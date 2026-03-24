from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


_ROW_TRAINING_STATE_ATTR = "_class_token_row_training_state"
_ROW_TRAINING_HOOKS_ATTR = "_class_token_row_training_hooks"


def _normalize_class_token_ids(class_token_ids: Sequence[int]) -> Tuple[int, ...]:
    ids = tuple(int(token_id) for token_id in class_token_ids)
    if not ids:
        raise ValueError("class_token_ids must not be empty")
    if any(token_id < 0 for token_id in ids):
        raise ValueError(f"class_token_ids must be non-negative, got {ids}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"class_token_ids must be unique, got {ids}")
    return ids


def _build_row_mask(
    num_rows: int,
    class_token_ids: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ids = _normalize_class_token_ids(class_token_ids)
    if max(ids) >= num_rows:
        raise ValueError(
            f"class_token_ids {ids} exceed parameter row count {num_rows}"
        )

    mask = torch.zeros(num_rows, 1, device=device, dtype=dtype)
    index = torch.tensor(ids, device=device, dtype=torch.long)
    mask.index_fill_(0, index, 1.0)
    return mask


def _make_mask_hook(mask: torch.Tensor):
    def hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad

        local_mask = mask
        if local_mask.device != grad.device or local_mask.dtype != grad.dtype:
            local_mask = local_mask.to(device=grad.device, dtype=grad.dtype)
        return grad * local_mask

    return hook


def _iter_unique_masked_parameters(model) -> Iterable[Tuple[torch.nn.Parameter, torch.Tensor]]:
    state = get_class_token_row_training_state(model)
    class_token_ids = state["class_token_ids"]

    seen_param_ids: set[int] = set()
    parameters = (
        model.llm.get_input_embeddings().weight,
        model.llm.lm_head.weight,
    )
    for parameter in parameters:
        parameter_id = id(parameter)
        if parameter_id in seen_param_ids:
            continue
        seen_param_ids.add(parameter_id)
        yield parameter, _build_row_mask(
            parameter.shape[0],
            class_token_ids,
            device=parameter.device,
            dtype=parameter.dtype,
        )


def register_class_token_row_training(model, class_token_ids: Sequence[int]) -> Dict[str, Any]:
    ids = _normalize_class_token_ids(class_token_ids)

    for handle in getattr(model, _ROW_TRAINING_HOOKS_ATTR, []):
        handle.remove()

    state = {"class_token_ids": ids}
    setattr(model, _ROW_TRAINING_STATE_ATTR, state)

    hook_handles = []
    for parameter, mask in _iter_unique_masked_parameters(model):
        parameter.requires_grad = True
        hook_handles.append(parameter.register_hook(_make_mask_hook(mask)))
    setattr(model, _ROW_TRAINING_HOOKS_ATTR, hook_handles)
    return state


def get_class_token_row_training_state(model) -> Dict[str, Any]:
    state = getattr(model, _ROW_TRAINING_STATE_ATTR, None)
    if state is None:
        raise RuntimeError(
            "Class-token row training has not been registered for this model."
        )
    return state


def get_class_token_ids(model) -> Tuple[int, ...]:
    return get_class_token_row_training_state(model)["class_token_ids"]


def get_class_token_trainable_parameters(model) -> List[torch.nn.Parameter]:
    return [parameter for parameter, _ in _iter_unique_masked_parameters(model)]


def sanitize_class_token_optimizer_state(optimizer, model) -> None:
    for parameter, mask in _iter_unique_masked_parameters(model):
        param_state = optimizer.state.get(parameter)
        if not param_state:
            continue

        for value in param_state.values():
            if torch.is_tensor(value) and value.shape == parameter.shape:
                local_mask = mask
                if local_mask.device != value.device or local_mask.dtype != value.dtype:
                    local_mask = local_mask.to(device=value.device, dtype=value.dtype)
                value.mul_(local_mask)


def save_class_token_rows_to_checkpoint(model, checkpoint: Dict[str, Any]) -> None:
    class_token_ids = list(get_class_token_ids(model))
    row_index = torch.tensor(
        class_token_ids,
        device=model.llm.get_input_embeddings().weight.device,
        dtype=torch.long,
    )

    checkpoint["class_token_ids"] = class_token_ids
    checkpoint["class_token_embedding_rows"] = (
        model.llm.get_input_embeddings().weight.detach().index_select(0, row_index).cpu()
    )
    checkpoint["class_token_lm_head_rows"] = (
        model.llm.lm_head.weight.detach().index_select(0, row_index).cpu()
    )
    checkpoint["tokenizer_vocab_size"] = len(model.tokenizer)


def load_class_token_rows_from_checkpoint(
    model,
    checkpoint: Dict[str, Any],
    *,
    device: str,
) -> bool:
    current_class_token_ids = list(get_class_token_ids(model))
    embedding_weight = model.llm.get_input_embeddings().weight
    lm_head_weight = model.llm.lm_head.weight

    if "class_token_embedding_rows" in checkpoint and "class_token_lm_head_rows" in checkpoint:
        saved_class_token_ids = [int(token_id) for token_id in checkpoint.get("class_token_ids", [])]
        if saved_class_token_ids != current_class_token_ids:
            raise ValueError(
                "Class token ids in checkpoint do not match the current tokenizer: "
                f"checkpoint={saved_class_token_ids}, current={current_class_token_ids}"
            )

        expected_rows = len(current_class_token_ids)
        embedding_rows = checkpoint["class_token_embedding_rows"].to(
            device=device,
            dtype=embedding_weight.dtype,
        )
        lm_head_rows = checkpoint["class_token_lm_head_rows"].to(
            device=device,
            dtype=lm_head_weight.dtype,
        )
        if embedding_rows.shape[0] != expected_rows or lm_head_rows.shape[0] != expected_rows:
            raise ValueError(
                "Class-token row checkpoint shape mismatch: "
                f"expected {expected_rows} rows, got "
                f"{embedding_rows.shape[0]} and {lm_head_rows.shape[0]}"
            )

        row_index = torch.tensor(
            current_class_token_ids,
            device=embedding_weight.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            embedding_weight.index_copy_(0, row_index, embedding_rows)
            lm_head_weight.index_copy_(0, row_index, lm_head_rows)
        return True

    if "embedding_weight" in checkpoint and "lm_head_weight" in checkpoint:
        full_embedding = checkpoint["embedding_weight"].to(
            device=device,
            dtype=embedding_weight.dtype,
        )
        full_lm_head = checkpoint["lm_head_weight"].to(
            device=device,
            dtype=lm_head_weight.dtype,
        )
        if full_embedding.shape != embedding_weight.shape or full_lm_head.shape != lm_head_weight.shape:
            raise ValueError(
                "Full embedding checkpoint shape mismatch: "
                f"embedding {full_embedding.shape} vs {embedding_weight.shape}, "
                f"lm_head {full_lm_head.shape} vs {lm_head_weight.shape}"
            )

        with torch.no_grad():
            embedding_weight.copy_(full_embedding)
            lm_head_weight.copy_(full_lm_head)
        return True

    return False
