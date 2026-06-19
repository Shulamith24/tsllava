#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Some environments install Hugging Face datasets without the tiny
# pyarrow_hotfix package. UCR visualization does not use datasets, but the
# training script imports dataset-family modules eagerly, so provide a harmless
# placeholder to keep this utility importable.
sys.modules.setdefault("pyarrow_hotfix", types.ModuleType("pyarrow_hotfix"))

from scripts import train_ucr_classification_pretrained_fewshot as train  # noqa: E402
from opentslm.time_series_datasets.univariate_fewshot import load_univariate_fewshot_bundle  # noqa: E402


DEFAULT_RUNS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ucr_batches"
    / "m2_pretrained"
    / "fewshot"
    / "timemorph_visual_10shot"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
    / "timemorph_visual_analysis"
    / "features"
)


class IndexedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: Iterable[int], split_name: str):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample_index = self.indices[item]
        sample = dict(self.dataset[sample_index])
        sample["sample_index"] = sample_index
        sample["visual_split"] = self.split_name
        return sample


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TimeMorph features, branch predictions, and morphology attention rollout artifacts."
    )
    parser.add_argument("--runs_root", type=str, default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--datasets", type=str, default="ECG200,TwoLeadECG,ACSF1,ElectricDevices")
    parser.add_argument("--shot", type=str, default="10")
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--splits", type=str, default="support,test")
    parser.add_argument("--runtime_branch_modes", type=str, default="both,ts_only,vision_only")
    parser.add_argument("--max_query_per_class", type=int, default=200)
    parser.add_argument("--max_support_per_class", type=int, default=0)
    parser.add_argument("--attention_samples_per_dataset", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset root from saved config.")
    parser.add_argument("--device", type=str, default=None, help="Override device from saved config.")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=False)
    return parser.parse_args(argv)


def install_peft_unpickle_shim() -> None:
    """Allow older PEFT checkpoints with LoraRuntimeConfig to load in newer PEFT versions."""
    try:
        import peft.tuners.lora.config as lora_config
    except Exception:
        return

    if hasattr(lora_config, "LoraRuntimeConfig"):
        return

    class LoraRuntimeConfig:  # pragma: no cover - only used for legacy checkpoint unpickling
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    LoraRuntimeConfig.__module__ = lora_config.__name__
    setattr(lora_config, "LoraRuntimeConfig", LoraRuntimeConfig)


def safe_torch_load(path: Path | str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    install_peft_unpickle_shim()
    return torch.load(path, map_location=map_location, weights_only=False)


def resolve_dataset_root(runs_root: Path, dataset: str) -> Path:
    candidates = [
        runs_root / "datasets" / dataset,
        runs_root / dataset,
        runs_root,
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists() or any(candidate.glob("shot_*")):
            return candidate
    return runs_root / "datasets" / dataset


def resolve_run_dir(runs_root: Path, dataset: str, shot: str, run_id: int) -> Path:
    run_name = f"run_{int(run_id):02d}"
    candidates = [
        runs_root / "datasets" / dataset / f"shot_{shot}" / run_name,
        runs_root / dataset / f"shot_{shot}" / run_name,
        runs_root / f"shot_{shot}" / run_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_saved_config(dataset_root: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    config_path = dataset_root / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_args = checkpoint.get("args")
    if isinstance(checkpoint_args, dict):
        return dict(checkpoint_args)
    raise FileNotFoundError(
        f"Cannot find config.json under {dataset_root}, and checkpoint does not contain args metadata."
    )


def build_args_from_saved_config(
    saved_config: dict[str, Any],
    *,
    dataset: str,
    shot: str,
    data_path_override: str | None,
    device_override: str | None,
    eval_batch_size_override: int | None,
    dataloader_num_workers: int,
    pin_memory: bool,
) -> argparse.Namespace:
    args = train.parse_args([])
    for key, value in saved_config.items():
        setattr(args, key, value)

    args.dataset_family = str(getattr(args, "dataset_family", "ucr")).lower()
    args.dataset = dataset
    args.protocol = "fewshot"
    args.shots = str(shot)
    args.num_runs = 1
    args.eval_decode_mode = "logits"
    args.disable_constrained_decoding = False
    args.use_lora = bool(getattr(args, "use_lora", not getattr(args, "no_lora", False)))
    args.no_lora = not args.use_lora

    if data_path_override is not None:
        args.data_path = data_path_override
    if device_override is not None:
        args.device = device_override
    if eval_batch_size_override is not None:
        args.eval_batch_size = int(eval_batch_size_override)
    args.dataloader_num_workers = int(dataloader_num_workers)
    args.pin_memory = bool(pin_memory)
    args.persistent_workers = False
    return args


def resolve_device(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "device", "cuda"))
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cpu"


def reset_univariate_dataset_cache(args: argparse.Namespace) -> None:
    """Avoid class-level dataset caches leaking across datasets in one export process."""
    family = str(getattr(args, "dataset_family", "ucr")).lower()
    cache_attrs = (
        "loaded",
        "_train_dataset",
        "_validation_dataset",
        "_test_dataset",
    )
    metadata_attrs = (
        "_dataset_name",
        "_label_to_token",
        "_token_to_label",
        "_num_classes",
        "_class_tokens",
    )
    dataset_classes = []

    if family == "ucr":
        from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset

        dataset_classes.append(UCRClassificationDataset)

    for dataset_cls in dataset_classes:
        for attr in (*cache_attrs, *metadata_attrs):
            if hasattr(dataset_cls, attr):
                delattr(dataset_cls, attr)


def build_model_from_phase_checkpoint(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    *,
    device: str,
):
    init_kwargs = train.resolve_model_init_kwargs_from_checkpoint(args, checkpoint)
    model = train.OpenTSLMSP(
        llm_id=init_kwargs["llm_id"],
        device=device,
        encoder_type=init_kwargs["encoder_type"],
        tslanet_config=init_kwargs["tslanet_config"],
        newts_dual_branch_config=init_kwargs["newts_dual_branch_config"],
        llm_attn_impl=getattr(args, "llm_attn_impl", "sdpa"),
    )

    if checkpoint.get("lora_enabled", False):
        lora_r, lora_alpha = train.OpenTSLM._resolve_lora_hparams_from_checkpoint(checkpoint)
        model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)
        args.use_lora = True
        args.no_lora = False

    if hasattr(model, "set_runtime_branch_mode"):
        model.set_runtime_branch_mode("both")
    return model


def load_timemorph_run(
    *,
    args: argparse.Namespace,
    dataset_bundle,
    phase2_checkpoint_path: Path,
    checkpoint: dict[str, Any],
    support_class_ids: list[int],
    device: str,
):
    model = build_model_from_phase_checkpoint(args, checkpoint, device=device)
    saved_class_token_ids = [int(token_id) for token_id in checkpoint.get("class_token_ids", [])]
    num_tokens_to_add = int(dataset_bundle.num_classes)
    if saved_class_token_ids:
        if len(saved_class_token_ids) < int(dataset_bundle.num_classes):
            raise ValueError(
                f"Checkpoint {phase2_checkpoint_path} contains only {len(saved_class_token_ids)} "
                f"class-token rows, but {dataset_bundle.dataset_name} has {dataset_bundle.num_classes} classes."
            )
        max_selected_class = max((int(class_id) for class_id in support_class_ids), default=-1)
        if max_selected_class >= len(saved_class_token_ids):
            raise ValueError(
                f"Selected class id {max_selected_class} is outside checkpoint class-token rows "
                f"({len(saved_class_token_ids)} rows) for {dataset_bundle.dataset_name}."
            )
        num_tokens_to_add = max(num_tokens_to_add, len(saved_class_token_ids))

    class_tokens, class_token_ids = train.add_class_tokens_to_model(
        model,
        num_classes=num_tokens_to_add,
        tokenizer_training_mode=args.tokenizer_training_mode,
        rank=0,
    )
    if dataset_bundle.class_tokens and class_tokens[: dataset_bundle.num_classes] != dataset_bundle.class_tokens:
        raise RuntimeError(
            f"Class-token mismatch for {dataset_bundle.dataset_name}: "
            f"{class_tokens[: dataset_bundle.num_classes]} vs {dataset_bundle.class_tokens}"
        )
    checkpoint_mode = train.get_checkpoint_tokenizer_training_mode(checkpoint)
    if checkpoint_mode != args.tokenizer_training_mode:
        raise ValueError(
            "Checkpoint tokenizer_training_mode mismatch: "
            f"expected {args.tokenizer_training_mode}, got {checkpoint_mode}."
        )

    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.projector.load_state_dict(checkpoint["projector_state"])
    model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    if args.tokenizer_training_mode == "class_rows":
        train.load_class_token_rows_from_checkpoint(model, checkpoint, device=device)
    elif not train.load_full_tokenizer_weights_from_checkpoint(model, checkpoint, device=device):
        raise ValueError(f"Checkpoint {phase2_checkpoint_path} does not contain full embedding/lm_head weights.")
    model.eval()
    selected_class_token_ids = [int(class_token_ids[class_id]) for class_id in support_class_ids]
    return model, class_tokens, class_token_ids, selected_class_token_ids


def validate_visual_run_metadata(
    *,
    dataset: str,
    run_dir: Path,
    support_info: dict[str, Any],
    run_metrics: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    for source_name, payload in (("fewshot_indices.json", support_info), ("run_metrics.json", run_metrics)):
        recorded_dataset = payload.get("dataset") if isinstance(payload, dict) else None
        if recorded_dataset is None:
            continue
        if str(recorded_dataset) != str(dataset):
            raise ValueError(
                f"Resolved run directory {run_dir} is for dataset {recorded_dataset!r}, "
                f"but the exporter is processing {dataset!r}. Check --runs_root and --datasets."
            )

    support_classes = support_info.get("selected_class_ids")
    metric_classes = run_metrics.get("selected_class_ids") if isinstance(run_metrics, dict) else None
    if isinstance(support_classes, list) and isinstance(metric_classes, list):
        if [int(item) for item in support_classes] != [int(item) for item in metric_classes]:
            warnings.append(
                "selected_class_ids differ between fewshot_indices.json and run_metrics.json"
            )
    return warnings


def sample_indices_per_class(
    label_to_indices: dict[int, list[int]],
    class_ids: list[int],
    *,
    max_per_class: int,
    seed: int,
) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for class_id in class_ids:
        indices = list(label_to_indices.get(int(class_id), []))
        if max_per_class > 0 and len(indices) > max_per_class:
            indices = sorted(rng.choice(indices, size=max_per_class, replace=False).astype(int).tolist())
        selected.extend(indices)
    return sorted(selected)


def build_loader(
    base_dataset: Dataset,
    indices: list[int],
    *,
    split_name: str,
    args: argparse.Namespace,
) -> DataLoader:
    subset = IndexedSubset(base_dataset, indices, split_name=split_name)
    return DataLoader(
        subset,
        batch_size=int(args.eval_batch_size),
        shuffle=False,
        collate_fn=train.make_collate_fn(args, is_train=False),
        **train.build_dataloader_kwargs(args),
    )


def tensor_to_numpy(tensor: torch.Tensor | None, batch_size: int) -> np.ndarray | None:
    if tensor is None:
        return np.empty((int(batch_size), 0), dtype=np.float32)
    return tensor.detach().float().cpu().numpy()


def finalize_optional_features(chunks: list[np.ndarray | None], total_count: int) -> np.ndarray:
    available = [chunk for chunk in chunks if chunk is not None and chunk.shape[1] > 0]
    if not chunks or not available:
        return np.empty((total_count, 0), dtype=np.float32)

    dim = int(available[0].shape[1])
    finalized = []
    for chunk in chunks:
        if chunk is None or chunk.shape[1] == 0:
            batch_size = 0 if chunk is None else int(chunk.shape[0])
            finalized.append(np.full((batch_size, dim), np.nan, dtype=np.float32))
        else:
            finalized.append(chunk.astype(np.float32, copy=False))
    return np.concatenate(finalized, axis=0) if finalized else np.empty((0, dim), dtype=np.float32)


@torch.no_grad()
def collect_split_mode_features(
    model,
    data_loader: DataLoader,
    *,
    runtime_branch_mode: str,
    selected_class_ids: list[int],
    selected_class_token_ids: list[int],
    desc: str,
) -> dict[str, np.ndarray | float | list[int]]:
    underlying_model = train.get_model(model)
    underlying_model.eval()

    pooled_ts_chunks: list[np.ndarray | None] = []
    pooled_vision_chunks: list[np.ndarray | None] = []
    pooled_fused_chunks: list[np.ndarray | None] = []
    decision_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []
    labels: list[int] = []
    target_positions: list[int] = []
    sample_indices: list[int] = []

    class_id_to_position = {int(class_id): idx for idx, class_id in enumerate(selected_class_ids)}
    class_token_tensor = torch.tensor(
        selected_class_token_ids,
        device=underlying_model.device,
        dtype=torch.long,
    )

    for batch in tqdm(data_loader, desc=desc):
        batch_outputs = underlying_model.pad_and_apply_batch(
            batch,
            runtime_branch_mode=runtime_branch_mode,
            return_encoder_outputs=True,
        )
        inputs_embeds, attention_mask, encoder_outputs = batch_outputs
        outputs = underlying_model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_positions = attention_mask.to(outputs.logits.device).long().sum(dim=1) - 1
        batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
        decision_states = outputs.hidden_states[-1][batch_indices, last_positions, :].float()
        next_token_logits = outputs.logits[batch_indices, last_positions, :]
        class_logits = next_token_logits.index_select(dim=-1, index=class_token_tensor)

        batch_labels = [int(sample["int_label"]) for sample in batch]
        batch_targets = [class_id_to_position[label] for label in batch_labels]
        labels.extend(batch_labels)
        target_positions.extend(batch_targets)
        sample_indices.extend(int(sample["sample_index"]) for sample in batch)

        pooled_ts_chunks.append(tensor_to_numpy(encoder_outputs.get("pooled_ts"), len(batch)))
        pooled_vision_chunks.append(tensor_to_numpy(encoder_outputs.get("pooled_vision"), len(batch)))
        pooled_fused_chunks.append(tensor_to_numpy(encoder_outputs.get("pooled_fused"), len(batch)))
        decision_chunks.append(decision_states.detach().cpu().numpy().astype(np.float32))
        logits_chunks.append(class_logits.detach().float().cpu().numpy().astype(np.float32))

    total_count = len(labels)
    class_logits_np = np.concatenate(logits_chunks, axis=0) if logits_chunks else np.empty((0, 0), dtype=np.float32)
    target_np = np.asarray(target_positions, dtype=np.int64)
    pred_positions = class_logits_np.argmax(axis=1).astype(np.int64) if class_logits_np.size else np.empty((0,), dtype=np.int64)
    predictions = np.asarray([selected_class_ids[int(pos)] for pos in pred_positions], dtype=np.int64)
    labels_np = np.asarray(labels, dtype=np.int64)
    correct = predictions == labels_np
    accuracy = float(correct.mean()) if correct.size else 0.0

    return {
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "labels": labels_np,
        "target_positions": target_np,
        "predictions": predictions,
        "pred_positions": pred_positions,
        "correct": correct.astype(bool),
        "class_logits": class_logits_np,
        "pooled_ts": finalize_optional_features(pooled_ts_chunks, total_count),
        "pooled_vision": finalize_optional_features(pooled_vision_chunks, total_count),
        "pooled_fused": finalize_optional_features(pooled_fused_chunks, total_count),
        "decision_state": np.concatenate(decision_chunks, axis=0) if decision_chunks else np.empty((0, 0), dtype=np.float32),
        "accuracy": accuracy,
        "selected_class_ids": list(selected_class_ids),
        "selected_class_token_ids": list(selected_class_token_ids),
    }


def save_feature_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np_payload = {key: value for key, value in payload.items() if isinstance(value, np.ndarray)}
    scalar_payload = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, np.ndarray)
    }
    np_payload["metadata_json"] = np.asarray(json.dumps(scalar_payload, ensure_ascii=False))
    np.savez_compressed(path, **np_payload)


def choose_attention_positions(
    *,
    labels: np.ndarray,
    correct: np.ndarray,
    max_samples: int,
) -> list[int]:
    if max_samples <= 0 or labels.size == 0:
        return []
    selected: list[int] = []
    for class_id in sorted(set(int(label) for label in labels.tolist())):
        class_positions = np.flatnonzero(labels == class_id)
        correct_positions = [int(pos) for pos in class_positions if bool(correct[int(pos)])]
        selected.append(correct_positions[0] if correct_positions else int(class_positions[0]))
        if len(selected) >= max_samples:
            return selected
    if len(selected) < max_samples:
        for pos in range(labels.size):
            if pos not in selected:
                selected.append(pos)
            if len(selected) >= max_samples:
                break
    return selected


def normalize_saliency(saliency: torch.Tensor) -> torch.Tensor:
    flat = saliency.flatten(1)
    min_vals = flat.amin(dim=1, keepdim=True)
    max_vals = flat.amax(dim=1, keepdim=True)
    flat = (flat - min_vals) / (max_vals - min_vals + 1e-6)
    return flat.view_as(saliency)


def ensure_attention_outputs_enabled(vit) -> None:
    """Force a HF vision backbone into an attention implementation that returns maps."""
    if hasattr(vit, "set_attn_implementation"):
        try:
            vit.set_attn_implementation("eager")
        except Exception:
            pass
    config = getattr(vit, "config", None)
    if config is not None:
        try:
            config.output_attentions = True
            config._attn_implementation = "eager"
            config._attn_implementation_internal = "eager"
        except Exception:
            pass


@torch.no_grad()
def attention_rollout_for_batch(model, batch: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    underlying_model = train.get_model(model)
    if getattr(underlying_model, "encoder_type", None) != "newts_dual_branch":
        raise RuntimeError("Attention rollout requires encoder_type='newts_dual_branch'.")
    vision_encoder = getattr(underlying_model.encoder, "vision_encoder", None)
    if vision_encoder is None:
        raise RuntimeError("Attention rollout requires an active vision branch.")

    series = torch.stack(
        [torch.as_tensor(sample["time_series"][0], dtype=torch.float32).flatten() for sample in batch],
        dim=0,
    ).to(underlying_model.device)
    past_values = series.unsqueeze(-1)
    morphology = vision_encoder.ts2grayscale_image(past_values)[:, 0].detach().cpu().numpy().astype(np.float32)
    images = vision_encoder.ts2image(past_values)
    pixel_values = vision_encoder._prepare_pixel_values(images)
    ensure_attention_outputs_enabled(vision_encoder.vit)
    outputs = vision_encoder.vit(
        pixel_values=pixel_values,
        output_attentions=True,
        return_dict=True,
    )
    attentions = getattr(outputs, "attentions", None)
    if not attentions:
        raise RuntimeError(
            "Vision backbone did not return attentions. Try a transformers/attention implementation "
            "that supports output_attentions for the configured DINOv2 model."
        )

    num_tokens = attentions[0].shape[-1]
    eye = torch.eye(num_tokens, device=attentions[0].device, dtype=attentions[0].dtype).unsqueeze(0)
    rollout = eye.expand(attentions[0].shape[0], -1, -1)
    max_layers = len(attentions)
    if getattr(vision_encoder, "feature_mode", "single") == "single":
        max_layers = min(max_layers, int(getattr(vision_encoder, "layer_idx", max_layers)))

    for attn in attentions[:max_layers]:
        attn_mean = attn.mean(dim=1)
        attn_mean = attn_mean + eye
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        rollout = torch.bmm(attn_mean, rollout)

    saliency = rollout[:, 0, 1:]
    num_patches = saliency.shape[-1]
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise RuntimeError(f"Cannot reshape {num_patches} ViT patch attentions into a square grid.")
    saliency_grid = saliency.view(saliency.shape[0], 1, grid_size, grid_size)
    saliency_grid = F.interpolate(
        saliency_grid,
        size=morphology.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    saliency_grid = normalize_saliency(saliency_grid)
    return (
        series.detach().cpu().numpy().astype(np.float32),
        morphology,
        saliency_grid.detach().cpu().numpy().astype(np.float32),
    )


def export_attention_rollout(
    *,
    model,
    base_dataset: Dataset,
    selected_global_indices: list[int],
    split_features: dict[str, Any],
    args: argparse.Namespace,
    output_path: Path,
    max_samples: int,
) -> dict[str, Any]:
    positions = choose_attention_positions(
        labels=split_features["labels"],
        correct=split_features["correct"],
        max_samples=max_samples,
    )
    if not positions:
        return {"enabled": False, "reason": "no selected positions"}

    global_indices = [int(selected_global_indices[position]) for position in positions]
    subset = IndexedSubset(base_dataset, global_indices, split_name="attention")
    loader = DataLoader(
        subset,
        batch_size=len(subset),
        shuffle=False,
        collate_fn=train.make_collate_fn(args, is_train=False),
        **train.build_dataloader_kwargs(args),
    )
    batch = next(iter(loader))
    raw_series, morphology, saliency = attention_rollout_for_batch(model, batch)
    labels = split_features["labels"][positions]
    predictions = split_features["predictions"][positions]
    correct = split_features["correct"][positions]
    sample_indices = np.asarray(global_indices, dtype=np.int64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        raw_series=raw_series,
        morphology=morphology,
        saliency=saliency,
        labels=labels,
        predictions=predictions,
        correct=correct,
        sample_indices=sample_indices,
    )
    return {
        "enabled": True,
        "artifact": str(output_path.resolve()),
        "num_samples": int(len(global_indices)),
        "sample_indices": [int(index) for index in global_indices],
    }


def validate_feature_payload(payload: dict[str, Any], *, expected_metric: float | None = None) -> dict[str, Any]:
    labels = payload["labels"]
    correct = payload["correct"]
    checks = {
        "num_samples": int(labels.shape[0]),
        "accuracy_from_logits": float(correct.mean()) if correct.size else 0.0,
        "finite": True,
    }
    for key in ("pooled_ts", "pooled_vision", "pooled_fused", "decision_state", "class_logits"):
        array = payload[key]
        finite = bool(np.isfinite(array).all()) if array.size else True
        checks[f"{key}_shape"] = list(array.shape)
        checks[f"{key}_finite"] = finite
        checks["finite"] = checks["finite"] and finite
    if expected_metric is not None:
        checks["expected_accuracy"] = float(expected_metric)
        checks["accuracy_abs_error"] = abs(checks["accuracy_from_logits"] - float(expected_metric))
    return checks


def export_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    runs_root = Path(args.runs_root).resolve()
    dataset_root = resolve_dataset_root(runs_root, dataset)
    run_dir = resolve_run_dir(runs_root, dataset, args.shot, args.run_id)
    phase2_checkpoint_path = run_dir / "phase2_last.pt"
    support_info_path = run_dir / "fewshot_indices.json"
    run_metrics_path = run_dir / "run_metrics.json"

    missing = [path for path in (phase2_checkpoint_path, support_info_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required visual-run artifacts: " + ", ".join(str(path) for path in missing)
        )

    checkpoint = safe_torch_load(phase2_checkpoint_path, map_location="cpu")
    saved_config = load_saved_config(dataset_root, checkpoint)
    train_args = build_args_from_saved_config(
        saved_config,
        dataset=dataset,
        shot=str(args.shot),
        data_path_override=args.data_path,
        device_override=args.device,
        eval_batch_size_override=args.eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
    )
    device = resolve_device(train_args)
    train_args.device = device

    dataset_eos = train.resolve_dataset_eos_token(train_args)
    reset_univariate_dataset_cache(train_args)
    dataset_bundle = load_univariate_fewshot_bundle(train_args, eos_token=dataset_eos)
    support_info = json.loads(support_info_path.read_text(encoding="utf-8"))
    support_class_ids = [int(class_id) for class_id in support_info["selected_class_ids"]]
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8")) if run_metrics_path.exists() else {}
    metadata_warnings = validate_visual_run_metadata(
        dataset=dataset,
        run_dir=run_dir,
        support_info=support_info,
        run_metrics=run_metrics,
    )

    model, class_tokens, class_token_ids, selected_class_token_ids = load_timemorph_run(
        args=train_args,
        dataset_bundle=dataset_bundle,
        phase2_checkpoint_path=phase2_checkpoint_path,
        checkpoint=checkpoint,
        support_class_ids=support_class_ids,
        device=device,
    )

    label_to_indices = train.build_label_to_indices(dataset_bundle.train_dataset)
    test_label_to_indices = train.build_label_to_indices(dataset_bundle.test_dataset)
    support_indices = [int(index) for index in support_info["selected_indices"]]
    if args.max_support_per_class > 0:
        support_indices = sample_indices_per_class(
            label_to_indices,
            support_class_ids,
            max_per_class=int(args.max_support_per_class),
            seed=int(support_info.get("seed", 0)),
        )
    query_indices = sample_indices_per_class(
        test_label_to_indices,
        support_class_ids,
        max_per_class=int(args.max_query_per_class),
        seed=int(support_info.get("seed", 0)) + 91000,
    )
    run_eval_indices = (
        run_metrics.get("query_eval_subset", {}).get("selected_indices")
        if isinstance(run_metrics.get("query_eval_subset"), dict)
        else None
    )
    query_matches_run_metric = (
        isinstance(run_eval_indices, list)
        and [int(index) for index in run_eval_indices] == list(query_indices)
    )

    split_indices = {
        "support": support_indices,
        "test": query_indices,
    }
    split_datasets = {
        "support": dataset_bundle.train_dataset,
        "test": dataset_bundle.test_dataset,
    }
    requested_splits = parse_csv(args.splits)
    requested_modes = parse_csv(args.runtime_branch_modes)

    output_dataset_dir = Path(args.output_dir).resolve() / dataset
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_manifest: dict[str, Any] = {
        "dataset": dataset,
        "dataset_family": dataset_bundle.dataset_family,
        "shot": str(args.shot),
        "run_id": int(args.run_id),
        "run_dir": str(run_dir.resolve()),
        "dataset_root": str(dataset_root.resolve()),
        "phase2_checkpoint": str(phase2_checkpoint_path.resolve()),
        "support_info": support_info,
        "run_metrics": run_metrics,
        "device": device,
        "class_tokens": class_tokens,
        "class_token_ids": [int(token_id) for token_id in class_token_ids],
        "dataset_num_classes": int(dataset_bundle.num_classes),
        "checkpoint_class_token_ids": [int(token_id) for token_id in checkpoint.get("class_token_ids", [])],
        "metadata_warnings": metadata_warnings,
        "selected_class_ids": support_class_ids,
        "selected_class_token_ids": selected_class_token_ids,
        "splits": {},
        "attention_rollout": {},
    }

    both_test_payload: dict[str, Any] | None = None
    for split_name in requested_splits:
        if split_name not in split_indices:
            raise ValueError(f"Unsupported split {split_name!r}; expected support or test.")
        loader = build_loader(
            split_datasets[split_name],
            split_indices[split_name],
            split_name=split_name,
            args=train_args,
        )
        dataset_manifest["splits"].setdefault(split_name, {"num_samples": len(split_indices[split_name]), "modes": {}})
        for mode in requested_modes:
            payload = collect_split_mode_features(
                model,
                loader,
                runtime_branch_mode=mode,
                selected_class_ids=support_class_ids,
                selected_class_token_ids=selected_class_token_ids,
                desc=f"{dataset}/{split_name}/{mode}",
            )
            expected_accuracy = None
            if (
                split_name == "test"
                and mode == "both"
                and "test_accuracy" in run_metrics
                and query_matches_run_metric
            ):
                expected_accuracy = float(run_metrics["test_accuracy"])
            checks = validate_feature_payload(payload, expected_metric=expected_accuracy)
            if split_name == "test" and mode == "both" and "test_accuracy" in run_metrics and not query_matches_run_metric:
                checks["full_run_accuracy"] = float(run_metrics["test_accuracy"])
                checks["full_run_accuracy_comparison"] = "skipped_query_subset_differs"
            feature_path = output_dataset_dir / f"{split_name}_{mode}.npz"
            save_feature_npz(feature_path, payload)
            dataset_manifest["splits"][split_name]["modes"][mode] = {
                "artifact": str(feature_path.resolve()),
                "checks": checks,
            }
            if split_name == "test" and mode == "both":
                both_test_payload = payload

    if both_test_payload is not None and int(args.attention_samples_per_dataset) > 0:
        try:
            dataset_manifest["attention_rollout"] = export_attention_rollout(
                model=model,
                base_dataset=dataset_bundle.test_dataset,
                selected_global_indices=query_indices,
                split_features=both_test_payload,
                args=train_args,
                output_path=output_dataset_dir / "attention_rollout.npz",
                max_samples=int(args.attention_samples_per_dataset),
            )
        except Exception as exc:
            dataset_manifest["attention_rollout"] = {
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    manifest_path = output_dataset_dir / "metadata.json"
    manifest_path.write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dataset_manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    datasets = parse_csv(args.datasets)
    if not datasets:
        raise ValueError("--datasets must contain at least one dataset")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    manifests = []
    for dataset in datasets:
        manifests.append(export_dataset(args, dataset))

    top_manifest = {
        "datasets": datasets,
        "shot": str(args.shot),
        "run_id": int(args.run_id),
        "runs_root": str(Path(args.runs_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "dataset_manifests": manifests,
    }
    manifest_path = Path(args.output_dir).resolve().parent / "feature_export_manifest.json"
    manifest_path.write_text(json.dumps(top_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote feature export manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
