from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import AdamW


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from opentslm.model.class_token_rows import (
    get_class_token_trainable_parameters,
    load_class_token_rows_from_checkpoint,
    register_class_token_row_training,
    sanitize_class_token_optimizer_state,
    save_class_token_rows_to_checkpoint,
)


class DummyTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.vocab_size


class DummyLLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 6, hidden_size: int = 4):
        super().__init__()
        self.llm = DummyLLM(vocab_size=vocab_size, hidden_size=hidden_size)
        self.tokenizer = DummyTokenizer(vocab_size)


def _build_dense_loss(model: DummyModel) -> torch.Tensor:
    input_ids = torch.tensor([[0, 1, 2, 4]])
    emb_loss = model.llm.get_input_embeddings()(input_ids).sum()
    hidden = torch.ones(3, model.llm.lm_head.in_features)
    head_loss = model.llm.lm_head(hidden).sum()
    return emb_loss + head_loss


def test_register_class_token_row_training_masks_gradients():
    model = DummyModel()
    class_token_ids = [1, 4]

    register_class_token_row_training(model, class_token_ids)
    trainable_params = get_class_token_trainable_parameters(model)
    assert len(trainable_params) == 2

    loss = _build_dense_loss(model)
    loss.backward()

    embedding_grad = model.llm.get_input_embeddings().weight.grad
    lm_head_grad = model.llm.lm_head.weight.grad
    assert embedding_grad is not None
    assert lm_head_grad is not None

    for row_idx in range(model.llm.get_input_embeddings().weight.shape[0]):
        emb_row_norm = embedding_grad[row_idx].abs().sum().item()
        head_row_norm = lm_head_grad[row_idx].abs().sum().item()
        if row_idx in class_token_ids:
            assert emb_row_norm > 0
            assert head_row_norm > 0
        else:
            assert emb_row_norm == 0
            assert head_row_norm == 0


def test_class_token_optimizer_step_only_updates_selected_rows_and_sanitizes_state():
    model = DummyModel()
    class_token_ids = [1, 4]
    register_class_token_row_training(model, class_token_ids)

    parameters = get_class_token_trainable_parameters(model)
    optimizer = AdamW(
        [{"params": parameters, "lr": 0.1, "weight_decay": 0.0}],
    )

    before_embedding = model.llm.get_input_embeddings().weight.detach().clone()
    before_lm_head = model.llm.lm_head.weight.detach().clone()

    loss = _build_dense_loss(model)
    loss.backward()
    optimizer.step()

    after_embedding = model.llm.get_input_embeddings().weight.detach()
    after_lm_head = model.llm.lm_head.weight.detach()

    for row_idx in range(before_embedding.shape[0]):
        embedding_changed = not torch.equal(before_embedding[row_idx], after_embedding[row_idx])
        lm_head_changed = not torch.equal(before_lm_head[row_idx], after_lm_head[row_idx])
        if row_idx in class_token_ids:
            assert embedding_changed
            assert lm_head_changed
        else:
            assert not embedding_changed
            assert not lm_head_changed

    for parameter in parameters:
        state = optimizer.state[parameter]
        state["exp_avg"].fill_(1.0)
        state["exp_avg_sq"].fill_(2.0)

    sanitize_class_token_optimizer_state(optimizer, model)

    for parameter in parameters:
        state = optimizer.state[parameter]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        for row_idx in range(parameter.shape[0]):
            if row_idx in class_token_ids:
                assert torch.all(exp_avg[row_idx] == 1.0)
                assert torch.all(exp_avg_sq[row_idx] == 2.0)
            else:
                assert torch.all(exp_avg[row_idx] == 0.0)
                assert torch.all(exp_avg_sq[row_idx] == 0.0)


def test_rows_only_checkpoint_roundtrip_restores_only_class_rows():
    source_model = DummyModel()
    target_model = DummyModel()
    class_token_ids = [1, 4]
    register_class_token_row_training(source_model, class_token_ids)
    register_class_token_row_training(target_model, class_token_ids)

    with torch.no_grad():
        source_model.llm.get_input_embeddings().weight[class_token_ids] = 10.0
        source_model.llm.lm_head.weight[class_token_ids] = 20.0

    target_embedding_before = target_model.llm.get_input_embeddings().weight.detach().clone()
    target_lm_head_before = target_model.llm.lm_head.weight.detach().clone()

    checkpoint = {}
    save_class_token_rows_to_checkpoint(source_model, checkpoint)
    restored = load_class_token_rows_from_checkpoint(target_model, checkpoint, device="cpu")
    assert restored is True

    target_embedding_after = target_model.llm.get_input_embeddings().weight.detach()
    target_lm_head_after = target_model.llm.lm_head.weight.detach()

    for row_idx in range(target_embedding_after.shape[0]):
        if row_idx in class_token_ids:
            torch.testing.assert_close(
                target_embedding_after[row_idx],
                source_model.llm.get_input_embeddings().weight.detach()[row_idx],
            )
            torch.testing.assert_close(
                target_lm_head_after[row_idx],
                source_model.llm.lm_head.weight.detach()[row_idx],
            )
        else:
            torch.testing.assert_close(target_embedding_after[row_idx], target_embedding_before[row_idx])
            torch.testing.assert_close(target_lm_head_after[row_idx], target_lm_head_before[row_idx])


def test_legacy_full_checkpoint_loading_is_still_supported():
    source_model = DummyModel()
    target_model = DummyModel()
    class_token_ids = [1, 4]
    register_class_token_row_training(source_model, class_token_ids)
    register_class_token_row_training(target_model, class_token_ids)

    with torch.no_grad():
        source_model.llm.get_input_embeddings().weight.copy_(
            torch.arange(24, dtype=torch.float32).view(6, 4)
        )
        source_model.llm.lm_head.weight.copy_(
            torch.arange(24, dtype=torch.float32).view(6, 4) * 3
        )

    checkpoint = {
        "embedding_weight": source_model.llm.get_input_embeddings().weight.detach().clone(),
        "lm_head_weight": source_model.llm.lm_head.weight.detach().clone(),
    }
    restored = load_class_token_rows_from_checkpoint(target_model, checkpoint, device="cpu")
    assert restored is True

    torch.testing.assert_close(
        target_model.llm.get_input_embeddings().weight.detach(),
        source_model.llm.get_input_embeddings().weight.detach(),
    )
    torch.testing.assert_close(
        target_model.llm.lm_head.weight.detach(),
        source_model.llm.lm_head.weight.detach(),
    )
