from __future__ import annotations

import unittest

import torch

from scripts.train_ucr_classification_pretrained_fewshot import evaluate, parse_args


class _DummyTokenizer:
    eos_token_id = 99

    def batch_decode(self, _gen_ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        return ["<c0>"]


class _DummyLLM:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[1, 2]])


class _DummyFewshotModel:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.llm = _DummyLLM()
        self.native_generate_calls = 0

    def eval(self):
        return self

    def compute_loss(self, _batch):
        return torch.tensor(0.0)

    def pad_and_apply_batch(self, _batch):
        return torch.zeros(1, 2, 3), torch.ones(1, 2, dtype=torch.long)

    def generate(self, batch, max_new_tokens: int):
        del max_new_tokens
        self.native_generate_calls += 1
        return [sample["answer"].replace(self.get_eos_token(), "").strip() for sample in batch]

    def get_eos_token(self) -> str:
        return "<eos>"


class M2PretrainedFewshotTest(unittest.TestCase):
    def test_parse_args_tracks_constrained_decoding_flag(self) -> None:
        default_args = parse_args([])
        self.assertTrue(default_args.constrained_decoding)
        self.assertFalse(default_args.disable_constrained_decoding)

        disabled_args = parse_args(["--disable_constrained_decoding"])
        self.assertFalse(disabled_args.constrained_decoding)
        self.assertTrue(disabled_args.disable_constrained_decoding)

    def test_evaluate_uses_constrained_logits_processor_by_default(self) -> None:
        model = _DummyFewshotModel()
        batch = [{"answer": "<c0><eos>"}]

        metrics = evaluate(
            model=model,
            data_loader=[batch],
            max_new_tokens=2,
            class_token_ids=[7],
            disable_constrained_decoding=False,
            rank=1,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(len(model.llm.calls), 1)
        self.assertIn("logits_processor", model.llm.calls[0])
        self.assertEqual(model.native_generate_calls, 0)

    def test_evaluate_can_disable_constrained_decoding(self) -> None:
        model = _DummyFewshotModel()
        batch = [{"answer": "<c0><eos>"}]

        metrics = evaluate(
            model=model,
            data_loader=[batch],
            max_new_tokens=2,
            class_token_ids=[7],
            disable_constrained_decoding=True,
            rank=1,
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(len(model.llm.calls), 0)
        self.assertEqual(model.native_generate_calls, 1)


if __name__ == "__main__":
    unittest.main()
