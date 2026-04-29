from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.train_ucr_classification_pretrained_fewshot import evaluate, parse_args
from opentslm.time_series_datasets.univariate_fewshot import load_univariate_fewshot_bundle


def _write_ucr_dataset(root: Path, dataset_name: str) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    dataset_dir = archive / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        [0, 0.00, 0.05, 0.00, 0.05],
        [0, 0.10, 0.00, 0.10, 0.00],
        [1, 1.00, 0.95, 1.00, 0.95],
        [1, 0.90, 1.00, 0.90, 1.00],
    ]
    test_rows = [
        [0, 0.02, 0.05, 0.02, 0.05],
        [1, 0.98, 0.96, 0.98, 0.96],
    ]

    def _write_rows(path: Path, rows: list[list[float]]) -> None:
        lines = ["\t".join(str(value) for value in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_rows(dataset_dir / f"{dataset_name}_TRAIN.tsv", train_rows)
    _write_rows(dataset_dir / f"{dataset_name}_TEST.tsv", test_rows)
    return archive


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

    def generate(self, batch, max_new_tokens: int, skip_special_tokens: bool = False):
        del max_new_tokens
        del skip_special_tokens
        self.native_generate_calls += 1
        return [sample["answer"].replace(self.get_eos_token(), "").strip() for sample in batch]

    def get_eos_token(self) -> str:
        return "<eos>"


class M2PretrainedFewshotTest(unittest.TestCase):
    def test_parse_args_tracks_constrained_decoding_flag(self) -> None:
        default_args = parse_args([])
        self.assertTrue(default_args.constrained_decoding)
        self.assertFalse(default_args.disable_constrained_decoding)
        self.assertEqual(default_args.dataset_family, "ucr")
        self.assertEqual(default_args.split_protocol, "default")

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
        self.assertEqual(metrics["macro_f1"], 1.0)
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
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(model.llm.calls), 0)
        self.assertEqual(model.native_generate_calls, 1)

    def test_ucr_bundle_loader_preserves_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = _write_ucr_dataset(Path(tmp), "ToyWave")
            args = SimpleNamespace(
                dataset_family="ucr",
                dataset="ToyWave",
                split_protocol="default",
                data_path=str(archive),
            )

            bundle = load_univariate_fewshot_bundle(args, eos_token="<eos>")

            self.assertEqual(bundle.dataset_family, "ucr")
            self.assertEqual(bundle.dataset_name, "ToyWave")
            self.assertEqual(bundle.split_protocol, "official")
            self.assertEqual(bundle.num_classes, 2)
            self.assertEqual(bundle.class_tokens, ["<c0>", "<c1>"])
            self.assertEqual(len(bundle.train_dataset), 4)
            self.assertEqual(len(bundle.test_dataset), 2)


if __name__ == "__main__":
    unittest.main()
