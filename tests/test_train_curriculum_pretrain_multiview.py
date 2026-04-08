from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch.utils.data import Dataset

from scripts.train_curriculum_pretrain_multiview import (
    build_stage12_datasets,
    evaluate_model,
    parse_args,
)


class _TinyDataset(Dataset):
    def __init__(self, size: int):
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        return {"idx": idx}


class _EvalModel:
    def __init__(self):
        self.generate_calls: list[int] = []

    def eval(self):
        return self

    def compute_losses(self, batch):
        return {
            "loss_total": torch.tensor(1.0),
            "loss_lm": torch.tensor(0.5),
        }

    def generate(self, batch, max_new_tokens: int):
        self.generate_calls.append(len(batch))
        return [sample["answer"] for sample in batch]

    def get_eos_token(self) -> str:
        return ""


class CurriculumMultiviewSpeedConfigTest(unittest.TestCase):
    def test_stage12_default_epoch_caps_apply_to_train_and_validation(self) -> None:
        args = parse_args([])
        fake_aligned = {
            "tsqa": _TinyDataset(10),
            "m4": _TinyDataset(20),
            "synthetic": _TinyDataset(30),
        }

        with mock.patch(
            "scripts.train_curriculum_pretrain_multiview.build_stage12_aligned_datasets",
            return_value=fake_aligned,
        ):
            stage1_train = build_stage12_datasets(
                args,
                split="train",
                eos_token="",
                stage_name="stage1_semantic_alignment",
            )
            stage2_train = build_stage12_datasets(
                args,
                split="train",
                eos_token="",
                stage_name="stage2_instruction_tuning",
            )
            stage1_val = build_stage12_datasets(
                args,
                split="validation",
                eos_token="",
                stage_name="stage1_semantic_alignment",
            )
            stage2_val = build_stage12_datasets(
                args,
                split="validation",
                eos_token="",
                stage_name="stage2_instruction_tuning",
            )

        self.assertEqual(len(stage1_train), 192000)
        self.assertEqual(len(stage2_train), 160000)
        self.assertEqual(len(stage1_val), 8000)
        self.assertEqual(len(stage2_val), 8000)

    def test_evaluate_model_skips_match_generation_by_default(self) -> None:
        model = _EvalModel()
        data_loader = [
            [
                {"sample_type": "match_mismatch", "answer": "match"},
                {"sample_type": "match_mismatch", "answer": "mismatch"},
            ]
        ]

        metrics = evaluate_model(
            model=model,
            data_loader=data_loader,
            device="cpu",
            amp_dtype=None,
            stage_name="stage1_semantic_alignment",
            max_new_tokens=8,
            eval_match_accuracy=False,
            eval_match_max_samples=512,
        )

        self.assertEqual(model.generate_calls, [])
        self.assertNotIn("match_accuracy", metrics)

    def test_evaluate_model_caps_match_generation_budget(self) -> None:
        model = _EvalModel()
        data_loader = [
            [
                {"sample_type": "match_mismatch", "answer": "match"},
                {"sample_type": "match_mismatch", "answer": "mismatch"},
            ],
            [
                {"sample_type": "match_mismatch", "answer": "match"},
                {"sample_type": "match_mismatch", "answer": "mismatch"},
            ],
        ]

        metrics = evaluate_model(
            model=model,
            data_loader=data_loader,
            device="cpu",
            amp_dtype=None,
            stage_name="stage2_instruction_tuning",
            max_new_tokens=8,
            eval_match_accuracy=True,
            eval_match_max_samples=3,
        )

        self.assertEqual(model.generate_calls, [2, 1])
        self.assertEqual(metrics["match_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
