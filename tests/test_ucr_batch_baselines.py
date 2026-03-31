from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.ablations.train_ucr_distance_classification_fewshot import (
    dtw_distance,
    predict_1nn,
)
from scripts.ablations.train_ucr_distance_classification_fewshot import main as distance_main
from scripts.ablations.train_ucr_simple_backbone_classification_fewshot import main as neural_main
from scripts.experiments.ucr_batch.registry import get_entry, list_experiments


def _write_ucr_dataset(root: Path, dataset_name: str) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    dataset_dir = archive / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        [0, 0.00, 0.05, 0.00, 0.05, 0.00, 0.05, 0.00, 0.05, 0.00, 0.05, 0.00, 0.05],
        [0, 0.10, 0.00, 0.10, 0.00, 0.10, 0.00, 0.10, 0.00, 0.10, 0.00, 0.10, 0.00],
        [0, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00, 0.10, 0.10, 0.00, 0.00, 0.10, 0.10],
        [1, 1.00, 0.95, 1.00, 0.95, 1.00, 0.95, 1.00, 0.95, 1.00, 0.95, 1.00, 0.95],
        [1, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00, 0.90, 1.00],
        [1, 1.00, 1.00, 0.90, 0.90, 1.00, 1.00, 0.90, 0.90, 1.00, 1.00, 0.90, 0.90],
    ]
    test_rows = [
        [0, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05, 0.02, 0.05],
        [0, 0.08, 0.01, 0.08, 0.01, 0.08, 0.01, 0.08, 0.01, 0.08, 0.01, 0.08, 0.01],
        [1, 0.98, 0.96, 0.98, 0.96, 0.98, 0.96, 0.98, 0.96, 0.98, 0.96, 0.98, 0.96],
        [1, 0.92, 0.99, 0.92, 0.99, 0.92, 0.99, 0.92, 0.99, 0.92, 0.99, 0.92, 0.99],
    ]

    def _write_rows(path: Path, rows: list[list[float]]) -> None:
        lines = ["\t".join(str(value) for value in row) for row in rows]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_rows(dataset_dir / f"{dataset_name}_TRAIN.tsv", train_rows)
    _write_rows(dataset_dir / f"{dataset_name}_TEST.tsv", test_rows)
    return archive


def _read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class UCRBatchBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_distance_helpers_predict_expected_labels(self) -> None:
        support_features = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0, 2.0],
            ],
            dtype=np.float32,
        )
        support_labels = np.asarray([0, 1], dtype=np.int64)
        query_features = np.asarray(
            [
                [0.1, 0.0, 0.1, 0.0],
                [1.9, 2.0, 1.9, 2.0],
            ],
            dtype=np.float32,
        )

        preds_ed, distances_ed, nearest_ed = predict_1nn(support_features, support_labels, query_features, metric="ed")
        self.assertEqual(preds_ed.tolist(), [0, 1])
        self.assertEqual(nearest_ed.tolist(), [0, 1])
        self.assertLess(float(distances_ed[0]), float(distances_ed[1]) + 1.0)

        preds_dtw, _, nearest_dtw = predict_1nn(support_features, support_labels, query_features, metric="dtw")
        self.assertEqual(preds_dtw.tolist(), [0, 1])
        self.assertEqual(nearest_dtw.tolist(), [0, 1])

        self.assertAlmostEqual(dtw_distance(np.asarray([1, 2, 3], dtype=np.float32), np.asarray([1, 2, 3], dtype=np.float32)), 0.0)
        self.assertGreater(
            dtw_distance(np.asarray([0, 0, 0], dtype=np.float32), np.asarray([1, 1, 1], dtype=np.float32)),
            0.0,
        )

    def test_registry_contains_new_fewshot_baselines(self) -> None:
        experiments = list_experiments()
        for experiment in ("1nn_ed", "1nn_dtw", "resnet", "tapnet"):
            self.assertIn(experiment, experiments)

        entry_ed = get_entry("1nn_ed", "fewshot")
        self.assertEqual(entry_ed.fixed_args, ("--metric", "ed"))
        self.assertEqual(entry_ed.default_shots, "1,2,5,10")
        self.assertTrue(entry_ed.script_path.name.endswith("train_ucr_distance_classification_fewshot.py"))

        entry_dtw = get_entry("1nn_dtw", "fewshot")
        self.assertEqual(entry_dtw.fixed_args, ("--metric", "dtw"))

        entry_resnet = get_entry("resnet", "fewshot")
        self.assertEqual(entry_resnet.fixed_args, ("--model", "resnet"))
        self.assertTrue(entry_resnet.supports_inner_resume)

        entry_tapnet = get_entry("tapnet", "fewshot")
        self.assertEqual(entry_tapnet.fixed_args, ("--model", "tapnet"))

    def test_smoke_runs_write_fewshot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_dataset(tmp_path, "ToyWave")

            specs = [
                (
                    "1nn_ed",
                    distance_main,
                    [
                        "--metric",
                        "ed",
                    ],
                ),
                (
                    "1nn_dtw",
                    distance_main,
                    [
                        "--metric",
                        "dtw",
                    ],
                ),
                (
                    "resnet",
                    neural_main,
                    [
                        "--model",
                        "resnet",
                        "--device",
                        "cpu",
                        "--epochs",
                        "1",
                        "--batch_size",
                        "2",
                        "--eval_batch_size",
                        "4",
                    ],
                ),
                (
                    "tapnet",
                    neural_main,
                    [
                        "--model",
                        "tapnet",
                        "--device",
                        "cpu",
                        "--epochs",
                        "1",
                        "--batch_size",
                        "2",
                        "--eval_batch_size",
                        "4",
                        "--dropout",
                        "0.0",
                    ],
                ),
            ]

            for experiment_name, entrypoint, extra_args in specs:
                save_dir = tmp_path / "results" / experiment_name
                with self.subTest(experiment=experiment_name):
                    entrypoint(
                        [
                            "--dataset",
                            "ToyWave",
                            "--data_path",
                            str(archive),
                            "--shots",
                            "1",
                            "--num_runs",
                            "1",
                            "--save_dir",
                            str(save_dir),
                            *extra_args,
                        ]
                    )

                    dataset_root = save_dir / "ToyWave"
                    self.assertTrue((dataset_root / "config.json").exists())
                    self.assertTrue((dataset_root / "fewshot_summary.json").exists())
                    summary_csv = dataset_root / "fewshot_summary.csv"
                    self.assertTrue(summary_csv.exists())
                    self.assertTrue((dataset_root / "shot_1" / "shot_summary.json").exists())
                    self.assertTrue((dataset_root / "shot_1" / "run_01" / "fewshot_indices.json").exists())
                    self.assertTrue((dataset_root / "shot_1" / "run_01" / "run_metrics.json").exists())
                    self.assertTrue((dataset_root / "shot_1" / "run_01" / "test_predictions.json").exists())

                    summary_rows = _read_summary_rows(summary_csv)
                    self.assertEqual(len(summary_rows), 1)
                    self.assertEqual(summary_rows[0]["shot"], "1")
                    self.assertEqual(summary_rows[0]["num_runs"], "1")


if __name__ == "__main__":
    unittest.main()
