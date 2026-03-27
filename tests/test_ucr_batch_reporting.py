from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.experiments.ucr_batch.reporting.latex import render_main_table
from scripts.experiments.ucr_batch.reporting.pipeline import generate_report


LEDGER_FIELDS = [
    "dataset",
    "shot",
    "status",
    "accuracy",
    "accuracy_std",
    "num_runs",
    "result_file",
    "log_file",
    "updated_at",
    "note",
]


def _write_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _make_row(dataset: str, shot: str, accuracy: str, *, num_runs: str = "3") -> dict[str, str]:
    return {
        "dataset": dataset,
        "shot": shot,
        "status": "success",
        "accuracy": accuracy,
        "accuracy_std": "0.01",
        "num_runs": num_runs,
        "result_file": "",
        "log_file": "",
        "updated_at": "2026-03-27T00:00:00Z",
        "note": "",
    }


def _make_ucr_archive(root: Path, datasets: list[str]) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    archive.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        dataset_dir = archive / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / f"{dataset}_TRAIN.tsv").write_text("0\t0\n", encoding="utf-8")
        (dataset_dir / f"{dataset}_TEST.tsv").write_text("0\t0\n", encoding="utf-8")
    return archive


class FewshotReportingTest(unittest.TestCase):
    def test_strict_mode_writes_coverage_report_before_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha", "Beta"])

            model_a = tmp_path / "model_a_results.txt"
            model_b = tmp_path / "model_b_results.txt"
            _write_results(
                model_a,
                [
                    _make_row("Alpha", "1", "0.80"),
                    _make_row("Alpha", "2", "0.82"),
                    _make_row("Beta", "1", "0.70"),
                    _make_row("Beta", "2", "0.72"),
                ],
            )
            _write_results(
                model_b,
                [
                    _make_row("Alpha", "1", "0.75"),
                    _make_row("Alpha", "2", "0.77"),
                    _make_row("Beta", "1", "0.68"),
                ],
            )

            config_path = tmp_path / "strict_report.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "strict coverage report",
                        "dataset_source": str(archive),
                        "coverage_mode": "strict",
                        "shots": ["1", "2"],
                        "models": [
                            {"key": "model_a", "label": "Model A", "results_txt": str(model_a)},
                            {"key": "model_b", "label": "Model B", "results_txt": str(model_b)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_root = tmp_path / "out"
            with self.assertRaisesRegex(ValueError, "Coverage validation failed"):
                generate_report(config_path, output_root=output_root)

            coverage_path = output_root / "strict_coverage_report" / "coverage_report.csv"
            self.assertTrue(coverage_path.exists())
            coverage = pd.read_csv(coverage_path)
            missing = coverage[
                (coverage["issue_type"] == "missing_result")
                & (coverage["model_key"] == "model_b")
                & (coverage["dataset"] == "Beta")
                & (coverage["shot"].astype(str) == "2")
            ]
            self.assertEqual(len(missing), 1)

    def test_intersection_mode_infers_sorted_shots_and_generates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha", "Beta"])

            model_a = tmp_path / "model_a_results.txt"
            model_b = tmp_path / "model_b_results.txt"
            _write_results(
                model_a,
                [
                    _make_row("Alpha", "10", "0.90"),
                    _make_row("Alpha", "2", "0.80"),
                    _make_row("Alpha", "1", "0.70"),
                    _make_row("Alpha", "full", "0.95"),
                    _make_row("Beta", "10", "0.88"),
                    _make_row("Beta", "2", "0.78"),
                    _make_row("Beta", "1", "0.68"),
                ],
            )
            _write_results(
                model_b,
                [
                    _make_row("Alpha", "2", "0.75"),
                    _make_row("Alpha", "10", "0.82"),
                    _make_row("Alpha", "1", "0.65"),
                    _make_row("Beta", "10", "0.80"),
                    _make_row("Beta", "1", "0.60"),
                ],
            )

            config_path = tmp_path / "intersection_report.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "intersection preview",
                        "dataset_source": str(archive),
                        "coverage_mode": "intersection",
                        "models": [
                            {"key": "model_a", "label": "Model A", "results_txt": str(model_a), "primary": True},
                            {"key": "model_b", "label": "Model B", "results_txt": str(model_b)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_root = tmp_path / "out"
            manifest = generate_report(config_path, output_root=output_root)

            self.assertEqual(manifest["shots"], ["1", "2", "10"])
            self.assertEqual(manifest["dataset_count"], 1)
            self.assertEqual(manifest["datasets"], ["Alpha"])

            report_dir = output_root / "intersection_preview"
            self.assertTrue((report_dir / "main_table.tex").exists())
            self.assertTrue((report_dir / "appendix_shot_1.tex").exists())
            self.assertTrue((report_dir / "appendix_shot_2.tex").exists())
            self.assertTrue((report_dir / "appendix_shot_10.tex").exists())
            self.assertTrue((report_dir / "fewshot_trend.png").exists())
            self.assertTrue((report_dir / "fewshot_trend.pdf").exists())

            merged = pd.read_csv(report_dir / "merged_results.csv")
            self.assertEqual(sorted(merged["dataset"].unique().tolist()), ["Alpha"])
            self.assertEqual(len(merged), 6)

            main_table = (report_dir / "main_table.tex").read_text(encoding="utf-8")
            self.assertIn("1-shot & 2-shot & 10-shot", main_table)

    def test_duplicate_success_rows_are_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha"])

            model_a = tmp_path / "model_a_results.txt"
            model_b = tmp_path / "model_b_results.txt"
            _write_results(
                model_a,
                [
                    _make_row("Alpha", "1", "0.70"),
                    _make_row("Alpha", "1", "0.71"),
                ],
            )
            _write_results(model_b, [_make_row("Alpha", "1", "0.69")])

            config_path = tmp_path / "duplicate_report.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "duplicate report",
                        "dataset_source": str(archive),
                        "coverage_mode": "intersection",
                        "shots": ["1"],
                        "models": [
                            {"key": "model_a", "label": "Model A", "results_txt": str(model_a)},
                            {"key": "model_b", "label": "Model B", "results_txt": str(model_b)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            output_root = tmp_path / "out"
            with self.assertRaisesRegex(ValueError, "Coverage validation failed"):
                generate_report(config_path, output_root=output_root)

            coverage = pd.read_csv(output_root / "duplicate_report" / "coverage_report.csv")
            duplicates = coverage[
                (coverage["issue_type"] == "duplicate_success") & (coverage["model_key"] == "model_a")
            ]
            self.assertEqual(len(duplicates), 1)

    def test_main_table_highlights_ties_and_second_best(self) -> None:
        summary_by_shot = pd.DataFrame(
            [
                {"model_key": "a", "model_label": "Model A", "shot": "1", "accuracy_mean_pct": 80.0},
                {"model_key": "b", "model_label": "Model B", "shot": "1", "accuracy_mean_pct": 80.0},
                {"model_key": "c", "model_label": "Model C", "shot": "1", "accuracy_mean_pct": 70.0},
                {"model_key": "a", "model_label": "Model A", "shot": "2", "accuracy_mean_pct": 80.0},
                {"model_key": "b", "model_label": "Model B", "shot": "2", "accuracy_mean_pct": 60.0},
                {"model_key": "c", "model_label": "Model C", "shot": "2", "accuracy_mean_pct": 90.0},
            ]
        )
        rank_summary = pd.DataFrame(
            [
                {"model_key": "a", "model_label": "Model A", "mean_rank": 1.5},
                {"model_key": "b", "model_label": "Model B", "mean_rank": 2.5},
                {"model_key": "c", "model_label": "Model C", "mean_rank": 2.0},
            ]
        )
        model_order = [
            {"key": "a", "label": "Model A"},
            {"key": "b", "label": "Model B"},
            {"key": "c", "label": "Model C"},
        ]

        rendered = render_main_table(
            report_name="tie demo",
            model_order=model_order,
            summary_by_shot=summary_by_shot,
            rank_summary=rank_summary,
            shots=["1", "2"],
        )

        self.assertIn(r"Model A & \textbf{80.00} & \underline{80.00}", rendered)
        self.assertIn(r"Model B & \textbf{80.00} & 60.00", rendered)
        self.assertIn(r"Model C & \underline{70.00} & \textbf{90.00}", rendered)


if __name__ == "__main__":
    unittest.main()
