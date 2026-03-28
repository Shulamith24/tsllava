from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.experiments.ucr_batch.reporting.config import load_report_config
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


def _write_job_dir(
    path: Path,
    rows: list[dict[str, str]],
    *,
    experiment: str = "demo_exp",
    protocol: str = "fewshot",
    summary_kind: str = "fewshot",
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_results(path / "results.txt", rows)
    (path / "batch_config.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "protocol": protocol,
                "summary_kind": summary_kind,
            }
        ),
        encoding="utf-8",
    )
    return path


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


class UCRReportingTest(unittest.TestCase):
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

    def test_ablation_config_supports_job_dir_and_stage_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha"])
            ref_job = _write_job_dir(tmp_path / "ref_job", [_make_row("Alpha", "1", "0.70")])
            variant_job = _write_job_dir(tmp_path / "variant_job", [_make_row("Alpha", "1", "0.80")])

            config_path = tmp_path / "ablation_preview.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "ablation preview",
                        "report_kind": "ablation",
                        "report_stage": "preview",
                        "family_label": "Pretraining",
                        "reference_key": "without_pretrain",
                        "dataset_source": str(archive),
                        "shots": ["1"],
                        "items": [
                            {"key": "fewshot_second", "label": "With Pretrain", "job_dir": str(variant_job)},
                            {"key": "without_pretrain", "label": "Without Pretrain", "job_dir": str(ref_job)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_report_config(config_path)
            self.assertEqual(config.report_kind, "ablation")
            self.assertEqual(config.report_stage, "preview")
            self.assertEqual(config.coverage_mode, "intersection")
            self.assertEqual(config.reference_key, "without_pretrain")
            self.assertEqual(config.items[0].results_txt, (variant_job / "results.txt").resolve())
            self.assertEqual(config.items[0].job_dir, variant_job.resolve())
            self.assertEqual(config.items[0].batch_config_path, (variant_job / "batch_config.json").resolve())

    def test_ablation_preview_uses_intersection_and_writes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha", "Beta"])
            ref_job = _write_job_dir(
                tmp_path / "ref_job",
                [
                    _make_row("Alpha", "1", "0.70"),
                    _make_row("Alpha", "2", "0.80"),
                    _make_row("Beta", "1", "0.50"),
                    _make_row("Beta", "2", "0.60"),
                ],
            )
            variant_job = _write_job_dir(
                tmp_path / "variant_job",
                [
                    _make_row("Alpha", "1", "0.80"),
                    _make_row("Alpha", "2", "0.90"),
                ],
            )

            config_path = tmp_path / "ablation_preview.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "pretrain ablation preview",
                        "report_kind": "ablation",
                        "report_stage": "preview",
                        "family_label": "Pretraining",
                        "reference_key": "without_pretrain",
                        "dataset_source": str(archive),
                        "shots": ["1", "2"],
                        "items": [
                            {
                                "key": "fewshot_second",
                                "label": "With Pretrain",
                                "job_dir": str(variant_job),
                                "primary": True,
                            },
                            {
                                "key": "without_pretrain",
                                "label": "Without Pretrain",
                                "job_dir": str(ref_job),
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = generate_report(config_path, output_root=tmp_path / "out")

            self.assertEqual(manifest["report_kind"], "ablation")
            self.assertEqual(manifest["report_stage"], "preview")
            self.assertEqual(manifest["dataset_count"], 1)
            self.assertEqual(manifest["shared_dataset_count"], 1)
            self.assertEqual(manifest["datasets"], ["Alpha"])

            report_dir = tmp_path / "out" / "pretrain_ablation_preview"
            self.assertTrue((report_dir / "ablation_summary.csv").exists())
            self.assertTrue((report_dir / "cell_deltas.csv").exists())
            self.assertTrue((report_dir / "main_table.tex").exists())
            self.assertTrue((report_dir / "ablation_trend.png").exists())
            self.assertTrue((report_dir / "ablation_trend.pdf").exists())

            summary = pd.read_csv(report_dir / "ablation_summary.csv")
            variant_row = summary[summary["model_key"] == "fewshot_second"].iloc[0]
            self.assertAlmostEqual(float(variant_row["delta_vs_reference_pct"]), 10.0, places=6)
            self.assertEqual(str(variant_row["win_tie_loss"]), "2/0/0")

            main_table = (report_dir / "main_table.tex").read_text(encoding="utf-8")
            self.assertIn("Preview uses the 1 shared datasets currently available.", main_table)
            self.assertIn("+10.00", main_table)

    def test_ablation_final_writes_coverage_report_before_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha", "Beta"])
            ref_job = _write_job_dir(
                tmp_path / "ref_job",
                [
                    _make_row("Alpha", "1", "0.70"),
                    _make_row("Alpha", "2", "0.80"),
                    _make_row("Beta", "1", "0.50"),
                    _make_row("Beta", "2", "0.60"),
                ],
            )
            variant_job = _write_job_dir(
                tmp_path / "variant_job",
                [
                    _make_row("Alpha", "1", "0.80"),
                    _make_row("Alpha", "2", "0.90"),
                ],
            )

            config_path = tmp_path / "ablation_final.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "pretrain ablation final",
                        "report_kind": "ablation",
                        "report_stage": "final",
                        "family_label": "Pretraining",
                        "reference_key": "without_pretrain",
                        "dataset_source": str(archive),
                        "shots": ["1", "2"],
                        "items": [
                            {"key": "fewshot_second", "label": "With Pretrain", "job_dir": str(variant_job)},
                            {"key": "without_pretrain", "label": "Without Pretrain", "job_dir": str(ref_job)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Coverage validation failed"):
                generate_report(config_path, output_root=tmp_path / "out")

            coverage = pd.read_csv(tmp_path / "out" / "pretrain_ablation_final" / "coverage_report.csv")
            missing = coverage[
                (coverage["issue_type"] == "missing_result")
                & (coverage["model_key"] == "fewshot_second")
                & (coverage["dataset"] == "Beta")
            ]
            self.assertEqual(len(missing), 2)

    def test_ablation_supports_three_variants_and_signed_appendix_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _make_ucr_archive(tmp_path, ["Alpha", "Beta"])
            ref_job = _write_job_dir(
                tmp_path / "ref_job",
                [
                    _make_row("Alpha", "1", "0.50"),
                    _make_row("Alpha", "2", "0.60"),
                    _make_row("Beta", "1", "0.70"),
                    _make_row("Beta", "2", "0.80"),
                ],
            )
            variant_a_job = _write_job_dir(
                tmp_path / "variant_a_job",
                [
                    _make_row("Alpha", "1", "0.60"),
                    _make_row("Alpha", "2", "0.60"),
                    _make_row("Beta", "1", "0.65"),
                    _make_row("Beta", "2", "0.90"),
                ],
            )
            variant_b_job = _write_job_dir(
                tmp_path / "variant_b_job",
                [
                    _make_row("Alpha", "1", "0.40"),
                    _make_row("Alpha", "2", "0.65"),
                    _make_row("Beta", "1", "0.70"),
                    _make_row("Beta", "2", "0.70"),
                ],
            )

            config_path = tmp_path / "ablation_three_way.json"
            config_path.write_text(
                json.dumps(
                    {
                        "report_name": "three way ablation",
                        "report_kind": "ablation",
                        "report_stage": "final",
                        "family_label": "Component Study",
                        "reference_key": "ref",
                        "dataset_source": str(archive),
                        "shots": ["1", "2"],
                        "items": [
                            {"key": "variant_a", "label": "Variant A", "job_dir": str(variant_a_job)},
                            {"key": "variant_b", "label": "Variant B", "job_dir": str(variant_b_job)},
                            {"key": "ref", "label": "Reference", "job_dir": str(ref_job)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = generate_report(config_path, output_root=tmp_path / "out")
            self.assertEqual(manifest["dataset_count"], 2)

            report_dir = tmp_path / "out" / "three_way_ablation"
            summary = pd.read_csv(report_dir / "ablation_summary.csv").set_index("model_key")
            self.assertAlmostEqual(float(summary.loc["variant_a", "delta_vs_reference_pct"]), 3.75, places=6)
            self.assertAlmostEqual(float(summary.loc["variant_b", "delta_vs_reference_pct"]), -3.75, places=6)
            self.assertEqual(str(summary.loc["variant_a", "win_tie_loss"]), "2/1/1")
            self.assertEqual(str(summary.loc["variant_b", "win_tie_loss"]), "1/1/2")
            self.assertTrue((report_dir / "ablation_trend.png").exists())
            self.assertTrue((report_dir / "ablation_trend.pdf").exists())

            appendix = (report_dir / "appendix_shot_1.tex").read_text(encoding="utf-8")
            self.assertLess(appendix.find("Alpha"), appendix.find("Beta"))
            self.assertIn("+10.00", appendix)
            self.assertIn("-10.00", appendix)


if __name__ == "__main__":
    unittest.main()
