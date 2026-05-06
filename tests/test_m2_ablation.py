from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.experiments.ucr_batch.build_m2_pretrain_ablation import main as build_pretrain_ablation_main
import scripts.experiments.ucr_batch.build_m2_pretrain_ablation as build_pretrain_ablation
from scripts.experiments.ucr_batch.m2_ablation_common import (
    build_forward_args_from_reference,
    resolve_selected_datasets,
    shared_complete_datasets,
)
from scripts.experiments.ucr_batch.run_m2_dual_view_ablation import main as run_dual_view_ablation_main
import scripts.experiments.ucr_batch.run_m2_dual_view_ablation as run_dual_view_ablation
from scripts.experiments.ucr_batch.run_m2_no_llm_backbone_ablation import main as run_no_llm_ablation_main
import scripts.experiments.ucr_batch.run_m2_no_llm_backbone_ablation as run_no_llm_ablation


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
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _make_row(dataset: str, shot: str, accuracy: str = "0.80") -> dict[str, str]:
    return {
        "dataset": dataset,
        "shot": shot,
        "status": "success",
        "accuracy": accuracy,
        "accuracy_std": "0.01",
        "num_runs": "3",
        "result_file": "",
        "log_file": "",
        "updated_at": "2026-04-24T00:00:00Z",
        "note": "",
    }


def _write_job_dir(path: Path, rows: list[dict[str, str]], *, forward_args: list[str] | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _write_results(path / "results.txt", rows)
    (path / "batch_config.json").write_text(
        json.dumps(
            {
                "experiment": "m2_pretrained",
                "protocol": "fewshot",
                "summary_kind": "fewshot",
                "forward_args": list(forward_args or []),
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_ucr_dataset(
    archive: Path,
    dataset: str,
    *,
    train_size: int,
    test_size: int,
    series_length: int,
    num_classes: int,
) -> None:
    dataset_dir = archive / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def _write_split(path: Path, size: int) -> None:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            for idx in range(size):
                label = idx % num_classes
                values = [f"{float(step):.1f}" for step in range(series_length)]
                writer.writerow([label, *values])

    _write_split(dataset_dir / f"{dataset}_TRAIN.tsv", train_size)
    _write_split(dataset_dir / f"{dataset}_TEST.tsv", test_size)


def _write_ucr_archive(root: Path, specs: dict[str, tuple[int, int, int, int]]) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    archive.mkdir(parents=True, exist_ok=True)
    for dataset, (train_size, test_size, series_length, num_classes) in specs.items():
        _write_ucr_dataset(
            archive,
            dataset,
            train_size=train_size,
            test_size=test_size,
            series_length=series_length,
            num_classes=num_classes,
        )
    return archive


def _complete_rows(datasets: list[str], *, shots: tuple[str, ...] = ("1", "2", "5", "10")) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset in datasets:
        for shot in shots:
            rows.append(_make_row(dataset, shot))
    return rows


class M2AblationTest(unittest.TestCase):
    def test_build_forward_args_from_reference_overrides_managed_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            batch_config_path = tmp_path / "batch_config.json"
            batch_config_path.write_text(
                json.dumps(
                    {
                        "forward_args": [
                            "--local_checkpoint",
                            "/old/checkpoint.pt",
                            "--shots",
                            "1,2",
                            "--num_runs",
                            "5",
                            "--runtime_branch_mode",
                            "both",
                            "--disable_constrained_decoding",
                            "--resume",
                            "--epochs",
                            "60",
                            "--gradient_checkpointing",
                        ]
                    }
                ),
                encoding="utf-8",
            )

            forward_args = build_forward_args_from_reference(
                reference_batch_config_path=batch_config_path,
                local_checkpoint="/new/checkpoint.pt",
                shots=("1", "2", "5", "10"),
                num_runs=3,
                runtime_branch_mode="ts_only",
            )
            self.assertNotIn("/old/checkpoint.pt", forward_args)
            self.assertNotIn("--resume", forward_args)
            self.assertNotIn("--disable_constrained_decoding", forward_args)
            self.assertIn("--epochs", forward_args)
            self.assertIn("--gradient_checkpointing", forward_args)
            self.assertIn("/new/checkpoint.pt", forward_args)
            self.assertIn("1,2,5,10", forward_args)
            self.assertIn("3", forward_args)
            self.assertIn("ts_only", forward_args)

            unconstrained_args = build_forward_args_from_reference(
                reference_batch_config_path=batch_config_path,
                local_checkpoint="/new/checkpoint.pt",
                disable_constrained_decoding=True,
            )
            self.assertIn("--disable_constrained_decoding", unconstrained_args)

    def test_resolve_selected_datasets_is_deterministic_and_honors_explicit_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(
                tmp_path,
                {
                    "Alpha": (12, 10, 32, 2),
                    "Beta": (20, 8, 24, 3),
                    "Gamma": (8, 8, 40, 2),
                    "Delta": (15, 10, 18, 4),
                    "Epsilon": (30, 12, 28, 5),
                    "Zeta": (10, 10, 36, 3),
                },
            )
            candidate = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]

            selected_a, mode_a, _metadata_a = resolve_selected_datasets(
                candidate_datasets=candidate,
                dataset_source=archive,
                num_datasets=3,
                sampling_seed=7,
            )
            selected_b, mode_b, _metadata_b = resolve_selected_datasets(
                candidate_datasets=candidate,
                dataset_source=archive,
                num_datasets=3,
                sampling_seed=7,
            )
            self.assertEqual(mode_a, "stratified")
            self.assertEqual(mode_b, "stratified")
            self.assertEqual(selected_a, selected_b)

            explicit_selected, explicit_mode, _metadata = resolve_selected_datasets(
                candidate_datasets=candidate,
                dataset_source=archive,
                num_datasets=3,
                sampling_seed=7,
                dataset_list=["Zeta", "Alpha"],
            )
            self.assertEqual(explicit_mode, "explicit_list")
            self.assertEqual(explicit_selected, ["Zeta", "Alpha"])

    def test_shared_complete_datasets_intersects_successful_shots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(
                tmp_path,
                {
                    "Alpha": (4, 4, 8, 2),
                    "Beta": (4, 4, 8, 2),
                    "Gamma": (4, 4, 8, 2),
                },
            )
            job_a = _write_job_dir(
                tmp_path / "job_a",
                _complete_rows(["Alpha", "Beta", "Gamma"]),
            )
            job_b_rows = _complete_rows(["Alpha", "Gamma"])
            job_b_rows.append(_make_row("Beta", "1"))
            job_b = _write_job_dir(tmp_path / "job_b", job_b_rows)

            shared = shared_complete_datasets(
                [job_a / "results.txt", job_b / "results.txt"],
                dataset_source=archive,
            )
            self.assertEqual(shared, ["Alpha", "Gamma"])

    def test_dual_view_launcher_runs_only_variant_jobs_and_writes_report_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(
                tmp_path,
                {
                    "Alpha": (6, 6, 12, 2),
                    "Beta": (8, 6, 16, 3),
                    "Gamma": (10, 8, 20, 4),
                },
            )
            forward_args = [
                "--local_checkpoint",
                "/old/checkpoint.pt",
                "--epochs",
                "60",
                "--num_runs",
                "5",
                "--shots",
                "1,2,5,10",
                "--runtime_branch_mode",
                "both",
                "--resume",
            ]
            reference_job = _write_job_dir(tmp_path / "reference_job", _complete_rows(["Alpha", "Beta", "Gamma"]), forward_args=forward_args)
            comparison_job = _write_job_dir(tmp_path / "comparison_job", _complete_rows(["Alpha", "Beta", "Gamma"]))

            captured_argvs: list[list[str]] = []

            def _fake_run_ucr_batch(argv):
                captured_argvs.append(list(argv))
                return 0

            report_root = tmp_path / "reports"
            with mock.patch.object(run_dual_view_ablation, "run_ucr_batch_main", side_effect=_fake_run_ucr_batch), mock.patch.object(
                run_dual_view_ablation,
                "default_report_dir",
                side_effect=lambda report_name: report_root / report_name.replace(" ", "_"),
            ):
                exit_code = run_dual_view_ablation_main(
                    [
                        "--local_checkpoint",
                        "/new/checkpoint.pt",
                        "--data_path",
                        str(archive.parent),
                        "--reference_job_dir",
                        str(reference_job),
                        "--comparison_job_dir",
                        str(comparison_job),
                        "--num_datasets",
                        "2",
                        "--sampling_seed",
                        "11",
                        "--report_name",
                        "dual_view_test",
                        "--dry_run",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_argvs), 2)
            self.assertTrue(all("--job-name" in argv for argv in captured_argvs))
            self.assertTrue(any("ts_only" in argv[argv.index("--job-name") + 1] for argv in captured_argvs))
            self.assertTrue(any("vision_only" in argv[argv.index("--job-name") + 1] for argv in captured_argvs))
            self.assertFalse(any("both" in argv[argv.index("--job-name") + 1] for argv in captured_argvs))

            report_dir = report_root / "dual_view_test"
            config = json.loads((report_dir / "report_config.generated.json").read_text(encoding="utf-8"))
            self.assertEqual(config["reference_key"], "both")
            self.assertEqual([item["key"] for item in config["items"]], ["both", "ts_only", "vision_only"])
            self.assertEqual(config["items"][0]["job_dir"], str(reference_job.resolve()))
            subset_manifest = json.loads((report_dir / "subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subset_manifest["selected_dataset_count"], 2)

    def test_no_llm_launcher_runs_linear_variant_and_writes_report_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(
                tmp_path,
                {
                    "Alpha": (6, 6, 12, 2),
                    "Beta": (8, 6, 16, 3),
                    "Gamma": (10, 8, 20, 4),
                },
            )
            forward_args = [
                "--local_checkpoint",
                "/old/checkpoint.pt",
                "--epochs",
                "60",
                "--num_runs",
                "5",
                "--shots",
                "1,2,5,10",
                "--resume",
            ]
            reference_job = _write_job_dir(tmp_path / "reference_job", _complete_rows(["Alpha", "Beta", "Gamma"]), forward_args=forward_args)
            comparison_job = _write_job_dir(tmp_path / "comparison_job", _complete_rows(["Alpha", "Beta", "Gamma"]))

            captured_argvs: list[list[str]] = []

            def _fake_run_ucr_batch(argv):
                captured_argvs.append(list(argv))
                return 0

            report_root = tmp_path / "reports"
            with mock.patch.object(run_no_llm_ablation, "run_ucr_batch_main", side_effect=_fake_run_ucr_batch), mock.patch.object(
                run_no_llm_ablation,
                "default_report_dir",
                side_effect=lambda report_name: report_root / report_name.replace(" ", "_"),
            ):
                exit_code = run_no_llm_ablation_main(
                    [
                        "--local_checkpoint",
                        "/new/checkpoint.pt",
                        "--data_path",
                        str(archive.parent),
                        "--reference_job_dir",
                        str(reference_job),
                        "--comparison_job_dir",
                        str(comparison_job),
                        "--num_datasets",
                        "2",
                        "--sampling_seed",
                        "11",
                        "--report_name",
                        "no_llm_test",
                        "--start_from",
                        "Beta",
                        "--dry_run",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(captured_argvs), 1)
            argv = captured_argvs[0]
            self.assertIn("m2_no_llm_linear", argv)
            self.assertIn("/new/checkpoint.pt", argv)
            self.assertNotIn("/old/checkpoint.pt", argv)
            self.assertIn("1,2,5,10", argv)
            self.assertEqual(argv[argv.index("--start-from") + 1], "Beta")

            report_dir = report_root / "no_llm_test"
            config = json.loads((report_dir / "report_config.generated.json").read_text(encoding="utf-8"))
            self.assertEqual(config["family_label"], "Effect of LLM Backbone")
            self.assertEqual(config["reference_key"], "m2_llm")
            self.assertEqual([item["key"] for item in config["items"]], ["m2_llm", "no_llm_linear"])
            self.assertEqual(config["items"][0]["job_dir"], str(reference_job.resolve()))
            subset_manifest = json.loads((report_dir / "subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subset_manifest["selected_dataset_count"], 2)
            request_payload = json.loads((report_dir / "ablation_request.json").read_text(encoding="utf-8"))
            self.assertEqual(request_payload["start_from"], "Beta")

    def test_pretrain_builder_auto_falls_back_to_two_way(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(
                tmp_path,
                {
                    "Alpha": (6, 6, 12, 2),
                    "Beta": (8, 6, 16, 3),
                    "Gamma": (10, 8, 20, 4),
                    "Delta": (12, 8, 24, 5),
                },
            )
            without_pretrain_job = _write_job_dir(tmp_path / "without_pretrain", _complete_rows(["Alpha", "Beta", "Gamma", "Delta"]))
            full_curriculum_job = _write_job_dir(tmp_path / "fewshot_second", _complete_rows(["Alpha", "Beta", "Gamma", "Delta"]))
            stage012_rows = _complete_rows(["Alpha", "Beta"])
            stage012_job = _write_job_dir(tmp_path / "stage012", stage012_rows)

            report_root = tmp_path / "reports"
            with mock.patch.object(
                build_pretrain_ablation,
                "default_report_dir",
                side_effect=lambda report_name: report_root / report_name.replace(" ", "_"),
            ):
                exit_code = build_pretrain_ablation_main(
                    [
                        "--data_path",
                        str(archive.parent),
                        "--without_pretrain_job_dir",
                        str(without_pretrain_job),
                        "--stage012_job_dir",
                        str(stage012_job),
                        "--full_curriculum_job_dir",
                        str(full_curriculum_job),
                        "--num_datasets",
                        "3",
                        "--report_name",
                        "pretrain_test",
                        "--dry_run",
                    ]
                )
            self.assertEqual(exit_code, 0)

            report_dir = report_root / "pretrain_test"
            request_payload = json.loads((report_dir / "ablation_request.json").read_text(encoding="utf-8"))
            self.assertEqual(request_payload["variant_mode"], "two_way")
            config = json.loads((report_dir / "report_config.generated.json").read_text(encoding="utf-8"))
            self.assertEqual([item["key"] for item in config["items"]], ["without_pretrain", "fewshot_second"])
            subset_manifest = json.loads((report_dir / "subset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(subset_manifest["selected_dataset_count"], 3)


if __name__ == "__main__":
    unittest.main()
