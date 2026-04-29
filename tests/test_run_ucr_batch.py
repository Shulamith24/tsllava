from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from scripts.experiments.ucr_batch.registry import ExperimentEntry
from scripts.experiments.ucr_batch.run_ucr_batch import main as run_ucr_batch_main
import scripts.experiments.ucr_batch.run_ucr_batch as run_ucr_batch


SUMMARY_COLUMNS = [
    "shot",
    "num_runs",
    "accuracy_mean",
    "accuracy_std",
    "loss_mean",
    "loss_std",
    "support_size_mean",
    "support_size_std",
    "any_shortage_in_shot",
]

STUB_TRAINER = textwrap.dedent(
    """
    import argparse
    import csv
    import json
    import os
    import sys
    import time
    from pathlib import Path

    SUMMARY_COLUMNS = [
        "shot",
        "num_runs",
        "accuracy_mean",
        "accuracy_std",
        "loss_mean",
        "loss_std",
        "support_size_mean",
        "support_size_std",
        "any_shortage_in_shot",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--protocol", default="fewshot")
    parser.add_argument("--resume", action="store_true")
    args, _unknown = parser.parse_known_args()

    dataset_dir = Path(args.save_dir) / args.dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    time.sleep(float(os.environ.get(f"STUB_SLEEP_{args.dataset}", "0")))
    finished = time.time()

    payload = {
        "dataset": args.dataset,
        "started_at": start,
        "finished_at": finished,
        "pid": os.getpid(),
        "argv": sys.argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
            "RANK": os.environ.get("RANK"),
            "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
            "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
            "MASTER_PORT": os.environ.get("MASTER_PORT"),
        },
    }
    (dataset_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")

    fail_dataset = os.environ.get("STUB_FAIL_DATASET")
    if fail_dataset == args.dataset:
        raise SystemExit(int(os.environ.get("STUB_FAIL_EXIT_CODE", "7")))

    with open(dataset_dir / "fewshot_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "shot": "1",
                "num_runs": "1",
                "accuracy_mean": "0.9",
                "accuracy_std": "0.0",
                "loss_mean": "0.1",
                "loss_std": "0.0",
                "support_size_mean": "2",
                "support_size_std": "0",
                "any_shortage_in_shot": "False",
            }
        )

    print(f"finished {args.dataset}")
    """
)


def _write_ucr_archive(root: Path, datasets: list[str]) -> Path:
    archive = root / "data" / "UCRArchive_2018"
    archive.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        dataset_dir = archive / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / f"{dataset}_TRAIN.tsv").write_text("0\t0\n", encoding="utf-8")
        (dataset_dir / f"{dataset}_TEST.tsv").write_text("0\t0\n", encoding="utf-8")
    return archive


def _write_stub_trainer(path: Path) -> None:
    path.write_text(STUB_TRAINER, encoding="utf-8")


def _write_summary_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "shot": "1",
                "num_runs": "1",
                "accuracy_mean": "0.8",
                "accuracy_std": "0.0",
                "loss_mean": "0.2",
                "loss_std": "0.0",
                "support_size_mean": "2",
                "support_size_std": "0",
                "any_shortage_in_shot": "False",
            }
        )


def _read_results(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class UCRBatchRunnerTest(unittest.TestCase):
    def _make_entry(self, script_path: Path) -> ExperimentEntry:
        return ExperimentEntry(
            experiment="demo_exp",
            protocol="fewshot",
            script_path=script_path,
            summary_kind="fewshot",
            add_protocol_flag=False,
            supports_inner_resume=False,
            default_shots="1",
        )

    def _patch_runner(self, repo_root: Path, entry: ExperimentEntry):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(run_ucr_batch, "REPO_ROOT", repo_root))
        stack.enter_context(mock.patch.object(run_ucr_batch, "list_experiments", return_value=["demo_exp"]))
        stack.enter_context(mock.patch.object(run_ucr_batch, "get_entry", return_value=entry))
        return stack

    def test_parallel_workers_write_single_results_file_and_clean_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(tmp_path, ["Alpha", "Beta", "Gamma"])
            trainer = tmp_path / "stub_trainer.py"
            _write_stub_trainer(trainer)
            entry = self._make_entry(trainer)

            args = [
                "--experiment",
                "demo_exp",
                "--protocol",
                "fewshot",
                "--job-name",
                "parallel_job",
                "--data-path",
                str(archive.parent),
                "--datasets",
                "Alpha,Beta,Gamma",
                "--gpu-ids",
                "2,5",
                "--shots",
                "1",
            ]
            env = {
                "LOCAL_RANK": "3",
                "RANK": "7",
                "WORLD_SIZE": "8",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "23456",
                "STUB_SLEEP_Alpha": "0.35",
                "STUB_SLEEP_Beta": "0.05",
                "STUB_SLEEP_Gamma": "0.05",
            }

            with self._patch_runner(tmp_path, entry), mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(run_ucr_batch_main(args), 0)

            job_root = tmp_path / "results" / "ucr_batches" / "demo_exp" / "fewshot" / "parallel_job"
            self.assertTrue((job_root / "results.txt").exists())
            self.assertTrue((job_root / "batch_config.json").exists())
            self.assertTrue((job_root / "logs").is_dir())
            self.assertTrue((job_root / "datasets").is_dir())
            self.assertEqual({path.name for path in job_root.iterdir() if path.is_dir()}, {"datasets", "logs"})

            with open(job_root / "batch_config.json", "r", encoding="utf-8") as handle:
                batch_config = json.load(handle)
            self.assertEqual(batch_config["launcher"], "single_gpu_workers")
            self.assertEqual(batch_config["gpu_ids"], ["2", "5"])
            self.assertEqual(batch_config["worker_count"], 2)

            rows = _read_results(job_root / "results.txt")
            self.assertEqual(
                sorted((row["dataset"], row["shot"], row["status"]) for row in rows),
                [("Alpha", "1", "success"), ("Beta", "1", "success"), ("Gamma", "1", "success")],
            )

            alpha_run = json.loads((job_root / "datasets" / "Alpha" / "run.json").read_text(encoding="utf-8"))
            beta_run = json.loads((job_root / "datasets" / "Beta" / "run.json").read_text(encoding="utf-8"))
            gamma_run = json.loads((job_root / "datasets" / "Gamma" / "run.json").read_text(encoding="utf-8"))

            self.assertLess(gamma_run["started_at"], alpha_run["finished_at"])
            for run_payload in (alpha_run, beta_run, gamma_run):
                self.assertIn(run_payload["env"]["CUDA_VISIBLE_DEVICES"], {"2", "5"})
                self.assertIsNone(run_payload["env"]["LOCAL_RANK"])
                self.assertIsNone(run_payload["env"]["RANK"])
                self.assertIsNone(run_payload["env"]["WORLD_SIZE"])
                self.assertIsNone(run_payload["env"]["MASTER_ADDR"])
                self.assertIsNone(run_payload["env"]["MASTER_PORT"])

    def test_resume_skips_completed_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(tmp_path, ["Alpha", "Beta"])
            trainer = tmp_path / "stub_trainer.py"
            _write_stub_trainer(trainer)
            entry = self._make_entry(trainer)

            job_root = tmp_path / "results" / "ucr_batches" / "demo_exp" / "fewshot" / "resume_job"
            alpha_dir = job_root / "datasets" / "Alpha"
            _write_summary_csv(alpha_dir / "fewshot_summary.csv")
            (job_root / "logs").mkdir(parents=True, exist_ok=True)
            (job_root / "logs" / "Alpha.log").write_text("existing\n", encoding="utf-8")

            args = [
                "--experiment",
                "demo_exp",
                "--protocol",
                "fewshot",
                "--job-name",
                "resume_job",
                "--data-path",
                str(archive.parent),
                "--datasets",
                "Alpha,Beta",
                "--gpu-ids",
                "0",
                "--shots",
                "1",
            ]

            with self._patch_runner(tmp_path, entry):
                self.assertEqual(run_ucr_batch_main(args), 0)

            self.assertFalse((job_root / "datasets" / "Alpha" / "run.json").exists())
            self.assertTrue((job_root / "datasets" / "Beta" / "run.json").exists())
            rows = _read_results(job_root / "results.txt")
            self.assertEqual(
                sorted((row["dataset"], row["shot"], row["status"]) for row in rows),
                [("Alpha", "1", "success"), ("Beta", "1", "success")],
            )

    def test_failure_row_does_not_stop_other_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(tmp_path, ["Alpha", "FailSet", "Gamma"])
            trainer = tmp_path / "stub_trainer.py"
            _write_stub_trainer(trainer)
            entry = self._make_entry(trainer)

            args = [
                "--experiment",
                "demo_exp",
                "--protocol",
                "fewshot",
                "--job-name",
                "failure_job",
                "--data-path",
                str(archive.parent),
                "--datasets",
                "Alpha,FailSet,Gamma",
                "--gpu-ids",
                "0,1",
                "--shots",
                "1",
            ]
            env = {
                "STUB_FAIL_DATASET": "FailSet",
                "STUB_FAIL_EXIT_CODE": "9",
                "STUB_SLEEP_Alpha": "0.05",
                "STUB_SLEEP_FailSet": "0.01",
                "STUB_SLEEP_Gamma": "0.05",
            }

            with self._patch_runner(tmp_path, entry), mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(run_ucr_batch_main(args), 0)

            rows = _read_results(
                tmp_path / "results" / "ucr_batches" / "demo_exp" / "fewshot" / "failure_job" / "results.txt"
            )
            self.assertEqual(
                sorted((row["dataset"], row["shot"], row["status"]) for row in rows),
                [
                    ("Alpha", "1", "success"),
                    ("FailSet", "__dataset__", "failed"),
                    ("Gamma", "1", "success"),
                ],
            )

    def test_external_dataset_family_uses_registered_external_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trainer = tmp_path / "stub_trainer.py"
            _write_stub_trainer(trainer)
            entry = self._make_entry(trainer)

            args = [
                "--experiment",
                "demo_exp",
                "--protocol",
                "fewshot",
                "--job-name",
                "external_family_job",
                "--data-path",
                str(tmp_path / "data"),
                "--shots",
                "1",
                "--dataset_family",
                "mitbih",
            ]

            with self._patch_runner(tmp_path, entry):
                self.assertEqual(run_ucr_batch_main(args), 0)

            job_root = tmp_path / "results" / "ucr_batches" / "demo_exp" / "fewshot" / "external_family_job"
            rows = _read_results(job_root / "results.txt")
            self.assertEqual(
                [(row["dataset"], row["shot"], row["status"]) for row in rows],
                [("MITBIHArrhythmia", "1", "success")],
            )

            run_payload = json.loads(
                (job_root / "datasets" / "MITBIHArrhythmia" / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_payload["dataset"], "MITBIHArrhythmia")
            self.assertIn("--dataset_family", run_payload["argv"])
            self.assertIn("mitbih", run_payload["argv"])

    def test_job_lock_blocks_second_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _write_ucr_archive(tmp_path, ["Alpha"])
            trainer = tmp_path / "stub_trainer.py"
            _write_stub_trainer(trainer)
            entry = self._make_entry(trainer)

            job_root = tmp_path / "results" / "ucr_batches" / "demo_exp" / "fewshot" / "locked_job"
            job_root.mkdir(parents=True, exist_ok=True)
            lock_path = job_root / ".job.lock"
            ready_path = tmp_path / "lock_ready"
            holder = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    textwrap.dedent(
                        """
                        import fcntl
                        import pathlib
                        import sys
                        import time

                        lock_path = pathlib.Path(sys.argv[1])
                        ready_path = pathlib.Path(sys.argv[2])
                        with open(lock_path, "a+", encoding="utf-8") as handle:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                            ready_path.write_text("ready", encoding="utf-8")
                            time.sleep(5)
                        """
                    ),
                    str(lock_path),
                    str(ready_path),
                ]
            )
            try:
                deadline = time.time() + 5
                while not ready_path.exists():
                    if time.time() > deadline:
                        self.fail("lock holder did not become ready in time")
                    time.sleep(0.05)

                args = [
                    "--experiment",
                    "demo_exp",
                    "--protocol",
                    "fewshot",
                    "--job-name",
                    "locked_job",
                    "--data-path",
                    str(archive.parent),
                    "--datasets",
                    "Alpha",
                    "--gpu-ids",
                    "0",
                    "--shots",
                    "1",
                ]
                with self._patch_runner(tmp_path, entry):
                    with self.assertRaisesRegex(RuntimeError, "already using job-name 'locked_job'"):
                        run_ucr_batch_main(args)
            finally:
                holder.terminate()
                holder.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
