#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

"""
#onefitsall fewshot example:
python scripts/experiments/ucr_batch/run_ucr_batch.py \
    --experiment onefitsall \
    --protocol fewshot \
    --job-name my_onefitsall_fewshot \
    --shots "1,2,5,10" \
    --num_runs 5 \
    --epochs 60 \
    --cleanup_checkpoints
tslib_former example:
export CUDA_VISIBLE_DEVICES=0 uv run python scripts/experiments/ucr_batch/run_ucr_batch.py \
    --experiment tslib_crossformer \
    --protocol fewshot \
    --job-name my_timesnet_fewshot \
    --gpu-ids 0,1 \
    --shots "1,2,5,10" \
    --num_runs 5


"""


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ledger import (  # noqa: E402
    build_failure_row,
    has_success,
    load_ledger,
    remove_row,
    success_result_file_exists,
    upsert_row,
    write_ledger,
)
from registry import REPO_ROOT, get_entry, list_experiments  # noqa: E402
from summary_parsers import parse_summary_rows  # noqa: E402
from ucr_datasets import discover_datasets, resolve_ucr_archive  # noqa: E402

OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
)
DDP_ENV_VARS = (
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)

EXTERNAL_DATASETS = {
    "mitbih": ["MITBIHArrhythmia"],
    "sleepedf": ["SleepEDFCassette"],
    "cinc2017af": ["CinC2017AF"],
    "cinc2016heart": ["CinC2016HeartSound"],
}


@dataclass(frozen=True)
class WorkerMessage:
    kind: str
    worker_id: int
    gpu_id: str | None
    dataset: str | None = None
    return_code: int | None = None
    log_path: Path | None = None
    error_message: str | None = None


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Run UCR batch experiments with resume-aware bookkeeping. "
            "Unknown extra args are forwarded to the underlying training script, "
            "for example --cleanup_checkpoints."
        )
    )
    parser.add_argument("--experiment", required=True, choices=list_experiments())
    parser.add_argument("--protocol", required=True, choices=["full", "fewshot"])
    parser.add_argument("--job-name", default="default")
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data"))
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset allowlist.")
    parser.add_argument("--exclude-datasets", default=None, help="Comma-separated dataset blocklist.")
    parser.add_argument("--start-from", default=None, help="Start execution from this dataset name.")
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated CUDA device ids for parallel workers.")
    parser.add_argument("--dry-run", action="store_true")
    args, forward_args = parser.parse_known_args(argv)
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    return args, forward_args


def parse_gpu_ids(raw_value: str | None) -> List[str]:
    if raw_value is None:
        return []

    gpu_ids: List[str] = []
    seen: set[str] = set()
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            raise ValueError("--gpu-ids must not contain empty items")
        if not stripped.isdigit():
            raise ValueError(f"--gpu-ids must be numeric CUDA device ids, got: {stripped}")
        if stripped in seen:
            raise ValueError(f"--gpu-ids contains duplicate device id: {stripped}")
        seen.add(stripped)
        gpu_ids.append(stripped)
    return gpu_ids


def parse_csv_arg(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def cli_option_value(argv: Sequence[str], flag_name: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag_name and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(f"{flag_name}="):
            return token.split("=", 1)[1]
    return None


def resolve_dataset_family(forward_args: Sequence[str]) -> str:
    return (cli_option_value(forward_args, "--dataset_family") or "ucr").strip().lower()


def validate_forward_args(forward_args: Sequence[str], blocked_flags: Iterable[str]) -> None:
    blocked = set(blocked_flags)
    matches = sorted({token.split("=", 1)[0] for token in forward_args if token.split("=", 1)[0] in blocked})
    if matches:
        raise ValueError(
            "These arguments are managed by the batch runner and must not be forwarded: "
            + ", ".join(matches)
        )


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def filter_datasets(
    datasets: List[str],
    *,
    allowlist: List[str],
    denylist: List[str],
    start_from: str | None,
) -> List[str]:
    available = set(datasets)
    if allowlist:
        missing = [item for item in allowlist if item not in available]
        if missing:
            raise ValueError(f"Unknown datasets in --datasets: {', '.join(missing)}")
        datasets = [item for item in allowlist]

    if denylist:
        missing = [item for item in denylist if item not in available]
        if missing:
            raise ValueError(f"Unknown datasets in --exclude-datasets: {', '.join(missing)}")
        denyset = set(denylist)
        datasets = [item for item in datasets if item not in denyset]

    if start_from:
        if start_from not in datasets:
            raise ValueError(f"--start-from dataset not found after filtering: {start_from}")
        datasets = datasets[datasets.index(start_from) :]
    return datasets


def requested_shots(protocol: str, entry_default_shots: str | None, forward_args: Sequence[str]) -> List[str]:
    if protocol == "full":
        return ["full"]
    shots = cli_option_value(forward_args, "--shots") or entry_default_shots or "full"
    return [item.strip() for item in shots.split(",") if item.strip()]


def dataset_is_complete(ledger, dataset: str, shots: Sequence[str]) -> bool:
    for shot in shots:
        if not has_success(ledger, dataset, shot):
            return False
        if not success_result_file_exists(ledger, dataset, shot):
            return False
    return True


def build_batch_config(args, entry, resolved_data_path: Path, forward_args: Sequence[str]) -> dict:
    worker_count = len(args.gpu_ids) if args.gpu_ids else 1
    return {
        "experiment": args.experiment,
        "protocol": args.protocol,
        "script_path": str(entry.script_path.resolve()),
        "data_path": str(resolved_data_path),
        "launcher": "single_gpu_workers" if args.gpu_ids else "single_process",
        "gpu_ids": list(args.gpu_ids),
        "worker_count": worker_count,
        "fixed_args": list(entry.fixed_args),
        "forward_args": list(forward_args),
        "supports_inner_resume": entry.supports_inner_resume,
        "summary_kind": entry.summary_kind,
    }


RUNTIME_BATCH_CONFIG_KEYS = {"launcher", "gpu_ids", "worker_count"}


def semantic_batch_config(config_payload: dict) -> dict:
    return {
        key: value
        for key, value in config_payload.items()
        if key not in RUNTIME_BATCH_CONFIG_KEYS
    }


def ensure_batch_config(config_path: Path, config_payload: dict) -> None:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if semantic_batch_config(existing) != semantic_batch_config(config_payload):
            raise ValueError(
                f"Existing batch config at {config_path} does not match this run. "
                "Use a new --job-name for a different experiment configuration."
            )
        if existing != config_payload:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config_payload, f, indent=2)
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)


def backfill_existing_results(ledger, *, datasets: Sequence[str], logs_dir: Path, datasets_dir: Path, summary_kind: str) -> None:
    for dataset in datasets:
        dataset_dir = datasets_dir / dataset
        log_file = logs_dir / f"{dataset}.log"
        rows = parse_summary_rows(
            dataset=dataset,
            dataset_dir=dataset_dir,
            log_file=log_file,
            summary_kind=summary_kind,
        )
        for row in rows:
            upsert_row(ledger, row)
            remove_row(ledger, dataset, "__dataset__")


def build_command(
    entry,
    *,
    protocol: str,
    dataset: str,
    data_path: str,
    save_dir: str,
    forward_args: Sequence[str],
) -> List[str]:
    command = [sys.executable, str(entry.script_path)]
    if entry.add_protocol_flag:
        command.extend(["--protocol", protocol])
    if entry.supports_inner_resume:
        command.append("--resume")
    command.extend(entry.fixed_args)
    command.extend(["--dataset", dataset, "--data_path", data_path, "--save_dir", save_dir])
    command.extend(forward_args)
    return command


def detect_failure_status(log_path: Path) -> str:
    if not log_path.exists():
        return "failed"
    content = log_path.read_text(encoding="utf-8", errors="ignore")
    for pattern in OOM_PATTERNS:
        if pattern in content:
            return "oom"
    return "failed"


def build_subprocess_env(gpu_id: str | None) -> dict[str, str]:
    env = os.environ.copy()
    for key in DDP_ENV_VARS:
        env.pop(key, None)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    return env


def run_dataset(command: Sequence[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        if env is not None and "CUDA_VISIBLE_DEVICES" in env:
            log_file.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        log_file.write("COMMAND: " + " ".join(shlex.quote(token) for token in command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            check=False,
        )
    return completed.returncode


def describe_lock_owner(lock_path: Path) -> str | None:
    try:
        raw_pid = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw_pid:
        return None
    try:
        pid = int(raw_pid)
    except ValueError:
        return f"lock file contains non-numeric owner marker {raw_pid!r}"

    status_path = Path("/proc") / str(pid) / "status"
    if not status_path.is_file():
        return f"lock file recorded pid={pid}"

    name = None
    state = None
    try:
        for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
    except OSError:
        return f"lock file recorded pid={pid}"

    details = [f"pid={pid}"]
    if name:
        details.append(f"name={name}")
    if state:
        details.append(f"state={state}")
    return ", ".join(details)


@contextmanager
def acquire_job_lock(job_root: Path):
    job_root.mkdir(parents=True, exist_ok=True)
    lock_path = job_root / ".job.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            owner = describe_lock_owner(lock_path)
            suffix = f" ({owner})" if owner else ""
            raise RuntimeError(
                f"Another batch run is already using job-name '{job_root.name}': {job_root}{suffix}"
            ) from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def worker_loop(
    *,
    worker_id: int,
    gpu_id: str | None,
    entry,
    protocol: str,
    data_path: str,
    save_dir: str,
    forward_args: Sequence[str],
    logs_dir: Path,
    task_queue: "queue.Queue[str | None]",
    result_queue: "queue.Queue[WorkerMessage]",
) -> None:
    try:
        while True:
            dataset = task_queue.get()
            if dataset is None:
                return

            log_path = logs_dir / f"{dataset}.log"
            result_queue.put(
                WorkerMessage(
                    kind="dataset_started",
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    dataset=dataset,
                    log_path=log_path,
                )
            )
            command = build_command(
                entry,
                protocol=protocol,
                dataset=dataset,
                data_path=data_path,
                save_dir=save_dir,
                forward_args=forward_args,
            )
            return_code = run_dataset(
                command,
                log_path,
                env=build_subprocess_env(gpu_id),
            )
            result_queue.put(
                WorkerMessage(
                    kind="dataset_finished",
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    dataset=dataset,
                    return_code=return_code,
                    log_path=log_path,
                )
            )
    except Exception as exc:  # pragma: no cover - defensive fallback
        result_queue.put(
            WorkerMessage(
                kind="worker_error",
                worker_id=worker_id,
                gpu_id=gpu_id,
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )
    finally:
        result_queue.put(
            WorkerMessage(
                kind="worker_done",
                worker_id=worker_id,
                gpu_id=gpu_id,
            )
        )


def main(argv: Sequence[str] | None = None) -> int:
    args, forward_args = parse_args(argv)
    entry = get_entry(args.experiment, args.protocol)
    validate_forward_args(forward_args, entry.blocked_forward_args)
    dataset_family = resolve_dataset_family(forward_args)
    if dataset_family == "ucr":
        ucr_archive_dir = resolve_ucr_archive(args.data_path)
        discovered = discover_datasets(ucr_archive_dir)
    elif dataset_family in EXTERNAL_DATASETS:
        discovered = list(EXTERNAL_DATASETS[dataset_family])
    else:
        raise ValueError(
            f"Unsupported --dataset_family for batch runner: {dataset_family}. "
            f"Expected one of: ucr, {', '.join(sorted(EXTERNAL_DATASETS))}"
        )
    selected_datasets = dedupe_preserve_order(
        filter_datasets(
            discovered,
            allowlist=parse_csv_arg(args.datasets),
            denylist=parse_csv_arg(args.exclude_datasets),
            start_from=args.start_from,
        )
    )
    shots = requested_shots(args.protocol, entry.default_shots, forward_args)

    job_root = REPO_ROOT / "results" / "ucr_batches" / args.experiment / args.protocol / args.job_name
    logs_dir = job_root / "logs"
    datasets_dir = job_root / "datasets"
    ledger_path = job_root / "results.txt"
    config_path = job_root / "batch_config.json"
    config_payload = build_batch_config(args, entry, Path(args.data_path).resolve(), forward_args)

    with acquire_job_lock(job_root):
        ensure_batch_config(config_path, config_payload)

        ledger = load_ledger(ledger_path)
        backfill_existing_results(
            ledger,
            datasets=discovered,
            logs_dir=logs_dir,
            datasets_dir=datasets_dir,
            summary_kind=entry.summary_kind,
        )
        write_ledger(ledger_path, ledger)

        print(f"Experiment: {args.experiment} | protocol={args.protocol} | datasets={len(selected_datasets)}")
        print(f"Job root: {job_root}")
        if args.gpu_ids:
            print(f"Workers: {len(args.gpu_ids)} | gpu_ids={','.join(args.gpu_ids)}")
        else:
            print("Workers: 1 | gpu_ids=inherit")
        if args.dry_run:
            print("Dry run enabled.")

        pending_datasets: List[str] = []
        for dataset in selected_datasets:
            if dataset_is_complete(ledger, dataset, shots):
                print(f"[SKIP] {dataset} already complete for shots={shots}")
                continue
            pending_datasets.append(dataset)

        if args.dry_run:
            preview_gpu_ids = args.gpu_ids or [None]
            for idx, dataset in enumerate(pending_datasets):
                gpu_id = preview_gpu_ids[idx % len(preview_gpu_ids)]
                command = build_command(
                    entry,
                    protocol=args.protocol,
                    dataset=dataset,
                    data_path=args.data_path,
                    save_dir=str(datasets_dir),
                    forward_args=forward_args,
                )
                prefix = f"CUDA_VISIBLE_DEVICES={gpu_id} " if gpu_id is not None else ""
                print(f"[DRY-RUN] {dataset}: {prefix}{' '.join(shlex.quote(token) for token in command)}")
            print(f"Ledger written to: {ledger_path}")
            return 0

        if not pending_datasets:
            print("No pending datasets remain.")
            print(f"Ledger written to: {ledger_path}")
            return 0

        task_queue: queue.Queue[str | None] = queue.Queue()
        result_queue: queue.Queue[WorkerMessage] = queue.Queue()
        worker_gpu_ids = args.gpu_ids or [None]

        for dataset in pending_datasets:
            task_queue.put(dataset)
        for _ in worker_gpu_ids:
            task_queue.put(None)

        workers: List[threading.Thread] = []
        for worker_id, gpu_id in enumerate(worker_gpu_ids, start=1):
            thread = threading.Thread(
                target=worker_loop,
                kwargs={
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "entry": entry,
                    "protocol": args.protocol,
                    "data_path": args.data_path,
                    "save_dir": str(datasets_dir),
                    "forward_args": forward_args,
                    "logs_dir": logs_dir,
                    "task_queue": task_queue,
                    "result_queue": result_queue,
                },
                name=f"ucr-batch-worker-{worker_id}",
            )
            thread.start()
            workers.append(thread)

        active_workers = len(workers)
        completed_datasets = 0
        worker_errors: List[str] = []

        while active_workers > 0:
            message = result_queue.get()

            if message.kind == "dataset_started":
                gpu_label = message.gpu_id if message.gpu_id is not None else "inherit"
                print(f"[RUN] worker={message.worker_id} gpu={gpu_label} dataset={message.dataset}")
                continue

            if message.kind == "worker_error":
                gpu_label = message.gpu_id if message.gpu_id is not None else "inherit"
                worker_errors.append(
                    f"worker={message.worker_id} gpu={gpu_label}: {message.error_message}"
                )
                print(f"[WORKER-ERROR] worker={message.worker_id} gpu={gpu_label}: {message.error_message}")
                continue

            if message.kind == "worker_done":
                active_workers -= 1
                continue

            if message.kind != "dataset_finished":  # pragma: no cover - defensive fallback
                raise RuntimeError(f"Unknown worker message: {message.kind}")

            dataset = str(message.dataset)
            log_path = message.log_path
            if log_path is None:
                raise RuntimeError(f"Worker result for {dataset} is missing log_path")
            dataset_dir = datasets_dir / dataset
            return_code = int(message.return_code if message.return_code is not None else -1)
            completed_datasets += 1

            if return_code == 0:
                rows = parse_summary_rows(
                    dataset=dataset,
                    dataset_dir=dataset_dir,
                    log_file=log_path,
                    summary_kind=entry.summary_kind,
                )
                if not rows:
                    failure_row = build_failure_row(
                        dataset=dataset,
                        status="failed",
                        result_file=str(dataset_dir.resolve()) if dataset_dir.exists() else "",
                        log_file=str(log_path.resolve()),
                        note="exit_code=0 but no summary file was produced",
                    )
                    upsert_row(ledger, failure_row)
                    print(f"[FAIL] {dataset} produced no summary output ({completed_datasets}/{len(pending_datasets)})")
                else:
                    for row in rows:
                        upsert_row(ledger, row)
                    remove_row(ledger, dataset, "__dataset__")
                    print(f"[OK] {dataset} ({completed_datasets}/{len(pending_datasets)})")
            else:
                status = detect_failure_status(log_path)
                failure_row = build_failure_row(
                    dataset=dataset,
                    status=status,
                    result_file=str(dataset_dir.resolve()) if dataset_dir.exists() else "",
                    log_file=str(log_path.resolve()),
                    note=f"exit_code={return_code}",
                )
                upsert_row(ledger, failure_row)
                print(f"[{status.upper()}] {dataset} (exit {return_code}) ({completed_datasets}/{len(pending_datasets)})")

            write_ledger(ledger_path, ledger)

        for thread in workers:
            thread.join()

        if worker_errors:
            raise RuntimeError("Worker threads failed internally:\n" + "\n".join(worker_errors))

        print(f"Ledger written to: {ledger_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
