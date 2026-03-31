#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
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
    parser.add_argument("--dry-run", action="store_true")
    args, forward_args = parser.parse_known_args(argv)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]
    return args, forward_args


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


def validate_forward_args(forward_args: Sequence[str], blocked_flags: Iterable[str]) -> None:
    blocked = set(blocked_flags)
    matches = sorted({token.split("=", 1)[0] for token in forward_args if token.split("=", 1)[0] in blocked})
    if matches:
        raise ValueError(
            "These arguments are managed by the batch runner and must not be forwarded: "
            + ", ".join(matches)
        )


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
    return {
        "experiment": args.experiment,
        "protocol": args.protocol,
        "script_path": str(entry.script_path.resolve()),
        "data_path": str(resolved_data_path),
        "fixed_args": list(entry.fixed_args),
        "forward_args": list(forward_args),
        "supports_inner_resume": entry.supports_inner_resume,
        "summary_kind": entry.summary_kind,
    }


def ensure_batch_config(config_path: Path, config_payload: dict) -> None:
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if existing != config_payload:
            raise ValueError(
                f"Existing batch config at {config_path} does not match this run. "
                "Use a new --job-name for a different experiment configuration."
            )
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


def build_command(entry, *, protocol: str, dataset: str, data_path: str, save_dir: str, forward_args: Sequence[str]) -> List[str]:
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


def run_dataset(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("COMMAND: " + " ".join(shlex.quote(token) for token in command) + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args, forward_args = parse_args(argv)
    entry = get_entry(args.experiment, args.protocol)
    validate_forward_args(forward_args, entry.blocked_forward_args)

    ucr_archive_dir = resolve_ucr_archive(args.data_path)
    discovered = discover_datasets(ucr_archive_dir)
    selected_datasets = filter_datasets(
        discovered,
        allowlist=parse_csv_arg(args.datasets),
        denylist=parse_csv_arg(args.exclude_datasets),
        start_from=args.start_from,
    )
    shots = requested_shots(args.protocol, entry.default_shots, forward_args)

    job_root = REPO_ROOT / "results" / "ucr_batches" / args.experiment / args.protocol / args.job_name
    logs_dir = job_root / "logs"
    datasets_dir = job_root / "datasets"
    ledger_path = job_root / "results.txt"
    config_path = job_root / "batch_config.json"

    config_payload = build_batch_config(args, entry, Path(args.data_path).resolve(), forward_args)
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
    if args.dry_run:
        print("Dry run enabled.")

    for dataset in selected_datasets:
        dataset_dir = datasets_dir / dataset
        log_path = logs_dir / f"{dataset}.log"

        if dataset_is_complete(ledger, dataset, shots):
            print(f"[SKIP] {dataset} already complete for shots={shots}")
            continue

        command = build_command(
            entry,
            protocol=args.protocol,
            dataset=dataset,
            data_path=args.data_path,
            save_dir=str(datasets_dir),
            forward_args=forward_args,
        )

        if args.dry_run:
            print(f"[DRY-RUN] {dataset}: {' '.join(shlex.quote(token) for token in command)}")
            continue

        print(f"[RUN] {dataset}")
        return_code = run_dataset(command, log_path)

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
                print(f"[FAIL] {dataset} produced no summary output")
            else:
                for row in rows:
                    upsert_row(ledger, row)
                remove_row(ledger, dataset, "__dataset__")
                print(f"[OK] {dataset}")
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
            print(f"[{status.upper()}] {dataset} (exit {return_code})")

        write_ledger(ledger_path, ledger)

    print(f"Ledger written to: {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
