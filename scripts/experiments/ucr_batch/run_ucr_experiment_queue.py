#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from registry import REPO_ROOT, list_experiments  # noqa: E402

RUN_UCR_BATCH = SCRIPT_DIR / "run_ucr_batch.py"
SUPPORTED_DATASET_FAMILIES = (
    "ucr",
    "mitbih",
    "sleepedf",
    "cinc2017af",
    "cinc2016heart",
)
BLOCKED_FORWARD_ARGS = {
    "--experiment",
    "--protocol",
    "--job-name",
    "--data-path",
    "--gpu-ids",
    "--dry-run",
}


@dataclass
class ActiveTask:
    queue_index: int
    experiment: str
    dataset_family: str | None
    job_name: str
    gpu_id: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[str]
    log_handle: IO[str]
    started_at_unix: float


@dataclass
class TaskResult:
    queue_index: int
    experiment: str
    dataset_family: str | None
    job_name: str
    gpu_id: str
    return_code: int
    status: str
    elapsed_seconds: float
    log_path: str
    command: list[str]


@dataclass(frozen=True)
class QueuedTask:
    queue_index: int
    experiment: str
    dataset_family: str | None


def parse_gpu_ids(raw_value: str) -> list[str]:
    gpu_ids: list[str] = []
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
    if not gpu_ids:
        raise ValueError("--gpu-ids must contain at least one GPU id")
    return gpu_ids


def parse_experiment_list(raw_value: str) -> list[str]:
    tokens = [item.strip() for item in re.split(r"[\s,]+", raw_value) if item.strip()]
    if not tokens:
        raise ValueError("--experiments must contain at least one experiment name")

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def parse_dataset_family_list(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    tokens = [item.strip().lower() for item in re.split(r"[\s,]+", raw_value) if item.strip()]
    if not tokens:
        raise ValueError("--dataset-families must contain at least one dataset family")
    supported = set(SUPPORTED_DATASET_FAMILIES)
    unknown = [family for family in tokens if family not in supported]
    if unknown:
        raise ValueError(
            "Unknown dataset families: " + ", ".join(unknown) + ". "
            "Use one of: " + ", ".join(SUPPORTED_DATASET_FAMILIES)
        )
    deduped: list[str] = []
    seen: set[str] = set()
    for family in tokens:
        if family in seen:
            continue
        seen.add(family)
        deduped.append(family)
    return deduped


def cli_option_value(argv: Sequence[str], flag_name: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag_name and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith(f"{flag_name}="):
            return token.split("=", 1)[1]
    return None


def validate_experiments(experiments: Sequence[str]) -> None:
    supported = set(list_experiments())
    unknown = [experiment for experiment in experiments if experiment not in supported]
    if unknown:
        raise ValueError(
            "Unknown experiments: " + ", ".join(unknown) + ". "
            "Use one of: " + ", ".join(sorted(supported))
        )


def validate_forward_args(forward_args: Sequence[str]) -> None:
    matches = sorted(
        {
            token.split("=", 1)[0]
            for token in forward_args
            if token.split("=", 1)[0] in BLOCKED_FORWARD_ARGS
        }
    )
    if matches:
        raise ValueError(
            "These arguments are managed by the queue launcher and must not be forwarded: "
            + ", ".join(matches)
        )


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "Schedule multiple run_ucr_batch jobs across a fixed GPU pool. "
            "Each GPU runs at most one experiment at a time, and the next pending "
            "experiment starts as soon as a GPU becomes free."
        )
    )
    parser.add_argument(
        "--experiments",
        required=True,
        help="Whitespace/comma-separated experiment names, e.g. 'onefitsall patchtst resnet'",
    )
    parser.add_argument("--protocol", required=True, choices=["full", "fewshot"])
    parser.add_argument("--job-name-template", default=None)
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data"))
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated physical CUDA device ids.")
    parser.add_argument(
        "--dataset-families",
        default=None,
        help=(
            "Optional whitespace/comma-separated dataset families. "
            "When provided, the queue runs every family x experiment pair."
        ),
    )
    parser.add_argument(
        "--launcher-name",
        default=None,
        help="Optional name for queue logs/summary. Defaults to a timestamped auto name.",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop launching new experiments after the first failure and terminate active children.",
    )
    args, forward_args = parser.parse_known_args(argv)
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    args.experiments = parse_experiment_list(args.experiments)
    args.gpu_ids = parse_gpu_ids(args.gpu_ids)
    args.dataset_families = parse_dataset_family_list(args.dataset_families)
    if args.dataset_families and cli_option_value(forward_args, "--dataset_family") is not None:
        raise ValueError("Use --dataset-families at the queue level instead of forwarding --dataset_family.")
    if args.job_name_template is None:
        args.job_name_template = "{dataset_family}_{experiment}" if args.dataset_families else "{experiment}"
    validate_experiments(args.experiments)
    validate_forward_args(forward_args)
    return args, forward_args


def default_launcher_name(*, protocol: str, experiments: Sequence[str], dataset_families: Sequence[str]) -> str:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
    family_part = f"_{len(dataset_families)}fam" if dataset_families else ""
    return f"queue_{protocol}_{len(experiments)}exp{family_part}_{timestamp}"


def format_job_name(
    template: str,
    *,
    experiment: str,
    protocol: str,
    queue_index: int,
    dataset_family: str,
) -> str:
    try:
        return template.format(
            experiment=experiment,
            protocol=protocol,
            index=queue_index,
            dataset_family=dataset_family,
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"--job-name-template references unknown placeholder {{{missing}}}. "
            "Supported placeholders: {experiment}, {protocol}, {index}, {dataset_family}"
        ) from exc


def build_command(
    *,
    experiment: str,
    protocol: str,
    job_name: str,
    data_path: str,
    gpu_id: str,
    dataset_family: str | None,
    forward_args: Sequence[str],
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(RUN_UCR_BATCH),
        "--experiment",
        experiment,
        "--protocol",
        protocol,
        "--job-name",
        job_name,
        "--data-path",
        data_path,
        "--gpu-ids",
        gpu_id,
    ]
    if dry_run:
        command.append("--dry-run")
    managed_forward_args: list[str] = []
    if dataset_family is not None:
        managed_forward_args.extend(["--dataset_family", dataset_family])
    managed_forward_args.extend(forward_args)
    if managed_forward_args:
        command.append("--")
        command.extend(managed_forward_args)
    return command


def write_launcher_config(
    *,
    config_path: Path,
    args,
    forward_args: Sequence[str],
) -> None:
    payload = {
        "experiments": list(args.experiments),
        "protocol": args.protocol,
        "job_name_template": args.job_name_template,
        "data_path": str(Path(args.data_path).resolve()),
        "dataset_families": list(args.dataset_families),
        "gpu_ids": list(args.gpu_ids),
        "poll_seconds": args.poll_seconds,
        "dry_run": bool(args.dry_run),
        "fail_fast": bool(args.fail_fast),
        "forward_args": list(forward_args),
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def launch_task(
    *,
    queued_task: QueuedTask,
    experiment: str,
    gpu_id: str,
    args,
    forward_args: Sequence[str],
    logs_dir: Path,
) -> ActiveTask:
    dataset_family_for_name = queued_task.dataset_family or cli_option_value(forward_args, "--dataset_family") or "ucr"
    job_name = format_job_name(
        args.job_name_template,
        experiment=experiment,
        protocol=args.protocol,
        queue_index=queued_task.queue_index,
        dataset_family=dataset_family_for_name,
    )
    command = build_command(
        experiment=experiment,
        protocol=args.protocol,
        job_name=job_name,
        data_path=args.data_path,
        gpu_id=gpu_id,
        dataset_family=queued_task.dataset_family,
        forward_args=forward_args,
        dry_run=args.dry_run,
    )
    family_prefix = f"{queued_task.dataset_family}_" if queued_task.dataset_family is not None else ""
    log_path = logs_dir / f"{queued_task.queue_index:02d}_{family_prefix}{experiment}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write(f"GPU_ID={gpu_id}\n")
    log_handle.write("COMMAND: " + " ".join(shlex.quote(token) for token in command) + "\n\n")
    log_handle.flush()

    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ActiveTask(
        queue_index=queued_task.queue_index,
        experiment=experiment,
        dataset_family=queued_task.dataset_family,
        job_name=job_name,
        gpu_id=gpu_id,
        command=command,
        log_path=log_path,
        process=process,
        log_handle=log_handle,
        started_at_unix=time.monotonic(),
    )


def finalize_task(task: ActiveTask, return_code: int) -> TaskResult:
    elapsed_seconds = time.monotonic() - task.started_at_unix
    task.log_handle.flush()
    task.log_handle.close()
    status = "ok" if return_code == 0 else "failed"
    return TaskResult(
        queue_index=task.queue_index,
        experiment=task.experiment,
        dataset_family=task.dataset_family,
        job_name=task.job_name,
        gpu_id=task.gpu_id,
        return_code=return_code,
        status=status,
        elapsed_seconds=elapsed_seconds,
        log_path=str(task.log_path),
        command=list(task.command),
    )


def write_summary(
    *,
    summary_path: Path,
    results: Sequence[TaskResult],
    pending_experiments: Sequence[str],
    interrupted: bool,
) -> None:
    payload = {
        "interrupted": interrupted,
        "pending_experiments": list(pending_experiments),
        "results": [asdict(result) for result in results],
        "finished_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def terminate_active_tasks(active_tasks: Sequence[ActiveTask]) -> None:
    for task in active_tasks:
        if task.process.poll() is None:
            task.process.terminate()
    for task in active_tasks:
        if task.process.poll() is None:
            try:
                task.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                task.process.kill()
                task.process.wait(timeout=10)


def main(argv: Sequence[str] | None = None) -> int:
    args, forward_args = parse_args(argv)

    launcher_name = args.launcher_name or default_launcher_name(
        protocol=args.protocol,
        experiments=args.experiments,
        dataset_families=args.dataset_families,
    )
    launcher_root = REPO_ROOT / "results" / "ucr_batches" / "_launchers" / launcher_name
    logs_dir = launcher_root / "logs"
    config_path = launcher_root / "launcher_config.json"
    summary_path = launcher_root / "summary.json"

    write_launcher_config(
        config_path=config_path,
        args=args,
        forward_args=forward_args,
    )

    if args.dataset_families:
        task_specs = []
        index = 1
        for dataset_family in args.dataset_families:
            for experiment in args.experiments:
                task_specs.append(
                    QueuedTask(
                        queue_index=index,
                        experiment=experiment,
                        dataset_family=dataset_family,
                    )
                )
                index += 1
    else:
        task_specs = [
            QueuedTask(queue_index=index, experiment=experiment, dataset_family=None)
            for index, experiment in enumerate(args.experiments, start=1)
        ]
    pending = deque(task_specs)
    available_gpus = deque(args.gpu_ids)
    active_tasks: list[ActiveTask] = []
    results: list[TaskResult] = []
    interrupted = False

    print(
        f"Queue launcher: {launcher_name} | protocol={args.protocol} | "
        f"tasks={len(task_specs)} | experiments={len(args.experiments)} | "
        f"families={','.join(args.dataset_families) if args.dataset_families else 'forward/default'} | "
        f"gpus={','.join(args.gpu_ids)}"
    )
    print(f"Launcher root: {launcher_root}")

    try:
        while pending or active_tasks:
            while pending and available_gpus:
                queued_task = pending.popleft()
                gpu_id = available_gpus.popleft()
                task = launch_task(
                    queued_task=queued_task,
                    experiment=queued_task.experiment,
                    gpu_id=gpu_id,
                    args=args,
                    forward_args=forward_args,
                    logs_dir=logs_dir,
                )
                active_tasks.append(task)
                family_label = task.dataset_family or "forward/default"
                print(
                    f"[LAUNCH] gpu={gpu_id} family={family_label} experiment={task.experiment} "
                    f"job={task.job_name} log={task.log_path}"
                )

            if not active_tasks:
                break

            completed_index = None
            for index, task in enumerate(active_tasks):
                return_code = task.process.poll()
                if return_code is not None:
                    completed_index = index
                    result = finalize_task(task, return_code)
                    results.append(result)
                    active_tasks.pop(index)
                    available_gpus.append(task.gpu_id)
                    tag = "OK" if result.return_code == 0 else "FAILED"
                    family_label = result.dataset_family or "forward/default"
                    print(
                        f"[{tag}] gpu={result.gpu_id} family={family_label} experiment={result.experiment} "
                        f"exit={result.return_code} elapsed={result.elapsed_seconds:.1f}s "
                        f"log={result.log_path}"
                    )
                    if result.return_code != 0 and args.fail_fast:
                        print("Fail-fast enabled; terminating remaining active experiments.")
                        terminate_active_tasks(active_tasks)
                        for leftover in active_tasks:
                            finalized = finalize_task(
                                leftover,
                                leftover.process.returncode if leftover.process.returncode is not None else -15,
                            )
                            results.append(finalized)
                        active_tasks.clear()
                        pending.clear()
                    break

            if completed_index is None:
                time.sleep(max(args.poll_seconds, 0.1))
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; terminating active experiments.")
        terminate_active_tasks(active_tasks)
        for task in active_tasks:
            results.append(
                finalize_task(
                    task,
                    task.process.returncode if task.process.returncode is not None else -15,
                )
            )
        active_tasks.clear()
    finally:
        pending_experiments = [
            f"{task.dataset_family}:{task.experiment}" if task.dataset_family else task.experiment
            for task in pending
        ]
        write_summary(
            summary_path=summary_path,
            results=results,
            pending_experiments=pending_experiments,
            interrupted=interrupted,
        )

    num_failed = sum(1 for result in results if result.return_code != 0)
    num_succeeded = sum(1 for result in results if result.return_code == 0)
    print(
        f"Summary: ok={num_succeeded} failed={num_failed} pending={len(pending)} "
        f"summary={summary_path}"
    )
    return 1 if interrupted or num_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
