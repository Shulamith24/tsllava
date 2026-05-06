#!/usr/bin/env python3

"""Run the M2 no-LLM linear-classifier ablation on a selected UCR subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m2_ablation_common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_FEWSHOT_SECOND_JOB_DIR,
    DEFAULT_REFERENCE_JOB_DIR,
    DEFAULT_SHOTS,
    REPO_ROOT,
    build_ablation_report_config,
    build_forward_args_from_reference,
    build_run_request_payload,
    build_subset_manifest,
    default_report_dir,
    load_subset_manifest,
    parse_csv_list,
    resolve_selected_datasets,
    shared_complete_datasets,
    write_json,
)
from reporting import generate_report  # noqa: E402
from run_ucr_batch import main as run_ucr_batch_main  # noqa: E402

NO_LLM_FEWSHOT_ROOT = REPO_ROOT / "results" / "ucr_batches" / "m2_no_llm_linear" / "fewshot"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M2 ablation that removes the LLM backbone and trains a linear head on dual-branch features."
    )
    parser.add_argument("--local_checkpoint", required=True, help="Checkpoint providing the pretrained dual-branch encoder.")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--reference_job_dir", default=str(DEFAULT_REFERENCE_JOB_DIR))
    parser.add_argument("--comparison_job_dir", default=str(DEFAULT_FEWSHOT_SECOND_JOB_DIR))
    parser.add_argument("--shots", default=",".join(DEFAULT_SHOTS))
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--num_datasets", type=int, default=None)
    parser.add_argument("--dataset_list", default=None, help="Comma-separated explicit dataset subset.")
    parser.add_argument("--sampling_seed", type=int, default=3407)
    parser.add_argument("--subset_manifest", default=None, help="Reuse a previously written subset_manifest.json.")
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated CUDA device ids for run_ucr_batch.")
    parser.add_argument(
        "--start_from",
        "--start-from",
        dest="start_from",
        default=None,
        help="Start the inner UCR batch runner from this dataset name.",
    )
    parser.add_argument("--job_prefix", default=None, help="Optional prefix for the generated batch job name.")
    parser.add_argument("--report_name", default=None, help="Optional override for the generated report name.")
    parser.add_argument("--appendix_tables", action="store_true", help="Also generate appendix_shot_*.tex tables.")
    parser.add_argument("--dry_run", action="store_true", help="Only write configs and batch commands.")
    return parser.parse_args(argv)


def _run_batch_job(
    *,
    job_name: str,
    data_path: str,
    datasets: list[str],
    forward_args: list[str],
    gpu_ids: str | None,
    start_from: str | None,
    dry_run: bool,
) -> Path:
    argv = [
        "--experiment",
        "m2_no_llm_linear",
        "--protocol",
        "fewshot",
        "--job-name",
        job_name,
        "--data-path",
        data_path,
        "--datasets",
        ",".join(datasets),
    ]
    if gpu_ids:
        argv.extend(["--gpu-ids", gpu_ids])
    if start_from:
        argv.extend(["--start-from", start_from])
    if dry_run:
        argv.append("--dry-run")
    argv.append("--")
    argv.extend(forward_args)

    exit_code = run_ucr_batch_main(argv)
    if exit_code != 0:
        raise RuntimeError(f"run_ucr_batch failed for {job_name} with exit code {exit_code}")
    return NO_LLM_FEWSHOT_ROOT / job_name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shots = tuple(str(item) for item in parse_csv_list(args.shots))
    if not shots:
        raise ValueError("--shots must contain at least one shot")
    if any(shot.lower() == "full" for shot in shots):
        raise ValueError("The no-LLM backbone ablation is few-shot only; remove 'full' from --shots.")

    reference_job_dir = Path(args.reference_job_dir).resolve()
    comparison_job_dir = Path(args.comparison_job_dir).resolve()
    candidate_datasets = shared_complete_datasets(
        [reference_job_dir / "results.txt", comparison_job_dir / "results.txt"],
        dataset_source=args.data_path,
        shots=shots,
    )
    if not candidate_datasets:
        raise ValueError("No shared complete datasets found for the no-LLM candidate pool")

    if args.subset_manifest:
        subset_manifest = load_subset_manifest(Path(args.subset_manifest).resolve())
        selected_datasets = list(subset_manifest["selected_datasets"])
        missing = [dataset for dataset in selected_datasets if dataset not in set(candidate_datasets)]
        if missing:
            raise ValueError(
                "subset_manifest contains datasets outside the current no-LLM candidate pool: "
                + ",".join(missing)
            )
    else:
        selected_datasets, selection_mode, metadata = resolve_selected_datasets(
            candidate_datasets=candidate_datasets,
            dataset_source=args.data_path,
            num_datasets=args.num_datasets,
            sampling_seed=args.sampling_seed,
            dataset_list=parse_csv_list(args.dataset_list),
        )
        subset_manifest = build_subset_manifest(
            candidate_pool_name=f"no_llm_shared_{len(candidate_datasets)}",
            dataset_source=args.data_path,
            shots=shots,
            candidate_datasets=candidate_datasets,
            selected_datasets=selected_datasets,
            selection_mode=selection_mode,
            sampling_seed=args.sampling_seed,
            num_datasets_requested=args.num_datasets,
            metadata=metadata,
        )

    subset_id = str(subset_manifest["subset_id"])
    report_name = args.report_name or f"m2_no_llm_backbone_ablation_{subset_id}"
    report_dir = default_report_dir(report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest_path = write_json(report_dir / "subset_manifest.json", subset_manifest)

    job_prefix = args.job_prefix or f"m2_no_llm_linear_{subset_id}"
    no_llm_job_name = f"{job_prefix}_linear"
    forward_args = build_forward_args_from_reference(
        reference_batch_config_path=reference_job_dir / "batch_config.json",
        local_checkpoint=args.local_checkpoint,
        shots=shots,
        num_runs=args.num_runs,
        runtime_branch_mode="both",
    )

    no_llm_job_dir = _run_batch_job(
        job_name=no_llm_job_name,
        data_path=str(args.data_path),
        datasets=selected_datasets,
        forward_args=forward_args,
        gpu_ids=args.gpu_ids,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )

    report_config = build_ablation_report_config(
        report_name=report_name,
        family_label="Effect of LLM Backbone",
        reference_key="m2_llm",
        dataset_source=args.data_path,
        dataset_allowlist=selected_datasets,
        shots=shots,
        appendix_tables_enabled=args.appendix_tables,
        items=[
            {
                "key": "m2_llm",
                "label": "M2 with LLM",
                "job_dir": str(reference_job_dir),
                "primary": True,
            },
            {
                "key": "no_llm_linear",
                "label": "Encoder + Linear",
                "job_dir": str(no_llm_job_dir),
            },
        ],
    )
    report_config_path = write_json(report_dir / "report_config.generated.json", report_config)

    request_payload = build_run_request_payload(
        script_name=Path(__file__).name,
        report_name=report_name,
        subset_manifest=subset_manifest,
        extra={
            "reference_job_dir": str(reference_job_dir),
            "comparison_job_dir": str(comparison_job_dir),
            "no_llm_job_name": no_llm_job_name,
            "local_checkpoint": str(args.local_checkpoint),
            "num_runs": args.num_runs,
            "shots": list(shots),
            "start_from": args.start_from,
            "appendix_tables_enabled": bool(args.appendix_tables),
            "dry_run": bool(args.dry_run),
            "subset_manifest_path": str(subset_manifest_path),
            "report_config_path": str(report_config_path),
        },
    )
    write_json(report_dir / "ablation_request.json", request_payload)

    if args.dry_run:
        print(f"Prepared no-LLM backbone ablation config at {report_config_path}")
        print(f"Subset manifest: {subset_manifest_path}")
        return 0

    manifest = generate_report(report_config_path)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {report_dir}")
    print(f"Datasets: {len(selected_datasets)} | subset_id={subset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
