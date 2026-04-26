#!/usr/bin/env python3

"""uv run scripts/run_m2_constrained_decoding_ablation.sh \
  --local_checkpoint /root/data1/tsllava/results/from_83/stage01/best_model.pt \
  --data_path /root/data1/tsllava/data \
  --reference_job_dir /root/data1/tsllava/results/ucr_batches/m2_pretrained/fewshot/m2_fewshot_6gpu \
  --comparison_job_dir /root/data1/tsllava/results/ucr_batches/m2_pretrained/fewshot/m2_fewshot_6gpu \
  --shots 1,5,10 \
  --num_runs 1 \
  --gpu_ids 0,1,2,3,4,5 \
  --job_prefix m2_cd_after_dualview \
  --report_name m2_cd_after_dualview
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m2_ablation_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEWSHOT_SECOND_JOB_DIR,
    DEFAULT_REFERENCE_JOB_DIR,
    DEFAULT_SHOTS,
    M2_FEWSHOT_ROOT,
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
from reporting import generate_report
from run_ucr_batch import main as run_ucr_batch_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M2 constrained-decoding ablation on the dual-view shared UCR subset."
    )
    parser.add_argument("--local_checkpoint", required=True, help="Checkpoint used for the unconstrained rerun.")
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
    dry_run: bool,
) -> Path:
    argv = [
        "--experiment",
        "m2_pretrained",
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
    if dry_run:
        argv.append("--dry-run")
    argv.append("--")
    argv.extend(forward_args)
    exit_code = run_ucr_batch_main(argv)
    if exit_code != 0:
        raise RuntimeError(f"run_ucr_batch failed for {job_name} with exit code {exit_code}")
    return M2_FEWSHOT_ROOT / job_name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shots = tuple(str(item) for item in parse_csv_list(args.shots))
    if not shots:
        raise ValueError("--shots must contain at least one shot")

    reference_job_dir = Path(args.reference_job_dir).resolve()
    comparison_job_dir = Path(args.comparison_job_dir).resolve()
    candidate_datasets = shared_complete_datasets(
        [reference_job_dir / "results.txt", comparison_job_dir / "results.txt"],
        dataset_source=args.data_path,
        shots=shots,
    )
    if not candidate_datasets:
        raise ValueError("No shared complete datasets found for the constrained-decoding candidate pool")

    if args.subset_manifest:
        subset_manifest = load_subset_manifest(Path(args.subset_manifest).resolve())
        selected_datasets = list(subset_manifest["selected_datasets"])
        missing = [dataset for dataset in selected_datasets if dataset not in set(candidate_datasets)]
        if missing:
            raise ValueError(
                "subset_manifest contains datasets outside the current constrained-decoding candidate pool: "
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
            candidate_pool_name=f"constrained_decoding_shared_{len(candidate_datasets)}",
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
    report_name = args.report_name or f"m2_constrained_decoding_ablation_{subset_id}"
    report_dir = default_report_dir(report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest_path = write_json(report_dir / "subset_manifest.json", subset_manifest)

    job_prefix = args.job_prefix or f"m2_constrained_decoding_{subset_id}"
    unconstrained_job_name = f"{job_prefix}_unconstrained"
    forward_args = build_forward_args_from_reference(
        reference_batch_config_path=reference_job_dir / "batch_config.json",
        local_checkpoint=args.local_checkpoint,
        shots=shots,
        num_runs=args.num_runs,
        disable_constrained_decoding=True,
    )

    unconstrained_job_dir = _run_batch_job(
        job_name=unconstrained_job_name,
        data_path=str(args.data_path),
        datasets=selected_datasets,
        forward_args=forward_args,
        gpu_ids=args.gpu_ids,
        dry_run=args.dry_run,
    )

    report_config = build_ablation_report_config(
        report_name=report_name,
        family_label="Effect of Constrained Decoding",
        reference_key="constrained",
        dataset_source=args.data_path,
        dataset_allowlist=selected_datasets,
        shots=shots,
        appendix_tables_enabled=args.appendix_tables,
        items=[
            {
                "key": "constrained",
                "label": "Constrained",
                "job_dir": str(reference_job_dir),
                "primary": True,
            },
            {
                "key": "unconstrained",
                "label": "Unconstrained",
                "job_dir": str(unconstrained_job_dir),
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
            "unconstrained_job_name": unconstrained_job_name,
            "local_checkpoint": str(args.local_checkpoint),
            "num_runs": args.num_runs,
            "shots": list(shots),
            "appendix_tables_enabled": bool(args.appendix_tables),
            "dry_run": bool(args.dry_run),
            "subset_manifest_path": str(subset_manifest_path),
            "report_config_path": str(report_config_path),
        },
    )
    write_json(report_dir / "ablation_request.json", request_payload)

    if args.dry_run:
        print(f"Prepared constrained-decoding ablation config at {report_config_path}")
        print(f"Subset manifest: {subset_manifest_path}")
        return 0

    manifest = generate_report(report_config_path)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {report_dir}")
    print(f"Datasets: {len(selected_datasets)} | subset_id={subset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
