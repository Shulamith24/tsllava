#!/usr/bin/env python3

"""Run no-LLM Transformer ablations for the 1D-to-2D vision backbone."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m2_ablation_common import (  # noqa: E402
    DEFAULT_DATA_PATH,
    DEFAULT_SHOTS,
    REPO_ROOT,
    build_ablation_report_config,
    build_run_request_payload,
    default_report_dir,
    parse_csv_list,
    write_json,
)
from reporting import generate_report  # noqa: E402
from run_ucr_batch import main as run_ucr_batch_main  # noqa: E402
from ucr_datasets import discover_datasets, resolve_ucr_archive  # noqa: E402


NO_LLM_TRANSFORMER_ROOT = REPO_ROOT / "results" / "ucr_batches" / "m2_no_llm_transformer" / "fewshot"

DEFAULT_VARIANTS = (
    "dinov2_frozen",
    "small_vit_trainable",
    "cnn_trainable",
)

VARIANT_SPECS = {
    "dinov2_frozen": {
        "label": "Frozen DINOv2",
        "forward_args": [
            "--vision_backbone_type",
            "hf_pretrained",
            "--vit_model_name",
            "facebook/dinov2-base",
            "--vit_feature_mode",
            "single",
            "--vit_layer_idx",
            "4",
            "--freeze_vision_backbone",
            "--vision_train_mode",
            "none",
        ],
    },
    "small_vit_trainable": {
        "label": "Trainable small ViT",
        "forward_args": [
            "--vision_backbone_type",
            "small_vit_scratch",
            "--vit_feature_mode",
            "single",
            "--vit_layer_idx",
            "4",
            "--no_freeze_vision_backbone",
            "--vision_train_mode",
            "all",
        ],
    },
    "cnn_trainable": {
        "label": "Trainable CNN",
        "forward_args": [
            "--vision_backbone_type",
            "cnn_scratch",
            "--vit_feature_mode",
            "single",
            "--vit_layer_idx",
            "4",
            "--no_freeze_vision_backbone",
            "--vision_train_mode",
            "all",
        ],
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TimeMorph no-LLM Transformer vision-backbone ablations with dataset-level GPU queues."
    )
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--gpu_ids", default=None, help="Comma-separated CUDA device ids for run_ucr_batch.")
    parser.add_argument("--shots", default=",".join(DEFAULT_SHOTS))
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="Comma-separated backbone variants.")
    parser.add_argument("--mode", default="both", choices=["both", "ts_only", "vision_only"])
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset allowlist.")
    parser.add_argument("--start_from", "--start-from", dest="start_from", default=None)
    parser.add_argument("--job_prefix", default="m2_vision_backbone")
    parser.add_argument("--report_name", default=None)
    parser.add_argument("--appendix_tables", action="store_true")
    parser.add_argument("--cleanup_checkpoints", action="store_true")
    parser.add_argument("--skip_checkpoints", "--skip-phase-checkpoints", dest="skip_checkpoints", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--vision_2d_mode", default="legacy_unfold")
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    return parser.parse_args(argv)


def _run_batch_job(
    *,
    job_name: str,
    data_path: str,
    datasets: list[str],
    gpu_ids: str | None,
    start_from: str | None,
    dry_run: bool,
    forward_args: list[str],
) -> Path:
    argv = [
        "--experiment",
        "m2_no_llm_transformer",
        "--protocol",
        "fewshot",
        "--job-name",
        job_name,
        "--data-path",
        data_path,
    ]
    if datasets:
        argv.extend(["--datasets", ",".join(datasets)])
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
    return NO_LLM_TRANSFORMER_ROOT / job_name


def _base_forward_args(args: argparse.Namespace, shots: tuple[str, ...]) -> list[str]:
    forward_args = [
        "--shots",
        ",".join(shots),
        "--num_runs",
        str(args.num_runs),
        "--runtime_branch_mode",
        str(args.mode),
        "--no_freeze_encoder",
        "--vision_2d_mode",
        str(args.vision_2d_mode),
        "--vit_patch_size",
        str(args.vit_patch_size),
        "--vit_stride",
        str(args.vit_stride),
    ]
    if args.cleanup_checkpoints:
        forward_args.append("--cleanup_checkpoints")
    if args.skip_checkpoints:
        forward_args.append("--skip_checkpoints")
    if args.dataloader_num_workers is not None:
        forward_args.extend(["--dataloader_num_workers", str(args.dataloader_num_workers)])
    if args.epochs is not None:
        forward_args.extend(["--epochs", str(args.epochs)])
    if args.batch_size is not None:
        forward_args.extend(["--batch_size", str(args.batch_size)])
    if args.eval_batch_size is not None:
        forward_args.extend(["--eval_batch_size", str(args.eval_batch_size)])
    if args.freeze_encoder:
        forward_args.append("--freeze_encoder")
    return forward_args


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return parse_csv_list(args.datasets)
    archive_dir = resolve_ucr_archive(args.data_path)
    return discover_datasets(archive_dir)


def _build_jobs(args: argparse.Namespace) -> list[dict[str, object]]:
    variants = parse_csv_list(args.variants)
    if not variants:
        raise ValueError("--variants must contain at least one variant")
    unknown = [variant for variant in variants if variant not in VARIANT_SPECS]
    if unknown:
        raise ValueError(f"Unsupported variants: {','.join(unknown)}. Valid variants: {list(VARIANT_SPECS)}")
    return [
        {
            "key": variant,
            "label": VARIANT_SPECS[variant]["label"],
            "job_name": f"{args.job_prefix}_{variant}",
            "forward_args": list(VARIANT_SPECS[variant]["forward_args"]),
        }
        for variant in variants
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shots = tuple(str(item) for item in parse_csv_list(args.shots))
    if not shots:
        raise ValueError("--shots must contain at least one shot")
    if any(shot.lower() == "full" for shot in shots):
        raise ValueError("The vision-backbone ablation is few-shot only; remove 'full' from --shots.")

    selected_datasets = _resolve_datasets(args)
    if not selected_datasets:
        raise ValueError("No datasets selected for vision-backbone ablation")

    jobs = _build_jobs(args)
    job_dirs: list[Path] = []
    for job in jobs:
        forward_args = _base_forward_args(args, shots)
        forward_args.extend(job["forward_args"])
        job_dir = _run_batch_job(
            job_name=str(job["job_name"]),
            data_path=str(args.data_path),
            datasets=selected_datasets,
            gpu_ids=args.gpu_ids,
            start_from=args.start_from,
            dry_run=args.dry_run,
            forward_args=forward_args,
        )
        job_dirs.append(job_dir)

    report_name = args.report_name or f"{args.job_prefix}_{args.mode}_report"
    report_dir = default_report_dir(report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for idx, (job, job_dir) in enumerate(zip(jobs, job_dirs)):
        items.append(
            {
                "key": str(job["key"]),
                "label": str(job["label"]),
                "job_dir": str(job_dir),
                "primary": idx == 0,
            }
        )
    reference_key = str(jobs[0]["key"])
    report_config = build_ablation_report_config(
        report_name=report_name,
        family_label="1D-to-2D Vision Backbone",
        reference_key=reference_key,
        dataset_source=args.data_path,
        dataset_allowlist=selected_datasets,
        shots=shots,
        appendix_tables_enabled=args.appendix_tables,
        items=items,
    )
    report_config_path = write_json(report_dir / "report_config.generated.json", report_config)

    request_payload = build_run_request_payload(
        script_name=Path(__file__).name,
        report_name=report_name,
        subset_manifest={
            "selected_datasets": selected_datasets,
            "selected_dataset_count": len(selected_datasets),
            "selection_mode": "explicit_or_full_ucr",
        },
        extra={
            "gpu_ids": args.gpu_ids,
            "shots": list(shots),
            "num_runs": args.num_runs,
            "mode": args.mode,
            "vision_2d_mode": args.vision_2d_mode,
            "vit_patch_size": args.vit_patch_size,
            "vit_stride": args.vit_stride,
            "jobs": jobs,
            "dry_run": bool(args.dry_run),
            "report_config_path": str(report_config_path),
        },
    )
    write_json(report_dir / "ablation_request.json", request_payload)

    if args.dry_run:
        print(f"Prepared vision-backbone ablation config at {report_config_path}")
        print(f"Jobs: {len(jobs)} | datasets: {len(selected_datasets)}")
        return 0

    manifest = generate_report(report_config_path)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {report_dir}")
    print(f"Jobs: {len(jobs)} | datasets: {len(selected_datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
