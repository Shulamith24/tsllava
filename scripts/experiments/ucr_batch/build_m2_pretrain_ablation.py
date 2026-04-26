#!/usr/bin/env python3

"""uv run python scripts/experiments/ucr_batch/run_ucr_batch.py \
  --experiment m2_pretrained \
  --protocol fewshot \
  --job-name m2_fewshot_6gpu \
  --data-path /root/data1/tsllava/data \
  --gpu-ids 0,1,2,3,4,5 \
  -- \
  --local_checkpoint /root/data1/tsllava/results/from_83/stage01/best_model.pt \
  --epochs 60 \
  --batch_size 8 \
  --eval_batch_size 8 \
  --shots 1,2,5,10 \
  --num_runs 5 \
  --fewshot_batch_mode manual \
  --gradient_accumulation_steps 4 \
  --device cuda \
  --gradient_checkpointing \
  --cleanup_checkpoints \
  --resume"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m2_ablation_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_FEWSHOT_SECOND_JOB_DIR,
    DEFAULT_SHOTS,
    DEFAULT_STAGE012_JOB_DIR,
    DEFAULT_WITHOUT_PRETRAIN_JOB_DIR,
    build_ablation_report_config,
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the M2 curriculum-pretraining ablation report.")
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--without_pretrain_job_dir", default=str(DEFAULT_WITHOUT_PRETRAIN_JOB_DIR))
    parser.add_argument("--stage012_job_dir", default=str(DEFAULT_STAGE012_JOB_DIR))
    parser.add_argument("--full_curriculum_job_dir", default=str(DEFAULT_FEWSHOT_SECOND_JOB_DIR))
    parser.add_argument("--variant_mode", choices=["auto", "three_way", "two_way"], default="auto")
    parser.add_argument("--shots", default=",".join(DEFAULT_SHOTS))
    parser.add_argument("--num_datasets", type=int, default=None)
    parser.add_argument("--dataset_list", default=None, help="Comma-separated explicit dataset subset.")
    parser.add_argument("--sampling_seed", type=int, default=3407)
    parser.add_argument("--subset_manifest", default=None, help="Reuse a previously written subset_manifest.json.")
    parser.add_argument("--report_name", default=None, help="Optional override for the generated report name.")
    parser.add_argument("--appendix_tables", action="store_true", help="Also generate appendix_shot_*.tex tables.")
    parser.add_argument("--dry_run", action="store_true", help="Only write subset/report config artifacts.")
    return parser.parse_args(argv)


def _resolve_variant_mode(
    *,
    requested_mode: str,
    dataset_list: list[str],
    num_datasets: int | None,
    two_way_candidate: list[str],
    three_way_candidate: list[str],
) -> str:
    two_way_set = set(two_way_candidate)
    three_way_set = set(three_way_candidate)
    has_three_way = bool(three_way_candidate)

    if requested_mode == "two_way":
        return "two_way"
    if requested_mode == "three_way":
        if not has_three_way:
            raise ValueError("variant_mode=three_way requested but stage012 coverage is unavailable")
        return "three_way"

    if dataset_list:
        dataset_set = set(dataset_list)
        if dataset_set <= three_way_set:
            return "three_way"
        if dataset_set <= two_way_set:
            return "two_way"
        raise ValueError("dataset_list is not contained in either the two-way or three-way shared pools")

    if has_three_way and (num_datasets is None or num_datasets <= len(three_way_candidate)):
        return "three_way"
    return "two_way"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shots = tuple(str(item) for item in parse_csv_list(args.shots))
    if not shots:
        raise ValueError("--shots must contain at least one shot")

    without_pretrain_job_dir = Path(args.without_pretrain_job_dir).resolve()
    stage012_job_dir = Path(args.stage012_job_dir).resolve()
    full_curriculum_job_dir = Path(args.full_curriculum_job_dir).resolve()

    two_way_candidate = shared_complete_datasets(
        [without_pretrain_job_dir / "results.txt", full_curriculum_job_dir / "results.txt"],
        dataset_source=args.data_path,
        shots=shots,
    )
    three_way_candidate = []
    if (stage012_job_dir / "results.txt").exists():
        three_way_candidate = shared_complete_datasets(
            [
                without_pretrain_job_dir / "results.txt",
                stage012_job_dir / "results.txt",
                full_curriculum_job_dir / "results.txt",
            ],
            dataset_source=args.data_path,
            shots=shots,
        )

    dataset_list = parse_csv_list(args.dataset_list)
    variant_mode = _resolve_variant_mode(
        requested_mode=args.variant_mode,
        dataset_list=dataset_list,
        num_datasets=args.num_datasets,
        two_way_candidate=two_way_candidate,
        three_way_candidate=three_way_candidate,
    )
    candidate_datasets = three_way_candidate if variant_mode == "three_way" else two_way_candidate
    if not candidate_datasets:
        raise ValueError(f"No shared complete datasets found for variant_mode={variant_mode}")

    if args.subset_manifest:
        subset_manifest = load_subset_manifest(Path(args.subset_manifest).resolve())
        selected_datasets = list(subset_manifest["selected_datasets"])
        missing = [dataset for dataset in selected_datasets if dataset not in set(candidate_datasets)]
        if missing:
            raise ValueError(
                "subset_manifest contains datasets outside the current pretrain candidate pool: "
                + ",".join(missing)
            )
    else:
        selected_datasets, selection_mode, metadata = resolve_selected_datasets(
            candidate_datasets=candidate_datasets,
            dataset_source=args.data_path,
            num_datasets=args.num_datasets,
            sampling_seed=args.sampling_seed,
            dataset_list=dataset_list,
        )
        subset_manifest = build_subset_manifest(
            candidate_pool_name=f"pretrain_{variant_mode}_shared_{len(candidate_datasets)}",
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
    report_name = args.report_name or f"m2_pretrain_ablation_{variant_mode}_{subset_id}"
    report_dir = default_report_dir(report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest_path = write_json(report_dir / "subset_manifest.json", subset_manifest)

    if variant_mode == "three_way":
        family_label = "Effect of Curriculum Pretraining"
        items = [
            {
                "key": "without_pretrain",
                "label": "Without Pretrain",
                "job_dir": str(without_pretrain_job_dir),
            },
            {
                "key": "stage012",
                "label": "Stage I-II",
                "job_dir": str(stage012_job_dir),
            },
            {
                "key": "fewshot_second",
                "label": "Full Curriculum",
                "job_dir": str(full_curriculum_job_dir),
                "primary": True,
            },
        ]
    else:
        family_label = "Effect of Full Curriculum Pretraining"
        items = [
            {
                "key": "without_pretrain",
                "label": "Without Pretrain",
                "job_dir": str(without_pretrain_job_dir),
            },
            {
                "key": "fewshot_second",
                "label": "Full Curriculum",
                "job_dir": str(full_curriculum_job_dir),
                "primary": True,
            },
        ]

    report_config = build_ablation_report_config(
        report_name=report_name,
        family_label=family_label,
        reference_key="without_pretrain",
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
        subset_manifest=subset_manifest,
        extra={
            "variant_mode": variant_mode,
            "without_pretrain_job_dir": str(without_pretrain_job_dir),
            "stage012_job_dir": str(stage012_job_dir),
            "full_curriculum_job_dir": str(full_curriculum_job_dir),
            "shots": list(shots),
            "appendix_tables_enabled": bool(args.appendix_tables),
            "dry_run": bool(args.dry_run),
            "subset_manifest_path": str(subset_manifest_path),
            "report_config_path": str(report_config_path),
        },
    )
    write_json(report_dir / "ablation_request.json", request_payload)

    if args.dry_run:
        print(f"Prepared pretraining ablation config at {report_config_path}")
        print(f"Subset manifest: {subset_manifest_path}")
        return 0

    manifest = generate_report(report_config_path)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {report_dir}")
    print(f"Variant mode: {variant_mode} | datasets={len(selected_datasets)} | subset_id={subset_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
