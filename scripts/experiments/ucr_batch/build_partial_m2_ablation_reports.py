#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from m2_ablation_common import DEFAULT_DATA_PATH, default_report_dir, write_json
from reporting import generate_report
from ucr_datasets import discover_datasets, resolve_ucr_archive


REPO_ROOT = Path(__file__).resolve().parents[3]
PARTIAL_ROOT = REPO_ROOT / "results" / "ucr_batches" / "m2_pretrained_fewshot"
REFERENCE_RESULTS = REPO_ROOT / "results" / "ucr_batches" / "m2_pretrained" / "fewshot" / "m2_fewshot_6gpu" / "results.txt"
ALL_SHOTS = ("1", "2", "5", "10")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sparse preview ablation tables from the current partial M2 few-shot runs."
    )
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--dry_run", action="store_true", help="Only write report configs without generating tables.")
    return parser.parse_args(argv)


def _read_success_datasets(results_txt: Path) -> set[str]:
    datasets: set[str] = set()
    with open(results_txt, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("status", "")).strip() != "success":
                continue
            dataset = str(row.get("dataset", "")).strip()
            shot = str(row.get("shot", "")).strip()
            if not dataset or not shot or shot == "__dataset__":
                continue
            datasets.add(dataset)
    return datasets


def _ordered_datasets(dataset_source: str | Path, selected: set[str]) -> list[str]:
    archive_dir = resolve_ucr_archive(dataset_source)
    return [dataset for dataset in discover_datasets(archive_dir) if dataset in selected]


def _build_config(
    *,
    report_name: str,
    family_label: str,
    reference_key: str,
    dataset_source: str | Path,
    dataset_allowlist: list[str],
    items: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "report_name": report_name,
        "report_kind": "ablation",
        "report_stage": "preview",
        "coverage_mode": "sparse",
        "family_label": family_label,
        "reference_key": reference_key,
        "dataset_source": str(resolve_ucr_archive(dataset_source)),
        "dataset_allowlist": dataset_allowlist,
        "shots": list(ALL_SHOTS),
        "appendix_tables_enabled": False,
        "items": items,
    }


def _write_and_maybe_generate(config: dict[str, object], *, dry_run: bool) -> Path:
    report_dir = default_report_dir(str(config["report_name"]))
    report_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_json(report_dir / "report_config.generated.json", config)
    if dry_run:
        print(f"Prepared config: {config_path}")
        return config_path

    manifest = generate_report(config_path)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {report_dir}")
    print(f"Datasets: {manifest['dataset_count']} | shots: {','.join(manifest['shots'])}")
    return config_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_path = Path(args.data_path).resolve()

    without_pretrain_results = PARTIAL_ROOT / "without_pretrain_aligned_m2_fewshot_2gpu" / "results.txt"
    ts_only_results = PARTIAL_ROOT / "m2_dualview_vs_main6gpu_ts_only" / "results.txt"
    vision_only_results = PARTIAL_ROOT / "m2_dualview_vs_main6gpu_vision_only" / "results.txt"
    unconstrained_results = PARTIAL_ROOT / "m2_cd_after_dualview_fixed_unconstrained" / "results.txt"

    pretrain_allowlist = _ordered_datasets(data_path, _read_success_datasets(without_pretrain_results))
    dual_view_allowlist = _ordered_datasets(
        data_path,
        _read_success_datasets(ts_only_results) | _read_success_datasets(vision_only_results),
    )
    decoding_allowlist = _ordered_datasets(data_path, _read_success_datasets(unconstrained_results))

    if not pretrain_allowlist:
        raise ValueError("No successful datasets found for the pretraining ablation preview")
    if not dual_view_allowlist:
        raise ValueError("No successful datasets found for the dual-view ablation preview")
    if not decoding_allowlist:
        raise ValueError("No successful datasets found for the constrained-decoding ablation preview")

    configs = [
        _build_config(
            report_name="m2_pretrain_ablation_partial_preview",
            family_label="Effect of Pretraining",
            reference_key="without_pretrain",
            dataset_source=data_path,
            dataset_allowlist=pretrain_allowlist,
            items=[
                {
                    "key": "duotsp",
                    "label": "DuoTSP",
                    "results_txt": str(REFERENCE_RESULTS),
                    "primary": True,
                },
                {
                    "key": "without_pretrain",
                    "label": "Without Pretrain",
                    "results_txt": str(without_pretrain_results),
                },
            ],
        ),
        _build_config(
            report_name="m2_dual_view_ablation_partial_preview",
            family_label="Effect of Dual-view Prompting",
            reference_key="both",
            dataset_source=data_path,
            dataset_allowlist=dual_view_allowlist,
            items=[
                {
                    "key": "both",
                    "label": "Dual-view",
                    "results_txt": str(REFERENCE_RESULTS),
                    "primary": True,
                },
                {
                    "key": "ts_only",
                    "label": "TS-only",
                    "results_txt": str(ts_only_results),
                },
                {
                    "key": "vision_only",
                    "label": "Vision-only",
                    "results_txt": str(vision_only_results),
                },
            ],
        ),
        _build_config(
            report_name="m2_constrained_decoding_ablation_partial_preview",
            family_label="Effect of Constrained Decoding",
            reference_key="constrained",
            dataset_source=data_path,
            dataset_allowlist=decoding_allowlist,
            items=[
                {
                    "key": "constrained",
                    "label": "Constrained",
                    "results_txt": str(REFERENCE_RESULTS),
                    "primary": True,
                },
                {
                    "key": "unconstrained",
                    "label": "Unconstrained",
                    "results_txt": str(unconstrained_results),
                },
            ],
        ),
    ]

    for config in configs:
        _write_and_maybe_generate(config, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
