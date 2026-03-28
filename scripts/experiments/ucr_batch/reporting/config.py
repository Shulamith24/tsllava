from __future__ import annotations

import json
from pathlib import Path

from .common import (
    ReportConfig,
    ReportItemSpec,
    VALID_COVERAGE_MODES,
    VALID_REPORT_KINDS,
    VALID_REPORT_STAGES,
)


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(raw_path: str, *, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (REPO_ROOT / path).resolve()


def _load_items(payload: dict[str, object], *, config_file: Path) -> tuple[ReportItemSpec, ...]:
    raw_items = payload.get("items")
    if raw_items is None:
        raw_items = payload.get("models", [])
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("report config must include a non-empty items/models list")

    items: list[ReportItemSpec] = []
    seen_keys: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("each item entry must be a JSON object")
        key = str(raw_item.get("key", "")).strip()
        label = str(raw_item.get("label", "")).strip() or key
        if not key:
            raise ValueError("each item entry requires a non-empty key")
        if key in seen_keys:
            raise ValueError(f"duplicate item key in report config: {key}")
        seen_keys.add(key)

        job_dir = None
        if raw_item.get("job_dir"):
            job_dir = _resolve_path(str(raw_item["job_dir"]), config_path=config_file)

        raw_results_txt = str(raw_item.get("results_txt", "")).strip()
        if raw_results_txt:
            results_txt = _resolve_path(raw_results_txt, config_path=config_file)
        elif job_dir is not None:
            results_txt = (job_dir / "results.txt").resolve()
        else:
            raise ValueError(f"item {key} must provide either job_dir or results_txt")

        if raw_item.get("batch_config_path"):
            batch_config_path = _resolve_path(str(raw_item["batch_config_path"]), config_path=config_file)
        elif job_dir is not None:
            batch_config_path = (job_dir / "batch_config.json").resolve()
        else:
            batch_config_path = None

        raw_variant_tags = raw_item.get("variant_tags", [])
        if raw_variant_tags in (None, ""):
            variant_tags: tuple[str, ...] = tuple()
        elif isinstance(raw_variant_tags, list):
            variant_tags = tuple(str(tag).strip() for tag in raw_variant_tags if str(tag).strip())
        else:
            raise ValueError(f"variant_tags for item {key} must be omitted or a list")

        items.append(
            ReportItemSpec(
                key=key,
                label=label,
                results_txt=results_txt,
                job_dir=job_dir,
                batch_config_path=batch_config_path,
                primary=bool(raw_item.get("primary", False)),
                color=str(raw_item["color"]).strip() if raw_item.get("color") else None,
                marker=str(raw_item["marker"]).strip() if raw_item.get("marker") else None,
                variant_tags=variant_tags,
            )
        )
    return tuple(items)


def load_report_config(config_path: str | Path) -> ReportConfig:
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    report_name = str(payload.get("report_name", "")).strip()
    if not report_name:
        raise ValueError("report_name is required in report config")

    report_kind = str(payload.get("report_kind", "leaderboard")).strip().lower() or "leaderboard"
    if report_kind not in VALID_REPORT_KINDS:
        raise ValueError(f"report_kind must be one of {sorted(VALID_REPORT_KINDS)}")

    raw_report_stage = payload.get("report_stage")
    if raw_report_stage is None:
        if report_kind == "leaderboard":
            report_stage = "final"
        else:
            raise ValueError("ablation reports must set report_stage to preview or final")
    else:
        report_stage = str(raw_report_stage).strip().lower()
        if report_stage not in VALID_REPORT_STAGES:
            raise ValueError(f"report_stage must be one of {sorted(VALID_REPORT_STAGES)}")

    items = _load_items(payload, config_file=config_file)

    dataset_source = payload.get("dataset_source", "data/UCRArchive_2018")
    raw_coverage_mode = str(payload.get("coverage_mode", "strict")).strip().lower() or "strict"
    if raw_coverage_mode not in VALID_COVERAGE_MODES:
        raise ValueError(f"coverage_mode must be one of {sorted(VALID_COVERAGE_MODES)}")

    reference_key = str(payload.get("reference_key", "")).strip() or None
    family_label = str(payload.get("family_label", "")).strip() or None

    if report_kind == "ablation":
        if len(items) < 2:
            raise ValueError("ablation reports require at least two items")
        if not reference_key:
            raise ValueError("ablation reports require a non-empty reference_key")
        item_keys = {item.key for item in items}
        if reference_key not in item_keys:
            raise ValueError(f"reference_key {reference_key} is not present in items")
        expected_coverage_mode = "intersection" if report_stage == "preview" else "strict"
        if "coverage_mode" in payload and raw_coverage_mode != expected_coverage_mode:
            raise ValueError(
                f"ablation report_stage={report_stage} requires coverage_mode={expected_coverage_mode}"
            )
        coverage_mode = expected_coverage_mode
        if not family_label:
            family_label = report_name
    else:
        coverage_mode = raw_coverage_mode

    raw_shots = payload.get("shots")
    if raw_shots is None:
        shots = None
    elif isinstance(raw_shots, list) and raw_shots:
        shots = tuple(str(item).strip() for item in raw_shots if str(item).strip())
        if not shots:
            raise ValueError("shots must not be empty when provided")
    else:
        raise ValueError("shots must be omitted or provided as a non-empty list")

    return ReportConfig(
        report_name=report_name,
        config_path=config_file,
        report_kind=report_kind,
        report_stage=report_stage,
        dataset_source=_resolve_path(str(dataset_source), config_path=config_file),
        coverage_mode=coverage_mode,
        shots=shots,
        items=items,
        reference_key=reference_key,
        family_label=family_label,
    )
