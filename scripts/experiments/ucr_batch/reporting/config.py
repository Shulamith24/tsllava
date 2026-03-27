from __future__ import annotations

import json
from pathlib import Path

from .common import ModelSpec, ReportConfig


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_path(raw_path: str, *, config_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path

    config_relative = (config_path.parent / path).resolve()
    if config_relative.exists():
        return config_relative
    return (REPO_ROOT / path).resolve()


def load_report_config(config_path: str | Path) -> ReportConfig:
    config_file = Path(config_path).resolve()
    with open(config_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    report_name = str(payload.get("report_name", "")).strip()
    if not report_name:
        raise ValueError("report_name is required in report config")

    raw_models = payload.get("models", [])
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("report config must include a non-empty models list")

    models: list[ModelSpec] = []
    seen_keys: set[str] = set()
    for item in raw_models:
        key = str(item.get("key", "")).strip()
        label = str(item.get("label", "")).strip() or key
        if not key:
            raise ValueError("each model entry requires a non-empty key")
        if key in seen_keys:
            raise ValueError(f"duplicate model key in report config: {key}")
        seen_keys.add(key)

        results_txt = _resolve_path(str(item.get("results_txt", "")), config_path=config_file)
        models.append(
            ModelSpec(
                key=key,
                label=label,
                results_txt=results_txt,
                primary=bool(item.get("primary", False)),
                color=str(item["color"]).strip() if item.get("color") else None,
                marker=str(item["marker"]).strip() if item.get("marker") else None,
            )
        )

    dataset_source = payload.get("dataset_source", "data/UCRArchive_2018")
    coverage_mode = str(payload.get("coverage_mode", "strict")).strip().lower() or "strict"
    if coverage_mode not in {"strict", "intersection"}:
        raise ValueError("coverage_mode must be either 'strict' or 'intersection'")

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
        dataset_source=_resolve_path(str(dataset_source), config_path=config_file),
        coverage_mode=coverage_mode,
        shots=shots,
        models=tuple(models),
    )
