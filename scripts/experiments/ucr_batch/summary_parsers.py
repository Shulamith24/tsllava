from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from ledger import build_success_row


def _fmt_float(value) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.10g}"


def parse_full_summary(dataset: str, dataset_dir: Path, log_file: Path) -> List[Dict[str, str]]:
    final_results_path = dataset_dir / "final_results.json"
    if not final_results_path.exists():
        return []

    with open(final_results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return [
        build_success_row(
            dataset=dataset,
            shot="full",
            accuracy=_fmt_float(payload.get("test_accuracy")),
            accuracy_std=_fmt_float(0.0),
            num_runs="1",
            result_file=str(final_results_path.resolve()),
            log_file=str(log_file.resolve()),
        )
    ]


def parse_fewshot_summary(dataset: str, dataset_dir: Path, log_file: Path) -> List[Dict[str, str]]:
    csv_path = dataset_dir / "fewshot_summary.csv"
    if csv_path.exists():
        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for item in reader:
                shot = str(item.get("shot", "")).strip()
                if not shot:
                    continue
                rows.append(
                    build_success_row(
                        dataset=dataset,
                        shot=shot,
                        accuracy=_fmt_float(item.get("accuracy_mean")),
                        accuracy_std=_fmt_float(item.get("accuracy_std")),
                        num_runs=str(item.get("num_runs", "")),
                        result_file=str(csv_path.resolve()),
                        log_file=str(log_file.resolve()),
                    )
                )
        if rows:
            return rows

    json_path = dataset_dir / "fewshot_summary.json"
    if not json_path.exists():
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = []
    for item in payload.get("shot_summaries", []):
        shot = str(item.get("shot", "")).strip()
        if not shot:
            continue
        rows.append(
            build_success_row(
                dataset=dataset,
                shot=shot,
                accuracy=_fmt_float(item.get("accuracy_mean")),
                accuracy_std=_fmt_float(item.get("accuracy_std")),
                num_runs=str(item.get("num_runs", "")),
                result_file=str(json_path.resolve()),
                log_file=str(log_file.resolve()),
            )
        )
    return rows


def parse_summary_rows(
    *,
    dataset: str,
    dataset_dir: Path,
    log_file: Path,
    summary_kind: str,
) -> List[Dict[str, str]]:
    if summary_kind == "full":
        return parse_full_summary(dataset, dataset_dir, log_file)
    if summary_kind == "fewshot":
        return parse_fewshot_summary(dataset, dataset_dir, log_file)
    raise ValueError(f"Unsupported summary kind: {summary_kind}")
