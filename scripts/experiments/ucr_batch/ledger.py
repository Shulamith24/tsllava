from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

LEDGER_FIELDS = [
    "dataset",
    "shot",
    "status",
    "accuracy",
    "accuracy_std",
    "num_runs",
    "result_file",
    "log_file",
    "updated_at",
    "note",
]

Ledger = Dict[Tuple[str, str], Dict[str, str]]


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_ledger(path: Path) -> Ledger:
    ledger: Ledger = {}
    if not path.exists():
        return ledger

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = row.get("dataset", "").strip()
            shot = row.get("shot", "").strip()
            if not dataset or not shot:
                continue
            ledger[(dataset, shot)] = {field: row.get(field, "") for field in LEDGER_FIELDS}
    return ledger


def upsert_row(ledger: Ledger, row: Dict[str, str]) -> None:
    normalized = {field: str(row.get(field, "")) for field in LEDGER_FIELDS}
    ledger[(normalized["dataset"], normalized["shot"])] = normalized


def remove_row(ledger: Ledger, dataset: str, shot: str) -> None:
    ledger.pop((dataset, shot), None)


def write_ledger(path: Path, ledger: Ledger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(ledger.values(), key=lambda item: (item["dataset"], item["shot"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def has_success(ledger: Ledger, dataset: str, shot: str) -> bool:
    row = ledger.get((dataset, shot))
    return row is not None and row.get("status") == "success"


def success_result_file_exists(ledger: Ledger, dataset: str, shot: str) -> bool:
    row = ledger.get((dataset, shot))
    if row is None or row.get("status") != "success":
        return False
    result_file = row.get("result_file", "").strip()
    return bool(result_file) and Path(result_file).exists()


def build_success_row(
    *,
    dataset: str,
    shot: str,
    accuracy: str,
    accuracy_std: str,
    num_runs: str,
    result_file: str,
    log_file: str,
    note: str = "",
) -> Dict[str, str]:
    return {
        "dataset": dataset,
        "shot": shot,
        "status": "success",
        "accuracy": accuracy,
        "accuracy_std": accuracy_std,
        "num_runs": num_runs,
        "result_file": result_file,
        "log_file": log_file,
        "updated_at": utc_timestamp(),
        "note": note,
    }


def build_failure_row(
    *,
    dataset: str,
    status: str,
    result_file: str,
    log_file: str,
    note: str,
) -> Dict[str, str]:
    return {
        "dataset": dataset,
        "shot": "__dataset__",
        "status": status,
        "accuracy": "",
        "accuracy_std": "",
        "num_runs": "",
        "result_file": result_file,
        "log_file": log_file,
        "updated_at": utc_timestamp(),
        "note": note,
    }


def iter_success_shots(ledger: Ledger, dataset: str) -> Iterable[str]:
    for (row_dataset, shot), row in sorted(ledger.items()):
        if row_dataset == dataset and row.get("status") == "success":
            yield shot
