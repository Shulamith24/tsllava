# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import Any, Iterable

import numpy as np

from opentslm.time_series_datasets.constants import RAW_DATA


MITBIH_DIR = os.path.join(RAW_DATA, "mitbih_arrhythmia")
MITBIH_DB_DIR = os.path.join(MITBIH_DIR, "mitdb")
MITBIH_DB_NAME = "mitdb"
MITBIH_WINDOW_SIZE = 256
MITBIH_PRIMARY_LEAD = "MLII"
MITBIH_AAMI_LABEL_ORDER = ("N", "S", "V", "F", "Q")

# Standard inter-patient split introduced by de Chazal et al. (2004).
DE_CHAZAL_DS1_RECORDS = (
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
)
DE_CHAZAL_DS2_RECORDS = (
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
)

_AAMI_SYMBOL_TO_LABEL = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


def resolve_mitbih_paths(raw_data_path: str) -> tuple[str, str]:
    normalized = os.path.abspath(os.path.expanduser(raw_data_path))
    if os.path.basename(os.path.normpath(normalized)) == MITBIH_DB_NAME:
        return os.path.dirname(normalized), normalized
    if os.path.basename(os.path.normpath(normalized)) == "mitbih_arrhythmia":
        return normalized, os.path.join(normalized, MITBIH_DB_NAME)
    return normalized, os.path.join(normalized, "mitbih_arrhythmia", MITBIH_DB_NAME)


def _require_wfdb():
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "wfdb is required for MIT-BIH loading. Install project dependencies first."
        ) from exc
    return wfdb


def ensure_mitbih_data(raw_data_path: str = MITBIH_DIR) -> str:
    root_dir, db_dir = resolve_mitbih_paths(raw_data_path)
    required_records = sorted(set(DE_CHAZAL_DS1_RECORDS) | set(DE_CHAZAL_DS2_RECORDS))

    missing = [
        record
        for record in required_records
        if not os.path.exists(os.path.join(db_dir, f"{record}.dat"))
        or not os.path.exists(os.path.join(db_dir, f"{record}.hea"))
        or not os.path.exists(os.path.join(db_dir, f"{record}.atr"))
    ]
    if not missing:
        return db_dir

    os.makedirs(root_dir, exist_ok=True)
    wfdb = _require_wfdb()
    wfdb.dl_database(MITBIH_DB_NAME, dl_dir=db_dir, records=required_records)
    return db_dir


def map_mitbih_symbol_to_aami(symbol: str) -> str | None:
    return _AAMI_SYMBOL_TO_LABEL.get(str(symbol).strip())


def extract_centered_window(
    signal: np.ndarray,
    center: int,
    *,
    window_size: int = MITBIH_WINDOW_SIZE,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    half_window = window_size // 2
    start = int(center) - half_window
    end = start + window_size

    clipped_start = max(0, start)
    clipped_end = min(int(signal.shape[0]), end)

    window = np.zeros((window_size,), dtype=np.float32)
    insert_start = clipped_start - start
    insert_end = insert_start + (clipped_end - clipped_start)
    if clipped_end > clipped_start:
        window[insert_start:insert_end] = signal[clipped_start:clipped_end].astype(
            np.float32,
            copy=False,
        )
    return window


def choose_mitbih_primary_lead(record) -> tuple[np.ndarray, str]:
    signal_matrix = record.p_signal
    if signal_matrix is None:
        signal_matrix = np.asarray(record.d_signal, dtype=np.float32)
    if signal_matrix is None:
        raise ValueError("MIT-BIH record does not contain signal data")

    signal_matrix = np.asarray(signal_matrix, dtype=np.float32)
    signal_matrix = np.nan_to_num(signal_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    signal_names = [str(name) for name in getattr(record, "sig_name", [])]
    if MITBIH_PRIMARY_LEAD in signal_names:
        lead_index = signal_names.index(MITBIH_PRIMARY_LEAD)
    else:
        lead_index = 0

    return signal_matrix[:, lead_index], signal_names[lead_index] if signal_names else "lead_0"


def load_mitbih_record_rows(
    record_name: str,
    *,
    db_dir: str,
    window_size: int = MITBIH_WINDOW_SIZE,
) -> list[dict[str, Any]]:
    wfdb = _require_wfdb()
    record_path = os.path.join(db_dir, record_name)
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal, lead_name = choose_mitbih_primary_lead(record)
    rows: list[dict[str, Any]] = []
    for beat_offset, (sample_index, symbol) in enumerate(
        zip(annotation.sample, annotation.symbol)
    ):
        label = map_mitbih_symbol_to_aami(symbol)
        if label is None:
            continue
        rows.append(
            {
                "record_name": record_name,
                "beat_offset": beat_offset,
                "sample_index": int(sample_index),
                "lead_name": lead_name,
                "label": label,
                "time_series": extract_centered_window(
                    signal,
                    int(sample_index),
                    window_size=window_size,
                ),
            }
        )
    return rows


def load_mitbih_arrhythmia_splits(
    *,
    raw_data_path: str = MITBIH_DIR,
    split_protocol: str = "de_chazal",
    window_size: int = MITBIH_WINDOW_SIZE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = str(split_protocol).strip().lower()
    if protocol != "de_chazal":
        raise ValueError(
            f"Unsupported MIT-BIH split_protocol: {split_protocol}. Expected 'de_chazal'."
        )

    db_dir = ensure_mitbih_data(raw_data_path=raw_data_path)
    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for record_name in DE_CHAZAL_DS1_RECORDS:
        train_rows.extend(
            load_mitbih_record_rows(
                record_name,
                db_dir=db_dir,
                window_size=window_size,
            )
        )
    for record_name in DE_CHAZAL_DS2_RECORDS:
        test_rows.extend(
            load_mitbih_record_rows(
                record_name,
                db_dir=db_dir,
                window_size=window_size,
            )
        )
    val_rows = list(test_rows)
    return train_rows, val_rows, test_rows


def summarize_split_records(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        record_name = str(row["record_name"])
        counts[record_name] = counts.get(record_name, 0) + 1
    return counts
