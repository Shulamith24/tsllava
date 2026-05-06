# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm.auto import tqdm

from opentslm.time_series_datasets.classification_utils import split_rows_stratified
from opentslm.time_series_datasets.constants import RAW_DATA


CINC2017AF_DIR = os.path.join(RAW_DATA, "cinc2017af")
CINC2017AF_URL = "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip"
CINC2017AF_SOURCE_SAMPLE_RATE = 300.0
CINC2017AF_SAMPLE_RATE = 100.0
CINC2017AF_TARGET_SECONDS = 30
CINC2017AF_TARGET_LENGTH = int(CINC2017AF_SAMPLE_RATE * CINC2017AF_TARGET_SECONDS)
CINC2017AF_LABEL_ORDER = ("N", "A", "O", "~")


def resolve_cinc2017af_paths(raw_data_path: str) -> tuple[str, str]:
    normalized = os.path.abspath(os.path.expanduser(raw_data_path))
    basename = os.path.basename(os.path.normpath(normalized))
    if basename in {"cinc2017af", "challenge-2017"}:
        root_dir = normalized
    elif basename in {"training", "training2017"}:
        root_dir = os.path.dirname(normalized)
    else:
        root_dir = os.path.join(normalized, "cinc2017af")
    return root_dir, os.path.join(root_dir, "training")


def _download_file(url: str, destination: str) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as handle, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {os.path.basename(destination)}",
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))


def _extract_training_zip(zip_path: str, *, root_dir: str, training_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [name for name in archive.namelist() if name and not name.endswith("/")]
        target_dir = root_dir if any(name.startswith("training/") for name in names) else training_dir
        os.makedirs(target_dir, exist_ok=True)
        archive.extractall(target_dir)


def discover_cinc2017af_reference(data_dir: str) -> Path:
    root = Path(data_dir)
    candidate_names = (
        "REFERENCE.csv",
        "REFERENCE-v3.csv",
        "REFERENCE-v2.csv",
        "REFERENCE-v1.csv",
        "REFERENCE-v0.csv",
    )
    for name in candidate_names:
        for path in sorted(root.rglob(name)):
            if path.is_file():
                return path
    raise RuntimeError(f"No CinC2017 AF REFERENCE file found under {data_dir}")


def ensure_cinc2017af_data(raw_data_path: str = CINC2017AF_DIR) -> str:
    root_dir, training_dir = resolve_cinc2017af_paths(raw_data_path)
    for candidate in (training_dir, root_dir):
        if os.path.isdir(candidate):
            try:
                discover_cinc2017af_reference(candidate)
                return candidate
            except RuntimeError:
                pass

    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "training2017.zip")
    if not os.path.exists(zip_path):
        _download_file(CINC2017AF_URL, zip_path)
    _extract_training_zip(zip_path, root_dir=root_dir, training_dir=training_dir)

    for candidate in (training_dir, root_dir):
        try:
            discover_cinc2017af_reference(candidate)
            return candidate
        except RuntimeError:
            pass
    raise RuntimeError(f"Failed to prepare CinC2017 AF data under {root_dir}")


def normalize_cinc2017af_label(label: str) -> str | None:
    normalized = str(label).strip()
    return normalized if normalized in CINC2017AF_LABEL_ORDER else None


def read_cinc2017af_reference(reference_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(reference_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            record_name = raw_row[0].strip()
            label = normalize_cinc2017af_label(raw_row[1])
            if not record_name or label is None:
                continue
            rows.append({"record_name": record_name, "label": label})
    if not rows:
        raise RuntimeError(f"CinC2017 AF reference is empty: {reference_path}")
    return rows


def _require_wfdb():
    try:
        import wfdb  # type: ignore
    except ImportError as exc:
        raise ImportError("wfdb is required for CinC2017 AF loading. Install project dependencies first.") from exc
    return wfdb


def resample_series(values: np.ndarray, *, source_rate: float, target_rate: float) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if series.size == 0:
        return series
    if abs(float(source_rate) - float(target_rate)) < 1e-6:
        return series

    target_length = max(1, int(round(series.size * float(target_rate) / float(source_rate))))
    old_positions = np.linspace(0.0, float(series.size - 1), num=series.size)
    new_positions = np.linspace(0.0, float(series.size - 1), num=target_length)
    return np.interp(new_positions, old_positions, series).astype(np.float32)


def center_crop_or_pad(values: np.ndarray, *, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be positive")
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)
    if series.size == target_length:
        return series
    if series.size > target_length:
        start = (series.size - target_length) // 2
        return series[start : start + target_length]

    padded = np.zeros((target_length,), dtype=np.float32)
    start = (target_length - series.size) // 2
    padded[start : start + series.size] = series
    return padded


def _record_base_path(reference_dir: Path, record_name: str) -> Path:
    record_path = reference_dir / record_name
    if record_path.suffix:
        return record_path.with_suffix("")
    return record_path


def load_cinc2017af_record(
    record_name: str,
    *,
    reference_dir: str | Path,
    target_rate: float = CINC2017AF_SAMPLE_RATE,
    target_length: int = CINC2017AF_TARGET_LENGTH,
) -> tuple[np.ndarray, float, int]:
    wfdb = _require_wfdb()
    base_path = _record_base_path(Path(reference_dir), record_name)
    signals, fields = wfdb.rdsamp(str(base_path))
    signal_matrix = np.asarray(signals, dtype=np.float32)
    if signal_matrix.ndim == 1:
        signal = signal_matrix
    elif signal_matrix.ndim == 2 and signal_matrix.shape[1] >= 1:
        signal = signal_matrix[:, 0]
    else:
        raise ValueError(f"Unexpected CinC2017 AF signal shape for {record_name}: {signal_matrix.shape}")
    source_rate = float(fields.get("fs", CINC2017AF_SOURCE_SAMPLE_RATE))
    original_length = int(signal.shape[0])
    signal = resample_series(signal, source_rate=source_rate, target_rate=target_rate)
    signal = center_crop_or_pad(signal, target_length=target_length)
    return signal, source_rate, original_length


def build_cinc2017af_rows(
    *,
    raw_data_path: str = CINC2017AF_DIR,
    target_rate: float = CINC2017AF_SAMPLE_RATE,
    target_length: int = CINC2017AF_TARGET_LENGTH,
) -> list[dict[str, Any]]:
    data_dir = ensure_cinc2017af_data(raw_data_path)
    reference_path = discover_cinc2017af_reference(data_dir)
    reference_dir = reference_path.parent
    reference_rows = read_cinc2017af_reference(reference_path)

    rows: list[dict[str, Any]] = []
    for row in tqdm(reference_rows, desc="Loading CinC2017 AF records"):
        signal, source_rate, original_length = load_cinc2017af_record(
            row["record_name"],
            reference_dir=reference_dir,
            target_rate=target_rate,
            target_length=target_length,
        )
        rows.append(
            {
                "record_name": row["record_name"],
                "label": row["label"],
                "time_series": signal,
                "sample_rate": float(target_rate),
                "source_sample_rate": float(source_rate),
                "original_length": int(original_length),
            }
        )
    return rows


def load_cinc2017af_splits(
    *,
    raw_data_path: str = CINC2017AF_DIR,
    split_protocol: str = "stratified",
    seed: int = 42,
    target_rate: float = CINC2017AF_SAMPLE_RATE,
    target_length: int = CINC2017AF_TARGET_LENGTH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = str(split_protocol).strip().lower()
    if protocol != "stratified":
        raise ValueError(
            f"Unsupported CinC2017 AF split_protocol: {split_protocol}. Expected 'stratified'."
        )
    rows = build_cinc2017af_rows(
        raw_data_path=raw_data_path,
        target_rate=target_rate,
        target_length=target_length,
    )
    return split_rows_stratified(rows, seed=seed, val_fraction=0.1, test_fraction=0.2)
