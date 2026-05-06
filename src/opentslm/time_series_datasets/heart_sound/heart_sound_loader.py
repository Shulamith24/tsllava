# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import csv
import os
import wave
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm.auto import tqdm

from opentslm.time_series_datasets.classification_utils import split_rows_stratified
from opentslm.time_series_datasets.constants import RAW_DATA


HEART_SOUND_DIR = os.path.join(RAW_DATA, "cinc2016heart")
HEART_SOUND_URL = "https://physionet.org/files/challenge-2016/1.0.0/training.zip"
HEART_SOUND_SOURCE_SAMPLE_RATE = 2000.0
HEART_SOUND_TARGET_SAMPLE_RATE = 500.0
HEART_SOUND_TARGET_SECONDS = 20
HEART_SOUND_TARGET_LENGTH = int(HEART_SOUND_TARGET_SAMPLE_RATE * HEART_SOUND_TARGET_SECONDS)
HEART_SOUND_LABEL_ORDER = ("normal", "abnormal")


def resolve_heart_sound_paths(raw_data_path: str) -> str:
    normalized = os.path.abspath(os.path.expanduser(raw_data_path))
    basename = os.path.basename(os.path.normpath(normalized))
    if basename == "cinc2016heart" or basename.startswith("training-"):
        return normalized if basename == "cinc2016heart" else os.path.dirname(normalized)
    return os.path.join(normalized, "cinc2016heart")


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


def discover_heart_sound_training_dirs(data_dir: str) -> list[Path]:
    root = Path(data_dir)
    dirs = sorted(
        {
            path.parent
            for path in root.rglob("REFERENCE.csv")
            if path.parent.name.startswith("training-")
        }
    )
    if dirs:
        return dirs
    if root.name.startswith("training-") and (root / "REFERENCE.csv").is_file():
        return [root]
    raise RuntimeError(f"No CinC2016 Heart Sound training directories found under {data_dir}")


def ensure_heart_sound_data(raw_data_path: str = HEART_SOUND_DIR) -> str:
    root_dir = resolve_heart_sound_paths(raw_data_path)
    if os.path.isdir(root_dir):
        try:
            discover_heart_sound_training_dirs(root_dir)
            return root_dir
        except RuntimeError:
            pass

    os.makedirs(root_dir, exist_ok=True)
    zip_path = os.path.join(root_dir, "training.zip")
    if not os.path.exists(zip_path):
        _download_file(HEART_SOUND_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(root_dir)

    discover_heart_sound_training_dirs(root_dir)
    return root_dir


def normalize_heart_sound_label(label: str | int | float) -> str | None:
    normalized = str(label).strip().lower()
    mapping = {
        "-1": "normal",
        "-1.0": "normal",
        "normal": "normal",
        "n": "normal",
        "1": "abnormal",
        "1.0": "abnormal",
        "abnormal": "abnormal",
        "a": "abnormal",
    }
    return mapping.get(normalized)


def read_heart_sound_reference(reference_path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(reference_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for raw_row in reader:
            if len(raw_row) < 2:
                continue
            record_name = raw_row[0].strip()
            label = normalize_heart_sound_label(raw_row[1])
            if not record_name or label is None:
                continue
            rows.append({"record_name": record_name, "label": label})
    if not rows:
        raise RuntimeError(f"Heart Sound reference is empty: {reference_path}")
    return rows


def _decode_pcm(raw_bytes: bytes, *, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        data = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
        return data / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        return signed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        data = np.frombuffer(raw_bytes, dtype="<i4").astype(np.float32)
        return data / 2147483648.0
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def read_wav_mono(wav_path: str | Path) -> tuple[np.ndarray, float]:
    with wave.open(str(wav_path), "rb") as reader:
        sample_rate = float(reader.getframerate())
        channels = int(reader.getnchannels())
        sample_width = int(reader.getsampwidth())
        frame_count = int(reader.getnframes())
        raw_bytes = reader.readframes(frame_count)
    values = _decode_pcm(raw_bytes, sample_width=sample_width)
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), sample_rate


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


def _resolve_wav_path(training_dir: Path, record_name: str) -> Path:
    record_path = training_dir / record_name
    if record_path.suffix:
        return record_path
    return record_path.with_suffix(".wav")


def build_heart_sound_rows(
    *,
    raw_data_path: str = HEART_SOUND_DIR,
    target_rate: float = HEART_SOUND_TARGET_SAMPLE_RATE,
    target_length: int = HEART_SOUND_TARGET_LENGTH,
) -> list[dict[str, Any]]:
    data_dir = ensure_heart_sound_data(raw_data_path)
    training_dirs = discover_heart_sound_training_dirs(data_dir)

    rows: list[dict[str, Any]] = []
    for training_dir in training_dirs:
        reference_rows = read_heart_sound_reference(training_dir / "REFERENCE.csv")
        for row in tqdm(reference_rows, desc=f"Loading {training_dir.name} heart sounds"):
            wav_path = _resolve_wav_path(training_dir, row["record_name"])
            signal, source_rate = read_wav_mono(wav_path)
            original_length = int(signal.shape[0])
            signal = resample_series(signal, source_rate=source_rate, target_rate=target_rate)
            signal = center_crop_or_pad(signal, target_length=target_length)
            rows.append(
                {
                    "record_name": row["record_name"],
                    "label": row["label"],
                    "source_database": training_dir.name,
                    "time_series": signal,
                    "sample_rate": float(target_rate),
                    "source_sample_rate": float(source_rate),
                    "original_length": int(original_length),
                    "wav_path": str(wav_path.resolve()),
                }
            )
    if not rows:
        raise RuntimeError("CinC2016 Heart Sound manifest is empty after preprocessing.")
    return rows


def load_heart_sound_splits(
    *,
    raw_data_path: str = HEART_SOUND_DIR,
    split_protocol: str = "stratified",
    seed: int = 42,
    target_rate: float = HEART_SOUND_TARGET_SAMPLE_RATE,
    target_length: int = HEART_SOUND_TARGET_LENGTH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = str(split_protocol).strip().lower()
    if protocol != "stratified":
        raise ValueError(
            f"Unsupported CinC2016 Heart Sound split_protocol: {split_protocol}. Expected 'stratified'."
        )
    rows = build_heart_sound_rows(
        raw_data_path=raw_data_path,
        target_rate=target_rate,
        target_length=target_length,
    )
    return split_rows_stratified(rows, seed=seed, val_fraction=0.1, test_fraction=0.2)
