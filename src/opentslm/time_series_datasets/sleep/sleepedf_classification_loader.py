# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

from opentslm.time_series_datasets.constants import RAW_DATA


SLEEPEDF_DIR = os.path.join(RAW_DATA, "sleep_edfx")
SLEEPEDF_CASSETTE_DIRNAME = "sleep-cassette"
SLEEPEDF_INDEX_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"
SLEEPEDF_DEFAULT_CHANNEL = "Fpz-Cz"
SLEEPEDF_DEFAULT_EPOCH_SECONDS = 30
SLEEPEDF_LABEL_ORDER = ("W", "N1", "N2", "N3", "REM")


def resolve_sleepedf_paths(
    raw_data_path: str,
    *,
    channel: str = SLEEPEDF_DEFAULT_CHANNEL,
    epoch_seconds: int = SLEEPEDF_DEFAULT_EPOCH_SECONDS,
) -> tuple[str, str, str]:
    normalized = os.path.abspath(os.path.expanduser(raw_data_path))
    if os.path.basename(os.path.normpath(normalized)) == SLEEPEDF_CASSETTE_DIRNAME:
        root_dir = os.path.dirname(normalized)
        cassette_dir = normalized
    elif os.path.basename(os.path.normpath(normalized)) == "sleep_edfx":
        root_dir = normalized
        cassette_dir = os.path.join(root_dir, SLEEPEDF_CASSETTE_DIRNAME)
    else:
        root_dir = os.path.join(normalized, "sleep_edfx")
        cassette_dir = os.path.join(root_dir, SLEEPEDF_CASSETTE_DIRNAME)

    safe_channel = re.sub(r"[^a-z0-9]+", "_", channel.lower()).strip("_")
    processed_dir = os.path.join(root_dir, f"processed_{safe_channel}_{int(epoch_seconds)}s")
    return root_dir, cassette_dir, processed_dir


def _require_pyedflib():
    try:
        import pyedflib  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pyedflib is required for Sleep-EDF classification loading. "
            "Install project dependencies first."
        ) from exc
    return pyedflib


def extract_sleepedf_subject_id(record_name: str) -> str:
    stem = str(record_name).split("-")[0]
    if not stem.startswith("SC") or len(stem) < 6:
        raise ValueError(f"Unexpected Sleep-EDF record name: {record_name}")
    return stem[:5]


def sleepedf_pair_key(filename: str) -> str:
    stem = Path(filename).stem
    record_stem = stem.split("-")[0]
    if len(record_stem) < 2:
        raise ValueError(f"Unexpected Sleep-EDF filename: {filename}")
    return record_stem[:-1]


def normalize_sleep_stage(description: str) -> str | None:
    desc = str(description).strip()
    mapping = {
        "Sleep stage W": "W",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",
        "Sleep stage R": "REM",
        "Sleep stage ?": None,
        "Movement time": None,
        "Sleep stage M": None,
    }
    if desc in mapping:
        return mapping[desc]
    return None


def expand_sleep_stage_annotations(
    *,
    record_name: str,
    subject_id: str,
    signal_path: str,
    sample_rate: float,
    signal_length: int,
    onsets: Iterable[float],
    durations: Iterable[float],
    descriptions: Iterable[str],
    epoch_seconds: int = SLEEPEDF_DEFAULT_EPOCH_SECONDS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    samples_per_epoch = int(round(float(sample_rate) * int(epoch_seconds)))
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")

    for onset, duration, description in zip(onsets, durations, descriptions):
        label = normalize_sleep_stage(description)
        if label is None:
            continue

        epoch_count = int(round(float(duration) / float(epoch_seconds)))
        if epoch_count <= 0:
            continue

        base_start = int(round(float(onset) * float(sample_rate)))
        for epoch_index in range(epoch_count):
            start_sample = base_start + epoch_index * samples_per_epoch
            if start_sample >= signal_length:
                continue
            rows.append(
                {
                    "record_name": record_name,
                    "subject_id": subject_id,
                    "signal_path": signal_path,
                    "start_sample": start_sample,
                    "num_samples": samples_per_epoch,
                    "sample_rate": float(sample_rate),
                    "label": label,
                }
            )
    return rows


def split_sleepedf_subject_ids(
    subject_ids: list[str],
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[str]]:
    unique_subjects = sorted(set(subject_ids))
    if len(unique_subjects) < 3:
        raise ValueError("Need at least three subjects to create train/validation/test splits")

    rng = random.Random(seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)

    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    if test_count + val_count >= len(shuffled):
        val_count = 1
        test_count = 1

    test_subjects = sorted(shuffled[:test_count])
    val_subjects = sorted(shuffled[test_count:test_count + val_count])
    train_subjects = sorted(shuffled[test_count + val_count :])
    if not train_subjects:
        raise ValueError("Subject split left no training subjects")

    return {
        "train": train_subjects,
        "validation": val_subjects,
        "test": test_subjects,
    }


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


def ensure_sleepedf_cassette_data(raw_data_path: str = SLEEPEDF_DIR) -> str:
    _root_dir, cassette_dir, _processed_dir = resolve_sleepedf_paths(raw_data_path)
    os.makedirs(cassette_dir, exist_ok=True)
    existing_files = [name for name in os.listdir(cassette_dir) if name.endswith(".edf")]
    if existing_files:
        try:
            if discover_sleepedf_cassette_pairs(cassette_dir):
                return cassette_dir
        except Exception:
            pass

    index_response = requests.get(SLEEPEDF_INDEX_URL, timeout=60)
    index_response.raise_for_status()
    file_names = sorted(
        {
            match.group(1)
            for match in re.finditer(r'href="([^"]+\.edf)"', index_response.text)
            if match.group(1).startswith("SC")
        }
    )
    if not file_names:
        raise RuntimeError("Failed to discover Sleep-EDF cassette EDF files from the official index.")

    for file_name in tqdm(file_names, desc="Downloading Sleep-EDF cassette EDF files"):
        destination = os.path.join(cassette_dir, file_name)
        if os.path.exists(destination):
            continue
        _download_file(f"{SLEEPEDF_INDEX_URL}{file_name}", destination)
    return cassette_dir


def discover_sleepedf_cassette_pairs(cassette_dir: str) -> list[tuple[str, str]]:
    files = sorted(
        file_name
        for file_name in os.listdir(cassette_dir)
        if file_name.startswith("SC") and file_name.endswith(".edf")
    )
    grouped: dict[str, dict[str, str]] = {}
    for file_name in files:
        key = sleepedf_pair_key(file_name)
        slot = grouped.setdefault(key, {})
        if "-PSG" in file_name:
            slot["psg"] = os.path.join(cassette_dir, file_name)
        elif "-Hypnogram" in file_name:
            slot["hypnogram"] = os.path.join(cassette_dir, file_name)

    pairs = []
    for key, item in sorted(grouped.items()):
        if "psg" not in item or "hypnogram" not in item:
            continue
        pairs.append((item["psg"], item["hypnogram"]))
    if not pairs:
        raise RuntimeError(f"No valid Sleep-EDF PSG/Hypnogram pairs found in {cassette_dir}")
    return pairs


def _normalize_sleepedf_channel_label(label: str) -> str:
    normalized = re.sub(r"\s+", " ", str(label).strip().lower())
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"^(eeg)\s+", "", normalized)
    return normalized


def _load_sleepedf_signal_channel(psg_path: str, *, channel: str) -> tuple[np.ndarray, float]:
    pyedflib = _require_pyedflib()
    with pyedflib.EdfReader(psg_path) as reader:
        signal_labels = [str(label).strip() for label in reader.getSignalLabels()]
        normalized_labels = [_normalize_sleepedf_channel_label(label) for label in signal_labels]
        target = _normalize_sleepedf_channel_label(channel)
        if target not in normalized_labels:
            raise ValueError(
                f"Channel {channel!r} not found in {psg_path}. Available labels: {signal_labels}"
            )
        signal_index = normalized_labels.index(target)
        signal = np.asarray(reader.readSignal(signal_index), dtype=np.float32)
        sample_rate = float(reader.samplefrequency(signal_index))
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal, sample_rate


def _load_sleepedf_annotations(hypnogram_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pyedflib = _require_pyedflib()
    with pyedflib.EdfReader(hypnogram_path) as reader:
        onsets, durations, descriptions = reader.readAnnotations()
    normalized_descriptions = [str(item) for item in descriptions]
    return np.asarray(onsets), np.asarray(durations), normalized_descriptions


def build_sleepedf_classification_manifest(
    *,
    raw_data_path: str = SLEEPEDF_DIR,
    channel: str = SLEEPEDF_DEFAULT_CHANNEL,
    epoch_seconds: int = SLEEPEDF_DEFAULT_EPOCH_SECONDS,
) -> pd.DataFrame:
    _root_dir, cassette_dir, processed_dir = resolve_sleepedf_paths(
        raw_data_path,
        channel=channel,
        epoch_seconds=epoch_seconds,
    )
    os.makedirs(processed_dir, exist_ok=True)
    manifest_path = os.path.join(processed_dir, "manifest.csv")
    if os.path.exists(manifest_path):
        manifest_df = pd.read_csv(manifest_path)
        if not manifest_df.empty and manifest_df["signal_path"].map(os.path.exists).all():
            return manifest_df

    ensure_sleepedf_cassette_data(raw_data_path)
    pairs = discover_sleepedf_cassette_pairs(cassette_dir)

    manifest_rows: list[dict[str, Any]] = []
    for psg_path, hypnogram_path in tqdm(pairs, desc="Processing Sleep-EDF cassette records"):
        record_name = Path(psg_path).stem
        subject_id = extract_sleepedf_subject_id(record_name)
        signal_array, sample_rate = _load_sleepedf_signal_channel(psg_path, channel=channel)
        signal_cache_path = os.path.join(processed_dir, f"{record_name}.npy")
        if not os.path.exists(signal_cache_path):
            np.save(signal_cache_path, signal_array.astype(np.float32))
        onsets, durations, descriptions = _load_sleepedf_annotations(hypnogram_path)
        manifest_rows.extend(
            expand_sleep_stage_annotations(
                record_name=record_name,
                subject_id=subject_id,
                signal_path=os.path.abspath(signal_cache_path),
                sample_rate=sample_rate,
                signal_length=int(signal_array.shape[0]),
                onsets=onsets,
                durations=durations,
                descriptions=descriptions,
                epoch_seconds=epoch_seconds,
            )
        )

    manifest_df = pd.DataFrame.from_records(manifest_rows)
    if manifest_df.empty:
        raise RuntimeError("Sleep-EDF manifest is empty after preprocessing.")
    manifest_df.to_csv(manifest_path, index=False)
    return manifest_df


def load_sleepedf_classification_splits(
    *,
    raw_data_path: str = SLEEPEDF_DIR,
    split_protocol: str = "subject",
    seed: int = 42,
    channel: str = SLEEPEDF_DEFAULT_CHANNEL,
    epoch_seconds: int = SLEEPEDF_DEFAULT_EPOCH_SECONDS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    protocol = str(split_protocol).strip().lower()
    if protocol != "subject":
        raise ValueError(
            f"Unsupported Sleep-EDF split_protocol: {split_protocol}. Expected 'subject'."
        )

    manifest_df = build_sleepedf_classification_manifest(
        raw_data_path=raw_data_path,
        channel=channel,
        epoch_seconds=epoch_seconds,
    )
    split_paths = resolve_sleepedf_paths(
        raw_data_path,
        channel=channel,
        epoch_seconds=epoch_seconds,
    )
    processed_dir = split_paths[2]
    split_cache_path = os.path.join(processed_dir, f"subject_split_seed_{int(seed)}.json")
    if os.path.exists(split_cache_path):
        with open(split_cache_path, "r", encoding="utf-8") as handle:
            split_subjects = json.load(handle)
    else:
        split_subjects = split_sleepedf_subject_ids(
            manifest_df["subject_id"].astype(str).tolist(),
            seed=seed,
        )
        with open(split_cache_path, "w", encoding="utf-8") as handle:
            json.dump(split_subjects, handle, indent=2, sort_keys=True)

    subsets: dict[str, list[dict[str, Any]]] = {}
    for split_name, subject_ids in split_subjects.items():
        split_df = manifest_df[manifest_df["subject_id"].astype(str).isin(subject_ids)].copy()
        subsets[split_name] = split_df.to_dict(orient="records")

    return subsets["train"], subsets["validation"], subsets["test"]
