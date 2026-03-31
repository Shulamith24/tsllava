#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Shared helpers for lightweight UCR few-shot baselines."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data")

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fewshot_utils import write_json  # noqa: E402


def cli_flag_was_provided(argv: Optional[Sequence[str]], flag_name: str) -> bool:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in tokens)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def cleanup_checkpoint_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            path.unlink()
            print(f"Removed checkpoint: {path}")
        except OSError as exc:
            print(f"Failed to remove checkpoint {path}: {exc}")


def _python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def resolve_ucr_dataset_dir(dataset_name: str, data_path: str | Path) -> Path:
    candidate = Path(data_path).resolve()
    if candidate.name == dataset_name and candidate.is_dir():
        dataset_dir = candidate
    elif (candidate / dataset_name).is_dir():
        dataset_dir = candidate / dataset_name
    else:
        dataset_dir = candidate / "UCRArchive_2018" / dataset_name

    train_path = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_path = dataset_dir / f"{dataset_name}_TEST.tsv"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"Unable to locate UCR TRAIN/TEST files for dataset {dataset_name} under {candidate}"
        )
    return dataset_dir


def load_ucr_dataframes(dataset_name: str, data_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    dataset_dir = resolve_ucr_dataset_dir(dataset_name, data_path)
    train_path = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_path = dataset_dir / f"{dataset_name}_TEST.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df = pd.read_csv(test_path, sep="\t", header=None)

    num_features = train_df.shape[1] - 1
    columns = ["label"] + [f"t{idx}" for idx in range(1, num_features + 1)]
    train_df.columns = columns
    test_df.columns = columns
    return train_df, test_df, dataset_dir


def encode_labels(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[Any, int], Dict[int, Any]]:
    unique_train_labels = sorted({_python_scalar(value) for value in train_labels.tolist()})
    label_to_index = {label: idx for idx, label in enumerate(unique_train_labels)}

    unseen_test_labels = sorted({_python_scalar(value) for value in test_labels.tolist()} - set(unique_train_labels))
    if unseen_test_labels:
        raise ValueError(f"Test split contains labels unseen in TRAIN: {unseen_test_labels}")

    train_encoded = np.asarray([label_to_index[_python_scalar(value)] for value in train_labels.tolist()], dtype=np.int64)
    test_encoded = np.asarray([label_to_index[_python_scalar(value)] for value in test_labels.tolist()], dtype=np.int64)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return train_encoded, test_encoded, label_to_index, index_to_label


def maybe_normalize_series(features: np.ndarray, *, normalize: bool) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if not normalize:
        return values

    means = values.mean(axis=1, keepdims=True)
    stds = values.std(axis=1, keepdims=True)
    return (values - means) / (stds + 1e-8)


def load_ucr_arrays(
    dataset_name: str,
    *,
    data_path: str,
    normalize: bool,
) -> Dict[str, Any]:
    train_df, test_df, dataset_dir = load_ucr_dataframes(dataset_name, data_path)
    feature_cols = [column for column in train_df.columns if column != "label"]

    train_features = train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    test_features = test_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    train_raw_labels = train_df["label"].to_numpy(copy=True)
    test_raw_labels = test_df["label"].to_numpy(copy=True)

    train_labels, test_labels, label_to_index, index_to_label = encode_labels(train_raw_labels, test_raw_labels)
    train_features = maybe_normalize_series(train_features, normalize=normalize)
    test_features = maybe_normalize_series(test_features, normalize=normalize)

    return {
        "train_features": train_features,
        "test_features": test_features,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "label_to_index": label_to_index,
        "index_to_label": index_to_label,
        "series_length": int(train_features.shape[1]),
        "dataset_dir": dataset_dir,
    }


def build_label_to_indices(labels: np.ndarray) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels.tolist()):
        label_to_indices.setdefault(int(label), []).append(index)
    return label_to_indices


def remap_labels_to_local(labels: np.ndarray, selected_class_ids: List[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    global_to_local = {int(class_id): local_idx for local_idx, class_id in enumerate(selected_class_ids)}
    remapped = np.asarray([global_to_local[int(label)] for label in labels.tolist()], dtype=np.int64)
    return remapped, global_to_local


class SeriesDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        if features.ndim != 2:
            raise ValueError(f"Expected 2D features [N, L], got shape {features.shape}")
        if len(features) != len(labels):
            raise ValueError("Features and labels must have the same number of samples.")
        self.features = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(1)
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


def write_support_info(
    path: Path,
    *,
    dataset: str,
    model: str,
    protocol: str,
    shot_name: str,
    run_id: int,
    run_seed: int,
    support_info: Dict[str, Any],
    index_to_label: Optional[Dict[int, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "protocol": protocol,
        "shot": shot_name,
        "run_id": run_id,
        "seed": run_seed,
        "way": support_info["way"],
        "selected_class_ids": support_info["selected_class_ids"],
        "selected_indices": support_info["selected_indices"],
        "selected_by_class": support_info["selected_by_class"],
        "k_eff_per_class": support_info["k_eff_per_class"],
        "class_train_counts": support_info["class_train_counts"],
        "classes_with_shortage": support_info["classes_with_shortage"],
        "any_shortage": support_info["any_shortage"],
        "support_size": support_info["support_size"],
    }
    if index_to_label is not None:
        payload["selected_original_labels"] = [
            index_to_label[int(class_id)] for class_id in support_info["selected_class_ids"]
        ]
    write_json(path, payload)
