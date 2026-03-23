#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Reusable few-shot experiment helpers for ablation scripts."""

from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

ShotType = Union[int, Literal["full"]]


def parse_shots(shots_str: str) -> List[ShotType]:
    shots: List[ShotType] = []
    for token in shots_str.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "full":
            shots.append("full")
            continue
        shot = int(item)
        if shot <= 0:
            raise ValueError(f"Shot must be a positive integer or 'full', got: {item}")
        shots.append(shot)

    if not shots:
        raise ValueError("No valid shots were provided.")

    deduped: List[ShotType] = []
    seen = set()
    for shot in shots:
        if shot in seen:
            continue
        seen.add(shot)
        deduped.append(shot)
    return deduped


def shot_to_name(shot: ShotType) -> str:
    return "full" if shot == "full" else str(shot)


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return _to_python_scalar(value)


def write_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)


def build_label_to_indices(
    labels_df,
    sample_ids: Optional[Iterable[Any]] = None,
) -> Dict[int, List[Any]]:
    label_to_indices: Dict[int, List[Any]] = defaultdict(list)
    ids = sample_ids if sample_ids is not None else labels_df.index.tolist()
    for sample_id in ids:
        label_value = int(labels_df.loc[sample_id].iloc[0])
        label_to_indices[label_value].append(_to_python_scalar(sample_id))
    return dict(label_to_indices)


def sample_support_info(
    label_to_indices: Dict[int, List[Any]],
    shot: ShotType,
    seed: int,
    way: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    all_class_ids = sorted(label_to_indices.keys())
    if way is None or way >= len(all_class_ids):
        selected_class_ids = all_class_ids
    else:
        selected_class_ids = sorted(rng.sample(all_class_ids, way))

    selected_indices: List[Any] = []
    selected_by_class: Dict[str, List[Any]] = {}
    class_train_counts: Dict[str, int] = {}
    k_eff_per_class: Dict[str, int] = {}
    classes_with_shortage: List[int] = []

    for class_id in selected_class_ids:
        class_indices = list(label_to_indices[class_id])
        class_train_counts[str(class_id)] = len(class_indices)

        if shot == "full":
            chosen = list(class_indices)
        else:
            requested_k = int(shot)
            k_eff = min(requested_k, len(class_indices))
            if len(class_indices) < requested_k:
                classes_with_shortage.append(class_id)
            chosen = rng.sample(class_indices, k_eff) if k_eff < len(class_indices) else list(class_indices)

        selected_by_class[str(class_id)] = to_jsonable(chosen)
        k_eff_per_class[str(class_id)] = len(chosen)
        selected_indices.extend(chosen)

    rng.shuffle(selected_indices)

    return {
        "selected_class_ids": selected_class_ids,
        "way": len(selected_class_ids),
        "selected_indices": to_jsonable(selected_indices),
        "selected_by_class": selected_by_class,
        "class_train_counts": class_train_counts,
        "k_eff_per_class": k_eff_per_class,
        "classes_with_shortage": classes_with_shortage,
        "any_shortage": bool(classes_with_shortage),
        "support_size": len(selected_indices),
    }


def filter_indices_by_class_ids(
    label_to_indices: Dict[int, List[Any]],
    class_ids: Iterable[int],
) -> List[Any]:
    selected_indices: List[Any] = []
    for class_id in class_ids:
        selected_indices.extend(label_to_indices.get(int(class_id), []))
    return to_jsonable(sorted(selected_indices))


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate_shot_results(
    shot: ShotType,
    run_metrics: List[Dict[str, Any]],
    metric_key: str = "test_accuracy",
) -> Dict[str, Any]:
    metric_values = [float(item[metric_key]) for item in run_metrics if item.get(metric_key) is not None]
    losses = [float(item["test_loss"]) for item in run_metrics if item.get("test_loss") is not None]
    support_sizes = [int(item["support_size"]) for item in run_metrics if item.get("support_size") is not None]

    metric_mean, metric_std = mean_std(metric_values)
    loss_mean, loss_std = mean_std(losses)
    support_mean, support_std = mean_std([float(size) for size in support_sizes])

    return {
        "shot": shot_to_name(shot),
        "num_runs": len(run_metrics),
        f"{metric_key.removeprefix('test_')}_mean": metric_mean,
        f"{metric_key.removeprefix('test_')}_std": metric_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "support_size_mean": support_mean,
        "support_size_std": support_std,
        "any_shortage_in_shot": any(bool(item.get("any_shortage")) for item in run_metrics),
        "run_metrics": run_metrics,
    }


def save_shot_summary_csv(save_path: Union[str, Path], shot_summaries: List[Dict[str, Any]]) -> None:
    columns = [
        "shot",
        "num_runs",
        "accuracy_mean",
        "accuracy_std",
        "loss_mean",
        "loss_std",
        "support_size_mean",
        "support_size_std",
        "any_shortage_in_shot",
    ]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in shot_summaries:
            writer.writerow({key: item.get(key) for key in columns})
