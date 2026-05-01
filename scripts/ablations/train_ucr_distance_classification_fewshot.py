#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Few-shot univariate classification with exact 1-NN distance baselines."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FEWSHOT_SAVE_DIRS = {
    "ed": "results/ablations/ucr_1nn_ed_fewshot",
    "dtw": "results/ablations/ucr_1nn_dtw_fewshot",
}

sys.path.insert(0, str(SCRIPT_DIR))
from fewshot_utils import (  # noqa: E402
    ShotType,
    aggregate_shot_results,
    filter_indices_by_class_ids,
    parse_shots,
    sample_support_info,
    save_shot_summary_csv,
    shot_to_name,
    write_json,
)
from ucr_fewshot_baseline_utils import (  # noqa: E402
    DEFAULT_DATA_PATH,
    build_label_to_indices,
    cli_flag_was_provided,
    load_univariate_arrays,
    remap_labels_to_local,
    set_seed,
    write_support_info,
)

try:
    from numba import njit
except ImportError:  # pragma: no cover - numba is expected in the project env
    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(function):
            return function

        return decorator


@njit
def _dtw_distance_numba(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    m = y.shape[0]
    prev = np.empty(m + 1, dtype=np.float32)
    curr = np.empty(m + 1, dtype=np.float32)

    for j in range(m + 1):
        prev[j] = np.inf
        curr[j] = np.inf
    prev[0] = 0.0

    for i in range(1, n + 1):
        curr[0] = np.inf
        xi = x[i - 1]
        for j in range(1, m + 1):
            cost = abs(xi - y[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m])


@njit
def _pairwise_dtw_distance_numba(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    distances = np.empty((query.shape[0], support.shape[0]), dtype=np.float32)
    for query_idx in range(query.shape[0]):
        for support_idx in range(support.shape[0]):
            distances[query_idx, support_idx] = _dtw_distance_numba(query[query_idx], support[support_idx])
    return distances


def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(_dtw_distance_numba(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)))


def pairwise_ed_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    diff = query[:, None, :] - support[None, :, :]
    squared = np.sum(diff * diff, axis=2, dtype=np.float32)
    return np.sqrt(squared).astype(np.float32, copy=False)


def pairwise_dtw_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    query_array = np.asarray(query, dtype=np.float32)
    support_array = np.asarray(support, dtype=np.float32)
    return _pairwise_dtw_distance_numba(query_array, support_array)


def predict_1nn(
    support_features: np.ndarray,
    support_labels_local: np.ndarray,
    query_features: np.ndarray,
    *,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if metric == "ed":
        distances = pairwise_ed_distance(query_features, support_features)
    elif metric == "dtw":
        distances = pairwise_dtw_distance(query_features, support_features)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    nearest_indices = np.argmin(distances, axis=1)
    predictions = support_labels_local[nearest_indices]
    nearest_distances = distances[np.arange(len(query_features)), nearest_indices]
    return predictions.astype(np.int64), nearest_distances.astype(np.float32), nearest_indices.astype(np.int64)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Few-shot univariate classification with exact 1-NN baselines")

    parser.add_argument("--metric", type=str, required=True, choices=["ed", "dtw"])
    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot"], help=argparse.SUPPRESS)
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)

    parser.add_argument(
        "--dataset_family",
        type=str,
        default="ucr",
        choices=["ucr", "mitbih", "sleepedf"],
        help="Univariate classification dataset family.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name within the selected family.")
    parser.add_argument(
        "--split_protocol",
        type=str,
        default="default",
        help="Dataset-family-specific split protocol.",
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply per-sample z-normalization before distance computation.",
    )

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed run outputs when available.")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Accepted for batch-runner compatibility; this baseline does not write checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)
    args.save_dir_explicit = cli_flag_was_provided(provided_argv, "--save_dir")
    if args.save_dir is None:
        args.save_dir = DEFAULT_FEWSHOT_SAVE_DIRS[args.metric]
    args.protocol = "fewshot"
    return args


def run_single_experiment(
    *,
    args: argparse.Namespace,
    save_root: Path,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    label_to_indices: Dict[int, List[int]],
    test_label_to_indices: Dict[int, List[int]],
    index_to_label: Dict[int, Any],
    num_classes: int,
    shot: ShotType,
    shot_idx: int,
    run_id: int,
    run_seed: int,
) -> Dict[str, Any]:
    shot_name = shot_to_name(shot)
    run_dir = save_root / f"shot_{shot_name}" / f"run_{run_id:02d}"
    run_metrics_path = run_dir / "run_metrics.json"
    support_info_path = run_dir / "fewshot_indices.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and run_metrics_path.exists():
        print(f"[metric={args.metric} shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
        with open(run_metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    if args.resume and support_info_path.exists():
        with open(support_info_path, "r", encoding="utf-8") as handle:
            support_info = json.load(handle)
    else:
        support_info = sample_support_info(label_to_indices, shot, run_seed, way=args.way)
        write_support_info(
            support_info_path,
            dataset=args.dataset,
            model=f"1nn_{args.metric}",
            protocol=args.protocol,
            shot_name=shot_name,
            run_id=run_id,
            run_seed=run_seed,
            support_info=support_info,
            index_to_label=index_to_label,
        )

    selected_class_ids = [int(class_id) for class_id in support_info["selected_class_ids"]]
    support_indices = [int(index) for index in support_info["selected_indices"]]
    query_indices = [int(index) for index in filter_indices_by_class_ids(test_label_to_indices, selected_class_ids)]
    if not query_indices:
        raise RuntimeError(f"No TEST examples found for selected classes {selected_class_ids} in dataset {args.dataset}.")

    support_features = train_features[support_indices]
    support_labels_global = train_labels[support_indices]
    query_features = test_features[query_indices]
    query_labels_global = test_labels[query_indices]

    support_labels_local, global_to_local = remap_labels_to_local(support_labels_global, selected_class_ids)
    query_labels_local, _ = remap_labels_to_local(query_labels_global, selected_class_ids)
    local_to_global = {local_idx: global_id for global_id, local_idx in global_to_local.items()}

    print("-" * 80)
    print(
        f"[metric={args.metric} shot={shot_name} run={run_id}] "
        f"seed={run_seed}, way={len(selected_class_ids)}, support={len(support_indices)}, query={len(query_indices)}"
    )
    print(f"selected global classes: {selected_class_ids}")
    print(f"selected original labels: {[index_to_label[class_id] for class_id in selected_class_ids]}")
    if support_info["any_shortage"]:
        print(f"classes with n<K use-all behavior: {support_info['classes_with_shortage']}")

    predictions_local, nearest_distances, nearest_support_offsets = predict_1nn(
        support_features,
        support_labels_local,
        query_features,
        metric=args.metric,
    )
    accuracy = float(np.mean(predictions_local == query_labels_local)) if len(query_labels_local) else 0.0
    test_loss = float(np.mean(nearest_distances)) if len(nearest_distances) else 0.0

    predictions_global = [int(local_to_global[int(local)]) for local in predictions_local.tolist()]
    labels_global = [int(local_to_global[int(local)]) for local in query_labels_local.tolist()]
    predictions_original = [index_to_label[int(class_id)] for class_id in predictions_global]
    labels_original = [index_to_label[int(class_id)] for class_id in labels_global]
    nearest_support_indices = [support_indices[int(offset)] for offset in nearest_support_offsets.tolist()]

    run_metrics = {
        "dataset": args.dataset,
        "model": f"1nn_{args.metric}",
        "metric": args.metric,
        "protocol": args.protocol,
        "way": len(selected_class_ids),
        "num_classes": num_classes,
        "selected_class_ids": selected_class_ids,
        "selected_original_labels": [index_to_label[class_id] for class_id in selected_class_ids],
        "global_to_local_label": {str(key): value for key, value in global_to_local.items()},
        "shot": shot_name,
        "run_id": run_id,
        "shot_index": shot_idx,
        "seed": run_seed,
        "support_size": len(support_indices),
        "query_size": len(query_indices),
        "k_eff_per_class": support_info["k_eff_per_class"],
        "class_train_counts": support_info["class_train_counts"],
        "classes_with_shortage": support_info["classes_with_shortage"],
        "any_shortage": support_info["any_shortage"],
        "normalize": bool(args.normalize),
        "series_length": int(train_features.shape[1]),
        "test_loss": test_loss,
        "test_accuracy": accuracy,
    }
    write_json(run_metrics_path, run_metrics)
    write_json(
        run_dir / "test_predictions.json",
        {
            "selected_class_ids": selected_class_ids,
            "selected_original_labels": [index_to_label[class_id] for class_id in selected_class_ids],
            "predictions_local": predictions_local.tolist(),
            "labels_local": query_labels_local.tolist(),
            "predictions_global": predictions_global,
            "labels_global": labels_global,
            "predictions_original": predictions_original,
            "labels_original": labels_original,
            "nearest_distances": nearest_distances.tolist(),
            "nearest_support_indices": nearest_support_indices,
        },
    )

    print(f"result: test_acc={accuracy:.4f}, mean_neighbor_distance={test_loss:.4f}")
    return run_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    shots = parse_shots(args.shots)
    num_runs = max(1, args.num_runs)

    set_seed(args.seed)
    data_bundle = load_univariate_arrays(
        args.dataset,
        data_path=args.data_path,
        normalize=bool(args.normalize),
        dataset_family=args.dataset_family,
        split_protocol=args.split_protocol,
    )
    args.dataset_family = str(data_bundle["dataset_family"])
    args.dataset = str(data_bundle["dataset_name"])
    args.split_protocol = str(data_bundle["split_protocol"])
    train_features = data_bundle["train_features"]
    test_features = data_bundle["test_features"]
    train_labels = data_bundle["train_labels"]
    test_labels = data_bundle["test_labels"]
    index_to_label = data_bundle["index_to_label"]
    num_classes = len(index_to_label)

    if args.way is not None and args.way > num_classes:
        raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes}).")

    label_to_indices = build_label_to_indices(train_labels)
    test_label_to_indices = build_label_to_indices(test_labels)

    save_root = Path(args.save_dir) / args.dataset
    save_root.mkdir(parents=True, exist_ok=True)
    write_json(
        save_root / "config.json",
        {
            **vars(args),
            "num_classes": num_classes,
            "series_length": data_bundle["series_length"],
            "label_to_index": {str(label): index for label, index in data_bundle["label_to_index"].items()},
            "index_to_label": {str(index): label for index, label in index_to_label.items()},
            "dataset_dir": str(data_bundle.get("dataset_dir", Path(args.data_path).resolve())),
        },
    )

    print("=" * 80)
    print(f"1NN-{args.metric.upper()}: Few-shot Univariate Classification")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset_family: {args.dataset_family}")
    print(f"dataset: {args.dataset}")
    print(f"split_protocol: {args.split_protocol}")
    print(f"data_source: {Path(args.data_path).resolve()}")
    print(f"protocol: {args.protocol}")
    print(f"shots: {[shot_to_name(shot) for shot in shots]}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"num_runs: {num_runs}")
    print(f"normalize: {bool(args.normalize)}")
    print(f"num_classes: {num_classes}")
    print(f"train_size: {len(train_labels)} | test_size: {len(test_labels)}")
    print(f"series_length: {data_bundle['series_length']}")
    print("=" * 80)

    shot_summaries = []
    for shot_idx, shot in enumerate(shots):
        shot_run_metrics: List[Dict[str, Any]] = []
        for run_id in range(1, num_runs + 1):
            run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
            set_seed(run_seed)
            run_metrics = run_single_experiment(
                args=args,
                save_root=save_root,
                train_features=train_features,
                test_features=test_features,
                train_labels=train_labels,
                test_labels=test_labels,
                label_to_indices=label_to_indices,
                test_label_to_indices=test_label_to_indices,
                index_to_label=index_to_label,
                num_classes=num_classes,
                shot=shot,
                shot_idx=shot_idx,
                run_id=run_id,
                run_seed=run_seed,
            )
            shot_run_metrics.append(run_metrics)

        shot_summary = aggregate_shot_results(shot=shot, run_metrics=shot_run_metrics)
        shot_summaries.append(shot_summary)
        shot_dir = save_root / f"shot_{shot_to_name(shot)}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        write_json(shot_dir / "shot_summary.json", shot_summary)
        print(
            f"[shot={shot_summary['shot']}] "
            f"acc={shot_summary['accuracy_mean']:.4f}+-{shot_summary['accuracy_std']:.4f}"
        )

    overall_summary = {
        "dataset": args.dataset,
        "model": f"1nn_{args.metric}",
        "metric": args.metric,
        "protocol": args.protocol,
        "way": args.way if args.way is not None else num_classes,
        "num_classes": num_classes,
        "shots": [shot_to_name(shot) for shot in shots],
        "num_runs": num_runs,
        "normalize": bool(args.normalize),
        "series_length": data_bundle["series_length"],
        "timestamp": str(datetime.datetime.now()),
        "shot_summaries": shot_summaries,
    }
    write_json(save_root / "fewshot_summary.json", overall_summary)
    save_shot_summary_csv(save_root / "fewshot_summary.csv", shot_summaries)

    print("=" * 80)
    print(f"Done. Results saved to: {save_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
