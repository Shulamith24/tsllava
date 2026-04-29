#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Few-shot univariate classification with the COSCO ResNet training recipe.

Protocol:
- official TRAIN split is the support pool
- official TEST split is the query/evaluation split
- support sets are sampled per shot/run from TRAIN
- labels are globally encoded once per dataset, then remapped to local 0..way-1 per run
- training follows COSCO's ResNet + SAM + prototypical-loss setup
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data")
DEFAULT_FEWSHOT_SAVE_DIR = "results/ablations/cosco_resnet_fewshot"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset  # noqa: E402

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
from ucr_fewshot_baseline_utils import load_univariate_arrays  # noqa: E402


def _load_python_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COSCO_ROOT: Optional[Path] = None
ResNet = None
enable_running_stats = None
disable_running_stats = None
PrototypicalLoss = None
prototypical_testing = None
SAM = None


def _is_valid_cosco_root(path: Path) -> bool:
    required_files = (
        path / "Baselines" / "ResNet.py",
        path / "Prototypical_Loss.py",
        path / "SAM.py",
    )
    return all(file_path.is_file() for file_path in required_files)


def resolve_cosco_root(explicit_root: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))

    env_root = os.environ.get("TSLLAVA_COSCO_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend(
        [
            PROJECT_ROOT / "temp" / "COSCO-main",
            PROJECT_ROOT / "temp" / "COSCO",
            PROJECT_ROOT / "temp" / "cosco",
        ]
    )

    temp_root = PROJECT_ROOT / "temp"
    if temp_root.is_dir():
        for candidate in sorted(temp_root.iterdir()):
            if candidate.is_dir():
                candidates.append(candidate)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_valid_cosco_root(resolved):
            return resolved

    searched = ", ".join(str(path) for path in seen) if seen else "<none>"
    raise FileNotFoundError(
        "Unable to locate a COSCO checkout relative to the repo. "
        "Pass --cosco_root or set TSLLAVA_COSCO_ROOT. "
        f"Searched: {searched}"
    )


def ensure_cosco_components_loaded(cosco_root: Path) -> None:
    global COSCO_ROOT
    global ResNet
    global enable_running_stats
    global disable_running_stats
    global PrototypicalLoss
    global prototypical_testing
    global SAM

    if ResNet is not None and COSCO_ROOT == cosco_root:
        return

    resnet_module = _load_python_module(
        f"cosco_resnet_module_{abs(hash(str(cosco_root)))}",
        cosco_root / "Baselines" / "ResNet.py",
    )
    proto_module = _load_python_module(
        f"cosco_proto_module_{abs(hash(str(cosco_root)))}",
        cosco_root / "Prototypical_Loss.py",
    )
    sam_module = _load_python_module(
        f"cosco_sam_module_{abs(hash(str(cosco_root)))}",
        cosco_root / "SAM.py",
    )

    COSCO_ROOT = cosco_root
    ResNet = resnet_module.ResNet
    enable_running_stats = resnet_module.enable_running_stats
    disable_running_stats = resnet_module.disable_running_stats
    PrototypicalLoss = proto_module.PrototypicalLoss
    prototypical_testing = proto_module.prototypical_testing
    SAM = sam_module.SAM


def cli_flag_was_provided(argv: Optional[Sequence[str]], flag_name: str) -> bool:
    tokens = list(argv) if argv is not None else sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in tokens)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Few-shot univariate classification with the COSCO ResNet recipe"
    )

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
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Base path containing the selected dataset family. Defaults to the repo's ./data directory.",
    )
    parser.add_argument(
        "--cosco_root",
        type=str,
        default=None,
        help="Optional path to the COSCO checkout. Defaults to auto-discovery under ./temp.",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply per-sample z-normalization before training and evaluation.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--rho", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default=DEFAULT_FEWSHOT_SAVE_DIR)
    parser.add_argument("--resume", action="store_true", help="Reuse completed run outputs when available.")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove per-run checkpoints after final results are written.",
    )

    args = parser.parse_args(argv)
    args.save_dir_explicit = cli_flag_was_provided(provided_argv, "--save_dir")
    args.protocol = "fewshot"
    return args


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def encode_labels(
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[Any, int], Dict[int, Any]]:
    unique_train_labels = sorted({_python_scalar(value) for value in train_labels.tolist()})
    label_to_index = {label: idx for idx, label in enumerate(unique_train_labels)}

    unseen_test_labels = sorted({_python_scalar(value) for value in test_labels.tolist()} - set(unique_train_labels))
    if unseen_test_labels:
        raise ValueError(
            f"Test split contains labels unseen in TRAIN for dataset {unseen_test_labels}"
        )

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
    train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=data_path)
    feature_cols = [column for column in train_df.columns if column != "label"]

    train_features = train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    test_features = test_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    train_raw_labels = train_df["label"].to_numpy(copy=True)
    test_raw_labels = test_df["label"].to_numpy(copy=True)

    train_labels, test_labels, label_to_index, index_to_label = encode_labels(
        train_raw_labels,
        test_raw_labels,
    )
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
    }


def build_label_to_indices(labels: np.ndarray) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        label_to_indices[int(label)].append(index)
    return dict(label_to_indices)


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


def train_cosco_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    rho: float,
    momentum: float,
) -> Dict[str, float]:
    criterion = PrototypicalLoss(flag="neg")
    base_optimizer = torch.optim.SGD
    optimizer = SAM(
        model.parameters(),
        base_optimizer,
        lr=learning_rate,
        momentum=momentum,
        rho=rho,
    )

    last_epoch_loss = 0.0
    last_epoch_second_loss = 0.0
    last_loss = 0.0
    last_second_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_total = 0.0
        epoch_second_loss_total = 0.0
        batch_count = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            enable_running_stats(model)
            optimizer.zero_grad()
            _, embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            disable_running_stats(model)
            _, second_embeddings = model(inputs)
            second_loss = criterion(second_embeddings, labels)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)

            last_loss = float(loss.item())
            last_second_loss = float(second_loss.item())
            epoch_loss_total += last_loss
            epoch_second_loss_total += last_second_loss
            batch_count += 1

        last_epoch_loss = epoch_loss_total / max(1, batch_count)
        last_epoch_second_loss = epoch_second_loss_total / max(1, batch_count)
        print(
            f"Epoch {epoch + 1:03d}/{num_epochs:03d} "
            f"train_loss={last_epoch_loss:.6f} second_loss={last_epoch_second_loss:.6f}"
        )

    return {
        "last_loss": last_loss,
        "last_second_loss": last_second_loss,
        "last_epoch_loss": last_epoch_loss,
        "last_epoch_second_loss": last_epoch_second_loss,
    }


def compute_support_centroids(
    model: torch.nn.Module,
    support_loader: DataLoader,
    *,
    device: torch.device,
) -> torch.Tensor:
    criterion = PrototypicalLoss(flag="neg")
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in support_loader:
            inputs = inputs.to(device)
            _, embeddings = model(inputs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    if not all_embeddings:
        raise RuntimeError("Support loader produced no embeddings; cannot compute centroids.")

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return criterion._compute_class_centroid(labels, embeddings)


def evaluate_model(
    model: torch.nn.Module,
    query_loader: DataLoader,
    *,
    centroids: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    criterion = PrototypicalLoss(flag="neg")
    centroids_device = centroids.to(device)

    total_loss = 0.0
    total_examples = 0
    predictions_local: List[int] = []
    labels_local: List[int] = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in query_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            _, embeddings = model(inputs)
            distances = criterion._distance_matrix(embeddings, centroids_device)
            loss = criterion._prototypical_loss_neg(distances, labels)
            batch_predictions = prototypical_testing(embeddings.detach().cpu(), centroids)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            predictions_local.extend(int(item) for item in batch_predictions.tolist())
            labels_local.extend(int(item) for item in labels.detach().cpu().tolist())

    if total_examples == 0:
        raise RuntimeError("Query loader produced no examples; cannot evaluate COSCO model.")

    accuracy = sum(int(pred == label) for pred, label in zip(predictions_local, labels_local)) / total_examples
    return {
        "loss": total_loss / total_examples,
        "accuracy": accuracy,
        "predictions_local": predictions_local,
        "labels_local": labels_local,
    }


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
    device: torch.device,
) -> Dict[str, Any]:
    shot_name = shot_to_name(shot)
    run_dir = save_root / f"shot_{shot_name}" / f"run_{run_id:02d}"
    run_metrics_path = run_dir / "run_metrics.json"
    support_info_path = run_dir / "fewshot_indices.json"
    checkpoint_path = run_dir / "last.pt"

    run_dir.mkdir(parents=True, exist_ok=True)
    completed_run_exists = args.resume and run_metrics_path.exists()
    if completed_run_exists:
        print(f"[shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
        with open(run_metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    if args.resume and support_info_path.exists():
        with open(support_info_path, "r", encoding="utf-8") as handle:
            support_info = json.load(handle)
    else:
        support_info = sample_support_info(
            label_to_indices,
            shot,
            run_seed,
            way=args.way,
        )
        write_json(
            support_info_path,
            {
                "dataset": args.dataset,
                "protocol": args.protocol,
                "shot": shot_name,
                "run_id": run_id,
                "seed": run_seed,
                "way": support_info["way"],
                "selected_class_ids": support_info["selected_class_ids"],
                "selected_original_labels": [
                    index_to_label[int(class_id)] for class_id in support_info["selected_class_ids"]
                ],
                "selected_indices": support_info["selected_indices"],
                "selected_by_class": support_info["selected_by_class"],
                "k_eff_per_class": support_info["k_eff_per_class"],
                "class_train_counts": support_info["class_train_counts"],
                "classes_with_shortage": support_info["classes_with_shortage"],
                "any_shortage": support_info["any_shortage"],
                "support_size": support_info["support_size"],
            },
        )

    selected_class_ids = [int(class_id) for class_id in support_info["selected_class_ids"]]
    support_indices = [int(index) for index in support_info["selected_indices"]]
    query_indices = [int(index) for index in filter_indices_by_class_ids(test_label_to_indices, selected_class_ids)]
    if not query_indices:
        raise RuntimeError(
            f"No TEST examples found for selected classes {selected_class_ids} in dataset {args.dataset}."
        )

    support_features = train_features[support_indices]
    support_labels_global = train_labels[support_indices]
    query_features = test_features[query_indices]
    query_labels_global = test_labels[query_indices]

    support_labels_local, global_to_local = remap_labels_to_local(support_labels_global, selected_class_ids)
    query_labels_local, _ = remap_labels_to_local(query_labels_global, selected_class_ids)
    local_to_global = {local_idx: global_id for global_id, local_idx in global_to_local.items()}

    support_dataset = SeriesDataset(support_features, support_labels_local)
    query_dataset = SeriesDataset(query_features, query_labels_local)
    train_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(args.batch_size, len(support_dataset))),
        shuffle=True,
    )
    support_eval_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(args.eval_batch_size, len(support_dataset))),
        shuffle=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=max(1, min(args.eval_batch_size, len(query_dataset))),
        shuffle=False,
    )

    model = ResNet(input_size=1, nb_classes=len(selected_class_ids)).to(device)

    print("-" * 80)
    print(
        f"[shot={shot_name} run={run_id}] seed={run_seed}, way={len(selected_class_ids)}, "
        f"support={len(support_indices)}, query={len(query_indices)}, "
        f"batch={train_loader.batch_size}, eval_batch={query_loader.batch_size}"
    )
    print(f"selected global classes: {selected_class_ids}")
    print(
        "selected original labels: "
        f"{[index_to_label[class_id] for class_id in selected_class_ids]}"
    )
    if support_info["any_shortage"]:
        print(f"classes with n<K use-all behavior: {support_info['classes_with_shortage']}")

    train_stats = train_cosco_model(
        model,
        train_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        rho=args.rho,
        momentum=args.momentum,
    )
    support_centroids = compute_support_centroids(
        model,
        support_eval_loader,
        device=device,
    )
    test_results = evaluate_model(
        model,
        query_loader,
        centroids=support_centroids,
        device=device,
    )

    checkpoint_payload = {
        "dataset": args.dataset,
        "protocol": args.protocol,
        "shot": shot_name,
        "run_id": run_id,
        "seed": run_seed,
        "selected_class_ids": selected_class_ids,
        "selected_original_labels": [index_to_label[class_id] for class_id in selected_class_ids],
        "local_to_global_label": local_to_global,
        "support_centroids": support_centroids.cpu(),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    predictions_global = [int(local_to_global[int(local)]) for local in test_results["predictions_local"]]
    labels_global = [int(local_to_global[int(local)]) for local in test_results["labels_local"]]
    predictions_original = [index_to_label[int(class_id)] for class_id in predictions_global]
    labels_original = [index_to_label[int(class_id)] for class_id in labels_global]

    run_metrics = {
        "dataset": args.dataset,
        "model": "cosco_resnet",
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
        "epochs": args.epochs,
        "train_batch_size": train_loader.batch_size,
        "eval_batch_size": query_loader.batch_size,
        "learning_rate": args.lr,
        "rho": args.rho,
        "momentum": args.momentum,
        "normalize": bool(args.normalize),
        "series_length": int(train_features.shape[1]),
        "last_train_loss": train_stats["last_loss"],
        "last_second_loss": train_stats["last_second_loss"],
        "last_epoch_loss": train_stats["last_epoch_loss"],
        "last_epoch_second_loss": train_stats["last_epoch_second_loss"],
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "model_checkpoint": checkpoint_path.name,
    }
    write_json(run_metrics_path, run_metrics)
    write_json(
        run_dir / "test_predictions.json",
        {
            "selected_class_ids": selected_class_ids,
            "selected_original_labels": [index_to_label[class_id] for class_id in selected_class_ids],
            "predictions_local": test_results["predictions_local"],
            "labels_local": test_results["labels_local"],
            "predictions_global": predictions_global,
            "labels_global": labels_global,
            "predictions_original": predictions_original,
            "labels_original": labels_original,
        },
    )

    if args.cleanup_checkpoints:
        cleanup_checkpoint_files([checkpoint_path])

    print(
        f"result: test_acc={test_results['accuracy']:.4f}, "
        f"test_loss={test_results['loss']:.4f}"
    )
    return run_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    shots = parse_shots(args.shots)
    num_runs = max(1, args.num_runs)

    cosco_root = resolve_cosco_root(args.cosco_root)
    ensure_cosco_components_loaded(cosco_root)

    set_seed(args.seed)
    device = resolve_device(args.device)

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
            "device": str(device),
            "num_classes": num_classes,
            "series_length": data_bundle["series_length"],
            "label_to_index": {str(label): index for label, index in data_bundle["label_to_index"].items()},
            "index_to_label": {str(index): label for index, label in index_to_label.items()},
            "temp_source": str(cosco_root),
        },
    )

    print("=" * 80)
    print("COSCO ResNet: Few-shot Univariate Classification")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset_family: {args.dataset_family}")
    print(f"dataset: {args.dataset}")
    print(f"split_protocol: {args.split_protocol}")
    print(f"data_source: {Path(args.data_path).resolve()}")
    print(f"cosco_source: {cosco_root}")
    print(f"protocol: {args.protocol}")
    print(f"shots: {[shot_to_name(shot) for shot in shots]}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"num_runs: {num_runs}")
    print(f"normalize: {bool(args.normalize)}")
    print(f"device: {device}")
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
                device=device,
            )
            shot_run_metrics.append(run_metrics)

        shot_summary = aggregate_shot_results(shot=shot, run_metrics=shot_run_metrics)
        shot_summaries.append(shot_summary)
        shot_dir = save_root / f"shot_{shot_to_name(shot)}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        write_json(shot_dir / "shot_summary.json", shot_summary)
        print(
            f"[shot={shot_summary['shot']}] "
            f"acc={shot_summary['accuracy_mean']:.4f}±{shot_summary['accuracy_std']:.4f}"
        )

    overall_summary = {
        "dataset": args.dataset,
        "model": "cosco_resnet",
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
