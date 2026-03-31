#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Few-shot UCR classification with lightweight local neural backbones."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FEWSHOT_SAVE_DIRS = {
    "resnet": "results/ablations/ucr_resnet_fewshot",
    "tapnet": "results/ablations/ucr_tapnet_fewshot",
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
from simple_backbone_models import build_simple_backbone  # noqa: E402
from ucr_fewshot_baseline_utils import (  # noqa: E402
    DEFAULT_DATA_PATH,
    SeriesDataset,
    build_label_to_indices,
    cleanup_checkpoint_files,
    cli_flag_was_provided,
    load_ucr_arrays,
    remap_labels_to_local,
    resolve_device,
    set_seed,
    write_support_info,
)


@dataclass(frozen=True)
class ModelDefaults:
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float


MODEL_DEFAULTS: Dict[str, ModelDefaults] = {
    "resnet": ModelDefaults(epochs=100, batch_size=64, lr=1e-3, weight_decay=4e-3, dropout=0.0),
    "tapnet": ModelDefaults(epochs=100, batch_size=16, lr=1e-3, weight_decay=1e-4, dropout=0.5),
}


@dataclass(frozen=True)
class ResolvedHParams:
    epochs: int
    batch_size: int
    eval_batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    grad_clip: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Few-shot UCR classification with local PyTorch baselines")

    parser.add_argument("--model", type=str, required=True, choices=["resnet", "tapnet"])
    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot"], help=argparse.SUPPRESS)
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)

    parser.add_argument("--dataset", type=str, required=True, help="UCR dataset name, e.g. ECG200.")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply per-sample z-normalization before training and evaluation.",
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=4.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse completed run outputs when available.")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove per-run checkpoints after final results are written.",
    )

    args = parser.parse_args(argv)
    args.save_dir_explicit = cli_flag_was_provided(provided_argv, "--save_dir")
    if args.save_dir is None:
        args.save_dir = DEFAULT_FEWSHOT_SAVE_DIRS[args.model]
    args.protocol = "fewshot"
    return args


def resolve_hparams(args: argparse.Namespace) -> ResolvedHParams:
    defaults = MODEL_DEFAULTS[args.model]
    batch_size = args.batch_size if args.batch_size is not None else defaults.batch_size
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else max(256, batch_size)
    return ResolvedHParams(
        epochs=args.epochs if args.epochs is not None else defaults.epochs,
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        lr=args.lr if args.lr is not None else defaults.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else defaults.weight_decay,
        dropout=args.dropout if args.dropout is not None else defaults.dropout,
        grad_clip=float(args.grad_clip),
    )


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: List[int] = []
    labels_list: List[int] = []

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits, labels)
        batch_predictions = torch.argmax(logits, dim=-1)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        predictions.extend(int(item) for item in batch_predictions.detach().cpu().tolist())
        labels_list.extend(int(item) for item in labels.detach().cpu().tolist())

    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels_list)) / max(total_examples, 1)
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy,
        "predictions_local": predictions,
        "labels_local": labels_list,
    }


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    support_eval_loader: DataLoader,
    *,
    device: torch.device,
    checkpoint_path: Path,
    args: argparse.Namespace,
    hparams: ResolvedHParams,
) -> Dict[str, Any]:
    optimizer = AdamW(model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)

    loss_history: List[float] = []
    best_support_accuracy = float("-inf")
    best_support_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss_history = [float(item) for item in checkpoint.get("loss_history", [])]
        best_support_accuracy = float(checkpoint.get("best_support_accuracy", best_support_accuracy))
        best_support_loss = float(checkpoint.get("best_support_loss", best_support_loss))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_model_state = checkpoint.get("best_model_state_dict", best_model_state)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resume training from epoch {start_epoch - 1}/{hparams.epochs}")
        if start_epoch > hparams.epochs:
            model.load_state_dict(best_model_state)
            return {
                "loss_history": loss_history,
                "last_train_loss": loss_history[-1] if loss_history else None,
                "best_support_accuracy": best_support_accuracy,
                "best_support_loss": best_support_loss,
                "best_epoch": best_epoch,
            }

    for epoch in range(start_epoch, hparams.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            grad_clip=hparams.grad_clip,
        )
        support_metrics = evaluate(model, support_eval_loader, device=device)
        loss_history.append(train_loss)

        is_better = support_metrics["accuracy"] > best_support_accuracy or (
            support_metrics["accuracy"] == best_support_accuracy and support_metrics["loss"] < best_support_loss
        )
        if is_better:
            best_support_accuracy = float(support_metrics["accuracy"])
            best_support_loss = float(support_metrics["loss"])
            best_epoch = epoch
            best_model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        torch.save(
            {
                "epoch": epoch,
                "loss_history": loss_history,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_support_accuracy": best_support_accuracy,
                "best_support_loss": best_support_loss,
                "best_epoch": best_epoch,
                "best_model_state_dict": best_model_state,
            },
            checkpoint_path,
        )
        print(
            f"Epoch {epoch:03d}/{hparams.epochs:03d} "
            f"train_loss={train_loss:.6f} support_acc={support_metrics['accuracy']:.4f} "
            f"support_loss={support_metrics['loss']:.6f}"
        )

    final_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    best_state_dict = final_checkpoint.get("best_model_state_dict", final_checkpoint["model_state_dict"])
    model.load_state_dict(best_state_dict)
    return {
        "loss_history": loss_history,
        "last_train_loss": loss_history[-1] if loss_history else None,
        "best_support_accuracy": best_support_accuracy,
        "best_support_loss": best_support_loss,
        "best_epoch": best_epoch,
    }


def run_single_experiment(
    *,
    args: argparse.Namespace,
    hparams: ResolvedHParams,
    save_root: Path,
    train_features: torch.Tensor | Any,
    test_features: torch.Tensor | Any,
    train_labels: torch.Tensor | Any,
    test_labels: torch.Tensor | Any,
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
    completed_run_exists = (
        args.resume
        and run_metrics_path.exists()
        and (args.cleanup_checkpoints or checkpoint_path.exists())
    )
    if completed_run_exists:
        print(f"[model={args.model} shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
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
            model=args.model,
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

    support_dataset = SeriesDataset(support_features, support_labels_local)
    query_dataset = SeriesDataset(query_features, query_labels_local)
    train_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(hparams.batch_size, len(support_dataset))),
        shuffle=True,
    )
    support_eval_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(hparams.eval_batch_size, len(support_dataset))),
        shuffle=False,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=max(1, min(hparams.eval_batch_size, len(query_dataset))),
        shuffle=False,
    )

    model = build_simple_backbone(
        args.model,
        input_channels=1,
        num_classes=len(selected_class_ids),
        dropout=hparams.dropout,
    ).to(device)

    print("-" * 80)
    print(
        f"[model={args.model} shot={shot_name} run={run_id}] "
        f"seed={run_seed}, way={len(selected_class_ids)}, support={len(support_indices)}, "
        f"query={len(query_indices)}, batch={train_loader.batch_size}, eval_batch={query_loader.batch_size}"
    )
    print(f"selected global classes: {selected_class_ids}")
    print(f"selected original labels: {[index_to_label[class_id] for class_id in selected_class_ids]}")
    if support_info["any_shortage"]:
        print(f"classes with n<K use-all behavior: {support_info['classes_with_shortage']}")

    train_stats = train_model(
        model,
        train_loader,
        support_eval_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        args=args,
        hparams=hparams,
    )
    test_results = evaluate(model, query_loader, device=device)

    predictions_global = [int(local_to_global[int(local)]) for local in test_results["predictions_local"]]
    labels_global = [int(local_to_global[int(local)]) for local in test_results["labels_local"]]
    predictions_original = [index_to_label[int(class_id)] for class_id in predictions_global]
    labels_original = [index_to_label[int(class_id)] for class_id in labels_global]

    run_metrics = {
        "dataset": args.dataset,
        "model": args.model,
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
        "epochs": hparams.epochs,
        "train_batch_size": train_loader.batch_size,
        "eval_batch_size": query_loader.batch_size,
        "learning_rate": hparams.lr,
        "weight_decay": hparams.weight_decay,
        "dropout": hparams.dropout,
        "grad_clip": hparams.grad_clip,
        "normalize": bool(args.normalize),
        "series_length": int(train_features.shape[1]),
        "last_train_loss": train_stats["last_train_loss"],
        "best_support_accuracy": train_stats["best_support_accuracy"],
        "best_support_loss": train_stats["best_support_loss"],
        "best_epoch": train_stats["best_epoch"],
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

    print(f"result: test_acc={test_results['accuracy']:.4f}, test_loss={test_results['loss']:.4f}")
    return run_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    hparams = resolve_hparams(args)
    shots = parse_shots(args.shots)
    num_runs = max(1, args.num_runs)

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_bundle = load_ucr_arrays(args.dataset, data_path=args.data_path, normalize=bool(args.normalize))
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
            "resolved_hparams": {
                "epochs": hparams.epochs,
                "batch_size": hparams.batch_size,
                "eval_batch_size": hparams.eval_batch_size,
                "lr": hparams.lr,
                "weight_decay": hparams.weight_decay,
                "dropout": hparams.dropout,
                "grad_clip": hparams.grad_clip,
            },
            "label_to_index": {str(label): index for label, index in data_bundle["label_to_index"].items()},
            "index_to_label": {str(index): label for index, label in index_to_label.items()},
            "dataset_dir": str(data_bundle["dataset_dir"]),
        },
    )

    print("=" * 80)
    print(f"{args.model.upper()}: Few-shot UCR Classification")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset: {args.dataset}")
    print(f"data_source: {Path(args.data_path).resolve()}")
    print(f"protocol: {args.protocol}")
    print(f"shots: {[shot_to_name(shot) for shot in shots]}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"num_runs: {num_runs}")
    print(f"normalize: {bool(args.normalize)}")
    print(f"device: {device}")
    print(f"num_classes: {num_classes}")
    print(f"train_size: {len(train_labels)} | test_size: {len(test_labels)}")
    print(f"series_length: {data_bundle['series_length']}")
    print(
        f"resolved_hparams: epochs={hparams.epochs}, batch_size={hparams.batch_size}, "
        f"eval_batch_size={hparams.eval_batch_size}, lr={hparams.lr}, "
        f"weight_decay={hparams.weight_decay}, dropout={hparams.dropout}"
    )
    print("=" * 80)

    shot_summaries = []
    for shot_idx, shot in enumerate(shots):
        shot_run_metrics: List[Dict[str, Any]] = []
        for run_id in range(1, num_runs + 1):
            run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
            set_seed(run_seed)
            run_metrics = run_single_experiment(
                args=args,
                hparams=hparams,
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
            f"acc={shot_summary['accuracy_mean']:.4f}+-{shot_summary['accuracy_std']:.4f}"
        )

    overall_summary = {
        "dataset": args.dataset,
        "model": args.model,
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
