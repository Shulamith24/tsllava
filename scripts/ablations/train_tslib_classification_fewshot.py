#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Unified few-shot UCR evaluation entrypoint for selected TSLib classifiers."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opentslm.model.TSLibClassification import (  # noqa: E402
    TSLibClassifierAdapter,
    normalize_model_name,
    prepare_tslib_classification_batch,
    resolve_model_profile,
)
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import (  # noqa: E402
    UCRClassificationDataset,
)

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

DEFAULT_FEWSHOT_SAVE_DIR = "results/ablations/tslib_ucr_fewshot"
DEFAULT_FULL_SAVE_DIR = "results/ablations/tslib_ucr_full"
AVAILABLE_MODEL_ALIASES = ("autoformer", "crossformer", "fedformer", "informer", "timesnet")


def cli_flag_was_provided(argv: Optional[List[str]], flag_name: str) -> bool:
    if argv is None:
        argv = sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in argv)


def parse_model_list(value: str) -> List[str]:
    models: List[str] = []
    seen = set()
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        normalized = normalize_model_name(item).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        models.append(normalized)
    if not models:
        raise ValueError("No valid models were provided.")
    return models


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Unified few-shot UCR evaluation for selected TSLib classification models"
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str, default=None, help="Single model alias, e.g. autoformer")
    model_group.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model aliases, e.g. autoformer,timesnet",
    )

    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot", "full"])
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)

    parser.add_argument("--dataset", type=str, default="CricketZ")
    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data"))

    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "last", "repeat"])
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--d_layers", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--factor", type=int, default=None)
    parser.add_argument("--moving_avg", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_kernels", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=4.0)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--save_dir", type=str, default=DEFAULT_FEWSHOT_SAVE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup_checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args(argv)
    args.save_dir_explicit = cli_flag_was_provided(provided_argv, "--save_dir")
    return args


def normalize_protocol_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.protocol == "full":
        if args.way is not None:
            raise ValueError("--way is not allowed when --protocol=full.")
        args.shots = "full"
        args.num_runs = 1
        if not args.save_dir_explicit:
            args.save_dir = DEFAULT_FULL_SAVE_DIR
    elif not args.save_dir_explicit:
        args.save_dir = DEFAULT_FEWSHOT_SAVE_DIR
    return args


def resolve_requested_models(args: argparse.Namespace) -> List[str]:
    if args.model is not None:
        return [normalize_model_name(args.model).lower()]
    return parse_model_list(args.models)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identity_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def cleanup_checkpoint_files(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            path.unlink()
            print(f"Removed checkpoint: {path}")
        except OSError as exc:
            print(f"Failed to remove checkpoint {path}: {exc}")


def build_label_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = {}
    for idx in range(len(dataset)):
        class_id = int(dataset[idx]["int_label"])
        label_to_indices.setdefault(class_id, []).append(idx)
    return label_to_indices


def infer_max_context_length(dataset: Dataset) -> int:
    max_len = 0
    for idx in range(len(dataset)):
        ts_container = dataset[idx]["time_series"]
        ts_raw = ts_container[0] if isinstance(ts_container, list) else ts_container
        ts = torch.as_tensor(ts_raw)
        max_len = max(max_len, int(ts.numel()))
    if max_len <= 0:
        raise RuntimeError("Unable to infer a valid context_length from the dataset.")
    return max_len


def resolve_context_length(args: argparse.Namespace, train_dataset: Dataset, test_dataset: Dataset) -> int:
    inferred = max(infer_max_context_length(train_dataset), infer_max_context_length(test_dataset))
    if args.context_length is None:
        return inferred
    if args.context_length < inferred:
        raise ValueError(
            f"--context_length ({args.context_length}) is smaller than dataset max length ({inferred})."
        )
    return args.context_length


def build_profile_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "e_layers": args.e_layers,
        "d_layers": args.d_layers,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "factor": args.factor,
        "moving_avg": args.moving_avg,
        "top_k": args.top_k,
        "num_kernels": args.num_kernels,
    }


def resolve_device(device_arg: str) -> str:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        return "cpu"
    return device_arg


def build_optimizer_scheduler(
    adapter: TSLibClassifierAdapter,
    *,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
):
    optimizer = RAdam(
        adapter.get_trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))
    return optimizer, scheduler


def train_one_epoch(
    adapter: TSLibClassifierAdapter,
    train_loader: DataLoader,
    *,
    context_length: int,
    device: str,
    pad_mode: str,
    optimizer,
    grad_clip: float,
    epoch_idx: int,
    epoch_total: int,
) -> float:
    adapter.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}/{epoch_total}")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        batch_inputs = prepare_tslib_classification_batch(
            batch,
            context_length=context_length,
            device=device,
            pad_mode=pad_mode,
        )
        loss, _ = adapter.forward_loss(batch_inputs)
        loss.backward()
        clip_grad_norm_(adapter.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    return total_loss / max(num_batches, 1)


def train_single_stage(
    adapter: TSLibClassifierAdapter,
    train_loader: DataLoader,
    *,
    context_length: int,
    device: str,
    pad_mode: str,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    num_epochs: int,
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    optimizer, scheduler = build_optimizer_scheduler(
        adapter,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
    )

    losses: List[float] = []
    start_epoch = 1
    if args.resume and checkpoint_path.exists():
        checkpoint = adapter.load_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        saved_epochs = int(checkpoint.get("num_epochs", 0) or 0)
        if saved_epochs and saved_epochs != num_epochs:
            raise ValueError(
                f"Checkpoint epoch count mismatch: checkpoint has {saved_epochs}, expected {num_epochs}"
            )
        losses = [float(item) for item in checkpoint.get("loss_history", [])]
        completed_epoch = int(checkpoint.get("epoch", 0) or 0)
        start_epoch = completed_epoch + 1
        print(f"Resume training from epoch {completed_epoch}/{num_epochs}")
        if completed_epoch >= num_epochs:
            return {
                "losses": losses,
                "last_loss": checkpoint.get("last_loss", losses[-1] if losses else None),
            }

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = train_one_epoch(
            adapter,
            train_loader,
            context_length=context_length,
            device=device,
            pad_mode=pad_mode,
            optimizer=optimizer,
            grad_clip=grad_clip,
            epoch_idx=epoch,
            epoch_total=num_epochs,
        )
        scheduler.step()
        losses.append(train_loss)
        print(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.6f}")

        adapter.save_checkpoint(
            save_path=str(checkpoint_path),
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=vars(args),
            extra_state={
                "num_epochs": num_epochs,
                "loss_history": losses,
                "last_loss": train_loss,
            },
        )

    return {"losses": losses, "last_loss": losses[-1] if losses else None}


@torch.no_grad()
def evaluate(
    adapter: TSLibClassifierAdapter,
    data_loader: DataLoader,
    *,
    context_length: int,
    device: str,
    pad_mode: str,
    selected_class_ids: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    adapter.eval()

    total_loss = 0.0
    num_batches = 0
    predictions: List[int] = []
    labels: List[int] = []

    for batch in tqdm(data_loader, desc="Testing"):
        batch_inputs = prepare_tslib_classification_batch(
            batch,
            context_length=context_length,
            device=device,
            pad_mode=pad_mode,
        )
        loss, logits = adapter.forward_loss(batch_inputs)
        masked_logits = adapter.mask_logits_for_selected_classes(logits, selected_class_ids)
        batch_predictions = torch.argmax(masked_logits, dim=-1)

        total_loss += float(loss.item())
        num_batches += 1
        predictions.extend(batch_predictions.cpu().tolist())
        labels.extend(batch_inputs["labels"].cpu().tolist())

    accuracy = (
        sum(int(pred == label) for pred, label in zip(predictions, labels)) / len(labels)
        if labels
        else 0.0
    )
    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
        "predictions": predictions,
        "labels": labels,
    }


def run_single_experiment(
    *,
    args: argparse.Namespace,
    model_name: str,
    model_root: Path,
    train_dataset: Dataset,
    test_dataset: Dataset,
    label_to_indices: Dict[int, List[int]],
    test_label_to_indices: Dict[int, List[int]],
    num_classes: int,
    context_length: int,
    profile_overrides: Dict[str, Any],
    shot: ShotType,
    shot_idx: int,
    run_id: int,
    run_seed: int,
    device: str,
) -> Dict[str, Any]:
    shot_name = shot_to_name(shot)
    run_dir = model_root / f"shot_{shot_name}" / f"run_{run_id:02d}"
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
        print(f"[shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
        with open(run_metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)

    support_info = None
    if args.resume and support_info_path.exists():
        with open(support_info_path, "r", encoding="utf-8") as f:
            support_info = json.load(f)
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
                "model": model_name,
                "way": support_info["way"],
                "shot": shot_name,
                "run_id": run_id,
                "seed": run_seed,
                "selected_class_ids": support_info["selected_class_ids"],
                "selected_indices": support_info["selected_indices"],
                "selected_by_class": support_info["selected_by_class"],
                "k_eff_per_class": support_info["k_eff_per_class"],
                "class_train_counts": support_info["class_train_counts"],
                "classes_with_shortage": support_info["classes_with_shortage"],
                "any_shortage": support_info["any_shortage"],
                "support_size": support_info["support_size"],
            },
        )

    support_indices = list(support_info["selected_indices"])
    query_indices = filter_indices_by_class_ids(
        test_label_to_indices,
        support_info["selected_class_ids"],
    )
    support_dataset = Subset(train_dataset, support_indices)
    query_dataset = Subset(test_dataset, query_indices)

    profile = resolve_model_profile(
        model_name,
        context_length=context_length,
        num_classes=num_classes,
        overrides=profile_overrides,
    )
    eval_batch_size = args.eval_batch_size or profile.batch_size

    print("-" * 80)
    print(
        f"[model={model_name.lower()} shot={shot_name} run={run_id}] "
        f"seed={run_seed}, way={support_info['way']}, support={len(support_indices)}, "
        f"query={len(query_indices)}, batch={profile.batch_size}, eval_batch={eval_batch_size}"
    )
    print(f"selected classes: {support_info['selected_class_ids']}")
    if support_info["any_shortage"]:
        print(f"classes with n<K use-all behavior: {support_info['classes_with_shortage']}")

    train_loader = DataLoader(
        support_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )
    test_loader = DataLoader(
        query_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=identity_collate,
    )

    adapter = TSLibClassifierAdapter.build_model(
        model_name=model_name,
        num_classes=num_classes,
        context_length=context_length,
        device=device,
        overrides=profile_overrides,
    )
    print(
        f"resolved profile: epochs={profile.train_epochs}, lr={profile.learning_rate}, "
        f"d_model={profile.d_model}, d_ff={profile.d_ff}, e_layers={profile.e_layers}"
    )

    train_stats = train_single_stage(
        adapter,
        train_loader,
        context_length=context_length,
        device=device,
        pad_mode=args.pad_mode,
        learning_rate=profile.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        num_epochs=profile.train_epochs,
        checkpoint_path=checkpoint_path,
        args=args,
    )

    adapter.load_checkpoint(checkpoint_path=str(checkpoint_path), device=device)
    test_results = evaluate(
        adapter,
        test_loader,
        context_length=context_length,
        device=device,
        pad_mode=args.pad_mode,
        selected_class_ids=support_info["selected_class_ids"],
    )

    run_metrics = {
        "dataset": args.dataset,
        "model": model_name,
        "protocol": args.protocol,
        "way": support_info["way"],
        "selected_class_ids": support_info["selected_class_ids"],
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
        "train_epochs": profile.train_epochs,
        "train_batch_size": profile.batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": profile.learning_rate,
        "context_length": context_length,
        "last_train_loss": train_stats["last_loss"],
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "model_checkpoint": checkpoint_path.name,
        "resolved_profile": profile.to_dict(),
    }
    write_json(run_metrics_path, run_metrics)
    write_json(
        run_dir / "test_predictions.json",
        {
            "predictions": test_results["predictions"],
            "labels": test_results["labels"],
        },
    )

    if args.cleanup_checkpoints:
        cleanup_checkpoint_files([checkpoint_path])

    print(
        f"result: test_acc={test_results['accuracy']:.4f}, "
        f"test_loss={test_results['loss']:.4f}"
    )
    return run_metrics


def run_model(
    *,
    args: argparse.Namespace,
    model_alias: str,
    model_root: Path,
    train_dataset: Dataset,
    test_dataset: Dataset,
    label_to_indices: Dict[int, List[int]],
    test_label_to_indices: Dict[int, List[int]],
    num_classes: int,
    context_length: int,
    shots: List[ShotType],
    num_runs: int,
    device: str,
) -> Dict[str, Any]:
    model_name = normalize_model_name(model_alias)
    profile_overrides = build_profile_overrides(args)
    resolved_profile = resolve_model_profile(
        model_name,
        context_length=context_length,
        num_classes=num_classes,
        overrides=profile_overrides,
    )

    model_root.mkdir(parents=True, exist_ok=True)
    write_json(
        model_root / "config.json",
        {
            **vars(args),
            "model": model_name,
            "context_length": context_length,
            "resolved_profile": resolved_profile.to_dict(),
        },
    )

    shot_summaries = []
    for shot_idx, shot in enumerate(shots):
        shot_name = shot_to_name(shot)
        shot_run_metrics = []
        for run_id in range(1, num_runs + 1):
            run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
            set_seed(run_seed)
            run_metrics = run_single_experiment(
                args=args,
                model_name=model_name,
                model_root=model_root,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                label_to_indices=label_to_indices,
                test_label_to_indices=test_label_to_indices,
                num_classes=num_classes,
                context_length=context_length,
                profile_overrides=profile_overrides,
                shot=shot,
                shot_idx=shot_idx,
                run_id=run_id,
                run_seed=run_seed,
                device=device,
            )
            shot_run_metrics.append(run_metrics)

        shot_summary = aggregate_shot_results(shot=shot, run_metrics=shot_run_metrics)
        shot_summaries.append(shot_summary)
        shot_dir = model_root / f"shot_{shot_name}"
        shot_dir.mkdir(parents=True, exist_ok=True)
        write_json(shot_dir / "shot_summary.json", shot_summary)
        print(
            f"[model={model_alias} shot={shot_name}] "
            f"acc={shot_summary['accuracy_mean']:.4f}±{shot_summary['accuracy_std']:.4f}"
        )

    overall_summary = {
        "dataset": args.dataset,
        "model": model_name,
        "protocol": args.protocol,
        "way": args.way if args.way is not None else num_classes,
        "num_classes": num_classes,
        "shots": [shot_to_name(item) for item in shots],
        "num_runs": num_runs,
        "context_length": context_length,
        "timestamp": str(datetime.datetime.now()),
        "resolved_profile": resolved_profile.to_dict(),
        "shot_summaries": shot_summaries,
    }
    write_json(model_root / "fewshot_summary.json", overall_summary)
    save_shot_summary_csv(model_root / "fewshot_summary.csv", shot_summaries)

    if args.protocol == "full" and shot_summaries:
        full_run = shot_summaries[0]["run_metrics"][0] if shot_summaries[0]["run_metrics"] else {}
        write_json(
            model_root / "final_results.json",
            {
                "dataset": args.dataset,
                "model": model_name,
                "protocol": "full",
                "test_accuracy": full_run.get("test_accuracy"),
                "test_loss": full_run.get("test_loss"),
                "context_length": context_length,
                "resolved_profile": resolved_profile.to_dict(),
            },
        )
    return overall_summary


def save_comparison_outputs(comparison_root: Path, payload: Dict[str, Any]) -> None:
    comparison_root.mkdir(parents=True, exist_ok=True)
    write_json(comparison_root / "comparison_summary.json", payload)

    csv_path = comparison_root / "comparison_summary.csv"
    columns = [
        "model",
        "shot",
        "num_runs",
        "accuracy_mean",
        "accuracy_std",
        "loss_mean",
        "loss_std",
        "support_size_mean",
        "support_size_std",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for model_summary in payload["model_summaries"]:
            model = model_summary["model"].lower()
            for shot_summary in model_summary["shot_summaries"]:
                writer.writerow(
                    {
                        "model": model,
                        "shot": shot_summary["shot"],
                        "num_runs": shot_summary["num_runs"],
                        "accuracy_mean": shot_summary["accuracy_mean"],
                        "accuracy_std": shot_summary["accuracy_std"],
                        "loss_mean": shot_summary["loss_mean"],
                        "loss_std": shot_summary["loss_std"],
                        "support_size_mean": shot_summary["support_size_mean"],
                        "support_size_std": shot_summary["support_size_std"],
                    }
                )


def main(argv: Optional[List[str]] = None) -> None:
    args = normalize_protocol_args(parse_args(argv))
    requested_models = resolve_requested_models(args)
    shots = parse_shots(args.shots) if args.protocol == "fewshot" else ["full"]
    num_runs = max(1, args.num_runs) if args.protocol == "fewshot" else 1
    device = resolve_device(args.device)

    set_seed(args.seed)

    train_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    test_dataset = UCRClassificationDataset(
        split="test",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )

    num_classes = UCRClassificationDataset.get_num_classes()
    label_to_indices = build_label_to_indices(train_dataset)
    test_label_to_indices = build_label_to_indices(test_dataset)
    if args.way is not None and args.way > num_classes:
        raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes})")

    context_length = resolve_context_length(args, train_dataset, test_dataset)

    print("=" * 80)
    print("TSLib UCR Few-shot Comparison")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset: {args.dataset}")
    print(f"models: {requested_models}")
    print(f"protocol: {args.protocol}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"shots: {[shot_to_name(item) for item in shots]}")
    print(f"num_runs: {num_runs}")
    print(f"context_length: {context_length}")
    print(f"pad_mode: {args.pad_mode}")
    print(f"device: {device}")
    print("=" * 80)

    model_summaries = []
    multi_model_mode = len(requested_models) > 1
    for model_alias in requested_models:
        model_root = Path(args.save_dir) / model_alias / args.dataset if multi_model_mode else Path(args.save_dir) / args.dataset
        model_summary = run_model(
            args=args,
            model_alias=model_alias,
            model_root=model_root,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            label_to_indices=label_to_indices,
            test_label_to_indices=test_label_to_indices,
            num_classes=num_classes,
            context_length=context_length,
            shots=shots,
            num_runs=num_runs,
            device=device,
        )
        model_summaries.append(model_summary)

    if multi_model_mode:
        comparison_root = Path(args.save_dir) / args.dataset
        comparison_payload = {
            "dataset": args.dataset,
            "protocol": args.protocol,
            "way": args.way if args.way is not None else num_classes,
            "num_classes": num_classes,
            "shots": [shot_to_name(item) for item in shots],
            "num_runs": num_runs,
            "timestamp": str(datetime.datetime.now()),
            "model_summaries": model_summaries,
        }
        save_comparison_outputs(comparison_root, comparison_payload)
        print(f"Comparison summary saved to: {comparison_root}")


if __name__ == "__main__":
    main()
