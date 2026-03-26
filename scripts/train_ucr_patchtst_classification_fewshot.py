#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Strict few-shot UCR classification with Hugging Face PatchTSTForClassification.

Protocol lock:
- support-set sampling per shot/run
- Phase1 warm-up only; final checkpoint is always Phase2 last
- few-shot full-batch priority for numeric K by default
- if class count < K, use all available samples from that class
"""

import argparse
import csv
import datetime
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.PatchTSTClassifier import (  # noqa: E402
    PatchTSTClassifierAdapter,
    prepare_patchtst_classification_batch,
)
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import (  # noqa: E402
    UCRClassificationDataset,
)

ShotType = Union[int, Literal["full"]]
STRICT_FEWSHOT_EPOCHS = 100
DEFAULT_FEWSHOT_SAVE_DIR = "results/patchtst_ucr_fewshot"
DEFAULT_FULL_SAVE_DIR = "results/patchtst_ucr_full"


def parse_args(argv=None):
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Strict few-shot supervised UCR classification with PatchTST"
    )

    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot", "full"])
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument(
        "--way",
        type=int,
        default=None,
        help="Number of classes to sample per run for strict N-way few-shot. Defaults to all classes.",
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)
    parser.add_argument("--model_select_metric", type=str, default="last", choices=["last", "train_loss"])
    parser.add_argument("--fewshot_batch_mode", type=str, default="manual", choices=["full", "manual"])

    parser.add_argument("--epochs", type=int, default=STRICT_FEWSHOT_EPOCHS)
    parser.add_argument("--phase1_epochs", type=int, default=5)

    parser.add_argument("--dataset", type=str, default="CricketZ")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--pad_mode", type=str, default="last", choices=["last", "repeat", "zero"])

    parser.add_argument("--patchtst_model_id", type=str, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--patch_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--head_dropout", type=float, default=None)
    parser.add_argument(
        "--use_cls_token",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CLS token pooling in PatchTST.",
    )
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument(
        "--reset_head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the classification head after loading pretrained weights.",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--save_dir", type=str, default=DEFAULT_FEWSHOT_SAVE_DIR)
    parser.add_argument("--resume", action="store_true", help="Resume from existing run checkpoints when available.")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Remove per-run phase checkpoints after writing final results to save disk space.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args(argv)
    args.save_dir_explicit = any(
        token == "--save_dir" or token.startswith("--save_dir=") for token in provided_argv
    )
    return args


def setup_distributed() -> Tuple[int, int, int]:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return local_rank, world_size, rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def cleanup_checkpoint_files(paths: List[str], rank: int = 0):
    """Remove no-longer-needed checkpoints without failing the run."""
    if rank != 0:
        return

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            os.remove(path)
            print(f"Removed checkpoint: {path}")
        except OSError as exc:
            print(f"Failed to remove checkpoint {path}: {exc}")


def get_model(model):
    return model.module if hasattr(model, "module") else model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_protocol_args(args):
    if args.protocol == "full":
        if args.way is not None:
            raise ValueError("--way is not allowed when --protocol=full; full supervision must use all classes.")
        args.shots = "full"
        args.num_runs = 1
        if not getattr(args, "save_dir_explicit", False):
            args.save_dir = DEFAULT_FULL_SAVE_DIR
        return args

    if not getattr(args, "save_dir_explicit", False):
        args.save_dir = DEFAULT_FEWSHOT_SAVE_DIR
    return args


def parse_shots(shots_str: str) -> List[ShotType]:
    shots: List[ShotType] = []
    for token in shots_str.split(","):
        item = token.strip().lower()
        if not item:
            continue
        if item == "full":
            shots.append("full")
            continue
        k = int(item)
        if k <= 0:
            raise ValueError(f"Shot must be positive integer or 'full', got: {item}")
        shots.append(k)

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


def build_label_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label_to_indices[int(dataset[idx]["int_label"])].append(idx)
    return dict(label_to_indices)


def sample_support_info(
    label_to_indices: Dict[int, List[int]],
    shot: ShotType,
    seed: int,
    way: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    selected_indices: List[int] = []
    selected_by_class: Dict[str, List[int]] = {}
    class_train_counts: Dict[str, int] = {}
    k_eff_per_class: Dict[str, int] = {}
    classes_with_shortage: List[int] = []

    all_class_ids = sorted(label_to_indices.keys())
    if way is None or way >= len(all_class_ids):
        selected_class_ids = all_class_ids
    else:
        selected_class_ids = sorted(rng.sample(all_class_ids, way))

    for class_id in selected_class_ids:
        class_indices = list(label_to_indices[class_id])
        class_train_counts[str(class_id)] = len(class_indices)

        if shot == "full":
            chosen = class_indices
        else:
            requested_k = int(shot)
            k_eff = min(requested_k, len(class_indices))
            if len(class_indices) < requested_k:
                classes_with_shortage.append(class_id)
            chosen = (
                rng.sample(class_indices, k_eff)
                if k_eff < len(class_indices)
                else class_indices
            )

        selected_by_class[str(class_id)] = chosen
        k_eff_per_class[str(class_id)] = len(chosen)
        selected_indices.extend(chosen)

    rng.shuffle(selected_indices)

    return {
        "selected_class_ids": selected_class_ids,
        "way": len(selected_class_ids),
        "selected_indices": selected_indices,
        "selected_by_class": selected_by_class,
        "class_train_counts": class_train_counts,
        "k_eff_per_class": k_eff_per_class,
        "classes_with_shortage": classes_with_shortage,
        "any_shortage": bool(classes_with_shortage),
        "support_size": len(selected_indices),
    }


def filter_indices_by_class_ids(
    label_to_indices: Dict[int, List[int]],
    class_ids: List[int],
) -> List[int]:
    selected_indices: List[int] = []
    for class_id in class_ids:
        selected_indices.extend(label_to_indices.get(class_id, []))
    return sorted(selected_indices)


def broadcast_object_from_rank0(obj, world_size: int, rank: int):
    if world_size == 1:
        return obj
    holder = [obj if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def resolve_phase_epochs(total_epochs: int, phase1_epochs: int) -> Tuple[int, int]:
    if total_epochs <= 0:
        return 0, 0

    safe_phase1 = min(max(0, phase1_epochs), max(0, total_epochs - 1))
    phase2 = total_epochs - safe_phase1
    if phase2 < 1:
        phase2 = 1
        safe_phase1 = max(0, total_epochs - 1)
    return safe_phase1, phase2


def compute_fewshot_train_hparams(
    args,
    shot: ShotType,
    support_size: int,
) -> Tuple[int, int]:
    if (
        args.protocol == "fewshot"
        and isinstance(shot, int)
        and args.fewshot_batch_mode == "full"
    ):
        return max(1, support_size), 1
    return args.batch_size, args.gradient_accumulation_steps


def infer_context_length(dataset: Dataset) -> int:
    max_len = 0
    for idx in range(len(dataset)):
        ts_container = dataset[idx]["time_series"]
        ts_raw = ts_container[0] if isinstance(ts_container, list) else ts_container
        ts = torch.as_tensor(ts_raw).flatten()
        max_len = max(max_len, int(ts.numel()))
    if max_len <= 0:
        raise RuntimeError("Unable to infer a valid context length from the dataset.")
    return max_len


def enforce_strict_fewshot_protocol(args):
    if args.protocol != "fewshot":
        return args

    args.epochs = STRICT_FEWSHOT_EPOCHS
    return args


def build_model(args, num_classes: int, context_length: int, device: str, rank: int):
    if rank == 0:
        print("🔧 Building PatchTST classifier...")
        if args.patchtst_model_id:
            print(f"   pretrained_source: {args.patchtst_model_id}")
            print(f"   reset_head: {args.reset_head}")
        else:
            print("   pretrained_source: none (random init)")

    model = PatchTSTClassifierAdapter.build_model(
        num_classes=num_classes,
        context_length=context_length,
        device=device,
        patchtst_model_id=args.patchtst_model_id,
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        use_cls_token=args.use_cls_token,
        pooling_type=args.pooling_type,
        reset_head=args.reset_head,
    )

    if rank == 0:
        print(f"   head_was_reset: {model.head_was_reset}")
        print(f"   use_cls_token: {model.config.use_cls_token}")
        print(f"   pooling_type: {model.config.pooling_type}")
        print(f"   patch_length: {model.config.patch_length}")
        print(f"   stride: {model.config.stride}")
        print(f"   d_model: {model.config.d_model}")

    return model


def build_optimizer_scheduler(
    model,
    train_loader: DataLoader,
    args,
    num_epochs: int,
    grad_acc_steps: int,
    include_backbone: bool,
):
    underlying = get_model(model)
    param_groups = underlying.get_param_groups(
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        include_backbone=include_backbone,
    )
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, grad_acc_steps)))
    total_steps = max(1, num_epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler, total_steps, warmup_steps


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    *,
    context_length: int,
    grad_clip: float,
    epoch_idx: int,
    epoch_total: int,
    gradient_accumulation_steps: int,
    device: str,
    pad_mode: str,
    rank: int,
    phase_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(
        train_loader,
        desc=f"{phase_name} Epoch {epoch_idx}/{epoch_total}",
        disable=(rank != 0),
    )
    for step, batch in enumerate(pbar):
        model_inputs = prepare_patchtst_classification_batch(
            batch,
            context_length=context_length,
            device=device,
            pad_mode=pad_mode,
        )
        outputs = model(return_dict=True, **model_inputs)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

        if rank == 0:
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

    if num_batches % gradient_accumulation_steps != 0:
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    *,
    context_length: int,
    device: str,
    pad_mode: str,
    rank: int = 0,
) -> Dict[str, Any]:
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Testing", disable=(rank != 0)):
        model_inputs = prepare_patchtst_classification_batch(
            batch,
            context_length=context_length,
            device=device,
            pad_mode=pad_mode,
        )
        outputs = model(return_dict=True, **model_inputs)
        predictions = torch.argmax(outputs.prediction_logits, dim=-1)

        total_loss += float(outputs.loss.item())
        num_batches += 1

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(model_inputs["target_values"].cpu().tolist())

    avg_loss = total_loss / max(num_batches, 1)
    correct = sum(int(p == y) for p, y in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def train_phase(
    model,
    train_loader: DataLoader,
    train_sampler,
    *,
    args,
    phase_name: str,
    phase_epochs: int,
    include_backbone: bool,
    grad_acc_steps: int,
    epoch_offset: int,
    context_length: int,
    device: str,
    num_classes: int,
    label_mapping: Dict[str, Any],
    run_dir: str,
    rank: int,
    resume: bool,
) -> Dict[str, Any]:
    if phase_epochs <= 0:
        return {"losses": [], "last_loss": None, "total_steps": 0, "warmup_steps": 0}

    underlying = get_model(model)
    if include_backbone:
        underlying.unfreeze_backbone()
    else:
        underlying.freeze_backbone()

    optimizer, scheduler, total_steps, warmup_steps = build_optimizer_scheduler(
        model=model,
        train_loader=train_loader,
        args=args,
        num_epochs=phase_epochs,
        grad_acc_steps=grad_acc_steps,
        include_backbone=include_backbone,
    )

    if rank == 0:
        print(
            f"   {phase_name}: epochs={phase_epochs}, include_backbone={include_backbone}, "
            f"steps={total_steps}, warmup={warmup_steps}"
        )

    ckpt_path = os.path.join(run_dir, f"{phase_name}.pt")
    losses: List[float] = []
    start_local_epoch = 1
    if resume and os.path.exists(ckpt_path):
        checkpoint = underlying.load_checkpoint(
            checkpoint_path=ckpt_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        checkpoint_phase = checkpoint.get("phase")
        if checkpoint_phase is not None and checkpoint_phase != phase_name:
            raise ValueError(f"Checkpoint phase mismatch: expected {phase_name}, got {checkpoint_phase}")

        saved_phase_epochs = checkpoint.get("phase_epochs")
        if saved_phase_epochs is not None and int(saved_phase_epochs) != phase_epochs:
            raise ValueError(
                f"Phase epoch mismatch for {phase_name}: checkpoint has {saved_phase_epochs}, expected {phase_epochs}"
            )

        losses = [float(item) for item in checkpoint.get("loss_history", [])]
        completed_epoch = int(checkpoint.get("epoch", 0) or 0)
        start_local_epoch = completed_epoch + 1
        if rank == 0:
            print(
                f"   {phase_name}: resume from epoch {completed_epoch}/{phase_epochs}"
            )
        if completed_epoch >= phase_epochs:
            return {
                "losses": losses,
                "last_loss": checkpoint.get("last_loss", losses[-1] if losses else None),
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
            }

    for local_epoch in range(start_local_epoch, phase_epochs + 1):
        global_epoch = epoch_offset + local_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            context_length=context_length,
            grad_clip=args.grad_clip,
            epoch_idx=local_epoch,
            epoch_total=phase_epochs,
            gradient_accumulation_steps=grad_acc_steps,
            device=device,
            pad_mode=args.pad_mode,
            rank=rank,
            phase_name=phase_name,
        )
        losses.append(train_loss)

        if rank == 0:
            print(f"   {phase_name} epoch {local_epoch}/{phase_epochs}: train_loss={train_loss:.6f}")
        underlying.save_checkpoint(
            save_path=ckpt_path,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=local_epoch,
            phase=phase_name,
            num_classes=num_classes,
            context_length=context_length,
            label_mapping=label_mapping,
            args=vars(args),
            extra_state={
                "phase_epochs": phase_epochs,
                "loss_history": losses,
                "last_loss": train_loss,
            },
            rank=rank,
        )

    return {
        "losses": losses,
        "last_loss": losses[-1] if losses else None,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.stdev(values))


def aggregate_shot_results(shot: ShotType, run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [float(r["test_accuracy"]) for r in run_metrics]
    losses = [float(r["test_loss"]) for r in run_metrics]
    support_sizes = [int(r["support_size"]) for r in run_metrics]

    acc_mean, acc_std = mean_std(accs)
    loss_mean, loss_std = mean_std(losses)
    support_mean, support_std = mean_std([float(x) for x in support_sizes])

    return {
        "shot": shot_to_name(shot),
        "num_runs": len(run_metrics),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "support_size_mean": support_mean,
        "support_size_std": support_std,
        "any_shortage_in_shot": any(bool(r["any_shortage"]) for r in run_metrics),
        "run_metrics": run_metrics,
    }


def save_shot_summary_csv(save_path: str, shot_summaries: List[Dict[str, Any]]):
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
    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for item in shot_summaries:
            writer.writerow({key: item[key] for key in columns})


def run_single_experiment(
    *,
    args,
    shot: ShotType,
    shot_idx: int,
    run_id: int,
    run_seed: int,
    base_save_dir: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    label_to_indices: Dict[int, List[int]],
    test_label_to_indices: Dict[int, List[int]],
    num_classes: int,
    label_mapping: Dict[str, Any],
    context_length: int,
    local_rank: int,
    world_size: int,
    rank: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    shot_name = shot_to_name(shot)
    run_dir = os.path.join(base_save_dir, f"shot_{shot_name}", f"run_{run_id:02d}")
    run_metrics_path = os.path.join(run_dir, "run_metrics.json")
    support_info_path = os.path.join(run_dir, "fewshot_indices.json")
    phase1_ckpt_path = os.path.join(run_dir, "phase1_warmup.pt")
    phase2_ckpt_path = os.path.join(run_dir, "phase2_last.pt")

    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    completed_run_exists_rank0 = (
        args.resume
        and rank == 0
        and os.path.exists(run_metrics_path)
        and (args.cleanup_checkpoints or os.path.exists(phase2_ckpt_path))
    )
    completed_run_exists = broadcast_object_from_rank0(
        completed_run_exists_rank0 if rank == 0 else None,
        world_size,
        rank,
    )
    if completed_run_exists:
        cached_metrics = None
        if rank == 0:
            print(f"[shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
            with open(run_metrics_path, "r", encoding="utf-8") as f:
                cached_metrics = json.load(f)
        if world_size > 1:
            dist.barrier()
        return cached_metrics

    set_seed(run_seed)

    support_info_rank0 = None
    if rank == 0:
        if args.resume and os.path.exists(support_info_path):
            with open(support_info_path, "r", encoding="utf-8") as f:
                support_info_rank0 = json.load(f)
        else:
            support_info_rank0 = sample_support_info(
                label_to_indices,
                shot,
                run_seed,
                way=args.way,
            )
            with open(support_info_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": args.dataset,
                        "way": support_info_rank0["way"],
                        "shot": shot_name,
                        "run_id": run_id,
                        "seed": run_seed,
                        "selected_class_ids": support_info_rank0["selected_class_ids"],
                        "selected_indices": support_info_rank0["selected_indices"],
                        "selected_by_class": support_info_rank0["selected_by_class"],
                        "k_eff_per_class": support_info_rank0["k_eff_per_class"],
                        "class_train_counts": support_info_rank0["class_train_counts"],
                        "classes_with_shortage": support_info_rank0["classes_with_shortage"],
                    },
                    f,
                    indent=2,
                )
    support_info = broadcast_object_from_rank0(support_info_rank0, world_size, rank)

    support_indices = support_info["selected_indices"]
    support_dataset = Subset(train_dataset, support_indices)
    query_indices = filter_indices_by_class_ids(
        test_label_to_indices,
        support_info["selected_class_ids"],
    )
    query_dataset = Subset(test_dataset, query_indices)
    train_batch_size, grad_acc_steps = compute_fewshot_train_hparams(
        args=args,
        shot=shot,
        support_size=len(support_indices),
    )

    if rank == 0:
        print("-" * 80)
        print(
            f"[shot={shot_name} run={run_id}] seed={run_seed}, "
            f"way={support_info['way']}, support={len(support_indices)}, "
            f"query={len(query_indices)}, batch={train_batch_size}, grad_acc={grad_acc_steps}"
        )
        print(f"   selected classes: {support_info['selected_class_ids']}")
        if support_info["any_shortage"]:
            print(f"   classes with n<K use-all behavior: {support_info['classes_with_shortage']}")

    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            support_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    train_loader = DataLoader(
        support_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )

    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            query_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
            num_workers=0,
        )

    model = build_model(
        args=args,
        num_classes=num_classes,
        context_length=context_length,
        device=device,
        rank=rank,
    )

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    phase1_epochs, phase2_epochs = resolve_phase_epochs(args.epochs, args.phase1_epochs)

    if rank == 0:
        print(
            f"   phase split: phase1={phase1_epochs} (head warm-up), "
            f"phase2={phase2_epochs} (joint fine-tune)"
        )
        if args.model_select_metric != "last":
            print("   note: model_select_metric is forced by design to phase2 last checkpoint.")

    phase1_stats = train_phase(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        args=args,
        phase_name="phase1_warmup",
        phase_epochs=phase1_epochs,
        include_backbone=False,
        grad_acc_steps=grad_acc_steps,
        epoch_offset=0,
        context_length=context_length,
        device=device,
        num_classes=num_classes,
        label_mapping=label_mapping,
        run_dir=run_dir,
        rank=rank,
        resume=args.resume,
    )

    phase2_stats = train_phase(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        args=args,
        phase_name="phase2_last",
        phase_epochs=phase2_epochs,
        include_backbone=True,
        grad_acc_steps=grad_acc_steps,
        epoch_offset=phase1_epochs,
        context_length=context_length,
        device=device,
        num_classes=num_classes,
        label_mapping=label_mapping,
        run_dir=run_dir,
        rank=rank,
        resume=args.resume,
    )

    if world_size > 1:
        dist.barrier()

    run_metrics = None
    if rank == 0:
        underlying = get_model(model)
        underlying.load_checkpoint(
            checkpoint_path=phase2_ckpt_path,
            device=device,
        )
        test_results = evaluate(
            model=underlying,
            data_loader=test_loader,
            context_length=context_length,
            device=device,
            pad_mode=args.pad_mode,
            rank=rank,
        )

        run_metrics = {
            "dataset": args.dataset,
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
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_acc_steps,
            "phase1_last_train_loss": phase1_stats["last_loss"],
            "phase2_last_train_loss": phase2_stats["last_loss"],
            "test_loss": test_results["loss"],
            "test_accuracy": test_results["accuracy"],
            "context_length": context_length,
            "head_was_reset": underlying.head_was_reset,
            "pretrained_source": underlying.pretrained_source,
            "model_checkpoint": "phase2_last.pt",
        }

        with open(run_metrics_path, "w", encoding="utf-8") as f:
            json.dump(run_metrics, f, indent=2)

        with open(os.path.join(run_dir, "test_predictions.json"), "w") as f:
            json.dump(
                {
                    "predictions": test_results["predictions"],
                    "labels": test_results["labels"],
                },
                f,
                indent=2,
            )

        if args.cleanup_checkpoints:
            cleanup_checkpoint_files(
                [phase1_ckpt_path, phase2_ckpt_path],
                rank=rank,
            )

        print(
            f"   result: test_acc={test_results['accuracy']:.4f}, "
            f"test_loss={test_results['loss']:.4f}"
        )

    if world_size > 1:
        dist.barrier()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_metrics


def main():
    args = normalize_protocol_args(parse_args())
    args = enforce_strict_fewshot_protocol(args)
    local_rank, world_size, rank = setup_distributed()

    try:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        if args.num_runs < 1:
            raise ValueError("--num_runs must be >= 1")
        if args.way is not None and args.way < 1:
            raise ValueError("--way must be >= 1 when provided")

        if args.protocol == "fewshot":
            shots: List[ShotType] = parse_shots(args.shots)
            num_runs = max(1, args.num_runs)
        else:
            shots = ["full"]
            num_runs = 1

        if world_size > 1:
            device = f"cuda:{local_rank}"
        elif args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        set_seed(args.seed)

        save_root = os.path.join(args.save_dir, args.dataset)
        if rank == 0:
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            print("=" * 80)
            print("PatchTST: Strict Few-shot Supervised Protocol")
            print("=" * 80)
            print(f"time: {datetime.datetime.now()}")
            print(f"dataset: {args.dataset}")
            print(f"protocol: {args.protocol}")
            print(f"way: {args.way if args.way is not None else 'all'}")
            print(f"shots: {[shot_to_name(s) for s in shots]}")
            print(f"num_runs: {num_runs}")
            print(f"pretrained_source: {args.patchtst_model_id}")
            print(f"strict few-shot epochs: {args.epochs if args.protocol == 'fewshot' else 'n/a'}")
            print(f"pad_mode: {args.pad_mode}")
            print(f"ddp world_size: {world_size}")
            print("=" * 80)

        if world_size > 1:
            dist.barrier()

        dataset_eos = "<eos>"
        train_dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN=dataset_eos,
            dataset_name=args.dataset,
            raw_data_path=args.data_path,
        )
        test_dataset = UCRClassificationDataset(
            split="test",
            EOS_TOKEN=dataset_eos,
            dataset_name=args.dataset,
            raw_data_path=args.data_path,
        )

        num_classes = UCRClassificationDataset.get_num_classes()
        label_mapping = UCRClassificationDataset.get_label_mapping()
        label_to_indices = build_label_to_indices(train_dataset)
        test_label_to_indices = build_label_to_indices(test_dataset)
        context_length = args.context_length or infer_context_length(train_dataset)

        if args.way is not None and args.way > num_classes:
            raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes})")

        if rank == 0:
            class_size_brief = {k: len(v) for k, v in label_to_indices.items()}
            print(f"num_classes: {num_classes}")
            print(f"train_size: {len(train_dataset)} | test_size: {len(test_dataset)}")
            print(f"class_train_counts: {class_size_brief}")
            print(f"context_length: {context_length}")

        shot_summaries = []
        for shot_idx, shot in enumerate(shots):
            shot_name = shot_to_name(shot)
            shot_run_metrics = []

            for run_id in range(1, num_runs + 1):
                run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
                run_metrics = run_single_experiment(
                    args=args,
                    shot=shot,
                    shot_idx=shot_idx,
                    run_id=run_id,
                    run_seed=run_seed,
                    base_save_dir=save_root,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                    label_to_indices=label_to_indices,
                    test_label_to_indices=test_label_to_indices,
                    num_classes=num_classes,
                    label_mapping=label_mapping,
                    context_length=context_length,
                    local_rank=local_rank,
                    world_size=world_size,
                    rank=rank,
                    device=device,
                )
                if rank == 0 and run_metrics is not None:
                    shot_run_metrics.append(run_metrics)

            if rank == 0:
                shot_summary = aggregate_shot_results(shot=shot, run_metrics=shot_run_metrics)
                shot_summaries.append(shot_summary)

                shot_dir = os.path.join(save_root, f"shot_{shot_name}")
                os.makedirs(shot_dir, exist_ok=True)
                with open(os.path.join(shot_dir, "shot_summary.json"), "w") as f:
                    json.dump(shot_summary, f, indent=2)

                print(
                    f"[shot={shot_name}] "
                    f"acc={shot_summary['accuracy_mean']:.4f}±{shot_summary['accuracy_std']:.4f}"
                )

            if world_size > 1:
                dist.barrier()

        if rank == 0:
            overall_summary = {
                "dataset": args.dataset,
                "protocol": args.protocol,
                "way": args.way if args.way is not None else num_classes,
                "num_classes": num_classes,
                "shots": [shot_to_name(s) for s in shots],
                "num_runs": num_runs,
                "timestamp": str(datetime.datetime.now()),
                "shot_summaries": shot_summaries,
            }

            with open(os.path.join(save_root, "fewshot_summary.json"), "w") as f:
                json.dump(overall_summary, f, indent=2)

            save_shot_summary_csv(
                save_path=os.path.join(save_root, "fewshot_summary.csv"),
                shot_summaries=shot_summaries,
            )

            if args.protocol == "full" and shot_summaries:
                final_results = {
                    "dataset": args.dataset,
                    "protocol": args.protocol,
                    "test_loss": shot_summaries[0]["loss_mean"],
                    "test_accuracy": shot_summaries[0]["accuracy_mean"],
                    "epochs_trained": args.epochs,
                }
                with open(os.path.join(save_root, "final_results.json"), "w", encoding="utf-8") as f:
                    json.dump(final_results, f, indent=2)

            print("=" * 80)
            print(f"Done. Results saved to: {save_root}")
            print("=" * 80)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
