#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Experiment A (few-shot supervised):
LLM as deep sequence aggregator with strict few-shot protocol.

Key defaults requested by design lock:
- model_select_metric = last
- Phase1 is warm-up only; final checkpoint is always Phase2 last
- few-shot batch mode defaults to full-batch priority
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

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMClassifier import OpenTSLMClassifier
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

ShotType = Union[int, Literal["full"]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment A: strict few-shot supervised classification protocol"
    )

    # Core behavior
    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot", "full"])
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)
    parser.add_argument("--model_select_metric", type=str, default="last", choices=["last", "train_loss"])
    parser.add_argument("--fewshot_batch_mode", type=str, default="full", choices=["full", "manual"])

    # Phase setup
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--phase1_epochs", type=int, default=5)

    # Must-specify switches (kept for compatibility with existing scripts)
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA in phase2")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    # Data
    parser.add_argument("--dataset", type=str, default="CricketZ")
    parser.add_argument("--data_path", type=str, default="./data")

    # Model
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="tslanet",
        choices=["transformer_cnn", "tslanet"],
    )
    parser.add_argument("--encoder_pretrained", type=str, default=None)
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_encoder", type=float, default=2e-4)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--lr_classifier", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    # Preprocessing and augmentation
    parser.add_argument("--pad_mode", type=str, default="last", choices=["last", "repeat", "zero"])
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--aug_jitter_std", type=float, default=0.02)
    parser.add_argument("--aug_scaling_min", type=float, default=0.9)
    parser.add_argument("--aug_scaling_max", type=float, default=1.1)
    parser.add_argument("--aug_time_mask_ratio", type=float, default=0.05)
    parser.add_argument("--aug_time_mask_prob", type=float, default=0.3)
    parser.add_argument("--aug_freq_dropout_ratio", type=float, default=0.05)
    parser.add_argument("--aug_freq_dropout_prob", type=float, default=0.2)

    # System and logging
    parser.add_argument("--save_dir", type=str, default="results/experiment_a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def setup_distributed() -> Tuple[int, int, int]:
    """Init distributed environment for torchrun."""
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


def get_model(model):
    return model.module if hasattr(model, "module") else model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # de-duplicate while preserving order
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


def make_collate_fn(args, is_train: bool):
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=PATCH_SIZE,
            normalize=True,
            normalize_eps=1e-5,
            pad_mode=args.pad_mode,
            augment=is_train and args.enable_augmentation,
            jitter_std=args.aug_jitter_std,
            scaling_range=(args.aug_scaling_min, args.aug_scaling_max),
            time_mask_ratio=args.aug_time_mask_ratio,
            time_mask_prob=args.aug_time_mask_prob,
            freq_dropout_ratio=args.aug_freq_dropout_ratio,
            freq_dropout_prob=args.aug_freq_dropout_prob,
            enable_freq_dropout=(args.encoder_type == "tslanet"),
        )

    return collate_fn


def build_label_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        label_to_indices[int(dataset[idx]["int_label"])].append(idx)
    return dict(label_to_indices)


def sample_support_info(
    label_to_indices: Dict[int, List[int]],
    shot: ShotType,
    seed: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    selected_indices: List[int] = []
    selected_by_class: Dict[str, List[int]] = {}
    class_train_counts: Dict[str, int] = {}
    k_eff_per_class: Dict[str, int] = {}
    classes_with_shortage: List[int] = []

    for class_id in sorted(label_to_indices.keys()):
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
    support_size = len(selected_indices)
    any_shortage = len(classes_with_shortage) > 0

    return {
        "selected_indices": selected_indices,
        "selected_by_class": selected_by_class,
        "class_train_counts": class_train_counts,
        "k_eff_per_class": k_eff_per_class,
        "classes_with_shortage": classes_with_shortage,
        "any_shortage": any_shortage,
        "support_size": support_size,
    }


def broadcast_object_from_rank0(obj, world_size: int, rank: int):
    if world_size == 1:
        return obj
    holder = [obj if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def resolve_phase_epochs(total_epochs: int, phase1_epochs: int, use_lora: bool) -> Tuple[int, int]:
    if total_epochs <= 0:
        return 0, 0

    if not use_lora:
        return 0, total_epochs

    # Force phase2 >= 1
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
        # Full-batch priority for few-shot numeric K
        return max(1, support_size), 1
    return args.batch_size, args.gradient_accumulation_steps


def build_model(args, num_classes: int, device: str, rank: int):
    tslanet_config = {"patch_size": PATCH_SIZE}
    model = OpenTSLMClassifier(
        num_classes=num_classes,
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
    )

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 Encoder parameters frozen.")

    if args.use_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    return model


def build_optimizer_scheduler(
    model,
    train_loader: DataLoader,
    args,
    num_epochs: int,
    grad_acc_steps: int,
    include_lora: bool,
):
    underlying = get_model(model)
    param_groups = []

    if not args.freeze_encoder:
        encoder_params = [p for p in underlying.encoder.parameters() if p.requires_grad]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_encoder})

    projector_params = [
        p for p in list(underlying.projector.parameters()) + list(underlying.projector_out_norm.parameters())
        if p.requires_grad
    ]
    if projector_params:
        param_groups.append({"params": projector_params, "lr": args.lr_projector})

    classifier_params = [
        underlying.ans_token,
        *list(underlying.classifier_head.parameters()),
        *list(underlying.ans_norm.parameters()),
    ]
    classifier_params = [p for p in classifier_params if p.requires_grad]
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": args.lr_classifier})

    if include_lora and args.use_lora:
        lora_params = [p for p in underlying.get_lora_parameters() if p.requires_grad]
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer.")

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
    grad_clip: float,
    epoch_idx: int,
    epoch_total: int,
    gradient_accumulation_steps: int,
    rank: int,
    phase_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    desc = f"{phase_name} Epoch {epoch_idx}/{epoch_total}"
    pbar = tqdm(train_loader, desc=desc, disable=(rank != 0))
    for step, batch in enumerate(pbar):
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
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
def evaluate(model, data_loader: DataLoader, rank: int = 0) -> Dict[str, Any]:
    model.eval()
    underlying_model = get_model(model)

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(data_loader, desc="Testing", disable=(rank != 0)):
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1

        predictions = underlying_model.predict(batch)

        for i, sample in enumerate(batch):
            all_predictions.append(predictions[i].item())
            all_labels.append(sample["int_label"])

    avg_loss = total_loss / max(num_batches, 1)
    correct = sum(int(p == y) for p, y in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: Optional[float],
    save_path: str,
    args,
    phase: str,
    rank: int = 0,
):
    if rank != 0:
        return

    underlying_model = get_model(model)
    checkpoint = {
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "projector_out_norm_state": underlying_model.projector_out_norm.state_dict(),
        "classifier_head_state": underlying_model.classifier_head.state_dict(),
        "ans_norm_state": underlying_model.ans_norm.state_dict(),
        "ans_token": underlying_model.ans_token.data.cpu(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "phase": phase,
        "num_classes": underlying_model.num_classes,
        "args": vars(args),
    }
    underlying_model.save_lora_state_to_checkpoint(checkpoint)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: str,
    optimizer=None,
    scheduler=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    underlying_model = get_model(model)

    underlying_model.encoder.load_state_dict(checkpoint["encoder_state"])
    underlying_model.projector.load_state_dict(checkpoint["projector_state"])

    if "projector_out_norm_state" in checkpoint:
        underlying_model.projector_out_norm.load_state_dict(checkpoint["projector_out_norm_state"])
    if "ans_norm_state" in checkpoint:
        underlying_model.ans_norm.load_state_dict(checkpoint["ans_norm_state"])

    underlying_model.classifier_head.load_state_dict(checkpoint["classifier_head_state"])
    underlying_model.ans_token.data.copy_(checkpoint["ans_token"].to(device))

    underlying_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def train_phase(
    model,
    train_loader: DataLoader,
    train_sampler,
    args,
    phase_name: str,
    phase_epochs: int,
    include_lora: bool,
    grad_acc_steps: int,
    epoch_offset: int,
    rank: int,
    ckpt_path: Optional[str] = None,
) -> Dict[str, Any]:
    if phase_epochs <= 0:
        return {"losses": [], "last_loss": None, "total_steps": 0, "warmup_steps": 0}

    underlying_model = get_model(model)
    if args.use_lora:
        underlying_model.set_lora_trainable(include_lora)

    optimizer, scheduler, total_steps, warmup_steps = build_optimizer_scheduler(
        model=model,
        train_loader=train_loader,
        args=args,
        num_epochs=phase_epochs,
        grad_acc_steps=grad_acc_steps,
        include_lora=include_lora,
    )

    if rank == 0:
        print(
            f"   {phase_name}: epochs={phase_epochs}, include_lora={include_lora}, "
            f"steps={total_steps}, warmup={warmup_steps}"
        )

    losses: List[float] = []
    for local_epoch in range(1, phase_epochs + 1):
        global_epoch = epoch_offset + local_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(global_epoch)

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            epoch_idx=local_epoch,
            epoch_total=phase_epochs,
            gradient_accumulation_steps=grad_acc_steps,
            rank=rank,
            phase_name=phase_name,
        )
        losses.append(train_loss)

        if rank == 0:
            print(f"   {phase_name} epoch {local_epoch}/{phase_epochs}: train_loss={train_loss:.6f}")

    last_loss = losses[-1] if losses else None
    if ckpt_path is not None:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=phase_epochs,
            train_loss=last_loss,
            save_path=ckpt_path,
            args=args,
            phase=phase_name,
            rank=rank,
        )

    return {
        "losses": losses,
        "last_loss": last_loss,
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
    args,
    shot: ShotType,
    shot_idx: int,
    run_id: int,
    run_seed: int,
    base_save_dir: str,
    train_dataset: Dataset,
    test_dataset: Dataset,
    label_to_indices: Dict[int, List[int]],
    num_classes: int,
    local_rank: int,
    world_size: int,
    rank: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    shot_name = shot_to_name(shot)
    run_dir = os.path.join(base_save_dir, f"shot_{shot_name}", f"run_{run_id:02d}")

    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    set_seed(run_seed)

    support_info_rank0 = None
    if rank == 0:
        support_info_rank0 = sample_support_info(label_to_indices, shot, run_seed)
    support_info = broadcast_object_from_rank0(support_info_rank0, world_size, rank)

    support_indices = support_info["selected_indices"]
    support_dataset = Subset(train_dataset, support_indices)

    # Few-shot default: full-batch for numeric K. "full" shot keeps regular mini-batch.
    train_batch_size, grad_acc_steps = compute_fewshot_train_hparams(
        args=args,
        shot=shot,
        support_size=len(support_indices),
    )

    if rank == 0:
        print("-" * 80)
        print(
            f"[shot={shot_name} run={run_id}] seed={run_seed}, support={len(support_indices)}, "
            f"batch={train_batch_size}, grad_acc={grad_acc_steps}"
        )
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
        collate_fn=make_collate_fn(args, is_train=True),
    )

    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(args, is_train=False),
        )

    model = build_model(args=args, num_classes=num_classes, device=device, rank=rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    phase1_epochs, phase2_epochs = resolve_phase_epochs(
        total_epochs=args.epochs,
        phase1_epochs=args.phase1_epochs,
        use_lora=args.use_lora,
    )

    if rank == 0:
        print(
            f"   phase split: phase1={phase1_epochs} (warm-up), "
            f"phase2={phase2_epochs} (joint)"
        )
        if args.model_select_metric != "last":
            print("   note: model_select_metric is forced by design to phase2 last checkpoint.")

    phase2_ckpt_path = os.path.join(run_dir, "phase2_last.pt")

    # Phase1: warm-up only, no final model selection.
    phase1_stats = train_phase(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        args=args,
        phase_name="phase1_warmup",
        phase_epochs=phase1_epochs,
        include_lora=False,
        grad_acc_steps=grad_acc_steps,
        epoch_offset=0,
        rank=rank,
        ckpt_path=None,
    )

    # Phase2: final model is always the last checkpoint.
    phase2_stats = train_phase(
        model=model,
        train_loader=train_loader,
        train_sampler=train_sampler,
        args=args,
        phase_name="phase2_joint",
        phase_epochs=phase2_epochs,
        include_lora=args.use_lora,
        grad_acc_steps=grad_acc_steps,
        epoch_offset=phase1_epochs,
        rank=rank,
        ckpt_path=phase2_ckpt_path,
    )

    if world_size > 1:
        dist.barrier()

    run_metrics = None
    if rank == 0:
        load_checkpoint(model=model, checkpoint_path=phase2_ckpt_path, device=device)
        test_results = evaluate(model=model, data_loader=test_loader, rank=rank)

        run_metrics = {
            "dataset": args.dataset,
            "protocol": args.protocol,
            "shot": shot_name,
            "run_id": run_id,
            "shot_index": shot_idx,
            "seed": run_seed,
            "support_size": len(support_indices),
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
            "model_checkpoint": "phase2_last.pt",
        }

        with open(os.path.join(run_dir, "run_metrics.json"), "w") as f:
            json.dump(run_metrics, f, indent=2)

        with open(os.path.join(run_dir, "fewshot_indices.json"), "w") as f:
            json.dump(
                {
                    "dataset": args.dataset,
                    "shot": shot_name,
                    "run_id": run_id,
                    "seed": run_seed,
                    "selected_indices": support_info["selected_indices"],
                    "selected_by_class": support_info["selected_by_class"],
                    "k_eff_per_class": support_info["k_eff_per_class"],
                    "class_train_counts": support_info["class_train_counts"],
                    "classes_with_shortage": support_info["classes_with_shortage"],
                },
                f,
                indent=2,
            )

        with open(os.path.join(run_dir, "test_predictions.json"), "w") as f:
            json.dump(
                {
                    "predictions": test_results["predictions"],
                    "labels": test_results["labels"],
                },
                f,
                indent=2,
            )

        print(
            f"   result: test_acc={test_results['accuracy']:.4f}, "
            f"test_loss={test_results['loss']:.4f}"
        )

    if world_size > 1:
        dist.barrier()

    # Cleanup memory per run.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_metrics


def main():
    args = parse_args()
    local_rank, world_size, rank = setup_distributed()

    try:
        if args.epochs < 1:
            raise ValueError("--epochs must be >= 1")
        if args.num_runs < 1:
            raise ValueError("--num_runs must be >= 1")
        if args.aug_scaling_min > args.aug_scaling_max:
            raise ValueError("--aug_scaling_min must be <= --aug_scaling_max")

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

            config_path = os.path.join(save_root, "config.json")
            with open(config_path, "w") as f:
                json.dump(vars(args), f, indent=2)

            print("=" * 80)
            print("Experiment A: Strict Few-shot Supervised Protocol")
            print("=" * 80)
            print(f"time: {datetime.datetime.now()}")
            print(f"dataset: {args.dataset}")
            print(f"protocol: {args.protocol}")
            print(f"shots: {[shot_to_name(s) for s in shots]}")
            print(f"num_runs: {num_runs}")
            print(f"encoder: {args.encoder_type}")
            print(f"use_lora: {args.use_lora}")
            print(f"pad_mode: {args.pad_mode}")
            print(f"augmentation: {args.enable_augmentation}")
            print(f"ddp world_size: {world_size}")
            print("=" * 80)

        if world_size > 1:
            dist.barrier()

        # Build base datasets once. For this classifier pipeline, EOS is not used by loss.
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
        label_to_indices = build_label_to_indices(train_dataset)

        if rank == 0:
            class_size_brief = {k: len(v) for k, v in label_to_indices.items()}
            print(f"num_classes: {num_classes}")
            print(f"train_size: {len(train_dataset)} | test_size: {len(test_dataset)}")
            print(f"class_train_counts: {class_size_brief}")

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
                    num_classes=num_classes,
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

            print("=" * 80)
            print(f"Done. Results saved to: {save_root}")
            print("=" * 80)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
