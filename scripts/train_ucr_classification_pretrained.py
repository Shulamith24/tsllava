#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M2: UCR single-dataset classification with pretrained SP models under a strict few-shot protocol.

This script aligns its protocol and training flow with Experiment A:
- strict support-set sampling per shot/run
- Phase1 warm-up only; final checkpoint is always Phase2 last
- few-shot full-batch priority for numeric K by default
- if class count < K, use all available samples from that class

The model-specific behavior from the original pretrained script is preserved:
- Hugging Face or local checkpoint loading
- generative classification with class tokens: <c0>, <c1>, ...
- constrained decoding during evaluation
- class-token embedding / lm_head checkpoint persistence
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
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    get_linear_schedule_with_warmup,
)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

ShotType = Union[int, Literal["full"]]


def parse_int_list(value: Optional[Union[str, List[int]]]) -> Optional[List[int]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value

    items = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        items.append(int(stripped))
    return items or None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="M2: strict few-shot UCR classification with pretrained SP models"
    )

    # Core protocol behavior
    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot", "full"])
    parser.add_argument("--shots", type=str, default="1,2,5,10,full")
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)
    parser.add_argument(
        "--model_select_metric",
        type=str,
        default="last",
        choices=["last", "train_loss"],
        help="Kept for compatibility; final checkpoint is always Phase2 last.",
    )
    parser.add_argument("--fewshot_batch_mode", type=str, default="full", choices=["full", "manual"])

    # Phase setup
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--phase1_epochs", type=int, default=5)

    # Must-specify switches / compatibility flags
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder parameters")

    # Data
    parser.add_argument("--dataset", type=str, default="CricketZ", help="UCR dataset name")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR data root")

    # Model loading
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="Pretrained Hugging Face repo_id, e.g. OpenTSLM/llama-3.2-1b-m4-sp",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default=None,
        help="Local checkpoint path, e.g. results/curriculum_pretrain/.../best_model.pt",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer_cnn",
        choices=["transformer_cnn", "tslanet", "newts_dual_branch"],
        help="Encoder type (required when using local checkpoints or training from scratch)",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base LLM id (used for local checkpoints or training from scratch)",
    )
    parser.add_argument(
        "--tslanet_patch_size",
        type=int,
        default=8,
        help="TSLANet patch_size (when encoder_type=tslanet)",
    )
    parser.add_argument(
        "--random_init_llm",
        action="store_true",
        help="Randomly initialize the LLM backbone for ablations",
    )

    # NewTS dual-branch encoder
    parser.add_argument("--branch_mode", type=str, default="both", choices=["both", "ts_only", "vision_only"])
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument(
        "--vit_feature_mode",
        type=str,
        default="single",
        choices=["last", "single", "scalar_mix"],
    )
    parser.add_argument("--vit_layer_idx", type=int, default=4)
    parser.add_argument(
        "--vit_mix_layers",
        type=str,
        default=None,
        help="Comma-separated 1-based layer indices used when vit_feature_mode=scalar_mix",
    )
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    parser.add_argument("--vit_num_hidden_layers", type=int, default=None)
    parser.add_argument(
        "--vit_truncate_to_feature_layer",
        dest="vit_truncate_to_feature_layer",
        action="store_true",
        help="Truncate the loaded vision backbone to the minimum depth needed by the selected feature layer",
    )
    parser.add_argument(
        "--no_vit_truncate_to_feature_layer",
        dest="vit_truncate_to_feature_layer",
        action="store_false",
        help="Disable feature-layer-based truncation for the vision backbone",
    )
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "linear"]) #分支内投影
    parser.add_argument("--projector_dropout", type=float, default=0.1) #分支内投影的dropout
    parser.add_argument("--freeze_ts_backbone", action="store_true")
    parser.add_argument("--freeze_vision_backbone",dest="freeze_vision_backbone",action="store_true",help="Freeze the vision backbone parameters",)
    parser.add_argument(
        "--no_freeze_vision_backbone",
        dest="freeze_vision_backbone",
        action="store_false",
        help="Leave the vision backbone trainable",
    )
    parser.set_defaults(
        vit_truncate_to_feature_layer=True,
        freeze_vision_backbone=True,
    )

    # LoRA
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Optimization
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_encoder", type=float, default=2e-4)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
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

    # Generation / eval
    parser.add_argument("--max_new_tokens", type=int, default=2, help="Class token + EOS")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=5,
        help="Unused in the few-shot protocol; kept for backward compatibility.",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Unused in the few-shot protocol; kept for backward compatibility.",
    )

    # System and logging
    parser.add_argument("--save_dir", type=str, default="results/m2_ucr_pretrained_fewshot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args(argv)
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    return args


def hydrate_args_from_local_checkpoint_metadata(args):
    args.local_checkpoint_model_config = None
    if not args.local_checkpoint:
        return args

    checkpoint = torch.load(args.local_checkpoint, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") or {}
    args.local_checkpoint_model_config = model_config

    llm_id = model_config.get("llm_id")
    if llm_id:
        args.llm_id = llm_id

    encoder_type = model_config.get("encoder_type")
    encoder_config = model_config.get("encoder_config") or {}
    if encoder_type:
        args.encoder_type = encoder_type

    if encoder_type == "tslanet":
        if "patch_size" in encoder_config:
            args.tslanet_patch_size = encoder_config["patch_size"]
    elif encoder_type == "newts_dual_branch":
        structural_keys = [
            "branch_mode",
            "context_length",
            "patch_length",
            "stride",
            "d_model",
            "num_attention_heads",
            "num_hidden_layers",
            "ffn_dim",
            "dropout",
            "vit_model_name",
            "vit_feature_mode",
            "vit_layer_idx",
            "vit_mix_layers",
            "vit_patch_size",
            "vit_stride",
            "vit_truncate_to_feature_layer",
            "vit_num_hidden_layers",
            "projector_type",
            "projector_dropout",
        ]
        for key in structural_keys:
            if key in encoder_config:
                setattr(args, key, encoder_config[key])

    return args


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


def resolve_collate_patch_size(args) -> int:
    if args.encoder_type == "tslanet":
        return args.tslanet_patch_size
    if args.encoder_type == "newts_dual_branch":
        return 1
    return PATCH_SIZE


def make_collate_fn(args, is_train: bool):
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=resolve_collate_patch_size(args),
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


def resolve_base_llm_id(args) -> str:
    if args.pretrained_model:
        return OpenTSLM._get_base_llm_id(args.pretrained_model)
    return args.llm_id


def resolve_dataset_eos_token(args) -> str:
    base_llm_id = resolve_base_llm_id(args)
    tokenizer = AutoTokenizer.from_pretrained(base_llm_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        raise RuntimeError(f"Tokenizer for {base_llm_id} has no EOS token.")
    return tokenizer.eos_token


def validate_args(args):
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")
    if args.aug_scaling_min > args.aug_scaling_max:
        raise ValueError("--aug_scaling_min must be <= --aug_scaling_max")

    if args.encoder_type != "newts_dual_branch":
        return

    if args.pretrained_model:
        raise ValueError("--pretrained_model is not supported with --encoder_type=newts_dual_branch")
    if args.patch_length <= 0:
        raise ValueError("--patch_length must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.context_length is not None and args.context_length <= 0:
        raise ValueError("--context_length must be positive when provided")
    if args.vit_num_hidden_layers is not None and args.vit_num_hidden_layers <= 0:
        raise ValueError("--vit_num_hidden_layers must be positive when provided")

    target_layer = None
    if args.vit_feature_mode == "single":
        if args.vit_layer_idx <= 0:
            raise ValueError("--vit_layer_idx must be positive when --vit_feature_mode=single")
        target_layer = args.vit_layer_idx
    elif args.vit_feature_mode == "scalar_mix":
        if not args.vit_mix_layers:
            raise ValueError("--vit_mix_layers is required when --vit_feature_mode=scalar_mix")
        if any(layer <= 0 for layer in args.vit_mix_layers):
            raise ValueError("--vit_mix_layers must contain positive 1-based layer indices")
        target_layer = max(args.vit_mix_layers)

    if args.vit_num_hidden_layers is not None and target_layer is not None:
        if args.vit_num_hidden_layers < target_layer:
            raise ValueError(
                "--vit_num_hidden_layers must be >= the requested target feature layer depth"
            )


def infer_context_length_from_dataset(dataset: Dataset, patch_length: int) -> int:
    if len(dataset) == 0:
        raise ValueError("Cannot infer context_length from an empty dataset")

    sample = dataset[0]["time_series"][0]
    sample_length = int(torch.as_tensor(sample).numel())
    return ((sample_length + patch_length - 1) // patch_length) * patch_length


def build_tslanet_config(args) -> Dict[str, Any]:
    return {
        "patch_size": args.tslanet_patch_size,
        "output_dim": ENCODER_OUTPUT_DIM,
    }


def build_newts_dual_branch_config(args) -> Dict[str, Any]:
    if args.context_length is None:
        raise ValueError("context_length must be resolved before building the newts_dual_branch config")

    return {
        "output_dim": ENCODER_OUTPUT_DIM,
        "context_length": args.context_length,
        "patch_length": args.patch_length,
        "stride": args.stride,
        "d_model": args.d_model,
        "num_attention_heads": args.num_attention_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "branch_mode": args.branch_mode,
        "vit_model_name": args.vit_model_name,
        "vit_feature_mode": args.vit_feature_mode,
        "vit_layer_idx": args.vit_layer_idx,
        "vit_mix_layers": list(args.vit_mix_layers) if args.vit_mix_layers else None,
        "vit_patch_size": args.vit_patch_size,
        "vit_stride": args.vit_stride,
        "vit_truncate_to_feature_layer": args.vit_truncate_to_feature_layer,
        "vit_num_hidden_layers": args.vit_num_hidden_layers,
        "projector_type": args.projector_type,
        "projector_dropout": args.projector_dropout,
        "freeze_ts_backbone": args.freeze_ts_backbone,
        "freeze_vision_backbone": args.freeze_vision_backbone,
    }


def resolve_model_init_kwargs(args) -> Dict[str, Any]:
    init_kwargs: Dict[str, Any] = {
        "llm_id": args.llm_id,
        "encoder_type": args.encoder_type,
        "tslanet_config": None,
        "newts_dual_branch_config": None,
    }
    if args.encoder_type == "tslanet":
        init_kwargs["tslanet_config"] = build_tslanet_config(args)
    elif args.encoder_type == "newts_dual_branch":
        init_kwargs["newts_dual_branch_config"] = build_newts_dual_branch_config(args)
    return init_kwargs


def resolve_model_init_kwargs_from_checkpoint(args, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    metadata = checkpoint.get("model_config") or {}
    init_kwargs = resolve_model_init_kwargs(args)

    llm_id = metadata.get("llm_id")
    encoder_type = metadata.get("encoder_type")
    encoder_config = metadata.get("encoder_config")

    if llm_id:
        init_kwargs["llm_id"] = llm_id
    if encoder_type:
        init_kwargs["encoder_type"] = encoder_type

    if encoder_type == "tslanet" and encoder_config:
        init_kwargs["tslanet_config"] = dict(encoder_config)
    elif encoder_type == "newts_dual_branch" and encoder_config:
        merged_config = dict(encoder_config)
        merged_config["freeze_ts_backbone"] = args.freeze_ts_backbone
        merged_config["freeze_vision_backbone"] = args.freeze_vision_backbone
        merged_config["output_dim"] = ENCODER_OUTPUT_DIM
        init_kwargs["newts_dual_branch_config"] = merged_config

    return init_kwargs


def build_model(args, device: str, rank: int):
    use_lora = args.use_lora

    if args.local_checkpoint:
        checkpoint = torch.load(args.local_checkpoint, map_location=device, weights_only=False)
        model_init_kwargs = resolve_model_init_kwargs_from_checkpoint(args, checkpoint)
        if rank == 0:
            print(f"📂 Loading local checkpoint: {args.local_checkpoint}")
            print(f"   encoder_type: {model_init_kwargs['encoder_type']}")
            print(f"   llm_id: {model_init_kwargs['llm_id']}")

        model = OpenTSLMSP(
            llm_id=model_init_kwargs["llm_id"],
            device=device,
            encoder_type=model_init_kwargs["encoder_type"],
            tslanet_config=model_init_kwargs["tslanet_config"],
            newts_dual_branch_config=model_init_kwargs["newts_dual_branch_config"],
        )

        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        if rank == 0:
            print("✅ Loaded encoder and projector from local checkpoint")

        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

    elif args.pretrained_model:
        if rank == 0:
            print(f"📂 Loading Hugging Face model: {args.pretrained_model}")

        model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=use_lora,
        )

        if use_lora and (args.lora_r != 16 or args.lora_alpha != 32):
            model.disable_lora()
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            if rank == 0:
                print(f"📎 Reconfigured LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    else:
        model_init_kwargs = resolve_model_init_kwargs(args)
        if rank == 0:
            print("🆕 Training without pretrained checkpoint")
            print(f"   encoder_type: {model_init_kwargs['encoder_type']}")
            print(f"   llm_id: {model_init_kwargs['llm_id']}")

        model = OpenTSLMSP(
            llm_id=model_init_kwargs["llm_id"],
            device=device,
            encoder_type=model_init_kwargs["encoder_type"],
            tslanet_config=model_init_kwargs["tslanet_config"],
            newts_dual_branch_config=model_init_kwargs["newts_dual_branch_config"],
        )

        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.random_init_llm:
        if rank == 0:
            print("🎲 Randomly initializing LLM weights...")
        from transformers import AutoModelForCausalLM

        llm_config = model.llm.config
        random_llm = AutoModelForCausalLM.from_config(
            llm_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        model.llm = random_llm

        for p in model.llm.parameters():
            p.requires_grad = False

        if use_lora:
            model.lora_enabled = False
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

        if rank == 0:
            print("✅ LLM reinitialized")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 Encoder parameters frozen")

    return model


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    import re

    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip()
        label_clean = label.strip()

        match = re.search(r"<c\d+>", pred_clean)
        pred_token = match.group() if match else pred_clean
        if pred_token == label_clean:
            correct += 1

    return correct / len(predictions) if predictions else 0.0


def add_class_tokens_to_model(model, num_classes: int, rank: int = 0):
    class_tokens = [f"<c{i}>" for i in range(num_classes)]

    num_added = model.tokenizer.add_tokens(class_tokens, special_tokens=True)
    if rank == 0:
        print(f"✅ Added {num_added} class tokens to tokenizer")

    old_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    model.llm.resize_token_embeddings(len(model.tokenizer))
    new_vocab_size = model.llm.get_input_embeddings().weight.shape[0]

    if rank == 0:
        print(f"   Vocabulary size: {old_vocab_size} -> {new_vocab_size}")

    with torch.no_grad():
        embedding = model.llm.get_input_embeddings()
        lm_head = model.llm.lm_head

        if num_added > 0:
            old_embeddings = embedding.weight[:-num_added]
            emb_mean = old_embeddings.mean(dim=0)
            emb_std = old_embeddings.std(dim=0)

            for i in range(num_added):
                noise = torch.randn_like(emb_mean) * emb_std * 0.1
                embedding.weight[-num_added + i] = emb_mean + noise

            old_head = lm_head.weight[:-num_added]
            head_mean = old_head.mean(dim=0)
            head_std = old_head.std(dim=0)

            for i in range(num_added):
                noise = torch.randn_like(head_mean) * head_std * 0.1
                lm_head.weight[-num_added + i] = head_mean + noise

            if rank == 0:
                print(f"   Initialized {num_added} class tokens with mean + random perturbation")

    embedding.weight.requires_grad = True
    lm_head.weight.requires_grad = True

    class_token_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in class_tokens]
    if rank == 0:
        preview = class_token_ids[:5]
        suffix = "..." if len(class_token_ids) > 5 else ""
        print(f"   Class token IDs: {preview}{suffix}")

    return class_tokens, class_token_ids


class AllowedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_token_ids:
            if token_id < scores.shape[-1]:
                mask[:, token_id] = 0
        return scores + mask


def set_lora_trainable(model, trainable: bool) -> int:
    underlying_model = get_model(model)
    if not getattr(underlying_model, "lora_enabled", False):
        return 0

    num_params = 0
    for name, param in underlying_model.llm.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = trainable
            num_params += 1
    return num_params


def build_optimizer_scheduler(
    model,
    train_loader: DataLoader,
    args,
    num_epochs: int,
    grad_acc_steps: int,
    include_lora: bool,
):
    underlying_model = get_model(model)
    param_groups = []
    seen_params: set[int] = set()

    def unique_params(params):
        unique = []
        for param in params:
            if param is None or not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)
            unique.append(param)
        return unique

    encoder_params = []
    if not args.freeze_encoder:
        encoder_params = unique_params(list(underlying_model.encoder.parameters()))
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": args.lr_encoder})

    projector_params = unique_params(list(underlying_model.projector.parameters()))
    if projector_params:
        param_groups.append({"params": projector_params, "lr": args.lr_projector})

    if include_lora and args.use_lora:
        lora_params = unique_params(list(underlying_model.get_lora_parameters()))
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})

    class_token_params = unique_params(
        [
            underlying_model.llm.get_input_embeddings().weight,
            underlying_model.llm.lm_head.weight,
        ]
    )
    if class_token_params:
        param_groups.append({"params": class_token_params, "lr": args.lr_lora * 2})

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

    pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch_idx}/{epoch_total}", disable=(rank != 0))
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
def evaluate(
    model,
    data_loader: DataLoader,
    max_new_tokens: int,
    class_token_ids: Optional[List[int]] = None,
    desc: str = "Testing",
    rank: int = 0,
) -> Dict[str, Any]:
    import re

    underlying_model = get_model(model)
    underlying_model.eval()

    total_loss = 0.0
    num_batches = 0
    all_predictions: List[str] = []
    all_labels: List[str] = []

    logits_processor = None
    if class_token_ids is not None:
        eos_token_id = underlying_model.tokenizer.eos_token_id
        allowed_ids = class_token_ids + [eos_token_id]
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])

    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1

        if logits_processor is not None:
            inputs_embeds, attention_mask = underlying_model.pad_and_apply_batch(batch)
            gen_ids = underlying_model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                do_sample=False,
            )
            decoded_predictions = underlying_model.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=False
            )
            predictions = []
            for pred in decoded_predictions:
                match = re.search(r"<c\d+>", pred)
                predictions.append(match.group() if match else pred.strip())
        else:
            predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)

        for sample, pred in zip(batch, predictions):
            label = sample["answer"].replace(underlying_model.get_eos_token(), "").strip()
            all_predictions.append(pred)
            all_labels.append(label)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels)

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
        "model_config": underlying_model.get_checkpoint_metadata(),
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "phase": phase,
        "args": vars(args),
        "embedding_weight": underlying_model.llm.get_input_embeddings().weight.detach().cpu(),
        "lm_head_weight": underlying_model.llm.lm_head.weight.detach().cpu(),
        "tokenizer_vocab_size": len(underlying_model.tokenizer),
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
    underlying_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

    if "embedding_weight" in checkpoint:
        with torch.no_grad():
            underlying_model.llm.get_input_embeddings().weight.copy_(
                checkpoint["embedding_weight"].to(device)
            )
            underlying_model.llm.lm_head.weight.copy_(checkpoint["lm_head_weight"].to(device))

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

    if args.use_lora:
        set_lora_trainable(model, include_lora)

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
    expected_class_tokens: List[str],
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

    model = build_model(args=args, device=device, rank=rank)
    underlying_model = get_model(model)
    class_tokens, class_token_ids = add_class_tokens_to_model(
        underlying_model,
        num_classes=num_classes,
        rank=rank,
    )

    if expected_class_tokens and class_tokens != expected_class_tokens:
        raise RuntimeError(
            f"Class tokens mismatch: expected {expected_class_tokens}, got {class_tokens}"
        )

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
        test_results = evaluate(
            model=model,
            data_loader=test_loader,
            max_new_tokens=args.max_new_tokens,
            class_token_ids=class_token_ids,
            desc="Testing",
            rank=rank,
        )

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_metrics


def main():
    args = parse_args()
    args.use_lora = not args.no_lora
    args = hydrate_args_from_local_checkpoint_metadata(args)
    local_rank, world_size, rank = setup_distributed()

    try:
        validate_args(args)

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
            if rank == 0 and args.device == "cuda":
                print("⚠️ CUDA unavailable, using CPU")
            device = "cpu"

        set_seed(args.seed)

        save_root = os.path.join(args.save_dir, args.dataset)

        eos_rank0 = resolve_dataset_eos_token(args) if rank == 0 else None
        dataset_eos = broadcast_object_from_rank0(eos_rank0, world_size, rank)

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
        class_tokens = UCRClassificationDataset.get_class_tokens()
        label_to_indices = build_label_to_indices(train_dataset)

        if args.encoder_type == "newts_dual_branch" and args.context_length is None:
            args.context_length = infer_context_length_from_dataset(train_dataset, args.patch_length)

        if rank == 0:
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            print("=" * 80)
            print("M2: Strict Few-shot UCR Classification with Pretrained SP Models")
            print("=" * 80)
            print(f"time: {datetime.datetime.now()}")
            print(f"dataset: {args.dataset}")
            print(f"protocol: {args.protocol}")
            print(f"shots: {[shot_to_name(s) for s in shots]}")
            print(f"num_runs: {num_runs}")
            print(f"pretrained_model: {args.pretrained_model}")
            print(f"local_checkpoint: {args.local_checkpoint}")
            print(f"encoder_type: {args.encoder_type}")
            print(f"llm_id: {args.llm_id}")
            print(f"use_lora: {args.use_lora}")
            print(f"pad_mode: {args.pad_mode}")
            print(f"augmentation: {args.enable_augmentation}")
            print(f"ddp world_size: {world_size}")
            if args.encoder_type == "newts_dual_branch":
                print(f"context_length: {args.context_length}")
                print(
                    "vision: "
                    f"mode={args.vit_feature_mode}, "
                    f"layer={args.vit_layer_idx if args.vit_feature_mode == 'single' else args.vit_mix_layers}, "
                    f"truncate={args.vit_truncate_to_feature_layer}, "
                    f"loaded_layers={args.vit_num_hidden_layers}"
                )
            print("=" * 80)

        if rank == 0:
            class_size_brief = {k: len(v) for k, v in label_to_indices.items()}
            print(f"dataset eos token: {repr(dataset_eos)}")
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
                    expected_class_tokens=class_tokens,
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
