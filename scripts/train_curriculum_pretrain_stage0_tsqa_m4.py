#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Curriculum pretraining V4 for UCR-oriented transfer.

Stages:
1. stage0_encoder_ssl
2. stage1_tsqa_transfer
3. stage2_semantic_bridge
4. stage3_m4_caption

This script keeps the stable stage12 TSQA transfer recipe as the downstream
reference checkpoint while replacing stage0 with a branch-preserving multi-view
SSL objective that better matches classification transfer.
"""

import argparse
import datetime
import faulthandler
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder
from opentslm.model.encoder.NewTSVisionEncoder import (
    SUPPORTED_VISION_2D_MODES,
    TIVIT_SQRT_OVERLAP_VISION_2D_MODE,
    resolve_effective_vit_patch_policy,
    resolve_effective_vision_stride,
    validate_vision_2d_mode,
    vision_mode_ignores_patch_size,
)
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import BATCH_SIZE, EARLY_STOP_PAT, ENCODER_OUTPUT_DIM
from opentslm.time_series_datasets.curriculum_pretrain_aux import (
    AlignmentTargetDataset,
    DEFAULT_SYNTHETIC_SAMPLE_TYPES,
    FULL_SYNTHETIC_SAMPLE_TYPES,
    DEFAULT_UCR_TRAIN_LIST,
    MixedPretrainDataset,
    RawSeriesDataset,
    SyntheticAttributeDataset,
    load_m4_raw_records,
    load_tsqa_raw_records,
    load_ucr_train_raw_records,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


faulthandler.enable(all_threads=True, file=sys.stderr)


STAGE0_MIX_WEIGHTS = (2, 1, 2)
STAGE0_VICREG_SIM_COEFF = 25.0
STAGE0_VICREG_VAR_COEFF = 25.0
STAGE0_VICREG_COV_COEFF = 1.0
STAGE0_VICREG_EPS = 1e-4
STAGE2_COMPONENT_NAMES = ("tsqa", "m4", "synthetic")
STAGE2_DEFAULT_MIX_WEIGHTS = (1, 1, 2)
STAGE2_LOSS_W_ALIGN = 0.2
STAGE2_LOSS_W_CONSISTENCY = 0.05
STAGE2_LEGACY_NAME = "stage2_synthetic_semantics"
STAGE2_CANONICAL_NAME = "stage2_semantic_bridge"

STAGE_ORDER = [
    "stage0_encoder_ssl",
    "stage1_tsqa_transfer",
    STAGE2_CANONICAL_NAME,
    "stage3_m4_caption",
]

STAGE_NAME_ALIASES = {
    STAGE2_LEGACY_NAME: STAGE2_CANONICAL_NAME,
}

STAGE_SPECS = {
    "stage0_encoder_ssl": {
        "default_epochs": 8,
        "description": "TS foundation pretraining on raw TSQA + M4 + UCR series",
        "selection": "loss",
        "recommended_use": "Warm-start encoder weights for stage1_tsqa_transfer.",
    },
    "stage1_tsqa_transfer": {
        "default_epochs": 20,
        "default_lr_encoder": 5e-5,
        "default_lr_projector": 1e-4,
        "description": "Reference TSQA transfer stage for downstream classification",
        "selection": "loss",
        "recommended_use": "Default UCR few-shot reference checkpoint.",
    },
    STAGE2_CANONICAL_NAME: {
        "default_epochs": 6,
        "default_lr_encoder": 1e-4,
        "default_lr_projector": 1e-4,
        "description": "Semantic bridge with TSQA retention, M4 alignment, and synthetic attributes",
        "selection": "loss",
        "recommended_use": "Intermediate semantic bridge checkpoint for caption specialization.",
    },
    "stage3_m4_caption": {
        "default_epochs": 12,
        "default_lr_encoder": 1e-4,
        "default_lr_projector": 1e-4,
        "description": "Caption specialization on M4 with frozen encoder and LoRA",
        "selection": "loss",
        "recommended_use": "Default captioning checkpoint.",
    },
}

STAGE_ALIAS_FILENAMES = {
    "stage1_tsqa_transfer": "stage1_transfer_checkpoint.pt",
    STAGE2_CANONICAL_NAME: "stage2_semantic_bridge_checkpoint.pt",
    "stage3_m4_caption": "stage3_m4_checkpoint.pt",
}


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    values = [int(token.strip()) for token in value.split(",") if token.strip()]
    return values or None


def parse_str_list(value: Optional[str]) -> Tuple[str, ...]:
    if value is None:
        return ()
    values = tuple(token.strip() for token in value.split(",") if token.strip())
    return values


def parse_stage_list(value: str) -> List[str]:
    stages = [stage.strip() for stage in value.split(",") if stage.strip()]
    if not stages:
        raise ValueError("At least one stage must be provided in --stages")
    normalized = [normalize_stage_name(stage) for stage in stages]
    unknown = [stage for stage in normalized if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid stages: {STAGE_ORDER}")
    return normalized


def normalize_stage_name(stage_name: str) -> str:
    return STAGE_NAME_ALIASES.get(stage_name, stage_name)


def parse_weight_list(value: Optional[str], *, expected_len: int) -> Tuple[int, ...]:
    weights = parse_int_list(value)
    if weights is None:
        raise ValueError("Weights must be provided")
    if len(weights) != expected_len:
        raise ValueError(f"Expected {expected_len} weights, got {len(weights)}")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Weights must all be positive")
    return tuple(int(weight) for weight in weights)


def stage_dependency_candidates(stage_name: str) -> List[str]:
    stage_name = normalize_stage_name(stage_name)
    if stage_name == "stage0_encoder_ssl":
        return []
    if stage_name == "stage1_tsqa_transfer":
        return ["stage0_encoder_ssl"]
    if stage_name == STAGE2_CANONICAL_NAME:
        return ["stage1_tsqa_transfer"]
    if stage_name == "stage3_m4_caption":
        return [STAGE2_CANONICAL_NAME, "stage1_tsqa_transfer"]
    raise ValueError(f"Unsupported stage: {stage_name}")


def sanitize_llm_id(llm_id: str) -> str:
    name = llm_id.split("/")[-1] if llm_id else "unknown_llm"
    name = name.replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def cli_flag_was_provided(argv: Optional[List[str]], flag_name: str) -> bool:
    if argv is None:
        argv = sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in argv)


def default_run_name(args) -> str:
    return "newts_dual_branch_curriculum_v4"


def parse_args(argv=None):
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Curriculum pretraining V4 with stage0/TSQA/semantic-bridge/M4 stages")

    parser.add_argument(
        "--stages",
        type=str,
        default=f"stage0_encoder_ssl,stage1_tsqa_transfer,{STAGE2_CANONICAL_NAME},stage3_m4_caption",
        help="Comma-separated stages to run.",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional subdirectory for this run")
    parser.add_argument("--save_dir", type=str, default="results/curriculum_pretrain_stage0_tsqa_m4")
    parser.add_argument("--resume", action="store_true", help="Resume the current stage from its checkpoint if present")

    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--llm_attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--random_init_llm", action="store_true", help="Replace the frozen base LLM with a random initialization")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_lora", action="store_true", help="Explicitly enable LoRA for stage3_m4_caption (it is enabled by default when not specified)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr_lora", type=float, default=1e-4)

    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vit_feature_mode", type=str, default="single", choices=["last", "single", "scalar_mix"])
    parser.add_argument("--vit_layer_idx", type=int, default=4)
    parser.add_argument("--vit_mix_layers", type=str, default=None)
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    parser.add_argument(
        "--vision_2d_mode",
        type=str,
        default=TIVIT_SQRT_OVERLAP_VISION_2D_MODE,
        choices=list(SUPPORTED_VISION_2D_MODES),
    )
    parser.add_argument("--vit_num_hidden_layers", type=int, default=None)
    parser.add_argument(
        "--vit_truncate_to_feature_layer",
        dest="vit_truncate_to_feature_layer",
        action="store_true",
    )
    parser.add_argument(
        "--no_vit_truncate_to_feature_layer",
        dest="vit_truncate_to_feature_layer",
        action="store_false",
    )
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--projector_dropout", type=float, default=0.1)
    parser.add_argument("--use_pma", action="store_true")
    parser.add_argument("--aggregator_layers", type=int, default=2)
    parser.add_argument("--aggregator_hidden_size", type=int, default=None)
    parser.add_argument("--aggregator_num_heads", type=int, default=8)
    parser.add_argument("--aggregator_ffn_dim", type=int, default=None)
    parser.add_argument("--aggregator_num_queries", type=int, default=2)
    parser.add_argument("--aggregator_query_mode", type=str, default="separate", choices=["shared", "separate"])
    parser.add_argument("--aggregator_fusion_mode", type=str, default="concat_linear", choices=["gated_sum", "concat_linear"])
    parser.add_argument("--aggregator_gate_type", type=str, default="dynamic", choices=["scalar", "slot", "dynamic"])
    parser.add_argument("--aggregator_fuse_layers", type=int, default=1)
    parser.add_argument("--freeze_ts_backbone", action="store_true")
    parser.add_argument("--freeze_vision_backbone", dest="freeze_vision_backbone", action="store_true")
    parser.add_argument("--no_freeze_vision_backbone", dest="freeze_vision_backbone", action="store_false")
    parser.set_defaults(vit_truncate_to_feature_layer=True, freeze_vision_backbone=True)

    parser.add_argument("--stage0_epochs", type=int, default=STAGE_SPECS["stage0_encoder_ssl"]["default_epochs"])
    parser.add_argument("--stage0_lr_ts", type=float, default=2e-4)
    parser.add_argument("--stage0_lr_vision", type=float, default=5e-5)
    parser.add_argument("--stage0_lr_heads", type=float, default=1e-4)
    parser.add_argument("--stage0_mask_ratio", type=float, default=0.4)
    parser.add_argument("--stage0_batch_multiplier", type=int, default=4)
    parser.add_argument(
        "--stage0_mix_weights",
        type=str,
        default="2,1,2",
        help="Comma-separated train mix weights for stage0 in TSQA,M4,UCR order.",
    )
    parser.add_argument("--stage0_downstream_pool", type=str, default="ucr_train_list", choices=["none", "ucr_train_list"])
    parser.add_argument(
        "--stage0_train_vision",
        action="store_true",
        help="Train the stage0 vision backbone instead of keeping the ViT backbone frozen.",
    )
    parser.add_argument(
        "--stage0_branch_dropout",
        type=float,
        default=0.1,
        help="Stage0-only random branch dropout probability for fused multi-view SSL.",
    )
    parser.add_argument("--stage0_w_ts_recon", type=float, default=1.0)
    parser.add_argument("--stage0_w_ts_vicreg", type=float, default=0.25)
    parser.add_argument("--stage0_w_vi_vicreg", type=float, default=0.25)
    parser.add_argument("--stage0_w_fuse_vicreg", type=float, default=0.5)
    parser.add_argument("--aug_jitter_std", type=float, default=0.03)
    parser.add_argument("--aug_scaling_min", type=float, default=0.8)
    parser.add_argument("--aug_scaling_max", type=float, default=1.2)
    parser.add_argument("--aug_time_mask_ratio", type=float, default=0.1)
    parser.add_argument("--ucr_data_path", type=str, default="./data")
    parser.add_argument("--ucr_train_list_path", type=str, default=str(DEFAULT_UCR_TRAIN_LIST))

    parser.add_argument("--stage1_epochs", type=int, default=STAGE_SPECS["stage1_tsqa_transfer"]["default_epochs"])
    parser.add_argument("--stage2_epochs", type=int, default=STAGE_SPECS[STAGE2_CANONICAL_NAME]["default_epochs"])
    parser.add_argument("--stage3_epochs", type=int, default=STAGE_SPECS["stage3_m4_caption"]["default_epochs"])
    parser.add_argument("--stage1_lr_encoder", type=float, default=STAGE_SPECS["stage1_tsqa_transfer"]["default_lr_encoder"])
    parser.add_argument("--stage1_lr_projector", type=float, default=STAGE_SPECS["stage1_tsqa_transfer"]["default_lr_projector"])
    parser.add_argument("--stage2_lr_encoder", type=float, default=STAGE_SPECS[STAGE2_CANONICAL_NAME]["default_lr_encoder"])
    parser.add_argument("--stage2_lr_projector", type=float, default=STAGE_SPECS[STAGE2_CANONICAL_NAME]["default_lr_projector"])
    parser.add_argument("--stage3_lr_encoder", type=float, default=STAGE_SPECS["stage3_m4_caption"]["default_lr_encoder"])
    parser.add_argument("--stage3_lr_projector", type=float, default=STAGE_SPECS["stage3_m4_caption"]["default_lr_projector"])
    parser.add_argument(
        "--stage1_projector_only_epochs",
        type=int,
        default=2,
        help="Freeze the stage1 encoder for the first N epochs before low-LR joint tuning.",
    )
    parser.add_argument(
        "--stage2_synthetic_sample_types",
        type=str,
        default=",".join(DEFAULT_SYNTHETIC_SAMPLE_TYPES),
        help="Comma-separated synthetic sample types for stage2 semantic bridge. Defaults exclude match_mismatch.",
    )
    parser.add_argument(
        "--stage2_mix_weights",
        type=str,
        default="1,1,2",
        help="Comma-separated train mix weights for stage2_semantic_bridge in TSQA,M4,SYN order.",
    )
    parser.add_argument(
        "--stage3_unfreeze_encoder",
        action="store_true",
        help="Allow stage3_m4_caption to update encoder weights instead of using the default frozen-encoder path.",
    )

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=EARLY_STOP_PAT)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "last", "repeat"])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--export_model_checkpoint",
        dest="export_model_checkpoint",
        action="store_true",
        help="Export a slim model_checkpoint.pt after the last requested SP stage",
    )
    parser.add_argument(
        "--no_export_model_checkpoint",
        dest="export_model_checkpoint",
        action="store_false",
    )
    parser.set_defaults(export_model_checkpoint=True)

    args = parser.parse_args(argv)
    args.stages = parse_stage_list(args.stages)
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    args.stage0_mix_weights = parse_weight_list(args.stage0_mix_weights, expected_len=3)
    args.stage2_synthetic_sample_types = parse_str_list(args.stage2_synthetic_sample_types)
    args.stage2_mix_weights = parse_weight_list(args.stage2_mix_weights, expected_len=len(STAGE2_COMPONENT_NAMES))
    args.pad_mode_explicit = cli_flag_was_provided(provided_argv, "--pad_mode")
    args.vit_patch_size_explicit = cli_flag_was_provided(provided_argv, "--vit_patch_size")
    args.vit_stride_explicit = cli_flag_was_provided(provided_argv, "--vit_stride")
    args.vision_2d_mode_explicit = cli_flag_was_provided(provided_argv, "--vision_2d_mode")
    args.use_lora_explicit = cli_flag_was_provided(provided_argv, "--use_lora")
    validate_args(args)
    return args


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.eval_batch_size < 1:
        raise ValueError("--eval_batch_size must be >= 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.early_stop < 1:
        raise ValueError("--early_stop must be >= 1")
    if args.patch_length <= 0:
        raise ValueError("--patch_length must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.vit_num_hidden_layers is not None and args.vit_num_hidden_layers <= 0:
        raise ValueError("--vit_num_hidden_layers must be positive when provided")
    if args.stage0_batch_multiplier < 1:
        raise ValueError("--stage0_batch_multiplier must be >= 1")
    if not 0.0 <= args.stage0_branch_dropout <= 0.5:
        raise ValueError("--stage0_branch_dropout must be in [0, 0.5]")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader_num_workers must be >= 0")
    if any(epoch <= 0 for epoch in [args.stage0_epochs, args.stage1_epochs, args.stage2_epochs, args.stage3_epochs]):
        raise ValueError("All stage epoch counts must be positive")
    if args.stage1_projector_only_epochs < 0:
        raise ValueError("--stage1_projector_only_epochs must be >= 0")
    if args.stage1_projector_only_epochs > args.stage1_epochs:
        raise ValueError("--stage1_projector_only_epochs must be <= --stage1_epochs")
    if args.aug_scaling_min <= 0 or args.aug_scaling_max <= 0 or args.aug_scaling_min > args.aug_scaling_max:
        raise ValueError("Invalid augmentation scaling range")
    if not 0.0 <= args.stage0_mask_ratio < 1.0:
        raise ValueError("--stage0_mask_ratio must be in [0, 1)")
    if not 0.0 <= args.aug_time_mask_ratio < 1.0:
        raise ValueError("--aug_time_mask_ratio must be in [0, 1)")
    if any(weight < 0.0 for weight in [
        args.stage0_w_ts_recon,
        args.stage0_w_ts_vicreg,
        args.stage0_w_vi_vicreg,
        args.stage0_w_fuse_vicreg,
    ]):
        raise ValueError("Stage0 loss weights must be non-negative")
    validate_vision_2d_mode(args.vision_2d_mode)

    if args.vit_feature_mode == "single":
        if args.vit_layer_idx <= 0:
            raise ValueError("--vit_layer_idx must be positive when --vit_feature_mode=single")
        target_layer = args.vit_layer_idx
    elif args.vit_feature_mode == "scalar_mix":
        if not args.vit_mix_layers:
            raise ValueError("--vit_mix_layers is required when --vit_feature_mode=scalar_mix")
        if any(layer <= 0 for layer in args.vit_mix_layers):
            raise ValueError("--vit_mix_layers must contain positive layer indices")
        target_layer = max(args.vit_mix_layers)
    else:
        target_layer = None

    if args.vit_num_hidden_layers is not None and target_layer is not None and args.vit_num_hidden_layers < target_layer:
        raise ValueError("--vit_num_hidden_layers must be >= selected feature layer depth")

    if args.use_pma:
        if args.aggregator_layers <= 0:
            raise ValueError("--aggregator_layers must be positive")
        if args.aggregator_num_heads <= 0:
            raise ValueError("--aggregator_num_heads must be positive")
        if args.aggregator_num_queries <= 0:
            raise ValueError("--aggregator_num_queries must be positive")
        if args.aggregator_fuse_layers < 0:
            raise ValueError("--aggregator_fuse_layers must be >= 0")
        hidden_size = args.aggregator_hidden_size or ENCODER_OUTPUT_DIM
        if hidden_size % args.aggregator_num_heads != 0:
            raise ValueError("--aggregator_num_heads must evenly divide the PMA hidden size")

    unknown_sample_types = sorted(set(args.stage2_synthetic_sample_types) - set(FULL_SYNTHETIC_SAMPLE_TYPES))
    if unknown_sample_types:
        raise ValueError(f"Unsupported --stage2_synthetic_sample_types entries: {unknown_sample_types}")
    if len(args.stage2_mix_weights) != len(STAGE2_COMPONENT_NAMES):
        raise ValueError(f"--stage2_mix_weights must contain {len(STAGE2_COMPONENT_NAMES)} entries")


def setup_distributed() -> Tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device("cuda", local_rank),
        )
    except TypeError:
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, world_size, rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(model):
    return model.module if hasattr(model, "module") else model


def broadcast_object_from_rank0(obj, rank: int):
    if not dist.is_initialized():
        return obj
    holder = [obj if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_effective_newts_vision_config(args) -> Dict[str, Any]:
    vision_2d_mode = validate_vision_2d_mode(args.vision_2d_mode)
    return {
        "vision_2d_mode": vision_2d_mode,
        "effective_vit_stride": resolve_effective_vision_stride(
            vision_2d_mode,
            args.vit_stride,
            stride_explicit=getattr(args, "vit_stride_explicit", False),
        ),
        "effective_vit_patch_policy": resolve_effective_vit_patch_policy(vision_2d_mode),
    }


def resolve_effective_pad_mode(args) -> str:
    if not getattr(args, "pad_mode_explicit", False):
        return "last"
    return args.pad_mode


def warn_newts_vision_runtime_config(args, rank: int):
    if rank != 0:
        return
    resolved = resolve_effective_newts_vision_config(args)
    if resolved["vision_2d_mode"] == TIVIT_SQRT_OVERLAP_VISION_2D_MODE and not getattr(args, "vit_stride_explicit", False):
        print(
            "ℹ️ --vit_stride not provided for tivit_sqrt_overlap; "
            f"using TiViT default overlap ratio {resolved['effective_vit_stride']:.1f}"
        )
    if vision_mode_ignores_patch_size(resolved["vision_2d_mode"]) and getattr(args, "vit_patch_size_explicit", False):
        print(
            "⚠️ --vit_patch_size is ignored when --vision_2d_mode=tivit_sqrt_overlap; "
            "patch_size is resolved dynamically as int(sqrt(T))"
        )


def build_newts_dual_branch_config(args, *, for_stage0: bool) -> Dict[str, Any]:
    resolved_vision_config = resolve_effective_newts_vision_config(args)
    config = {
        "output_dim": ENCODER_OUTPUT_DIM,
        "dynamic_length": True,
        "ts_positional_encoding": "sinusoidal",
        "patch_length": args.patch_length,
        "stride": args.stride,
        "d_model": args.d_model,
        "num_attention_heads": args.num_attention_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "ffn_dim": args.ffn_dim,
        "dropout": args.dropout,
        "branch_mode": "both",
        "vit_model_name": args.vit_model_name,
        "vit_feature_mode": args.vit_feature_mode,
        "vit_layer_idx": args.vit_layer_idx,
        "vit_mix_layers": list(args.vit_mix_layers) if args.vit_mix_layers else None,
        "vit_patch_size": args.vit_patch_size,
        "vit_stride": resolved_vision_config["effective_vit_stride"],
        "vision_2d_mode": resolved_vision_config["vision_2d_mode"],
        "vit_truncate_to_feature_layer": args.vit_truncate_to_feature_layer,
        "vit_num_hidden_layers": args.vit_num_hidden_layers,
        "projector_type": args.projector_type,
        "projector_dropout": args.projector_dropout,
        "use_pma": False if for_stage0 else args.use_pma,
        "aggregator_layers": args.aggregator_layers,
        "aggregator_hidden_size": args.aggregator_hidden_size,
        "aggregator_num_heads": args.aggregator_num_heads,
        "aggregator_ffn_dim": args.aggregator_ffn_dim,
        "aggregator_num_queries": args.aggregator_num_queries,
        "aggregator_query_mode": args.aggregator_query_mode,
        "aggregator_fusion_mode": args.aggregator_fusion_mode,
        "aggregator_gate_type": args.aggregator_gate_type,
        "aggregator_fuse_layers": args.aggregator_fuse_layers,
        "freeze_ts_backbone": False if for_stage0 else args.freeze_ts_backbone,
        "freeze_vision_backbone": (not args.stage0_train_vision) if for_stage0 else args.freeze_vision_backbone,
        "enable_modality_embeddings": False,
        "branch_dropout": args.stage0_branch_dropout if for_stage0 else 0.0,
    }
    return config


def resolve_model_init_kwargs(args) -> Dict[str, Any]:
    return {
        "llm_id": args.llm_id,
        "encoder_type": "newts_dual_branch",
        "newts_dual_branch_config": build_newts_dual_branch_config(args, for_stage0=False),
    }


def calculate_accuracy(predictions: Sequence[str], gold_answers: Sequence[str]) -> float:
    correct = 0
    total = len(predictions)
    for pred, gold in zip(predictions, gold_answers):
        pred_clean = pred.strip()
        gold_clean = gold.strip()
        if gold_clean.startswith(pred_clean) or pred_clean == gold_clean:
            correct += 1
    return correct / total if total > 0 else 0.0


def save_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def effective_config_snapshot(args, world_size: int) -> Dict[str, Any]:
    snapshot = dict(vars(args))
    snapshot["effective_pad_mode"] = resolve_effective_pad_mode(args)
    snapshot["world_size"] = int(world_size)
    snapshot["effective_global_batch_size"] = (
        int(world_size) * int(args.batch_size) * int(args.gradient_accumulation_steps)
    )
    resolved_vision_config = resolve_effective_newts_vision_config(args)
    snapshot["vision_2d_mode"] = resolved_vision_config["vision_2d_mode"]
    snapshot["effective_vit_stride"] = resolved_vision_config["effective_vit_stride"]
    snapshot["effective_vit_patch_policy"] = resolved_vision_config["effective_vit_patch_policy"]
    return snapshot


def save_launch_configs(run_dir: str, args, world_size: int):
    latest_config = dict(vars(args))
    effective_config = effective_config_snapshot(args, world_size)
    save_json(os.path.join(run_dir, "config.json"), latest_config)
    save_json(os.path.join(run_dir, "effective_config.json"), effective_config)
    launch_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_json(os.path.join(run_dir, "launch_history", f"{launch_stamp}.json"), effective_config)


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
        bucket_size_multiplier: int = 20,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
        self.seed = int(seed)
        self.epoch = 0
        self.sample_lengths = [self._resolve_sample_length(dataset, idx) for idx in range(len(dataset))]

    @staticmethod
    def _infer_sample_length(sample: Dict[str, Any]) -> int:
        max_len = 0
        for ts in sample.get("time_series", []):
            max_len = max(max_len, int(torch.as_tensor(ts).numel()))
        if max_len <= 0:
            raise ValueError("Encountered a sample without a positive time-series length")
        return max_len

    @classmethod
    def _resolve_sample_length(cls, dataset: Dataset, idx: int) -> int:
        get_sample_length = getattr(dataset, "get_sample_length", None)
        if callable(get_sample_length):
            return int(get_sample_length(idx))
        return cls._infer_sample_length(dataset[idx])

    def _build_all_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.sample_lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        indices.sort(key=self.sample_lengths.__getitem__)

        batches: List[List[int]] = []
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket = list(indices[bucket_start : bucket_start + self.bucket_size])
            if self.shuffle:
                rng.shuffle(bucket)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def _get_rank_batches(self) -> List[List[int]]:
        batches = self._build_all_batches()
        if self.num_replicas == 1:
            return batches
        if self.drop_last:
            total_batches = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:total_batches]
        elif batches:
            total_batches = math.ceil(len(batches) / self.num_replicas) * self.num_replicas
            if len(batches) < total_batches:
                batches.extend(batches[: total_batches - len(batches)])
        return batches[self.rank : len(batches) : self.num_replicas]

    def __iter__(self):
        yield from self._get_rank_batches()

    def __len__(self) -> int:
        total_batches = len(self.sample_lengths) // self.batch_size
        if not self.drop_last and len(self.sample_lengths) % self.batch_size != 0:
            total_batches += 1
        if self.num_replicas == 1:
            return total_batches
        if self.drop_last:
            return total_batches // self.num_replicas
        return math.ceil(total_batches / self.num_replicas)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


def create_sp_data_loader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_patch_size: int,
    pad_mode: str,
    world_size: int,
    rank: int,
    distribute_data: bool,
    use_length_bucket: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    sampler = None
    batch_sampler = None
    if use_length_bucket:
        batch_sampler = LengthBucketBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_replicas=world_size if distribute_data else 1,
            rank=rank if distribute_data else 0,
            seed=seed,
        )
        shuffle = False
    elif distribute_data and world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    collate = lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch,
        patch_size=collate_patch_size,
        pad_mode=pad_mode,
    )

    loader_kwargs = {
        "dataset": dataset,
        "collate_fn": collate,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    if batch_sampler is not None:
        loader_kwargs["batch_sampler"] = batch_sampler
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["sampler"] = sampler
    return DataLoader(**loader_kwargs)


def pad_1d_series(ts: torch.Tensor, padded_len: int, pad_mode: str = "last") -> torch.Tensor:
    if ts.numel() >= padded_len:
        return ts[:padded_len]
    pad_amt = padded_len - ts.numel()
    if pad_mode == "zero":
        return F.pad(ts, (0, pad_amt), value=0.0)
    if pad_mode == "repeat":
        repeats = math.ceil(pad_amt / max(ts.numel(), 1))
        return torch.cat([ts, ts.repeat(repeats)[:pad_amt]], dim=0)
    last_val = ts[-1]
    return torch.cat([ts, torch.full((pad_amt,), last_val, dtype=ts.dtype)], dim=0)


def collate_raw_series_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    series_list = [sample["series"].flatten() for sample in batch]
    padded_len = max(series.numel() for series in series_list)
    padded = [pad_1d_series(series, padded_len, pad_mode="last") for series in series_list]
    return {
        "series": torch.stack(padded, dim=0),
        "series_id": [sample["series_id"] for sample in batch],
        "source_name": [sample["source_name"] for sample in batch],
    }


def create_simple_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    world_size: int,
    rank: int,
    collate_fn,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(**loader_kwargs)


def build_stage0_datasets(args) -> Dict[str, Dataset]:
    train_sets: List[Dataset] = [RawSeriesDataset(load_tsqa_raw_records("train")), RawSeriesDataset(load_m4_raw_records("train"))]
    mix_weights = list(args.stage0_mix_weights[:2])
    if args.stage0_downstream_pool == "ucr_train_list":
        train_sets.append(
            RawSeriesDataset(
                load_ucr_train_raw_records(
                    raw_data_path=args.ucr_data_path,
                    dataset_list_path=args.ucr_train_list_path,
                )
            )
        )
        mix_weights.append(args.stage0_mix_weights[2])
    tsqa_units = math.ceil(len(train_sets[0]) / max(mix_weights[0], 1))
    epoch_size = tsqa_units * sum(mix_weights)
    return {
        "train": MixedPretrainDataset(train_sets, mix_weights, seed=args.seed, epoch_size=epoch_size),
        "validation": ConcatDataset(
            [
                RawSeriesDataset(load_tsqa_raw_records("validation")),
                RawSeriesDataset(load_m4_raw_records("validation")),
            ]
        ),
        "test": ConcatDataset(
            [
                RawSeriesDataset(load_tsqa_raw_records("test")),
                RawSeriesDataset(load_m4_raw_records("test")),
            ]
        ),
    }


def build_stage2_component_datasets(split: str, eos_token: str, args) -> Dict[str, Dataset]:
    from opentslm.time_series_datasets.TSQADataset import TSQADataset
    from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset

    return {
        "tsqa": AlignmentTargetDataset(
            TSQADataset(split, EOS_TOKEN=eos_token),
            eos_token=eos_token,
            source_name="tsqa",
            alignment_from_answer=False,
        ),
        "m4": AlignmentTargetDataset(
            M4QADataset(split, EOS_TOKEN=eos_token),
            eos_token=eos_token,
            source_name="m4",
            alignment_from_answer=True,
        ),
        "synthetic": SyntheticAttributeDataset(
            split,
            eos_token=eos_token,
            seed=args.seed,
            sample_types=args.stage2_synthetic_sample_types,
        ),
    }


def create_stage_dataset(stage_name: str, split: str, eos_token: str, args) -> Dataset:
    stage_name = normalize_stage_name(stage_name)
    if stage_name == "stage1_tsqa_transfer":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset

        return TSQADataset(split, EOS_TOKEN=eos_token)
    if stage_name == STAGE2_CANONICAL_NAME:
        split_components = build_stage2_component_datasets(split, eos_token, args)
        if split == "train":
            return MixedPretrainDataset(
                [split_components[name] for name in STAGE2_COMPONENT_NAMES],
                args.stage2_mix_weights,
                seed=args.seed,
            )
        return ConcatDataset([split_components[name] for name in STAGE2_COMPONENT_NAMES])
    if stage_name == "stage3_m4_caption":
        from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset

        return M4QADataset(split, EOS_TOKEN=eos_token)
    raise ValueError(f"Unsupported stage: {stage_name}")


class DualViewSSLModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_config: Dict[str, Any],
        device: str,
        train_vision: bool,
        mask_ratio: float,
        jitter_std: float,
        scaling_range: Tuple[float, float],
        time_mask_ratio: float,
        loss_w_ts_recon: float,
        loss_w_ts_vicreg: float,
        loss_w_vi_vicreg: float,
        loss_w_fuse_vicreg: float,
    ):
        super().__init__()
        self.device = device
        self.train_vision = bool(train_vision)
        self.mask_ratio = float(mask_ratio)
        self.jitter_std = float(jitter_std)
        self.scaling_range = scaling_range
        self.time_mask_ratio = float(time_mask_ratio)
        self.loss_w_ts_recon = float(loss_w_ts_recon)
        self.loss_w_ts_vicreg = float(loss_w_ts_vicreg)
        self.loss_w_vi_vicreg = float(loss_w_vi_vicreg)
        self.loss_w_fuse_vicreg = float(loss_w_fuse_vicreg)

        self.encoder = NewTSDualBranchEncoder(**encoder_config, device=device).to(device)
        self.ts_recon_head = nn.Linear(self.encoder.output_dim, self.encoder.patch_length).to(device)
        self.ts_ssl_head = self._build_ssl_head(self.encoder.output_dim).to(device)
        self.vision_ssl_head = self._build_ssl_head(self.encoder.output_dim).to(device)
        self.fuse_ssl_head = self._build_ssl_head(self.encoder.output_dim).to(device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.compute_losses(batch)["loss_total"]

    @staticmethod
    def _build_ssl_head(hidden_size: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _apply_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0:
            return x
        masked = x.clone()
        length = masked.size(1)
        mask_len = max(1, int(round(length * self.mask_ratio)))
        for idx in range(masked.size(0)):
            start = torch.randint(0, max(length - mask_len + 1, 1), (1,), device=masked.device).item()
            masked[idx, start : start + mask_len] = 0.0
        return masked

    def _apply_augment(self, x: torch.Tensor) -> torch.Tensor:
        augmented = x + torch.randn_like(x) * self.jitter_std
        scaling = torch.empty((augmented.size(0), 1), device=augmented.device).uniform_(
            self.scaling_range[0],
            self.scaling_range[1],
        )
        augmented = augmented * scaling
        mask_len = max(1, int(round(augmented.size(1) * self.time_mask_ratio)))
        for idx in range(augmented.size(0)):
            start = torch.randint(0, max(augmented.size(1) - mask_len + 1, 1), (1,), device=augmented.device).item()
            augmented[idx, start : start + mask_len] = 0.0
        return augmented

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        if n != m:
            raise ValueError("off_diagonal expects a square matrix")
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    @classmethod
    def _vicreg_loss(cls, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        if z_a.shape != z_b.shape:
            raise ValueError(f"VICReg tensors must share a shape, got {tuple(z_a.shape)} vs {tuple(z_b.shape)}")

        repr_loss = F.mse_loss(z_a, z_b)

        std_a = torch.sqrt(z_a.var(dim=0, unbiased=False) + STAGE0_VICREG_EPS)
        std_b = torch.sqrt(z_b.var(dim=0, unbiased=False) + STAGE0_VICREG_EPS)
        var_loss = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))

        centered_a = z_a - z_a.mean(dim=0, keepdim=True)
        centered_b = z_b - z_b.mean(dim=0, keepdim=True)
        cov_a = (centered_a.transpose(0, 1) @ centered_a) / max(z_a.size(0) - 1, 1)
        cov_b = (centered_b.transpose(0, 1) @ centered_b) / max(z_b.size(0) - 1, 1)
        cov_loss = cls._off_diagonal(cov_a).pow(2).sum() / max(z_a.size(1), 1)
        cov_loss = cov_loss + cls._off_diagonal(cov_b).pow(2).sum() / max(z_b.size(1), 1)

        return (
            STAGE0_VICREG_SIM_COEFF * repr_loss
            + STAGE0_VICREG_VAR_COEFF * var_loss
            + STAGE0_VICREG_COV_COEFF * cov_loss
        )

    def _encode_view(
        self,
        x: torch.Tensor,
        *,
        runtime_branch_mode: str,
        apply_mask: bool = False,
    ) -> Dict[str, Any]:
        source = self._apply_mask(x) if apply_mask else x
        return self.encoder(
            source,
            return_intermediates=True,
            runtime_branch_mode=runtime_branch_mode,
        )

    def _masked_reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        masked_outputs = self._encode_view(x, runtime_branch_mode="ts_only", apply_mask=True)
        target_patches = self.encoder.ts_backbone._extract_patches(x)
        pred_patches = self.ts_recon_head(masked_outputs["ts_tokens"].float()).to(target_patches.dtype)
        return F.mse_loss(pred_patches, target_patches)

    def compute_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch["series"].to(self.device, non_blocking=True)
        x_a = self._apply_augment(x)
        x_b = self._apply_augment(x)

        loss_ts_recon = 0.5 * (
            self._masked_reconstruction_loss(x_a) + self._masked_reconstruction_loss(x_b)
        )

        ts_outputs_a = self._encode_view(x_a, runtime_branch_mode="ts_only")
        ts_outputs_b = self._encode_view(x_b, runtime_branch_mode="ts_only")
        vi_outputs_a = self._encode_view(x_a, runtime_branch_mode="vision_only")
        vi_outputs_b = self._encode_view(x_b, runtime_branch_mode="vision_only")
        fuse_outputs_a = self._encode_view(x_a, runtime_branch_mode="both")
        fuse_outputs_b = self._encode_view(x_b, runtime_branch_mode="both")

        z_ts_a = self.ts_ssl_head(ts_outputs_a["pooled_ts"].float())
        z_ts_b = self.ts_ssl_head(ts_outputs_b["pooled_ts"].float())
        loss_ts_vicreg = self._vicreg_loss(z_ts_a, z_ts_b)

        z_vi_a = self.vision_ssl_head(vi_outputs_a["pooled_vision"].float())
        z_vi_b = self.vision_ssl_head(vi_outputs_b["pooled_vision"].float())
        loss_vi_vicreg = self._vicreg_loss(z_vi_a, z_vi_b)

        z_fuse_a = self.fuse_ssl_head(fuse_outputs_a["pooled_fused"].float())
        z_fuse_b = self.fuse_ssl_head(fuse_outputs_b["pooled_fused"].float())
        loss_fuse_vicreg = self._vicreg_loss(z_fuse_a, z_fuse_b)

        loss_total = (
            self.loss_w_ts_recon * loss_ts_recon
            + self.loss_w_ts_vicreg * loss_ts_vicreg
            + self.loss_w_vi_vicreg * loss_vi_vicreg
            + self.loss_w_fuse_vicreg * loss_fuse_vicreg
        )
        return {
            "loss_total": loss_total,
            "loss_ts_recon": loss_ts_recon,
            "loss_ts_vicreg": loss_ts_vicreg,
            "loss_vi_vicreg": loss_vi_vicreg,
            "loss_fuse_vicreg": loss_fuse_vicreg,
        }

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "encoder_type": "newts_dual_branch",
            "encoder_config": self.encoder.get_config(),
        }


def sanitize_checkpoint_metadata(model_config: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(model_config or {})
    encoder_config = dict(sanitized.get("encoder_config") or {})
    for key in ["branch_dropout", "enable_modality_embeddings", "vision_train_mode", "vision_topk_blocks"]:
        encoder_config.pop(key, None)
    sanitized["encoder_config"] = encoder_config
    for key in ["alignment_losses_enabled", "loss_w_align", "loss_w_consistency", "alignment_temperature"]:
        sanitized.pop(key, None)
    return sanitized


def build_sp_export_payload(model) -> Dict[str, Any]:
    underlying = get_model(model)
    payload = {
        "model_config": sanitize_checkpoint_metadata(underlying.get_checkpoint_metadata()),
        "encoder_state": underlying.encoder.state_dict(),
        "projector_state": underlying.projector.state_dict(),
    }
    underlying.save_lora_state_to_checkpoint(payload)
    return payload


def export_stage_alias_checkpoint(model, alias_path: str, rank: int):
    if rank != 0:
        return
    os.makedirs(os.path.dirname(alias_path), exist_ok=True)
    torch.save(build_sp_export_payload(model), alias_path)
    print(f"📦 Exported stage alias checkpoint to: {alias_path}")


def export_final_model_checkpoint(model, export_path: str, random_init_llm: bool, rank: int):
    if rank != 0:
        return
    if random_init_llm:
        print("⚠️ Skipping model_checkpoint.pt export because --random_init_llm was used")
        return
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    torch.save(build_sp_export_payload(model), export_path)
    print(f"📦 Exported OpenTSLM-compatible checkpoint to: {export_path}")


def extract_sp_component_states_from_checkpoint(checkpoint: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    encoder_state = checkpoint.get("encoder_state")
    projector_state = checkpoint.get("projector_state")
    if encoder_state and projector_state:
        return encoder_state, projector_state
    model_state = checkpoint.get("model_state") or {}
    if model_state:
        encoder_state = {
            key[len("encoder.") :]: value
            for key, value in model_state.items()
            if key.startswith("encoder.")
        }
        projector_state = {
            key[len("projector.") :]: value
            for key, value in model_state.items()
            if key.startswith("projector.")
        }
        if encoder_state and projector_state:
            return encoder_state, projector_state
    raise KeyError("Checkpoint is missing encoder/projector weights.")


def save_stage0_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    save_path: str,
    args,
    epoch: int,
    metrics: Dict[str, float],
    rank: int,
):
    if rank != 0:
        return
    underlying = get_model(model)
    payload = {
        "stage_name": "stage0_encoder_ssl",
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "model_state": underlying.state_dict(),
        "encoder_state": underlying.encoder.state_dict(),
        "model_config": sanitize_checkpoint_metadata(underlying.get_checkpoint_metadata()),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def load_stage0_checkpoint(
    *,
    model,
    checkpoint_path: str,
    device: str,
    optimizer=None,
    scheduler=None,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    get_model(model).load_state_dict(checkpoint["model_state"], strict=True)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def save_sp_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    metrics: Dict[str, float],
    save_path: str,
    args,
    stage_name: str,
    rank: int,
):
    if rank != 0:
        return
    underlying = get_model(model)
    checkpoint = {
        "model_config": sanitize_checkpoint_metadata(underlying.get_checkpoint_metadata()),
        "encoder_state": underlying.encoder.state_dict(),
        "projector_state": underlying.projector.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": float(metrics.get("loss_total", metrics.get("test_loss", float("inf")))),
        "metrics": metrics,
        "stage_name": stage_name,
        "args": vars(args),
    }
    underlying.save_lora_state_to_checkpoint(checkpoint)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def load_sp_checkpoint(
    *,
    model,
    checkpoint_path: str,
    device: str,
    optimizer=None,
    scheduler=None,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    underlying = get_model(model)
    encoder_state, projector_state = extract_sp_component_states_from_checkpoint(checkpoint)
    underlying.encoder.load_state_dict(encoder_state)
    underlying.projector.load_state_dict(projector_state)
    underlying.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint


def load_stage0_encoder_into_sp(model, checkpoint_path: str, device: str) -> Dict[str, List[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = sanitize_checkpoint_metadata(
        {
            "encoder_config": dict((checkpoint.get("model_config") or {}).get("encoder_config") or {}),
        }
    )["encoder_config"]
    current_config = sanitize_checkpoint_metadata(
        {
            "encoder_config": get_model(model).encoder.get_config(),
        }
    )["encoder_config"]
    if checkpoint_config != current_config:
        raise RuntimeError(
            "Stage0 encoder config mismatch while initializing stage1. "
            f"checkpoint={checkpoint_config!r}, current={current_config!r}"
        )
    current_state = get_model(model).encoder.state_dict()
    filtered_encoder_state = dict(checkpoint["encoder_state"])
    dropped_stage0_only_keys: List[str] = []
    for key in list(filtered_encoder_state.keys()):
        if key in current_state:
            continue
        if key.startswith("fused_pool_proj."):
            dropped_stage0_only_keys.append(key)
            filtered_encoder_state.pop(key)

    missing = get_model(model).encoder.load_state_dict(filtered_encoder_state, strict=True)
    return {
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "dropped_stage0_only_keys": dropped_stage0_only_keys,
    }


def build_stage0_optimizer(model: DualViewSSLModel, args):
    def collect(parameters: Sequence[torch.nn.Parameter], lr: float):
        params = [param for param in parameters if param.requires_grad]
        return {"params": params, "lr": lr} if params else None

    groups = []
    groups.append(collect(model.encoder.ts_backbone.parameters(), lr=args.stage0_lr_ts))
    excluded_modules = [model.encoder.ts_backbone]
    if args.stage0_train_vision and model.encoder.vision_encoder is not None:
        groups.append(collect(model.encoder.vision_encoder.vit.parameters(), lr=args.stage0_lr_vision))
        excluded_modules.append(model.encoder.vision_encoder.vit)
    excluded = {
        id(param)
        for module in excluded_modules
        for param in module.parameters()
    }
    head_params = [param for param in model.parameters() if param.requires_grad and id(param) not in excluded]
    groups.append(collect(head_params, lr=args.stage0_lr_heads))
    return AdamW([group for group in groups if group is not None], weight_decay=args.weight_decay)


def build_sp_model(args, device: str, rank: int, *, enable_lora: bool):
    init_kwargs = resolve_model_init_kwargs(args)
    if rank == 0:
        print("🔧 Building OpenTSLMSP model")
        print(f"   LLM: {init_kwargs['llm_id']}")
        print("   Encoder: newts_dual_branch")

    model = OpenTSLMSP(
        llm_id=init_kwargs["llm_id"],
        device=device,
        encoder_type=init_kwargs["encoder_type"],
        newts_dual_branch_config=init_kwargs["newts_dual_branch_config"],
        llm_attn_impl=args.llm_attn_impl,
    )

    if args.random_init_llm:
        if rank == 0:
            print("🎲 Replacing frozen base LLM with a random initialization")
        llm_config = model.llm.config
        random_llm = AutoModelForCausalLM.from_config(
            llm_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).to(device)
        model.llm = random_llm
        for param in model.llm.parameters():
            param.requires_grad = False

    if enable_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    return model


def stage_uses_lora(args, stage_name: str) -> bool:
    stage_name = normalize_stage_name(stage_name)
    if stage_name != "stage3_m4_caption":
        return False
    if args.use_lora_explicit:
        return bool(args.use_lora)
    return True


def configure_sp_trainable_parameters(model, args, stage_name: str, rank: int):
    stage_name = normalize_stage_name(stage_name)
    underlying = get_model(model)
    if stage_name == "stage1_tsqa_transfer":
        current_epoch = int(getattr(underlying, "_curriculum_stage_epoch", 1))
        freeze_encoder = current_epoch <= args.stage1_projector_only_epochs
        for param in underlying.encoder.parameters():
            param.requires_grad = not freeze_encoder
        last_state = getattr(underlying, "_stage1_encoder_frozen", None)
        if rank == 0 and last_state != freeze_encoder:
            if freeze_encoder:
                print(f"🧊 Stage1 encoder frozen for projector-only warm-start (epoch {current_epoch})")
            else:
                print(f"🔥 Stage1 encoder unfrozen for joint TSQA transfer (epoch {current_epoch})")
        underlying._stage1_encoder_frozen = freeze_encoder
        return
    if stage_name == "stage3_m4_caption" and not args.stage3_unfreeze_encoder:
        for param in underlying.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 Stage3 encoder frozen (default caption specialization path)")


def build_sp_optimizer(model, args, stage_name: str):
    stage_name = normalize_stage_name(stage_name)
    underlying = get_model(model)

    if stage_name == "stage1_tsqa_transfer":
        encoder_lr = args.stage1_lr_encoder
        projector_lr = args.stage1_lr_projector
    elif stage_name == STAGE2_CANONICAL_NAME:
        encoder_lr = args.stage2_lr_encoder
        projector_lr = args.stage2_lr_projector
    elif stage_name == "stage3_m4_caption":
        encoder_lr = args.stage3_lr_encoder
        projector_lr = args.stage3_lr_projector
    else:
        raise ValueError(f"Unsupported stage: {stage_name}")

    param_groups = []
    encoder_params = list(underlying.encoder.parameters())
    projector_params = list(underlying.projector.parameters())
    covered_param_ids = {id(param) for param in encoder_params + projector_params}

    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": projector_lr})

    extra_params = [
        param
        for param in underlying.parameters()
        if param.requires_grad and id(param) not in covered_param_ids
    ]
    if underlying.lora_enabled:
        lora_params = [param for param in underlying.get_lora_parameters() if param.requires_grad]
        covered_param_ids.update(id(param) for param in lora_params)
        extra_params = [param for param in extra_params if id(param) not in covered_param_ids]
        param_groups.append({"params": lora_params, "lr": args.lr_lora} if lora_params else None)
    param_groups.append({"params": extra_params, "lr": projector_lr} if extra_params else None)

    return AdamW([group for group in param_groups if group is not None], weight_decay=args.weight_decay)


def resolve_stage_num_epochs(args, stage_name: str) -> int:
    stage_name = normalize_stage_name(stage_name)
    if stage_name == "stage1_tsqa_transfer":
        return args.stage1_epochs
    if stage_name == STAGE2_CANONICAL_NAME:
        return args.stage2_epochs
    if stage_name == "stage3_m4_caption":
        return args.stage3_epochs
    raise ValueError(f"Unsupported stage: {stage_name}")


def resolve_stage_gradient_accumulation_steps(args, stage_name: str, world_size: int) -> int:
    stage_name = normalize_stage_name(stage_name)
    if stage_name != "stage1_tsqa_transfer":
        return args.gradient_accumulation_steps
    desired_effective_global_batch = args.batch_size * args.gradient_accumulation_steps
    return max(1, math.ceil(desired_effective_global_batch / max(world_size * args.batch_size, 1)))


def optimizer_step_count(num_batches: int, grad_accum_steps: int) -> int:
    return max(1, math.ceil(num_batches / grad_accum_steps))


def maybe_set_epoch(data_loader: DataLoader, epoch: int):
    dataset = getattr(data_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(data_loader, "batch_sampler", None) or getattr(data_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def train_one_epoch(
    *,
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    gradient_accumulation_steps: int,
    rank: int,
    epoch: int,
    num_epochs: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    for step, batch in enumerate(pbar, start=1):
        loss = model(batch)
        (loss / gradient_accumulation_steps).backward()

        should_step = step % gradient_accumulation_steps == 0 or step == len(train_loader)
        if should_step:
            if trainable_params:
                clip_grad_norm_(trainable_params, max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item())
        num_batches += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_loss_metrics(model, data_loader: DataLoader) -> Dict[str, float]:
    underlying = get_model(model)
    underlying.eval()
    totals: Dict[str, float] = {}
    num_batches = 0

    for batch in tqdm(data_loader, desc="Eval", leave=False):
        losses = underlying.compute_losses(batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())
        num_batches += 1

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_sp_stage(
    *,
    model,
    data_loader: DataLoader,
    metric_type: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    underlying = get_model(model)
    underlying.eval()
    totals: Dict[str, float] = {}
    num_batches = 0
    predictions_preview: List[str] = []
    labels_preview: List[str] = []
    all_predictions: List[str] = []
    all_golds: List[str] = []
    eos_token = underlying.get_eos_token()

    for batch in tqdm(data_loader, desc="Eval", leave=False):
        losses = underlying.compute_losses(batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())
        num_batches += 1

        if metric_type == "accuracy":
            predictions = underlying.generate(batch, max_new_tokens=max_new_tokens)
            golds = [sample["answer"].replace(eos_token, "").strip() for sample in batch]
            all_predictions.extend(predictions)
            all_golds.extend(golds)
            if len(predictions_preview) < 10:
                remaining = 10 - len(predictions_preview)
                predictions_preview.extend(predictions[:remaining])
                labels_preview.extend(golds[:remaining])

    metrics: Dict[str, Any] = {key: value / max(num_batches, 1) for key, value in totals.items()}
    if metric_type == "accuracy":
        metrics["accuracy"] = calculate_accuracy(all_predictions, all_golds)
        metrics["predictions_preview"] = predictions_preview
        metrics["labels_preview"] = labels_preview
    return metrics


def stage_metrics_improved(stage_name: str, current_metrics: Dict[str, Any], best_metrics: Optional[Dict[str, Any]]) -> bool:
    stage_name = normalize_stage_name(stage_name)
    if not best_metrics:
        return True
    selection = STAGE_SPECS[stage_name]["selection"]
    current_loss = float(current_metrics.get("loss_total", float("inf")))
    best_loss = float(best_metrics.get("loss_total", float("inf")))
    if selection == "accuracy":
        current_acc = float(current_metrics.get("accuracy", float("-inf")))
        best_acc = float(best_metrics.get("accuracy", float("-inf")))
        if current_acc > best_acc + 1e-6:
            return True
        if abs(current_acc - best_acc) <= 1e-6 and current_loss + 1e-4 < best_loss:
            return True
        return False
    return current_loss + 1e-4 < best_loss


def accuracy_metrics_improved(current_metrics: Dict[str, Any], best_metrics: Optional[Dict[str, Any]]) -> bool:
    if not best_metrics:
        return True
    current_acc = float(current_metrics.get("accuracy", float("-inf")))
    best_acc = float(best_metrics.get("accuracy", float("-inf")))
    current_loss = float(current_metrics.get("loss_total", float("inf")))
    best_loss = float(best_metrics.get("loss_total", float("inf")))
    if current_acc > best_acc + 1e-6:
        return True
    if abs(current_acc - best_acc) <= 1e-6 and current_loss + 1e-4 < best_loss:
        return True
    return False


def build_checkpoint_index_entry(
    *,
    stage_name: str,
    run_dir: str,
    best_checkpoint_path: str,
    metrics_path: str,
    model_config: Dict[str, Any],
    best_accuracy_checkpoint_path: Optional[str] = None,
) -> Dict[str, Any]:
    stage_name = normalize_stage_name(stage_name)
    entry = {
        "stage_name": stage_name,
        "description": STAGE_SPECS[stage_name]["description"],
        "best_checkpoint": os.path.relpath(best_checkpoint_path, run_dir),
        "metrics_file": os.path.relpath(metrics_path, run_dir),
        "model_config": model_config,
        "encoder_config": dict(model_config.get("encoder_config") or {}),
        "recommended_use": STAGE_SPECS[stage_name]["recommended_use"],
        "reference_downstream_checkpoint": stage_name == "stage1_tsqa_transfer",
    }
    if best_accuracy_checkpoint_path:
        entry["best_accuracy_checkpoint"] = os.path.relpath(best_accuracy_checkpoint_path, run_dir)
    alias_filename = STAGE_ALIAS_FILENAMES.get(stage_name)
    if alias_filename:
        entry["alias_checkpoint"] = alias_filename
    return entry


def resolve_stage_checkpoint_path(run_dir: str, stage_name: str) -> str:
    stage_name = normalize_stage_name(stage_name)
    if stage_name == "stage0_encoder_ssl":
        return os.path.join(run_dir, stage_name, "checkpoints", "best_encoder.pt")
    return os.path.join(run_dir, stage_name, "checkpoints", "best_model.pt")


def resolve_stage_accuracy_checkpoint_path(run_dir: str, stage_name: str) -> Optional[str]:
    stage_name = normalize_stage_name(stage_name)
    if stage_name != "stage1_tsqa_transfer":
        return None
    return os.path.join(run_dir, stage_name, "checkpoints", "best_accuracy.pt")


def initialize_sp_model_from_previous_stage(
    *,
    model,
    stage_name: str,
    run_dir: str,
    device: str,
    rank: int,
):
    stage_name = normalize_stage_name(stage_name)
    if stage_name == "stage1_tsqa_transfer":
        stage0_checkpoint = resolve_stage_checkpoint_path(run_dir, "stage0_encoder_ssl")
        if not os.path.exists(stage0_checkpoint):
            raise RuntimeError(
                f"{stage_name} requires a stage0 encoder checkpoint, but it was not found: {stage0_checkpoint}"
            )
        load_info = load_stage0_encoder_into_sp(model, stage0_checkpoint, device=device)
        if rank == 0:
            print(f"📂 Loaded stage0 encoder into stage1 (strict=True): {load_info}")
        return

    if stage_name == STAGE2_CANONICAL_NAME:
        previous_checkpoint = resolve_stage_checkpoint_path(run_dir, "stage1_tsqa_transfer")
    elif stage_name == "stage3_m4_caption":
        stage2_checkpoint = resolve_stage_checkpoint_path(run_dir, STAGE2_CANONICAL_NAME)
        stage1_checkpoint = resolve_stage_checkpoint_path(run_dir, "stage1_tsqa_transfer")
        previous_checkpoint = stage2_checkpoint if os.path.exists(stage2_checkpoint) else stage1_checkpoint
    else:
        raise ValueError(f"Unsupported stage: {stage_name}")

    if not os.path.exists(previous_checkpoint):
        raise RuntimeError(
            f"{stage_name} requires a previous-stage checkpoint, but it was not found: {previous_checkpoint}"
        )
    load_sp_checkpoint(model=model, checkpoint_path=previous_checkpoint, device=device)
    if rank == 0:
        print(f"📂 Loaded previous stage checkpoint from {previous_checkpoint}")


def train_stage0(
    *,
    args,
    run_dir: str,
    device: str,
    world_size: int,
    rank: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stage_name = "stage0_encoder_ssl"
    stage_dir = os.path.join(run_dir, stage_name)
    checkpoint_path = resolve_stage_checkpoint_path(run_dir, stage_name)
    history_path = os.path.join(stage_dir, "checkpoints", "history.jsonl")
    metrics_path = os.path.join(stage_dir, "results", "metrics.json")

    datasets = build_stage0_datasets(args)
    train_loader = create_simple_loader(
        datasets["train"],
        batch_size=max(1, args.batch_size * args.stage0_batch_multiplier),
        shuffle=True,
        world_size=world_size,
        rank=rank,
        collate_fn=collate_raw_series_batch,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = create_simple_loader(
        datasets["validation"],
        batch_size=max(1, args.eval_batch_size * args.stage0_batch_multiplier),
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate_raw_series_batch,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    test_loader = create_simple_loader(
        datasets["test"],
        batch_size=max(1, args.eval_batch_size * args.stage0_batch_multiplier),
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate_raw_series_batch,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    model = DualViewSSLModel(
        encoder_config=build_newts_dual_branch_config(args, for_stage0=True),
        device=device,
        train_vision=args.stage0_train_vision,
        mask_ratio=args.stage0_mask_ratio,
        jitter_std=args.aug_jitter_std,
        scaling_range=(args.aug_scaling_min, args.aug_scaling_max),
        time_mask_ratio=args.aug_time_mask_ratio,
        loss_w_ts_recon=args.stage0_w_ts_recon,
        loss_w_ts_vicreg=args.stage0_w_ts_vicreg,
        loss_w_vi_vicreg=args.stage0_w_vi_vicreg,
        loss_w_fuse_vicreg=args.stage0_w_fuse_vicreg,
    )
    if args.gradient_checkpointing:
        model.encoder.enable_gradient_checkpointing()
    model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[int(device.split(":")[-1])] if device.startswith("cuda:") else None,
        )

    optimizer = build_stage0_optimizer(get_model(model), args)
    total_steps = optimizer_step_count(len(train_loader), args.gradient_accumulation_steps) * args.stage0_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_metrics: Optional[Dict[str, Any]] = None
    start_epoch = 1
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_stage0_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if checkpoint is not None:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metrics = checkpoint.get("metrics") or None
            if rank == 0:
                print(f"📂 Resuming {stage_name} from epoch {checkpoint.get('epoch', 0)}")

    epochs_no_improve = 0
    if rank == 0:
        print("\n" + "=" * 72)
        print(f"🚀 Starting {stage_name}: {STAGE_SPECS[stage_name]['description']}")
        print(f"   epochs={args.stage0_epochs}")
        print(f"   batch_size={max(1, args.batch_size * args.stage0_batch_multiplier)}")
        print(f"   grad_accum={args.gradient_accumulation_steps}")
        print(f"   train_vision={args.stage0_train_vision}")
        used_weights = args.stage0_mix_weights if args.stage0_downstream_pool == "ucr_train_list" else args.stage0_mix_weights[:2]
        print(f"   train_mix_weights={used_weights}")
        print(f"   branch_dropout={args.stage0_branch_dropout:.2f}")
        print(
            "   loss_weights="
            f"ts_recon:{args.stage0_w_ts_recon:.2f},"
            f"ts_vicreg:{args.stage0_w_ts_vicreg:.2f},"
            f"vi_vicreg:{args.stage0_w_vi_vicreg:.2f},"
            f"fuse_vicreg:{args.stage0_w_fuse_vicreg:.2f}"
        )
        print("=" * 72)

    for epoch in range(start_epoch, args.stage0_epochs + 1):
        maybe_set_epoch(train_loader, epoch)
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            rank=rank,
            epoch=epoch,
            num_epochs=args.stage0_epochs,
        )

        val_metrics = evaluate_loss_metrics(model, val_loader) if rank == 0 else None
        val_metrics = broadcast_object_from_rank0(val_metrics, rank)

        if rank == 0:
            append_jsonl(history_path, {"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
            print(
                f"{stage_name} epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss_total']:.4f}"
            )

        if rank == 0 and stage_metrics_improved(stage_name, val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            epochs_no_improve = 0
            save_stage0_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_path=checkpoint_path,
                args=args,
                epoch=epoch,
                metrics=best_metrics,
                rank=rank,
            )
        elif rank == 0:
            epochs_no_improve += 1

        state = broadcast_object_from_rank0(
            {
                "best_metrics": best_metrics,
                "epochs_no_improve": epochs_no_improve,
                "stop": epochs_no_improve >= args.early_stop,
            },
            rank,
        )
        best_metrics = state["best_metrics"]
        epochs_no_improve = int(state["epochs_no_improve"])
        if state["stop"]:
            if rank == 0:
                print(f"⏹️ Early stopping triggered for {stage_name}")
            break

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"No checkpoint was produced for {stage_name}: {checkpoint_path}")

    load_stage0_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    if dist.is_initialized():
        dist.barrier()

    test_metrics = evaluate_loss_metrics(model, test_loader) if rank == 0 else None
    test_metrics = broadcast_object_from_rank0(test_metrics, rank)

    if rank == 0:
        encoder_config = sanitize_checkpoint_metadata(get_model(model).get_checkpoint_metadata())
        save_json(metrics_path, test_metrics)
        save_json(os.path.join(stage_dir, "encoder_config.json"), encoder_config)
    else:
        encoder_config = None
    encoder_config = broadcast_object_from_rank0(encoder_config, rank)

    if dist.is_initialized():
        dist.barrier()

    index_entry = build_checkpoint_index_entry(
        stage_name=stage_name,
        run_dir=run_dir,
        best_checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        model_config=encoder_config,
    )
    return test_metrics, index_entry


def train_sp_stage(
    *,
    args,
    stage_name: str,
    run_dir: str,
    device: str,
    world_size: int,
    rank: int,
    export_root_model_checkpoint: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    stage_name = normalize_stage_name(stage_name)
    stage_dir = os.path.join(run_dir, stage_name)
    checkpoint_path = resolve_stage_checkpoint_path(run_dir, stage_name)
    best_accuracy_checkpoint_path = resolve_stage_accuracy_checkpoint_path(run_dir, stage_name)
    history_path = os.path.join(stage_dir, "checkpoints", "history.jsonl")
    metrics_path = os.path.join(stage_dir, "results", "metrics.json")
    alias_path = os.path.join(run_dir, STAGE_ALIAS_FILENAMES[stage_name])

    enable_lora = stage_uses_lora(args, stage_name)
    model = build_sp_model(args, device=device, rank=rank, enable_lora=enable_lora)
    if stage_name == STAGE2_CANONICAL_NAME:
        model.enable_alignment_losses(
            loss_w_align=STAGE2_LOSS_W_ALIGN,
            loss_w_consistency=STAGE2_LOSS_W_CONSISTENCY,
        )
    get_model(model)._curriculum_stage_epoch = 1
    configure_sp_trainable_parameters(model, args, stage_name, rank)
    model.to(device)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[int(device.split(":")[-1])] if device.startswith("cuda:") else None,
        )

    eos_token = get_model(model).get_eos_token()
    train_dataset = create_stage_dataset(stage_name, "train", eos_token, args)
    val_dataset = create_stage_dataset(stage_name, "validation", eos_token, args)
    test_dataset = create_stage_dataset(stage_name, "test", eos_token, args)

    collate_patch_size = 1
    pad_mode = resolve_effective_pad_mode(args)
    use_length_bucket = True

    train_loader = create_sp_data_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=world_size,
        rank=rank,
        distribute_data=True,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    val_loader = create_sp_data_loader(
        dataset=val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=1,
        rank=0,
        distribute_data=False,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    test_loader = create_sp_data_loader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=1,
        rank=0,
        distribute_data=False,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )

    optimizer = build_sp_optimizer(model, args, stage_name)
    num_epochs = resolve_stage_num_epochs(args, stage_name)
    stage_grad_accum = resolve_stage_gradient_accumulation_steps(args, stage_name, world_size)
    total_steps = optimizer_step_count(len(train_loader), stage_grad_accum) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_metrics: Optional[Dict[str, Any]] = None
    best_accuracy_metrics: Optional[Dict[str, Any]] = None
    start_epoch = 1
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_sp_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if checkpoint is not None:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_metrics = checkpoint.get("metrics") or None
            if best_accuracy_checkpoint_path and os.path.exists(best_accuracy_checkpoint_path):
                best_accuracy_metrics = (
                    torch.load(best_accuracy_checkpoint_path, map_location=device, weights_only=False).get("metrics") or None
                )
            if rank == 0:
                print(f"📂 Resuming {stage_name} from epoch {checkpoint.get('epoch', 0)}")
    else:
        initialize_sp_model_from_previous_stage(
            model=model,
            stage_name=stage_name,
            run_dir=run_dir,
            device=device,
            rank=rank,
        )

    metric_type = "accuracy" if stage_name == "stage1_tsqa_transfer" else "loss"
    epochs_no_improve = 0
    if rank == 0:
        print("\n" + "=" * 72)
        print(f"🚀 Starting {stage_name}: {STAGE_SPECS[stage_name]['description']}")
        print(f"   epochs={num_epochs}")
        print(f"   batch_size={args.batch_size}")
        print(f"   grad_accum={stage_grad_accum}")
        print(f"   effective_global_batch={world_size * args.batch_size * stage_grad_accum}")
        print(f"   pad_mode={pad_mode}")
        print(f"   lora_enabled={enable_lora}")
        if stage_name == "stage1_tsqa_transfer":
            print(f"   projector_only_epochs={args.stage1_projector_only_epochs}")
        if stage_name == STAGE2_CANONICAL_NAME:
            print(f"   stage2_mix_weights={args.stage2_mix_weights}")
            print(
                f"   alignment_losses=align:{STAGE2_LOSS_W_ALIGN:.2f},"
                f"consistency:{STAGE2_LOSS_W_CONSISTENCY:.2f}"
            )
        print("   length_bucket_batching=True")
        print("=" * 72)

    for epoch in range(start_epoch, num_epochs + 1):
        get_model(model)._curriculum_stage_epoch = epoch
        configure_sp_trainable_parameters(model, args, stage_name, rank)
        maybe_set_epoch(train_loader, epoch)
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=stage_grad_accum,
            rank=rank,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        val_metrics = (
            evaluate_sp_stage(
                model=model,
                data_loader=val_loader,
                metric_type=metric_type,
                max_new_tokens=args.max_new_tokens,
            )
            if rank == 0
            else None
        )
        val_metrics = broadcast_object_from_rank0(val_metrics, rank)

        if rank == 0:
            append_jsonl(history_path, {"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
            print(
                f"{stage_name} epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss_total']:.4f}"
                + (
                    f" val_accuracy={val_metrics['accuracy']:.4f}"
                    if "accuracy" in val_metrics
                    else ""
                )
            )

        if rank == 0 and stage_metrics_improved(stage_name, val_metrics, best_metrics):
            best_metrics = dict(val_metrics)
            epochs_no_improve = 0
            save_sp_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                metrics=best_metrics,
                save_path=checkpoint_path,
                args=args,
                stage_name=stage_name,
                rank=rank,
            )
        elif rank == 0:
            epochs_no_improve += 1

        if (
            rank == 0
            and stage_name == "stage1_tsqa_transfer"
            and best_accuracy_checkpoint_path is not None
            and accuracy_metrics_improved(val_metrics, best_accuracy_metrics)
        ):
            best_accuracy_metrics = dict(val_metrics)
            save_sp_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                metrics=best_accuracy_metrics,
                save_path=best_accuracy_checkpoint_path,
                args=args,
                stage_name=stage_name,
                rank=rank,
            )

        state = broadcast_object_from_rank0(
            {
                "best_metrics": best_metrics,
                "best_accuracy_metrics": best_accuracy_metrics,
                "epochs_no_improve": epochs_no_improve,
                "stop": epochs_no_improve >= args.early_stop,
            },
            rank,
        )
        best_metrics = state["best_metrics"]
        best_accuracy_metrics = state["best_accuracy_metrics"]
        epochs_no_improve = int(state["epochs_no_improve"])
        if state["stop"]:
            if rank == 0:
                print(f"⏹️ Early stopping triggered for {stage_name}")
            break

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"No checkpoint was produced for {stage_name}: {checkpoint_path}")

    load_sp_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    if dist.is_initialized():
        dist.barrier()

    test_metrics = (
        evaluate_sp_stage(
            model=model,
            data_loader=test_loader,
            metric_type=metric_type,
            max_new_tokens=args.max_new_tokens,
        )
        if rank == 0
        else None
    )
    test_metrics = broadcast_object_from_rank0(test_metrics, rank)

    if rank == 0:
        save_json(metrics_path, test_metrics)
        export_stage_alias_checkpoint(model, alias_path, rank=rank)
        if export_root_model_checkpoint:
            export_final_model_checkpoint(
                model=model,
                export_path=os.path.join(run_dir, "model_checkpoint.pt"),
                random_init_llm=args.random_init_llm,
                rank=rank,
            )
        encoder_config = sanitize_checkpoint_metadata(get_model(model).get_checkpoint_metadata())
    else:
        encoder_config = None
    encoder_config = broadcast_object_from_rank0(encoder_config, rank)

    if dist.is_initialized():
        dist.barrier()

    index_entry = build_checkpoint_index_entry(
        stage_name=stage_name,
        run_dir=run_dir,
        best_checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        model_config=encoder_config,
        best_accuracy_checkpoint_path=(
            best_accuracy_checkpoint_path if best_accuracy_checkpoint_path and os.path.exists(best_accuracy_checkpoint_path) else None
        ),
    )
    return test_metrics, index_entry


def main():
    args = parse_args()
    local_rank, world_size, rank = setup_distributed()

    try:
        if world_size > 1:
            device = f"cuda:{local_rank}"
        elif args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        set_seed(args.seed + rank)
        warn_newts_vision_runtime_config(args, rank)

        run_name = args.run_name or default_run_name(args)
        run_dir = os.path.join(args.save_dir, sanitize_llm_id(args.llm_id), run_name)

        if rank == 0:
            print("=" * 72)
            print("Curriculum Pretraining V4")
            print("=" * 72)
            print(f"Time: {datetime.datetime.now()}")
            print(f"Device: {device}")
            print(f"LLM: {args.llm_id}")
            print("Encoder: newts_dual_branch")
            print(f"Stages: {args.stages}")
            print(f"Run dir: {run_dir}")
            print(f"Resume: {args.resume}")
            print(f"LoRA(stage3 default): {stage_uses_lora(args, 'stage3_m4_caption')}")
            resolved_vision_config = resolve_effective_newts_vision_config(args)
            print("Dynamic length: enabled")
            print(f"Pad mode: {resolve_effective_pad_mode(args)}")
            print(f"Vision 2D mode: {resolved_vision_config['vision_2d_mode']}")
            print(f"Effective ViT stride: {resolved_vision_config['effective_vit_stride']}")
            print(f"Effective ViT patch policy: {resolved_vision_config['effective_vit_patch_policy']}")
            print("=" * 72)

            os.makedirs(run_dir, exist_ok=True)
            save_launch_configs(run_dir, args, world_size)

        if dist.is_initialized():
            dist.barrier()

        results: Dict[str, Any] = {}
        checkpoint_index: Dict[str, Any] = {}

        if "stage0_encoder_ssl" in args.stages:
            stage_metrics, index_entry = train_stage0(
                args=args,
                run_dir=run_dir,
                device=device,
                world_size=world_size,
                rank=rank,
            )
            results["stage0_encoder_ssl"] = stage_metrics
            checkpoint_index["stage0_encoder_ssl"] = index_entry
            if rank == 0:
                save_json(os.path.join(run_dir, "checkpoint_index.json"), checkpoint_index)

        sp_stages = [stage_name for stage_name in args.stages if stage_name != "stage0_encoder_ssl"]
        last_sp_stage = sp_stages[-1] if sp_stages else None
        for stage_name in sp_stages:
            stage_metrics, index_entry = train_sp_stage(
                args=args,
                stage_name=stage_name,
                run_dir=run_dir,
                device=device,
                world_size=world_size,
                rank=rank,
                export_root_model_checkpoint=bool(args.export_model_checkpoint and stage_name == last_sp_stage),
            )
            results[stage_name] = stage_metrics
            checkpoint_index[stage_name] = index_entry
            if rank == 0:
                save_json(os.path.join(run_dir, "checkpoint_index.json"), checkpoint_index)

        if rank == 0:
            save_json(os.path.join(run_dir, "curriculum_results.json"), results)
            print("\n🎉 Done")
            print(f"Results saved under: {run_dir}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
