#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Reproduce curriculum pretraining stage1 + stage2 with configurable encoders.

This script is intended to validate whether a Stage2 SP checkpoint is consistent
with the first two curriculum stages:
1. stage1_mcq        (TSQA multiple-choice QA)
2. stage2_captioning (M4 caption generation)

Compared with `train_curriculum_pretrain.py`, this version:
- supports `transformer_cnn`, `tslanet`, and `newts_dual_branch`
- stores `model_config` in checkpoints so downstream pretrained classification
  scripts can recover encoder / LLM configuration automatically
- resumes per-stage training and automatically chains stage1 -> stage2
- exports a slim `model_checkpoint.pt` after the final stage by default

Example:
    uv run torchrun --nproc_per_node=2 scripts/train_curriculum_pretrain_stage12.py \
    --encoder_type newts_dual_branch \
    --branch_mode both \
    --batch_size 4 --eval_batch_size 4 --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --run_name second \
    --vit_layer_idx 4   --vit_num_hidden_layers 4 \
    --stages stage1_mcq,stage2_captioning \
    --resume

"""
import argparse
import datetime
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model_config import BATCH_SIZE, EARLY_STOP_PAT, ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


STAGE_ORDER = ["stage1_mcq", "stage2_captioning"]
STAGE_SPECS = {
    "stage1_mcq": {
        "default_epochs": 25,
        "default_lr_encoder": 2e-4,
        "default_lr_projector": 1e-4,
        "metric_type": "accuracy",
        "description": "TSQA multiple-choice QA",
    },
    "stage2_captioning": {
        "default_epochs": 20,
        "default_lr_encoder": 2e-4,
        "default_lr_projector": 1e-4,
        "metric_type": "loss",
        "description": "M4 caption generation",
    },
}


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None

    items: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    return items or None


def parse_stage_list(value: str) -> List[str]:
    stages = [stage.strip() for stage in value.split(",") if stage.strip()]
    if not stages:
        raise ValueError("At least one stage must be provided in --stages")

    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid stages: {STAGE_ORDER}")
    return stages


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
    if args.encoder_type == "transformer_cnn":
        return "transformer_cnn"
    if args.encoder_type == "tslanet":
        return f"tslanet_ps{args.tslanet_patch_size}"
    return f"newts_dual_branch_{args.branch_mode}_dynamic"


def get_dataset_class(stage_name: str):
    if stage_name == "stage1_mcq":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset

        return TSQADataset
    if stage_name == "stage2_captioning":
        from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset

        return M4QADataset
    raise ValueError(f"Unsupported stage: {stage_name}")


def parse_args(argv=None):
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Reproduce stage1/stage2 curriculum pretraining with configurable encoders"
    )

    parser.add_argument(
        "--stages",
        type=str,
        default="stage1_mcq,stage2_captioning",
        help="Comma-separated stages to run. Choices: stage1_mcq, stage2_captioning",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Optional subdirectory name for this run")
    parser.add_argument("--save_dir", type=str, default="results/curriculum_pretrain_stage12")
    parser.add_argument("--resume", action="store_true", help="Resume current stage from its saved checkpoint if present")

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="transformer_cnn",
        choices=["transformer_cnn", "tslanet", "newts_dual_branch"],
    )
    parser.add_argument("--encoder_pretrained", type=str, default=None, help="Optional external encoder checkpoint (mainly for TSLANet)")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--random_init_llm", action="store_true", help="Replace the frozen base LLM with a random initialization")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_encoder", action="store_true")

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA during stage1/2. Disabled by default to match the original curriculum.",
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr_lora", type=float, default=1e-4)

    parser.add_argument("--tslanet_patch_size", type=int, default=8)

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
    parser.add_argument("--vit_feature_mode", type=str, default="single", choices=["last", "single", "scalar_mix"])
    parser.add_argument("--vit_layer_idx", type=int, default=4)
    parser.add_argument("--vit_mix_layers", type=str, default=None)
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
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

    parser.add_argument("--stage1_epochs", type=int, default=STAGE_SPECS["stage1_mcq"]["default_epochs"])
    parser.add_argument("--stage2_epochs", type=int, default=STAGE_SPECS["stage2_captioning"]["default_epochs"])
    parser.add_argument("--stage1_lr_encoder", type=float, default=STAGE_SPECS["stage1_mcq"]["default_lr_encoder"])
    parser.add_argument("--stage1_lr_projector", type=float, default=STAGE_SPECS["stage1_mcq"]["default_lr_projector"])
    parser.add_argument("--stage2_lr_encoder", type=float, default=STAGE_SPECS["stage2_captioning"]["default_lr_encoder"])
    parser.add_argument("--stage2_lr_projector", type=float, default=STAGE_SPECS["stage2_captioning"]["default_lr_projector"])

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=EARLY_STOP_PAT)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "last", "repeat"])
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Generation length used for stage1 evaluation")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--export_model_checkpoint",
        dest="export_model_checkpoint",
        action="store_true",
        help="Export a slim OpenTSLM-compatible model_checkpoint.pt after the last stage",
    )
    parser.add_argument(
        "--no_export_model_checkpoint",
        dest="export_model_checkpoint",
        action="store_false",
        help="Disable final slim checkpoint export",
    )
    parser.set_defaults(export_model_checkpoint=True)

    args = parser.parse_args(argv)
    args.stages = parse_stage_list(args.stages)
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    args.context_length_explicit = cli_flag_was_provided(provided_argv, "--context_length")
    args.pad_mode_explicit = cli_flag_was_provided(provided_argv, "--pad_mode")
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

    if args.encoder_type != "newts_dual_branch":
        return

    if args.patch_length <= 0:
        raise ValueError("--patch_length must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
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
            raise ValueError("--vit_num_hidden_layers must be >= the selected feature layer depth")

    if not args.use_pma:
        return

    if args.aggregator_layers <= 0:
        raise ValueError("--aggregator_layers must be positive")
    if args.aggregator_num_heads <= 0:
        raise ValueError("--aggregator_num_heads must be positive")
    if args.aggregator_num_queries <= 0:
        raise ValueError("--aggregator_num_queries must be positive")
    if args.aggregator_fuse_layers < 0:
        raise ValueError("--aggregator_fuse_layers must be >= 0")
    if args.aggregator_hidden_size is not None and args.aggregator_hidden_size <= 0:
        raise ValueError("--aggregator_hidden_size must be positive when provided")
    if args.aggregator_ffn_dim is not None and args.aggregator_ffn_dim <= 0:
        raise ValueError("--aggregator_ffn_dim must be positive when provided")

    hidden_size = args.aggregator_hidden_size or ENCODER_OUTPUT_DIM
    if hidden_size % args.aggregator_num_heads != 0:
        raise ValueError("--aggregator_num_heads must evenly divide the PMA hidden size")


def setup_distributed() -> Tuple[int, int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, 0

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    torch.cuda.set_device(local_rank)
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


def build_tslanet_config(args) -> Dict[str, Any]:
    return {
        "patch_size": args.tslanet_patch_size,
        "output_dim": ENCODER_OUTPUT_DIM,
    }


def build_newts_dual_branch_config(args) -> Dict[str, Any]:
    return {
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
        "use_pma": args.use_pma,
        "aggregator_layers": args.aggregator_layers,
        "aggregator_hidden_size": args.aggregator_hidden_size,
        "aggregator_num_heads": args.aggregator_num_heads,
        "aggregator_ffn_dim": args.aggregator_ffn_dim,
        "aggregator_num_queries": args.aggregator_num_queries,
        "aggregator_query_mode": args.aggregator_query_mode,
        "aggregator_fusion_mode": args.aggregator_fusion_mode,
        "aggregator_gate_type": args.aggregator_gate_type,
        "aggregator_fuse_layers": args.aggregator_fuse_layers,
        "freeze_ts_backbone": args.freeze_ts_backbone,
        "freeze_vision_backbone": args.freeze_vision_backbone,
    }


def resolve_model_init_kwargs(args) -> Dict[str, Any]:
    init_kwargs: Dict[str, Any] = {
        "llm_id": args.llm_id,
        "encoder_type": args.encoder_type,
        "encoder_pretrained_path": args.encoder_pretrained,
        "tslanet_config": None,
        "newts_dual_branch_config": None,
    }
    if args.encoder_type == "tslanet":
        init_kwargs["tslanet_config"] = build_tslanet_config(args)
    elif args.encoder_type == "newts_dual_branch":
        init_kwargs["newts_dual_branch_config"] = build_newts_dual_branch_config(args)
    return init_kwargs


def resolve_collate_patch_size(args) -> int:
    if args.encoder_type == "tslanet":
        return args.tslanet_patch_size
    if args.encoder_type == "newts_dual_branch":
        return 1
    return PATCH_SIZE


def resolve_effective_pad_mode(args) -> str:
    if args.encoder_type == "newts_dual_branch" and not getattr(args, "pad_mode_explicit", False):
        return "last"
    return args.pad_mode


def resolve_base_eos_token(llm_id: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        raise RuntimeError(f"Tokenizer for {llm_id} has no EOS token")
    return tokenizer.eos_token


def warn_deprecated_newts_context_length(args, rank: int):
    if rank != 0:
        return
    if args.encoder_type != "newts_dual_branch":
        return
    if getattr(args, "context_length_explicit", False):
        print("⚠️ --context_length is deprecated for newts_dual_branch and is ignored in dynamic-length mode")


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
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be within [0, num_replicas)")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.bucket_size = max(self.batch_size, self.batch_size * int(bucket_size_multiplier))
        self.seed = int(seed)
        self.epoch = 0
        self.sample_lengths = [self._infer_sample_length(dataset[idx]) for idx in range(len(dataset))]

    @staticmethod
    def _infer_sample_length(sample: Dict[str, Any]) -> int:
        max_len = 0
        for ts in sample.get("time_series", []):
            max_len = max(max_len, int(torch.as_tensor(ts).numel()))
        if max_len <= 0:
            raise ValueError("Encountered a sample without a positive time-series length")
        return max_len

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


def create_data_loader(
    *,
    dataset_class,
    split: str,
    eos_token: str,
    batch_size: int,
    shuffle: bool,
    collate_patch_size: int,
    pad_mode: str,
    world_size: int,
    rank: int,
    distribute_data: bool,
    use_length_bucket: bool,
    seed: int,
) -> DataLoader:
    dataset = dataset_class(split, EOS_TOKEN=eos_token)
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
    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate,
    )


def build_model(args, device: str, rank: int):
    init_kwargs = resolve_model_init_kwargs(args)
    if rank == 0:
        print("🔧 Building OpenTSLMSP model")
        print(f"   LLM: {init_kwargs['llm_id']}")
        print(f"   Encoder: {init_kwargs['encoder_type']}")

    model = OpenTSLMSP(
        llm_id=init_kwargs["llm_id"],
        device=device,
        encoder_type=init_kwargs["encoder_type"],
        encoder_pretrained_path=init_kwargs["encoder_pretrained_path"],
        tslanet_config=init_kwargs["tslanet_config"],
        newts_dual_branch_config=init_kwargs["newts_dual_branch_config"],
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

    if args.use_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 Encoder parameters frozen")

    return model


def stage_config(stage_name: str, args) -> Dict[str, Any]:
    if stage_name == "stage1_mcq":
        return {
            "dataset_class": get_dataset_class(stage_name),
            "epochs": args.stage1_epochs,
            "lr_encoder": args.stage1_lr_encoder,
            "lr_projector": args.stage1_lr_projector,
            "metric_type": "accuracy",
        }
    if stage_name == "stage2_captioning":
        return {
            "dataset_class": get_dataset_class(stage_name),
            "epochs": args.stage2_epochs,
            "lr_encoder": args.stage2_lr_encoder,
            "lr_projector": args.stage2_lr_projector,
            "metric_type": "loss",
        }
    raise ValueError(f"Unsupported stage: {stage_name}")


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def effective_config_snapshot(args, world_size: int) -> Dict[str, Any]:
    snapshot = dict(vars(args))
    snapshot["effective_pad_mode"] = resolve_effective_pad_mode(args)
    snapshot["world_size"] = int(world_size)
    snapshot["effective_global_batch_size"] = (
        int(world_size) * int(args.batch_size) * int(args.gradient_accumulation_steps)
    )
    return snapshot


def save_launch_configs(run_dir: str, args, world_size: int):
    latest_config = dict(vars(args))
    effective_config = effective_config_snapshot(args, world_size)
    save_json(os.path.join(run_dir, "config.json"), latest_config)
    save_json(os.path.join(run_dir, "effective_config.json"), effective_config)

    launch_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_dir = os.path.join(run_dir, "launch_history")
    save_json(os.path.join(launch_dir, f"{launch_stamp}.json"), effective_config)


def _normalize_comparable_arg_value(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def collect_checkpoint_arg_mismatches(saved_args: Optional[Dict[str, Any]], current_args) -> List[Tuple[str, Any, Any]]:
    if not saved_args:
        return []

    current = vars(current_args)
    keys_to_compare = [
        "stages",
        "resume",
        "llm_id",
        "encoder_type",
        "encoder_pretrained",
        "random_init_llm",
        "gradient_checkpointing",
        "freeze_encoder",
        "use_lora",
        "lora_r",
        "lora_alpha",
        "lr_lora",
        "tslanet_patch_size",
        "branch_mode",
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
        "use_pma",
        "aggregator_layers",
        "aggregator_hidden_size",
        "aggregator_num_heads",
        "aggregator_ffn_dim",
        "aggregator_num_queries",
        "aggregator_query_mode",
        "aggregator_fusion_mode",
        "aggregator_gate_type",
        "aggregator_fuse_layers",
        "freeze_ts_backbone",
        "freeze_vision_backbone",
        "stage1_epochs",
        "stage2_epochs",
        "stage1_lr_encoder",
        "stage1_lr_projector",
        "stage2_lr_encoder",
        "stage2_lr_projector",
        "batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "grad_clip",
        "warmup_ratio",
        "early_stop",
        "pad_mode",
        "pad_mode_explicit",
        "max_new_tokens",
        "seed",
    ]
    mismatches: List[Tuple[str, Any, Any]] = []
    for key in keys_to_compare:
        saved_value = _normalize_comparable_arg_value(saved_args.get(key))
        current_value = _normalize_comparable_arg_value(current.get(key))
        if saved_value != current_value:
            mismatches.append((key, saved_value, current_value))
    return mismatches


def warn_on_checkpoint_arg_mismatch(
    *,
    checkpoint_args: Optional[Dict[str, Any]],
    current_args,
    context: str,
    rank: int,
):
    if rank != 0:
        return

    mismatches = collect_checkpoint_arg_mismatches(checkpoint_args, current_args)
    if not mismatches:
        return

    print(f"⚠️ Checkpoint args mismatch while {context}")
    for key, saved_value, current_value in mismatches[:12]:
        print(f"   {key}: checkpoint={saved_value!r}, current={current_value!r}")
    if len(mismatches) > 12:
        print(f"   ... and {len(mismatches) - 12} more differences")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    train_loss: float,
    val_loss: float,
    save_path: str,
    args,
    stage_name: str,
    rank: int,
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
        "val_loss": val_loss,
        "stage_name": stage_name,
        "args": vars(args),
    }
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: str,
    optimizer=None,
    scheduler=None,
) -> Optional[Dict[str, Any]]:
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    underlying_model = get_model(model)
    underlying_model.encoder.load_state_dict(checkpoint["encoder_state"])
    underlying_model.projector.load_state_dict(checkpoint["projector_state"])
    underlying_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def save_loss_history(path: str, epoch: int, train_loss: float, val_loss: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Epoch\tTrain_Loss\tVal_Loss\n")
            f.write("-" * 40 + "\n")
        f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")


def evaluate_loss(model, data_loader: DataLoader) -> float:
    underlying_model = get_model(model)
    underlying_model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            total_loss += underlying_model.compute_loss(batch).item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_stage(
    model,
    data_loader: DataLoader,
    metric_type: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    underlying_model = get_model(model)
    underlying_model.eval()

    total_loss = 0.0
    num_batches = 0
    predictions_preview: List[str] = []
    labels_preview: List[str] = []
    all_predictions: List[str] = []
    all_golds: List[str] = []
    eos_token = underlying_model.get_eos_token()

    for batch in tqdm(data_loader, desc="Testing", leave=False):
        total_loss += underlying_model.compute_loss(batch).item()
        num_batches += 1

        if metric_type == "accuracy":
            predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
            golds = [sample["answer"].replace(eos_token, "").strip() for sample in batch]
            all_predictions.extend(predictions)
            all_golds.extend(golds)
            if len(predictions_preview) < 10:
                remaining = 10 - len(predictions_preview)
                predictions_preview.extend(predictions[:remaining])
                labels_preview.extend(golds[:remaining])

    metrics: Dict[str, Any] = {"test_loss": total_loss / max(num_batches, 1)}
    if metric_type == "accuracy":
        metrics["accuracy"] = calculate_accuracy(all_predictions, all_golds)
        metrics["predictions_preview"] = predictions_preview
        metrics["labels_preview"] = labels_preview
    return metrics


def optimizer_step_count(num_batches: int, grad_accum_steps: int) -> int:
    return max(1, math.ceil(num_batches / grad_accum_steps))


def train_one_epoch(
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

        running_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return running_loss / max(num_batches, 1)


def load_previous_stage_if_needed(
    model,
    stage_name: str,
    run_dir: str,
    device: str,
    rank: int,
    already_completed: set[str],
    current_args,
):
    stage_index = STAGE_ORDER.index(stage_name)
    if stage_index == 0:
        return

    previous_stage = STAGE_ORDER[stage_index - 1]
    if previous_stage in already_completed:
        return

    checkpoint_path = os.path.join(run_dir, previous_stage, "checkpoints", "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"{stage_name} requires the previous stage checkpoint, but it was not found: {checkpoint_path}"
        )

    checkpoint = load_checkpoint(model, checkpoint_path, device=device)
    if checkpoint is None:
        raise RuntimeError(f"Failed to load previous stage checkpoint: {checkpoint_path}")

    warn_on_checkpoint_arg_mismatch(
        checkpoint_args=checkpoint.get("args"),
        current_args=current_args,
        context=f"loading previous stage '{previous_stage}' for '{stage_name}'",
        rank=rank,
    )

    if rank == 0:
        print(f"📂 Loaded previous stage checkpoint from {checkpoint_path}")


def train_stage(
    *,
    stage_name: str,
    model,
    args,
    run_dir: str,
    device: str,
    world_size: int,
    rank: int,
) -> Dict[str, Any]:
    cfg = stage_config(stage_name, args)
    dataset_class = cfg["dataset_class"]
    num_epochs = cfg["epochs"]
    lr_encoder = cfg["lr_encoder"]
    lr_projector = cfg["lr_projector"]
    metric_type = cfg["metric_type"]
    collate_patch_size = resolve_collate_patch_size(args)
    pad_mode = resolve_effective_pad_mode(args)
    use_length_bucket = args.encoder_type == "newts_dual_branch"

    if rank == 0:
        print("\n" + "=" * 72)
        print(f"🚀 Starting {stage_name}: {STAGE_SPECS[stage_name]['description']}")
        print(f"   epochs={num_epochs}")
        print(f"   encoder_lr={lr_encoder:.2e}")
        print(f"   projector_lr={lr_projector:.2e}")
        print(f"   batch_size={args.batch_size}")
        print(f"   grad_accum={args.gradient_accumulation_steps}")
        print(f"   collate_patch_size={collate_patch_size}")
        print(f"   pad_mode={pad_mode}")
        if use_length_bucket:
            print("   length_bucket_batching=True")
        print("=" * 72)

    eos_token = get_model(model).get_eos_token()
    train_loader = create_data_loader(
        dataset_class=dataset_class,
        split="train",
        eos_token=eos_token,
        batch_size=args.batch_size,
        shuffle=True,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=world_size,
        rank=rank,
        distribute_data=True,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
    )
    val_loader = create_data_loader(
        dataset_class=dataset_class,
        split="validation",
        eos_token=eos_token,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=1,
        rank=0,
        distribute_data=False,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
    )
    test_loader = create_data_loader(
        dataset_class=dataset_class,
        split="test",
        eos_token=eos_token,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_patch_size=collate_patch_size,
        pad_mode=pad_mode,
        world_size=1,
        rank=0,
        distribute_data=False,
        use_length_bucket=use_length_bucket,
        seed=args.seed,
    )

    underlying_model = get_model(model)
    param_groups = []
    if not args.freeze_encoder:
        param_groups.append({"params": underlying_model.encoder.parameters(), "lr": lr_encoder})
    param_groups.append({"params": underlying_model.projector.parameters(), "lr": lr_projector})
    if args.use_lora:
        lora_params = underlying_model.get_lora_parameters()
        if lora_params:
            param_groups.append({"params": lora_params, "lr": args.lr_lora})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = optimizer_step_count(len(train_loader), args.gradient_accumulation_steps) * num_epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if rank == 0:
        print(f"📈 Total optimizer steps: {total_steps}")
        print(f"🔥 Warmup steps: {warmup_steps}")

    stage_dir = os.path.join(run_dir, stage_name)
    checkpoint_path = os.path.join(stage_dir, "checkpoints", "best_model.pt")
    loss_history_path = os.path.join(stage_dir, "checkpoints", "loss_history.txt")
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if checkpoint is not None:
            warn_on_checkpoint_arg_mismatch(
                checkpoint_args=checkpoint.get("args"),
                current_args=args,
                context=f"resuming stage '{stage_name}'",
                rank=rank,
            )
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val_loss = float(checkpoint.get("val_loss", float("inf")))
            if rank == 0:
                print(
                    f"📂 Resuming {stage_name} from epoch {checkpoint.get('epoch', 0)} "
                    f"(best val_loss={best_val_loss:.4f})"
                )

    epochs_no_improve = 0
    if start_epoch <= num_epochs:
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_sampler = getattr(train_loader, "batch_sampler", None) or getattr(train_loader, "sampler", None)
            if hasattr(epoch_sampler, "set_epoch"):
                epoch_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_clip=args.grad_clip,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                rank=rank,
                epoch=epoch,
                num_epochs=num_epochs,
            )

            val_loss = evaluate_loss(model, val_loader)

            if rank == 0:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                save_loss_history(loss_history_path, epoch, train_loss, val_loss)

            if rank == 0:
                if val_loss + 1e-4 < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        save_path=checkpoint_path,
                        args=args,
                        stage_name=stage_name,
                        rank=rank,
                    )
                    print("✔️ New best checkpoint saved")
                else:
                    epochs_no_improve += 1
                    print(f"   No improvement for {epochs_no_improve}/{args.early_stop} epochs")

            state = {
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
                "stop": epochs_no_improve >= args.early_stop,
            }
            state = broadcast_object_from_rank0(state, rank)
            best_val_loss = float(state["best_val_loss"])
            epochs_no_improve = int(state["epochs_no_improve"])

            if state["stop"]:
                if rank == 0:
                    print(f"⏹️ Early stopping triggered for {stage_name}")
                break
    elif rank == 0:
        print(f"⏭️ Skipping training for {stage_name}: checkpoint epoch already reached the configured num_epochs")

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"No checkpoint was produced for {stage_name}: {checkpoint_path}")

    load_checkpoint(model, checkpoint_path, device=device)
    if dist.is_initialized():
        dist.barrier()

    metrics: Dict[str, Any] = {}
    if rank == 0:
        print(f"📊 Evaluating {stage_name} test split")
        metrics = evaluate_stage(
            model=model,
            data_loader=test_loader,
            metric_type=metric_type,
            max_new_tokens=args.max_new_tokens,
        )
        metrics["stage_name"] = stage_name
        metrics["checkpoint"] = checkpoint_path
        save_json(os.path.join(stage_dir, "results", "metrics.json"), metrics)
        print(f"✅ {stage_name} finished")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")

    metrics = broadcast_object_from_rank0(metrics, rank)
    if dist.is_initialized():
        dist.barrier()
    return metrics


def export_final_model_checkpoint(model, export_path: str, random_init_llm: bool, rank: int):
    if rank != 0:
        return
    if random_init_llm:
        print("⚠️ Skipping slim model_checkpoint export because --random_init_llm was used")
        return

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    get_model(model).store_to_file(export_path)
    print(f"📦 Exported OpenTSLM-compatible checkpoint to: {export_path}")


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
        warn_deprecated_newts_context_length(args, rank)

        run_name = args.run_name or default_run_name(args)
        run_dir = os.path.join(args.save_dir, sanitize_llm_id(args.llm_id), run_name)

        if rank == 0:
            print("=" * 72)
            print("Curriculum Pretraining Stage1+2")
            print("=" * 72)
            print(f"Time: {datetime.datetime.now()}")
            print(f"Device: {device}")
            print(f"LLM: {args.llm_id}")
            print(f"Encoder: {args.encoder_type}")
            print(f"Stages: {args.stages}")
            print(f"Run dir: {run_dir}")
            print(f"Resume: {args.resume}")
            print(f"LoRA: {args.use_lora}")
            if args.encoder_type == "newts_dual_branch":
                print("Dynamic length: enabled")
                print(f"Pad mode: {resolve_effective_pad_mode(args)}")
            print("=" * 72)

            os.makedirs(run_dir, exist_ok=True)
            save_launch_configs(run_dir, args, world_size)

        if dist.is_initialized():
            dist.barrier()

        model = build_model(args, device=device, rank=rank)
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
            if rank == 0:
                print(f"✅ Wrapped model with DDP (world_size={world_size})")

        results: Dict[str, Any] = {}
        completed_stages: set[str] = set()

        for stage_name in args.stages:
            load_previous_stage_if_needed(
                model=model,
                stage_name=stage_name,
                run_dir=run_dir,
                device=device,
                rank=rank,
                already_completed=completed_stages,
                current_args=args,
            )
            stage_results = train_stage(
                stage_name=stage_name,
                model=model,
                args=args,
                run_dir=run_dir,
                device=device,
                world_size=world_size,
                rank=rank,
            )
            results[stage_name] = stage_results
            completed_stages.add(stage_name)

        if rank == 0:
            save_json(os.path.join(run_dir, "curriculum_results.json"), results)

        if args.export_model_checkpoint and args.stages:
            export_final_model_checkpoint(
                model=model,
                export_path=os.path.join(run_dir, "model_checkpoint.pt"),
                random_init_llm=args.random_init_llm,
                rank=rank,
            )

        if rank == 0:
            print("\n🎉 Done")
            print(f"Results saved under: {run_dir}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
