#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Multiview curriculum pretraining for UCR-oriented transfer.

Stages:
1. stage0_dual_view_ssl
2. stage1_semantic_alignment
3. stage2_instruction_tuning
"""

import argparse
import contextlib
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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.multiview_pretrain import (
    AlignmentTargetDataset,
    MixedPretrainDataset,
    RawSeriesDataset,
    SyntheticAttributeDataset,
    build_stage12_aligned_datasets,
    load_m4_raw_records,
    load_tsqa_raw_records,
    load_ucr_train_raw_records,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


STAGE_ORDER = [
    "stage0_dual_view_ssl",
    "stage1_semantic_alignment",
    "stage2_instruction_tuning",
]

STAGE_SPECS = {
    "stage0_dual_view_ssl": {
        "epochs": 10,
        "description": "Dual-view self-supervised encoder pretraining",
    },
    "stage1_semantic_alignment": {
        "epochs": 25,
        "description": "Semantic alignment with TSQA + M4 + synthetic attributes",
    },
    "stage2_instruction_tuning": {
        "epochs": 20,
        "description": "Instruction tuning with auxiliary alignment losses",
    },
}


def parse_stage_list(value: str) -> List[str]:
    stages = [stage.strip() for stage in value.split(",") if stage.strip()]
    if not stages:
        raise ValueError("At least one stage must be provided")
    unknown = [stage for stage in stages if stage not in STAGE_ORDER]
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}; valid={STAGE_ORDER}")
    return stages


def sanitize_llm_id(llm_id: str) -> str:
    name = llm_id.split("/")[-1] if llm_id else "unknown_llm"
    name = name.replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def broadcast_object_from_rank0(obj, rank: int):
    if not dist.is_initialized():
        return obj
    holder = [obj if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def save_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_precision(device: str, precision: str) -> Tuple[str, Optional[torch.dtype]]:
    if not device.startswith("cuda"):
        return "fp32", None
    precision = precision.lower()
    if precision == "auto":
        if torch.cuda.is_bf16_supported():
            return "bf16", torch.bfloat16
        return "fp16", torch.float16
    if precision == "bf16":
        return "bf16", torch.bfloat16
    if precision == "fp16":
        return "fp16", torch.float16
    if precision == "fp32":
        return "fp32", None
    raise ValueError(f"Unsupported precision: {precision}")


def autocast_context(device: str, amp_dtype: Optional[torch.dtype]):
    if not device.startswith("cuda") or amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def build_newts_config(args, *, for_stage0: bool) -> Dict[str, Any]:
    return {
        "output_dim": 128,
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
        "vit_feature_mode": "single",
        "vit_layer_idx": 4,
        "vit_mix_layers": None,
        "vit_patch_size": args.vit_patch_size,
        "vit_stride": args.vit_stride,
        "vision_2d_mode": args.vision_2d_mode,
        "vit_truncate_to_feature_layer": True,
        "vit_num_hidden_layers": 4,
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
        "enable_modality_embeddings": False if for_stage0 else args.enable_modality_embeddings,
        "branch_dropout": 0.0 if for_stage0 else args.branch_dropout,
        "vision_train_mode": args.vision_train_mode,
        "vision_topk_blocks": args.vision_topk_blocks,
        "freeze_ts_backbone": False,
        "freeze_vision_backbone": False if not for_stage0 else False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Multiview curriculum pretraining")
    parser.add_argument(
        "--stages",
        type=str,
        default="stage0_dual_view_ssl,stage1_semantic_alignment,stage2_instruction_tuning",
    )
    parser.add_argument("--run_name", type=str, default="multiview_single4")
    parser.add_argument("--save_dir", type=str, default="results/curriculum_pretrain_multiview")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_hidden_layers", type=int, default=3)
    parser.add_argument("--ffn_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.1)
    parser.add_argument("--vision_2d_mode", type=str, default="tivit_sqrt_overlap")
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--projector_dropout", type=float, default=0.1)

    parser.add_argument("--use_pma", action="store_true")
    parser.add_argument("--aggregator_layers", type=int, default=2)
    parser.add_argument("--aggregator_hidden_size", type=int, default=None)
    parser.add_argument("--aggregator_num_heads", type=int, default=8)
    parser.add_argument("--aggregator_ffn_dim", type=int, default=None)
    parser.add_argument("--aggregator_num_queries", type=int, default=4)
    parser.add_argument("--aggregator_query_mode", type=str, default="separate", choices=["shared", "separate"])
    parser.add_argument("--aggregator_fusion_mode", type=str, default="concat_linear", choices=["gated_sum", "concat_linear"])
    parser.add_argument("--aggregator_gate_type", type=str, default="dynamic", choices=["scalar", "slot", "dynamic"])
    parser.add_argument("--aggregator_fuse_layers", type=int, default=1)

    parser.add_argument("--vision_train_mode", type=str, default="topk", choices=["none", "topk", "all"])
    parser.add_argument("--vision_topk_blocks", type=int, default=4)
    parser.add_argument("--enable_modality_embeddings", dest="enable_modality_embeddings", action="store_true")
    parser.add_argument("--disable_modality_embeddings", dest="enable_modality_embeddings", action="store_false")
    parser.add_argument("--branch_dropout", type=float, default=0.15)
    parser.add_argument("--runtime_branch_mode", type=str, default="both", choices=["both", "ts_only", "vision_only"])
    parser.add_argument("--use_alignment_losses", dest="use_alignment_losses", action="store_true")
    parser.add_argument("--disable_alignment_losses", dest="use_alignment_losses", action="store_false")
    parser.add_argument("--loss_w_align", type=float, default=0.2)
    parser.add_argument("--loss_w_consistency", type=float, default=0.1)
    parser.add_argument("--stage2_loss_w_align", type=float, default=0.1)
    parser.add_argument("--stage2_loss_w_consistency", type=float, default=0.05)

    parser.add_argument("--stage0_epochs", type=int, default=STAGE_SPECS["stage0_dual_view_ssl"]["epochs"])
    parser.add_argument("--stage1_epochs", type=int, default=STAGE_SPECS["stage1_semantic_alignment"]["epochs"])
    parser.add_argument("--stage2_epochs", type=int, default=STAGE_SPECS["stage2_instruction_tuning"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--early_stop", type=int, default=5)

    parser.add_argument("--stage0_lr_ts", type=float, default=2e-4)
    parser.add_argument("--stage0_lr_vision", type=float, default=5e-5)
    parser.add_argument("--stage0_lr_heads", type=float, default=1e-4)
    parser.add_argument("--stage1_lr_ts", type=float, default=1e-4)
    parser.add_argument("--stage1_lr_vision", type=float, default=5e-5)
    parser.add_argument("--stage1_lr_other", type=float, default=1e-4)
    parser.add_argument("--stage2_lr_ts", type=float, default=5e-5)
    parser.add_argument("--stage2_lr_vision", type=float, default=2e-5)
    parser.add_argument("--stage2_lr_other", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    parser.add_argument("--stage0_mask_ratio", type=float, default=0.4)
    parser.add_argument("--stage0_downstream_pool", type=str, default="none", choices=["none", "ucr_train_list"])
    parser.add_argument("--ucr_data_path", type=str, default="./data")
    parser.add_argument("--ucr_train_list_path", type=str, default=None)

    parser.add_argument("--aug_jitter_std", type=float, default=0.02)
    parser.add_argument("--aug_scaling_min", type=float, default=0.9)
    parser.add_argument("--aug_scaling_max", type=float, default=1.1)
    parser.add_argument("--aug_time_mask_ratio", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--export_model_checkpoint", action="store_true", default=True)
    parser.set_defaults(enable_modality_embeddings=True, use_alignment_losses=True)
    args = parser.parse_args(argv)
    args.stages = parse_stage_list(args.stages)
    return args


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


class DualViewSSLModel(nn.Module):
    def __init__(
        self,
        *,
        encoder_config: Dict[str, Any],
        device: str,
        mask_ratio: float,
        jitter_std: float,
        scaling_range: Tuple[float, float],
        time_mask_ratio: float,
    ):
        super().__init__()
        self.device = device
        self.mask_ratio = float(mask_ratio)
        self.jitter_std = float(jitter_std)
        self.scaling_range = scaling_range
        self.time_mask_ratio = float(time_mask_ratio)

        self.encoder = NewTSDualBranchEncoder(**encoder_config, device=device).to(device)
        self.ts_recon_head = nn.Linear(self.encoder.output_dim, self.encoder.patch_length).to(device)
        self.ts_align_head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, 256),
        ).to(device)
        self.vision_align_head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, 256),
        ).to(device)

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.compute_losses(batch)["loss_total"]

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
    def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        logits = torch.matmul(z_a, z_b.transpose(0, 1)) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.transpose(0, 1), targets))

    def compute_losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch["series"].to(self.device, non_blocking=True)
        base_outputs = self.encoder(x, return_intermediates=True, runtime_branch_mode="both")
        masked_outputs = self.encoder(self._apply_mask(x), return_intermediates=True, runtime_branch_mode="both")
        aug_outputs = self.encoder(self._apply_augment(x), return_intermediates=True, runtime_branch_mode="both")

        target_patches = self.encoder.ts_backbone._extract_patches(x)
        pred_patches = self.ts_recon_head(masked_outputs["ts_tokens"].float()).to(target_patches.dtype)
        loss_recon = F.mse_loss(pred_patches, target_patches)

        z_ts = F.normalize(self.ts_align_head(base_outputs["pooled_ts"].float()), dim=-1)
        z_vi = F.normalize(self.vision_align_head(base_outputs["pooled_vision"].float()), dim=-1)
        loss_info = self._info_nce(z_ts, z_vi)

        z_ts_aug = F.normalize(self.ts_align_head(aug_outputs["pooled_ts"].float()), dim=-1)
        z_vi_aug = F.normalize(self.vision_align_head(aug_outputs["pooled_vision"].float()), dim=-1)
        loss_aug = 0.5 * (
            (1.0 - F.cosine_similarity(z_ts, z_ts_aug, dim=-1)).mean()
            + (1.0 - F.cosine_similarity(z_vi, z_vi_aug, dim=-1)).mean()
        )

        loss_total = loss_recon + 0.5 * loss_info + 0.1 * loss_aug
        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_info": loss_info,
            "loss_aug": loss_aug,
        }

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "encoder_type": "newts_dual_branch",
            "encoder_config": self.encoder.get_config(),
        }


def create_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    world_size: int,
    rank: int,
    collate_fn,
) -> DataLoader:
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
    )


def build_stage0_datasets(args) -> Dict[str, Dataset]:
    train_sets: List[Dataset] = [
        RawSeriesDataset(load_tsqa_raw_records("train")),
        RawSeriesDataset(load_m4_raw_records("train")),
    ]
    if args.stage0_downstream_pool == "ucr_train_list":
        train_sets.append(
            RawSeriesDataset(
                load_ucr_train_raw_records(
                    raw_data_path=args.ucr_data_path,
                    dataset_list_path=args.ucr_train_list_path,
                )
            )
        )
    train_dataset = MixedPretrainDataset(train_sets, [1] * len(train_sets), seed=args.seed)
    val_dataset = MixedPretrainDataset(
        [
            RawSeriesDataset(load_tsqa_raw_records("validation")),
            RawSeriesDataset(load_m4_raw_records("validation")),
        ],
        [1, 1],
        seed=args.seed,
    )
    test_dataset = MixedPretrainDataset(
        [
            RawSeriesDataset(load_tsqa_raw_records("test")),
            RawSeriesDataset(load_m4_raw_records("test")),
        ],
        [1, 1],
        seed=args.seed,
    )
    return {"train": train_dataset, "validation": val_dataset, "test": test_dataset}


def build_stage12_datasets(args, *, split: str, eos_token: str, stage_name: str) -> Dataset:
    aligned = build_stage12_aligned_datasets(split=split, eos_token=eos_token, seed=args.seed)
    if stage_name == "stage1_semantic_alignment":
        weights = [1, 1, 1]
    else:
        weights = [3, 4, 3]  # TSQA, M4, Synthetic
    return MixedPretrainDataset(
        [aligned["tsqa"], aligned["m4"], aligned["synthetic"]],
        weights,
        seed=args.seed,
    )


def collect_param_group(
    parameters: Sequence[torch.nn.Parameter],
    *,
    lr: float,
) -> Optional[Dict[str, Any]]:
    params = [param for param in parameters if param.requires_grad]
    if not params:
        return None
    return {"params": params, "lr": lr}


def build_stage0_optimizer(model: DualViewSSLModel, args):
    groups = []
    groups.append(collect_param_group(model.encoder.ts_backbone.parameters(), lr=args.stage0_lr_ts))
    groups.append(collect_param_group(model.encoder.vision_encoder.vit.parameters(), lr=args.stage0_lr_vision))

    excluded = {
        id(param)
        for module in [model.encoder.ts_backbone, model.encoder.vision_encoder.vit]
        for param in module.parameters()
    }
    head_params = [
        param
        for param in model.parameters()
        if param.requires_grad and id(param) not in excluded
    ]
    groups.append(collect_param_group(head_params, lr=args.stage0_lr_heads))
    return AdamW([group for group in groups if group is not None], weight_decay=args.weight_decay)


def build_sp_model(args, device: str, *, stage_name: str, enable_lora: bool) -> OpenTSLMSP:
    model = OpenTSLMSP(
        llm_id=args.llm_id,
        device=device,
        encoder_type="newts_dual_branch",
        newts_dual_branch_config=build_newts_config(args, for_stage0=False),
    )
    model.set_runtime_branch_mode(args.runtime_branch_mode)
    if args.use_alignment_losses:
        if stage_name == "stage2_instruction_tuning":
            model.enable_alignment_losses(
                loss_w_align=args.stage2_loss_w_align,
                loss_w_consistency=args.stage2_loss_w_consistency,
            )
        else:
            model.enable_alignment_losses(
                loss_w_align=args.loss_w_align,
                loss_w_consistency=args.loss_w_consistency,
            )
    if enable_lora:
        model.enable_lora(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def build_sp_optimizer(model: OpenTSLMSP, args, stage_name: str):
    encoder = model.encoder
    groups = []
    ts_lr = args.stage1_lr_ts if stage_name == "stage1_semantic_alignment" else args.stage2_lr_ts
    vision_lr = args.stage1_lr_vision if stage_name == "stage1_semantic_alignment" else args.stage2_lr_vision
    other_lr = args.stage1_lr_other if stage_name == "stage1_semantic_alignment" else args.stage2_lr_other

    groups.append(collect_param_group(encoder.ts_backbone.parameters(), lr=ts_lr))
    groups.append(collect_param_group(encoder.vision_encoder.vit.parameters(), lr=vision_lr))

    excluded = {
        id(param)
        for module in [encoder.ts_backbone, encoder.vision_encoder.vit]
        for param in module.parameters()
    }
    other_params = [
        param
        for param in encoder.parameters()
        if param.requires_grad and id(param) not in excluded
    ]
    groups.append(collect_param_group(other_params, lr=other_lr))
    groups.append(collect_param_group(model.projector.parameters(), lr=other_lr))
    if model.alignment_losses_enabled:
        for module in [model.ts_align_head, model.vision_align_head, model.fused_align_head, model.text_align_head]:
            if module is not None:
                groups.append(collect_param_group(module.parameters(), lr=other_lr))
    if model.lora_enabled:
        groups.append(collect_param_group(model.get_lora_parameters(), lr=args.lr_lora))

    return AdamW([group for group in groups if group is not None], weight_decay=args.weight_decay)


def optimizer_step_count(num_batches: int, grad_accum_steps: int) -> int:
    return max(1, math.ceil(num_batches / grad_accum_steps))


def save_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    save_path: str,
    args,
    stage_name: str,
    epoch: int,
    metrics: Dict[str, float],
    rank: int,
):
    if rank != 0:
        return

    underlying = get_model(model)
    payload = {
        "stage_name": stage_name,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    if isinstance(underlying, DualViewSSLModel):
        payload["encoder_state"] = underlying.encoder.state_dict()
        payload["model_config"] = underlying.get_checkpoint_metadata()
    else:
        payload["model_config"] = underlying.get_checkpoint_metadata()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(payload, save_path)


def load_full_checkpoint(
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


def load_stage0_encoder_into_sp(model: OpenTSLMSP, checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing = model.encoder.load_state_dict(checkpoint["encoder_state"], strict=False)
    return {
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
    }


def train_epoch(
    *,
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    amp_dtype: Optional[torch.dtype],
    grad_clip: float,
    gradient_accumulation_steps: int,
    epoch: int,
    num_epochs: int,
    rank: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda") and amp_dtype == torch.float16))

    running_loss = 0.0
    num_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    for step, batch in enumerate(pbar, start=1):
        with autocast_context(device, amp_dtype):
            loss = model(batch)

        scaled_loss = loss / gradient_accumulation_steps
        if scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = step % gradient_accumulation_steps == 0 or step == len(train_loader)
        if should_step:
            if trainable_params:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(trainable_params, max_norm=grad_clip)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item()
        num_batches += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_model(
    *,
    model,
    data_loader: DataLoader,
    device: str,
    amp_dtype: Optional[torch.dtype],
    stage_name: str,
    max_new_tokens: int,
) -> Dict[str, float]:
    underlying = get_model(model)
    underlying.eval()
    totals: Dict[str, float] = {}
    num_batches = 0
    match_correct = 0
    match_total = 0

    for batch in tqdm(data_loader, desc=f"Eval {stage_name}", leave=False):
        with autocast_context(device, amp_dtype):
            losses = underlying.compute_losses(batch)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value.item())
        num_batches += 1

        match_subset = [sample for sample in batch if sample.get("sample_type") == "match_mismatch"]
        if match_subset:
            predictions = underlying.generate(match_subset, max_new_tokens=max_new_tokens)
            for pred, sample in zip(predictions, match_subset):
                pred_clean = pred.replace(underlying.get_eos_token(), "").strip().lower()
                gold = sample["answer"].replace(underlying.get_eos_token(), "").strip().lower()
                if gold.startswith(pred_clean) or pred_clean == gold:
                    match_correct += 1
                match_total += 1

    metrics = {key: value / max(num_batches, 1) for key, value in totals.items()}
    if match_total > 0:
        metrics["match_accuracy"] = match_correct / match_total
    return metrics


def maybe_set_dataset_epoch(data_loader: DataLoader, epoch: int):
    dataset = getattr(data_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)
    sampler = getattr(data_loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def train_stage0(
    *,
    args,
    run_dir: str,
    device: str,
    world_size: int,
    rank: int,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    stage_name = "stage0_dual_view_ssl"
    stage_dir = os.path.join(run_dir, stage_name)
    checkpoint_path = os.path.join(stage_dir, "checkpoints", "best_encoder.pt")

    datasets = build_stage0_datasets(args)
    train_loader = create_loader(
        datasets["train"],
        batch_size=max(1, args.batch_size * 4),
        shuffle=True,
        world_size=world_size,
        rank=rank,
        collate_fn=collate_raw_series_batch,
    )
    val_loader = create_loader(
        datasets["validation"],
        batch_size=max(1, args.eval_batch_size * 4),
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate_raw_series_batch,
    )
    test_loader = create_loader(
        datasets["test"],
        batch_size=max(1, args.eval_batch_size * 4),
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate_raw_series_batch,
    )

    model = DualViewSSLModel(
        encoder_config=build_newts_config(args, for_stage0=True),
        device=device,
        mask_ratio=args.stage0_mask_ratio,
        jitter_std=args.aug_jitter_std,
        scaling_range=(args.aug_scaling_min, args.aug_scaling_max),
        time_mask_ratio=args.aug_time_mask_ratio,
    )
    if args.gradient_checkpointing:
        model.encoder.enable_gradient_checkpointing()
    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[int(device.split(":")[-1])] if device.startswith("cuda:") else None)

    optimizer = build_stage0_optimizer(get_model(model), args)
    total_steps = optimizer_step_count(len(train_loader), args.gradient_accumulation_steps) * args.stage0_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    start_epoch = 1
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = load_full_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if checkpoint is not None:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val = float(checkpoint.get("metrics", {}).get("loss_total", float("inf")))

    for epoch in range(start_epoch, args.stage0_epochs + 1):
        maybe_set_dataset_epoch(train_loader, epoch)
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=epoch,
            num_epochs=args.stage0_epochs,
            rank=rank,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            stage_name=stage_name,
            max_new_tokens=args.max_new_tokens,
        )
        val_loss = val_metrics["loss_total"]
        if rank == 0:
            print(f"{stage_name} epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if rank == 0 and val_loss + 1e-4 < best_val:
            best_val = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_path=checkpoint_path,
                args=args,
                stage_name=stage_name,
                epoch=epoch,
                metrics={"loss_total": val_loss},
                rank=rank,
            )
        state = broadcast_object_from_rank0(
            {"best_val": best_val},
            rank,
        )
        best_val = float(state["best_val"])

    load_full_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        amp_dtype=amp_dtype,
        stage_name=stage_name,
        max_new_tokens=args.max_new_tokens,
    )
    if rank == 0:
        save_json(os.path.join(stage_dir, "results", "metrics.json"), test_metrics)
        save_json(
            os.path.join(stage_dir, "encoder_config.json"),
            get_model(model).get_checkpoint_metadata(),
        )
    return broadcast_object_from_rank0(test_metrics, rank)


def train_sp_stage(
    *,
    args,
    stage_name: str,
    run_dir: str,
    device: str,
    world_size: int,
    rank: int,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    stage_dir = os.path.join(run_dir, stage_name)
    checkpoint_path = os.path.join(stage_dir, "checkpoints", "best_model.pt")
    resume_from_stage = args.resume and os.path.exists(checkpoint_path)

    enable_lora = stage_name == "stage2_instruction_tuning" and resume_from_stage
    model = build_sp_model(args, device=device, stage_name=stage_name, enable_lora=enable_lora)

    if not resume_from_stage and stage_name == "stage1_semantic_alignment":
        stage0_checkpoint = os.path.join(run_dir, "stage0_dual_view_ssl", "checkpoints", "best_encoder.pt")
        if os.path.exists(stage0_checkpoint):
            load_info = load_stage0_encoder_into_sp(model, stage0_checkpoint, device=device)
            if rank == 0:
                print(f"Loaded stage0 encoder into stage1 (strict=False): {load_info}")

    if not resume_from_stage and stage_name == "stage2_instruction_tuning":
        previous_checkpoint = os.path.join(run_dir, "stage1_semantic_alignment", "checkpoints", "best_model.pt")
        if not os.path.exists(previous_checkpoint):
            raise RuntimeError(f"Stage2 requires stage1 checkpoint: {previous_checkpoint}")
        load_full_checkpoint(model=model, checkpoint_path=previous_checkpoint, device=device)
        model.enable_lora(
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    model.to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[int(device.split(":")[-1])] if device.startswith("cuda:") else None)

    optimizer = build_sp_optimizer(get_model(model), args, stage_name)
    num_epochs = args.stage1_epochs if stage_name == "stage1_semantic_alignment" else args.stage2_epochs

    eos_token = get_model(model).get_eos_token()
    train_dataset = build_stage12_datasets(args, split="train", eos_token=eos_token, stage_name=stage_name)
    val_dataset = build_stage12_datasets(args, split="validation", eos_token=eos_token, stage_name=stage_name)
    test_dataset = build_stage12_datasets(args, split="test", eos_token=eos_token, stage_name=stage_name)

    collate = lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch,
        patch_size=1,
        pad_mode="last",
    )
    train_loader = create_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        collate_fn=collate,
    )
    val_loader = create_loader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate,
    )
    test_loader = create_loader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        world_size=1,
        rank=0,
        collate_fn=collate,
    )

    total_steps = optimizer_step_count(len(train_loader), args.gradient_accumulation_steps) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    start_epoch = 1
    best_val = float("inf")
    if resume_from_stage:
        checkpoint = load_full_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if checkpoint is not None:
            start_epoch = int(checkpoint.get("epoch", 0)) + 1
            best_val = float(checkpoint.get("metrics", {}).get("loss_total", float("inf")))

    if rank == 0 and not args.use_pma:
        print("ℹ️ use_pma=False; no_pma may increase sequence length and memory.")

    epochs_without_improvement = 0
    for epoch in range(start_epoch, num_epochs + 1):
        maybe_set_dataset_epoch(train_loader, epoch)
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            epoch=epoch,
            num_epochs=num_epochs,
            rank=rank,
        )
        val_metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            device=device,
            amp_dtype=amp_dtype,
            stage_name=stage_name,
            max_new_tokens=args.max_new_tokens,
        )
        val_loss = val_metrics["loss_total"]
        if rank == 0:
            print(
                f"{stage_name} epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} align={val_metrics.get('loss_align', 0.0):.4f} "
                f"cons={val_metrics.get('loss_consistency', 0.0):.4f}"
            )
        if rank == 0 and val_loss + 1e-4 < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_path=checkpoint_path,
                args=args,
                stage_name=stage_name,
                epoch=epoch,
                metrics=val_metrics,
                rank=rank,
            )
        elif rank == 0:
            epochs_without_improvement += 1

        state = broadcast_object_from_rank0(
            {
                "best_val": best_val,
                "epochs_without_improvement": epochs_without_improvement,
                "stop": epochs_without_improvement >= args.early_stop,
            },
            rank,
        )
        best_val = float(state["best_val"])
        epochs_without_improvement = int(state["epochs_without_improvement"])
        if state["stop"]:
            break

    load_full_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        amp_dtype=amp_dtype,
        stage_name=stage_name,
        max_new_tokens=args.max_new_tokens,
    )
    if rank == 0:
        save_json(os.path.join(stage_dir, "results", "metrics.json"), test_metrics)
    return broadcast_object_from_rank0(test_metrics, rank)


def export_final_model_checkpoint(model: OpenTSLMSP, export_path: str, rank: int):
    if rank != 0:
        return
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.store_to_file(export_path)
    print(f"📦 Exported model_checkpoint.pt to {export_path}")


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

        resolved_precision, amp_dtype = resolve_precision(device, args.precision)
        set_seed(args.seed + rank)
        run_dir = os.path.join(args.save_dir, sanitize_llm_id(args.llm_id), args.run_name)

        if rank == 0:
            os.makedirs(run_dir, exist_ok=True)
            save_json(os.path.join(run_dir, "config.json"), vars(args))
            save_json(
                os.path.join(run_dir, "effective_config.json"),
                {
                    **vars(args),
                    "resolved_precision": resolved_precision,
                    "world_size": world_size,
                    "effective_global_batch_size": world_size * args.batch_size * args.gradient_accumulation_steps,
                    "vit_feature_mode": "single",
                    "vit_layer_idx": 4,
                    "vit_num_hidden_layers": 4,
                },
            )
            print("=" * 72)
            print("Multiview Curriculum Pretraining")
            print("=" * 72)
            print(f"time: {datetime.datetime.now()}")
            print(f"device: {device}")
            print(f"precision: {resolved_precision}")
            print(f"run_dir: {run_dir}")
            print(f"stages: {args.stages}")
            print(f"vision feature: single@4 / loaded_layers=4")
            print(f"use_pma: {args.use_pma}")
            print(f"vision_train_mode: {args.vision_train_mode}")
            print(f"vision_topk_blocks: {args.vision_topk_blocks}")
            print("=" * 72)

        if dist.is_initialized():
            dist.barrier()

        results: Dict[str, Any] = {}
        final_sp_model: Optional[OpenTSLMSP] = None

        for stage_name in args.stages:
            if stage_name == "stage0_dual_view_ssl":
                stage_results = train_stage0(
                    args=args,
                    run_dir=run_dir,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    amp_dtype=amp_dtype,
                )
            else:
                stage_results = train_sp_stage(
                    args=args,
                    stage_name=stage_name,
                    run_dir=run_dir,
                    device=device,
                    world_size=world_size,
                    rank=rank,
                    amp_dtype=amp_dtype,
                )
                if rank == 0 and stage_name == args.stages[-1]:
                    final_sp_model = build_sp_model(
                        args,
                        device=device,
                        stage_name=stage_name,
                        enable_lora=(stage_name == "stage2_instruction_tuning"),
                    )
                    load_full_checkpoint(
                        model=final_sp_model,
                        checkpoint_path=os.path.join(run_dir, stage_name, "checkpoints", "best_model.pt"),
                        device=device,
                    )
            results[stage_name] = stage_results

        if rank == 0:
            save_json(os.path.join(run_dir, "curriculum_results.json"), results)
            if args.export_model_checkpoint and final_sp_model is not None:
                export_final_model_checkpoint(
                    final_sp_model,
                    os.path.join(run_dir, "model_checkpoint.pt"),
                    rank=rank,
                )
            print(f"🎉 Done. Results saved to {run_dir}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
