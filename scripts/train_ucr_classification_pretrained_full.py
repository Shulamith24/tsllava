#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M2: UCR single-dataset classification training with pretrained SP models.

加载curriculum learning的stage2预训练模型进行分类微调。
编码器和投影层解冻，LLM使用LoRA训练。
使用特殊类别token: <c0>, <c1>, ... <cK-1>

使用方法：
    python scripts/train_ucr_classification_pretrained_full.py \
        --pretrained_model OpenTSLM/llama-3.2-1b-m4-sp \
        --dataset ECG5000 \
        --epochs 30 \
        --batch_size 4

训练配置：
- LoRA: r=16, alpha=32 (默认启用)
- Encoder LR: 2e-4
- Projector LR: 1e-4
- LoRA LR: 1e-4
- 使用特殊类别token (<c0>, <c1>, ...) 替代字母标签
- 约束解码：只允许输出类别token + EOS
"""

import argparse
import datetime
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    get_linear_schedule_with_warmup,
)

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.class_token_rows import (
    get_class_token_trainable_parameters,
    load_class_token_rows_from_checkpoint,
    register_class_token_row_training,
    sanitize_class_token_optimizer_state,
    save_class_token_rows_to_checkpoint,
)
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


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


def cli_flag_was_provided(argv: Optional[List[str]], flag_name: str) -> bool:
    if argv is None:
        argv = sys.argv[1:]
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in argv)


def parse_args(argv=None):
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="M2: UCR单数据集分类训练（基于Stage2预训练模型）")

    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结编码器参数")

    parser.add_argument("--dataset", type=str, default="CricketZ", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")

    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        help="预训练模型ID (HuggingFace repo_id，如 OpenTSLM/llama-3.2-1b-m4-sp)",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default=None,
        help="本地checkpoint路径 (如 results/curriculum_pretrain/.../best_model.pt)",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="tslanet",
        choices=["transformer_cnn", "tslanet", "newts_dual_branch"],
        help="编码器类型（使用checkpoint时会被checkpoint元数据覆盖）",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM模型ID（使用local_checkpoint时需要）",
    )
    parser.add_argument(
        "--tslanet_patch_size",
        type=int,
        default=8,
        help="TSLANet的patch_size（使用tslanet编码器时）",
    )
    parser.add_argument("--random_init_llm", action="store_true", help="随机初始化LLM权重（用于测试完全随机初始化的模型）")

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
    parser.add_argument("--projector_type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--projector_dropout", type=float, default=0.1)
    parser.add_argument("--use_pma", action="store_true", help="Enable PMA slot aggregation for newts_dual_branch")
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
    parser.add_argument(
        "--freeze_vision_backbone",
        dest="freeze_vision_backbone",
        action="store_true",
        help="Freeze the vision backbone parameters",
    )
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

    parser.add_argument("--no_lora", action="store_true", help="禁用LoRA（不推荐）")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "last", "repeat"], help="时序padding策略")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="预热比例")

    parser.add_argument("--save_dir", type=str, default="results/ucr_pretrained", help="结果保存目录")
    parser.add_argument("--resume", action="store_true", help="从 save_dir/dataset/last_checkpoint.pt 断点续训")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="训练完成并写出结果后删除 best_model.pt / last_checkpoint.pt 以节省磁盘空间",
    )

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--max_new_tokens", type=int, default=2, help="生成最大token数（类别token + EOS）")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="评估批次大小")

    args = parser.parse_args(argv)
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    args.encoder_type_explicit = cli_flag_was_provided(provided_argv, "--encoder_type")
    args.context_length_explicit = cli_flag_was_provided(provided_argv, "--context_length")
    args.pad_mode_explicit = cli_flag_was_provided(provided_argv, "--pad_mode")
    return args


def setup_distributed():
    """初始化分布式训练环境（用于torchrun）"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        return local_rank, world_size, rank
    return 0, 1, 0


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(model):
    """获取底层模型（兼容DDP包装）"""
    return model.module if hasattr(model, "module") else model


def broadcast_object_from_rank0(obj, world_size: int, rank: int):
    if world_size == 1:
        return obj
    holder = [obj if rank == 0 else None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cleanup_checkpoint_files(paths: List[str], rank: int = 0):
    """删除不再需要的 checkpoint 文件；失败时仅告警，不中断训练。"""
    if rank != 0:
        return

    for path in paths:
        if not path or not os.path.exists(path):
            continue
        try:
            os.remove(path)
            print(f"🧹 Removed checkpoint: {path}")
        except OSError as exc:
            print(f"⚠️ Failed to remove checkpoint {path}: {exc}")


def hydrate_args_from_model_config(args, model_config: Dict[str, Any]):
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
        ]
        for key in structural_keys:
            if key in encoder_config:
                setattr(args, key, encoder_config[key])

    return args


def hydrate_args_from_checkpoint(args, checkpoint: Dict[str, Any], fallback_llm_id: str):
    resolved = OpenTSLM._resolve_sp_init_kwargs_from_checkpoint(checkpoint, fallback_llm_id=fallback_llm_id)
    encoder_config = resolved["tslanet_config"] or resolved["newts_dual_branch_config"] or {}
    model_config = {
        "llm_id": resolved["llm_id"],
        "encoder_type": resolved["encoder_type"],
        "encoder_config": encoder_config,
    }
    return hydrate_args_from_model_config(args, model_config), model_config


def hydrate_args_from_local_checkpoint_metadata(args):
    args.local_checkpoint_model_config = None
    if not args.local_checkpoint:
        return args

    checkpoint = torch.load(args.local_checkpoint, map_location="cpu", weights_only=False)
    args, model_config = hydrate_args_from_checkpoint(args, checkpoint, fallback_llm_id=args.llm_id)
    args.local_checkpoint_model_config = model_config
    return args


def hydrate_args_from_pretrained_model_metadata(args):
    args.pretrained_model_checkpoint = None
    args.pretrained_model_model_config = None
    if not args.pretrained_model:
        return args

    checkpoint_path = OpenTSLM._download_model_files(args.pretrained_model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args, model_config = hydrate_args_from_checkpoint(
        args,
        checkpoint,
        fallback_llm_id=OpenTSLM._get_base_llm_id(args.pretrained_model),
    )

    args.pretrained_model_checkpoint = checkpoint_path
    args.pretrained_model_model_config = model_config
    return args


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


def warn_deprecated_newts_context_length(args, rank: int):
    if rank != 0:
        return
    if args.encoder_type != "newts_dual_branch":
        return
    if getattr(args, "context_length_explicit", False):
        print("⚠️ --context_length is deprecated for newts_dual_branch and is ignored in dynamic-length mode")


def resolve_base_llm_id(args) -> str:
    pretrained_model_config = getattr(args, "pretrained_model_model_config", None) or {}
    if args.pretrained_model and pretrained_model_config.get("llm_id"):
        return pretrained_model_config["llm_id"]
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

    if args.pretrained_model and getattr(args, "encoder_type_explicit", False):
        raise ValueError("--pretrained_model and --encoder_type cannot be specified together")

    encoder_type = getattr(args, "encoder_type", None)
    if encoder_type is None:
        raise ValueError("--encoder_type could not be resolved from CLI args or checkpoint metadata")

    if encoder_type != "newts_dual_branch":
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
            raise ValueError(
                "--vit_num_hidden_layers must be >= the requested target feature layer depth"
            )

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

    resolved_aggregator_hidden_size = args.aggregator_hidden_size or ENCODER_OUTPUT_DIM
    if resolved_aggregator_hidden_size % args.aggregator_num_heads != 0:
        raise ValueError("--aggregator_num_heads must evenly divide the PMA hidden size")


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
        merged_config.pop("context_length", None)
        merged_config["dynamic_length"] = True
        merged_config["ts_positional_encoding"] = "sinusoidal"
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
            print(f"📂 从本地checkpoint加载: {args.local_checkpoint}")
            print(f"   编码器类型: {model_init_kwargs['encoder_type']}")
            print(f"   LLM: {model_init_kwargs['llm_id']}")

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
            print("✅ 已加载encoder和projector权重")

        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)

    elif args.pretrained_model:
        if rank == 0:
            print(f"📂 从HuggingFace加载: {args.pretrained_model}")

        model = OpenTSLM.load_pretrained(
            repo_id=args.pretrained_model,
            device=device,
            enable_lora=use_lora,
            checkpoint_path=getattr(args, "pretrained_model_checkpoint", None),
        )

        if use_lora and (args.lora_r != 16 or args.lora_alpha != 32):
            model.disable_lora()
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            if rank == 0:
                print(f"📎 重新配置LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    else:
        model_init_kwargs = resolve_model_init_kwargs(args)
        if rank == 0:
            print("🆕 从零开始训练（无预训练权重）")
            print(f"   编码器类型: {model_init_kwargs['encoder_type']}")
            print(f"   LLM: {model_init_kwargs['llm_id']}")

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
            print("🎲 随机初始化LLM权重...")
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
            print("✅ LLM已随机初始化")

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        if rank == 0:
            print("🧊 编码器参数已冻结")

    return model


def calculate_accuracy(predictions: List[str], labels: List[str]) -> float:
    """
    计算分类准确率 - 适配特殊token格式 (<c0>, <c1>, ...)
    
    直接比较生成的token与真实标签
    """
    import re
    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = pred.strip()
        label_clean = label.strip()
        
        # 尝试从预测中提取 <cN> 格式的token
        match = re.search(r'<c\d+>', pred_clean)
        if match:
            pred_token = match.group()
        else:
            # 如果没有找到，使用整个预测
            pred_token = pred_clean
        
        # 直接比较
        if pred_token == label_clean:
            correct += 1
    
    return correct / len(predictions) if predictions else 0.0


def add_class_tokens_to_model(model, num_classes: int, device: str, rank: int = 0):
    """
    添加类别特殊token到tokenizer和embedding层
    
    Args:
        model: OpenTSLMSP 模型
        num_classes: 类别数量
        device: 设备
        rank: DDP rank
    
    Returns:
        class_tokens: 类别token列表 ['<c0>', '<c1>', ...]
        class_token_ids: 对应的token ID列表
    """
    class_tokens = [f"<c{i}>" for i in range(num_classes)]
    
    # 添加到tokenizer
    num_added = model.tokenizer.add_tokens(class_tokens, special_tokens=True)
    if rank == 0:
        print(f"✅ Added {num_added} class tokens to tokenizer")
    
    # 调整embedding大小
    old_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    model.llm.resize_token_embeddings(len(model.tokenizer))
    new_vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    
    if rank == 0:
        print(f"   Vocabulary size: {old_vocab_size} -> {new_vocab_size}")
    
    # 改进的初始化：每个类别token使用不同的初始化
    # 从已有token中随机采样，并添加小的扰动
    with torch.no_grad():
        embedding = model.llm.get_input_embeddings()
        lm_head = model.llm.lm_head
        
        if num_added > 0:
            # 获取已有embedding的统计信息
            old_embeddings = embedding.weight[:-num_added]
            emb_mean = old_embeddings.mean(dim=0)
            emb_std = old_embeddings.std(dim=0)
            
            # 为每个类别token生成不同的初始化
            for i in range(num_added):
                # 方法：均值 + 随机扰动 (扰动幅度为标准差的10%)
                noise = torch.randn_like(emb_mean) * emb_std * 0.1
                embedding.weight[-num_added + i] = emb_mean + noise
            
            # 同样处理lm_head
            old_head = lm_head.weight[:-num_added]
            head_mean = old_head.mean(dim=0)
            head_std = old_head.std(dim=0)
            
            for i in range(num_added):
                noise = torch.randn_like(head_mean) * head_std * 0.1
                lm_head.weight[-num_added + i] = head_mean + noise
            
            if rank == 0:
                print(f"   Initialized {num_added} class tokens with mean + random perturbation")
    
    # 获取token IDs
    class_token_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in class_tokens]
    register_class_token_row_training(model, class_token_ids)
    if rank == 0:
        print(
            f"   Class token IDs: {class_token_ids[:5]}..."
            if len(class_token_ids) > 5
            else f"   Class token IDs: {class_token_ids}"
        )
        print("   Restricted embedding/lm_head updates to class-token rows only")
    
    return class_tokens, class_token_ids


class AllowedTokensLogitsProcessor(LogitsProcessor):
    """
    约束解码的Logits处理器：只允许特定token被生成
    """
    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = set(allowed_token_ids)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 创建mask，只保留允许的token
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            if token_id < scores.shape[-1]:
                mask[:, token_id] = 0
        return scores + mask


class IndexedDataset(torch.utils.data.Dataset):
    """
    为数据集包装一个索引，用于分布式评估时的去重
    """
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # 添加原始索引到样本中
        sample["_sample_idx"] = idx
        return sample


def create_data_loaders(args, eos_token: str, world_size: int = 1, rank: int = 0):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRClassificationDataset(
        split="validation",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRClassificationDataset(
        split="test",
        EOS_TOKEN=eos_token,
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    # 用IndexedDataset包装评估数据集，为每个样本添加索引
    indexed_val_dataset = IndexedDataset(val_dataset)
    indexed_test_dataset = IndexedDataset(test_dataset)
    
    # Collate函数
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=resolve_collate_patch_size(args),
            pad_mode=resolve_effective_pad_mode(args),
        )
    
    # 分布式采样器
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        # 评估集使用分布式采样器（shuffle=False保持顺序）
        val_sampler = DistributedSampler(
            indexed_val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            indexed_test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    
    # 评估用DataLoader（使用分布式采样+索引跟踪）
    eval_batch_size = getattr(args, 'eval_batch_size', 8)
    
    val_loader = DataLoader(
        indexed_val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        indexed_test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        collate_fn=collate_fn,
    )
    
    return (
        train_loader,
        val_loader,
        test_loader,
        train_sampler,
        len(val_dataset),
        len(test_dataset),
        train_dataset,
    )


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    epoch: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    rank: int = 0,
) -> float:
    """训练一个epoch（支持梯度累积和DDP）"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", disable=(rank != 0))
    for step, batch in enumerate(pbar):
        # 计算损失（缩放用于梯度累积）
        # 使用model(batch)调用forward方法，DDP梯度同步在backward()时自动进行
        loss = model(batch)
        loss = loss / gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积完成后更新
        if (step + 1) % gradient_accumulation_steps == 0:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
    
    # 处理最后不足accumulation_steps的batch
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
    class_token_ids: List[int] | None = None,
    desc: str = "Evaluating",
    rank: int = 0,
    world_size: int = 1,
    total_samples: int | None = None,
) -> Dict[str, Any]:
    """
    分布式评估模型（使用样本索引正确去重）
    
    Args:
        model: 模型（DDP 包装或底层模型都可以）
        data_loader: 数据加载器（使用IndexedDataset + DistributedSampler）
        max_new_tokens: 最大生成token数
        class_token_ids: 类别token的ID列表，用于约束解码
        desc: 进度条描述
        rank: DDP rank
        world_size: GPU 数量
        total_samples: 真实样本数，用于验证去重结果
    """
    import re
    import pickle
    
    # 始终使用底层模型评估
    underlying_model = get_model(model)
    underlying_model.eval()
    
    # 使用字典按索引存储结果（自动去重）
    results_by_idx = {}
    total_loss = 0.0
    num_batches = 0
    
    # 设置约束解码处理器
    logits_processor = None
    if class_token_ids is not None:
        eos_token_id = underlying_model.tokenizer.eos_token_id
        allowed_ids = class_token_ids + [eos_token_id]
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])
    
    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        # 使用底层模型
        loss = underlying_model.compute_loss(batch)
        total_loss += loss.item()
        num_batches += 1
        
        # 生成预测（使用约束解码）
        if logits_processor is not None:
            inputs_embeds, attention_mask = underlying_model.pad_and_apply_batch(batch)
            gen_ids = underlying_model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                do_sample=False,
            )
            predictions = underlying_model.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)
            # 清理多余的特殊token，保留<cN>格式
            cleaned_predictions = []
            for p in predictions:
                match = re.search(r'<c\d+>', p)
                if match:
                    cleaned_predictions.append(match.group())
                else:
                    cleaned_predictions.append(p.strip())
            predictions = cleaned_predictions
        else:
            predictions = underlying_model.generate(batch, max_new_tokens=max_new_tokens)
        
        # 收集结果（使用样本索引作为key）
        for sample, pred in zip(batch, predictions):
            idx = sample.get("_sample_idx", -1)
            label = sample["answer"].replace(underlying_model.get_eos_token(), "").strip()
            results_by_idx[idx] = {"prediction": pred, "label": label}
    
    # 分布式聚合：收集所有 rank 的结果
    if world_size > 1:
        # 序列化本地结果
        local_data = pickle.dumps({
            "results_by_idx": results_by_idx,
            "loss": total_loss,
            "num_batches": num_batches,
        })
        local_size = torch.tensor([len(local_data)], device=underlying_model.device)
        
        # 收集所有 rank 的数据大小
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)
        max_size = max(s.item() for s in all_sizes)
        
        # 填充到相同大小
        local_tensor = torch.zeros(int(max_size), dtype=torch.uint8, device=underlying_model.device)
        local_tensor[:len(local_data)] = torch.tensor(list(local_data), dtype=torch.uint8, device=underlying_model.device)
        
        # 收集所有数据
        all_tensors = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(all_tensors, local_tensor)
        
        # 反序列化并合并（字典自动去重：相同索引只保留一份）
        merged_results = {}
        total_loss = 0.0
        num_batches = 0
        
        for tensor, size in zip(all_tensors, all_sizes):
            data = pickle.loads(bytes(tensor[:size.item()].cpu().tolist()))
            merged_results.update(data["results_by_idx"])  # 自动去重
            total_loss += data["loss"]
            num_batches += data["num_batches"]
        
        results_by_idx = merged_results
    
    # 按索引排序并提取结果
    sorted_indices = sorted(results_by_idx.keys())
    all_predictions = [results_by_idx[idx]["prediction"] for idx in sorted_indices]
    all_labels = [results_by_idx[idx]["label"] for idx in sorted_indices]
    
    # 验证样本数量
    if total_samples is not None and len(all_predictions) != total_samples:
        if rank == 0:
            print(f"⚠️ 警告: 期望 {total_samples} 个样本，实际 {len(all_predictions)} 个")
    
    # 计算指标
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
    val_loss: Optional[float],
    val_acc: Optional[float],
    save_path: str,
    args,
    extra_state: Optional[Dict[str, Any]] = None,
    rank: int = 0,
):
    """保存checkpoint（仅rank=0执行）"""
    if rank != 0:
        return
    
    underlying_model = get_model(model)
    if optimizer is not None:
        sanitize_class_token_optimizer_state(optimizer, underlying_model)

    checkpoint = {
        "model_config": underlying_model.get_checkpoint_metadata(),
        "encoder_state": underlying_model.encoder.state_dict(),
        "projector_state": underlying_model.projector.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "args": vars(args),
    }
    if extra_state:
        checkpoint.update(extra_state)

    # 保存LoRA权重
    underlying_model.save_lora_state_to_checkpoint(checkpoint)

    save_class_token_rows_to_checkpoint(underlying_model, checkpoint)

    torch.save(checkpoint, save_path)
    print(f"💾 Saved checkpoint to: {save_path}")


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
    load_class_token_rows_from_checkpoint(underlying_model, checkpoint, device=device)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        sanitize_class_token_optimizer_state(optimizer, underlying_model)
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


def resolve_training_resume_state(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    last_epoch = int(checkpoint.get("epoch", 0) or 0)
    return {
        "start_epoch": last_epoch + 1,
        "best_val_acc": float(checkpoint.get("best_val_acc", float("-inf"))),
        "patience_counter": int(checkpoint.get("patience_counter", 0) or 0),
        "loss_history": list(checkpoint.get("loss_history", [])),
    }


def main():
    args = parse_args()
    args.use_lora = not args.no_lora
    args = hydrate_args_from_local_checkpoint_metadata(args)
    args = hydrate_args_from_pretrained_model_metadata(args)

    local_rank, world_size, rank = setup_distributed()

    try:
        validate_args(args)

        if world_size > 1:
            device = f"cuda:{local_rank}"
        elif args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            if rank == 0 and args.device == "cuda":
                print("⚠️ CUDA不可用，使用CPU")
            device = "cpu"

        set_seed(args.seed + rank)
        warn_deprecated_newts_context_length(args, rank)

        eos_rank0 = resolve_dataset_eos_token(args) if rank == 0 else None
        dataset_eos = broadcast_object_from_rank0(eos_rank0, world_size, rank)

        if rank == 0:
            print("\n📂 加载数据...")
        (
            train_loader,
            val_loader,
            test_loader,
            train_sampler,
            val_size,
            test_size,
            train_dataset,
        ) = create_data_loaders(args, dataset_eos, world_size, rank)

        save_dir = os.path.join(args.save_dir, args.dataset)
        num_classes = UCRClassificationDataset.get_num_classes()

        if rank == 0:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "config.json"), "w") as f:
                json.dump(vars(args), f, indent=2)

            print("=" * 60)
            print("M2: UCR单数据集分类训练（基于Stage2预训练模型）")
            print("=" * 60)
            print(f"时间: {datetime.datetime.now()}")
            print(f"数据集: {args.dataset}")
            print(f"预训练模型: {args.pretrained_model}")
            print(f"本地checkpoint: {args.local_checkpoint}")
            print(f"编码器类型: {args.encoder_type}")
            print(f"LLM: {args.llm_id}")
            print(f"LoRA: {args.use_lora}")
            print(f"DDP: world_size={world_size}")
            print(f"梯度累积: {args.gradient_accumulation_steps}")
            print(f"梯度检查点: {args.gradient_checkpointing}")
            if args.encoder_type == "newts_dual_branch":
                print("dynamic_length: enabled")
                print(f"pad_mode: {resolve_effective_pad_mode(args)}")
            print("=" * 60)
            print(f"   Train batches: {len(train_loader)}")
            print(f"   Val batches: {len(val_loader)}")
            print(f"   Test batches: {len(test_loader)}")

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            print("\n🔧 加载模型...")
        model = build_model(args=args, device=device, rank=rank)

        if rank == 0:
            print("\n🎯 添加类别token...")
        add_class_tokens_to_model(get_model(model), num_classes, device, rank)

        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])
            if rank == 0:
                print(f"✅ 模型已用DDP包装 (world_size={world_size})")

        underlying_model = get_model(model)
        class_token_ids = [
            underlying_model.tokenizer.convert_tokens_to_ids(token)
            for token in UCRClassificationDataset.get_class_tokens()
        ]

        if rank == 0:
            print("\n⚙️ 创建优化器...")

        param_groups = []
        if not args.freeze_encoder:
            param_groups.append({"params": underlying_model.encoder.parameters(), "lr": args.lr_encoder})
        param_groups.append({"params": underlying_model.projector.parameters(), "lr": args.lr_projector})

        if args.use_lora:
            lora_params = underlying_model.get_lora_parameters()
            if lora_params:
                param_groups.append({"params": lora_params, "lr": args.lr_lora})

        class_token_params = get_class_token_trainable_parameters(underlying_model)
        param_groups.append(
            {
                "params": class_token_params,
                "lr": args.lr_lora * 2,
                "weight_decay": 0.0,
            }
        )
        if rank == 0:
            print(f"   Added embedding and lm_head to optimizer (lr={args.lr_lora * 2:.2e})")

        optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

        effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
        steps_per_epoch = max(1, len(train_loader) // max(1, args.gradient_accumulation_steps))
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_steps = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        if rank == 0:
            print(f"   Effective batch size: {effective_batch_size}")
            print(f"   Total steps: {total_steps}")
            print(f"   Warmup steps: {warmup_steps}")
            print("\n🚀 开始训练...")

        best_checkpoint_path = os.path.join(save_dir, "best_model.pt")
        last_checkpoint_path = os.path.join(save_dir, "last_checkpoint.pt")
        best_val_acc = float("-inf")
        patience_counter = 0
        loss_history = []
        start_epoch = 1
        completed_epochs = 0

        if args.resume:
            if os.path.exists(last_checkpoint_path):
                resume_checkpoint = load_checkpoint(
                    model,
                    last_checkpoint_path,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                resume_state = resolve_training_resume_state(resume_checkpoint)
                start_epoch = resume_state["start_epoch"]
                best_val_acc = resume_state["best_val_acc"]
                patience_counter = resume_state["patience_counter"]
                loss_history = resume_state["loss_history"]
                completed_epochs = start_epoch - 1
                if rank == 0:
                    print(
                        f"📂 断点续训: 已完成 {completed_epochs} / {args.epochs} 个 epoch，"
                        f"从 epoch {start_epoch} 继续"
                    )
            elif rank == 0:
                print("⚠️ 未找到 last_checkpoint.pt，将从头开始训练")

        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                args.grad_clip,
                epoch,
                args.epochs,
                args.gradient_accumulation_steps,
                rank,
            )

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                if rank == 0:
                    print(f"\n📊 Epoch {epoch} 评估...")

                val_results = evaluate(
                    model,
                    val_loader,
                    args.max_new_tokens,
                    class_token_ids=class_token_ids,
                    desc="Validating",
                    rank=rank,
                    world_size=world_size,
                    total_samples=val_size,
                )
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]

                if rank == 0:
                    print(f"   Train Loss: {train_loss:.4f}")
                    print(f"   Val Loss: {val_loss:.4f}")
                    print(f"   Val Accuracy: {val_acc:.4f}")

                    print("   Sample predictions:")
                    for i in range(min(3, len(val_results["predictions"]))):
                        pred = val_results["predictions"][i]
                        label = val_results["labels"][i]
                        pred_short = pred[-50:] if len(pred) > 50 else pred
                        print(f"     Pred: '{pred_short}' | Label: '{label}'")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        save_checkpoint(
                            model, optimizer, scheduler, epoch,
                            val_loss, val_acc,
                            best_checkpoint_path,
                            args,
                            extra_state={
                                "best_val_acc": best_val_acc,
                                "patience_counter": patience_counter,
                                "loss_history": loss_history,
                            },
                            rank=rank,
                        )
                    else:
                        patience_counter += 1
                        print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")

                    loss_history.append({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                        json.dump(loss_history, f, indent=2)

                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        val_loss,
                        val_acc,
                        last_checkpoint_path,
                        args,
                        extra_state={
                            "best_val_acc": best_val_acc,
                            "patience_counter": patience_counter,
                            "loss_history": loss_history,
                            "last_train_loss": train_loss,
                        },
                        rank=rank,
                    )

                if world_size > 1:
                    patience_tensor = torch.tensor([patience_counter], device=device)
                    best_val_acc_tensor = torch.tensor([best_val_acc], device=device)
                    dist.broadcast(patience_tensor, src=0)
                    dist.broadcast(best_val_acc_tensor, src=0)
                    patience_counter = int(patience_tensor.item())
                    best_val_acc = float(best_val_acc_tensor.item())
            else:
                if rank == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        None,
                        None,
                        last_checkpoint_path,
                        args,
                        extra_state={
                            "best_val_acc": best_val_acc,
                            "patience_counter": patience_counter,
                            "loss_history": loss_history,
                            "last_train_loss": train_loss,
                        },
                        rank=rank,
                    )

            completed_epochs = epoch

            if patience_counter >= args.early_stop:
                if rank == 0:
                    print(f"\n⏹️ 早停! 验证准确率 {args.early_stop} 轮未改进")
                break

        if rank == 0:
            print("\n" + "=" * 60)
            print("📋 最终测试评估...")

        final_checkpoint_path = best_checkpoint_path
        if not os.path.exists(final_checkpoint_path):
            final_checkpoint_path = last_checkpoint_path
            if rank == 0:
                print("⚠️ 未找到 best_model.pt，回退到 last_checkpoint.pt 进行测试")

        load_checkpoint(model, final_checkpoint_path, device=device)

        if world_size > 1:
            dist.barrier()

        test_results = evaluate(
            model, test_loader, args.max_new_tokens,
            class_token_ids=class_token_ids, desc="Testing",
            rank=rank, world_size=world_size, total_samples=test_size
        )

        if rank == 0:
            print(f"\n✅ 测试结果:")
            print(f"   Test Loss: {test_results['loss']:.4f}")
            print(f"   Test Accuracy: {test_results['accuracy']:.4f}")

            final_results = {
                "dataset": args.dataset,
                "pretrained_model": args.pretrained_model,
                "local_checkpoint": args.local_checkpoint,
                "best_val_acc": best_val_acc,
                "test_loss": test_results["loss"],
                "test_accuracy": test_results["accuracy"],
                "epochs_trained": completed_epochs,
            }

            with open(os.path.join(save_dir, "final_results.json"), "w") as f:
                json.dump(final_results, f, indent=2)

            with open(os.path.join(save_dir, "test_predictions.json"), "w") as f:
                json.dump({
                    "predictions": test_results["predictions"],
                    "labels": test_results["labels"],
                }, f, indent=2)

            if args.cleanup_checkpoints:
                cleanup_checkpoint_files(
                    [best_checkpoint_path, last_checkpoint_path],
                    rank=rank,
                )
            
            print("=" * 60)
            print(f"结果保存到: {save_dir}")
            print("=" * 60)

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
