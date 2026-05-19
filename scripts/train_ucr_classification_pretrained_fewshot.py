#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M2: Univariate classification with pretrained SP models under a few-shot protocol.

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
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.distributed as dist
import torch.nn.functional as F
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

from opentslm.model.encoder.NewTSVisionEncoder import (
    LEGACY_VISION_2D_MODE,
    SUPPORTED_VISION_2D_MODES,
    resolve_effective_vision_stride,
    validate_vision_2d_mode,
)
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.llm.hf_local import resolve_local_hf_snapshot
from opentslm.model.class_token_rows import (
    get_class_token_trainable_parameters,
    load_class_token_rows_from_checkpoint,
    register_class_token_row_training,
    sanitize_class_token_optimizer_state,
    save_class_token_rows_to_checkpoint,
)
from opentslm.model_config import ENCODER_OUTPUT_DIM, PATCH_SIZE
from opentslm.time_series_datasets.classification_utils import (
    class_token_to_index,
    extract_class_token,
)
from opentslm.time_series_datasets.univariate_fewshot import (
    load_univariate_fewshot_bundle,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

ShotType = Union[int, Literal["full"]]
DEFAULT_EPOCHS = 50


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
    parser = argparse.ArgumentParser(
        description="M2: few-shot univariate classification with pretrained SP models"
    )

    # Core protocol behavior
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
    parser.add_argument(
        "--model_select_metric",
        type=str,
        default="last",
        choices=["last", "train_loss"],
        help="Kept for compatibility; final checkpoint is always Phase2 last.",
    )
    parser.add_argument("--fewshot_batch_mode", type=str, default="manual", choices=["full", "manual"])

    # Phase setup
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--phase1_epochs", type=int, default=5)

    # Must-specify switches / compatibility flags
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder parameters")
    parser.add_argument(
        "--runtime_branch_mode",
        type=str,
        default="both",
        choices=["both", "ts_only", "vision_only"],
        help="Runtime branch masking for dual-branch checkpoints.",
    )

    # Data
    parser.add_argument(
        "--dataset_family",
        type=str,
        default="ucr",
        choices=["ucr", "mitbih", "sleepedf", "cinc2017af", "cinc2016heart"],
        help="Univariate classification dataset family.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name within the selected family")
    parser.add_argument(
        "--split_protocol",
        type=str,
        default="default",
        help="Dataset-family-specific split protocol.",
    )
    parser.add_argument("--data_path", type=str, default="./data", help="Dataset root directory")
    parser.add_argument(
        "--label_interface",
        type=str,
        default="anonymous",
        choices=["anonymous", "semantic"],
        help="Use anonymous class tokens or natural-language label verbalizers.",
    )
    parser.add_argument(
        "--verbalizer_set",
        type=str,
        default="canonical",
        choices=["canonical"],
        help="Label verbalizer set used when --label_interface=semantic.",
    )
    parser.add_argument(
        "--verbalizer_mode",
        type=str,
        default="multi",
        choices=["canonical", "multi"],
        help="Use one canonical label phrase or a multi-verbalizer label-card prototype.",
    )
    parser.add_argument(
        "--semantic_target_mode",
        type=str,
        default="class_token",
        choices=["class_token", "phrase"],
        help=(
            "In semantic label mode, train single class tokens by default; "
            "use 'phrase' only for the legacy pure phrase-likelihood diagnostic."
        ),
    )
    parser.add_argument(
        "--class_token_init",
        type=str,
        default="random",
        choices=["random", "semantic"],
        help="Initialize class-token rows randomly or from label-verbalizer prototypes.",
    )
    parser.add_argument(
        "--label_proto_source",
        type=str,
        default="token_mean",
        choices=["token_mean", "contextual_lm", "sentence_encoder"],
        help="Prototype source for semantic class-token priors. Only token_mean is implemented.",
    )
    parser.add_argument(
        "--semantic_row_reg_weight",
        type=float,
        default=None,
        help=(
            "Weight for class-token row-to-label-prototype regularization. "
            "Defaults to 0.01 for semantic class-token initialization and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--semantic_row_reg_type",
        type=str,
        default="cosine",
        choices=["cosine", "l2"],
        help="Distance used by semantic row regularization.",
    )
    parser.add_argument(
        "--semantic_decision_reg_weight",
        type=float,
        default=0.0,
        help="Optional decision-state-to-label-prototype contrastive regularization weight.",
    )
    parser.add_argument(
        "--semantic_decision_temperature",
        type=float,
        default=0.07,
        help="Temperature for optional decision-state semantic contrastive regularization.",
    )

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
        help="Local checkpoint path, e.g. results/curriculum_pretrain/.../best_accuracy.pt",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="newts_dual_branch",
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
    parser.add_argument("--vit_feature_mode", type=str, default="single", choices=["last", "single", "scalar_mix"],)
    parser.add_argument("--vit_layer_idx", type=int, default=4)
    parser.add_argument("--vit_mix_layers", type=str, default=None, help="Comma-separated 1-based layer indices used when vit_feature_mode=scalar_mix",)
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    parser.add_argument(
        "--vision_2d_mode",
        type=str,
        default=LEGACY_VISION_2D_MODE,
        choices=list(SUPPORTED_VISION_2D_MODES),
        help="1D->2D image construction for the NewTS vision branch",
    )
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
    parser.add_argument("--use_pma", action="store_true", help="Enable PMA slot aggregation for newts_dual_branch")
    parser.add_argument("--aggregator_layers", type=int, default=1)
    parser.add_argument("--aggregator_hidden_size", type=int, default=None)
    parser.add_argument("--aggregator_num_heads", type=int, default=8)
    parser.add_argument("--aggregator_ffn_dim", type=int, default=None)
    parser.add_argument("--aggregator_num_queries", type=int, default=2)
    parser.add_argument("--aggregator_query_mode", type=str, default="separate", choices=["shared", "separate"],)
    parser.add_argument("--aggregator_fusion_mode", type=str, default="concat_linear", choices=["gated_sum", "concat_linear"],)
    parser.add_argument("--aggregator_gate_type", type=str, default="dynamic", choices=["scalar", "slot", "dynamic"],)
    parser.add_argument("--aggregator_fuse_layers", type=int, default=1)
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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument(
        "--eval_max_samples_per_class",
        type=int,
        default=0,
        help=(
            "Pilot-mode evaluation cap: sample at most this many TEST examples per selected class. "
            "0 keeps the full query set."
        ),
    )
    parser.add_argument(
        "--eval_max_total_samples",
        type=int,
        default=0,
        help=(
            "Pilot-mode evaluation cap over the whole query set after per-class capping. "
            "0 keeps all selected query examples."
        ),
    )
    parser.add_argument(
        "--eval_subset_seed_offset",
        type=int,
        default=91000,
        help="Seed offset used for deterministic pilot evaluation subset sampling.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr_encoder", type=float, default=2e-4)
    parser.add_argument("--lr_projector", type=float, default=1e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument(
        "--llm_attn_impl",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention implementation for the LLM backbone.",
    )
    parser.add_argument(
        "--tokenizer_training_mode",
        type=str,
        default="class_rows",
        choices=["class_rows", "full_embedding_head"],
        help=(
            "Train only the added class-token rows (current behavior) or "
            "the full embedding/lm_head matrices (legacy pre-df831bd behavior)."
        ),
    )

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
        "--eval_decode_mode",
        type=str,
        default="generate",
        choices=["generate", "logits", "phrase_likelihood"],
        help=(
            "Evaluation prediction mode. 'generate' preserves the legacy constrained "
            "decoding path; 'logits' scores class tokens with one forward pass; "
            "'phrase_likelihood' scores semantic label verbalizers."
        ),
    )
    parser.add_argument(
        "--semantic_score_mode",
        type=str,
        default="calibrated",
        choices=["raw", "calibrated", "zero_cal", "support_cal"],
        help="Semantic phrase likelihood scoring mode.",
    )
    parser.add_argument(
        "--phrase_diag_score",
        type=str,
        default="raw",
        choices=["raw", "zero_cal", "support_cal"],
        help="Phrase-likelihood diagnostic calibration mode.",
    )
    parser.add_argument(
        "--phrase_diag_use_eos",
        action="store_true",
        help="Include EOS in phrase diagnostic scoring (disabled by default).",
    )
    parser.add_argument(
        "--label_shuffle_control",
        action="store_true",
        help="Shuffle phrase diagnostic label-to-class mapping as a control.",
    )
    parser.add_argument(
        "--prompt_label_order",
        type=str,
        default="fixed",
        choices=["fixed", "random"],
        help="Prompt label-card order. Random mode is reserved for diagnostic runs.",
    )
    parser.add_argument(
        "--disable_constrained_decoding",
        action="store_true",
        help="Disable constrained label-token decoding during evaluation.",
    )
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
    parser.add_argument("--save_dir", type=str, default="results/ucr_pretrained_fewshot")
    parser.add_argument("--resume", action="store_true", help="从已有 run_dir checkpoint 断点续训")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="每个 few-shot run 结束并写出结果后删除 phase checkpoint 以节省磁盘空间",
    )
    parser.add_argument(
        "--skip_phase_checkpoints",
        action="store_true",
        help=(
            "Do not write phase1/phase2 checkpoint .pt files. "
            "The final in-memory model is evaluated directly. Incompatible with --resume."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")

    parser.set_defaults(pin_memory=True, persistent_workers=True)
    args = parser.parse_args(argv)
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    args.encoder_type_explicit = cli_flag_was_provided(provided_argv, "--encoder_type")
    args.context_length_explicit = cli_flag_was_provided(provided_argv, "--context_length")
    args.pad_mode_explicit = cli_flag_was_provided(provided_argv, "--pad_mode")
    args.vit_patch_size_explicit = cli_flag_was_provided(provided_argv, "--vit_patch_size")
    args.vit_stride_explicit = cli_flag_was_provided(provided_argv, "--vit_stride")
    args.vision_2d_mode_explicit = cli_flag_was_provided(provided_argv, "--vision_2d_mode")
    args.eval_decode_mode_explicit = cli_flag_was_provided(provided_argv, "--eval_decode_mode")
    args.phrase_diag_score_explicit = cli_flag_was_provided(provided_argv, "--phrase_diag_score")
    args.semantic_row_reg_weight_explicit = cli_flag_was_provided(
        provided_argv,
        "--semantic_row_reg_weight",
    )
    if args.label_interface == "semantic" and not args.eval_decode_mode_explicit:
        args.eval_decode_mode = "logits" if args.semantic_target_mode == "class_token" else "phrase_likelihood"
    if args.semantic_row_reg_weight is None:
        args.semantic_row_reg_weight = (
            0.01
            if args.label_interface == "semantic"
            and args.semantic_target_mode == "class_token"
            and args.class_token_init == "semantic"
            else 0.0
        )
    if args.eval_decode_mode == "phrase_likelihood" and args.phrase_diag_score_explicit:
        args.semantic_score_mode = args.phrase_diag_score
    args.constrained_decoding = not args.disable_constrained_decoding
    return args


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
            "vision_2d_mode",
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


def use_anonymous_label_interface(args) -> bool:
    return getattr(args, "label_interface", "anonymous") == "anonymous"


def use_class_token_label_interface(args) -> bool:
    if use_anonymous_label_interface(args):
        return True
    return (
        getattr(args, "label_interface", "anonymous") == "semantic"
        and getattr(args, "semantic_target_mode", "class_token") == "class_token"
    )


def use_phrase_label_interface(args) -> bool:
    return (
        getattr(args, "label_interface", "anonymous") == "semantic"
        and getattr(args, "semantic_target_mode", "class_token") == "phrase"
    )


def use_class_token_row_training(args) -> bool:
    return (
        use_class_token_label_interface(args)
        and getattr(args, "tokenizer_training_mode", "class_rows") == "class_rows"
    )


def should_save_tokenizer_training_state(args) -> bool:
    return use_class_token_label_interface(args)


def semantic_priors_requested(args) -> bool:
    return (
        getattr(args, "class_token_init", "random") == "semantic"
        or float(getattr(args, "semantic_row_reg_weight", 0.0) or 0.0) > 0.0
        or float(getattr(args, "semantic_decision_reg_weight", 0.0) or 0.0) > 0.0
    )


def get_checkpoint_tokenizer_training_mode(checkpoint: Dict[str, Any]) -> str:
    checkpoint_args = checkpoint.get("args") or {}
    mode = checkpoint_args.get("tokenizer_training_mode")
    if mode in {"class_rows", "full_embedding_head"}:
        return mode
    if "class_token_embedding_rows" in checkpoint and "class_token_lm_head_rows" in checkpoint:
        return "class_rows"
    if "embedding_weight" in checkpoint and "lm_head_weight" in checkpoint:
        return "full_embedding_head"
    return "class_rows"


def save_tokenizer_training_state(model, checkpoint: Dict[str, Any], tokenizer_training_mode: str):
    if tokenizer_training_mode == "class_rows":
        save_class_token_rows_to_checkpoint(model, checkpoint)
        return

    checkpoint["embedding_weight"] = model.llm.get_input_embeddings().weight.detach().cpu()
    checkpoint["lm_head_weight"] = model.llm.lm_head.weight.detach().cpu()
    checkpoint["tokenizer_vocab_size"] = len(model.tokenizer)


def load_full_tokenizer_weights_from_checkpoint(model, checkpoint: Dict[str, Any], device: str) -> bool:
    if "embedding_weight" not in checkpoint or "lm_head_weight" not in checkpoint:
        return False

    embedding_weight = model.llm.get_input_embeddings().weight
    lm_head_weight = model.llm.lm_head.weight
    full_embedding = checkpoint["embedding_weight"].to(
        device=device,
        dtype=embedding_weight.dtype,
    )
    full_lm_head = checkpoint["lm_head_weight"].to(
        device=device,
        dtype=lm_head_weight.dtype,
    )
    if full_embedding.shape != embedding_weight.shape or full_lm_head.shape != lm_head_weight.shape:
        raise ValueError(
            "Full embedding checkpoint shape mismatch: "
            f"embedding {full_embedding.shape} vs {embedding_weight.shape}, "
            f"lm_head {full_lm_head.shape} vs {lm_head_weight.shape}"
        )

    with torch.no_grad():
        embedding_weight.copy_(full_embedding)
        lm_head_weight.copy_(full_lm_head)
    return True


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


def resolve_effective_pad_mode(args) -> str:
    if args.encoder_type == "newts_dual_branch" and not getattr(args, "pad_mode_explicit", False):
        return "last"
    return args.pad_mode


def resolve_effective_newts_vision_config(args) -> Dict[str, Any]:
    vision_2d_mode = validate_vision_2d_mode(args.vision_2d_mode)
    return {
        "vision_2d_mode": vision_2d_mode,
        "effective_vit_stride": resolve_effective_vision_stride(
            vision_2d_mode,
            args.vit_stride,
            stride_explicit=getattr(args, "vit_stride_explicit", False),
        ),
    }


def warn_deprecated_newts_context_length(args, rank: int):
    if rank != 0:
        return
    if args.encoder_type != "newts_dual_branch":
        return
    if getattr(args, "context_length_explicit", False):
        print("⚠️ --context_length is deprecated for newts_dual_branch and is ignored in dynamic-length mode")


def make_collate_fn(args, is_train: bool):
    def collate_fn(batch):
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=resolve_collate_patch_size(args),
            normalize=True,
            normalize_eps=1e-5,
            pad_mode=resolve_effective_pad_mode(args),
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


def build_dataloader_kwargs(args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return kwargs


def build_label_to_indices(dataset: Dataset) -> Dict[int, List[int]]:
    if hasattr(dataset, "get_int_labels"):
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(dataset.get_int_labels()):
            label_to_indices[int(label)].append(idx)
        return dict(label_to_indices)

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
    support_size = len(selected_indices)
    any_shortage = len(classes_with_shortage) > 0

    return {
        "selected_class_ids": selected_class_ids,
        "way": len(selected_class_ids),
        "selected_indices": selected_indices,
        "selected_by_class": selected_by_class,
        "class_train_counts": class_train_counts,
        "k_eff_per_class": k_eff_per_class,
        "classes_with_shortage": classes_with_shortage,
        "any_shortage": any_shortage,
        "support_size": support_size,
    }


def filter_indices_by_class_ids(
    label_to_indices: Dict[int, List[int]],
    class_ids: List[int],
) -> List[int]:
    selected_indices: List[int] = []
    for class_id in class_ids:
        selected_indices.extend(label_to_indices.get(class_id, []))
    return sorted(selected_indices)


def sample_query_indices_for_fast_eval(
    label_to_indices: Dict[int, List[int]],
    class_ids: List[int],
    *,
    max_samples_per_class: int = 0,
    max_total_samples: int = 0,
    seed: int = 0,
) -> Tuple[List[int], Dict[str, Any]]:
    rng = random.Random(seed)
    selected_by_class: Dict[int, List[int]] = {}
    for class_id in class_ids:
        indices = sorted(label_to_indices.get(class_id, []))
        if max_samples_per_class and len(indices) > max_samples_per_class:
            indices = sorted(rng.sample(indices, max_samples_per_class))
        selected_by_class[int(class_id)] = indices

    flattened = sorted(
        index
        for indices in selected_by_class.values()
        for index in indices
    )
    if max_total_samples and len(flattened) > max_total_samples:
        flattened = sorted(rng.sample(flattened, max_total_samples))

    selected_set = set(flattened)
    counts = {
        str(class_id): sum(1 for index in indices if index in selected_set)
        for class_id, indices in selected_by_class.items()
    }
    metadata = {
        "enabled": bool(max_samples_per_class or max_total_samples),
        "max_samples_per_class": int(max_samples_per_class),
        "max_total_samples": int(max_total_samples),
        "seed": int(seed),
        "class_counts": counts,
        "selected_indices": flattened,
    }
    return flattened, metadata


def summarize_subset_class_counts(
    label_to_indices: Dict[int, List[int]],
    class_ids: List[int],
) -> Dict[str, int]:
    return {
        str(class_id): len(label_to_indices.get(class_id, []))
        for class_id in class_ids
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
    pretrained_model_config = getattr(args, "pretrained_model_model_config", None) or {}
    if args.pretrained_model and pretrained_model_config.get("llm_id"):
        return pretrained_model_config["llm_id"]
    if args.pretrained_model:
        return OpenTSLM._get_base_llm_id(args.pretrained_model)
    return args.llm_id


def resolve_dataset_eos_token(args) -> str:
    base_llm_id = resolve_base_llm_id(args)
    tokenizer_source = resolve_local_hf_snapshot(base_llm_id)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=Path(tokenizer_source).exists(),
    )
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
    if args.way is not None and args.way < 1:
        raise ValueError("--way must be >= 1 when provided")
    if args.aug_scaling_min > args.aug_scaling_max:
        raise ValueError("--aug_scaling_min must be <= --aug_scaling_max")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader_num_workers must be >= 0")
    if args.eval_max_samples_per_class < 0:
        raise ValueError("--eval_max_samples_per_class must be >= 0")
    if args.eval_max_total_samples < 0:
        raise ValueError("--eval_max_total_samples must be >= 0")
    if args.semantic_row_reg_weight < 0:
        raise ValueError("--semantic_row_reg_weight must be >= 0")
    if args.semantic_decision_reg_weight < 0:
        raise ValueError("--semantic_decision_reg_weight must be >= 0")
    if args.semantic_decision_temperature <= 0:
        raise ValueError("--semantic_decision_temperature must be > 0")
    if args.class_token_init == "semantic" and args.label_proto_source != "token_mean":
        raise NotImplementedError(
            "Only --label_proto_source token_mean is currently implemented for semantic class-token initialization."
        )
    if args.semantic_row_reg_weight > 0 and args.label_proto_source != "token_mean":
        raise NotImplementedError(
            "Only --label_proto_source token_mean is currently implemented for semantic row regularization."
        )
    if args.phrase_diag_use_eos:
        raise NotImplementedError("--phrase_diag_use_eos is reserved for a future diagnostic implementation.")

    if args.pretrained_model and getattr(args, "encoder_type_explicit", False):
        raise ValueError("--pretrained_model and --encoder_type cannot be specified together")
    if args.resume and args.skip_phase_checkpoints:
        raise ValueError("--resume cannot be used with --skip_phase_checkpoints")
    if args.label_interface == "semantic":
        if args.dataset_family not in {"mitbih", "sleepedf", "cinc2017af"}:
            raise ValueError(
                "--label_interface semantic is supported only for mitbih, sleepedf, and cinc2017af"
            )
        if args.semantic_target_mode == "phrase" and args.eval_decode_mode != "phrase_likelihood":
            raise ValueError(
                "--semantic_target_mode phrase requires --eval_decode_mode phrase_likelihood"
            )
        if args.semantic_target_mode == "class_token" and args.eval_decode_mode == "phrase_likelihood":
            # Supported as a diagnostic: train class tokens, evaluate phrase likelihood.
            pass
        if args.tokenizer_training_mode == "class_rows":
            # In semantic class-token mode this restricts adaptation to the label rows.
            # In phrase mode no rows are registered; the flag remains in the config for compatibility.
            pass
    elif args.eval_decode_mode == "phrase_likelihood":
        raise ValueError("--eval_decode_mode phrase_likelihood requires --label_interface semantic")
    if args.prompt_label_order == "random":
        raise NotImplementedError("--prompt_label_order random is reserved for prompt-order diagnostics.")

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
    validate_vision_2d_mode(args.vision_2d_mode)

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
    resolved_vision_config = resolve_effective_newts_vision_config(args)
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
        "vit_stride": resolved_vision_config["effective_vit_stride"],
        "vision_2d_mode": resolved_vision_config["vision_2d_mode"],
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
        if getattr(args, "vision_2d_mode_explicit", False):
            resolved_vision_config = resolve_effective_newts_vision_config(args)
            merged_config["vision_2d_mode"] = resolved_vision_config["vision_2d_mode"]
            merged_config["vit_stride"] = resolved_vision_config["effective_vit_stride"]
        elif getattr(args, "vit_stride_explicit", False):
            merged_config["vit_stride"] = args.vit_stride
        merged_config["output_dim"] = ENCODER_OUTPUT_DIM
        init_kwargs["newts_dual_branch_config"] = merged_config

    return init_kwargs


def extract_sp_component_states_from_checkpoint(
    checkpoint: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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

    raise KeyError(
        "Checkpoint does not contain encoder/projector weights. "
        "Expected either top-level 'encoder_state'/'projector_state' or "
        "a full 'model_state' with 'encoder.' and 'projector.' prefixes."
    )


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
            llm_attn_impl=args.llm_attn_impl,
        )

        encoder_state, projector_state = extract_sp_component_states_from_checkpoint(checkpoint)
        model.encoder.load_state_dict(encoder_state)
        model.projector.load_state_dict(projector_state)
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
            checkpoint_path=getattr(args, "pretrained_model_checkpoint", None),
            llm_attn_impl=args.llm_attn_impl,
        )
        if (
            (getattr(args, "vision_2d_mode_explicit", False) or getattr(args, "vit_stride_explicit", False))
            and getattr(model, "encoder_type", None) == "newts_dual_branch"
            and getattr(model.encoder, "vision_encoder", None) is not None
        ):
            resolved_vision_config = resolve_effective_newts_vision_config(args)
            if getattr(args, "vision_2d_mode_explicit", False):
                model.encoder.vision_encoder.vision_2d_mode = resolved_vision_config["vision_2d_mode"]
            model.encoder.vision_encoder.ts_stride = resolved_vision_config["effective_vit_stride"]

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
            llm_attn_impl=args.llm_attn_impl,
        )

        if use_lora:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    if args.random_init_llm:
        if rank == 0:
            print("🎲 Randomly initializing LLM weights...")
        from transformers import AutoModelForCausalLM

        llm_config = model.llm.config
        attn_candidates = [args.llm_attn_impl]
        if args.llm_attn_impl == "flash_attention_2":
            attn_candidates.extend(["sdpa", "eager"])
        elif args.llm_attn_impl != "eager":
            attn_candidates.append("eager")

        random_llm = None
        last_error = None
        resolved_attn_impl = args.llm_attn_impl
        for candidate in attn_candidates:
            try:
                random_llm = AutoModelForCausalLM.from_config(
                    llm_config,
                    torch_dtype=torch.bfloat16,
                    attn_implementation=candidate,
                ).to(device)
                resolved_attn_impl = candidate
                if candidate != args.llm_attn_impl and rank == 0:
                    print(
                        f"⚠️ Failed to reinitialize LLM with attn_implementation={args.llm_attn_impl}; "
                        f"falling back to {candidate}."
                    )
                break
            except Exception as exc:
                last_error = exc

        if random_llm is None:
            raise RuntimeError(
                f"Failed to reinitialize LLM with attention implementations {attn_candidates}: {last_error}"
            ) from last_error
        model.llm = random_llm
        model.llm_attn_impl = resolved_attn_impl

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

    if hasattr(model, "set_runtime_branch_mode"):
        model.set_runtime_branch_mode(args.runtime_branch_mode)
        if rank == 0:
            print(f"🌿 runtime_branch_mode: {args.runtime_branch_mode}")

    return model


def _normalize_label_text(text: str) -> str:
    return str(text).strip()


def calculate_accuracy(
    predictions: List[str],
    labels: List[str],
    label_to_class_id: Optional[Dict[str, int]] = None,
) -> float:
    correct = 0
    for pred, label in zip(predictions, labels):
        pred_clean = _normalize_label_text(pred)
        label_clean = _normalize_label_text(label)

        if label_to_class_id:
            pred_key = extract_class_token(pred_clean) or pred_clean
            label_key = extract_class_token(label_clean) or label_clean
            label_id = label_to_class_id.get(label_key)
            pred_id = label_to_class_id.get(pred_key)
            if label_id is not None and pred_id == label_id:
                correct += 1
        else:
            pred_token = extract_class_token(pred_clean) or pred_clean
            if pred_token == label_clean:
                correct += 1

    return correct / len(predictions) if predictions else 0.0


def calculate_macro_f1(
    predictions: List[str],
    labels: List[str],
    label_to_class_id: Optional[Dict[str, int]] = None,
) -> float:
    if not predictions or not labels:
        return 0.0

    true_ids: List[int] = []
    pred_ids: List[int] = []
    for pred, label in zip(predictions, labels):
        true_token = extract_class_token(label) or label.strip()
        pred_token = extract_class_token(pred) or pred.strip()
        if label_to_class_id:
            true_id = label_to_class_id.get(true_token)
            pred_id = label_to_class_id.get(pred_token)
        else:
            true_id = class_token_to_index(true_token)
            pred_id = class_token_to_index(pred_token)
        if true_id is None:
            continue
        true_ids.append(int(true_id))
        pred_ids.append(-1 if pred_id is None else int(pred_id))

    if not true_ids:
        return 0.0

    label_space = sorted(set(true_ids))
    return float(
        f1_score(
            true_ids,
            pred_ids,
            labels=label_space,
            average="macro",
            zero_division=0.0,
        )
    )


def add_class_tokens_to_model(
    model,
    num_classes: int,
    tokenizer_training_mode: str,
    rank: int = 0,
):
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

    class_token_ids = [model.tokenizer.convert_tokens_to_ids(t) for t in class_tokens]
    if rank == 0:
        preview = class_token_ids[:5]
        suffix = "..." if len(class_token_ids) > 5 else ""
        print(f"   Class token IDs: {preview}{suffix}")

    if tokenizer_training_mode == "class_rows":
        register_class_token_row_training(model, class_token_ids)
        if rank == 0:
            print("   Restricted embedding/lm_head updates to class-token rows only")
    else:
        embedding.weight.requires_grad = True
        lm_head.weight.requires_grad = True
        if rank == 0:
            print("   Enabled full embedding/lm_head training (legacy behavior)")

    return class_tokens, class_token_ids


def _mean_vocab_embedding_norm(
    embedding_weight: torch.Tensor,
    excluded_token_ids: List[int],
) -> torch.Tensor:
    if not excluded_token_ids:
        return embedding_weight.norm(dim=-1).mean()
    mask = torch.ones(
        embedding_weight.shape[0],
        device=embedding_weight.device,
        dtype=torch.bool,
    )
    valid_ids = [
        int(token_id)
        for token_id in excluded_token_ids
        if 0 <= int(token_id) < embedding_weight.shape[0]
    ]
    if valid_ids:
        mask[torch.tensor(valid_ids, device=embedding_weight.device, dtype=torch.long)] = False
    return embedding_weight[mask].norm(dim=-1).mean()


def build_token_mean_label_prototypes(
    model,
    *,
    label_verbalizers: Dict[int, List[str]],
    class_token_ids: List[int],
) -> Tuple[List[int], torch.Tensor]:
    embedding = model.llm.get_input_embeddings()
    embedding_weight = embedding.weight.detach()
    target_norm = _mean_vocab_embedding_norm(embedding_weight, class_token_ids)

    class_ids: List[int] = []
    prototypes: List[torch.Tensor] = []
    for class_id in sorted(label_verbalizers.keys()):
        if class_id >= len(class_token_ids):
            continue
        phrase_vectors: List[torch.Tensor] = []
        for verbalizer in label_verbalizers[class_id]:
            token_ids = model.tokenizer.encode(
                str(verbalizer).strip(),
                add_special_tokens=False,
            )
            token_ids = [
                int(token_id)
                for token_id in token_ids
                if 0 <= int(token_id) < embedding_weight.shape[0]
            ]
            if not token_ids:
                continue
            token_tensor = torch.tensor(
                token_ids,
                device=embedding_weight.device,
                dtype=torch.long,
            )
            phrase_vectors.append(embedding_weight.index_select(0, token_tensor).mean(dim=0))
        if not phrase_vectors:
            continue
        prototype = torch.stack(phrase_vectors, dim=0).mean(dim=0)
        prototype = prototype * (target_norm / prototype.norm().clamp_min(1e-6))
        class_ids.append(int(class_id))
        prototypes.append(prototype)

    if not prototypes:
        raise ValueError("No semantic label prototypes could be built from label_verbalizers.")
    return class_ids, torch.stack(prototypes, dim=0).detach()


def configure_semantic_label_priors(
    model,
    *,
    args,
    class_token_ids: List[int],
    label_verbalizers: Dict[int, List[str]],
    selected_class_ids: List[int],
    rank: int = 0,
) -> Dict[str, Any]:
    if not use_class_token_label_interface(args) or not semantic_priors_requested(args):
        return {}
    if not label_verbalizers:
        raise ValueError("Semantic class-token priors require non-empty label_verbalizers.")
    if args.label_proto_source != "token_mean":
        raise NotImplementedError("Only token_mean label prototypes are implemented.")

    all_class_ids, all_prototypes = build_token_mean_label_prototypes(
        model,
        label_verbalizers=label_verbalizers,
        class_token_ids=class_token_ids,
    )
    class_id_to_proto = {
        class_id: all_prototypes[index]
        for index, class_id in enumerate(all_class_ids)
    }
    usable_selected_class_ids = [
        int(class_id)
        for class_id in selected_class_ids
        if int(class_id) in class_id_to_proto and int(class_id) < len(class_token_ids)
    ]
    if not usable_selected_class_ids:
        raise ValueError("No selected classes have semantic prototypes.")

    selected_prototypes = torch.stack(
        [class_id_to_proto[class_id] for class_id in usable_selected_class_ids],
        dim=0,
    ).to(device=model.llm.get_input_embeddings().weight.device)
    selected_token_ids = [int(class_token_ids[class_id]) for class_id in usable_selected_class_ids]

    if args.class_token_init == "semantic":
        row_index = torch.tensor(
            [int(class_token_ids[class_id]) for class_id in all_class_ids],
            device=model.llm.get_input_embeddings().weight.device,
            dtype=torch.long,
        )
        init_rows = all_prototypes.to(
            device=model.llm.get_input_embeddings().weight.device,
            dtype=model.llm.get_input_embeddings().weight.dtype,
        )
        with torch.no_grad():
            input_weight = model.llm.get_input_embeddings().weight
            input_weight.index_copy_(0, row_index, init_rows.to(dtype=input_weight.dtype))
            lm_head_weight = model.llm.lm_head.weight
            if lm_head_weight.data_ptr() != input_weight.data_ptr():
                lm_head_weight.index_copy_(0, row_index, init_rows.to(dtype=lm_head_weight.dtype))
        if rank == 0:
            print(
                f"   Initialized {len(all_class_ids)} class tokens from "
                f"{args.verbalizer_mode} label prototypes"
            )

    setattr(model, "_semantic_prior_class_ids", tuple(usable_selected_class_ids))
    setattr(model, "_semantic_prior_token_ids", tuple(selected_token_ids))
    setattr(model, "_semantic_prior_prototypes", selected_prototypes.detach())
    setattr(
        model,
        "_semantic_prior_class_id_to_position",
        {class_id: index for index, class_id in enumerate(usable_selected_class_ids)},
    )
    metadata = {
        "enabled": True,
        "class_token_init": args.class_token_init,
        "label_proto_source": args.label_proto_source,
        "verbalizer_mode": args.verbalizer_mode,
        "selected_class_ids": usable_selected_class_ids,
        "semantic_row_reg_weight": float(args.semantic_row_reg_weight),
        "semantic_row_reg_type": args.semantic_row_reg_type,
        "semantic_decision_reg_weight": float(args.semantic_decision_reg_weight),
        "semantic_decision_temperature": float(args.semantic_decision_temperature),
    }
    setattr(model, "_semantic_prior_metadata", metadata)
    return metadata


def compute_semantic_row_regularization(model, args) -> torch.Tensor:
    prototypes = getattr(model, "_semantic_prior_prototypes", None)
    token_ids = getattr(model, "_semantic_prior_token_ids", None)
    weight = float(getattr(args, "semantic_row_reg_weight", 0.0) or 0.0)
    if weight <= 0.0 or prototypes is None or not token_ids:
        return torch.zeros((), device=model.device)

    token_tensor = torch.tensor(
        list(token_ids),
        device=model.llm.get_input_embeddings().weight.device,
        dtype=torch.long,
    )
    prototypes = prototypes.to(
        device=token_tensor.device,
        dtype=model.llm.get_input_embeddings().weight.dtype,
    )
    input_rows = model.llm.get_input_embeddings().weight.index_select(0, token_tensor)
    row_reg_type = getattr(args, "semantic_row_reg_type", "cosine")
    if row_reg_type == "cosine":
        loss = (1.0 - F.cosine_similarity(input_rows.float(), prototypes.float(), dim=-1)).mean()
    elif row_reg_type == "l2":
        loss = F.mse_loss(input_rows.float(), prototypes.float())
    else:
        raise ValueError(f"Unsupported semantic_row_reg_type: {row_reg_type}")

    lm_head_weight = model.llm.lm_head.weight
    if lm_head_weight.data_ptr() != model.llm.get_input_embeddings().weight.data_ptr():
        head_rows = lm_head_weight.index_select(0, token_tensor)
        head_prototypes = prototypes.to(dtype=head_rows.dtype)
        if row_reg_type == "cosine":
            loss = loss + (
                1.0 - F.cosine_similarity(head_rows.float(), head_prototypes.float(), dim=-1)
            ).mean()
        else:
            loss = loss + F.mse_loss(head_rows.float(), head_prototypes.float())
    return loss.to(device=model.device) * weight


def compute_semantic_decision_regularization(
    model,
    batch: List[Dict[str, Any]],
    args,
) -> torch.Tensor:
    prototypes = getattr(model, "_semantic_prior_prototypes", None)
    class_id_to_position = getattr(model, "_semantic_prior_class_id_to_position", None)
    weight = float(getattr(args, "semantic_decision_reg_weight", 0.0) or 0.0)
    if weight <= 0.0 or prototypes is None or not class_id_to_position:
        return torch.zeros((), device=model.device)

    target_positions: List[int] = []
    for sample in batch:
        class_id = int(sample["int_label"])
        if class_id not in class_id_to_position:
            return torch.zeros((), device=model.device)
        target_positions.append(int(class_id_to_position[class_id]))

    inputs_embeds, attention_mask = model.pad_and_apply_batch(batch)
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    last_positions = attention_mask.to(outputs.hidden_states[-1].device).long().sum(dim=1) - 1
    batch_indices = torch.arange(outputs.hidden_states[-1].size(0), device=last_positions.device)
    decision_states = outputs.hidden_states[-1][batch_indices, last_positions, :].float()
    proto = prototypes.to(device=decision_states.device, dtype=decision_states.dtype)
    logits = torch.matmul(
        F.normalize(decision_states, dim=-1),
        F.normalize(proto, dim=-1).transpose(0, 1),
    ) / float(args.semantic_decision_temperature)
    targets = torch.tensor(target_positions, device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits, targets) * weight


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

    if use_class_token_row_training(args):
        tokenizer_params = unique_params(
            list(get_class_token_trainable_parameters(underlying_model))
        )
        if tokenizer_params:
            param_groups.append(
                {
                    "params": tokenizer_params,
                    "lr": args.lr_lora * 2,
                    "weight_decay": 0.0,
                }
            )
    else:
        tokenizer_params = unique_params(
            [
                underlying_model.llm.get_input_embeddings().weight,
                underlying_model.llm.lm_head.weight,
            ]
        )
        if tokenizer_params:
            param_groups.append({"params": tokenizer_params, "lr": args.lr_lora * 2})

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


def answer_token_nll_from_prompt(
    underlying_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    answers: List[str],
) -> torch.Tensor:
    B, L, _H = inputs_embeds.size()
    ans_tok = underlying_model.tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False,
    )
    ans_ids = ans_tok.input_ids.to(underlying_model.device, non_blocking=True)
    ans_mask = ans_tok.attention_mask.to(underlying_model.device, non_blocking=True)
    ans_emb = underlying_model.llm.get_input_embeddings()(ans_ids)

    full_inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)
    full_attention_mask = torch.cat([attention_mask, ans_mask], dim=1)

    labels = torch.full(
        (B, full_attention_mask.size(1)),
        -100,
        device=underlying_model.device,
        dtype=torch.long,
    )
    labels[:, L:] = torch.where(
        ans_mask.bool(),
        ans_ids,
        torch.full_like(ans_ids, -100),
    )

    outputs = underlying_model.llm(
        inputs_embeds=full_inputs_embeds,
        attention_mask=full_attention_mask,
        return_dict=True,
    )
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)
    token_losses = F.cross_entropy(
        shift_logits.float().reshape(-1, shift_logits.size(-1)),
        safe_labels.reshape(-1),
        reduction="none",
    ).view(B, -1)
    token_counts = valid_mask.sum(dim=1).clamp_min(1)
    return (token_losses * valid_mask).sum(dim=1) / token_counts


def compute_training_loss(model, batch: List[Dict[str, Any]], args) -> torch.Tensor:
    loss = model(batch)
    underlying_model = get_model(model)
    row_reg = compute_semantic_row_regularization(underlying_model, args)
    decision_reg = compute_semantic_decision_regularization(underlying_model, batch, args)
    return loss + row_reg + decision_reg


def score_phrase_batch(
    underlying_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    phrase: str,
) -> torch.Tensor:
    batch_size = inputs_embeds.size(0)
    nll = answer_token_nll_from_prompt(
        underlying_model,
        inputs_embeds,
        attention_mask,
        [phrase] * batch_size,
    )
    return -nll


def score_candidate_verbalizers(
    underlying_model,
    batch: List[Dict[str, Any]],
    candidate_verbalizers: List[List[str]],
) -> torch.Tensor:
    if not candidate_verbalizers:
        raise ValueError("candidate_verbalizers must not be empty")

    inputs_embeds, attention_mask = underlying_model.pad_and_apply_batch(batch)
    class_scores: List[torch.Tensor] = []
    for verbalizers in candidate_verbalizers:
        if not verbalizers:
            raise ValueError("Each candidate class must have at least one verbalizer")
        verbalizer_scores = torch.stack(
            [
                score_phrase_batch(
                    underlying_model,
                    inputs_embeds,
                    attention_mask,
                    verbalizer,
                )
                for verbalizer in verbalizers
            ],
            dim=1,
        )
        class_scores.append(
            torch.logsumexp(verbalizer_scores, dim=1) - math.log(len(verbalizers))
        )
    return torch.stack(class_scores, dim=1)


def build_null_time_series_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    null_batch: List[Dict[str, Any]] = []
    for sample in batch:
        null_sample = dict(sample)
        null_sample["time_series"] = [
            torch.zeros_like(torch.as_tensor(series)) for series in sample["time_series"]
        ]
        null_batch.append(null_sample)
    return null_batch


def score_semantic_candidates(
    underlying_model,
    batch: List[Dict[str, Any]],
    candidate_verbalizers: List[List[str]],
    semantic_score_mode: str,
    support_calibration_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    raw_scores = score_candidate_verbalizers(
        underlying_model,
        batch,
        candidate_verbalizers,
    )
    if semantic_score_mode == "raw":
        return raw_scores
    if semantic_score_mode in {"support_cal"}:
        if support_calibration_scores is None:
            raise ValueError("support_cal scoring requires support_calibration_scores")
        return raw_scores - support_calibration_scores.to(
            device=raw_scores.device,
            dtype=raw_scores.dtype,
        ).unsqueeze(0)
    if semantic_score_mode in {"calibrated", "zero_cal"}:
        null_scores = score_candidate_verbalizers(
            underlying_model,
            build_null_time_series_batch(batch),
            candidate_verbalizers,
        )
        return raw_scores - null_scores
    raise ValueError(f"Unsupported semantic_score_mode: {semantic_score_mode}")


@torch.no_grad()
def estimate_support_phrase_calibration(
    model,
    data_loader: DataLoader,
    candidate_verbalizers: List[List[str]],
    *,
    rank: int = 0,
) -> torch.Tensor:
    underlying_model = get_model(model)
    underlying_model.eval()
    score_sum: Optional[torch.Tensor] = None
    sample_count = 0
    for batch in tqdm(data_loader, desc="Estimating support phrase bias", disable=(rank != 0)):
        scores = score_candidate_verbalizers(
            underlying_model,
            batch,
            candidate_verbalizers,
        )
        batch_sum = scores.detach().float().sum(dim=0)
        score_sum = batch_sum if score_sum is None else score_sum + batch_sum
        sample_count += scores.size(0)
    if score_sum is None or sample_count == 0:
        raise ValueError("Cannot estimate support phrase calibration from an empty support loader")
    return score_sum / float(sample_count)


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    grad_clip: float,
    epoch_idx: int,
    epoch_total: int,
    gradient_accumulation_steps: int,
    args,
    rank: int,
    phase_name: str,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch_idx}/{epoch_total}", disable=(rank != 0))
    for step, batch in enumerate(pbar):
        loss = compute_training_loss(model, batch, args)
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
    disable_constrained_decoding: bool = False,
    eval_decode_mode: str = "generate",
    label_verbalizers: Optional[Dict[int, List[str]]] = None,
    selected_class_ids: Optional[List[int]] = None,
    semantic_score_mode: str = "raw",
    support_calibration_scores: Optional[torch.Tensor] = None,
    label_to_class_id: Optional[Dict[str, int]] = None,
    desc: str = "Testing",
    rank: int = 0,
) -> Dict[str, Any]:
    underlying_model = get_model(model)
    underlying_model.eval()

    total_loss = 0.0
    loss_denominator = 0
    all_predictions: List[str] = []
    all_labels: List[str] = []
    eval_decode_mode = str(eval_decode_mode).lower()
    if eval_decode_mode not in {"generate", "logits", "phrase_likelihood"}:
        raise ValueError(f"Unsupported eval_decode_mode: {eval_decode_mode}")
    if eval_decode_mode == "logits" and not class_token_ids:
        raise ValueError("eval_decode_mode='logits' requires non-empty class_token_ids")
    if eval_decode_mode == "phrase_likelihood":
        if not label_verbalizers:
            raise ValueError("eval_decode_mode='phrase_likelihood' requires label_verbalizers")
        if not label_to_class_id:
            raise ValueError("eval_decode_mode='phrase_likelihood' requires label_to_class_id")
    eval_loss_type = {
        "logits": "class_token_ce",
        "phrase_likelihood": f"semantic_phrase_{semantic_score_mode}_ce",
    }.get(eval_decode_mode, "lm_answer_ce")

    candidate_class_ids: List[int] = []
    candidate_verbalizers: List[List[str]] = []
    if eval_decode_mode == "phrase_likelihood":
        assert label_verbalizers is not None
        candidate_class_ids = list(selected_class_ids or sorted(label_verbalizers.keys()))
        candidate_verbalizers = [label_verbalizers[class_id] for class_id in candidate_class_ids]
        if any(not verbalizers for verbalizers in candidate_verbalizers):
            raise ValueError("Every selected class must have at least one label verbalizer")

    logits_processor = None
    if (
        eval_decode_mode == "generate"
        and class_token_ids is not None
        and not disable_constrained_decoding
    ):
        eos_token_id = underlying_model.tokenizer.eos_token_id
        allowed_ids = class_token_ids + [eos_token_id]
        logits_processor = LogitsProcessorList([AllowedTokensLogitsProcessor(allowed_ids)])

    for batch in tqdm(data_loader, desc=desc, disable=(rank != 0)):
        labels = [
            sample["answer"].replace(underlying_model.get_eos_token(), "").strip()
            for sample in batch
        ]

        if eval_decode_mode == "phrase_likelihood":
            scores = score_semantic_candidates(
                underlying_model,
                batch,
                candidate_verbalizers,
                semantic_score_mode=semantic_score_mode,
                support_calibration_scores=support_calibration_scores,
            )
            class_id_to_position = {
                class_id: position for position, class_id in enumerate(candidate_class_ids)
            }
            target_positions = []
            for label in labels:
                label_key = extract_class_token(label) or label.strip()
                class_id = label_to_class_id.get(label_key) if label_to_class_id else None
                if class_id is None or class_id not in class_id_to_position:
                    raise ValueError(
                        f"Label {label_key!r} cannot be mapped to a selected semantic class"
                    )
                target_positions.append(class_id_to_position[class_id])
            targets = torch.tensor(
                target_positions,
                device=scores.device,
                dtype=torch.long,
            )
            loss = F.cross_entropy(scores.float(), targets)
            total_loss += loss.item() * len(batch)
            loss_denominator += len(batch)

            pred_positions = scores.argmax(dim=-1).detach().cpu().tolist()
            predictions = [
                candidate_verbalizers[int(position)][0] for position in pred_positions
            ]
        elif eval_decode_mode == "logits":
            inputs_embeds, attention_mask = underlying_model.pad_and_apply_batch(batch)
            outputs = underlying_model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            class_token_tensor = torch.tensor(
                class_token_ids,
                device=outputs.logits.device,
                dtype=torch.long,
            )
            last_token_positions = attention_mask.to(outputs.logits.device).long().sum(dim=1) - 1
            batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
            next_token_logits = outputs.logits[batch_indices, last_token_positions, :]
            class_logits = next_token_logits.index_select(dim=-1, index=class_token_tensor)

            token_id_to_position = {
                int(token_id): position for position, token_id in enumerate(class_token_ids)
            }
            target_positions: List[int] = []
            for label in labels:
                label_token = extract_class_token(label) or label.strip()
                label_token_id = int(underlying_model.tokenizer.convert_tokens_to_ids(label_token))
                if label_token_id not in token_id_to_position:
                    raise ValueError(
                        f"Label token {label_token!r} is not in selected class_token_ids"
                    )
                target_positions.append(token_id_to_position[label_token_id])
            targets = torch.tensor(
                target_positions,
                device=class_logits.device,
                dtype=torch.long,
            )
            loss = F.cross_entropy(class_logits.float(), targets)
            total_loss += loss.item() * len(batch)
            loss_denominator += len(batch)

            pred_positions = class_logits.argmax(dim=-1)
            pred_token_ids = class_token_tensor[pred_positions].detach().cpu().tolist()
            predictions = underlying_model.tokenizer.convert_ids_to_tokens(pred_token_ids)
            if isinstance(predictions, str):
                predictions = [predictions]
        else:
            loss = underlying_model.compute_loss(batch)
            total_loss += loss.item()
            loss_denominator += 1

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
                    predictions.append(extract_class_token(pred) or pred.strip())
            else:
                decoded_predictions = underlying_model.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    skip_special_tokens=False,
                )
                predictions = []
                for pred in decoded_predictions:
                    predictions.append(extract_class_token(pred) or pred.strip())

        for label, pred in zip(labels, predictions):
            all_predictions.append(pred)
            all_labels.append(label)

    avg_loss = total_loss / max(loss_denominator, 1)
    accuracy = calculate_accuracy(all_predictions, all_labels, label_to_class_id=label_to_class_id)
    macro_f1 = calculate_macro_f1(all_predictions, all_labels, label_to_class_id=label_to_class_id)

    return {
        "loss": avg_loss,
        "eval_decode_mode": eval_decode_mode,
        "eval_loss_type": eval_loss_type,
        "semantic_score_mode": semantic_score_mode if eval_decode_mode == "phrase_likelihood" else None,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
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
    extra_state: Optional[Dict[str, Any]] = None,
    rank: int = 0,
):
    if rank != 0:
        return

    underlying_model = get_model(model)
    if optimizer is not None and use_class_token_row_training(args):
        sanitize_class_token_optimizer_state(optimizer, underlying_model)
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
    }
    if extra_state:
        checkpoint.update(extra_state)
    underlying_model.save_lora_state_to_checkpoint(checkpoint)
    if should_save_tokenizer_training_state(args):
        save_tokenizer_training_state(
            underlying_model,
            checkpoint,
            args.tokenizer_training_mode,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_save_path = f"{save_path}.tmp.{os.getpid()}"
    try:
        torch.save(checkpoint, tmp_save_path)
        os.replace(tmp_save_path, save_path)
    except (OSError, RuntimeError) as exc:
        message = str(exc)
        is_write_failure = (
            "PytorchStreamWriter failed writing file" in message
            or "unexpected pos" in message
            or "No space left on device" in message
        )
        partial_size = os.path.getsize(tmp_save_path) if os.path.exists(tmp_save_path) else 0
        try:
            free_bytes = shutil.disk_usage(os.path.dirname(save_path)).free
        except OSError:
            free_bytes = None

        if os.path.exists(tmp_save_path):
            try:
                os.remove(tmp_save_path)
            except OSError:
                pass

        if is_write_failure:
            free_text = "unknown" if free_bytes is None else f"{free_bytes / (1024 ** 3):.2f} GiB"
            partial_text = (
                f"{partial_size / (1024 ** 2):.2f} MiB"
                if partial_size > 0
                else "0 MiB"
            )
            raise RuntimeError(
                "Checkpoint save failed while writing "
                f"{save_path}. This usually means the destination disk is full. "
                f"Free space before the failure: {free_text}; partial checkpoint size: {partial_text}. "
                "The incomplete checkpoint file has been removed when possible. "
                "If you only need evaluation metrics, rerun with --skip_phase_checkpoints."
            ) from exc

        raise


def is_recoverable_checkpoint_load_error(exc: BaseException) -> bool:
    message = str(exc)
    recoverable_patterns = (
        "PytorchStreamReader failed reading zip archive",
        "failed finding central directory",
        "failed reading zip archive",
        "invalid header or archive is corrupted",
        "not a ZIP archive",
        "Ran out of input",
    )
    return any(pattern in message for pattern in recoverable_patterns)


def quarantine_corrupt_checkpoint(checkpoint_path: str, rank: int = 0) -> Optional[str]:
    if rank != 0 or not os.path.exists(checkpoint_path):
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantined_path = f"{checkpoint_path}.corrupt_{timestamp}"
    suffix = 1
    while os.path.exists(quarantined_path):
        quarantined_path = f"{checkpoint_path}.corrupt_{timestamp}_{suffix}"
        suffix += 1
    try:
        os.replace(checkpoint_path, quarantined_path)
        return quarantined_path
    except OSError as exc:
        print(f"⚠️ Failed to quarantine corrupt checkpoint {checkpoint_path}: {exc}")
        return None


def quarantine_corrupt_json(json_path: str, rank: int = 0) -> Optional[str]:
    if rank != 0 or not os.path.exists(json_path):
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantined_path = f"{json_path}.corrupt_{timestamp}"
    suffix = 1
    while os.path.exists(quarantined_path):
        quarantined_path = f"{json_path}.corrupt_{timestamp}_{suffix}"
        suffix += 1
    try:
        os.replace(json_path, quarantined_path)
        return quarantined_path
    except OSError as exc:
        print(f"⚠️ Failed to quarantine corrupt JSON {json_path}: {exc}")
        return None


def load_json_or_quarantine(json_path: str, *, rank: int = 0, description: str = "JSON"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        quarantined_path = quarantine_corrupt_json(json_path, rank=rank)
        if rank == 0:
            target_text = quarantined_path or "not moved"
            print(f"⚠️  {description} is unreadable and will be ignored: {json_path}")
            print(f"   quarantined: {target_text}")
            print(f"   reason: {exc}")
        return None


def atomic_write_json(save_path: str, payload: Any, *, indent: Optional[int] = 2):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tmp_save_path = f"{save_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp_save_path, save_path)
    except Exception:
        if os.path.exists(tmp_save_path):
            try:
                os.remove(tmp_save_path)
            except OSError:
                pass
        raise


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: str,
    tokenizer_training_mode: str,
    label_interface: str = "anonymous",
    semantic_target_mode: str = "class_token",
    optimizer=None,
    scheduler=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    underlying_model = get_model(model)
    uses_class_tokens = label_interface == "anonymous" or semantic_target_mode == "class_token"
    if uses_class_tokens:
        checkpoint_mode = get_checkpoint_tokenizer_training_mode(checkpoint)
        if checkpoint_mode != tokenizer_training_mode:
            raise ValueError(
                "Checkpoint tokenizer_training_mode mismatch: "
                f"expected {tokenizer_training_mode}, got {checkpoint_mode}. "
                "Please resume/evaluate with the matching --tokenizer_training_mode."
            )

    underlying_model.encoder.load_state_dict(checkpoint["encoder_state"])
    underlying_model.projector.load_state_dict(checkpoint["projector_state"])
    underlying_model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
    if uses_class_tokens:
        if tokenizer_training_mode == "class_rows":
            load_class_token_rows_from_checkpoint(underlying_model, checkpoint, device=device)
        elif not load_full_tokenizer_weights_from_checkpoint(underlying_model, checkpoint, device=device):
            raise ValueError(
                f"Checkpoint {checkpoint_path} does not contain full embedding/lm_head weights."
            )

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if uses_class_tokens and tokenizer_training_mode == "class_rows":
            sanitize_class_token_optimizer_state(optimizer, underlying_model)
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    return checkpoint


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


def resolve_phase_resume_state(
    checkpoint: Dict[str, Any],
    *,
    phase_name: str,
    phase_epochs: int,
) -> Dict[str, Any]:
    checkpoint_phase = checkpoint.get("phase")
    if checkpoint_phase is not None and checkpoint_phase != phase_name:
        raise ValueError(
            f"Checkpoint phase mismatch: expected {phase_name}, got {checkpoint_phase}"
        )

    saved_phase_epochs = checkpoint.get("phase_epochs")
    if saved_phase_epochs is not None and int(saved_phase_epochs) != int(phase_epochs):
        raise ValueError(
            f"Checkpoint phase_epochs mismatch: expected {phase_epochs}, got {saved_phase_epochs}"
        )

    completed_epoch = int(checkpoint.get("epoch", 0) or 0)
    loss_history = list(checkpoint.get("loss_history", []))
    return {
        "completed_epoch": completed_epoch,
        "start_epoch": completed_epoch + 1,
        "loss_history": loss_history,
        "last_loss": checkpoint.get("train_loss"),
        "is_complete": completed_epoch >= phase_epochs,
    }


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
    device: str,
    ckpt_path: Optional[str] = None,
    resume: bool = False,
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
    start_local_epoch = 1

    if resume and ckpt_path and os.path.exists(ckpt_path):
        try:
            checkpoint = load_checkpoint(
                model=model,
                checkpoint_path=ckpt_path,
                device=device,
                tokenizer_training_mode=args.tokenizer_training_mode,
                label_interface=args.label_interface,
                semantic_target_mode=args.semantic_target_mode,
                optimizer=optimizer,
                scheduler=scheduler,
            )
        except Exception as exc:
            if not is_recoverable_checkpoint_load_error(exc):
                raise
            quarantined_path = quarantine_corrupt_checkpoint(ckpt_path, rank=rank)
            if rank == 0:
                target_text = quarantined_path or "not moved"
                print(
                    f"⚠️  {phase_name}: resume checkpoint is unreadable and will be ignored: {ckpt_path}"
                )
                print(f"   quarantined: {target_text}")
                print(f"   reason: {exc}")
                print(f"   restarting {phase_name} from epoch 1")
            checkpoint = None

        if checkpoint is None:
            start_local_epoch = 1
            losses = []
        else:
            resume_state = resolve_phase_resume_state(
                checkpoint,
                phase_name=phase_name,
                phase_epochs=phase_epochs,
            )
            start_local_epoch = resume_state["start_epoch"]
            losses = resume_state["loss_history"]
            if rank == 0:
                print(
                    f"   {phase_name}: 断点续训，已完成 {resume_state['completed_epoch']} / {phase_epochs} 个 epoch"
                )
            if resume_state["is_complete"]:
                return {
                    "losses": losses,
                    "last_loss": resume_state["last_loss"],
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
            grad_clip=args.grad_clip,
            epoch_idx=local_epoch,
            epoch_total=phase_epochs,
            gradient_accumulation_steps=grad_acc_steps,
            args=args,
            rank=rank,
            phase_name=phase_name,
        )
        losses.append(train_loss)

        if rank == 0:
            print(f"   {phase_name} epoch {local_epoch}/{phase_epochs}: train_loss={train_loss:.6f}")

        if ckpt_path is not None:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=local_epoch,
                train_loss=train_loss,
                save_path=ckpt_path,
                args=args,
                phase=phase_name,
                extra_state={
                    "phase_epochs": phase_epochs,
                    "include_lora": include_lora,
                    "gradient_accumulation_steps": grad_acc_steps,
                    "epoch_offset": epoch_offset,
                    "loss_history": losses,
                    "total_steps": total_steps,
                    "warmup_steps": warmup_steps,
                },
                rank=rank,
            )

    last_loss = losses[-1] if losses else None

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
    macro_f1s = [float(r["test_macro_f1"]) for r in run_metrics]
    losses = [float(r["test_loss"]) for r in run_metrics]
    support_sizes = [int(r["support_size"]) for r in run_metrics]

    acc_mean, acc_std = mean_std(accs)
    macro_f1_mean, macro_f1_std = mean_std(macro_f1s)
    loss_mean, loss_std = mean_std(losses)
    support_mean, support_std = mean_std([float(x) for x in support_sizes])

    return {
        "shot": shot_to_name(shot),
        "num_runs": len(run_metrics),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_mean": macro_f1_mean,
        "macro_f1_std": macro_f1_std,
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
        "macro_f1_mean",
        "macro_f1_std",
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
    test_label_to_indices: Dict[int, List[int]],
    num_classes: int,
    expected_class_tokens: List[str],
    label_verbalizers: Dict[int, List[str]],
    label_to_class_id: Dict[str, int],
    label_cards: Dict[int, Dict[str, Any]],
    local_rank: int,
    world_size: int,
    rank: int,
    device: str,
) -> Optional[Dict[str, Any]]:
    shot_name = shot_to_name(shot)
    run_dir = os.path.join(base_save_dir, f"shot_{shot_name}", f"run_{run_id:02d}")
    run_metrics_path = os.path.join(run_dir, "run_metrics.json")
    support_info_path = os.path.join(run_dir, "fewshot_indices.json")
    phase1_ckpt_path = os.path.join(run_dir, "phase1_last.pt")
    phase2_ckpt_path = os.path.join(run_dir, "phase2_last.pt")
    save_phase_checkpoints = not args.skip_phase_checkpoints

    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)

    if world_size > 1:
        dist.barrier()

    cached_metrics_rank0 = None
    completed_run_exists_rank0 = False
    if (
        args.resume
        and rank == 0
        and os.path.exists(run_metrics_path)
        and (args.cleanup_checkpoints or os.path.exists(phase2_ckpt_path))
    ):
        cached_metrics_rank0 = load_json_or_quarantine(
            run_metrics_path,
            rank=rank,
            description="cached run_metrics.json",
        )
        completed_run_exists_rank0 = cached_metrics_rank0 is not None
    completed_run_exists = broadcast_object_from_rank0(
        completed_run_exists_rank0 if rank == 0 else None,
        world_size,
        rank,
    )
    if completed_run_exists:
        cached_metrics = None
        if rank == 0:
            print(f"[shot={shot_name} run={run_id}] 检测到已完成结果，直接复用: {run_metrics_path}")
            cached_metrics = cached_metrics_rank0
        if world_size > 1:
            dist.barrier()
        return cached_metrics

    set_seed(run_seed)

    support_info_rank0 = None
    if rank == 0:
        if args.resume and os.path.exists(support_info_path):
            support_info_rank0 = load_json_or_quarantine(
                support_info_path,
                rank=rank,
                description="few-shot support index JSON",
            )
        if support_info_rank0 is None:
            support_info_rank0 = sample_support_info(
                label_to_indices,
                shot,
                run_seed,
                way=args.way,
            )
            atomic_write_json(
                support_info_path,
                {
                    "dataset": args.dataset,
                    "dataset_family": args.dataset_family,
                    "label_interface": args.label_interface,
                    "split_protocol": args.split_protocol,
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
                    "any_shortage": support_info_rank0["any_shortage"],
                    "support_size": support_info_rank0["support_size"],
                },
                indent=2,
            )
    support_info = broadcast_object_from_rank0(support_info_rank0, world_size, rank)
    support_info.setdefault("selected_class_ids", [])
    support_info.setdefault("selected_indices", [])
    support_info.setdefault("selected_by_class", {})
    support_info.setdefault("k_eff_per_class", {})
    support_info.setdefault("class_train_counts", {})
    support_info.setdefault("classes_with_shortage", [])
    support_info.setdefault("any_shortage", bool(support_info["classes_with_shortage"]))
    support_info.setdefault("support_size", len(support_info["selected_indices"]))
    support_info.setdefault("way", len(support_info["selected_class_ids"]))

    support_indices = support_info["selected_indices"]
    support_dataset = Subset(train_dataset, support_indices)
    query_indices = filter_indices_by_class_ids(
        test_label_to_indices,
        support_info["selected_class_ids"],
    )
    full_query_indices = list(query_indices)
    query_eval_subset = {
        "enabled": False,
        "max_samples_per_class": int(args.eval_max_samples_per_class),
        "max_total_samples": int(args.eval_max_total_samples),
        "seed": None,
        "class_counts": summarize_subset_class_counts(
            test_label_to_indices,
            support_info["selected_class_ids"],
        ),
        "selected_indices": list(query_indices),
    }
    if args.eval_max_samples_per_class or args.eval_max_total_samples:
        query_indices, query_eval_subset = sample_query_indices_for_fast_eval(
            test_label_to_indices,
            support_info["selected_class_ids"],
            max_samples_per_class=args.eval_max_samples_per_class,
            max_total_samples=args.eval_max_total_samples,
            seed=run_seed + int(args.eval_subset_seed_offset),
        )
    query_dataset = Subset(test_dataset, query_indices)
    query_class_counts = dict(query_eval_subset["class_counts"])

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
            f"query={len(query_indices)}/{len(full_query_indices)}, "
            f"batch={train_batch_size}, grad_acc={grad_acc_steps}"
        )
        if query_eval_subset["enabled"]:
            print(f"   fast eval subset: {query_eval_subset}")
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
        collate_fn=make_collate_fn(args, is_train=True),
        **build_dataloader_kwargs(args),
    )

    test_loader = None
    if rank == 0:
        test_loader = DataLoader(
            query_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(args, is_train=False),
            **build_dataloader_kwargs(args),
        )
        support_eval_loader = DataLoader(
            support_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=make_collate_fn(args, is_train=False),
            **build_dataloader_kwargs(args),
        )

    model = build_model(args=args, device=device, rank=rank)
    underlying_model = get_model(model)
    class_tokens: List[str] = []
    class_token_ids: List[int] = []
    selected_class_token_ids: List[int] = []
    semantic_prior_metadata: Dict[str, Any] = {}
    if use_class_token_label_interface(args):
        class_tokens, class_token_ids = add_class_tokens_to_model(
            underlying_model,
            num_classes=num_classes,
            tokenizer_training_mode=args.tokenizer_training_mode,
            rank=rank,
        )
        selected_class_token_ids = [
            class_token_ids[class_id] for class_id in support_info["selected_class_ids"]
        ]

        if expected_class_tokens and class_tokens != expected_class_tokens:
            raise RuntimeError(
                f"Class tokens mismatch: expected {expected_class_tokens}, got {class_tokens}"
            )
        semantic_prior_metadata = configure_semantic_label_priors(
            underlying_model,
            args=args,
            class_token_ids=class_token_ids,
            label_verbalizers=label_verbalizers,
            selected_class_ids=support_info["selected_class_ids"],
            rank=rank,
        )
    elif rank == 0:
        print("🏷️ Semantic phrase diagnostic: using natural-language verbalizers; no class tokens added")

    if world_size > 1:
        # Few-shot training dynamically toggles LoRA participation across phase1/phase2.
        # DDP therefore needs unused-parameter detection enabled to tolerate parameters
        # that are intentionally skipped in one phase and re-enabled in the next.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
        eval_loss_type = {
            "logits": "class_token_ce",
            "phrase_likelihood": f"semantic_phrase_{args.semantic_score_mode}_ce",
        }.get(args.eval_decode_mode, "lm_answer_ce")
        print(
            f"   evaluation: decode_mode={args.eval_decode_mode}, "
            f"loss_type={eval_loss_type}"
        )
        if args.eval_decode_mode == "logits" and args.disable_constrained_decoding:
            print("   note: --disable_constrained_decoding is ignored by logits evaluation")
        if not save_phase_checkpoints:
            print("   checkpointing: disabled (--skip_phase_checkpoints); evaluation will use in-memory weights")
        if args.model_select_metric != "last":
            print("   note: model_select_metric is forced by design to phase2 last checkpoint.")

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
        device=device,
        ckpt_path=phase1_ckpt_path if save_phase_checkpoints else None,
        resume=args.resume,
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
        device=device,
        ckpt_path=phase2_ckpt_path if save_phase_checkpoints else None,
        resume=args.resume,
    )

    if world_size > 1:
        dist.barrier()

    run_metrics = None
    if rank == 0:
        if save_phase_checkpoints:
            load_checkpoint(
                model=model,
                checkpoint_path=phase2_ckpt_path,
                device=device,
                tokenizer_training_mode=args.tokenizer_training_mode,
                label_interface=args.label_interface,
                semantic_target_mode=args.semantic_target_mode,
            )
        eval_label_verbalizers = label_verbalizers
        if args.label_shuffle_control and args.eval_decode_mode == "phrase_likelihood":
            rng = random.Random(run_seed)
            selected_ids = list(support_info["selected_class_ids"])
            shuffled_ids = list(selected_ids)
            rng.shuffle(shuffled_ids)
            eval_label_verbalizers = dict(label_verbalizers)
            for class_id, source_class_id in zip(selected_ids, shuffled_ids):
                eval_label_verbalizers[class_id] = list(label_verbalizers[source_class_id])

        phrase_support_bias = None
        phrase_score_mode = args.semantic_score_mode
        if args.eval_decode_mode == "phrase_likelihood":
            phrase_score_mode = args.phrase_diag_score if args.phrase_diag_score_explicit else args.semantic_score_mode
            if phrase_score_mode == "support_cal":
                candidate_verbalizers = [
                    eval_label_verbalizers[class_id]
                    for class_id in support_info["selected_class_ids"]
                ]
                phrase_support_bias = estimate_support_phrase_calibration(
                    model,
                    support_eval_loader,
                    candidate_verbalizers,
                    rank=rank,
                )
        test_results = evaluate(
            model=model,
            data_loader=test_loader,
            max_new_tokens=args.max_new_tokens,
            class_token_ids=selected_class_token_ids,
            disable_constrained_decoding=args.disable_constrained_decoding,
            eval_decode_mode=args.eval_decode_mode,
            label_verbalizers=eval_label_verbalizers,
            selected_class_ids=support_info["selected_class_ids"],
            semantic_score_mode=phrase_score_mode,
            support_calibration_scores=phrase_support_bias,
            label_to_class_id=label_to_class_id,
            desc="Testing",
            rank=rank,
        )

        run_metrics = {
            "dataset": args.dataset,
            "dataset_family": args.dataset_family,
            "split_protocol": args.split_protocol,
            "protocol": args.protocol,
            "label_interface": args.label_interface,
            "verbalizer_set": args.verbalizer_set,
            "verbalizer_mode": args.verbalizer_mode,
            "semantic_target_mode": args.semantic_target_mode,
            "class_token_init": args.class_token_init,
            "label_proto_source": args.label_proto_source,
            "semantic_row_reg_weight": args.semantic_row_reg_weight,
            "semantic_row_reg_type": args.semantic_row_reg_type,
            "semantic_decision_reg_weight": args.semantic_decision_reg_weight,
            "semantic_decision_temperature": args.semantic_decision_temperature,
            "semantic_prior": semantic_prior_metadata,
            "label_cards": label_cards,
            "phrase_diagnostic": {
                "enabled": args.eval_decode_mode == "phrase_likelihood",
                "score_mode": phrase_score_mode if args.eval_decode_mode == "phrase_likelihood" else None,
                "use_eos": bool(args.phrase_diag_use_eos),
                "label_shuffle_control": bool(args.label_shuffle_control),
            },
            "tokenizer_training_mode": args.tokenizer_training_mode,
            "way": support_info["way"],
            "selected_class_ids": support_info["selected_class_ids"],
            "shot": shot_name,
            "run_id": run_id,
            "shot_index": shot_idx,
            "seed": run_seed,
            "support_size": len(support_indices),
            "query_size": len(query_indices),
            "full_query_size": len(full_query_indices),
            "query_eval_subset": query_eval_subset,
            "k_eff_per_class": support_info["k_eff_per_class"],
            "class_train_counts": support_info["class_train_counts"],
            "support_class_counts": support_info["k_eff_per_class"],
            "query_class_counts": query_class_counts,
            "classes_with_shortage": support_info["classes_with_shortage"],
            "any_shortage": support_info["any_shortage"],
            "phase1_epochs": phase1_epochs,
            "phase2_epochs": phase2_epochs,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": grad_acc_steps,
            "constrained_decoding": not args.disable_constrained_decoding,
            "eval_decode_mode": test_results["eval_decode_mode"],
            "eval_loss_type": test_results["eval_loss_type"],
            "semantic_score_mode": test_results["semantic_score_mode"],
            "phase1_last_train_loss": phase1_stats["last_loss"],
            "phase2_last_train_loss": phase2_stats["last_loss"],
            "test_loss": test_results["loss"],
            "test_accuracy": test_results["accuracy"],
            "test_macro_f1": test_results["macro_f1"],
            "model_checkpoint": "phase2_last.pt" if save_phase_checkpoints else None,
        }

        atomic_write_json(run_metrics_path, run_metrics, indent=2)

        atomic_write_json(
            os.path.join(run_dir, "test_predictions.json"),
            {
                "predictions": test_results["predictions"],
                "labels": test_results["labels"],
                "macro_f1": test_results["macro_f1"],
                "eval_decode_mode": test_results["eval_decode_mode"],
                "semantic_score_mode": test_results["semantic_score_mode"],
            },
            indent=2,
        )

        if save_phase_checkpoints and args.cleanup_checkpoints:
            cleanup_checkpoint_files(
                [phase1_ckpt_path, phase2_ckpt_path],
                rank=rank,
            )

        print(
            f"   result: test_acc={test_results['accuracy']:.4f}, "
            f"test_macro_f1={test_results['macro_f1']:.4f}, "
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
    args = hydrate_args_from_pretrained_model_metadata(args)
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
        warn_deprecated_newts_context_length(args, rank)

        eos_rank0 = resolve_dataset_eos_token(args) if rank == 0 else None
        dataset_eos = broadcast_object_from_rank0(eos_rank0, world_size, rank)

        dataset_bundle = load_univariate_fewshot_bundle(args, eos_token=dataset_eos)
        args.dataset_family = dataset_bundle.dataset_family
        args.dataset = dataset_bundle.dataset_name
        args.split_protocol = dataset_bundle.split_protocol
        save_root = os.path.join(args.save_dir, args.dataset)

        train_dataset = dataset_bundle.train_dataset
        test_dataset = dataset_bundle.test_dataset
        num_classes = dataset_bundle.num_classes
        class_tokens = dataset_bundle.class_tokens
        label_verbalizers = dataset_bundle.label_verbalizers
        label_to_class_id = dataset_bundle.label_to_class_id
        label_cards = dataset_bundle.label_cards
        label_to_indices = build_label_to_indices(train_dataset)
        test_label_to_indices = build_label_to_indices(test_dataset)

        if args.way is not None and args.way > num_classes:
            raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes})")

        if rank == 0:
            os.makedirs(save_root, exist_ok=True)
            atomic_write_json(os.path.join(save_root, "config.json"), vars(args), indent=2)

            print("=" * 80)
            print("M2: Few-shot Univariate Classification with Pretrained SP Models")
            print("=" * 80)
            print(f"time: {datetime.datetime.now()}")
            print(f"dataset_family: {args.dataset_family}")
            print(f"dataset: {args.dataset}")
            print(f"split_protocol: {args.split_protocol}")
            print(f"protocol: {args.protocol}")
            print(f"way: {args.way if args.way is not None else 'all'}")
            print(f"shots: {[shot_to_name(s) for s in shots]}")
            print(f"num_runs: {num_runs}")
            print(f"pretrained_model: {args.pretrained_model}")
            print(f"local_checkpoint: {args.local_checkpoint}")
            print(f"encoder_type: {args.encoder_type}")
            print(f"llm_id: {args.llm_id}")
            print(f"use_lora: {args.use_lora}")
            print(f"label_interface: {args.label_interface}")
            print(f"verbalizer_set: {args.verbalizer_set}")
            print(f"verbalizer_mode: {args.verbalizer_mode}")
            print(f"semantic_target_mode: {args.semantic_target_mode}")
            print(f"class_token_init: {args.class_token_init}")
            print(f"semantic_row_reg_weight: {args.semantic_row_reg_weight}")
            print(f"semantic_score_mode: {args.semantic_score_mode}")
            print(f"tokenizer_training_mode: {args.tokenizer_training_mode}")
            print(f"constrained_decoding: {args.constrained_decoding}")
            print(f"epochs: {args.epochs}")
            print(f"pad_mode: {resolve_effective_pad_mode(args)}")
            print(f"augmentation: {args.enable_augmentation}")
            print(f"ddp world_size: {world_size}")
            if args.encoder_type == "newts_dual_branch":
                print("dynamic_length: enabled")
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
            if label_cards:
                print(f"label_cards: {label_cards}")

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
                    expected_class_tokens=class_tokens,
                    label_verbalizers=label_verbalizers,
                    label_to_class_id=label_to_class_id,
                    label_cards=label_cards,
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
                atomic_write_json(os.path.join(shot_dir, "shot_summary.json"), shot_summary, indent=2)

                print(
                    f"[shot={shot_name}] "
                    f"acc={shot_summary['accuracy_mean']:.4f}±{shot_summary['accuracy_std']:.4f}, "
                    f"macro_f1={shot_summary['macro_f1_mean']:.4f}±{shot_summary['macro_f1_std']:.4f}"
                )

            if world_size > 1:
                dist.barrier()

        if rank == 0:
            overall_summary = {
                "dataset": args.dataset,
                "dataset_family": args.dataset_family,
                "split_protocol": args.split_protocol,
                "protocol": args.protocol,
                "way": args.way if args.way is not None else num_classes,
                "num_classes": num_classes,
                "shots": [shot_to_name(s) for s in shots],
                "num_runs": num_runs,
                "timestamp": str(datetime.datetime.now()),
                "shot_summaries": shot_summaries,
            }

            atomic_write_json(os.path.join(save_root, "fewshot_summary.json"), overall_summary, indent=2)

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
