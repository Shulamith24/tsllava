#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Few-shot classification with the pretrained M2 dual-branch encoder and no LLM."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SAVE_DIR = "results/ablations/m2_no_llm_linear_fewshot"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fewshot_utils import (  # noqa: E402
    ShotType,
    filter_indices_by_class_ids,
    parse_shots,
    sample_support_info,
    shot_to_name,
    write_json,
)
from ucr_fewshot_baseline_utils import (  # noqa: E402
    DEFAULT_DATA_PATH,
    cleanup_checkpoint_files,
    cli_flag_was_provided,
    resolve_device,
    set_seed,
)
from opentslm.model.encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder  # noqa: E402
from opentslm.model.encoder.NewTSVisionEncoder import (  # noqa: E402
    LEGACY_VISION_2D_MODE,
    SUPPORTED_VISION_2D_MODES,
    resolve_effective_vision_stride,
    validate_vision_2d_mode,
)
from opentslm.model_config import ENCODER_OUTPUT_DIM  # noqa: E402
from opentslm.time_series_datasets.univariate_fewshot import load_univariate_fewshot_bundle  # noqa: E402
from opentslm.time_series_datasets.util import (  # noqa: E402
    extend_time_series_to_match_patch_size_and_aggregate,
)

NoFullShotType = int


@dataclass(frozen=True)
class ResolvedHParams:
    epochs: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    lr_head: float
    lr_encoder: float
    weight_decay: float
    grad_clip: float


def parse_int_list(value: Optional[Union[str, List[int]]]) -> Optional[List[int]]:
    if value is None or isinstance(value, list):
        return value
    items = [int(token.strip()) for token in value.split(",") if token.strip()]
    return items or None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="M2 no-LLM ablation: dual-branch encoder features plus a classification head."
    )

    parser.add_argument("--protocol", type=str, default="fewshot", choices=["fewshot"], help=argparse.SUPPRESS)
    parser.add_argument("--shots", type=str, default="1,2,5,10")
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--fewshot_seed_base", type=int, default=3407)
    parser.add_argument("--fewshot_batch_mode", type=str, default="manual", choices=["full", "manual"])

    parser.add_argument(
        "--dataset_family",
        type=str,
        default="ucr",
        choices=["ucr", "mitbih", "sleepedf", "cinc2017af", "cinc2016heart"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split_protocol", type=str, default="default")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--eos_token", type=str, default="<eos>")

    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional pretrained TimeMorph/NewTS dual-branch encoder checkpoint. "
            "When omitted, the no-LLM encoder is initialized from the script defaults; "
            "the frozen ViT backbone still uses its configured pretrained vision model."
        ),
    )
    parser.add_argument("--pretrained_model", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--encoder_init",
        type=str,
        default="auto",
        choices=["auto", "scratch", "checkpoint"],
        help=(
            "Encoder initialization for the no-LLM ablation. 'auto' uses checkpoint "
            "initialization when --local_checkpoint is provided and scratch otherwise."
        ),
    )
    parser.add_argument("--runtime_branch_mode", type=str, default="both", choices=["both", "ts_only", "vision_only"])
    parser.add_argument(
        "--classifier_head",
        type=str,
        default="linear",
        choices=["linear", "transformer"],
        help="Classification head attached to the pretrained dual-branch encoder.",
    )
    parser.add_argument(
        "--allow_single_branch",
        action="store_true",
        help="Allow ts_only/vision_only feature extraction by zero-filling the missing branch.",
    )
    parser.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the loaded dual-branch encoder and train only the classification head.",
    )
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--freeze_ts_backbone", action="store_true")
    parser.add_argument(
        "--freeze_vision_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the vision backbone when encoder fine-tuning is enabled.",
    )
    parser.add_argument("--no_freeze_vision_backbone", dest="freeze_vision_backbone", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--phase1_epochs", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_head", "--lr_projector", dest="lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_encoder", type=float, default=2e-4)
    parser.add_argument("--lr_lora", type=float, default=1e-4, help=argparse.SUPPRESS)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--classifier_dropout", type=float, default=0.1)
    parser.add_argument("--transformer_head_layers", type=int, default=2)
    parser.add_argument("--transformer_head_heads", type=int, default=4)
    parser.add_argument("--transformer_head_ffn_dim", type=int, default=512)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help=argparse.SUPPRESS)

    parser.add_argument("--pad_mode", type=str, default="last", choices=["last", "repeat", "zero"])
    parser.add_argument("--enable_augmentation", action="store_true")
    parser.add_argument("--aug_jitter_std", type=float, default=0.02)
    parser.add_argument("--aug_scaling_min", type=float, default=0.9)
    parser.add_argument("--aug_scaling_max", type=float, default=1.1)
    parser.add_argument("--aug_time_mask_ratio", type=float, default=0.05)
    parser.add_argument("--aug_time_mask_prob", type=float, default=0.3)
    parser.add_argument("--aug_freq_dropout_ratio", type=float, default=0.05, help=argparse.SUPPRESS)
    parser.add_argument("--aug_freq_dropout_prob", type=float, default=0.2, help=argparse.SUPPRESS)

    parser.add_argument("--vision_2d_mode", type=str, default=LEGACY_VISION_2D_MODE, choices=list(SUPPORTED_VISION_2D_MODES))
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    parser.add_argument("--vit_mix_layers", type=str, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--no_lora", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--lora_r", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--lora_alpha", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--llm_id", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--llm_attn_impl", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--random_init_llm", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model_select_metric", type=str, default="last", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer_training_mode", type=str, default="class_rows", help=argparse.SUPPRESS)
    parser.add_argument("--max_new_tokens", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--disable_constrained_decoding", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--eval_every", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument("--early_stop", type=int, default=10, help=argparse.SUPPRESS)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent_workers", dest="persistent_workers", action="store_true")
    parser.add_argument("--no_persistent_workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cleanup_checkpoints", action="store_true")
    parser.add_argument("--skip_checkpoints", "--skip_phase_checkpoints", dest="skip_checkpoints", action="store_true")

    parser.set_defaults(pin_memory=True, persistent_workers=True)
    args, unknown_args = parser.parse_known_args(argv)
    if unknown_args:
        print("Ignoring unsupported no-LLM ablation args: " + " ".join(unknown_args))
    args.vit_mix_layers = parse_int_list(args.vit_mix_layers)
    args.vit_patch_size_explicit = cli_flag_was_provided(provided_argv, "--vit_patch_size")
    args.vit_stride_explicit = cli_flag_was_provided(provided_argv, "--vit_stride")
    args.vision_2d_mode_explicit = cli_flag_was_provided(provided_argv, "--vision_2d_mode")
    args.protocol = "fewshot"
    return args


def validate_args(args: argparse.Namespace, shots: Sequence[ShotType]) -> None:
    if any(shot == "full" for shot in shots):
        raise ValueError("This no-LLM ablation only supports few-shot K values; remove 'full' from --shots.")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.num_runs < 1:
        raise ValueError("--num_runs must be >= 1")
    if args.way is not None and args.way < 1:
        raise ValueError("--way must be >= 1 when provided")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader_num_workers must be >= 0")
    if args.aug_scaling_min > args.aug_scaling_max:
        raise ValueError("--aug_scaling_min must be <= --aug_scaling_max")
    if args.vit_patch_size < 1:
        raise ValueError("--vit_patch_size must be >= 1")
    if args.transformer_head_layers < 1:
        raise ValueError("--transformer_head_layers must be >= 1")
    if args.transformer_head_heads < 1:
        raise ValueError("--transformer_head_heads must be >= 1")
    if args.transformer_head_ffn_dim < 1:
        raise ValueError("--transformer_head_ffn_dim must be >= 1")
    if args.resume and args.skip_checkpoints:
        raise ValueError("--resume cannot be used with --skip_checkpoints")
    if args.encoder_init == "checkpoint" and not args.local_checkpoint:
        raise ValueError("--encoder_init checkpoint requires --local_checkpoint")
    if args.encoder_init == "scratch" and args.local_checkpoint:
        raise ValueError("--encoder_init scratch should not be combined with --local_checkpoint")
    validate_vision_2d_mode(args.vision_2d_mode)


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.classifier_head == "transformer":
        return "m2_no_llm_transformer"
    return "m2_no_llm_linear"


def resolve_hparams(args: argparse.Namespace, support_size: int, shot: NoFullShotType) -> ResolvedHParams:
    if args.fewshot_batch_mode == "full":
        batch_size = max(1, support_size)
        grad_acc = 1
    else:
        batch_size = args.batch_size
        grad_acc = args.gradient_accumulation_steps
    return ResolvedHParams(
        epochs=int(args.epochs),
        batch_size=max(1, int(batch_size)),
        eval_batch_size=max(1, int(args.eval_batch_size)),
        gradient_accumulation_steps=max(1, int(grad_acc)),
        lr_head=float(args.lr_head),
        lr_encoder=float(args.lr_encoder),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
    )


def default_newts_dual_branch_config() -> Dict[str, Any]:
    return {
        "output_dim": ENCODER_OUTPUT_DIM,
        "patch_length": 16,
        "stride": 8,
        "d_model": 128,
        "num_attention_heads": 8,
        "num_hidden_layers": 3,
        "ffn_dim": 512,
        "dropout": 0.1,
        "dynamic_length": True,
        "ts_positional_encoding": "sinusoidal",
        "branch_mode": "both",
        "vit_model_name": "facebook/dinov2-base",
        "vit_feature_mode": "single",
        "vit_layer_idx": 4,
        "vit_mix_layers": None,
        "vit_patch_size": 16,
        "vit_stride": 0.5,
        "vision_2d_mode": LEGACY_VISION_2D_MODE,
        "vit_truncate_to_feature_layer": True,
        "vit_num_hidden_layers": None,
        "projector_type": "mlp",
        "projector_dropout": 0.1,
        "use_pma": False,
        "aggregator_layers": 2,
        "aggregator_hidden_size": ENCODER_OUTPUT_DIM,
        "aggregator_num_heads": 8,
        "aggregator_ffn_dim": ENCODER_OUTPUT_DIM * 4,
        "aggregator_num_queries": 4,
        "aggregator_query_mode": "shared",
        "aggregator_fusion_mode": "gated_sum",
        "aggregator_gate_type": "dynamic",
        "aggregator_fuse_layers": 1,
        "enable_modality_embeddings": False,
        "branch_dropout": 0.0,
        "vision_train_mode": "none",
        "vision_topk_blocks": 4,
        "freeze_ts_backbone": False,
        "freeze_vision_backbone": True,
    }


def extract_encoder_state_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    encoder_state = checkpoint.get("encoder_state")
    if encoder_state:
        return encoder_state

    model_state = checkpoint.get("model_state") or {}
    encoder_state = {
        key[len("encoder.") :]: value
        for key, value in model_state.items()
        if key.startswith("encoder.")
    }
    if encoder_state:
        return encoder_state

    raise KeyError(
        "Checkpoint does not contain encoder weights. Expected top-level "
        "'encoder_state' or a full 'model_state' with 'encoder.' prefixes."
    )


def resolve_encoder_config_from_checkpoint(
    checkpoint: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metadata = checkpoint.get("model_config") or {}
    encoder_type = metadata.get("encoder_type")
    if encoder_type and encoder_type != "newts_dual_branch":
        raise ValueError(
            "The no-LLM linear ablation expects a newts_dual_branch checkpoint, "
            f"got encoder_type={encoder_type!r}."
        )

    config = default_newts_dual_branch_config()
    checkpoint_config = dict(metadata.get("encoder_config") or checkpoint.get("encoder_config") or {})
    config.update(checkpoint_config)
    config.pop("context_length", None)
    config["output_dim"] = ENCODER_OUTPUT_DIM
    config["dynamic_length"] = True
    config["ts_positional_encoding"] = "sinusoidal"

    if args.vision_2d_mode_explicit or args.vit_stride_explicit or args.vit_patch_size_explicit:
        vision_2d_mode = validate_vision_2d_mode(args.vision_2d_mode if args.vision_2d_mode_explicit else config["vision_2d_mode"])
        config["vision_2d_mode"] = vision_2d_mode
        if args.vit_patch_size_explicit:
            config["vit_patch_size"] = int(args.vit_patch_size)
        config["vit_stride"] = resolve_effective_vision_stride(
            vision_2d_mode,
            args.vit_stride if args.vit_stride_explicit else config["vit_stride"],
            stride_explicit=args.vit_stride_explicit,
        )

    config["freeze_ts_backbone"] = bool(args.freeze_ts_backbone)
    config["freeze_vision_backbone"] = bool(args.freeze_vision_backbone)
    return config


class TransformerClassificationHead(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError(
                f"--transformer_head_heads ({num_heads}) must divide head input dim ({input_dim})."
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        encoded = self.encoder(sequence)
        cls_state = self.norm(encoded[:, 0])
        logits = self.classifier(self.dropout(cls_state))
        return logits, cls_state


class DualBranchNoLLMClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: NewTSDualBranchEncoder,
        num_classes: int,
        runtime_branch_mode: str = "both",
        classifier_head: str = "linear",
        classifier_dropout: float = 0.0,
        allow_single_branch: bool = False,
        transformer_head_layers: int = 2,
        transformer_head_heads: int = 4,
        transformer_head_ffn_dim: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.runtime_branch_mode = runtime_branch_mode
        self.allow_single_branch = bool(allow_single_branch)
        self.classifier_head = str(classifier_head)
        self.branch_feature_dim = int(getattr(encoder, "token_output_dim", encoder.output_dim))
        self.token_dim = int(getattr(encoder, "output_dim", self.branch_feature_dim))
        self.dropout = nn.Dropout(float(classifier_dropout))
        self.transformer_head_config = {
            "d_model": self.token_dim,
            "num_layers": int(transformer_head_layers),
            "num_heads": int(transformer_head_heads),
            "ffn_dim": int(transformer_head_ffn_dim),
            "dropout": float(classifier_dropout),
            "cls_pooling": True,
        }
        if self.classifier_head == "linear":
            self.head = nn.Linear(self.branch_feature_dim * 2, num_classes)
        elif self.classifier_head == "transformer":
            self.head = TransformerClassificationHead(
                input_dim=self.token_dim,
                num_classes=num_classes,
                num_layers=transformer_head_layers,
                num_heads=transformer_head_heads,
                ffn_dim=transformer_head_ffn_dim,
                dropout=classifier_dropout,
            )
        else:
            raise ValueError(f"Unsupported classifier_head: {classifier_head}")

    @property
    def num_classes(self) -> int:
        if isinstance(self.head, nn.Linear):
            return int(self.head.out_features)
        return int(self.head.classifier.out_features)

    def _zero_branch_like(self, reference: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if reference is not None:
            return torch.zeros_like(reference)
        dtype = next(self.parameters()).dtype
        return torch.zeros(batch_size, self.branch_feature_dim, device=device, dtype=dtype)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(
            inputs,
            runtime_branch_mode=self.runtime_branch_mode,
            return_intermediates=True,
        )
        pooled_ts = outputs.get("pooled_ts")
        pooled_vision = outputs.get("pooled_vision")
        if (pooled_ts is None or pooled_vision is None) and not self.allow_single_branch:
            raise RuntimeError(
                "Both TS and vision pooled features are required for this ablation. "
                "Use --allow_single_branch only for debugging branch-specific runs."
            )
        batch_size = int(inputs.shape[0])
        pooled_ts = pooled_ts if pooled_ts is not None else self._zero_branch_like(pooled_vision, batch_size, inputs.device)
        pooled_vision = (
            pooled_vision
            if pooled_vision is not None
            else self._zero_branch_like(pooled_ts, batch_size, inputs.device)
        )
        return torch.cat([pooled_ts, pooled_vision], dim=-1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.classifier_head == "transformer":
            outputs = self.encoder(
                inputs,
                runtime_branch_mode=self.runtime_branch_mode,
                return_intermediates=True,
            )
            tokens = outputs["fused_tokens"]
            return self.head(tokens)

        features = self.extract_features(inputs)
        logits = self.head(self.dropout(features))
        return logits, features


def build_encoder_linear_model(
    *,
    args: argparse.Namespace,
    num_classes: int,
    device: torch.device,
) -> Tuple[DualBranchNoLLMClassifier, Dict[str, Any]]:
    use_checkpoint = bool(args.local_checkpoint) if args.encoder_init == "auto" else args.encoder_init == "checkpoint"
    checkpoint = (
        torch.load(args.local_checkpoint, map_location="cpu", weights_only=False)
        if use_checkpoint
        else None
    )
    encoder_config = (
        resolve_encoder_config_from_checkpoint(checkpoint, args)
        if checkpoint is not None
        else resolve_encoder_config_from_checkpoint({}, args)
    )
    encoder = NewTSDualBranchEncoder(**encoder_config, device=str(device)).to(device)
    if checkpoint is not None:
        encoder.load_state_dict(extract_encoder_state_from_checkpoint(checkpoint), strict=True)
    if args.gradient_checkpointing:
        encoder.enable_gradient_checkpointing()

    model = DualBranchNoLLMClassifier(
        encoder=encoder,
        num_classes=num_classes,
        runtime_branch_mode=args.runtime_branch_mode,
        classifier_head=args.classifier_head,
        classifier_dropout=args.classifier_dropout,
        allow_single_branch=args.allow_single_branch,
        transformer_head_layers=args.transformer_head_layers,
        transformer_head_heads=args.transformer_head_heads,
        transformer_head_ffn_dim=args.transformer_head_ffn_dim,
    ).to(device)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    encoder_config["encoder_init"] = "checkpoint" if checkpoint is not None else "scratch"
    encoder_config["local_checkpoint"] = str(args.local_checkpoint) if checkpoint is not None else None
    return model, encoder_config


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


def summarize_subset_class_counts(
    label_to_indices: Dict[int, List[int]],
    class_ids: Iterable[int],
) -> Dict[str, int]:
    return {str(class_id): len(label_to_indices.get(int(class_id), [])) for class_id in class_ids}


def make_collate_fn(args: argparse.Namespace, *, is_train: bool, global_to_local: Dict[int, int]):
    def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processed = extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=1,
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
            enable_freq_dropout=False,
        )

        features: List[torch.Tensor] = []
        local_labels: List[int] = []
        global_labels: List[int] = []
        for item in processed:
            time_series = item["time_series"]
            if time_series.dim() != 2 or time_series.shape[0] != 1:
                raise ValueError(f"Expected one univariate series per sample, got shape {tuple(time_series.shape)}")
            global_label = int(item["int_label"])
            features.append(time_series.squeeze(0))
            global_labels.append(global_label)
            local_labels.append(int(global_to_local[global_label]))

        return (
            torch.stack(features, dim=0),
            torch.tensor(local_labels, dtype=torch.long),
            torch.tensor(global_labels, dtype=torch.long),
        )

    return collate_fn


def build_dataloader_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "num_workers": args.dataloader_num_workers,
        "pin_memory": bool(args.pin_memory),
    }
    if args.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = bool(args.persistent_workers)
    return kwargs


def build_optimizer(model: DualBranchNoLLMClassifier, args: argparse.Namespace, hparams: ResolvedHParams) -> AdamW:
    param_groups: List[Dict[str, Any]] = []
    if not args.freeze_encoder:
        encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": hparams.lr_encoder})

    head_params = [param for param in model.head.parameters() if param.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": hparams.lr_head})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for the no-LLM ablation.")
    return AdamW(param_groups, weight_decay=hparams.weight_decay)


def train_one_epoch(
    model: DualBranchNoLLMClassifier,
    data_loader: DataLoader,
    *,
    device: torch.device,
    optimizer: AdamW,
    hparams: ResolvedHParams,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (inputs, labels, _global_labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits, _features = model(inputs)
        loss = F.cross_entropy(logits, labels)
        (loss / hparams.gradient_accumulation_steps).backward()

        should_step = (step + 1) % hparams.gradient_accumulation_steps == 0
        if should_step:
            nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=hparams.grad_clip,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    if len(data_loader) % hparams.gradient_accumulation_steps != 0:
        nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=hparams.grad_clip,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: DualBranchNoLLMClassifier,
    data_loader: DataLoader,
    *,
    device: torch.device,
    num_local_classes: int,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions_local: List[int] = []
    labels_local: List[int] = []
    labels_global: List[int] = []

    for inputs, labels, global_labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits, _features = model(inputs)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        predictions_local.extend(int(item) for item in preds.detach().cpu().tolist())
        labels_local.extend(int(item) for item in labels.detach().cpu().tolist())
        labels_global.extend(int(item) for item in global_labels.detach().cpu().tolist())

    accuracy = sum(int(pred == label) for pred, label in zip(predictions_local, labels_local)) / max(total_examples, 1)
    macro_f1 = float(
        f1_score(
            labels_local,
            predictions_local,
            labels=list(range(num_local_classes)),
            average="macro",
            zero_division=0.0,
        )
    )
    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "predictions_local": predictions_local,
        "labels_local": labels_local,
        "labels_global": labels_global,
    }


def save_linear_checkpoint(
    checkpoint_path: Path,
    *,
    model: DualBranchNoLLMClassifier,
    optimizer: AdamW,
    epoch: int,
    loss_history: List[float],
    support_history: List[Dict[str, float]],
    encoder_trainable: bool,
) -> None:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "loss_history": loss_history,
        "support_history": support_history,
        "head_state_dict": model.head.state_dict(),
        "classifier_head": model.classifier_head,
        "optimizer_state_dict": optimizer.state_dict(),
        "encoder_trainable": encoder_trainable,
    }
    if encoder_trainable:
        payload["encoder_state_dict"] = model.encoder.state_dict()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_linear_checkpoint(
    checkpoint_path: Path,
    *,
    model: DualBranchNoLLMClassifier,
    optimizer: Optional[AdamW] = None,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        head_state = checkpoint.get("head_state_dict") or checkpoint.get("classifier_state_dict")
        if head_state is not None:
            model.head.load_state_dict(head_state)
        if checkpoint.get("encoder_state_dict") is not None:
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model(
    model: DualBranchNoLLMClassifier,
    train_loader: DataLoader,
    support_eval_loader: DataLoader,
    *,
    device: torch.device,
    checkpoint_path: Path,
    args: argparse.Namespace,
    hparams: ResolvedHParams,
) -> Dict[str, Any]:
    optimizer = build_optimizer(model, args, hparams)
    loss_history: List[float] = []
    support_history: List[Dict[str, float]] = []
    start_epoch = 1

    if args.resume and checkpoint_path.exists():
        checkpoint = load_linear_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )
        loss_history = [float(item) for item in checkpoint.get("loss_history", [])]
        support_history = list(checkpoint.get("support_history", []))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resume no-LLM training from epoch {start_epoch - 1}/{hparams.epochs}")
        if start_epoch > hparams.epochs:
            last_support = support_history[-1] if support_history else {}
            return {
                "loss_history": loss_history,
                "last_train_loss": loss_history[-1] if loss_history else None,
                "last_support_accuracy": last_support.get("accuracy"),
                "last_support_loss": last_support.get("loss"),
                "last_support_macro_f1": last_support.get("macro_f1"),
            }

    for epoch in range(start_epoch, hparams.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            hparams=hparams,
        )
        support_metrics = evaluate(
            model,
            support_eval_loader,
            device=device,
            num_local_classes=model.num_classes,
        )
        loss_history.append(train_loss)
        support_history.append(
            {
                "epoch": epoch,
                "loss": float(support_metrics["loss"]),
                "accuracy": float(support_metrics["accuracy"]),
                "macro_f1": float(support_metrics["macro_f1"]),
            }
        )

        if not args.skip_checkpoints:
            save_linear_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss_history=loss_history,
                support_history=support_history,
                encoder_trainable=not args.freeze_encoder,
            )
        print(
            f"Epoch {epoch:03d}/{hparams.epochs:03d} "
            f"train_loss={train_loss:.6f} support_acc={support_metrics['accuracy']:.4f} "
            f"support_macro_f1={support_metrics['macro_f1']:.4f}"
        )

    if not args.skip_checkpoints:
        load_linear_checkpoint(checkpoint_path, model=model, device=device)

    last_support = support_history[-1] if support_history else {}
    return {
        "loss_history": loss_history,
        "last_train_loss": loss_history[-1] if loss_history else None,
        "last_support_accuracy": last_support.get("accuracy"),
        "last_support_loss": last_support.get("loss"),
        "last_support_macro_f1": last_support.get("macro_f1"),
    }


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(np.mean(values)), float(np.std(values, ddof=1))


def aggregate_shot_results(shot: NoFullShotType, run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    acc_mean, acc_std = mean_std([float(item["test_accuracy"]) for item in run_metrics])
    f1_mean, f1_std = mean_std([float(item["test_macro_f1"]) for item in run_metrics])
    loss_mean, loss_std = mean_std([float(item["test_loss"]) for item in run_metrics])
    support_mean, support_std = mean_std([float(item["support_size"]) for item in run_metrics])
    return {
        "shot": shot_to_name(shot),
        "num_runs": len(run_metrics),
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "macro_f1_mean": f1_mean,
        "macro_f1_std": f1_std,
        "loss_mean": loss_mean,
        "loss_std": loss_std,
        "support_size_mean": support_mean,
        "support_size_std": support_std,
        "any_shortage_in_shot": any(bool(item.get("any_shortage")) for item in run_metrics),
        "run_metrics": run_metrics,
    }


def save_shot_summary_csv(save_path: Path, shot_summaries: List[Dict[str, Any]]) -> None:
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
    with open(save_path, "w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in shot_summaries:
            writer.writerow({key: item.get(key) for key in columns})


def run_single_experiment(
    *,
    args: argparse.Namespace,
    save_root: Path,
    train_dataset: Dataset,
    test_dataset: Dataset,
    label_to_indices: Dict[int, List[int]],
    test_label_to_indices: Dict[int, List[int]],
    num_classes: int,
    shot: NoFullShotType,
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
    model_name = resolve_model_name(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    completed_run_exists = (
        args.resume
        and run_metrics_path.exists()
        and (args.cleanup_checkpoints or args.skip_checkpoints or checkpoint_path.exists())
    )
    if completed_run_exists:
        print(f"[{model_name} shot={shot_name} run={run_id}] reuse completed run: {run_metrics_path}")
        with open(run_metrics_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    if args.resume and support_info_path.exists():
        with open(support_info_path, "r", encoding="utf-8") as handle:
            support_info = json.load(handle)
    else:
        support_info = sample_support_info(label_to_indices, shot, run_seed, way=args.way)
        write_json(
            support_info_path,
            {
                "dataset": args.dataset,
                "dataset_family": args.dataset_family,
                "split_protocol": args.split_protocol,
                "model": model_name,
                "protocol": args.protocol,
                "shot": shot_name,
                "run_id": run_id,
                "seed": run_seed,
                **support_info,
            },
        )

    selected_class_ids = [int(class_id) for class_id in support_info["selected_class_ids"]]
    support_indices = [int(index) for index in support_info["selected_indices"]]
    query_indices = [
        int(index)
        for index in filter_indices_by_class_ids(test_label_to_indices, selected_class_ids)
    ]
    if not query_indices:
        raise RuntimeError(f"No TEST examples found for selected classes {selected_class_ids} in {args.dataset}.")

    global_to_local = {class_id: local_idx for local_idx, class_id in enumerate(selected_class_ids)}
    local_to_global = {local_idx: class_id for class_id, local_idx in global_to_local.items()}
    support_dataset = Subset(train_dataset, support_indices)
    query_dataset = Subset(test_dataset, query_indices)
    hparams = resolve_hparams(args, support_size=len(support_indices), shot=shot)

    train_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(hparams.batch_size, len(support_dataset))),
        shuffle=True,
        collate_fn=make_collate_fn(args, is_train=True, global_to_local=global_to_local),
        **build_dataloader_kwargs(args),
    )
    support_eval_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(hparams.eval_batch_size, len(support_dataset))),
        shuffle=False,
        collate_fn=make_collate_fn(args, is_train=False, global_to_local=global_to_local),
        **build_dataloader_kwargs(args),
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=max(1, min(hparams.eval_batch_size, len(query_dataset))),
        shuffle=False,
        collate_fn=make_collate_fn(args, is_train=False, global_to_local=global_to_local),
        **build_dataloader_kwargs(args),
    )

    model, encoder_config = build_encoder_linear_model(
        args=args,
        num_classes=len(selected_class_ids),
        device=device,
    )

    print("-" * 80)
    print(
        f"[{model_name} shot={shot_name} run={run_id}] "
        f"seed={run_seed}, way={len(selected_class_ids)}, support={len(support_indices)}, "
        f"query={len(query_indices)}, batch={train_loader.batch_size}, "
        f"grad_acc={hparams.gradient_accumulation_steps}"
    )
    print(f"selected classes: {selected_class_ids}")
    if args.classifier_head == "transformer":
        print(
            "transformer head input: "
            f"fused_tokens(dim={model.token_dim}), config={model.transformer_head_config}"
        )
    else:
        print(
            "linear head input: "
            f"pooled_ts({model.branch_feature_dim}) + pooled_vision({model.branch_feature_dim})"
        )
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
    test_results = evaluate(
        model,
        query_loader,
        device=device,
        num_local_classes=len(selected_class_ids),
    )

    predictions_global = [int(local_to_global[int(local)]) for local in test_results["predictions_local"]]
    labels_global = [int(local_to_global[int(local)]) for local in test_results["labels_local"]]
    query_class_counts = summarize_subset_class_counts(test_label_to_indices, selected_class_ids)

    run_metrics = {
        "dataset": args.dataset,
        "dataset_family": args.dataset_family,
        "split_protocol": args.split_protocol,
        "model": model_name,
        "protocol": args.protocol,
        "way": len(selected_class_ids),
        "num_classes": num_classes,
        "selected_class_ids": selected_class_ids,
        "shot": shot_name,
        "run_id": run_id,
        "shot_index": shot_idx,
        "seed": run_seed,
        "support_size": len(support_indices),
        "query_size": len(query_indices),
        "k_eff_per_class": support_info["k_eff_per_class"],
        "class_train_counts": support_info["class_train_counts"],
        "support_class_counts": support_info["k_eff_per_class"],
        "query_class_counts": query_class_counts,
        "classes_with_shortage": support_info["classes_with_shortage"],
        "any_shortage": support_info["any_shortage"],
        "epochs": hparams.epochs,
        "train_batch_size": train_loader.batch_size,
        "eval_batch_size": query_loader.batch_size,
        "gradient_accumulation_steps": hparams.gradient_accumulation_steps,
        "lr_head": hparams.lr_head,
        "lr_encoder": hparams.lr_encoder,
        "weight_decay": hparams.weight_decay,
        "classifier_dropout": args.classifier_dropout,
        "classifier_head": args.classifier_head,
        "transformer_head_config": model.transformer_head_config if args.classifier_head == "transformer" else None,
        "grad_clip": hparams.grad_clip,
        "freeze_encoder": bool(args.freeze_encoder),
        "train_ts_encoder": not bool(args.freeze_encoder),
        "encoder_init": encoder_config.get("encoder_init"),
        "local_checkpoint": encoder_config.get("local_checkpoint"),
        "branch_mode": args.runtime_branch_mode,
        "freeze_vision_backbone": bool(args.freeze_vision_backbone),
        "vision_2d_mode": encoder_config.get("vision_2d_mode"),
        "vit_patch_size": encoder_config.get("vit_patch_size"),
        "vit_stride": encoder_config.get("vit_stride"),
        "runtime_branch_mode": args.runtime_branch_mode,
        "encoder_config": encoder_config,
        "linear_head_input_dim": model.branch_feature_dim * 2 if args.classifier_head == "linear" else None,
        "transformer_head_input_dim": model.token_dim if args.classifier_head == "transformer" else None,
        "last_train_loss": train_stats["last_train_loss"],
        "last_support_accuracy": train_stats["last_support_accuracy"],
        "last_support_loss": train_stats["last_support_loss"],
        "last_support_macro_f1": train_stats["last_support_macro_f1"],
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "test_macro_f1": test_results["macro_f1"],
        "model_checkpoint": None if args.skip_checkpoints else checkpoint_path.name,
    }
    write_json(run_metrics_path, run_metrics)
    write_json(
        run_dir / "test_predictions.json",
        {
            "selected_class_ids": selected_class_ids,
            "predictions_local": test_results["predictions_local"],
            "labels_local": test_results["labels_local"],
            "predictions_global": predictions_global,
            "labels_global": labels_global,
            "labels_global_from_loader": test_results["labels_global"],
            "macro_f1": test_results["macro_f1"],
        },
    )

    if args.cleanup_checkpoints and not args.skip_checkpoints:
        cleanup_checkpoint_files([checkpoint_path])

    print(
        f"result: test_acc={test_results['accuracy']:.4f}, "
        f"test_macro_f1={test_results['macro_f1']:.4f}, test_loss={test_results['loss']:.4f}"
    )
    return run_metrics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    shots = parse_shots(args.shots)
    validate_args(args, shots)
    numeric_shots = [int(shot) for shot in shots]
    num_runs = max(1, args.num_runs)
    model_name = resolve_model_name(args)

    set_seed(args.seed)
    device = resolve_device(args.device)
    dataset_bundle = load_univariate_fewshot_bundle(args, eos_token=args.eos_token)
    args.dataset_family = dataset_bundle.dataset_family
    args.dataset = dataset_bundle.dataset_name
    args.split_protocol = dataset_bundle.split_protocol

    train_dataset = dataset_bundle.train_dataset
    test_dataset = dataset_bundle.test_dataset
    num_classes = dataset_bundle.num_classes
    label_to_indices = build_label_to_indices(train_dataset)
    test_label_to_indices = build_label_to_indices(test_dataset)
    if args.way is not None and args.way > num_classes:
        raise ValueError(f"--way ({args.way}) cannot exceed num_classes ({num_classes}).")

    save_root = Path(args.save_dir) / args.dataset
    save_root.mkdir(parents=True, exist_ok=True)
    write_json(
        save_root / "config.json",
        {
            **vars(args),
            "device": str(device),
            "num_classes": num_classes,
            "class_tokens": dataset_bundle.class_tokens,
            "label_mapping": dataset_bundle.label_mapping,
        },
    )

    print("=" * 80)
    print("M2 No-LLM: Few-shot Univariate Classification")
    print("=" * 80)
    print(f"time: {datetime.datetime.now()}")
    print(f"dataset_family: {args.dataset_family}")
    print(f"dataset: {args.dataset}")
    print(f"split_protocol: {args.split_protocol}")
    print(f"data_source: {Path(args.data_path).resolve()}")
    print(f"shots: {[shot_to_name(shot) for shot in numeric_shots]}")
    print(f"way: {args.way if args.way is not None else 'all'}")
    print(f"num_runs: {num_runs}")
    print(f"device: {device}")
    print(f"local_checkpoint: {args.local_checkpoint}")
    print(f"encoder_init: {args.encoder_init}")
    print(f"model: {model_name}")
    print(f"classifier_head: {args.classifier_head}")
    print(f"freeze_encoder: {args.freeze_encoder}")
    print(f"freeze_vision_backbone: {args.freeze_vision_backbone}")
    print(f"runtime_branch_mode: {args.runtime_branch_mode}")
    print(f"vision_2d_mode: {args.vision_2d_mode}")
    print(f"vit_patch_size: {args.vit_patch_size}")
    print(f"vit_stride: {args.vit_stride}")
    print(f"classifier_dropout: {args.classifier_dropout}")
    print(f"num_classes: {num_classes}")
    print(f"train_size: {len(train_dataset)} | test_size: {len(test_dataset)}")
    print("=" * 80)

    shot_summaries = []
    for shot_idx, shot in enumerate(numeric_shots):
        shot_run_metrics: List[Dict[str, Any]] = []
        for run_id in range(1, num_runs + 1):
            run_seed = args.fewshot_seed_base + shot_idx * 1000 + run_id
            set_seed(run_seed)
            run_metrics = run_single_experiment(
                args=args,
                save_root=save_root,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                label_to_indices=label_to_indices,
                test_label_to_indices=test_label_to_indices,
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
            f"acc={shot_summary['accuracy_mean']:.4f}+-{shot_summary['accuracy_std']:.4f}, "
            f"macro_f1={shot_summary['macro_f1_mean']:.4f}+-{shot_summary['macro_f1_std']:.4f}"
        )

    overall_summary = {
        "dataset": args.dataset,
        "dataset_family": args.dataset_family,
        "split_protocol": args.split_protocol,
        "model": model_name,
        "protocol": args.protocol,
        "way": args.way if args.way is not None else num_classes,
        "num_classes": num_classes,
        "shots": [shot_to_name(shot) for shot in numeric_shots],
        "num_runs": num_runs,
        "classifier_head": args.classifier_head,
        "runtime_branch_mode": args.runtime_branch_mode,
        "vision_2d_mode": args.vision_2d_mode,
        "vit_patch_size": args.vit_patch_size,
        "vit_stride": args.vit_stride,
        "encoder_init": args.encoder_init,
        "freeze_encoder": bool(args.freeze_encoder),
        "freeze_vision_backbone": bool(args.freeze_vision_backbone),
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
