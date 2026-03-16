# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification
from transformers.models.patchtst.modeling_patchtst import PatchTSTClassificationHead


def _resolve_optional(value, default):
    return default if value is None else value


def prepare_patchtst_classification_batch(
    batch: List[Dict[str, Any]],
    *,
    context_length: int,
    device: str,
    pad_mode: str = "zero",
) -> Dict[str, torch.Tensor]:
    """
    Convert a raw UCRClassificationDataset batch into PatchTST classification tensors.

    Returns:
        Dict with `past_values`, `target_values`, and `past_observed_mask`.
    """
    if pad_mode not in {"zero", "last", "repeat"}:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    past_values_list = []
    observed_masks = []
    labels = []

    for sample in batch:
        ts_container = sample["time_series"]
        ts_raw = ts_container[0] if isinstance(ts_container, list) else ts_container
        ts = torch.as_tensor(ts_raw, dtype=torch.float32).flatten()
        if ts.numel() == 0:
            raise ValueError("Encountered empty time series.")

        observed_len = min(ts.numel(), context_length)
        observed = ts[:observed_len]

        if observed_len < context_length:
            pad_len = context_length - observed_len
            if pad_mode == "zero":
                pad = observed.new_zeros(pad_len)
            elif pad_mode == "last":
                pad = observed.new_full((pad_len,), float(observed[-1]))
            else:
                repeat_times = math.ceil(pad_len / max(1, observed.numel()))
                pad = observed.repeat(repeat_times)[:pad_len]
            padded = torch.cat([observed, pad], dim=0)
        else:
            padded = observed

        mask = torch.zeros(context_length, 1, dtype=torch.bool)
        mask[:observed_len, 0] = True

        past_values_list.append(padded.unsqueeze(-1))
        observed_masks.append(mask)
        labels.append(int(sample["int_label"]))

    return {
        "past_values": torch.stack(past_values_list, dim=0).to(device),
        "target_values": torch.tensor(labels, device=device, dtype=torch.long),
        "past_observed_mask": torch.stack(observed_masks, dim=0).to(device),
    }


class PatchTSTClassifierAdapter(nn.Module):
    """
    Thin wrapper around Hugging Face `PatchTSTForClassification`.
    """

    def __init__(
        self,
        model: PatchTSTForClassification,
        *,
        head_was_reset: bool = False,
        pretrained_source: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.head_was_reset = head_was_reset
        self.pretrained_source = pretrained_source

    @classmethod
    def build_model(
        cls,
        *,
        num_classes: int,
        context_length: int,
        device: str,
        patchtst_model_id: Optional[str] = None,
        num_input_channels: int = 1,
        patch_length: Optional[int] = None,
        stride: Optional[int] = None,
        d_model: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        head_dropout: Optional[float] = None,
        use_cls_token: bool = True,
        pooling_type: str = "mean",
        reset_head: bool = True,
    ) -> "PatchTSTClassifierAdapter":
        pretrained_source = patchtst_model_id
        head_was_reset = False

        if patchtst_model_id:
            pretrained_config = PatchTSTConfig.from_pretrained(patchtst_model_id)
            if pretrained_config.num_targets != num_classes and not reset_head:
                raise ValueError(
                    "Pretrained PatchTST num_targets does not match dataset num_classes. "
                    "Enable reset_head to rebuild the classifier head."
                )

            config = PatchTSTConfig.from_pretrained(patchtst_model_id)
            config.num_input_channels = num_input_channels
            config.num_targets = num_classes
            config.context_length = context_length
            config.patch_length = _resolve_optional(patch_length, config.patch_length)
            config.stride = _resolve_optional(
                stride,
                config.stride if config.stride is not None else config.patch_length,
            )
            config.d_model = _resolve_optional(d_model, config.d_model)
            config.num_attention_heads = _resolve_optional(
                num_attention_heads, config.num_attention_heads
            )
            config.num_hidden_layers = _resolve_optional(
                num_hidden_layers, config.num_hidden_layers
            )
            config.ffn_dim = _resolve_optional(ffn_dim, config.ffn_dim)
            config.dropout = _resolve_optional(dropout, config.dropout)
            config.head_dropout = _resolve_optional(head_dropout, config.head_dropout)
            config.use_cls_token = use_cls_token
            config.pooling_type = pooling_type

            model = PatchTSTForClassification.from_pretrained(
                patchtst_model_id,
                config=config,
                ignore_mismatched_sizes=reset_head,
            )

            if reset_head:
                model.head = PatchTSTClassificationHead(model.config)
                model.head.apply(model._init_weights)
                head_was_reset = True
        else:
            resolved_patch_length = _resolve_optional(patch_length, 12)
            config = PatchTSTConfig(
                num_input_channels=num_input_channels,
                num_targets=num_classes,
                context_length=context_length,
                patch_length=resolved_patch_length,
                stride=_resolve_optional(stride, resolved_patch_length),
                d_model=_resolve_optional(d_model, 128),
                num_attention_heads=_resolve_optional(num_attention_heads, 4),
                num_hidden_layers=_resolve_optional(num_hidden_layers, 3),
                ffn_dim=_resolve_optional(ffn_dim, 512),
                dropout=_resolve_optional(dropout, 0.0),
                head_dropout=_resolve_optional(head_dropout, 0.0),
                use_cls_token=use_cls_token,
                pooling_type=pooling_type,
            )
            model = PatchTSTForClassification(config)

        return cls(
            model=model.to(device),
            head_was_reset=head_was_reset,
            pretrained_source=pretrained_source,
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    @property
    def config(self) -> PatchTSTConfig:
        return self.model.config

    def freeze_backbone(self):
        for param in self.model.model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def predict(
        self,
        *,
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            return_dict=True,
        )
        return torch.argmax(outputs.prediction_logits, dim=-1)

    def get_param_groups(
        self,
        *,
        lr_backbone: float,
        lr_head: float,
        include_backbone: bool,
    ) -> List[Dict[str, Any]]:
        param_groups: List[Dict[str, Any]] = []

        head_params = [p for p in self.model.head.parameters() if p.requires_grad]
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head})

        if include_backbone:
            backbone_params = [p for p in self.model.model.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.insert(0, {"params": backbone_params, "lr": lr_backbone})

        if not param_groups:
            raise RuntimeError("No trainable PatchTST parameters found.")

        return param_groups

    def save_checkpoint(
        self,
        *,
        save_path: str,
        optimizer,
        scheduler,
        epoch: int,
        phase: str,
        num_classes: int,
        context_length: int,
        label_mapping: Dict[str, Any],
        args: Dict[str, Any],
        rank: int = 0,
    ):
        if rank != 0:
            return

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "phase": phase,
            "num_classes": num_classes,
            "context_length": context_length,
            "patchtst_config": self.model.config.to_dict(),
            "label_mapping": label_mapping,
            "args": args,
            "head_was_reset": self.head_was_reset,
            "pretrained_source": self.pretrained_source,
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

    def load_checkpoint(
        self,
        *,
        checkpoint_path: str,
        device: str,
        optimizer=None,
        scheduler=None,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint
