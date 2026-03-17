# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTModel

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.NewTSVisionEncoder import NewTSVisionEncoder
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class NewTSDualBranchEncoder(TimeSeriesEncoderBase):
    """
    PatchTST + vision dual-branch encoder adapted to the OpenTSLM encoder API.

    Input:
        ``[B, L]`` raw univariate time series.

    Output:
        ``[B, N_tokens, ENCODER_OUTPUT_DIM]`` patch/token embeddings, where the
        token dimension is the concatenation of the active branch outputs.
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        context_length: int = 512,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        branch_mode: str = "both",
        vit_model_name: str = "facebook/dinov2-base",
        vit_feature_mode: str = "single",
        vit_layer_idx: int = 4,
        vit_mix_layers: Optional[Sequence[int]] = None,
        vit_patch_size: int = 16,
        vit_stride: float = 0.5,
        vit_truncate_to_feature_layer: bool = True,
        vit_num_hidden_layers: Optional[int] = None,
        projector_type: str = "mlp",
        projector_dropout: float = 0.1,
        freeze_ts_backbone: bool = False,
        freeze_vision_backbone: bool = True,
        device: str = "cuda",
    ):
        super().__init__(output_dim=output_dim, dropout=dropout)

        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if branch_mode not in {"both", "ts_only", "vision_only"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        if projector_type not in {"mlp", "linear"}:
            raise ValueError(f"Unsupported projector_type: {projector_type}")

        self.output_dim = output_dim
        self.context_length = int(context_length)
        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.branch_mode = branch_mode
        self.projector_type = projector_type
        self.projector_dropout = float(projector_dropout)
        self.freeze_ts_backbone_default = bool(freeze_ts_backbone)
        self.freeze_vision_backbone_default = bool(freeze_vision_backbone)
        self.device = device

        if branch_mode in {"both", "ts_only"}:
            patchtst_config = PatchTSTConfig(
                num_input_channels=1,
                context_length=self.context_length,
                patch_length=self.patch_length,
                patch_stride=self.stride,
                d_model=self.d_model,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_cls_token=False,
            )
            self.ts_backbone = PatchTSTModel(config=patchtst_config)
            self.ts_num_patches = max(0, (self.context_length - self.patch_length) // self.stride + 1)
            self.ts_projector = self._build_projector(self.d_model, self.output_dim)
        else:
            self.ts_backbone = None
            self.ts_num_patches = 0
            self.ts_projector = None

        if branch_mode in {"both", "vision_only"}:
            self.vision_encoder = NewTSVisionEncoder(
                model_name=vit_model_name,
                layer_idx=vit_layer_idx,
                feature_mode=vit_feature_mode,
                mix_layers=vit_mix_layers,
                ts_patch_size=vit_patch_size,
                ts_stride=vit_stride,
                truncate_to_feature_layer=vit_truncate_to_feature_layer,
                num_hidden_layers=vit_num_hidden_layers,
                device=device,
            )
            self.vision_hidden_dim = self.vision_encoder.get_output_dim()
            self.vision_num_patches = self.vision_encoder.get_num_patches()
            self.vision_projector = self._build_projector(self.vision_hidden_dim, self.output_dim)
        else:
            self.vision_encoder = None
            self.vision_hidden_dim = 0
            self.vision_num_patches = 0
            self.vision_projector = None

        if freeze_ts_backbone and self.ts_backbone is not None:
            self.freeze_ts_backbone()
        if freeze_vision_backbone and self.vision_encoder is not None:
            self.freeze_vision_backbone()

    def _build_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        if input_dim == output_dim:
            return nn.Identity()
        if self.projector_type == "linear":
            return nn.Linear(input_dim, output_dim)
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(self.projector_dropout),
        )

    def freeze_ts_backbone(self):
        if self.ts_backbone is None:
            return
        for param in self.ts_backbone.parameters():
            param.requires_grad = False

    def unfreeze_ts_backbone(self):
        if self.ts_backbone is None:
            return
        for param in self.ts_backbone.parameters():
            param.requires_grad = True

    def freeze_vision_backbone(self):
        if self.vision_encoder is not None:
            self.vision_encoder.freeze()

    def unfreeze_vision_backbone(self):
        if self.vision_encoder is not None:
            self.vision_encoder.unfreeze()

    def enable_gradient_checkpointing(self):
        if self.ts_backbone is not None and hasattr(self.ts_backbone, "gradient_checkpointing_enable"):
            try:
                self.ts_backbone.gradient_checkpointing_enable()
            except ValueError:
                pass
        if self.vision_encoder is not None:
            self.vision_encoder.enable_gradient_checkpointing()

    def _prepare_past_values(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        past_values = x.unsqueeze(-1)

        if seq_len < self.context_length:
            pad_len = self.context_length - seq_len
            pad = past_values.new_zeros(batch_size, pad_len, 1)
            past_values = torch.cat([past_values, pad], dim=1)
        elif seq_len > self.context_length:
            past_values = past_values[:, : self.context_length, :]

        return past_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input [B, L], got shape {tuple(x.shape)}")

        past_values = self._prepare_past_values(x)
        embeddings = []

        if self.ts_backbone is not None:
            ts_output = self.ts_backbone(past_values=past_values)
            ts_embeddings = ts_output.last_hidden_state
            if ts_embeddings.dim() == 4:
                ts_embeddings = ts_embeddings.squeeze(1)
            ts_embeddings = self.ts_projector(ts_embeddings)
            embeddings.append(ts_embeddings)

        if self.vision_encoder is not None:
            vision_embeddings = self.vision_encoder(past_values)
            vision_embeddings = self.vision_projector(vision_embeddings)
            embeddings.append(vision_embeddings)

        if not embeddings:
            raise RuntimeError("No active branch is available in NewTSDualBranchEncoder")
        if len(embeddings) == 1:
            return embeddings[0]
        return torch.cat(embeddings, dim=1)

    def get_config(self) -> Dict[str, Any]:
        config = {
            "output_dim": self.output_dim,
            "context_length": self.context_length,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "d_model": self.d_model,
            "branch_mode": self.branch_mode,
            "projector_type": self.projector_type,
            "projector_dropout": self.projector_dropout,
            "freeze_ts_backbone": self.freeze_ts_backbone_default,
            "freeze_vision_backbone": self.freeze_vision_backbone_default,
        }
        if self.ts_backbone is not None:
            ts_config = self.ts_backbone.config
            config.update(
                {
                    "num_attention_heads": ts_config.num_attention_heads,
                    "num_hidden_layers": ts_config.num_hidden_layers,
                    "ffn_dim": ts_config.ffn_dim,
                    "dropout": ts_config.dropout,
                }
            )
        if self.vision_encoder is not None:
            config.update(
                {
                    "vit_model_name": self.vision_encoder.model_name,
                    "vit_feature_mode": self.vision_encoder.feature_mode,
                    "vit_layer_idx": self.vision_encoder.layer_idx,
                    "vit_mix_layers": list(self.vision_encoder.mix_layers),
                    "vit_patch_size": self.vision_encoder.ts_patch_size,
                    "vit_stride": self.vision_encoder.ts_stride,
                    "vit_truncate_to_feature_layer": self.vision_encoder.truncate_to_feature_layer,
                    "vit_num_hidden_layers": self.vision_encoder.requested_num_hidden_layers,
                }
            )
        return config
