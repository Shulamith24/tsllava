# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.DynamicPatchTSTBackbone import DynamicPatchTSTBackbone
from opentslm.model.encoder.NewTSPMAAggregator import NewTSPMAAggregator
from opentslm.model.encoder.NewTSVisionEncoder import NewTSVisionEncoder
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class NewTSDualBranchEncoder(TimeSeriesEncoderBase):
    """
    PatchTST + vision dual-branch encoder adapted to the OpenTSLM encoder API.

    Input:
        ``[B, L]`` raw univariate time series.

    Output:
        ``[B, N_tokens, ENCODER_OUTPUT_DIM]`` patch/token embeddings. Without
        PMA, tokens are the concatenation of the active branch outputs; with
        PMA, tokens are the learned slot states.
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        context_length: Optional[int] = None,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        dynamic_length: bool = True,
        ts_positional_encoding: str = "sinusoidal",
        branch_mode: str = "both",
        vit_model_name: str = "facebook/dinov2-base",
        vision_backbone_type: str = "hf_pretrained",
        vit_feature_mode: str = "single",
        vit_layer_idx: int = 4,
        vit_mix_layers: Optional[Sequence[int]] = None,
        vit_patch_size: int = 16,
        vit_stride: float = 0.5,
        vision_2d_mode: str = "legacy_unfold",
        vit_truncate_to_feature_layer: bool = True,
        vit_num_hidden_layers: Optional[int] = None,
        projector_type: str = "mlp",
        projector_dropout: float = 0.1,
        use_pma: bool = False,
        aggregator_layers: int = 2,
        aggregator_hidden_size: Optional[int] = None,
        aggregator_num_heads: int = 8,
        aggregator_ffn_dim: Optional[int] = None,
        aggregator_num_queries: int = 4,
        aggregator_query_mode: str = "shared",
        aggregator_fusion_mode: str = "gated_sum",
        aggregator_gate_type: str = "dynamic",
        aggregator_fuse_layers: int = 1,
        enable_modality_embeddings: bool = False,
        branch_dropout: float = 0.0,
        vision_train_mode: str = "none",
        vision_topk_blocks: int = 4,
        freeze_ts_backbone: bool = False,
        freeze_vision_backbone: bool = True,
        device: str = "cuda",
    ):
        super().__init__(output_dim=output_dim, dropout=dropout)

        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if branch_mode not in {"both", "ts_only", "vision_only"}:
            raise ValueError(f"Unsupported branch_mode: {branch_mode}")
        if projector_type not in {"mlp", "linear"}:
            raise ValueError(f"Unsupported projector_type: {projector_type}")
        if aggregator_layers <= 0:
            raise ValueError("aggregator_layers must be positive")
        if aggregator_num_heads <= 0:
            raise ValueError("aggregator_num_heads must be positive")
        if aggregator_num_queries <= 0:
            raise ValueError("aggregator_num_queries must be positive")
        if aggregator_fuse_layers < 0:
            raise ValueError("aggregator_fuse_layers must be non-negative")
        if not dynamic_length:
            raise ValueError("NewTSDualBranchEncoder now requires dynamic_length=True")
        if ts_positional_encoding != "sinusoidal":
            raise ValueError(f"Unsupported ts_positional_encoding: {ts_positional_encoding}")

        self.output_dim = output_dim
        self.context_length = int(context_length) if context_length is not None else None
        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.branch_mode = branch_mode
        self.dynamic_length = bool(dynamic_length)
        self.ts_positional_encoding = ts_positional_encoding
        self.projector_type = projector_type
        self.projector_dropout = float(projector_dropout)
        self.use_pma = bool(use_pma)
        self.aggregator_layers = int(aggregator_layers)
        self.aggregator_hidden_size = (
            int(aggregator_hidden_size) if aggregator_hidden_size is not None else self.output_dim
        )
        self.aggregator_num_heads = int(aggregator_num_heads)
        self.aggregator_ffn_dim = (
            int(aggregator_ffn_dim)
            if aggregator_ffn_dim is not None
            else self.aggregator_hidden_size * 4
        )
        self.aggregator_num_queries = int(aggregator_num_queries)
        self.aggregator_query_mode = aggregator_query_mode
        self.aggregator_fusion_mode = aggregator_fusion_mode
        self.aggregator_gate_type = aggregator_gate_type
        self.aggregator_fuse_layers = int(aggregator_fuse_layers)
        self.enable_modality_embeddings = bool(enable_modality_embeddings)
        self.branch_dropout = float(branch_dropout)
        self.vision_backbone_type = str(vision_backbone_type)
        self.vision_train_mode = str(vision_train_mode)
        self.vision_topk_blocks = int(vision_topk_blocks)
        self.freeze_ts_backbone_default = bool(freeze_ts_backbone)
        self.freeze_vision_backbone_default = bool(freeze_vision_backbone)
        self.device = device
        self.token_output_dim = self.aggregator_hidden_size if self.use_pma else self.output_dim

        if self.aggregator_hidden_size <= 0:
            raise ValueError("aggregator_hidden_size must be positive")
        if self.aggregator_ffn_dim <= 0:
            raise ValueError("aggregator_ffn_dim must be positive")
        if self.use_pma and self.aggregator_hidden_size % self.aggregator_num_heads != 0:
            raise ValueError("aggregator_hidden_size must be divisible by aggregator_num_heads")
        if not 0.0 <= self.branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0, 1)")

        if branch_mode in {"both", "ts_only"}:
            self.ts_backbone = DynamicPatchTSTBackbone(
                patch_length=self.patch_length,
                d_model=self.d_model,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                stride=self.stride,
                positional_encoding=self.ts_positional_encoding,
            )
            self.ts_projector = self._build_projector(self.d_model, self.token_output_dim)
        else:
            self.ts_backbone = None
            self.ts_projector = None

        if branch_mode in {"both", "vision_only"}:
            self.vision_encoder = NewTSVisionEncoder(
                model_name=vit_model_name,
                vision_backbone_type=vision_backbone_type,
                layer_idx=vit_layer_idx,
                feature_mode=vit_feature_mode,
                mix_layers=vit_mix_layers,
                ts_patch_size=vit_patch_size,
                ts_stride=vit_stride,
                vision_2d_mode=vision_2d_mode,
                truncate_to_feature_layer=vit_truncate_to_feature_layer,
                num_hidden_layers=vit_num_hidden_layers,
                vision_train_mode=self.vision_train_mode,
                vision_topk_blocks=self.vision_topk_blocks,
                device=device,
            )
            self.vision_hidden_dim = self.vision_encoder.get_output_dim()
            self.vision_num_patches = self.vision_encoder.get_num_patches()
            self.vision_projector = self._build_projector(self.vision_hidden_dim, self.token_output_dim)
        else:
            self.vision_encoder = None
            self.vision_hidden_dim = 0
            self.vision_num_patches = 0
            self.vision_projector = None

        if self.use_pma:
            self.aggregator = NewTSPMAAggregator(
                num_layers=self.aggregator_layers,
                hidden_size=self.aggregator_hidden_size,
                num_heads=self.aggregator_num_heads,
                ffn_dim=self.aggregator_ffn_dim,
                num_queries=self.aggregator_num_queries,
                query_mode=self.aggregator_query_mode,
                fusion_mode=self.aggregator_fusion_mode,
                gate_type=self.aggregator_gate_type,
                fuse_layers=self.aggregator_fuse_layers,
                dropout=dropout,
            )
            self.post_aggregator_projector = self._build_projector(
                self.aggregator_hidden_size,
                self.output_dim,
            )
        else:
            self.aggregator = None
            self.post_aggregator_projector = None

        if self.enable_modality_embeddings:
            self.ts_modality_embed = nn.Parameter(torch.zeros(1, 1, self.token_output_dim))
            self.vision_modality_embed = nn.Parameter(torch.zeros(1, 1, self.token_output_dim))
            nn.init.normal_(self.ts_modality_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.vision_modality_embed, mean=0.0, std=0.02)
        else:
            self.register_parameter("ts_modality_embed", None)
            self.register_parameter("vision_modality_embed", None)

        if self.enable_modality_embeddings or self.branch_dropout > 0:
            self.fused_pool_proj = nn.Sequential(
                nn.LayerNorm(self.output_dim),
                nn.Linear(self.output_dim, self.output_dim),
            )
        else:
            self.fused_pool_proj = nn.Identity()

        if freeze_ts_backbone and self.ts_backbone is not None:
            self.freeze_ts_backbone()
        if freeze_vision_backbone and self.vision_encoder is not None:
            self.freeze_vision_backbone()
        elif self.vision_encoder is not None:
            self.vision_encoder.set_trainable_blocks(
                mode=self.vision_train_mode,
                topk=self.vision_topk_blocks,
            )

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

    @staticmethod
    def _mean_pool(tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        return tokens.mean(dim=1)

    def _resolve_runtime_branch_mode(self, runtime_branch_mode: str) -> str:
        runtime_branch_mode = str(runtime_branch_mode).lower()
        if runtime_branch_mode not in {"both", "ts_only", "vision_only"}:
            raise ValueError(f"Unsupported runtime_branch_mode: {runtime_branch_mode}")

        if self.branch_mode != "both":
            return self.branch_mode

        if not self.training or self.branch_dropout <= 0.0 or runtime_branch_mode != "both":
            return runtime_branch_mode

        draw = torch.rand(1).item()
        if draw < self.branch_dropout:
            return "ts_only"
        if draw < 2 * self.branch_dropout:
            return "vision_only"
        return "both"

    def _apply_runtime_branch_mode(
        self,
        ts_embeddings: Optional[torch.Tensor],
        vision_embeddings: Optional[torch.Tensor],
        runtime_branch_mode: str,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        effective_mode = self._resolve_runtime_branch_mode(runtime_branch_mode)
        if effective_mode == "ts_only":
            return ts_embeddings, None, effective_mode
        if effective_mode == "vision_only":
            return None, vision_embeddings, effective_mode
        return ts_embeddings, vision_embeddings, effective_mode

    def _apply_modality_embedding(
        self,
        tokens: Optional[torch.Tensor],
        modality_embed: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        if modality_embed is None:
            return tokens
        return tokens + modality_embed.to(device=tokens.device, dtype=tokens.dtype)

    def forward(
        self,
        x: torch.Tensor,
        *,
        runtime_branch_mode: str = "both",
        return_intermediates: bool = False,
    ):
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input [B, L], got shape {tuple(x.shape)}")

        ts_embeddings = None
        vision_embeddings = None

        if self.ts_backbone is not None:
            ts_embeddings = self.ts_backbone(x)
            ts_embeddings = self.ts_projector(ts_embeddings)

        if self.vision_encoder is not None:
            vision_embeddings = self.vision_encoder(x.unsqueeze(-1))
            vision_embeddings = self.vision_projector(vision_embeddings)

        if ts_embeddings is None and vision_embeddings is None:
            raise RuntimeError("No active branch is available in NewTSDualBranchEncoder")

        ts_runtime_tokens, vision_runtime_tokens, effective_mode = self._apply_runtime_branch_mode(
            ts_embeddings,
            vision_embeddings,
            runtime_branch_mode=runtime_branch_mode,
        )
        pooled_ts = self._mean_pool(ts_runtime_tokens)
        pooled_vision = self._mean_pool(vision_runtime_tokens)

        ts_fused_input = self._apply_modality_embedding(ts_runtime_tokens, self.ts_modality_embed)
        vision_fused_input = self._apply_modality_embedding(vision_runtime_tokens, self.vision_modality_embed)

        if self.use_pma:
            agg_outputs = self.aggregator(
                ts_tokens=ts_fused_input,
                vision_tokens=vision_fused_input,
            )
            fused_tokens = self.post_aggregator_projector(agg_outputs["slot_states"])
        elif ts_fused_input is None:
            fused_tokens = vision_fused_input
        elif vision_fused_input is None:
            fused_tokens = ts_fused_input
        else:
            fused_tokens = torch.cat([ts_fused_input, vision_fused_input], dim=1)

        if not return_intermediates:
            return fused_tokens

        pooled_fused = self._mean_pool(fused_tokens)
        if pooled_fused is not None:
            pooled_fused = self.fused_pool_proj(pooled_fused)

        return {
            "ts_tokens": ts_runtime_tokens,
            "vision_tokens": vision_runtime_tokens,
            "fused_tokens": fused_tokens,
            "pooled_ts": pooled_ts,
            "pooled_vision": pooled_vision,
            "pooled_fused": pooled_fused,
            "effective_branch_mode": effective_mode,
        }

    def get_config(self) -> Dict[str, Any]:
        config = {
            "output_dim": self.output_dim,
            "dynamic_length": self.dynamic_length,
            "ts_positional_encoding": self.ts_positional_encoding,
            "patch_length": self.patch_length,
            "stride": self.stride,
            "d_model": self.d_model,
            "branch_mode": self.branch_mode,
            "projector_type": self.projector_type,
            "projector_dropout": self.projector_dropout,
            "use_pma": self.use_pma,
            "aggregator_layers": self.aggregator_layers,
            "aggregator_hidden_size": self.aggregator_hidden_size,
            "aggregator_num_heads": self.aggregator_num_heads,
            "aggregator_ffn_dim": self.aggregator_ffn_dim,
            "aggregator_num_queries": self.aggregator_num_queries,
            "aggregator_query_mode": self.aggregator_query_mode,
            "aggregator_fusion_mode": self.aggregator_fusion_mode,
            "aggregator_gate_type": self.aggregator_gate_type,
            "aggregator_fuse_layers": self.aggregator_fuse_layers,
            "enable_modality_embeddings": self.enable_modality_embeddings,
            "branch_dropout": self.branch_dropout,
            "vision_train_mode": self.vision_train_mode,
            "vision_topk_blocks": self.vision_topk_blocks,
            "freeze_ts_backbone": self.freeze_ts_backbone_default,
            "freeze_vision_backbone": self.freeze_vision_backbone_default,
        }
        if self.ts_backbone is not None:
            config.update(
                {
                    "num_attention_heads": self.ts_backbone.num_attention_heads,
                    "num_hidden_layers": self.ts_backbone.num_hidden_layers,
                    "ffn_dim": self.ts_backbone.ffn_dim,
                    "dropout": self.ts_backbone.dropout,
                }
            )
        if self.vision_encoder is not None:
            config.update(
                {
                    "vit_model_name": self.vision_encoder.model_name,
                    "vision_backbone_type": self.vision_encoder.vision_backbone_type,
                    "vit_feature_mode": self.vision_encoder.feature_mode,
                    "vit_layer_idx": self.vision_encoder.layer_idx,
                    "vit_mix_layers": list(self.vision_encoder.mix_layers),
                    "vit_patch_size": self.vision_encoder.ts_patch_size,
                    "vit_stride": self.vision_encoder.ts_stride,
                    "vision_2d_mode": self.vision_encoder.vision_2d_mode,
                    "vit_truncate_to_feature_layer": self.vision_encoder.truncate_to_feature_layer,
                    "vit_num_hidden_layers": self.vision_encoder.requested_num_hidden_layers,
                }
            )
        return config
