# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from typing import Dict, Optional

import torch
import torch.nn as nn


def _build_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class PMABlock(nn.Module):
    """Single PMA block: LN(Q + MHA(Q, X, X)) followed by LN(Q + FFN(Q))."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.cross_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class SlotFusionBlock(nn.Module):
    """Self-attention block operating on PMA slots."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_dim),
            _build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(query=q, key=q, value=q, need_weights=False)
        q = self.norm1(q + attn_out)
        q = self.norm2(q + self.ffn(q))
        return q


class NewTSPMAAggregator(nn.Module):
    """
    Per-modality PMA aggregator with optional slot fusion.

    This adapts the standalone NewTS aggregator into the OpenTSLM encoder path
    and returns slot tokens that can be fed into the LLM prompt directly.
    """

    def __init__(
        self,
        num_layers: int = 2,
        hidden_size: int = 128,
        num_heads: int = 8,
        ffn_dim: int = 512,
        num_queries: int = 4,
        query_mode: str = "shared",
        fusion_mode: str = "gated_sum",
        gate_type: str = "dynamic",
        fuse_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if fuse_layers < 0:
            raise ValueError("fuse_layers must be non-negative")
        if query_mode not in {"shared", "separate"}:
            raise ValueError(f"Unsupported query_mode: {query_mode}")
        if fusion_mode not in {"gated_sum", "concat_linear"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
        if gate_type not in {"scalar", "slot", "dynamic"}:
            raise ValueError(f"Unsupported gate_type: {gate_type}")

        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.ffn_dim = int(ffn_dim)
        self.num_queries = int(num_queries)
        self.query_mode = query_mode
        self.fusion_mode = fusion_mode
        self.gate_type = gate_type
        self.fuse_layers = int(fuse_layers)

        self.query_slots_shared = nn.Parameter(torch.empty(1, self.num_queries, self.hidden_size))
        if self.query_mode == "separate":
            self.query_slots_ts = nn.Parameter(torch.empty(1, self.num_queries, self.hidden_size))
            self.query_slots_vi = nn.Parameter(torch.empty(1, self.num_queries, self.hidden_size))
        else:
            self.query_slots_ts = None
            self.query_slots_vi = None

        self.ts_blocks = nn.ModuleList(
            [
                PMABlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.vi_blocks = nn.ModuleList(
            [
                PMABlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.fusion_mode == "concat_linear":
            self.fusion_proj = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.gate_scalar_logit = None
            self.gate_slot_logits = None
            self.gate_mlp = None
        else:
            self.fusion_proj = None
            if self.gate_type == "scalar":
                self.gate_scalar_logit = nn.Parameter(torch.zeros(1, 1, 1))
                self.gate_slot_logits = None
                self.gate_mlp = None
            elif self.gate_type == "slot":
                self.gate_scalar_logit = None
                self.gate_slot_logits = nn.Parameter(torch.zeros(1, self.num_queries, 1))
                self.gate_mlp = None
            else:
                self.gate_scalar_logit = None
                self.gate_slot_logits = None
                self.gate_mlp = nn.Sequential(
                    nn.Linear(self.hidden_size * 2, self.hidden_size),
                    _build_activation(activation),
                    nn.Linear(self.hidden_size, self.num_queries),
                )

        self.fusion_blocks = nn.ModuleList(
            [
                SlotFusionBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(self.fuse_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(self.hidden_size)
        self.final_dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1 and "query_slots" not in name:
                nn.init.xavier_uniform_(param)
        nn.init.normal_(self.query_slots_shared, mean=0.0, std=0.02)
        if self.query_slots_ts is not None:
            nn.init.normal_(self.query_slots_ts, mean=0.0, std=0.02)
        if self.query_slots_vi is not None:
            nn.init.normal_(self.query_slots_vi, mean=0.0, std=0.02)

    def _expand_queries(self, batch_size: int, modality: str) -> torch.Tensor:
        if self.query_mode == "shared":
            return self.query_slots_shared.expand(batch_size, -1, -1)
        if modality == "ts":
            return self.query_slots_ts.expand(batch_size, -1, -1)
        return self.query_slots_vi.expand(batch_size, -1, -1)

    @staticmethod
    def _to_padding_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        return attention_mask == 0

    def _run_pma(
        self,
        x: torch.Tensor,
        q_init: torch.Tensor,
        blocks: nn.ModuleList,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = q_init
        key_padding_mask = self._to_padding_mask(attention_mask)
        for block in blocks:
            q = block(q, x, key_padding_mask=key_padding_mask)
        return q

    def _fuse_modal_slots(
        self,
        q_ts: torch.Tensor,
        q_vi: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if self.fusion_mode == "concat_linear":
            fused = self.fusion_proj(torch.cat([q_ts, q_vi], dim=-1))
            return {"slot_states": fused, "fusion_alpha": None}

        if self.gate_type == "scalar":
            alpha = torch.sigmoid(self.gate_scalar_logit).expand(q_ts.size(0), -1, -1)
        elif self.gate_type == "slot":
            alpha = torch.sigmoid(self.gate_slot_logits).expand(q_ts.size(0), -1, -1)
        else:
            gate_input = torch.cat([q_ts.mean(dim=1), q_vi.mean(dim=1)], dim=-1)
            alpha = torch.sigmoid(self.gate_mlp(gate_input)).unsqueeze(-1)

        fused = alpha * q_ts + (1.0 - alpha) * q_vi
        return {"slot_states": fused, "fusion_alpha": alpha}

    def forward(
        self,
        ts_tokens: Optional[torch.Tensor] = None,
        vision_tokens: Optional[torch.Tensor] = None,
        ts_attention_mask: Optional[torch.Tensor] = None,
        vision_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if ts_tokens is None and vision_tokens is None:
            raise ValueError("ts_tokens and vision_tokens cannot both be None")

        q_ts = None
        q_vi = None
        batch_size = ts_tokens.size(0) if ts_tokens is not None else vision_tokens.size(0)

        if ts_tokens is not None:
            q_ts = self._run_pma(
                x=ts_tokens,
                q_init=self._expand_queries(batch_size, "ts"),
                blocks=self.ts_blocks,
                attention_mask=ts_attention_mask,
            )

        if vision_tokens is not None:
            q_vi = self._run_pma(
                x=vision_tokens,
                q_init=self._expand_queries(batch_size, "vi"),
                blocks=self.vi_blocks,
                attention_mask=vision_attention_mask,
            )

        if q_ts is not None and q_vi is not None:
            fused_outputs = self._fuse_modal_slots(q_ts, q_vi)
            q = fused_outputs["slot_states"]
            alpha = fused_outputs["fusion_alpha"]
        else:
            q = q_ts if q_ts is not None else q_vi
            alpha = None

        for block in self.fusion_blocks:
            q = block(q)

        q = self.final_norm(q)
        q = self.final_dropout(q)

        return {
            "slot_states": q,
            "slot_states_ts": q_ts,
            "slot_states_vi": q_vi,
            "fusion_alpha": alpha,
        }

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
