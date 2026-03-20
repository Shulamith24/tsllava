# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn


class DynamicPatchTSTBackbone(nn.Module):
    """
    Lightweight PatchTST-style backbone that derives its token count from the
    runtime batch length instead of a fixed init-time context_length.

    v1 intentionally mirrors the batch-dynamic behavior used by
    ``TransformerCNNEncoder``: upstream collate pads sequences to the current
    batch max length, and this module performs only the local replicate padding
    needed to align the final patch.
    """

    def __init__(
        self,
        *,
        patch_length: int,
        stride: int,
        d_model: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        ffn_dim: int,
        dropout: float = 0.1,
        positional_encoding: str = "sinusoidal",
    ):
        super().__init__()

        if patch_length <= 0:
            raise ValueError("patch_length must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if d_model % num_attention_heads != 0:
            raise ValueError("d_model must be divisible by num_attention_heads")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if positional_encoding != "sinusoidal":
            raise ValueError(f"Unsupported positional_encoding: {positional_encoding}")

        self.patch_length = int(patch_length)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.num_attention_heads = int(num_attention_heads)
        self.num_hidden_layers = int(num_hidden_layers)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)
        self.positional_encoding = positional_encoding
        self.gradient_checkpointing = False

        self.patch_projection = nn.Linear(self.patch_length, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.position_dropout = nn.Dropout(self.dropout)
        self.output_norm = nn.LayerNorm(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_attention_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_hidden_layers)

    def gradient_checkpointing_enable(self):
        # v1 keeps the implementation simple and accepts the flag as a no-op.
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_num_patches(self, seq_len: int) -> int:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        padded_len = seq_len + self._compute_pad_right(seq_len)
        return ((padded_len - self.patch_length) // self.stride) + 1

    def _compute_pad_right(self, seq_len: int) -> int:
        if seq_len < self.patch_length:
            return self.patch_length - seq_len
        remainder = (seq_len - self.patch_length) % self.stride
        return (self.stride - remainder) % self.stride

    def _replicate_pad_right(self, x: torch.Tensor, pad_right: int) -> torch.Tensor:
        if pad_right <= 0:
            return x
        last_value = x[:, -1:].expand(-1, pad_right)
        return torch.cat([x, last_value], dim=1)

    def _extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected [B, L] input, got shape {tuple(x.shape)}")
        if x.size(1) <= 0:
            raise ValueError("Sequence length must be positive")

        pad_right = self._compute_pad_right(x.size(1))
        x = self._replicate_pad_right(x, pad_right)
        return x.unfold(dimension=1, size=self.patch_length, step=self.stride)

    def _build_sinusoidal_position_encoding(
        self,
        *,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )

        encoding = torch.zeros(length, self.d_model, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[: encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self._extract_patches(x)
        token_states = self.patch_projection(patches)
        token_states = self.input_norm(token_states)

        position_encoding = self._build_sinusoidal_position_encoding(
            length=token_states.size(1),
            device=token_states.device,
            dtype=token_states.dtype,
        )
        token_states = self.position_dropout(token_states + position_encoding)
        token_states = self.encoder(token_states)
        return self.output_norm(token_states)
