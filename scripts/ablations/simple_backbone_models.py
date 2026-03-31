#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Local PyTorch backbones for simple UCR few-shot baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernels: tuple[int, int, int]):
        super().__init__()
        k1, k2, k3 = kernels
        self.expand = in_channels != out_channels

        self.conv_x = nn.Conv1d(in_channels, out_channels, k1, padding=k1 // 2)
        self.bn_x = nn.BatchNorm1d(out_channels)
        self.conv_y = nn.Conv1d(out_channels, out_channels, k2, padding=k2 // 2)
        self.bn_y = nn.BatchNorm1d(out_channels)
        self.conv_z = nn.Conv1d(out_channels, out_channels, k3, padding=k3 // 2)
        self.bn_z = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if self.expand else nn.Identity()
        self.bn_shortcut = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_x(self.conv_x(x)))
        out = F.relu(self.bn_y(self.conv_y(out)))
        out = self.bn_z(self.conv_z(out))

        shortcut = self.shortcut(x)
        shortcut = self.bn_shortcut(shortcut)
        out = F.relu(out + shortcut)
        return out


class SimpleResNetClassifier(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, feature_maps: int = 128):
        super().__init__()
        kernels = (7, 5, 3)
        self.block_1 = ResNetBlock(input_channels, feature_maps, kernels)
        self.block_2 = ResNetBlock(feature_maps, feature_maps, kernels)
        self.block_3 = ResNetBlock(feature_maps, feature_maps, kernels)
        self.classifier = nn.Linear(feature_maps, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        embedding = F.avg_pool1d(x, x.shape[-1]).squeeze(-1)
        logits = self.classifier(embedding)
        return logits, embedding


class SelfAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        return attended.mean(dim=1)


class SimpleTapNetClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        dropout: float = 0.5,
        use_attention: bool = True,
    ):
        super().__init__()
        if input_channels != 1:
            raise ValueError("SimpleTapNetClassifier currently supports univariate inputs only.")

        self.use_attention = use_attention

        self.conv_1 = nn.Conv1d(input_channels, 256, kernel_size=8, padding="same")
        self.bn_1 = nn.BatchNorm1d(256)
        self.conv_2 = nn.Conv1d(256, 256, kernel_size=5, padding="same")
        self.bn_2 = nn.BatchNorm1d(256)
        self.conv_3 = nn.Conv1d(256, 128, kernel_size=3, padding="same")
        self.bn_3 = nn.BatchNorm1d(128)

        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=128, batch_first=True)
        self.sequence_dropout = nn.Dropout(dropout)

        self.cnn_attention = SelfAttentionPooling(embed_dim=128, dropout=dropout)
        self.lstm_attention = SelfAttentionPooling(embed_dim=128, dropout=dropout)

        self.fc_1 = nn.Linear(256, 500)
        self.norm_1 = nn.LayerNorm(500)
        self.fc_2 = nn.Linear(500, 300)
        self.norm_2 = nn.LayerNorm(300)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(300, num_classes)

    def _encode_cnn(self, x: torch.Tensor) -> torch.Tensor:
        features = F.leaky_relu(self.bn_1(self.conv_1(x)))
        features = F.leaky_relu(self.bn_2(self.conv_2(features)))
        features = F.leaky_relu(self.bn_3(self.conv_3(features)))
        features = features.transpose(1, 2)
        if self.use_attention:
            return self.cnn_attention(features)
        return features.mean(dim=1)

    def _encode_lstm(self, x: torch.Tensor) -> torch.Tensor:
        sequence = x.transpose(1, 2)
        sequence, _ = self.lstm(sequence)
        sequence = self.sequence_dropout(sequence)
        if self.use_attention:
            return self.lstm_attention(sequence)
        return sequence.mean(dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cnn_embedding = self._encode_cnn(x)
        lstm_embedding = self._encode_lstm(x)
        fused = torch.cat([cnn_embedding, lstm_embedding], dim=1)

        hidden = self.fc_1(fused)
        hidden = F.leaky_relu(self.norm_1(hidden))
        hidden = self.head_dropout(hidden)
        embedding = self.fc_2(hidden)
        embedding = F.leaky_relu(self.norm_2(embedding))
        logits = self.classifier(self.head_dropout(embedding))
        return logits, embedding


class InceptionModule(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        nb_filters: int,
        kernel_size: int,
        use_bottleneck: bool,
        bottleneck_size: int = 32,
    ):
        super().__init__()
        self.use_bottleneck = use_bottleneck and input_channels > 1

        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(input_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_input_channels = bottleneck_size
        else:
            self.bottleneck = nn.Identity()
            conv_input_channels = input_channels

        reduced_kernel_size = max(1, int(kernel_size) - 1)
        kernel_sizes = [max(1, reduced_kernel_size // (2**i)) for i in range(3)]
        self.conv_branches = nn.ModuleList(
            [
                nn.Conv1d(
                    conv_input_channels,
                    nb_filters,
                    kernel_size=branch_kernel_size,
                    padding="same",
                    bias=False,
                )
                for branch_kernel_size in kernel_sizes
            ]
        )
        self.max_pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_channels, nb_filters, kernel_size=1, bias=False),
        )
        self.norm = nn.BatchNorm1d(nb_filters * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inception_input = self.bottleneck(x)
        branches = [conv(inception_input) for conv in self.conv_branches]
        branches.append(self.max_pool_branch(x))
        merged = torch.cat(branches, dim=1)
        return F.relu(self.norm(merged))


class InceptionResidualShortcut(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, output_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(output_channels)

    def forward(self, residual: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        projected = self.norm(self.proj(residual))
        return F.relu(projected + x)


class InceptionTimeClassifier(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_size: int = 41,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        bottleneck_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual

        modules = []
        shortcuts = []
        current_channels = input_channels
        output_channels = nb_filters * 4

        for depth_idx in range(depth):
            modules.append(
                InceptionModule(
                    input_channels=current_channels,
                    nb_filters=nb_filters,
                    kernel_size=kernel_size,
                    use_bottleneck=use_bottleneck,
                    bottleneck_size=bottleneck_size,
                )
            )
            current_channels = output_channels
            if self.use_residual and depth_idx % 3 == 2:
                shortcut_in_channels = input_channels if depth_idx == 2 else output_channels
                shortcuts.append(InceptionResidualShortcut(shortcut_in_channels, output_channels))

        self.inception_blocks = nn.ModuleList(modules)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_channels, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual_input = x
        shortcut_idx = 0

        for depth_idx, block in enumerate(self.inception_blocks):
            x = block(x)
            if self.use_residual and depth_idx % 3 == 2:
                x = self.shortcuts[shortcut_idx](residual_input, x)
                residual_input = x
                shortcut_idx += 1

        embedding = F.adaptive_avg_pool1d(x, output_size=1).squeeze(-1)
        embedding = self.head_dropout(embedding)
        logits = self.classifier(embedding)
        return logits, embedding


def build_simple_backbone(model_name: str, *, input_channels: int, num_classes: int, dropout: float) -> nn.Module:
    normalized = model_name.strip().lower()
    if normalized == "resnet":
        return SimpleResNetClassifier(input_channels=input_channels, num_classes=num_classes)
    if normalized == "tapnet":
        return SimpleTapNetClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    if normalized in {"inceptiontime", "inception_time", "inception"}:
        return InceptionTimeClassifier(
            input_channels=input_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported simple backbone: {model_name}")
