# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet编码器适配器 - 用于OpenTSLM架构

基于TSLANet (Time Series Lightweight Adaptive Network)，包含：
- Adaptive Spectral Block (ASB): 频域自适应滤波
- Inverted Convolutional Block (ICB): 特征混合
- 掩码预训练支持

参考: src/TSLANet_Classification/TSLANet_classification.py
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class ICB(nn.Module):
    """Inverted Convolutional Block - 特征混合模块"""
    
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] -> transpose to [B, C, N] for Conv1d
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)  # back to [B, N, C]
        return x


class PatchEmbed(nn.Module):
    """Patch嵌入层 - 使用Conv1d实现重叠patch"""
    
    def __init__(
        self,
        seq_len: int,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 128
    ):
        super().__init__()
        stride = patch_size // 2  # 50%重叠
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L] 时间序列
        Returns:
            [B, N, embed_dim] patch嵌入
        """
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out
    
    def compute_num_patches(self, seq_len: int) -> int:
        """动态计算patch数量"""
        return int((seq_len - self.patch_size) / self.stride + 1)


class AdaptiveSpectralBlock(nn.Module):
    """Adaptive Spectral Block - 频域自适应滤波"""
    
    def __init__(self, dim: int, adaptive_filter: bool = True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )

        trunc_normal_(self.complex_weight_high, std=0.02)
        trunc_normal_(self.complex_weight, std=0.02)
        self.threshold_param = nn.Parameter(torch.rand(1))

    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        """创建自适应高频掩码"""
        B, _, _ = x_fft.shape

        # 计算频域能量
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # 展平并计算中位数
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]
        median_energy = median_energy.view(B, 1)

        # 归一化能量
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)

        # 自适应掩码 (straight-through estimator)
        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_in: [B, N, C]
        Returns:
            [B, N, C]
        """
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # FFT沿时间维度
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # 自适应高频掩码
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # IFFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)

        return x


class TSLANetLayer(nn.Module):
    """TSLANet层 - 包含ASB和ICB"""
    
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 3.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_icb: bool = True,
        use_asb: bool = True,
        adaptive_filter: bool = True,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.use_icb = use_icb
        self.use_asb = use_asb
        
        self.norm1 = norm_layer(dim)
        self.asb = AdaptiveSpectralBlock(dim, adaptive_filter=adaptive_filter)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_icb and self.use_asb:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        elif self.use_icb:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        elif self.use_asb:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        return x


def random_masking_3D(
    xb: torch.Tensor, 
    mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    3D随机掩码 (来自PatchTST)
    
    Args:
        xb: [B, N, D] patch序列
        mask_ratio: 掩码比例
    
    Returns:
        x_masked: 掩码后的序列
        x_kept: 保留的patch
        mask: 二进制掩码 (0=keep, 1=mask)
        ids_restore: 恢复顺序的索引
    """
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)

    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)
    x_ = torch.cat([x_kept, x_removed], dim=1)

    x_masked = torch.gather(
        x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
    )

    mask = torch.ones([bs, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, x_kept, mask, ids_restore


class TSLANetEncoder(TimeSeriesEncoderBase):
    """
    TSLANet编码器 - 适配OpenTSLM架构
    
    特性：
    - Adaptive Spectral Block (ASB): 频域自适应滤波
    - Inverted Convolutional Block (ICB): 特征混合
    - 支持掩码预训练
    - 动态序列长度支持
    
    Args:
        output_dim: 输出维度 (默认128)
        dropout: Dropout比例 (默认0.15)
        patch_size: Patch大小 (默认8)
        emb_dim: 嵌入维度 (默认128)
        depth: Transformer层数 (默认2)
        mlp_ratio: MLP隐藏层比例 (默认3.0)
        use_icb: 是否使用ICB (默认True)
        use_asb: 是否使用ASB (默认True)  
        adaptive_filter: 是否使用自适应滤波 (默认True)
        max_seq_len: 最大序列长度 (默认4096)
    """
    
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.15,
        patch_size: int = 8,
        emb_dim: int = 128,
        depth: int = 2,
        mlp_ratio: float = 3.0,
        use_icb: bool = True,
        use_asb: bool = True,
        adaptive_filter: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__(output_dim, dropout)
        
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.use_icb = use_icb
        self.use_asb = use_asb
        self.adaptive_filter = adaptive_filter
        
        # Patch嵌入 (使用max_seq_len计算最大patch数)
        self.patch_embed = PatchEmbed(
            seq_len=max_seq_len,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=emb_dim
        )
        max_patches = self.patch_embed.num_patches
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_patches, emb_dim), requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        
        # TSLANet层
        self.tsla_blocks = nn.ModuleList([
            TSLANetLayer(
                dim=emb_dim,
                mlp_ratio=mlp_ratio,
                drop=dropout,
                drop_path=dpr[i],
                use_icb=use_icb,
                use_asb=use_asb,
                adaptive_filter=adaptive_filter
            )
            for i in range(depth)
        ])
        
        # 输出投影 (如果output_dim != emb_dim)
        if output_dim != emb_dim:
            self.output_proj = nn.Linear(emb_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
        
        # 初始化权重
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, L] 原始时间序列
        
        Returns:
            [B, N, output_dim] patch级别特征
        """
        B, L = x.shape
        
        # 检查序列长度
        if L % self.patch_size != 0:
            # 填充到patch_size的倍数
            pad_len = self.patch_size - (L % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len))
            L = x.shape[1]
        
        # 添加通道维度 [B, L] -> [B, 1, L]
        x = x.unsqueeze(1)
        
        # Patch嵌入 [B, 1, L] -> [B, N, emb_dim]
        x = self.patch_embed(x)
        N = x.size(1)
        
        # 添加位置嵌入 (截断到实际patch数)
        if N > self.pos_embed.size(1):
            raise ValueError(
                f"序列过长: {L} 时间点产生 {N} patches, 超过最大 {self.pos_embed.size(1)} patches"
            )
        pos = self.pos_embed[:, :N, :]
        x = x + pos
        x = self.pos_drop(x)
        
        # TSLANet层
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x
    
    def pretrain_forward(
        self, 
        x: torch.Tensor, 
        mask_ratio: float = 0.4
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        掩码预训练前向传播
        
        Args:
            x: [B, L] 原始时间序列
            mask_ratio: 掩码比例
        
        Returns:
            preds: [B, N, emb_dim] 预测
            target: [B, N, emb_dim] 目标
            mask: [B, N] 掩码 (1=masked, 0=kept)
        """
        B, L = x.shape
        
        # 填充
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len))
        
        # Patch嵌入
        x = x.unsqueeze(1)
        x = self.patch_embed(x)
        N = x.size(1)
        
        # 位置嵌入
        pos = self.pos_embed[:, :N, :]
        x = x + pos
        x_patched = self.pos_drop(x)  # target
        
        # 随机掩码
        x_masked, _, mask, _ = random_masking_3D(x, mask_ratio=mask_ratio)
        mask = mask.bool()
        
        # 通过TSLANet层
        for tsla_blk in self.tsla_blocks:
            x_masked = tsla_blk(x_masked)
        
        return x_masked, x_patched, mask
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取全局嵌入 (用于RAG检索)
        
        Args:
            x: [B, L] 时间序列
        
        Returns:
            [B, emb_dim] 全局嵌入
        """
        features = self.forward(x)  # [B, N, output_dim]
        return features.mean(dim=1)  # [B, output_dim]
    
    def load_pretrained(self, path: str, strict: bool = True):
        """
        加载预训练权重
        
        Args:
            path: 权重文件路径
            strict: 是否严格匹配
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'encoder_state' in checkpoint:
            state_dict = checkpoint['encoder_state']
        elif 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除可能的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                k = k[6:]
            new_state_dict[k] = v
        
        missing, unexpected = self.load_state_dict(new_state_dict, strict=strict)
        
        if missing:
            print(f"⚠️ 加载预训练权重时缺少: {len(missing)} 个参数")
        if unexpected:
            print(f"⚠️ 加载预训练权重时多余: {len(unexpected)} 个参数")
        
        print(f"✅ 成功加载预训练权重: {path}")
