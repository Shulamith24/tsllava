# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
SpecVisNet - Spectral Vision Network

融合 TSLANet 频谱思想的图像分支架构，包含：
- LearnableWaveletTransform: 可学习连续小波变换
- SwinBackbone: Swin Transformer 骨干网络
- FrequencyAttentionModule: 频率注意力模块
- AdaptiveSpectralBlock2D: 2D 自适应频谱块
- SpecVisNetEncoder: 统一编码器接口
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
from timm.layers.weight_init import trunc_normal_


class LearnableWaveletTransform(nn.Module):
    """
    可学习的连续小波变换层
    
    使用 Conv1d 实现，卷积核初始化为 Morlet 小波，
    中心频率和带宽作为可学习参数。
    
    Args:
        num_scales: 尺度数量（时频图高度）
        min_freq: 最小中心频率
        max_freq: 最大中心频率
        wavelet_length: 小波卷积核长度
        learnable_params: 是否学习小波参数
        output_size: 输出图像尺寸
        output_channels: 输出通道数 (3 for RGB)
    """
    
    def __init__(
        self,
        num_scales: int = 64,
        min_freq: float = 0.5,
        max_freq: float = 50.0,
        wavelet_length: int = 64,
        learnable_params: bool = True,
        output_size: int = 224,
        output_channels: int = 3,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        self.wavelet_length = wavelet_length
        self.output_size = output_size
        self.output_channels = output_channels
        
        # 初始化中心频率（对数间隔）
        init_freqs = torch.logspace(
            math.log10(min_freq), 
            math.log10(max_freq), 
            num_scales
        )
        
        # 初始化带宽（与频率成正比）
        init_bandwidths = init_freqs / 2.0
        
        if learnable_params:
            self.center_freqs = nn.Parameter(init_freqs)
            self.bandwidths = nn.Parameter(init_bandwidths)
        else:
            self.register_buffer('center_freqs', init_freqs)
            self.register_buffer('bandwidths', init_bandwidths)
        
        # 时间轴 (用于构建小波核)
        t = torch.linspace(-wavelet_length // 2, wavelet_length // 2, wavelet_length)
        self.register_buffer('time_axis', t)
        
        # 伪彩色映射 (将幅度图转为 RGB)
        self.colormap = nn.Conv2d(1, output_channels, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.colormap.weight)
        nn.init.zeros_(self.colormap.bias)
    
    def _build_wavelet_kernels(self) -> torch.Tensor:
        """
        构建 Morlet 小波卷积核
        
        Returns:
            [num_scales, 1, wavelet_length] 复数小波核
        """
        t = self.time_axis.unsqueeze(0)  # [1, L]
        freqs = self.center_freqs.unsqueeze(1)  # [S, 1]
        sigmas = 1.0 / (self.bandwidths.unsqueeze(1) + 1e-6)  # [S, 1]
        
        # Morlet 小波: ψ(t) = exp(-t²/2σ²) * exp(2πift)
        gaussian = torch.exp(-0.5 * (t / sigmas) ** 2)
        sinusoid_real = torch.cos(2 * math.pi * freqs * t / self.wavelet_length)
        sinusoid_imag = torch.sin(2 * math.pi * freqs * t / self.wavelet_length)
        
        wavelet_real = gaussian * sinusoid_real  # [S, L]
        wavelet_imag = gaussian * sinusoid_imag  # [S, L]
        
        # 归一化
        norm = torch.sqrt(torch.sum(wavelet_real ** 2 + wavelet_imag ** 2, dim=1, keepdim=True) + 1e-8)
        wavelet_real = wavelet_real / norm
        wavelet_imag = wavelet_imag / norm
        
        return wavelet_real.unsqueeze(1), wavelet_imag.unsqueeze(1)  # [S, 1, L]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入时间序列 (D 通常为 1)
            
        Returns:
            [B, output_channels, output_size, output_size] RGB 时频图
        """
        B, L, D = x.shape
        
        # 取第一个通道 [B, L]
        x_1d = x[:, :, 0]
        
        # 构建小波核
        wavelet_real, wavelet_imag = self._build_wavelet_kernels()  # [S, 1, K]
        
        # 卷积计算 CWT (padding='same' 保持长度)
        x_1d = x_1d.unsqueeze(1)  # [B, 1, L]
        
        # 手动 padding
        pad_left = self.wavelet_length // 2
        pad_right = self.wavelet_length - pad_left - 1
        x_padded = F.pad(x_1d, (pad_left, pad_right), mode='reflect')
        
        # 实部和虚部卷积
        cwt_real = F.conv1d(x_padded, wavelet_real)  # [B, S, L]
        cwt_imag = F.conv1d(x_padded, wavelet_imag)  # [B, S, L]
        
        # 计算幅度
        magnitude = torch.sqrt(cwt_real ** 2 + cwt_imag ** 2 + 1e-8)  # [B, S, L]
        
        # 归一化到 [0, 1]
        min_val = magnitude.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1)
        max_val = magnitude.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1)
        magnitude = (magnitude - min_val) / (max_val - min_val + 1e-8)
        
        # 添加通道维度 [B, 1, S, L]
        magnitude = magnitude.unsqueeze(1)
        
        # 调整到目标尺寸
        scalogram = F.interpolate(
            magnitude,
            size=(self.output_size, self.output_size),
            mode='bilinear',
            align_corners=False,
        )  # [B, 1, H, W]
        
        # 转换为 RGB
        rgb_image = self.colormap(scalogram)  # [B, 3, H, W]
        
        # 归一化到 [0, 1] 范围（适配预训练模型）
        rgb_image = torch.sigmoid(rgb_image)
        
        return rgb_image


class SwinBackbone(nn.Module):
    """
    Swin Transformer 骨干网络封装
    
    使用 timm 加载预训练模型，输出 patch 级别特征。
    
    Args:
        model_name: Swin 模型名称
        pretrained: 是否加载预训练权重
        finetune: 是否微调
    """
    
    def __init__(
        self,
        model_name: Literal["swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224"] = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        finetune: bool = False,
    ):
        super().__init__()
        
        from timm import create_model
        
        # 加载模型（去除分类头）
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去除分类头
            global_pool='',  # 去除全局池化，保留 patch 特征
        )
        
        # Swin Tiny: 输出 768 维特征
        self.hidden_size = self.model.num_features
        
        # 计算输出 patch 数量 (224/32 = 7, 7x7=49)
        self.num_patches = 49
        
        # 设置是否微调
        self._set_requires_grad(finetune)
        
        print(f"✅ 加载 {model_name}, hidden_size={self.hidden_size}, num_patches={self.num_patches}")
    
    def _set_requires_grad(self, finetune: bool):
        for param in self.model.parameters():
            param.requires_grad = finetune
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] RGB 图像
            
        Returns:
            [B, num_patches, hidden_size] patch 特征
        """
        # timm Swin 输出格式: [B, H, W, C] (注意不是 [B, C, H, W])
        features = self.model(images)
        
        # 处理不同的输出格式
        if features.dim() == 4:
            # [B, H, W, C] 格式 -> [B, H*W, C]
            B, H, W, C = features.shape
            features = features.view(B, H * W, C)  # [B, H*W, C]
            self.num_patches = H * W
        elif features.dim() == 3:
            # [B, N, C] 格式，直接使用
            self.num_patches = features.shape[1]
        else:
            raise ValueError(f"Unexpected Swin output shape: {features.shape}")
        
        return features


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt 骨干网络封装 (轻量备选)
    
    Args:
        model_name: ConvNeXt 模型名称
        pretrained: 是否加载预训练权重
        finetune: 是否微调
    """
    
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        finetune: bool = False,
    ):
        super().__init__()
        
        from timm import create_model
        
        self.model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
        
        self.hidden_size = self.model.num_features
        self.num_patches = 49  # 7x7 for 224x224 input
        
        self._set_requires_grad(finetune)
        
        print(f"✅ 加载 {model_name}, hidden_size={self.hidden_size}, num_patches={self.num_patches}")
    
    def _set_requires_grad(self, finetune: bool):
        for param in self.model.parameters():
            param.requires_grad = finetune
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] RGB 图像
            
        Returns:
            [B, num_patches, hidden_size] patch 特征
        """
        # ConvNeXt 输出 [B, C, H, W] 格式
        features = self.model(images)  # [B, 768, 7, 7]
        B, C, H, W = features.shape
        
        # 转换为 [B, H*W, C]
        features = features.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, 768]
        
        return features


class FrequencyAttentionModule(nn.Module):
    """
    频率注意力模块 (FAM)
    
    对特征进行频带级别的自适应加权，模拟 TSLANet ASB 的去噪功能。
    
    Args:
        hidden_size: 特征维度
        num_scales: 频率尺度数（对应时频图高度方向）
        reduction: 通道压缩比例
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_scales: int = 7,  # 对应 Swin 输出 7x7
        reduction: int = 4,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # 频率权重生成器
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size // reduction, num_scales),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] patch 特征 (N = num_scales * num_time, e.g., 7*7=49)
            
        Returns:
            [B, N, C] 加权后的特征
        """
        B, N, C = x.shape
        
        # 尝试将 N 分解为 num_scales * num_time
        # 对于 Swin 7x7 输出，N=49, num_scales=7, num_time=7
        num_time = N // self.num_scales
        if N != self.num_scales * num_time:
            # 如果不能整除，跳过 FAM
            return x
        
        # 重塑为 [B, S, T, C] (S=频率轴, T=时间轴)
        x_reshaped = x.view(B, self.num_scales, num_time, C)
        
        # 全局特征：先沿时间和频率维度池化 [B, C]
        global_feat = x_reshaped.mean(dim=(1, 2))  # [B, C]
        
        # 生成频率权重 [B, S]
        freq_weights = self.fc(global_feat)  # [B, num_scales]
        
        # 应用权重 [B, S, 1, 1]
        freq_weights = freq_weights.view(B, self.num_scales, 1, 1)
        x_weighted = x_reshaped * freq_weights
        
        # 恢复形状 [B, N, C]
        return x_weighted.view(B, N, C)


class AdaptiveSpectralBlock2D(nn.Module):
    """
    2D 自适应频谱块
    
    基于 TSLANet 的 ASB，适配 2D 图像特征。
    
    Args:
        dim: 特征维度
        adaptive_filter: 是否使用自适应滤波
    """
    
    def __init__(self, dim: int, adaptive_filter: bool = True):
        super().__init__()
        
        self.adaptive_filter = adaptive_filter
        
        # 复数权重（用于频域加权）
        self.complex_weight = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        self.complex_weight_high = nn.Parameter(
            torch.randn(dim, 2, dtype=torch.float32) * 0.02
        )
        
        trunc_normal_(self.complex_weight, std=0.02)
        trunc_normal_(self.complex_weight_high, std=0.02)
        
        self.threshold_param = nn.Parameter(torch.rand(1))
    
    def create_adaptive_high_freq_mask(self, x_fft: torch.Tensor) -> torch.Tensor:
        """创建自适应高频掩码"""
        B, N, _ = x_fft.shape
        
        # 计算频域能量
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        
        # 计算中位数
        flat_energy = energy.view(B, -1)
        median_energy = flat_energy.median(dim=1, keepdim=True)[0].view(B, 1)
        
        # 归一化能量
        epsilon = 1e-6
        normalized_energy = energy / (median_energy + epsilon)
        
        # 自适应掩码 (straight-through estimator)
        adaptive_mask = (
            (normalized_energy > self.threshold_param).float() - self.threshold_param
        ).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        
        return adaptive_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] patch 特征
            
        Returns:
            [B, N, C] 频域增强后的特征
        """
        B, N, C = x.shape
        dtype = x.dtype
        x = x.to(torch.float32)
        
        # FFT 沿 patch 维度
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        
        if self.adaptive_filter:
            # 自适应高频掩码
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            
            x_weighted = x_weighted + x_weighted2
        
        # IFFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        
        return x


class SpecVisNetEncoder(nn.Module):
    """
    SpecVisNet 统一编码器
    
    组合可学习小波变换、视觉骨干和频谱增强模块。
    
    Args:
        backbone: 骨干网络类型
        num_scales: 小波尺度数
        learnable_wavelet: 是否学习小波参数
        use_fam: 是否使用频率注意力模块
        use_asb: 是否使用自适应频谱块
        finetune: 是否微调骨干网络
        output_size: 生成图像尺寸
    """
    
    def __init__(
        self,
        backbone: Literal["swin_tiny", "swin_small", "convnext_tiny"] = "swin_tiny",
        num_scales: int = 64,
        learnable_wavelet: bool = True,
        use_fam: bool = True,
        use_asb: bool = True,
        finetune: bool = False,
        output_size: int = 224,
    ):
        super().__init__()
        
        self.use_fam = use_fam
        self.use_asb = use_asb
        
        # 1. 可学习小波变换
        self.wavelet_transform = LearnableWaveletTransform(
            num_scales=num_scales,
            learnable_params=learnable_wavelet,
            output_size=output_size,
        )
        
        # 2. 视觉骨干网络
        if backbone == "swin_tiny":
            self.vision_backbone = SwinBackbone(
                model_name="swin_tiny_patch4_window7_224",
                pretrained=True,
                finetune=finetune,
            )
        elif backbone == "swin_small":
            self.vision_backbone = SwinBackbone(
                model_name="swin_small_patch4_window7_224",
                pretrained=True,
                finetune=finetune,
            )
        elif backbone == "convnext_tiny":
            self.vision_backbone = ConvNeXtBackbone(
                model_name="convnext_tiny",
                pretrained=True,
                finetune=finetune,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        self.hidden_size = self.vision_backbone.hidden_size
        self.num_patches = self.vision_backbone.num_patches
        
        # 3. 频率注意力模块
        if use_fam:
            self.fam = FrequencyAttentionModule(
                hidden_size=self.hidden_size,
                num_scales=7,  # Swin/ConvNeXt 输出 7x7
            )
        else:
            self.fam = None
        
        # 4. 自适应频谱块
        if use_asb:
            self.asb = AdaptiveSpectralBlock2D(
                dim=self.hidden_size,
                adaptive_filter=True,
            )
            self.asb_norm = nn.LayerNorm(self.hidden_size)
        else:
            self.asb = None
            self.asb_norm = None
        
        # 打印模型信息
        self._print_info(backbone)
    
    def _print_info(self, backbone: str):
        wavelet_params = sum(p.numel() for p in self.wavelet_transform.parameters())
        backbone_params = sum(p.numel() for p in self.vision_backbone.parameters())
        fam_params = sum(p.numel() for p in self.fam.parameters()) if self.fam else 0
        asb_params = sum(p.numel() for p in self.asb.parameters()) if self.asb else 0
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n{'='*60}")
        print(f"SpecVisNet Encoder 信息")
        print(f"{'='*60}")
        print(f"骨干网络: {backbone}")
        print(f"输出维度: num_patches={self.num_patches}, hidden_size={self.hidden_size}")
        print(f"参数量:")
        print(f"  - 小波变换: {wavelet_params:,}")
        print(f"  - 骨干网络: {backbone_params:,}")
        print(f"  - FAM: {fam_params:,}")
        print(f"  - ASB: {asb_params:,}")
        print(f"  - 总计: {total_params:,}")
        print(f"{'='*60}\n")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入时间序列
            
        Returns:
            [B, num_patches, hidden_size] patch 特征
        """
        # 1. 可学习小波变换 -> RGB 时频图
        images = self.wavelet_transform(x)  # [B, 3, 224, 224]
        
        # 2. 视觉骨干提取特征
        features = self.vision_backbone(images)  # [B, N, C]
        
        # 3. 频率注意力
        if self.fam is not None:
            features = self.fam(features)
        
        # 4. 自适应频谱块
        if self.asb is not None:
            features = features + self.asb(self.asb_norm(features))
        
        return features
    
    def get_output_info(self) -> Tuple[int, int]:
        """返回输出信息: (num_patches, hidden_size)"""
        return self.num_patches, self.hidden_size
