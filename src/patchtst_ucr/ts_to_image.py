# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Time Series to Image: 时序转图像模块

将单变量时间序列转换为图像格式供VLM处理

支持两种模式：
- Learnable: 可学习的时序转图像模块（使用1D/2D卷积）
- Simple: 简单的时序转图像模块（直接reshape + 插值）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LearnableTimeSeriesToImage(nn.Module):
    """
    可学习的时序转图像模块
    
    将时间序列通过1D和2D卷积转换为图像，捕获趋势、周期模式和多尺度特征。
    
    Args:
        input_dim: 输入变量维度（单变量为1）
        hidden_dim: 隐藏层通道数
        output_channels: 输出图像通道数（RGB为3，灰度为1）
        image_size: 输出图像尺寸（高度=宽度）
        periodicity: 时序周期性，用于生成周期编码
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 48,
        output_channels: int = 3,
        image_size: int = 224,
        periodicity: int = 24,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.periodicity = periodicity
        
        # 输入特征: 原始值(1) + FFT幅度(1) + 周期编码(2) = 4通道
        in_channels = 4
        
        # 1D卷积层：处理时序特征
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        
        # 2D卷积层：生成图像
        self.conv2d_1 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim // 2,
            kernel_size=3,
            padding=1,
        )
        self.conv2d_2 = nn.Conv2d(
            in_channels=hidden_dim // 2,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入时间序列
               - B: batch size
               - L: 序列长度
               - D: 变量数（通常为1）
            
        Returns:
            [B, output_channels, image_size, image_size] 图像张量
        """
        B, L, D = x.shape
        device = x.device
        
        # 1. 生成周期编码 (sin/cos)
        time_steps = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        sin_encoding = torch.sin(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)  # [B, L, 1]
        cos_encoding = torch.cos(time_steps / self.periodicity * (2 * torch.pi)).unsqueeze(-1)  # [B, L, 1]
        periodicity_encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)  # [B, L, 2]
        
        # 2. FFT频率编码（幅度）
        x_fft = torch.fft.rfft(x, dim=1)
        x_fft_mag = torch.abs(x_fft)  # [B, L//2+1, D]
        
        # 填充到与x相同长度
        if x_fft_mag.shape[1] < L:
            pad = torch.zeros(B, L - x_fft_mag.shape[1], D, device=device, dtype=x_fft_mag.dtype)
            x_fft_mag = torch.cat([x_fft_mag, pad], dim=1)  # [B, L, D]
        
        # 3. 组合所有特征: [B, L, 4] (raw + fft + sin + cos)
        # 对于单变量，取第一个通道
        x_raw = x[:, :, 0:1]  # [B, L, 1]
        x_fft_mag = x_fft_mag[:, :, 0:1]  # [B, L, 1]
        
        combined = torch.cat([x_raw, x_fft_mag, periodicity_encoding], dim=-1)  # [B, L, 4]
        
        # 4. 1D卷积: [B, L, 4] -> [B, 4, L] -> [B, hidden_dim, L]
        combined = combined.permute(0, 2, 1)  # [B, 4, L]
        combined = F.relu(self.conv1d(combined))  # [B, hidden_dim, L]
        
        # 5. Reshape为2D: [B, hidden_dim, L] -> [B, hidden_dim, sqrt(L), sqrt(L)]
        # 需要补齐到完美平方
        sqrt_L = int(L ** 0.5) + 1
        target_len = sqrt_L * sqrt_L
        if combined.shape[2] < target_len:
            pad_len = target_len - combined.shape[2]
            combined = F.pad(combined, (0, pad_len), mode='replicate')
        combined = combined[:, :, :target_len].view(B, self.hidden_dim, sqrt_L, sqrt_L)
        
        # 6. 2D卷积生成图像
        combined = F.tanh(self.conv2d_1(combined))
        combined = F.tanh(self.conv2d_2(combined))  # [B, output_channels, sqrt_L, sqrt_L]
        
        # 7. 调整到目标尺寸
        output = F.interpolate(
            combined,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        
        return output  # [B, output_channels, image_size, image_size]


class SimpleTimeSeriesToImage(nn.Module):
    """
    简单的时序转图像模块（非可学习）
    
    将时间序列直接reshape并插值为图像格式。
    
    Args:
        image_size: 输出图像尺寸
        periodicity: 时序周期性，用于2D reshape
        output_channels: 输出图像通道数
    """
    
    def __init__(
        self,
        image_size: int = 224,
        periodicity: int = 24,
        output_channels: int = 3,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.periodicity = periodicity
        self.output_channels = output_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入时间序列
            
        Returns:
            [B, output_channels, image_size, image_size] 图像张量
        """
        B, L, D = x.shape
        
        # 1. 计算padding使L成为periodicity的倍数
        pad_left = 0
        if L % self.periodicity != 0:
            pad_left = self.periodicity - L % self.periodicity
        
        # 2. Reshape: [B, L, D] -> [B, D, L]
        x = x.permute(0, 2, 1)  # [B, D, L]
        
        # 3. Pad
        if pad_left > 0:
            x = F.pad(x, (pad_left, 0), mode='replicate')  # [B, D, L+pad]
        
        # 4. Reshape为2D: [B, D, L] -> [B, 1, H, W]
        new_L = x.shape[2]
        H = self.periodicity
        W = new_L // self.periodicity
        x = x.view(B * D, 1, H, W)  # [B*D, 1, H, W]
        
        # 5. 插值到目标尺寸
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )  # [B*D, 1, image_size, image_size]
        
        # 6. 扩展到output_channels
        x = x.expand(-1, self.output_channels, -1, -1)  # [B*D, output_channels, H, W]
        
        # 7. 对于多变量，取平均；对于单变量，直接reshape
        x = x.view(B, D, self.output_channels, self.image_size, self.image_size)
        x = x.mean(dim=1)  # [B, output_channels, image_size, image_size]
        
        return x


class TimeSeriesToImage(nn.Module):
    """
    统一的时序转图像接口
    
    Args:
        mode: 转换模式，"learnable" 或 "simple"
        image_size: 输出图像尺寸
        periodicity: 时序周期性
        output_channels: 输出图像通道数
        hidden_dim: 可学习模式的隐藏层维度
    """
    
    def __init__(
        self,
        mode: str = "learnable",
        image_size: int = 224,
        periodicity: int = 24,
        output_channels: int = 3,
        hidden_dim: int = 48,
    ):
        super().__init__()
        
        self.mode = mode
        
        if mode == "learnable":
            self.converter = LearnableTimeSeriesToImage(
                input_dim=1,
                hidden_dim=hidden_dim,
                output_channels=output_channels,
                image_size=image_size,
                periodicity=periodicity,
            )
        elif mode == "simple":
            self.converter = SimpleTimeSeriesToImage(
                image_size=image_size,
                periodicity=periodicity,
                output_channels=output_channels,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from ['learnable', 'simple']")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] 输入时间序列
            
        Returns:
            [B, output_channels, image_size, image_size] 图像张量
        """
        return self.converter(x)


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    将图像归一化到[0, 1]范围
    
    Args:
        images: [B, C, H, W] 图像张量
        
    Returns:
        归一化后的图像张量
    """
    # 计算每个图像的min和max
    B = images.shape[0]
    min_vals = images.view(B, -1).min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_vals = images.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    # 避免除零
    epsilon = 1e-5
    scale = (max_vals - min_vals).clamp(min=epsilon)
    
    # 归一化到[0, 1]
    normalized = (images - min_vals) / scale
    
    return normalized
