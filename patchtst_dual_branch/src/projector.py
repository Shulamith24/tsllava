# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
投影层模块

用于将PatchTST输出投影到聚合器的输入维度
"""

import torch.nn as nn


class MLPProjector(nn.Module):
    """
    MLP投影层
    
    结构: LayerNorm → Linear → GELU → Dropout
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        dropout: Dropout概率（默认0.1）
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            [B, L, output_dim]
        """
        return self.projector(x)


class LinearProjector(nn.Module):
    """
    简单线性投影层
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x: [B, L, input_dim]
        Returns:
            [B, L, output_dim]
        """
        return self.projector(x)
