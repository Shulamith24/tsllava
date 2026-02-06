# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
SmallTransformerAggregator: 小 Transformer 聚合器

设计特点：
- 2-6 层 Transformer encoder
- 正弦位置编码
- 可配置参数量（约 5-15M）
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
        Returns:
            [B, L, D] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SmallTransformerAggregator(nn.Module):
    """
    小 Transformer 聚合器
    
    Args:
        num_layers: Transformer 层数（2-6）
        hidden_size: Hidden 维度（512, 768, 1024）
        num_heads: Attention heads（通常 8 或 12）
        ffn_dim: FFN 中间层维度（通常 2048 或 4096）
        dropout: Dropout 概率
        activation: 激活函数（'relu' 或 'gelu'）
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        hidden_size: int = 768,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_seq_len: int = 5000,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model=hidden_size,
            dropout=dropout,
            max_len=max_seq_len,
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-LN（更稳定）
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),  # 最终 LayerNorm
        )
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier 初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, L, H] 输入 embeddings
            attention_mask: [B, L] 1 表示保留，0 表示 padding
            
        Returns:
            [B, L, H] 输出 hidden states
        """
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 转换 attention mask 到 Transformer 格式
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)  # [B, L]
        else:
            src_key_padding_mask = None
        
        # Transformer 前向传播
        output = self.transformer(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        return output  # [B, L, H]
    
    def count_parameters(self) -> int:
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
