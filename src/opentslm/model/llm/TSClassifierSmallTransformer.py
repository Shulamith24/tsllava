# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSClassifierSmallTransformer: 基于小 Transformer 的时间序列分类器

实验 D: 验证预训练 LLM 是否必要

核心设计：
- 用小 Transformer 替代预训练 LLM
- 参数量对齐 LoRA（约 5-15M）
- 从头训练所有参数
- 输入序列：[TS tokens] + [ANS]
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence

from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model_config import ENCODER_OUTPUT_DIM
from ..projector.MLPProjector import MLPProjector
from ..aggregator.SmallTransformerAggregator import SmallTransformerAggregator


class TSClassifierSmallTransformer(TimeSeriesLLM):
    """
    基于小 Transformer 的时间序列分类器
    
    核心组件：
    - encoder: 时间序列编码器
    - projector: 将编码器输出投影到 hidden_size
    - aggregator: SmallTransformerAggregator（替代 LLM）
    - ans_token: 可学习的 [ANS] 查询向量
    - classifier_head: 分类头 (hidden_size -> num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        aggregator_config: str = "medium",  # "small", "medium", "large"
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        encoder_pretrained_path: Optional[str] = None,
        tslanet_config: Optional[Dict] = None,
    ):
        """
        Args:
            num_classes: 分类类别数
            aggregator_config: 聚合器配置 ("small", "medium", "large")
            num_layers: Transformer 层数（覆盖预设配置）
            hidden_size: Hidden 维度（覆盖预设配置）
            num_heads: Attention heads（覆盖预设配置）
            ffn_dim: FFN 维度（覆盖预设配置）
            dropout: Dropout 概率
            device: 设备
            encoder_type: 编码器类型
            encoder_pretrained_path: 编码器预训练权重
            tslanet_config: TSLANet 配置
        """
        super().__init__(device)
        
        self.num_classes = num_classes

        # 1) encoder
        self.encoder_type = encoder_type
        if encoder_type == "tslanet":
            from ..encoder.TSLANetEncoder import TSLANetEncoder
            config = tslanet_config or {}
            default_config = {
                "output_dim": ENCODER_OUTPUT_DIM,
                "patch_size": 8,
                "emb_dim": 128,
                "depth": 2,
                "dropout": 0.15,
            }
            default_config.update(config)
            self.encoder = TSLANetEncoder(**default_config).to(device)
            self.patch_size = default_config.get("patch_size", 8)
            
            if encoder_pretrained_path:
                self.encoder.load_pretrained(encoder_pretrained_path)
                print(f"✅ Loaded TSLANet pretrained weights from: {encoder_pretrained_path}")
        else:
            self.encoder = TransformerCNNEncoder().to(device)
            self.patch_size = 4
        
        # 2) 创建聚合器配置
        aggregator_kwargs = {}
        if num_layers is not None:
            aggregator_kwargs["num_layers"] = num_layers
        if hidden_size is not None:
            aggregator_kwargs["hidden_size"] = hidden_size
        if num_heads is not None:
            aggregator_kwargs["num_heads"] = num_heads
        if ffn_dim is not None:
            aggregator_kwargs["ffn_dim"] = ffn_dim
        aggregator_kwargs["dropout"] = dropout
        
        # 创建聚合器
        from ..aggregator.SmallTransformerAggregator import create_small_transformer_aggregator
        self.aggregator = create_small_transformer_aggregator(
            config_name=aggregator_config,
            **aggregator_kwargs
        ).to(device)
        
        # 获取 hidden_size
        self.hidden_size = self.aggregator.hidden_size
        
        # 3) projector: ENCODER_OUTPUT_DIM -> hidden_size
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.hidden_size, device=device
        ).to(device)

        # 4) [ANS] token
        self.ans_token = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, device=device) * 0.02
        )

        # 5) 分类头
        self.classifier_head = nn.Linear(
            self.hidden_size, num_classes, device=device
        )
        
        # 打印模型信息
        total_params = self.count_parameters()
        agg_params = self.aggregator.count_parameters()
        print(f"\n{'='*60}")
        print(f"TSClassifierSmallTransformer 模型信息")
        print(f"{'='*60}")
        print(f"聚合器配置: {aggregator_config}")
        print(f"  - 层数: {self.aggregator.num_layers}")
        print(f"  - Hidden size: {self.hidden_size}")
        print(f"  - Num heads: {self.aggregator.num_heads}")
        print(f"  - 聚合器参数: {agg_params:,}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*60}\n")

    def count_parameters(self) -> int:
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def enable_gradient_checkpointing(self):
        """启用梯度检查点（小模型通常不需要）"""
        pass  # 小模型可以不用梯度检查点

    def forward(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """DDP 兼容的 forward 方法"""
        return self.compute_loss(batch)

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理批次数据
        
        输入序列：[TS tokens] + [ANS]
        
        Returns:
            inputs_embeds: [B, L, H]
            attention_mask: [B, L]
            ans_positions: [B]
        """
        device = self.device
        H = self.hidden_size

        # 1) 批量编码时间序列
        ts_list: List[torch.Tensor] = []
        ts_counts: List[int] = []
        
        for sample in batch:
            ts_counts.append(len(sample["time_series"]))
            for ts in sample["time_series"]:
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            # 填充到 patch_size 的倍数
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)

            ts_enc = self.encoder(ts_padded.squeeze(-1))
            ts_proj = self.projector(ts_enc)  # [N_total, N_patches, H]
        else:
            ts_proj = torch.empty(0, 0, H, device=device)

        # 2) 为每个样本构建序列：[TS tokens] + [ANS]
        all_seq_embeds = []
        ts_offset = 0
        
        for i, n_ts in enumerate(ts_counts):
            seq_parts = []
            
            # 添加 TS tokens
            for j in range(n_ts):
                seq_parts.append(ts_proj[ts_offset + j])  # [N_patches, H]
            
            ts_offset += n_ts
            
            # 添加 [ANS]
            seq_parts.append(self.ans_token.squeeze(0))  # [1, H]
            
            # 拼接
            seq = torch.cat(seq_parts, dim=0)  # [L_sample, H]
            all_seq_embeds.append(seq)

        # 3) 批量填充
        inputs_embeds = pad_sequence(
            all_seq_embeds, batch_first=True, padding_value=0.0
        )  # [B, L_max, H]
        
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=device, dtype=torch.long
        )  # [B, L_max]
        
        # 处理填充位置的 mask
        for i, seq in enumerate(all_seq_embeds):
            seq_len = seq.shape[0]
            if seq_len < inputs_embeds.shape[1]:
                attention_mask[i, seq_len:] = 0

        # 4) 计算 ans_positions
        ans_positions = torch.tensor(
            [seq.shape[0] - 1 for seq in all_seq_embeds],
            device=device, dtype=torch.long
        )  # [B]

        return inputs_embeds, attention_mask, ans_positions

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        计算分类损失
        
        Args:
            batch: 批次数据
            
        Returns:
            交叉熵损失
        """
        # 提取整数标签
        labels = torch.tensor(
            [b["int_label"] for b in batch], device=self.device, dtype=torch.long
        )

        # 获取输入 embeddings
        inputs_embeds, attention_mask, ans_positions = self.pad_and_apply_batch(batch)

        # Transformer 聚合
        hidden_states = self.aggregator(inputs_embeds, attention_mask)  # [B, L, H]

        # 提取 [ANS] 位置的 hidden state
        B = hidden_states.size(0)
        ans_hidden = hidden_states[torch.arange(B, device=self.device), ans_positions, :]  # [B, H]

        # 分类头
        logits = self.classifier_head(ans_hidden)  # [B, num_classes]

        # 计算交叉熵损失
        loss = nn.functional.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def predict(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        预测类别
        
        Args:
            batch: 批次数据
            
        Returns:
            预测的类别索引 [B]
        """
        # 获取输入 embeddings
        inputs_embeds, attention_mask, ans_positions = self.pad_and_apply_batch(batch)

        # Transformer 聚合
        hidden_states = self.aggregator(inputs_embeds, attention_mask)

        # 提取 [ANS] 位置的 hidden state
        B = hidden_states.size(0)
        ans_hidden = hidden_states[torch.arange(B, device=self.device), ans_positions, :]

        # 分类头
        logits = self.classifier_head(ans_hidden)

        # 预测
        predictions = torch.argmax(logits, dim=-1)  # [B]

        return predictions

    def get_eos_token(self):
        """返回 EOS token（占位符，小模型不需要）"""
        return "<eos>"
