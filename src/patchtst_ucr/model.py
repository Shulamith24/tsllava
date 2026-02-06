# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTSTWithAggregator: PatchTST Backbone + Transformer 聚合头

核心设计：
- 使用 HuggingFace PatchTSTModel (use_cls_token=False) 提取 patch 级特征
- 添加可学习 [ANS] token 到序列末尾
- 使用 SmallTransformerAggregator 进行特征聚合
- 提取 [ANS] 位置的 hidden state 进行分类
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from transformers import PatchTSTConfig, PatchTSTModel

from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector


class PatchTSTWithAggregator(nn.Module):
    """
    PatchTST Backbone + SmallTransformerAggregator 分类器
    
    Args:
        num_classes: 分类类别数
        context_length: 输入序列长度
        patch_length: Patch 长度
        stride: Patch 步长
        d_model: PatchTST 模型维度
        num_attention_heads: PatchTST attention heads
        num_hidden_layers: PatchTST Transformer 层数
        ffn_dim: PatchTST FFN 维度
        dropout: Dropout 概率
        aggregator_layers: 聚合头 Transformer 层数
        aggregator_hidden_size: 聚合头 hidden size (None 则与 d_model 相同)
        aggregator_num_heads: 聚合头 attention heads
        aggregator_ffn_dim: 聚合头 FFN 维度 (None 则自动计算)
    """

    def __init__(
        self,
        num_classes: int,
        context_length: int,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        # 聚合头参数
        aggregator_layers: int = 1,
        aggregator_hidden_size: Optional[int] = None,
        aggregator_num_heads: int = 8,
        aggregator_ffn_dim: Optional[int] = None,
        # 投影层参数
        projector_type: Literal["mlp", "linear", "none"] = "mlp",
        projector_dropout: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.context_length = context_length
        self.d_model = d_model
        self.device = device
        self.projector_type = projector_type
        
        # 1) PatchTST Backbone (use_cls_token=False)
        patchtst_config = PatchTSTConfig(
            num_input_channels=1,  # 单变量时间序列
            context_length=context_length,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_cls_token=False,  # 不使用 CLS token，输出纯 patch 特征
        )
        
        self.backbone = PatchTSTModel(config=patchtst_config)
        
        # 计算 patch 数量
        self.num_patches = (context_length - patch_length) // stride + 1
        
        # 2) 聚合头配置
        self.aggregator_hidden_size = aggregator_hidden_size or d_model
        self.aggregator_ffn_dim = aggregator_ffn_dim or (self.aggregator_hidden_size * 4)
        
        # 3) 投影层（根据类型选择）
        if projector_type == "none":
            # 无投影层，强制aggregator维度与d_model相同
            if aggregator_hidden_size is not None and aggregator_hidden_size != d_model:
                print(f"⚠️  projector_type='none' 时，aggregator_hidden_size被强制设为{d_model}")
            self.aggregator_hidden_size = d_model
            self.projector = None
        elif self.aggregator_hidden_size != d_model:
            # 需要投影层
            if projector_type == "mlp":
                self.projector = MLPProjector(d_model, self.aggregator_hidden_size, dropout=projector_dropout)
            elif projector_type == "linear":
                self.projector = LinearProjector(d_model, self.aggregator_hidden_size)
            else:
                raise ValueError(f"Unknown projector_type: {projector_type}")
        else:
            # 维度相同，不需要投影
            self.projector = None
        
        # 4) 聚合头
        self.aggregator = SmallTransformerAggregator(
            num_layers=aggregator_layers,
            hidden_size=self.aggregator_hidden_size,
            num_heads=aggregator_num_heads,
            ffn_dim=self.aggregator_ffn_dim,
            dropout=dropout,
        )
        
        # 4) 可学习 [ANS] token
        self.ans_token = nn.Parameter(
            torch.randn(1, 1, self.aggregator_hidden_size) * 0.02
        )
        
        # 5) 分类头
        self.classifier_head = nn.Linear(self.aggregator_hidden_size, num_classes)
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        aggregator_params = self.aggregator.count_parameters()
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n{'='*60}")
        print(f"PatchTSTWithAggregator 模型信息")
        print(f"{'='*60}")
        print(f"PatchTST Backbone:")
        print(f"  - context_length: {self.context_length}")
        print(f"  - num_patches: {self.num_patches}")
        print(f"  - d_model: {self.d_model}")
        print(f"  - 参数量: {backbone_params:,}")
        print(f"Aggregator:")
        print(f"  - 层数: {self.aggregator.num_layers}")
        print(f"  - hidden_size: {self.aggregator_hidden_size}")
        print(f"  - 参数量: {aggregator_params:,}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*60}\n")
    
    def freeze_backbone(self):
        """冻结 PatchTST backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("🧊 PatchTST backbone 已冻结")
    
    def unfreeze_backbone(self):
        """解冻 PatchTST backbone 参数"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("🔥 PatchTST backbone 已解冻")
    
    def forward(
        self,
        past_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            past_values: [B, context_length, 1] 输入时间序列
            labels: [B] 可选的分类标签
            attention_mask: [B, context_length] 可选的注意力掩码，1表示有效，0表示padding
            
        Returns:
            包含 loss (如果提供 labels) 和 logits 的字典
        """
        B = past_values.size(0)
        device = past_values.device
        
        # 1) PatchTST backbone 提取特征
        backbone_output = self.backbone(past_values=past_values)
        # PatchTST 输出: [B, num_channels, num_patches, d_model]
        # 对于单变量 (num_channels=1)，squeeze 掉 channel 维度
        patch_embeddings = backbone_output.last_hidden_state  # [B, 1, num_patches, d_model]
        if patch_embeddings.dim() == 4:
            patch_embeddings = patch_embeddings.squeeze(1)  # [B, num_patches, d_model]
        
        num_patches = patch_embeddings.size(1)
        
        # 2) 投影到聚合头维度（如果需要）
        if self.projector is not None:
            patch_embeddings = self.projector(patch_embeddings)  # [B, num_patches, aggregator_hidden_size]
        
        # 3) 添加 [ANS] token
        ans_tokens = self.ans_token.expand(B, -1, -1).to(device)  # [B, 1, aggregator_hidden_size]
        sequence = torch.cat([patch_embeddings, ans_tokens], dim=1)  # [B, num_patches+1, H]
        
        # 4) 构建 aggregator 的 attention mask
        if attention_mask is not None:
            # 将时间序列级别的mask转换为patch级别的mask
            # 策略：如果一个patch内有任何有效点，则该patch有效
            patch_length = self.backbone.config.patch_length
            stride = self.backbone.config.stride
            
            # 计算每个patch的有效性
            patch_mask = []
            for i in range(num_patches):
                start_idx = i * stride
                end_idx = min(start_idx + patch_length, attention_mask.size(1))
                # 如果patch范围内有任何有效点，则该patch有效
                patch_valid = attention_mask[:, start_idx:end_idx].sum(dim=1) > 0  # [B]
                patch_mask.append(patch_valid)
            
            patch_mask = torch.stack(patch_mask, dim=1).long()  # [B, num_patches]
            
            # 为 [ANS] token 添加 mask（始终有效）
            ans_mask = torch.ones(B, 1, device=device, dtype=torch.long)
            aggregator_mask = torch.cat([patch_mask, ans_mask], dim=1)  # [B, num_patches+1]
        else:
            aggregator_mask = None
        
        # 5) 聚合头处理
        hidden_states = self.aggregator(sequence, attention_mask=aggregator_mask)  # [B, num_patches+1, H]
        
        # 6) 提取 [ANS] 位置的 hidden state（最后一个位置）
        ans_hidden = hidden_states[:, -1, :]  # [B, H]
        
        # 7) 分类
        logits = self.classifier_head(ans_hidden)  # [B, num_classes]
        
        # 8) 计算损失（如果提供标签）
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "ans_hidden": ans_hidden,
        }
    
    def predict(self, past_values: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        with torch.no_grad():
            outputs = self.forward(past_values)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        return predictions
    
    def count_parameters(self) -> int:
        """计算总参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """返回配置"""
        return {
            "num_classes": self.num_classes,
            "context_length": self.context_length,
            "num_patches": self.num_patches,
            "d_model": self.d_model,
            "aggregator_layers": self.aggregator.num_layers,
            "aggregator_hidden_size": self.aggregator_hidden_size,
            "total_params": self.count_parameters(),
        }
