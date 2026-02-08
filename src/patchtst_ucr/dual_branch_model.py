# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTSTWithVisionBranch: 双分支时序分类模型

核心设计：
- 时序分支：PatchTST backbone 提取 patch 特征
- 图像分支：时序图像化 + ViT 编码器提取 patch 特征
- 融合：拼接两个分支的 patch 序列 + [ANS] token
- 聚合：Transformer Aggregator 进行特征聚合和分类
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from transformers import PatchTSTConfig, PatchTSTModel

from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector
from .vision_encoder import VisionEncoder


class PatchTSTWithVisionBranch(nn.Module):
    """
    双分支时序分类模型：PatchTST + VisionEncoder + Aggregator
    
    Args:
        num_classes: 分类类别数
        context_length: 输入序列长度
        
        # PatchTST 时序分支参数
        patch_length: Patch 长度
        stride: Patch 步长
        d_model: PatchTST 模型维度
        num_attention_heads: PatchTST attention heads
        num_hidden_layers: PatchTST Transformer 层数
        ffn_dim: PatchTST FFN 维度
        dropout: Dropout 概率
        
        # Vision 分支参数
        vit_model_name: ViT 模型名称
        vit_layer_idx: ViT 特征提取层索引
        vit_patch_size: 时序图像化 patch 大小
        vit_stride: 时序图像化步长比例
        
        # Aggregator 参数
        aggregator_layers: 聚合头 Transformer 层数
        aggregator_hidden_size: 聚合头 hidden size
        aggregator_num_heads: 聚合头 attention heads
        aggregator_ffn_dim: 聚合头 FFN 维度
        
        # 投影层参数
        projector_type: 投影层类型
        projector_dropout: 投影层 dropout
        
        # 分支控制
        branch_mode: 分支模式 ("both", "ts_only", "vision_only")
        freeze_ts_backbone: 是否冻结 PatchTST backbone
        freeze_vision_backbone: 是否冻结 Vision backbone
    """

    def __init__(
        self,
        num_classes: int,
        context_length: int,
        # PatchTST 时序分支参数
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        # Vision 分支参数
        vit_model_name: str = "facebook/dinov2-base",
        vit_layer_idx: int = -1,
        vit_patch_size: int = 16,
        vit_stride: float = 0.5,
        # Aggregator 参数
        aggregator_layers: int = 1,
        aggregator_hidden_size: Optional[int] = None,
        aggregator_num_heads: int = 8,
        aggregator_ffn_dim: Optional[int] = None,
        # 投影层参数
        projector_type: Literal["mlp", "linear", "none"] = "mlp",
        projector_dropout: float = 0.1,
        # 分支控制
        branch_mode: Literal["both", "ts_only", "vision_only"] = "both",
        freeze_ts_backbone: bool = False,
        freeze_vision_backbone: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.context_length = context_length
        self.d_model = d_model
        self.device = device
        self.branch_mode = branch_mode
        self.projector_type = projector_type
        
        # ============ 1) 时序分支：PatchTST Backbone ============
        patchtst_config = PatchTSTConfig(
            num_input_channels=1,
            context_length=context_length,
            patch_length=patch_length,
            stride=stride,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            use_cls_token=False,
        )
        
        self.ts_backbone = PatchTSTModel(config=patchtst_config)
        self.ts_num_patches = (context_length - patch_length) // stride + 1
        
        # ============ 2) 图像分支：VisionEncoder ============
        if branch_mode in ["both", "vision_only"]:
            self.vision_encoder = VisionEncoder(
                model_name=vit_model_name,
                layer_idx=vit_layer_idx,
                ts_patch_size=vit_patch_size,
                ts_stride=vit_stride,
                device=device,
            )
            self.vision_hidden_dim = self.vision_encoder.get_output_dim()
            self.vision_num_patches = self.vision_encoder.get_num_patches()
        else:
            self.vision_encoder = None
            self.vision_hidden_dim = 0
            self.vision_num_patches = 0
        
        # ============ 3) Aggregator 配置 ============
        self.aggregator_hidden_size = aggregator_hidden_size or d_model
        self.aggregator_ffn_dim = aggregator_ffn_dim or (self.aggregator_hidden_size * 4)
        
        # ============ 4) 投影层 ============
        # 时序分支投影层
        if branch_mode in ["both", "ts_only"]:
            if projector_type == "none":
                if self.aggregator_hidden_size != d_model:
                    print(f"⚠️  projector_type='none' 时，aggregator_hidden_size 被强制设为 {d_model}")
                    self.aggregator_hidden_size = d_model
                self.ts_projector = None
            elif self.aggregator_hidden_size != d_model:
                if projector_type == "mlp":
                    self.ts_projector = MLPProjector(d_model, self.aggregator_hidden_size, dropout=projector_dropout)
                else:
                    self.ts_projector = LinearProjector(d_model, self.aggregator_hidden_size)
            else:
                self.ts_projector = None
        else:
            self.ts_projector = None
        
        # 图像分支投影层
        if branch_mode in ["both", "vision_only"]:
            if self.vision_hidden_dim != self.aggregator_hidden_size:
                if projector_type == "mlp":
                    self.vision_projector = MLPProjector(
                        self.vision_hidden_dim, self.aggregator_hidden_size, dropout=projector_dropout
                    )
                else:
                    self.vision_projector = LinearProjector(
                        self.vision_hidden_dim, self.aggregator_hidden_size
                    )
            else:
                self.vision_projector = None
        else:
            self.vision_projector = None
        
        # ============ 5) 计算总 patch 数量 ============
        if branch_mode == "both":
            self.total_patches = self.ts_num_patches + self.vision_num_patches
        elif branch_mode == "ts_only":
            self.total_patches = self.ts_num_patches
        else:  # vision_only
            self.total_patches = self.vision_num_patches
        
        # ============ 6) Aggregator ============
        self.aggregator = SmallTransformerAggregator(
            num_layers=aggregator_layers,
            hidden_size=self.aggregator_hidden_size,
            num_heads=aggregator_num_heads,
            ffn_dim=self.aggregator_ffn_dim,
            dropout=dropout,
        )
        
        # ============ 7) [ANS] Token ============
        self.ans_token = nn.Parameter(
            torch.randn(1, 1, self.aggregator_hidden_size) * 0.02
        )
        
        # ============ 8) 分类头 ============
        self.classifier_head = nn.Linear(self.aggregator_hidden_size, num_classes)
        
        # ============ 9) 冻结控制 ============
        if freeze_ts_backbone and branch_mode in ["both", "ts_only"]:
            self.freeze_ts_backbone()
        
        if freeze_vision_backbone and branch_mode in ["both", "vision_only"]:
            self.freeze_vision_backbone()
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        ts_params = sum(p.numel() for p in self.ts_backbone.parameters()) if self.branch_mode != "vision_only" else 0
        vision_params = self.vision_encoder.count_parameters() if self.vision_encoder else 0
        aggregator_params = self.aggregator.count_parameters()
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"PatchTSTWithVisionBranch 模型信息")
        print(f"{'='*60}")
        print(f"分支模式: {self.branch_mode}")
        if self.branch_mode in ["both", "ts_only"]:
            print(f"时序分支 (PatchTST):")
            print(f"  - context_length: {self.context_length}")
            print(f"  - num_patches: {self.ts_num_patches}")
            print(f"  - d_model: {self.d_model}")
            print(f"  - 参数量: {ts_params:,}")
        if self.branch_mode in ["both", "vision_only"]:
            print(f"图像分支 (VisionEncoder):")
            print(f"  - num_patches: {self.vision_num_patches}")
            print(f"  - hidden_dim: {self.vision_hidden_dim}")
            print(f"  - 参数量: {vision_params:,}")
        print(f"Aggregator:")
        print(f"  - 层数: {self.aggregator.num_layers}")
        print(f"  - hidden_size: {self.aggregator_hidden_size}")
        print(f"  - total_patches (含 ANS): {self.total_patches + 1}")
        print(f"  - 参数量: {aggregator_params:,}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def freeze_ts_backbone(self):
        """冻结 PatchTST backbone"""
        for param in self.ts_backbone.parameters():
            param.requires_grad = False
        print("🧊 PatchTST backbone 已冻结")
    
    def unfreeze_ts_backbone(self):
        """解冻 PatchTST backbone"""
        for param in self.ts_backbone.parameters():
            param.requires_grad = True
        print("🔥 PatchTST backbone 已解冻")
    
    def freeze_vision_backbone(self):
        """冻结 Vision backbone"""
        if self.vision_encoder:
            self.vision_encoder.freeze()
    
    def unfreeze_vision_backbone(self):
        """解冻 Vision backbone"""
        if self.vision_encoder:
            self.vision_encoder.unfreeze()
    
    def forward(
        self,
        past_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            past_values: [B, context_length, 1] 输入时间序列
            labels: [B] 可选的分类标签
            
        Returns:
            包含 loss (如果提供 labels) 和 logits 的字典
        """
        B = past_values.size(0)
        device = past_values.device
        
        patch_sequences = []
        
        # 1) 时序分支
        if self.branch_mode in ["both", "ts_only"]:
            ts_output = self.ts_backbone(past_values=past_values)
            ts_embeddings = ts_output.last_hidden_state  # [B, 1, num_patches, d_model]
            if ts_embeddings.dim() == 4:
                ts_embeddings = ts_embeddings.squeeze(1)  # [B, num_patches, d_model]
            
            if self.ts_projector is not None:
                ts_embeddings = self.ts_projector(ts_embeddings)
            
            patch_sequences.append(ts_embeddings)
        
        # 2) 图像分支
        if self.branch_mode in ["both", "vision_only"]:
            vision_embeddings = self.vision_encoder(past_values)  # [B, num_patches, vision_dim]
            
            if self.vision_projector is not None:
                vision_embeddings = self.vision_projector(vision_embeddings)
            
            patch_sequences.append(vision_embeddings)
        
        # 3) 拼接 patch 序列
        combined = torch.cat(patch_sequences, dim=1)  # [B, total_patches, H]
        
        # 4) 添加 [ANS] token
        ans_tokens = self.ans_token.expand(B, -1, -1).to(device)
        sequence = torch.cat([combined, ans_tokens], dim=1)  # [B, total_patches+1, H]
        
        # 5) Aggregator 处理
        hidden_states = self.aggregator(sequence)  # [B, total_patches+1, H]
        
        # 6) 提取 [ANS] 位置的 hidden state
        ans_hidden = hidden_states[:, -1, :]  # [B, H]
        
        # 7) 分类
        logits = self.classifier_head(ans_hidden)  # [B, num_classes]
        
        # 8) 计算损失
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
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """返回配置"""
        return {
            "num_classes": self.num_classes,
            "context_length": self.context_length,
            "branch_mode": self.branch_mode,
            "ts_num_patches": self.ts_num_patches if self.branch_mode != "vision_only" else 0,
            "vision_num_patches": self.vision_num_patches if self.branch_mode != "ts_only" else 0,
            "total_patches": self.total_patches,
            "d_model": self.d_model,
            "aggregator_layers": self.aggregator.num_layers,
            "aggregator_hidden_size": self.aggregator_hidden_size,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": self.count_parameters(),
        }
