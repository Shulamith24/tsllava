# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + VLM图像分支双分支融合模型

核心设计：
- 时序数据同时经过PatchTST分支和图像分支
- 图像分支：时序→图像 → 图像编码器 → patch特征
- 两分支经过独立投影层后融合
- 支持两种融合方式：前后拼接(concat)、交叉注意力(cross_attention)
- 聚合头处理融合后的特征进行分类
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
from transformers import PatchTSTConfig, PatchTSTModel

from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector
from .vision_encoder import VisionEncoder
from .ts_to_image import TimeSeriesToImage, normalize_images


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块
    
    使用时序分支特征作为Query，图像分支特征作为Key/Value
    
    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        dropout: Dropout概率
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        ts_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ts_features: [B, num_ts_patches, hidden_size] 时序分支特征
            vision_features: [B, num_vision_patches, hidden_size] 图像分支特征
            
        Returns:
            [B, num_ts_patches, hidden_size] 融合后的特征
        """
        # 交叉注意力：ts作为query，vision作为key/value
        attn_output, _ = self.cross_attention(
            query=ts_features,
            key=vision_features,
            value=vision_features,
        )
        
        # 残差连接 + LayerNorm
        output = self.layer_norm(ts_features + self.dropout(attn_output))
        
        return output


class PatchTSTWithVisionBranch(nn.Module):
    """
    PatchTST + VLM图像分支双分支融合模型
    
    时序数据同时经过两个分支：
    1. PatchTST分支：提取时序patch特征
    2. 图像分支：时序→图像 → 图像编码器 → patch特征
    
    两分支特征经过投影后融合，再通过聚合头进行分类。
    
    Args:
        num_classes: 分类类别数
        context_length: 输入序列长度
        
        # PatchTST分支参数
        patch_length: Patch长度
        stride: Patch步长
        d_model: PatchTST模型维度
        num_attention_heads: PatchTST attention heads
        num_hidden_layers: PatchTST Transformer层数
        ffn_dim: PatchTST FFN维度
        dropout: Dropout概率
        
        # 图像分支参数
        image_encoder_type: 图像编码器类型 ("vit", "resnet", "cnn")
        image_size: 生成图像尺寸
        learnable_image: 是否使用可学习图像转换
        finetune_vision: 是否微调图像编码器
        resnet_variant: ResNet变体 ("resnet18", "resnet50")
        cnn_hidden_size: CNN隐藏层大小
        periodicity: 时序周期性
        
        # 融合参数
        fusion_type: 融合方式 ("concat", "cross_attention")
        fusion_hidden_size: 融合后的隐藏层大小
        
        # 投影层参数
        projector_type: 投影层类型 ("mlp", "linear", "none")
        projector_dropout: 投影层Dropout
        
        # 聚合头参数
        aggregator_layers: 聚合头层数
        aggregator_num_heads: 聚合头attention heads
        aggregator_ffn_dim: 聚合头FFN维度
    """
    
    def __init__(
        self,
        num_classes: int,
        context_length: int,
        # PatchTST分支参数
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        # 图像分支参数
        image_encoder_type: Literal["vit", "resnet", "cnn"] = "vit",
        image_size: int = 224,
        learnable_image: bool = True,
        finetune_vision: bool = False,
        resnet_variant: Literal["resnet18", "resnet50"] = "resnet18",
        cnn_hidden_size: int = 256,
        periodicity: int = 24,
        # 融合参数
        fusion_type: Literal["concat", "cross_attention"] = "concat",
        fusion_hidden_size: Optional[int] = None,
        # 投影层参数
        projector_type: Literal["mlp", "linear", "none"] = "mlp",
        projector_dropout: float = 0.1,
        # 聚合头参数
        aggregator_layers: int = 1,
        aggregator_num_heads: int = 8,
        aggregator_ffn_dim: Optional[int] = None,
        # 设备
        device: str = "cuda",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.context_length = context_length
        self.d_model = d_model
        self.device = device
        self.fusion_type = fusion_type
        self.projector_type = projector_type
        self.image_encoder_type = image_encoder_type
        
        # ========== 1. PatchTST分支 ==========
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
        
        self.patchtst_backbone = PatchTSTModel(config=patchtst_config)
        self.num_ts_patches = (context_length - patch_length) // stride + 1
        
        # ========== 2. 图像分支 ==========
        # 2.1 时序转图像
        self.ts_to_image = TimeSeriesToImage(
            mode="learnable" if learnable_image else "simple",
            image_size=image_size,
            periodicity=periodicity,
            output_channels=3,
        )
        
        # 2.2 图像编码器
        self.vision_encoder = VisionEncoder(
            encoder_type=image_encoder_type,
            finetune=finetune_vision,
            resnet_variant=resnet_variant,
            cnn_hidden_size=cnn_hidden_size,
        )
        self.num_vision_patches, self.vision_hidden_size = self.vision_encoder.get_output_info()
        
        # ========== 3. 融合相关设置 ==========
        # 确定融合后的隐藏层大小
        self.fusion_hidden_size = fusion_hidden_size or max(d_model, self.vision_hidden_size)
        
        # 3.1 PatchTST分支投影层
        if projector_type == "none":
            if d_model != self.fusion_hidden_size:
                print(f"⚠️  projector_type='none' 时，fusion_hidden_size被强制设为{d_model}")
                self.fusion_hidden_size = d_model
            self.ts_projector = None
        elif d_model != self.fusion_hidden_size:
            if projector_type == "mlp":
                self.ts_projector = MLPProjector(d_model, self.fusion_hidden_size, dropout=projector_dropout)
            else:
                self.ts_projector = LinearProjector(d_model, self.fusion_hidden_size)
        else:
            self.ts_projector = None
        
        # 3.2 图像分支投影层（始终需要，因为视觉编码器输出维度通常不同）
        if self.vision_hidden_size != self.fusion_hidden_size:
            if projector_type == "mlp" or projector_type == "none":
                # 即使projector_type='none'，vision分支仍需要投影
                self.vision_projector = MLPProjector(
                    self.vision_hidden_size, self.fusion_hidden_size, dropout=projector_dropout
                )
            else:
                self.vision_projector = LinearProjector(self.vision_hidden_size, self.fusion_hidden_size)
        else:
            self.vision_projector = None
        
        # 3.3 融合模块
        if fusion_type == "cross_attention":
            self.fusion_module = CrossAttentionFusion(
                hidden_size=self.fusion_hidden_size,
                num_heads=aggregator_num_heads,
                dropout=dropout,
            )
            # 交叉注意力后序列长度为时序分支长度
            self.total_patches = self.num_ts_patches
        else:
            # concat模式
            self.fusion_module = None
            self.total_patches = self.num_ts_patches + self.num_vision_patches
        
        # ========== 4. 聚合头 ==========
        self.aggregator_ffn_dim = aggregator_ffn_dim or (self.fusion_hidden_size * 4)
        
        self.aggregator = SmallTransformerAggregator(
            num_layers=aggregator_layers,
            hidden_size=self.fusion_hidden_size,
            num_heads=aggregator_num_heads,
            ffn_dim=self.aggregator_ffn_dim,
            dropout=dropout,
        )
        
        # ========== 5. [ANS] Token ==========
        self.ans_token = nn.Parameter(
            torch.randn(1, 1, self.fusion_hidden_size) * 0.02
        )
        
        # ========== 6. 分类头 ==========
        self.classifier_head = nn.Linear(self.fusion_hidden_size, num_classes)
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        patchtst_params = sum(p.numel() for p in self.patchtst_backbone.parameters())
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        ts_to_image_params = sum(p.numel() for p in self.ts_to_image.parameters())
        aggregator_params = self.aggregator.count_parameters()
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"\n{'='*70}")
        print(f"PatchTSTWithVisionBranch 模型信息")
        print(f"{'='*70}")
        print(f"PatchTST分支:")
        print(f"  - context_length: {self.context_length}")
        print(f"  - num_patches: {self.num_ts_patches}")
        print(f"  - d_model: {self.d_model}")
        print(f"  - 参数量: {patchtst_params:,}")
        print(f"图像分支:")
        print(f"  - encoder_type: {self.image_encoder_type}")
        print(f"  - num_patches: {self.num_vision_patches}")
        print(f"  - hidden_size: {self.vision_hidden_size}")
        print(f"  - 时序转图像参数量: {ts_to_image_params:,}")
        print(f"  - 视觉编码器参数量: {vision_params:,}")
        print(f"融合:")
        print(f"  - fusion_type: {self.fusion_type}")
        print(f"  - fusion_hidden_size: {self.fusion_hidden_size}")
        print(f"  - total_patches: {self.total_patches}")
        print(f"Aggregator:")
        print(f"  - 层数: {self.aggregator.num_layers}")
        print(f"  - 参数量: {aggregator_params:,}")
        print(f"总参数量: {total_params:,}")
        print(f"{'='*70}\n")
    
    def freeze_patchtst(self):
        """冻结PatchTST backbone参数"""
        for param in self.patchtst_backbone.parameters():
            param.requires_grad = False
        print("🧊 PatchTST backbone 已冻结")
    
    def unfreeze_patchtst(self):
        """解冻PatchTST backbone参数"""
        for param in self.patchtst_backbone.parameters():
            param.requires_grad = True
        print("🔥 PatchTST backbone 已解冻")
    
    def freeze_vision(self):
        """冻结图像编码器参数"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("🧊 Vision encoder 已冻结")
    
    def unfreeze_vision(self):
        """解冻图像编码器参数"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
        print("🔥 Vision encoder 已解冻")
    
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
            attention_mask: [B, context_length] 可选的注意力掩码
            
        Returns:
            包含 loss (如果提供 labels) 和 logits 的字典
        """
        B = past_values.size(0)
        device = past_values.device
        
        # ========== 1. PatchTST分支 ==========
        patchtst_output = self.patchtst_backbone(past_values=past_values)
        ts_embeddings = patchtst_output.last_hidden_state  # [B, 1, num_patches, d_model]
        if ts_embeddings.dim() == 4:
            ts_embeddings = ts_embeddings.squeeze(1)  # [B, num_patches, d_model]
        
        # 投影到融合维度
        if self.ts_projector is not None:
            ts_embeddings = self.ts_projector(ts_embeddings)  # [B, num_patches, fusion_hidden_size]
        
        # ========== 2. 图像分支 ==========
        # 2.1 时序转图像
        images = self.ts_to_image(past_values)  # [B, C, H, W]
        
        # 2.2 归一化图像到[0, 1]
        images = normalize_images(images)
        
        # 2.3 图像编码
        vision_embeddings = self.vision_encoder(images)  # [B, num_vision_patches, vision_hidden_size]
        
        # 2.4 投影到融合维度
        if self.vision_projector is not None:
            vision_embeddings = self.vision_projector(vision_embeddings)  # [B, num_vision_patches, fusion_hidden_size]
        
        # ========== 3. 融合 ==========
        if self.fusion_type == "cross_attention":
            # 交叉注意力融合：ts作为query，vision作为key/value
            fused_embeddings = self.fusion_module(ts_embeddings, vision_embeddings)
            # 输出: [B, num_ts_patches, fusion_hidden_size]
        else:
            # concat融合：沿序列维度拼接
            fused_embeddings = torch.cat([ts_embeddings, vision_embeddings], dim=1)
            # 输出: [B, num_ts_patches + num_vision_patches, fusion_hidden_size]
        
        # ========== 4. 添加[ANS] token ==========
        ans_tokens = self.ans_token.expand(B, -1, -1).to(device)
        sequence = torch.cat([fused_embeddings, ans_tokens], dim=1)  # [B, total_patches+1, H]
        
        # ========== 5. 聚合头处理 ==========
        hidden_states = self.aggregator(sequence)  # [B, total_patches+1, H]
        
        # ========== 6. 提取[ANS]位置的hidden state ==========
        ans_hidden = hidden_states[:, -1, :]  # [B, H]
        
        # ========== 7. 分类 ==========
        logits = self.classifier_head(ans_hidden)  # [B, num_classes]
        
        # ========== 8. 计算损失 ==========
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "ans_hidden": ans_hidden,
            "ts_embeddings": ts_embeddings,
            "vision_embeddings": vision_embeddings,
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
            "num_ts_patches": self.num_ts_patches,
            "num_vision_patches": self.num_vision_patches,
            "d_model": self.d_model,
            "vision_hidden_size": self.vision_hidden_size,
            "fusion_type": self.fusion_type,
            "fusion_hidden_size": self.fusion_hidden_size,
            "total_patches": self.total_patches,
            "image_encoder_type": self.image_encoder_type,
            "aggregator_layers": self.aggregator.num_layers,
            "total_params": self.count_parameters(),
        }
