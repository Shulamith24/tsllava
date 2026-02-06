# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Vision Encoder: 图像编码器模块

支持三种图像编码器类型：
- ViT: 预训练ViT-base-patch16，输出patch格式特征
- ResNet: 预训练ResNet18/50，输出特征图reshape后的patch格式
- CNN: 轻量级CNN，输出特征图reshape后的patch格式
"""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple


class ViTEncoder(nn.Module):
    """
    ViT图像编码器
    
    使用HuggingFace预训练ViT，输出池化前的patch特征
    
    输出格式: [B, num_patches, hidden_size]
    - ViT-base-patch16-224: num_patches=196, hidden_size=768
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        finetune: bool = False,
    ):
        super().__init__()
        from transformers import ViTModel
        
        self.model = ViTModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size  # 768 for base
        self.num_patches = (self.model.config.image_size // self.model.config.patch_size) ** 2  # 196
        
        # 设置是否微调
        self._set_requires_grad(finetune)
    
    def _set_requires_grad(self, finetune: bool):
        for param in self.model.parameters():
            param.requires_grad = finetune
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 图像张量，需要是224x224
            
        Returns:
            [B, num_patches, hidden_size] patch特征（去除CLS token）
        """
        outputs = self.model(pixel_values=images)
        # last_hidden_state: [B, 1+num_patches, hidden_size]
        # 去除第一个CLS token
        patch_features = outputs.last_hidden_state[:, 1:, :]  # [B, 196, 768]
        return patch_features


class ResNetEncoder(nn.Module):
    """
    ResNet图像编码器
    
    使用预训练ResNet18/50，取最后卷积层输出并reshape为patch格式
    
    输出格式: [B, num_patches, hidden_size]
    - ResNet18: num_patches=49 (7x7), hidden_size=512
    - ResNet50: num_patches=49 (7x7), hidden_size=2048
    """
    
    def __init__(
        self,
        model_name: Literal["resnet18", "resnet50"] = "resnet18",
        finetune: bool = False,
    ):
        super().__init__()
        import torchvision.models as models
        
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.hidden_size = 512
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.hidden_size = 2048
        
        # 移除最后的全局平均池化和全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_patches = 49  # 7x7 for 224x224 input
        
        # 设置是否微调
        self._set_requires_grad(finetune)
    
    def _set_requires_grad(self, finetune: bool):
        for param in self.features.parameters():
            param.requires_grad = finetune
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 图像张量
            
        Returns:
            [B, num_patches, hidden_size] patch特征
        """
        # 提取特征图
        features = self.features(images)  # [B, hidden_size, H', W'] e.g., [B, 512, 7, 7]
        B, C, H, W = features.shape
        
        # Reshape为patch格式: [B, H*W, C]
        patch_features = features.view(B, C, H * W).permute(0, 2, 1)  # [B, 49, 512/2048]
        return patch_features


class LightweightCNNEncoder(nn.Module):
    """
    轻量级CNN图像编码器
    
    自定义轻量级CNN网络，适合计算资源有限的场景
    
    输出格式: [B, num_patches, hidden_size]
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 构建卷积层
        layers = []
        current_channels = in_channels
        
        for i in range(num_layers):
            out_channels = hidden_size if i == num_layers - 1 else min(64 * (2 ** i), hidden_size)
            layers.extend([
                nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            current_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # 对于224x224输入，4层stride=2的卷积后大小为14x14
        self.num_patches = 14 * 14  # 196
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 图像张量
            
        Returns:
            [B, num_patches, hidden_size] patch特征
        """
        features = self.features(images)  # [B, hidden_size, H', W']
        B, C, H, W = features.shape
        
        # Reshape为patch格式: [B, H*W, C]
        patch_features = features.view(B, C, H * W).permute(0, 2, 1)
        
        # 更新实际的num_patches
        self.num_patches = H * W
        
        return patch_features


class VisionEncoder(nn.Module):
    """
    统一的图像编码器接口
    
    支持三种图像编码器类型：
    - vit: 预训练ViT-base-patch16
    - resnet: 预训练ResNet18/50
    - cnn: 轻量级CNN
    
    Args:
        encoder_type: 编码器类型
        finetune: 是否微调预训练模型（仅对vit/resnet有效）
        resnet_variant: ResNet变体（仅对resnet有效）
        cnn_hidden_size: CNN隐藏层大小（仅对cnn有效）
    """
    
    def __init__(
        self,
        encoder_type: Literal["vit", "resnet", "cnn"] = "vit",
        finetune: bool = False,
        resnet_variant: Literal["resnet18", "resnet50"] = "resnet18",
        cnn_hidden_size: int = 256,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == "vit":
            self.encoder = ViTEncoder(finetune=finetune)
        elif encoder_type == "resnet":
            self.encoder = ResNetEncoder(model_name=resnet_variant, finetune=finetune)
        elif encoder_type == "cnn":
            self.encoder = LightweightCNNEncoder(hidden_size=cnn_hidden_size)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        self.hidden_size = self.encoder.hidden_size
        self.num_patches = self.encoder.num_patches
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] 图像张量
            
        Returns:
            [B, num_patches, hidden_size] patch特征
        """
        return self.encoder(images)
    
    def get_output_info(self) -> Tuple[int, int]:
        """返回输出信息: (num_patches, hidden_size)"""
        return self.num_patches, self.hidden_size
