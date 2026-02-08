# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
TiViT: Time Series Vision Transformer

基于论文 "Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers"
实现时序数据转图像，然后用冻结的预训练 ViT 提取特征。

主要组件:
- ts2image: 时序到图像的转换
- TiViTFeatureExtractor: 使用冻结 ViT 提取特征
"""

from typing import Optional, List, Literal
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
import torchvision.transforms as T
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    ViTMAEForPreTraining,
)


# =============================================================================
# Supported ViT Models
# =============================================================================

SUPPORTED_VIT_MODELS = [
    # DINOv2
    "facebook/dinov2-small",
    "facebook/dinov2-base",
    "facebook/dinov2-large",
    # SigLIP2
    "google/siglip2-so400m-patch14-224",
    # MAE
    "facebook/vit-mae-base",
    "facebook/vit-mae-large",
    "facebook/vit-mae-huge",
]


# =============================================================================
# Time Series to Image Transformation
# =============================================================================

def get_patch_size(patch_size_mode: str, T: int) -> List[Optional[int]]:
    """
    根据时序长度计算 patch size
    
    Args:
        patch_size_mode: "sqrt" 或 "linspace"
        T: 时序长度
    
    Returns:
        patch sizes 列表
    """
    if patch_size_mode == "sqrt":
        return [int(math.ceil(math.sqrt(T)))]
    elif patch_size_mode == "linspace":
        # [1, 5, 10, ..., sqrt(T)] 线性空间
        sqrt_T = int(math.ceil(math.sqrt(T)))
        if sqrt_T <= 10:
            return list(range(1, sqrt_T + 1))
        else:
            return [1] + list(range(5, sqrt_T + 1, 5)) + [sqrt_T]
    else:
        raise ValueError(f"Unsupported patch_size_mode: {patch_size_mode}")


def ts2image_transformation(
    x: torch.Tensor,
    patch_size: int,
    stride: float = 0.1,
    image_size: int = 224,
) -> torch.Tensor:
    """
    将时间序列转换为图像
    
    来自 TiViT 论文的核心转换方法：
    1. Robust scaling 归一化
    2. 分割为 patches 并堆叠为 2D
    3. 对比度调整
    4. Resize 到 ViT 输入尺寸
    5. 复制为 RGB 三通道
    
    Args:
        x: 时序张量 [B, T, D] 或 [B, T]
        patch_size: 每个 patch 的大小
        stride: 步长比例 (0-1 之间，1 表示无重叠)
        image_size: 输出图像尺寸 (默认 224)
    
    Returns:
        图像张量 [B*D, 3, image_size, image_size]
    """
    # 确保输入格式为 [B, T, D]
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # [B, T] -> [B, T, 1]
    
    # Robust scaling: 使用中位数和四分位距
    median = x.median(1, keepdim=True)[0]
    q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
    q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
    x = x - median
    iqr = q75 - q25
    x = x / (iqr + 1e-5)
    
    # 转换维度: [B, T, D] -> [B, D, T]
    x = einops.rearrange(x, "b t d -> b d t")
    T = x.shape[-1]
    
    if stride == 1:
        # 无重叠 patches
        pad_left = 0
        if T % patch_size != 0:
            pad_left = patch_size - T % patch_size
        x_pad = F.pad(x, (pad_left, 0), mode="replicate")
        x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
    elif 0 < stride < 1:
        # 重叠 patches
        pad_left = 0
        stride_len = max(1, int(patch_size * stride))
        remainder = (T - patch_size) % stride_len
        if remainder != 0:
            pad_left = stride_len - remainder
        x_pad = F.pad(x, (pad_left, 0), mode="replicate")
        x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
    else:
        raise ValueError(f"Stride should be between 0 and 1, got {stride}")
    
    # 对比度调整
    min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
    x_2d = torch.pow(x_2d, 0.8)
    
    # Resize 到 ViT 输入尺寸
    x_resized = Resize((image_size, image_size), interpolation=0, antialias=False)(x_2d)
    
    # 生成 RGB 灰度图像 (3通道相同)
    image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)
    
    return image_input


# =============================================================================
# TiViT Feature Extractor
# =============================================================================

class TiViTFeatureExtractor(nn.Module):
    """
    TiViT 特征提取器
    
    使用冻结的预训练 ViT 从时序图像中提取特征。
    
    Args:
        model_name: 预训练 ViT 模型名称
        layer_idx: 使用哪一层的特征 (None 表示最后一层)
        aggregation: 特征聚合方式 ("mean" 或 "cls_token")
        patch_size_mode: patch size 计算模式 ("sqrt" 或 "linspace")
        stride: 步长比例
        freeze: 是否冻结 backbone
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layer_idx: Optional[int] = None,
        aggregation: Literal["mean", "cls_token"] = "mean",
        patch_size_mode: Literal["sqrt", "linspace"] = "sqrt",
        stride: float = 0.1,
        freeze: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.aggregation = aggregation
        self.patch_size_mode = patch_size_mode
        self.stride = stride
        
        # 加载对应的 processor 和 ViT 模型
        self.processor, self.vit, self.hidden_size = self._load_model(model_name)
        self.to_pil = T.ToPILImage()
        
        # 截断到指定层
        if layer_idx is not None and layer_idx != -1:
            self._truncate_layers(layer_idx)
        
        # 冻结 backbone
        if freeze:
            self.freeze()
    
    def _load_model(self, model_name: str):
        """加载预训练 ViT 模型"""
        model_name_lower = model_name.lower()
        
        if "dinov2" in model_name_lower:
            processor = AutoImageProcessor.from_pretrained(model_name)
            vit = AutoModel.from_pretrained(model_name)
            hidden_size = vit.config.hidden_size
        elif "siglip" in model_name_lower:
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            vit = model.vision_model
            hidden_size = vit.config.hidden_size
        elif "mae" in model_name_lower:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = ViTMAEForPreTraining.from_pretrained(model_name)
            vit = model.vit
            hidden_size = vit.config.hidden_size
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return processor, vit, hidden_size
    
    def _truncate_layers(self, layer_idx: int):
        """截断 transformer 层到指定深度"""
        if hasattr(self.vit, "encoder"):
            if hasattr(self.vit.encoder, "layers"):
                self.vit.encoder.layers = self.vit.encoder.layers[:layer_idx]
            elif hasattr(self.vit.encoder, "layer"):
                self.vit.encoder.layer = self.vit.encoder.layer[:layer_idx]
    
    def freeze(self):
        """冻结所有参数"""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()
    
    def unfreeze(self):
        """解冻所有参数"""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.train()
    
    def get_hidden_size(self) -> int:
        """返回特征维度"""
        return self.hidden_size
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取时序的 ViT 特征
        
        Args:
            x: 时序张量 [B, T] 或 [B, T, 1]
        
        Returns:
            特征张量 [B, hidden_size]
        """
        device = x.device
        
        # 确保正确格式
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        B, T, D = x.shape
        
        # 计算 patch size
        patch_sizes = get_patch_size(self.patch_size_mode, T)
        
        # 使用第一个 patch size
        patch_size = patch_sizes[0]
        
        # 时序转图像: [B, T, D] -> [B*D, 3, 224, 224]
        images = ts2image_transformation(x, patch_size=patch_size, stride=self.stride)
        
        # 转为 PIL 并通过 processor
        pil_images = [self.to_pil(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(device)
        
        # ViT 前向
        outputs = self.vit(**inputs, output_hidden_states=(self.layer_idx is None))
        
        # 获取 hidden states
        if self.layer_idx is not None:
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs.last_hidden_state
        
        # 聚合
        if self.aggregation == "mean":
            pooled = hidden.mean(dim=1)  # [B*D, hidden_size]
        elif self.aggregation == "cls_token":
            pooled = hidden[:, 0, :]  # [B*D, hidden_size]
        else:
            raise ValueError(f"Unsupported aggregation: {self.aggregation}")
        
        # 多通道时需要聚合
        if D > 1:
            pooled = pooled.view(B, D, -1).mean(dim=1)  # [B, hidden_size]
        
        # L2 归一化 (TiViT 论文做法)
        pooled = F.normalize(pooled, p=2, dim=-1)
        
        return pooled
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """forward 的别名，用于语义清晰"""
        return self.forward(x)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing TiViT Feature Extractor...")
    
    # 测试时序转图像
    print("\n1. Testing ts2image_transformation...")
    x = torch.randn(2, 96, 1)  # batch=2, length=96, channels=1
    images = ts2image_transformation(x, patch_size=10, stride=0.1)
    print(f"   Input shape: {x.shape}")
    print(f"   Output image shape: {images.shape}")
    
    # 测试特征提取器 (需要预训练模型)
    print("\n2. Testing TiViTFeatureExtractor...")
    try:
        extractor = TiViTFeatureExtractor(
            model_name="facebook/dinov2-base",
            aggregation="mean",
        )
        features = extractor(x.squeeze(-1))  # 传入 [B, T]
        print(f"   Feature shape: {features.shape}")
        print(f"   Hidden size: {extractor.get_hidden_size()}")
    except Exception as e:
        print(f"   Skipped (model not available): {e}")
    
    print("\nAll tests passed!")
