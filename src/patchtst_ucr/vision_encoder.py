# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
VisionEncoder: 时序图像化 + ViT 编码器

核心设计：
- 将时间序列转换为 2D 图像（参考 TiViT 方法）
- 使用预训练 ViT 提取 patch-level 特征
- 支持多种 ViT 模型（dinov2, clip, siglip, mae）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Optional, Tuple, Literal
from torchvision.transforms import Resize
import torchvision.transforms as T


def get_vit_model(model_name: str, device: str = "cuda"):
    """
    加载预训练 ViT 模型
    
    支持的模型：
    - facebook/dinov2-base, facebook/dinov2-small, facebook/dinov2-large
    - openai/clip-vit-base-patch16, openai/clip-vit-large-patch14
    - google/siglip-base-patch16-224
    - facebook/vit-mae-base
    
    Returns:
        processor: 图像处理器（用于标准化）
        vit: ViT 模型
        hidden_dim: 输出隐藏维度
        num_patches: patch 数量（不含 CLS token）
    """
    model_name_lower = model_name.lower()
    
    if "dinov2" in model_name_lower:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_name)
        vit = AutoModel.from_pretrained(model_name)
        hidden_dim = vit.config.hidden_size
        # DINOv2: 224x224 图像，14x14 patch → 256 patches（不含 CLS）
        num_patches = (224 // vit.config.patch_size) ** 2
        
    elif "clip" in model_name_lower:
        from transformers import CLIPProcessor, CLIPModel
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        vit = model.vision_model
        hidden_dim = vit.config.hidden_size
        num_patches = (vit.config.image_size // vit.config.patch_size) ** 2
        
    elif "siglip" in model_name_lower:
        from transformers import AutoProcessor, AutoModel
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        vit = model.vision_model
        hidden_dim = vit.config.hidden_size
        num_patches = (vit.config.image_size // vit.config.patch_size) ** 2
        
    elif "mae" in model_name_lower:
        from transformers import AutoImageProcessor, ViTMAEForPreTraining
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = ViTMAEForPreTraining.from_pretrained(model_name)
        vit = model.vit
        hidden_dim = vit.config.hidden_size
        num_patches = (224 // vit.config.patch_size) ** 2
        
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return processor, vit, hidden_dim, num_patches


class VisionEncoder(nn.Module):
    """
    时序图像化 + ViT 编码器
    
    Args:
        model_name: 预训练 ViT 模型名称
        layer_idx: 提取特征的层索引（-1 表示最后一层）
        ts_patch_size: 时序切片的 patch 大小
        ts_stride: 时序切片的步长比例（0-1 之间）
        image_size: 输出图像大小（默认 224）
        return_cls_token: 是否返回 CLS token（用于某些下游任务）
    """
    
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        layer_idx: int = -1,
        ts_patch_size: int = 16,
        ts_stride: float = 0.5,
        image_size: int = 224,
        return_cls_token: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        
        self.model_name = model_name
        self.layer_idx = layer_idx
        self.ts_patch_size = ts_patch_size
        self.ts_stride = ts_stride
        self.image_size = image_size
        self.return_cls_token = return_cls_token
        self.device = device
        
        # 加载 ViT 模型
        self.processor, self.vit, self.hidden_dim, self.num_vit_patches = get_vit_model(
            model_name, device
        )
        
        # 用于将 tensor 转为 PIL 图像（某些 processor 需要）
        self.to_pil = T.ToPILImage()
        
        # 截断层（如果需要）
        self._truncate_layers()
        
        print(f"\n{'='*50}")
        print(f"VisionEncoder 初始化完成")
        print(f"  模型: {model_name}")
        print(f"  隐藏维度: {self.hidden_dim}")
        print(f"  ViT patches: {self.num_vit_patches}")
        print(f"  时序 patch size: {ts_patch_size}")
        print(f"  时序 stride: {ts_stride}")
        print(f"{'='*50}\n")
    
    def _truncate_layers(self):
        """截断 ViT 层（用于提取中间层特征）"""
        if self.layer_idx is not None and self.layer_idx != -1:
            if hasattr(self.vit, 'encoder'):
                if hasattr(self.vit.encoder, 'layers'):
                    self.vit.encoder.layers = self.vit.encoder.layers[:self.layer_idx]
                elif hasattr(self.vit.encoder, 'layer'):
                    self.vit.encoder.layer = self.vit.encoder.layer[:self.layer_idx]
    
    def ts2image(
        self,
        x: torch.Tensor,
        patch_size: Optional[int] = None,
        stride: Optional[float] = None,
    ) -> torch.Tensor:
        """
        时间序列转图像（参考 TiViT 实现）
        
        Args:
            x: [B, T, D] 时间序列，D 通常为 1（单变量）
            patch_size: 可选覆盖 patch 大小
            stride: 可选覆盖 stride
            
        Returns:
            [B, 3, image_size, image_size] RGB 图像
        """
        patch_size = patch_size or self.ts_patch_size
        stride = stride or self.ts_stride
        
        # 1) 鲁棒归一化（中位数 + IQR）
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        iqr = q75 - q25
        x = x / (iqr + 1e-5)
        
        # 2) 重排维度: [B, T, D] -> [B, D, T]
        x = einops.rearrange(x, "b t d -> b d t")
        T_len = x.shape[-1]
        
        # 3) 时序切片
        if stride == 1:  # 无重叠
            pad_left = 0
            if T_len % patch_size != 0:
                pad_left = patch_size - T_len % patch_size
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
        elif 0 < stride < 1:  # 重叠切片
            stride_len = max(1, int(patch_size * stride))
            remainder = (T_len - patch_size) % stride_len
            pad_left = stride_len - remainder if remainder != 0 else 0
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
            # [B, D, num_patches, patch_size] -> [B*D, 1, num_patches, patch_size]
            x_2d = einops.rearrange(x_2d, "b d n p -> (b d) 1 n p")
        else:
            raise ValueError(f"stride 应在 (0, 1] 范围内，当前值: {stride}")
        
        # 4) 对比度调整
        min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        x_2d = torch.pow(x_2d, 0.8)  # gamma 校正
        
        # 5) 缩放到 ViT 输入分辨率
        x_resized = Resize(
            (self.image_size, self.image_size), 
            interpolation=T.InterpolationMode.NEAREST,
            antialias=False
        )(x_2d)
        
        # 6) 灰度转 RGB（复制通道）
        image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)
        
        return image_input
    
    def forward_vit(self, images: torch.Tensor) -> torch.Tensor:
        """
        ViT 前向传播，获取 patch-level 特征
        
        Args:
            images: [B, 3, H, W] RGB 图像
            
        Returns:
            [B, num_patches, hidden_dim] patch 特征（不含 CLS token）
        """
        device = images.device
        
        # 使用 processor 进行标准化
        # 注意：某些 processor 需要 PIL 图像
        if hasattr(self.processor, 'image_processor'):
            # CLIP/SigLIP 风格
            images_list = [self.to_pil(img.cpu()) for img in images]
            inputs = self.processor(images=images_list, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        else:
            # DINOv2/MAE 风格
            images_list = [self.to_pil(img.cpu()) for img in images]
            inputs = self.processor(images=images_list, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
        
        # ViT 前向传播
        outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=(self.layer_idx is None),
        )
        
        # 获取 hidden states
        hidden_states = outputs.last_hidden_state  # [B, 1+num_patches, hidden_dim]
        
        # 返回 patch 特征（可选是否包含 CLS token）
        if self.return_cls_token:
            return hidden_states  # [B, 1+num_patches, hidden_dim]
        else:
            return hidden_states[:, 1:, :]  # [B, num_patches, hidden_dim]，去除 CLS
    
    def forward(
        self,
        past_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        完整前向传播：时序 → 图像 → patch 特征
        
        Args:
            past_values: [B, T, 1] 或 [B, T, D] 时间序列
            
        Returns:
            [B, num_patches, hidden_dim] patch-level 特征
        """
        # 1) 时序图像化
        images = self.ts2image(past_values)  # [B*D, 3, H, W]
        
        # 2) ViT 编码
        patch_features = self.forward_vit(images)  # [B*D, num_patches, hidden_dim]
        
        # 如果是多变量，需要合并通道维度
        B = past_values.size(0)
        D = past_values.size(-1)
        if D > 1:
            # [B*D, num_patches, H] -> [B, D*num_patches, H]
            patch_features = einops.rearrange(
                patch_features, "(b d) n h -> b (d n) h", b=B, d=D
            )
        
        return patch_features
    
    def get_output_dim(self) -> int:
        """返回输出特征维度"""
        return self.hidden_dim
    
    def get_num_patches(self) -> int:
        """返回 patch 数量"""
        return self.num_vit_patches
    
    def freeze(self):
        """冻结 ViT 参数"""
        for param in self.vit.parameters():
            param.requires_grad = False
        print("🧊 VisionEncoder (ViT) 已冻结")
    
    def unfreeze(self):
        """解冻 ViT 参数"""
        for param in self.vit.parameters():
            param.requires_grad = True
        print("🔥 VisionEncoder (ViT) 已解冻")
    
    def count_parameters(self) -> int:
        """计算参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
