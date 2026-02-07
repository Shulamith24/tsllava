#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
SpecVisNet 冒烟测试脚本

验证 SpecVisNet 模块可正常导入和执行前向传播。

用法:
    python -m patchtst_ucr.test_specvisnet
"""

import sys
import torch

def test_specvisnet_encoder():
    """测试 SpecVisNetEncoder 的导入和前向传播"""
    print("=" * 60)
    print("SpecVisNet 冒烟测试")
    print("=" * 60)
    
    # 测试导入
    print("\n1️⃣  测试导入...")
    try:
        from patchtst_ucr.specvisnet import (
            LearnableWaveletTransform,
            SwinBackbone,
            FrequencyAttentionModule,
            AdaptiveSpectralBlock2D,
            SpecVisNetEncoder,
        )
        print("   ✅ 所有模块导入成功")
    except ImportError as e:
        print(f"   ❌ 导入失败: {e}")
        return False
    
    # 测试 LearnableWaveletTransform
    print("\n2️⃣  测试 LearnableWaveletTransform...")
    try:
        wavelet = LearnableWaveletTransform(num_scales=32, output_size=224)
        x = torch.randn(2, 100, 1)  # [B, L, D]
        out = wavelet(x)
        print(f"   输入: {x.shape} -> 输出: {out.shape}")
        assert out.shape == (2, 3, 224, 224), f"期望 (2, 3, 224, 224)，得到 {out.shape}"
        print("   ✅ LearnableWaveletTransform 测试通过")
    except Exception as e:
        print(f"   ❌ LearnableWaveletTransform 测试失败: {e}")
        return False
    
    # 测试 SpecVisNetEncoder
    print("\n3️⃣  测试 SpecVisNetEncoder (swin_tiny)...")
    try:
        encoder = SpecVisNetEncoder(
            backbone="swin_tiny",
            num_scales=64,
            learnable_wavelet=True,
            use_fam=True,
            use_asb=True,
            finetune=False,
        )
        x = torch.randn(2, 200, 1)  # [B, L, D]
        out = encoder(x)
        print(f"   输入: {x.shape} -> 输出: {out.shape}")
        print(f"   num_patches={encoder.num_patches}, hidden_size={encoder.hidden_size}")
        assert out.shape[0] == 2, "Batch size 不匹配"
        assert out.shape[1] == encoder.num_patches, "Patch 数量不匹配"
        assert out.shape[2] == encoder.hidden_size, "Hidden size 不匹配"
        print("   ✅ SpecVisNetEncoder 测试通过")
    except Exception as e:
        print(f"   ❌ SpecVisNetEncoder 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试集成到 VisionEncoder
    print("\n4️⃣  测试 VisionEncoder 集成...")
    try:
        from patchtst_ucr.vision_encoder import VisionEncoder
        vision_enc = VisionEncoder(
            encoder_type="specvisnet",
            specvisnet_backbone="swin_tiny",
            learnable_wavelet=True,
            use_fam=True,
            use_asb=True,
            finetune=False,
        )
        # 注意：VisionEncoder 接收时序数据 [B, L, D]
        x = torch.randn(2, 150, 1)
        out = vision_enc(x)
        print(f"   输入: {x.shape} -> 输出: {out.shape}")
        print("   ✅ VisionEncoder 集成测试通过")
    except Exception as e:
        print(f"   ❌ VisionEncoder 集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试 PatchTSTWithVisionBranch (vision_only 模式)
    print("\n5️⃣  测试 PatchTSTWithVisionBranch (vision_only + specvisnet)...")
    try:
        from patchtst_ucr.dual_branch_model import PatchTSTWithVisionBranch
        model = PatchTSTWithVisionBranch(
            num_classes=5,
            context_length=256,
            image_encoder_type="specvisnet",
            specvisnet_backbone="swin_tiny",
            learnable_wavelet=True,
            use_fam=True,
            use_asb=True,
            branch_mode="vision_only",
            device="cpu",
        )
        x = torch.randn(2, 256, 1)  # [B, context_length, 1]
        outputs = model(x)
        logits = outputs["logits"]
        print(f"   输入: {x.shape} -> logits: {logits.shape}")
        assert logits.shape == (2, 5), f"期望 (2, 5)，得到 {logits.shape}"
        print("   ✅ PatchTSTWithVisionBranch 测试通过")
    except Exception as e:
        print(f"   ❌ PatchTSTWithVisionBranch 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有冒烟测试通过!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_specvisnet_encoder()
    sys.exit(0 if success else 1)
