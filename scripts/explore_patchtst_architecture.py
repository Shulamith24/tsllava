#!/usr/bin/env python3
"""
探索 PatchTST 模型架构

目标：
1. 理解 PatchTST 的前向传播流程
2. 分析 use_cls_token=True 的作用
3. 查看输出的形状和结构
"""

import torch
from transformers import PatchTSTConfig, PatchTSTForClassification, PatchTSTModel

def explore_patchtst_architecture():
    """探索 PatchTST 架构"""
    print("=" * 80)
    print("PatchTST 架构探索")
    print("=" * 80)
    
    # ========== 配置 1: use_cls_token=True ==========
    print("\n" + "=" * 80)
    print("配置 1: use_cls_token=True")
    print("=" * 80)
    
    config_with_cls = PatchTSTConfig(
        num_input_channels=1,  # 单变量时间序列
        num_targets=5,  # 5 类分类
        context_length=128,  # 上下文长度
        patch_length=16,  # 每个 patch 的长度
        stride=8,  # patch 步长
        d_model=64,  # 模型维度
        num_attention_heads=4,
        num_hidden_layers=2,
        use_cls_token=True,  # 使用 CLS token
    )
    
    print(f"\n配置参数:")
    print(f"  num_input_channels: {config_with_cls.num_input_channels}")
    print(f"  num_targets: {config_with_cls.num_targets}")
    print(f"  context_length: {config_with_cls.context_length}")
    print(f"  patch_length: {config_with_cls.patch_length}")
    print(f"  stride: {config_with_cls.stride}")
    print(f"  d_model: {config_with_cls.d_model}")
    print(f"  num_hidden_layers: {config_with_cls.num_hidden_layers}")
    print(f"  use_cls_token: {config_with_cls.use_cls_token}")
    
    # 创建模型
    model_with_cls = PatchTSTForClassification(config=config_with_cls)
    print(f"\n✅ 模型创建成功")
    
    # 打印模型结构
    print(f"\n模型结构:")
    for name, module in model_with_cls.named_children():
        print(f"  - {name}: {module.__class__.__name__}")
    
    # 创建测试输入
    batch_size = 4
    past_values = torch.randn(batch_size, 128, 1)  # [B, L, C]
    print(f"\n输入形状: {past_values.shape}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - context_length: 128")
    print(f"  - num_channels: 1")
    
    # 前向传播
    print(f"\n执行前向传播...")
    model_with_cls.eval()
    with torch.no_grad():
        outputs = model_with_cls(past_values=past_values)
    
    print(f"\n输出结构:")
    print(f"  - prediction_logits shape: {outputs.prediction_logits.shape}")
    if outputs.hidden_states is not None:
        print(f"  - hidden_states: {len(outputs.hidden_states)} layers")
    
    # 计算 patch 数量
    num_patches = (128 - 16) // 8 + 1
    print(f"\n计算的 patch 数量:")
    print(f"  num_patches = (context_length - patch_length) / stride + 1")
    print(f"  num_patches = (128 - 16) / 8 + 1 = {num_patches}")
    
    # ========== 配置 2: use_cls_token=False ==========
    print("\n" + "=" * 80)
    print("配置 2: use_cls_token=False")
    print("=" * 80)
    
    config_no_cls = PatchTSTConfig(
        num_input_channels=1,
        num_targets=5,
        context_length=128,
        patch_length=16,
        stride=8,
        d_model=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        use_cls_token=False,  # 不使用 CLS token
    )
    
    model_no_cls = PatchTSTForClassification(config=config_no_cls)
    print(f"✅ 模型创建成功 (use_cls_token=False)")
    
    with torch.no_grad():
        outputs_no_cls = model_no_cls(past_values=past_values)
    
    print(f"\n输出形状:")
    print(f"  - prediction_logits: {outputs_no_cls.prediction_logits.shape}")
    
    # ========== 对比分析 ==========
    print("\n" + "=" * 80)
    print("对比分析")
    print("=" * 80)
    
    print(f"\nuse_cls_token=True vs use_cls_token=False:")
    print(f"  CLS=True  logits shape: {outputs.prediction_logits.shape}")
    print(f"  CLS=False logits shape: {outputs_no_cls.prediction_logits.shape}")
    
    return model_with_cls, config_with_cls


def explore_intermediate_outputs():
    """探索中间层输出"""
    print("\n" + "=" * 80)
    print("探索中间层输出")
    print("=" * 80)
    
    config = PatchTSTConfig(
        num_input_channels=1,
        num_targets=5,
        context_length=128,
        patch_length=16,
        stride=8,
        d_model=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        use_cls_token=True,
    )
    
    model = PatchTSTForClassification(config=config)
    
    # 测试输入
    past_values = torch.randn(2, 128, 1)
    
    # 获取所有 hidden states
    with torch.no_grad():
        outputs = model(
            past_values=past_values,
            output_hidden_states=True,
            output_attentions=True,
        )
    
    print(f"\n详细输出:")
    print(f"  prediction_logits: {outputs.prediction_logits.shape}")
    
    if outputs.hidden_states is not None:
        print(f"\n  hidden_states ({len(outputs.hidden_states)} 层):")
        for i, hidden in enumerate(outputs.hidden_states):
            print(f"    Layer {i}: {hidden.shape}")
    
    if outputs.attentions is not None:
        print(f"\n  attentions ({len(outputs.attentions)} 层):")
        for i, attn in enumerate(outputs.attentions):
            print(f"    Layer {i}: {attn.shape}")
    
    return outputs


def explore_backbone_only():
    """探索 PatchTSTModel（仅 backbone，不含分类头）"""
    print("\n" + "=" * 80)
    print("探索 PatchTSTModel (Backbone only)")
    print("=" * 80)
    
    config = PatchTSTConfig(
        num_input_channels=1,
        context_length=128,
        patch_length=16,
        stride=8,
        d_model=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        use_cls_token=True,
    )
    
    # PatchTSTModel 不含分类头
    backbone = PatchTSTModel(config=config)
    
    print(f"✅ Backbone 创建成功")
    print(f"\nBackbone 结构:")
    for name, module in backbone.named_children():
        print(f"  - {name}: {module.__class__.__name__}")
    
    # 测试
    past_values = torch.randn(2, 128, 1)
    
    with torch.no_grad():
        outputs = backbone(past_values=past_values)
    
    print(f"\nBackbone 输出:")
    print(f"  last_hidden_state: {outputs.last_hidden_state.shape}")
    
    # 分析序列长度
    seq_len = outputs.last_hidden_state.shape[1]
    num_patches = (128 - 16) // 8 + 1
    
    print(f"\n序列长度分析:")
    print(f"  输出序列长度: {seq_len}")
    print(f"  预期 patch 数: {num_patches}")
    if config.use_cls_token:
        print(f"  = {num_patches} patches + 1 CLS token")
    
    return backbone, outputs


def test_with_different_lengths():
    """测试不同长度的输入"""
    print("\n" + "=" * 80)
    print("测试不同长度的输入")
    print("=" * 80)
    
    config = PatchTSTConfig(
        num_input_channels=1,
        num_targets=5,
        context_length=256,  # 更长的上下文
        patch_length=16,
        stride=8,
        d_model=64,
        use_cls_token=True,
    )
    
    model = PatchTSTForClassification(config=config)
    
    test_cases = [
        (64, 1),
        (128, 1),
        (256, 1),
    ]
    
    for length, channels in test_cases:
        past_values = torch.randn(2, length, channels)
        
        try:
            with torch.no_grad():
                outputs = model(past_values=past_values)
            
            num_patches = (length - 16) // 8 + 1
            print(f"\n输入长度={length}:")
            print(f"  预期 patches: {num_patches}")
            print(f"  输出 logits: {outputs.prediction_logits.shape}")
            print(f"  ✅ 成功")
        except Exception as e:
            print(f"\n输入长度={length}:")
            print(f"  ❌ 失败: {e}")


def main():
    print("\n" + "🔍" * 40)
    print("PatchTST 架构深度探索")
    print("🔍" * 40)
    
    try:
        # 1. 基本架构探索
        model, config = explore_patchtst_architecture()
        
        # 2. 中间层输出
        outputs = explore_intermediate_outputs()
        
        # 3. Backbone 探索
        backbone, backbone_outputs = explore_backbone_only()
        
        # 4. 不同长度测试
        test_with_different_lengths()
        
        # ========== 总结 ==========
        print("\n" + "=" * 80)
        print("总结")
        print("=" * 80)
        
        print(f"\n**use_cls_token=True 时的前向传播流程**:")
        print(f"")
        print(f"1. 输入: past_values [batch_size, context_length, num_channels]")
        print(f"")
        print(f"2. Patching:")
        print(f"   - 将时间序列切分为 patches")
        print(f"   - num_patches = (context_length - patch_length) / stride + 1")
        print(f"   - 每个 patch 嵌入到 d_model 维")
        print(f"")
        print(f"3. CLS Token 添加:")
        print(f"   - 在 patch embeddings 前添加一个可学习的 CLS token")
        print(f"   - 序列变为: [CLS] + [Patch_1, Patch_2, ..., Patch_N]")
        print(f"   - 序列长度: num_patches + 1")
        print(f"")
        print(f"4. Transformer Encoder:")
        print(f"   - 多层 self-attention")
        print(f"   - 输出: [batch_size, num_patches+1, d_model]")
        print(f"")
        print(f"5. 分类 (use_cls_token=True):")
        print(f"   - 提取 CLS token 的输出: output[:, 0, :]")
        print(f"   - 通过分类头: Linear(d_model -> num_targets)")
        print(f"   - 输出: [batch_size, num_targets]")
        print(f"")
        print(f"6. 分类 (use_cls_token=False):")
        print(f"   - 对所有 patch 输出取平均: output.mean(dim=1)")
        print(f"   - 通过分类头")
        print(f"   - 输出: [batch_size, num_targets]")
        
        print("\n" + "=" * 80)
        print("✅ 探索完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
