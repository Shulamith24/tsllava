#!/usr/bin/env python3
"""
测试 PatchTST + Transformer 聚合头模型

验证：
1. 模型创建成功
2. 前向传播正常
3. 不同聚合头配置可用
4. backbone 冻结功能正常
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch


def test_model_creation():
    """测试模型创建"""
    print("=" * 60)
    print("测试 1: 模型创建")
    print("=" * 60)
    
    try:
        from opentslm.model.llm.PatchTSTWithAggregator import PatchTSTWithAggregator
        
        model = PatchTSTWithAggregator(
            num_classes=5,
            context_length=128,
            patch_length=16,
            stride=8,
            d_model=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            aggregator_layers=1,
            device="cpu",  # 测试用 CPU
        )
        
        print("✅ 模型创建成功")
        print(f"   配置: {model.get_config()}")
        return True, model
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)
    
    try:
        batch_size = 4
        context_length = 128
        
        past_values = torch.randn(batch_size, context_length, 1)
        labels = torch.randint(0, 5, (batch_size,))
        
        print(f"   输入形状: {past_values.shape}")
        print(f"   标签形状: {labels.shape}")
        
        model.eval()
        with torch.no_grad():
            outputs = model(past_values=past_values, labels=labels)
        
        print("✅ 前向传播成功")
        print(f"   Loss: {outputs['loss'].item():.4f}")
        print(f"   Logits shape: {outputs['logits'].shape}")
        print(f"   ANS hidden shape: {outputs['ans_hidden'].shape}")
        
        # 测试预测
        predictions = model.predict(past_values)
        print(f"   Predictions: {predictions.tolist()}")
        
        return True
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_aggregator_configs():
    """测试不同聚合头配置"""
    print("\n" + "=" * 60)
    print("测试 3: 不同聚合头配置")
    print("=" * 60)
    
    from opentslm.model.llm.PatchTSTWithAggregator import PatchTSTWithAggregator
    
    configs = [
        {"aggregator_layers": 1, "aggregator_hidden_size": None},  # 默认
        {"aggregator_layers": 2, "aggregator_hidden_size": None},  # 2层
        {"aggregator_layers": 1, "aggregator_hidden_size": 128},   # 不同维度
        {"aggregator_layers": 3, "aggregator_hidden_size": 256},   # 3层+大维度
    ]
    
    past_values = torch.randn(2, 128, 1)
    
    for i, config in enumerate(configs):
        try:
            model = PatchTSTWithAggregator(
                num_classes=5,
                context_length=128,
                patch_length=16,
                stride=8,
                d_model=64,
                num_hidden_layers=2,
                device="cpu",
                **config,
            )
            
            with torch.no_grad():
                outputs = model(past_values=past_values)
            
            print(f"   配置 {i+1}: {config}")
            print(f"      ✅ Logits shape: {outputs['logits'].shape}")
            print(f"      参数量: {model.count_parameters():,}")
        
        except Exception as e:
            print(f"   配置 {i+1}: {config}")
            print(f"      ❌ 失败: {e}")
            return False
    
    print("✅ 所有配置测试通过")
    return True


def test_freeze_backbone():
    """测试 backbone 冻结功能"""
    print("\n" + "=" * 60)
    print("测试 4: Backbone 冻结")
    print("=" * 60)
    
    try:
        from opentslm.model.llm.PatchTSTWithAggregator import PatchTSTWithAggregator
        
        model = PatchTSTWithAggregator(
            num_classes=5,
            context_length=128,
            patch_length=16,
            stride=8,
            d_model=64,
            num_hidden_layers=2,
            aggregator_layers=1,
            device="cpu",
        )
        
        # 测试冻结前
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   冻结前可训练参数: {trainable_before:,}")
        
        # 冻结
        model.freeze_backbone()
        
        # 测试冻结后
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   冻结后可训练参数: {trainable_after:,}")
        
        # 验证冻结成功
        backbone_params = sum(p.numel() for p in model.backbone.parameters())
        reduced = trainable_before - trainable_after
        
        if reduced == backbone_params:
            print("✅ Backbone 冻结成功")
        else:
            print(f"⚠️ 冻结参数不匹配: 减少了 {reduced:,}, backbone 有 {backbone_params:,}")
        
        # 测试解冻
        model.unfreeze_backbone()
        trainable_unfreeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   解冻后可训练参数: {trainable_unfreeze:,}")
        
        if trainable_unfreeze == trainable_before:
            print("✅ Backbone 解冻成功")
            return True
        else:
            print("❌ 解冻后参数不匹配")
            return False
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🧪" * 30)
    print("PatchTST + Aggregator 模型测试")
    print("🧪" * 30 + "\n")
    
    # 测试 1: 模型创建
    success, model = test_model_creation()
    if not success:
        return 1
    
    # 测试 2: 前向传播
    success = test_forward_pass(model)
    if not success:
        return 1
    
    # 测试 3: 不同配置
    success = test_aggregator_configs()
    if not success:
        return 1
    
    # 测试 4: backbone 冻结
    success = test_freeze_backbone()
    if not success:
        return 1
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
    print("\n下一步: 运行训练")
    print("  python scripts/train_patchtst_aggregator_ucr.py \\")
    print("      --dataset ECG200 \\")
    print("      --aggregator_layers 1 \\")
    print("      --epochs 20")
    
    return 0


if __name__ == "__main__":
    exit(main())
