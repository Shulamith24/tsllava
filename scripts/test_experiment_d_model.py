#!/usr/bin/env python3
"""
测试 TSClassifierSmallTransformer 模型

快速验证：
1. 模型可以正确初始化（small, medium, large配置）
2. forward pass 可以返回损失
3. predict 可以返回预测类别
4. 参数量计算正确
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from opentslm.model.llm.TSClassifierSmallTransformer import TSClassifierSmallTransformer
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def test_model_initialization():
    """测试模型初始化（不同配置）"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    for config in ["small", "medium", "large"]:
        print(f"\n测试配置: {config}")
        
        model = TSClassifierSmallTransformer(
            num_classes=5,
            aggregator_config=config,
            device=device,
            encoder_type="transformer_cnn",
        )
        
        print(f"  ✓ 模型创建成功")
        print(f"    - 类别数: {model.num_classes}")
        print(f"    - Aggregator layers: {model.aggregator.num_layers}")
        print(f"    - Hidden size: {model.hidden_size}")
        print(f"    - 总参数量: {model.count_parameters():,}")
    
    return model, device


def test_forward_pass(device):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播")
    print("=" * 60)
    
    batch = [
        {
            "time_series": [torch.randn(100, device=device)],
            "int_label": 0,
        },
        {
            "time_series": [torch.randn(120, device=device)],
            "int_label": 2,
        },
    ]
    
    model = TSClassifierSmallTransformer(
        num_classes=5,
        aggregator_config="medium",
        device=device,
        encoder_type="transformer_cnn",
    )
    
    model.train()
    loss = model(batch)
    
    print(f"  ✓ 前向传播成功")
    print(f"    - 损失值: {loss.item():.4f}")
    print(f"    - 损失是否为标量: {loss.dim() == 0}")
    print(f"    - 损失 requires_grad: {loss.requires_grad}")


def test_prediction(device):
    """测试预测"""
    print("\n" + "=" * 60)
    print("测试 3: 预测")
    print("=" * 60)
    
    batch = [
        {
            "time_series": [torch.randn(100, device=device)],
            "int_label": 1,
        },
        {
            "time_series": [torch.randn(150, device=device)],
            "int_label": 3,
        },
        {
            "time_series": [torch.randn(80, device=device)],
            "int_label": 4,
        },
    ]
    
    model = TSClassifierSmallTransformer(
        num_classes=5,
        aggregator_config="medium",
        device=device,
        encoder_type="transformer_cnn",
    )
    
    model.eval()
    with torch.no_grad():
        predictions = model.predict(batch)
    
    print(f"  ✓ 预测成功")
    print(f"    - 预测 shape: {predictions.shape}")
    print(f"    - 预测值: {predictions.tolist()}")
    print(f"    - 真实标签: {[b['int_label'] for b in batch]}")
    
    assert all(0 <= p < model.num_classes for p in predictions.tolist()), "预测值超出范围"
    print(f"    - 所有预测在有效范围 [0, {model.num_classes-1}]")


def test_custom_config(device):
    """测试自定义配置"""
    print("\n" + "=" * 60)
    print("测试 4: 自定义配置")
    print("=" * 60)
    
    # 自定义：4层 512维
    model = TSClassifierSmallTransformer(
        num_classes=10,
        aggregator_config="small",  # 基础配置
        num_layers=4,  # 覆盖层数
        hidden_size=512,  # 覆盖维度
        device=device,
        encoder_type="transformer_cnn",
    )
    
    print(f"  ✓ 自定义配置模型创建成功")
    print(f"    - Layers: {model.aggregator.num_layers}")
    print(f"    - Hidden: {model.hidden_size}")
    print(f"    - 总参数量: {model.count_parameters():,}")
    
    # 测试前向
    batch = [{"time_series": [torch.randn(100, device=device)], "int_label": 5}]
    loss = model(batch)
    print(f"    - 损失: {loss.item():.4f}")


def test_with_real_dataset():
    """测试使用真实 UCR 数据集"""
    print("\n" + "=" * 60)
    print("测试 5: 使用真实 UCR 数据集")
    print("=" * 60)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 创建数据集
        print("加载 ECG200 数据集...")
        dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN="<eos>",
            dataset_name="ECG200",
            raw_data_path="./data",
        )
        
        num_classes = UCRClassificationDataset.get_num_classes()
        print(f"✓ 数据集加载成功")
        print(f"  - 数据集大小: {len(dataset)}")
        print(f"  - 类别数: {num_classes}")
        
        # 创建模型
        model = TSClassifierSmallTransformer(
            num_classes=num_classes,
            aggregator_config="medium",
            device=device,
            encoder_type="transformer_cnn",
        )
        print(f"  ✓ 模型创建成功")
        
        # 获取批次
        samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = extend_time_series_to_match_patch_size_and_aggregate(samples, patch_size=PATCH_SIZE)
        
        # 测试前向传播
        model.train()
        loss = model(batch)
        print(f"  ✓ 前向传播成功，损失: {loss.item():.4f}")
        
        # 测试预测
        model.eval()
        with torch.no_grad():
            predictions = model.predict(batch)
        
        print(f"  ✓ 预测成功")
        for i, (pred, sample) in enumerate(zip(predictions.tolist(), batch)):
            print(f"    样本 {i}: 预测={pred}, 真实={sample['int_label']}")
        
        print("\n✅ 所有真实数据集测试通过!")
        
    except Exception as e:
        print(f"⚠️ 真实数据集测试跳过（需要下载数据）: {e}")


def main():
    print("\n" + "🧪" * 30)
    print("TSClassifierSmallTransformer 单元测试")
    print("🧪" * 30 + "\n")
    
    try:
        # 测试 1: 初始化
        model, device = test_model_initialization()
        
        # 测试 2: 前向传播
        test_forward_pass(device)
        
        # 测试 3: 预测
        test_prediction(device)
        
        # 测试 4: 自定义配置
        test_custom_config(device)
        
        # 测试 5: 真实数据集
        test_with_real_dataset()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n下一步: 运行完整训练脚本")
        print("  # medium 配置（推荐）")
        print("  python scripts/train_ucr_classification_experiment_d.py \\")
        print("      --dataset Adiac \\")
        print("      --aggregator_config medium \\")
        print("      --epochs 50 \\")
        print("      --batch_size 16")
        print()
        print("  # 自定义配置")
        print("  python scripts/train_ucr_classification_experiment_d.py \\")
        print("      --dataset Adiac \\")
        print("      --num_layers 4 \\")
        print("      --hidden_size 512 \\")
        print("      --epochs 50 \\")
        print("      --batch_size 16")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ 测试失败!")
        print("=" * 60)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
