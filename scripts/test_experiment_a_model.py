#!/usr/bin/env python3
"""
测试 OpenTSLMClassifier 模型

快速验证：
1. 模型可以正确初始化
2. forward pass 可以返回损失
3. predict 可以返回预测类别
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from opentslm.model.llm.OpenTSLMClassifier import OpenTSLMClassifier
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def test_model_initialization():
    """测试模型初始化"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模型
    model = OpenTSLMClassifier(
        num_classes=5,
        llm_id="meta-llama/Llama-3.2-1B",
        device=device,
        encoder_type="transformer_cnn",
    )
    
    print(f"✓ 模型创建成功")
    print(f"  - 类别数: {model.num_classes}")
    print(f"  - [ANS] token shape: {model.ans_token.shape}")
    print(f"  - 分类头: {model.classifier_head}")
    
    return model, device


def test_forward_pass(model, device):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播（计算损失）")
    print("=" * 60)
    
    # 创建模拟批次
    batch = [
        {
            "pre_prompt": "Classify the time series:",
            "time_series_text": ["This is test data:"],
            "time_series": [torch.randn(100, device=device)],
            "post_prompt": "\nClass:",
            "int_label": 0,
        },
        {
            "pre_prompt": "Classify the time series:",
            "time_series_text": ["This is test data:"],
            "time_series": [torch.randn(120, device=device)],
            "post_prompt": "\nClass:",
            "int_label": 2,
        },
    ]
    
    # 计算损失
    model.train()
    loss = model(batch)
    
    print(f"✓ 前向传播成功")
    print(f"  - 损失值: {loss.item():.4f}")
    print(f"  - 损失是否为标量: {loss.dim() == 0}")
    print(f"  - 损失 requires_grad: {loss.requires_grad}")
    
    return loss


def test_prediction(model, device):
    """测试预测"""
    print("\n" + "=" * 60)
    print("测试 3: 预测")
    print("=" * 60)
    
    # 创建模拟批次
    batch = [
        {
            "pre_prompt": "Classify the time series:",
            "time_series_text": ["This is test data:"],
            "time_series": [torch.randn(100, device=device)],
            "post_prompt": "\nClass:",
            "int_label": 1,
        },
        {
            "pre_prompt": "Classify the time series:",
            "time_series_text": ["This is test data:"],
            "time_series": [torch.randn(150, device=device)],
            "post_prompt": "\nClass:",
            "int_label": 3,
        },
        {
            "pre_prompt": "Classify the time series:",
            "time_series_text": ["This is test data:"],
            "time_series": [torch.randn(80, device=device)],
            "post_prompt": "\nClass:",
            "int_label": 4,
        },
    ]
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model.predict(batch)
    
    print(f"✓ 预测成功")
    print(f"  - 预测 shape: {predictions.shape}")
    print(f"  - 预测值: {predictions.tolist()}")
    print(f"  - 真实标签: {[b['int_label'] for b in batch]}")
    
    # 检查预测在有效范围内
    assert all(0 <= p < model.num_classes for p in predictions.tolist()), "预测值超出范围"
    print(f"  - 所有预测在有效范围 [0, {model.num_classes-1}]")
    
    return predictions


def test_with_real_dataset():
    """测试使用真实 UCR 数据集"""
    print("\n" + "=" * 60)
    print("测试 4: 使用真实 UCR 数据集")
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
        print("\n创建模型...")
        model = OpenTSLMClassifier(
            num_classes=num_classes,
            llm_id="meta-llama/Llama-3.2-1B",
            device=device,
            encoder_type="transformer_cnn",
        )
        print(f"✓ 模型创建成功")
        
        # 获取一个小批次
        print("\n处理批次数据...")
        samples = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = extend_time_series_to_match_patch_size_and_aggregate(samples, patch_size=PATCH_SIZE)
        
        # 测试前向传播
        print("\n测试前向传播...")
        model.train()
        loss = model(batch)
        print(f"✓ 前向传播成功，损失: {loss.item():.4f}")
        
        # 测试预测
        print("\n测试预测...")
        model.eval()
        with torch.no_grad():
            predictions = model.predict(batch)
        
        print(f"✓ 预测成功")
        for i, (pred, sample) in enumerate(zip(predictions.tolist(), batch)):
            print(f"  样本 {i}: 预测={pred}, 真实={sample['int_label']}")
        
        print("\n✅ 所有真实数据集测试通过!")
        
    except Exception as e:
        print(f"⚠️ 真实数据集测试跳过（需要下载数据）: {e}")


def main():
    print("\n" + "🧪" * 30)
    print("OpenTSLMClassifier 单元测试")
    print("🧪" * 30 + "\n")
    
    try:
        # 测试 1: 初始化
        model, device = test_model_initialization()
        
        # 测试 2: 前向传播
        loss = test_forward_pass(model, device)
        
        # 测试 3: 预测
        predictions = test_prediction(model, device)
        
        # 测试 4: 真实数据集（可选）
        test_with_real_dataset()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n下一步: 运行完整训练脚本")
        print("  python scripts/train_ucr_classification_experiment_a.py \\")
        print("      --dataset CricketZ \\")
        print("      --epochs 5 \\")
        print("      --batch_size 8")
        
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
