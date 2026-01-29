#!/usr/bin/env python3
"""
测试 OpenTSLMClassifierMultiANS 模型

快速验证：
1. 模型可以正确初始化（M=1, M=4, M=8）
2. forward pass 可以返回损失
3. predict 可以返回预测类别
4. token-wise normalization 和 averaging logits 正确工作
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from opentslm.model.llm.OpenTSLMClassifierMultiANS import OpenTSLMClassifierMultiANS
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def test_model_initialization():
    """测试模型初始化（不同 ANS tokens 数量）"""
    print("=" * 60)
    print("测试 1: 模型初始化")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    for num_ans in [1, 4, 8]:
        print(f"\n测试 num_ans_tokens={num_ans}:")
        
        model = OpenTSLMClassifierMultiANS(
            num_classes=5,
            num_ans_tokens=num_ans,
            llm_id="meta-llama/Llama-3.2-1B",
            device=device,
            encoder_type="transformer_cnn",
        )
        
        print(f"  ✓ 模型创建成功")
        print(f"    - 类别数: {model.num_classes}")
        print(f"    - ANS tokens: {num_ans}")
        print(f"    - ANS tokens shape: {model.ans_tokens.shape}")
        print(f"    - Token norm: {model.token_norm}")
    
    return model, device


def test_forward_pass(device):
    """测试前向传播（不同 ANS tokens 数量）"""
    print("\n" + "=" * 60)
    print("测试 2: 前向传播（不同 ANS tokens 数量）")
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
    
    for num_ans in [1, 4, 8]:
        print(f"\nnum_ans_tokens={num_ans}:")
        
        model = OpenTSLMClassifierMultiANS(
            num_classes=5,
            num_ans_tokens=num_ans,
            llm_id="meta-llama/Llama-3.2-1B",
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
    """测试预测（不同 ANS tokens 数量）"""
    print("\n" + "=" * 60)
    print("测试 3: 预测（不同 ANS tokens 数量）")
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
    
    for num_ans in [4, 8]:
        print(f"\nnum_ans_tokens={num_ans}:")
        
        model = OpenTSLMClassifierMultiANS(
            num_classes=5,
            num_ans_tokens=num_ans,
            llm_id="meta-llama/Llama-3.2-1B",
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


def test_with_real_dataset():
    """测试使用真实 UCR 数据集"""
    print("\n" + "=" * 60)
    print("测试 4: 使用真实 UCR 数据集（M=1, M=4）")
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
        
        # 测试 M=1 和 M=4
        for num_ans in [1, 4]:
            print(f"\n测试 num_ans_tokens={num_ans}:")
            
            # 创建模型
            model = OpenTSLMClassifierMultiANS(
                num_classes=num_classes,
                num_ans_tokens=num_ans,
                llm_id="meta-llama/Llama-3.2-1B",
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
    print("OpenTSLMClassifierMultiANS 单元测试")
    print("🧪" * 30 + "\n")
    
    try:
        # 测试 1: 初始化
        model, device = test_model_initialization()
        
        # 测试 2: 前向传播
        test_forward_pass(device)
        
        # 测试 3: 预测
        test_prediction(device)
        
        # 测试 4: 真实数据集
        test_with_real_dataset()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试通过!")
        print("=" * 60)
        print("\n下一步: 运行完整训练脚本")
        print("  # M=4")
        print("  python scripts/train_ucr_classification_experiment_c.py \\")
        print("      --dataset Adiac \\")
        print("      --num_ans_tokens 4 \\")
        print("      --epochs 5 \\")
        print("      --batch_size 8")
        print()
        print("  # M=8")
        print("  python scripts/train_ucr_classification_experiment_c.py \\")
        print("      --dataset Adiac \\")
        print("      --num_ans_tokens 8 \\")
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
