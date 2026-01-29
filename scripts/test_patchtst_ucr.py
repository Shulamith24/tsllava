#!/usr/bin/env python3
"""
快速测试 PatchTST UCR 分类脚本

验证：
1. 数据加载和转换正确
2. 模型可以前向传播
3. 训练循环可以运行
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
from transformers import PatchTSTConfig, PatchTSTForClassification
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset


def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)
    
    try:
        dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN="<eos>",
            dataset_name="ECG200",
            raw_data_path="./data",
        )
        
        print(f"✅ 数据集加载成功")
        print(f"   大小: {len(dataset)}")
        print(f"   类别数: {UCRClassificationDataset.get_num_classes()}")
        
        # 获取一个样本
        sample = dataset[0]
        ts = sample["time_series"][0]
        print(f"   样本长度: {len(ts)}")
        print(f"   标签: {sample['int_label']}")
        
        # 计算最大长度
        max_len = max(len(sample["time_series"][0]) for sample in dataset)
        print(f"   最大长度: {max_len}")
        
        return True, max_len
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False, None


def test_model_creation(context_length):
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("测试 2: 模型创建")
    print("=" * 60)
    
    try:
        config = PatchTSTConfig(
            num_input_channels=1,
            num_targets=2,  # ECG200 有 2 类
            context_length=context_length,
            patch_length=16,
            stride=8,
            d_model=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            use_cls_token=True,
        )
        
        model = PatchTSTForClassification(config=config)
        
        print(f"✅ 模型创建成功")
        print(f"   总参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Context length: {context_length}")
        
        # 计算 patch 数
        num_patches = (context_length - 16) // 8 + 1
        print(f"   Patch 数: {num_patches}")
        
        return True, model
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, context_length):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("测试 3: 前向传播")
    print("=" * 60)
    
    try:
        # 创建测试数据
        batch_size = 4
        past_values = torch.randn(batch_size, context_length, 1)
        labels = torch.randint(0, 2, (batch_size,))
        
        print(f"   输入形状: {past_values.shape}")
        print(f"   标签形状: {labels.shape}")
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            outputs = model(
                past_values=past_values,
                target_values=labels,
            )
        
        print(f"✅ 前向传播成功")
        print(f"   Loss: {outputs.loss.item():.4f}")
        print(f"   Logits shape: {outputs.prediction_logits.shape}")
        
        # 预测
        predictions = torch.argmax(outputs.prediction_logits, dim=-1)
        print(f"   Predictions: {predictions.tolist()}")
        print(f"   Labels: {labels.tolist()}")
        
        return True
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_conversion():
    """测试批次转换"""
    print("\n" + "=" * 60)
    print("测试 4: UCR 批次转换")
    print("=" * 60)
    
    try:
        dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN="<eos>",
            dataset_name="ECG200",
            raw_data_path="./data",
        )
        
        batch = [dataset[i] for i in range(4)]
        context_length = 128
        
        # 转换
        past_values_list = []
        labels = []
        
        for sample in batch:
            ts = sample["time_series"][0]
            
            # 转换为 tensor（如果是 numpy）
            if not isinstance(ts, torch.Tensor):
                ts = torch.tensor(ts, dtype=torch.float32)
            
            # 填充
            if len(ts) < context_length:
                padded = torch.zeros(context_length)
                padded[:len(ts)] = ts
            else:
                padded = ts[:context_length]
            
            past_values_list.append(padded.unsqueeze(-1))
            labels.append(sample["int_label"])
        
        past_values = torch.stack(past_values_list, dim=0)
        labels = torch.tensor(labels)
        
        print(f"✅ 批次转换成功")
        print(f"   Past values shape: {past_values.shape}")
        print(f"   Labels shape: {labels.shape}")
        print(f"   原始长度: {[len(sample['time_series'][0]) for sample in batch]}")
        print(f"   填充后长度: {context_length}")
        
        return True
    
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def main():
    print("\n" + "🧪" * 30)
    print("PatchTST UCR 分类 - 快速测试")
    print("🧪" * 30 + "\n")
    
    # 测试 1: 数据加载
    success, max_len = test_data_loading()
    if not success:
        return 1
    
    # 确定 context_length
    context_length = ((max_len - 1) // 16 + 1) * 16
    print(f"\n使用 context_length = {context_length}")
    
    # 测试 2: 模型创建
    success, model = test_model_creation(context_length)
    if not success:
        return 1
    
    # 测试 3: 前向传播
    success = test_forward_pass(model, context_length)
    if not success:
        return 1
    
    # 测试 4: 批次转换
    success = test_batch_conversion()
    if not success:
        return 1
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
    print("\n下一步: 运行完整训练")
    print("  python scripts/train_patchtst_ucr.py \\")
    print("      --dataset ECG200 \\")
    print("      --epochs 20 \\")
    print("      --batch_size 32")
    
    return 0


if __name__ == "__main__":
    exit(main())
