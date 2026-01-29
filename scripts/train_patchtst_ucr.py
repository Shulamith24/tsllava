#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
PatchTST 在 UCR 数据集上的分类

使用 HuggingFace 的 PatchTSTForClassification 进行时间序列分类

使用方法：
    python scripts/train_patchtst_ucr.py \
        --dataset Adiac \
        --epochs 50 \
        --batch_size 32 \
        --lr 1e-3
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PatchTSTConfig, PatchTSTForClassification
from transformers import get_cosine_schedule_with_warmup

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset


def parse_args():
    parser = argparse.ArgumentParser(description="PatchTST UCR 分类")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="Adiac", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # PatchTST 模型配置
    parser.add_argument("--context_length", type=int, default=None, 
                       help="上下文长度（None则自动设置为数据集最大长度）")
    parser.add_argument("--patch_length", type=int, default=16, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=8, help="Patch 步长")
    parser.add_argument("--d_model", type=int, default=128, help="模型维度")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="Transformer 层数")
    parser.add_argument("--ffn_dim", type=int, default=512, help="FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--use_cls_token", action="store_true", default=True, help="使用 CLS token")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/patchtst", help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--eval_every", type=int, default=5, help="每N轮评估一次")
    parser.add_argument("--early_stop", type=int, default=15, help="早停耐心值")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="评估批次大小")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def get_dataset_stats(dataset_name: str, data_path: str):
    """获取数据集统计信息"""
    temp_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name=dataset_name,
        raw_data_path=data_path,
    )
    
    num_classes = UCRClassificationDataset.get_num_classes()
    
    # 计算最大长度
    max_length = 0
    for sample in temp_dataset:
        for ts in sample["time_series"]:
            max_length = max(max_length, len(ts))
    
    return num_classes, max_length


def prepare_batch_for_patchtst(
    batch: List[Dict],
    context_length: int,
    device: str,
):
    """
    将 UCR 批次转换为 PatchTST 格式
    
    Args:
        batch: UCR 格式
        context_length: 固定上下文长度
        device: 设备
    
    Returns:
        past_values: [B, context_length, 1]
        labels: [B]
    """
    past_values_list = []
    labels = []
    
    for sample in batch:
        # 获取第一个时间序列
        ts = sample["time_series"][0]
        
        # 填充或截断到 context_length
        if len(ts) < context_length:
            # 零填充
            padded = torch.zeros(context_length, device=device)
            padded[:len(ts)] = ts.to(device)
        else:
            # 截断
            padded = ts[:context_length].to(device)
        
        past_values_list.append(padded.unsqueeze(-1))  # [L, 1]
        labels.append(sample["int_label"])
    
    past_values = torch.stack(past_values_list, dim=0)  # [B, L, 1]
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    
    return past_values, labels


def create_data_loaders(args, num_classes: int, context_length: int):
    """创建数据加载器"""
    # 创建数据集
    train_dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRClassificationDataset(
        split="validation",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRClassificationDataset(
        split="test",
        EOS_TOKEN="<eos>",
        dataset_name=args.dataset,
        raw_data_path=args.data_path,
    )
    
    # 简单的 collate（不做转换，在训练循环中转换）
    def collate_fn(batch):
        return batch
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer,
    scheduler,
    context_length: int,
    grad_clip: float,
    device: str,
    epoch: int,
    num_epochs: int,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        # 转换为 PatchTST 格式
        past_values, labels = prepare_batch_for_patchtst(batch, context_length, device)
        
        # 前向传播
        outputs = model(
            past_values=past_values,
            target_values=labels,  # PatchTST 接受 target_values 计算损失
        )
        
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    data_loader: DataLoader,
    context_length: int,
    device: str,
    desc: str = "Evaluating",
) -> Dict[str, Any]:
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc):
        # 转换为 PatchTST 格式
        past_values, labels = prepare_batch_for_patchtst(batch, context_length, device)
        
        # 前向传播
        outputs = model(
            past_values=past_values,
            target_values=labels,
        )
        
        total_loss += outputs.loss.item()
        num_batches += 1
        
        # 预测
        logits = outputs.prediction_logits  # [B, num_classes]
        predictions = torch.argmax(logits, dim=-1)  # [B]
        
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    avg_loss = total_loss / max(num_batches, 1)
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    accuracy = correct / len(all_labels) if all_labels else 0.0
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": all_predictions,
        "labels": all_labels,
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("PatchTST UCR 分类")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    # 获取数据集统计信息
    print("\n📂 分析数据集...")
    num_classes, max_length = get_dataset_stats(args.dataset, args.data_path)
    
    # 确定 context_length
    if args.context_length is None:
        # 向上取整到 patch_length 的倍数
        context_length = ((max_length - 1) // args.patch_length + 1) * args.patch_length
    else:
        context_length = args.context_length
    
    print(f"   类别数: {num_classes}")
    print(f"   最大长度: {max_length}")
    print(f"   Context length: {context_length}")
    
    # 计算 patch 数量
    num_patches = (context_length - args.patch_length) // args.stride + 1
    print(f"   预期 patch 数: {num_patches}")
    
    # 创建保存目录
    save_dir = os.path.join(
        args.save_dir, 
        args.dataset, 
        f"L{context_length}_P{args.patch_length}_S{args.stride}"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建模型
    print("\n🔧 创建 PatchTST 模型...")
    config = PatchTSTConfig(
        num_input_channels=1,  # UCR 单变量
        num_targets=num_classes,
        context_length=context_length,
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        use_cls_token=args.use_cls_token,
    )
    
    model = PatchTSTForClassification(config=config).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {total_params:,}")
    print(f"   d_model: {args.d_model}")
    print(f"   num_layers: {args.num_hidden_layers}")
    print(f"   use_cls_token: {args.use_cls_token}")
    
    # 创建数据加载器
    print("\n📂 加载数据...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args, num_classes, context_length
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 创建优化器
    print("\n⚙️  创建优化器...")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 创建学习率调度器
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_val_acc = 0.0
    patience_counter = 0
    loss_history = []
    
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                context_length, args.grad_clip, device,
                epoch, args.epochs
            )
            
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                print(f"\n📊 Epoch {epoch} 评估...")
                
                val_results = evaluate(
                    model, val_loader, context_length, device, "Validating"
                )
                val_loss = val_results["loss"]
                val_acc = val_results["accuracy"]
                
                print(f"   Train Loss: {train_loss:.4f}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val Accuracy: {val_acc:.4f}")
                
                # 显示样本预测
                print("   Sample predictions (first 5):")
                for i in range(min(5, len(val_results["predictions"]))):
                    pred = val_results["predictions"][i]
                    label = val_results["labels"][i]
                    print(f"     Pred: {pred} | Label: {label} | {'✓' if pred == label else '✗'}")
                
                # 保存最佳模型
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    
                    checkpoint = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "config": config.to_dict(),
                        "args": vars(args),
                    }
                    torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                    print(f"   💾 保存最佳模型")
                else:
                    patience_counter += 1
                    print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
                # 记录历史
                loss_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                })
                with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
                    json.dump(loss_history, f, indent=2)
            else:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # 早停
            if patience_counter >= args.early_stop:
                print(f"\n⏹️  早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        # 最终测试
        print("\n" + "=" * 60)
        print("📋 最终测试评估...")
        
        # 加载最佳模型
        best_ckpt = torch.load(
            os.path.join(save_dir, "best_model.pt"),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(best_ckpt["model_state"])
        
        test_results = evaluate(
            model, test_loader, context_length, device, "Testing"
        )
        
        print(f"\n✅ 测试结果:")
        print(f"   Test Loss: {test_results['loss']:.4f}")
        print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
        
        # 保存最终结果
        final_results = {
            "dataset": args.dataset,
            "num_classes": num_classes,
            "context_length": context_length,
            "total_params": total_params,
            "best_val_acc": best_val_acc,
            "test_loss": test_results["loss"],
            "test_accuracy": test_results["accuracy"],
            "epochs_trained": epoch,
        }
        
        with open(os.path.join(save_dir, "final_results.json"), "w") as f:
            json.dump(final_results, f, indent=2)
        
        with open(os.path.join(save_dir, "test_predictions.json"), "w") as f:
            json.dump({
                "predictions": test_results["predictions"],
                "labels": test_results["labels"],
            }, f, indent=2)
        
        print("=" * 60)
        print(f"结果保存到: {save_dir}")
        print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n⚠️  训练被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
