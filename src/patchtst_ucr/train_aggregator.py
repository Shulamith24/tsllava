#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + Transformer 聚合头在 UCR 数据集上的分类

使用 PatchTST backbone 提取 patch 特征，然后通过 Transformer 聚合头进行分类
独立模块，不依赖 opentslm 包

使用方法：
    python -m patchtst_ucr.train_aggregator --dataset Adiac --epochs 50

    # 自定义聚合头配置
    python -m patchtst_ucr.train_aggregator --dataset Adiac --aggregator_layers 2

    # 冻结 backbone
    python -m patchtst_ucr.train_aggregator --dataset Adiac --freeze_backbone
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

# 添加 src 目录到路径（支持直接运行）
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patchtst_ucr.model import PatchTSTWithAggregator
from patchtst_ucr.ucr_dataset import UCRDatasetForPatchTST, get_dataset_info


def parse_args():
    parser = argparse.ArgumentParser(description="PatchTST + Aggregator UCR 分类")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="Adiac", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # PatchTST 模型配置
    parser.add_argument("--context_length", type=int, default=None, 
                       help="上下文长度（None则自动设置为数据集最大长度）")
    parser.add_argument("--patch_length", type=int, default=16, help="Patch 长度")
    parser.add_argument("--stride", type=int, default=8, help="Patch 步长")
    parser.add_argument("--d_model", type=int, default=128, help="PatchTST 模型维度")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="PatchTST Attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=3, help="PatchTST Transformer 层数")
    parser.add_argument("--ffn_dim", type=int, default=512, help="PatchTST FFN 维度")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    
    # 聚合头配置（核心参数）
    parser.add_argument("--aggregator_layers", type=int, default=1, help="聚合头 Transformer 层数")
    parser.add_argument("--aggregator_hidden_size", type=int, default=None, 
                       help="聚合头 hidden size（None则与d_model相同）")
    parser.add_argument("--aggregator_num_heads", type=int, default=8, help="聚合头 attention heads")
    parser.add_argument("--aggregator_ffn_dim", type=int, default=None, 
                       help="聚合头 FFN 维度（None则自动计算）")
    
    # 冻结选项
    parser.add_argument("--freeze_backbone", action="store_true", help="冻结 PatchTST backbone")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/patchtst_aggregator", help="结果保存目录")
    
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


def prepare_batch(
    batch: List[Dict],
    context_length: int,
    device: str,
):
    """
    将 UCR 批次转换为模型输入格式
    
    Returns:
        past_values: [B, context_length, 1]
        labels: [B]
    """
    past_values_list = []
    labels = []
    
    for sample in batch:
        ts = sample["time_series"][0]
        
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        
        if len(ts) < context_length:
            padded = torch.zeros(context_length, device=device)
            padded[:len(ts)] = ts.to(device)
        else:
            padded = ts[:context_length].to(device)
        
        past_values_list.append(padded.unsqueeze(-1))
        labels.append(sample["int_label"])
    
    past_values = torch.stack(past_values_list, dim=0)
    labels = torch.tensor(labels, device=device, dtype=torch.long)
    
    return past_values, labels


def create_data_loaders(args, num_classes: int, context_length: int):
    """创建数据加载器"""
    train_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="train",
        raw_data_path=args.data_path,
    )
    
    val_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="validation",
        raw_data_path=args.data_path,
    )
    
    test_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="test",
        raw_data_path=args.data_path,
    )
    
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
        past_values, labels = prepare_batch(batch, context_length, device)
        
        outputs = model(past_values=past_values, labels=labels)
        loss = outputs["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        
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
        past_values, labels = prepare_batch(batch, context_length, device)
        
        outputs = model(past_values=past_values, labels=labels)
        
        total_loss += outputs["loss"].item()
        num_batches += 1
        
        predictions = torch.argmax(outputs["logits"], dim=-1)
        
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
    print("PatchTST + Transformer 聚合头 UCR 分类")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"聚合头层数: {args.aggregator_layers}")
    print(f"冻结 backbone: {args.freeze_backbone}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n使用设备: {device}")
    
    print("\n📂 分析数据集...")
    num_classes, max_length = get_dataset_info(args.dataset, args.data_path)
    
    if args.context_length is None:
        context_length = ((max_length - 1) // args.patch_length + 1) * args.patch_length
    else:
        context_length = args.context_length
    
    print(f"   类别数: {num_classes}")
    print(f"   最大长度: {max_length}")
    print(f"   Context length: {context_length}")
    
    num_patches = (context_length - args.patch_length) // args.stride + 1
    print(f"   预期 patch 数: {num_patches}")
    
    # 创建保存目录
    agg_str = f"L{args.aggregator_layers}"
    if args.aggregator_hidden_size:
        agg_str += f"_H{args.aggregator_hidden_size}"
    if args.freeze_backbone:
        agg_str += "_frozen"
    
    save_dir = os.path.join(
        args.save_dir, 
        args.dataset, 
        f"P{args.patch_length}_S{args.stride}_{agg_str}"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n🔧 创建模型...")
    model = PatchTSTWithAggregator(
        num_classes=num_classes,
        context_length=context_length,
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        aggregator_layers=args.aggregator_layers,
        aggregator_hidden_size=args.aggregator_hidden_size,
        aggregator_num_heads=args.aggregator_num_heads,
        aggregator_ffn_dim=args.aggregator_ffn_dim,
        device=device,
    ).to(device)
    
    if args.freeze_backbone:
        model.freeze_backbone()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n📂 加载数据...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args, num_classes, context_length
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    print("\n⚙️  创建优化器...")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    
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
                
                print("   Sample predictions (first 5):")
                for i in range(min(5, len(val_results["predictions"]))):
                    pred = val_results["predictions"][i]
                    label = val_results["labels"][i]
                    print(f"     Pred: {pred} | Label: {label} | {'✓' if pred == label else '✗'}")
                
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
                        "config": model.get_config(),
                        "args": vars(args),
                    }
                    torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
                    print(f"   💾 保存最佳模型")
                else:
                    patience_counter += 1
                    print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
                
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
            
            if patience_counter >= args.early_stop:
                print(f"\n⏹️  早停! 验证准确率 {args.early_stop} 轮未改进")
                break
        
        print("\n" + "=" * 60)
        print("📋 最终测试评估...")
        
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
        
        final_results = {
            "dataset": args.dataset,
            "num_classes": num_classes,
            "context_length": context_length,
            "aggregator_layers": args.aggregator_layers,
            "aggregator_hidden_size": args.aggregator_hidden_size or args.d_model,
            "freeze_backbone": args.freeze_backbone,
            "total_params": model.count_parameters(),
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
