#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet掩码预训练脚本

在UCR数据集上只进行自监督掩码预训练（不使用标签），用于ICL分类的相似样本检索。

与分类训练不同，掩码预训练只学习时间序列的表征，不涉及类别信息，
因此在ICL场景中不存在信息泄露问题。

使用方法：
    python scripts/pretrain_tslanet_ucr.py \\
        --dataset ECG5000 \\
        --epochs 100 \\
        --mask_ratio 0.4

训练流程：
1. 加载UCR数据集（只使用时间序列，忽略标签）
2. 使用掩码重建任务进行自监督预训练
3. 保存encoder checkpoint用于后续检索
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
from opentslm.time_series_datasets.ucr.ucr_loader import (
    load_ucr_dataset, 
    ensure_ucr_data,
    UCRDataset,
    collate_fn
)


def parse_args():
    parser = argparse.ArgumentParser(description="TSLANet掩码预训练（自监督）")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关
    parser.add_argument("--emb_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--depth", type=int, default=2, help="TSLANet层数")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch大小")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout比例")
    
    # 预训练相关
    parser.add_argument("--epochs", type=int, default=200, help="预训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="掩码比例")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/tslanet_pretrain", help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例(从训练集划分)")
    parser.add_argument("--patience", type=int, default=20, help="早停耐心值")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(args):
    """创建数据加载器"""
    ensure_ucr_data()
    
    # 加载数据
    train_df, test_df = load_ucr_dataset(args.dataset, raw_data_path=args.data_path)
    
    # 获取数据集信息
    all_labels = sorted(train_df["label"].unique().tolist())
    num_classes = len(all_labels)
    seq_len = train_df.shape[1] - 1  # 减去label列
    
    # 标签重映射到0-indexed (虽然预训练不需要标签，但保留以便后续使用)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    train_df["label"] = train_df["label"].map(label_to_idx)
    test_df["label"] = test_df["label"].map(label_to_idx)
    
    # 从训练集划分验证集
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    val_size = int(len(train_df) * args.val_ratio)
    
    if val_size > 0:
        val_df = train_df.iloc[:val_size]
        train_df = train_df.iloc[val_size:]
    else:
        val_df = test_df.copy()
    
    print(f"📊 Dataset: {args.dataset}")
    print(f"   Classes: {num_classes} (不用于预训练)")
    print(f"   Sequence length: {seq_len}")
    print(f"   Train samples: {len(train_df)}")
    print(f"   Val samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # 创建Dataset
    train_dataset = UCRDataset(train_df)
    val_dataset = UCRDataset(val_df)
    test_dataset = UCRDataset(test_df)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=len(train_dataset) > args.batch_size
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, num_classes, seq_len, label_to_idx


def pretrain_one_epoch(
    encoder: TSLANetEncoder,
    train_loader: DataLoader,
    optimizer,
    mask_ratio: float,
    epoch: int,
    num_epochs: int,
    device: str
) -> float:
    """预训练一个epoch"""
    encoder.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        features, _ = batch  # 忽略标签
        features = features.to(device)  # [B, L]
        
        # 掩码预训练
        preds, target, mask = encoder.pretrain_forward(features, mask_ratio=mask_ratio)
        
        # 计算掩码位置的MSE损失
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_pretrain(
    encoder: TSLANetEncoder,
    data_loader: DataLoader,
    mask_ratio: float,
    device: str,
    desc: str = "Evaluating"
) -> float:
    """评估预训练模型"""
    encoder.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(data_loader, desc=desc):
        features, _ = batch
        features = features.to(device)
        
        preds, target, mask = encoder.pretrain_forward(features, mask_ratio=mask_ratio)
        
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask.float()).sum() / mask.float().sum()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TSLANet掩码预训练（自监督）")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"掩码比例: {args.mask_ratio}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        print("⚠️ CUDA不可用，使用CPU")
        device = "cpu"
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 创建数据加载器
    print("\n📂 加载数据...")
    train_loader, val_loader, test_loader, num_classes, seq_len, label_to_idx = create_data_loaders(args)
    
    # 计算需要的patch数量和序列长度
    padded_seq_len = seq_len
    if seq_len % args.patch_size != 0:
        padded_seq_len = seq_len + (args.patch_size - seq_len % args.patch_size)
    max_seq_len = max(padded_seq_len, 512)
    
    # 创建模型
    print("\n🔧 创建TSLANet Encoder...")
    encoder = TSLANetEncoder(
        output_dim=args.emb_dim,
        dropout=args.dropout,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        max_seq_len=max_seq_len
    ).to(device)
    
    print(f"   Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # 创建优化器
    optimizer = AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 预训练
    print("\n🚀 开始掩码预训练...")
    best_val_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = pretrain_one_epoch(
            encoder, train_loader, optimizer,
            args.mask_ratio, epoch, args.epochs, device
        )
        
        # 验证
        val_loss = evaluate_pretrain(encoder, val_loader, args.mask_ratio, device, "Validating")
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 记录历史
        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存checkpoint (只保存encoder，不含分类头)
            checkpoint = {
                "encoder_state": encoder.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "num_classes": num_classes,
                "seq_len": seq_len,
                "max_seq_len": max_seq_len,
                "label_to_idx": label_to_idx,
                "config": vars(args),
                "pretrain_only": True  # 标记为纯预训练模型
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            print(f"💾 Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹️ 早停! 验证损失 {args.patience} 轮未改进")
                break
    
    # 保存训练历史
    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("📋 最终测试评估...")
    
    # 加载最佳模型
    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    encoder.load_state_dict(best_ckpt["encoder_state"])
    
    test_loss = evaluate_pretrain(encoder, test_loader, args.mask_ratio, device, "Testing")
    
    print(f"\n✅ 测试结果:")
    print(f"   Test Loss (重建误差): {test_loss:.4f}")
    
    # 保存最终结果
    final_results = {
        "dataset": args.dataset,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "epochs_trained": epoch,
        "num_classes": num_classes,
        "seq_len": seq_len,
        "mask_ratio": args.mask_ratio,
        "pretrain_only": True
    }
    
    with open(os.path.join(save_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("=" * 60)
    print(f"结果保存到: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
