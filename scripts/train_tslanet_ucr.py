#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet单数据集分类训练脚本

用于在UCR数据集上训练TSLANet分类器，训练好的encoder用于ICL分类的相似样本检索。

使用方法：
    python scripts/train_tslanet_ucr.py \\
        --dataset ECG5000 \\
        --epochs 100 \\
        --batch_size 16 \\
        --lr 1e-3

训练流程：
1. 加载UCR数据集
2. 使用TSLANetEncoder + 分类头进行训练
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
    parser = argparse.ArgumentParser(description="TSLANet单数据集分类训练")

    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关
    parser.add_argument("--emb_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--depth", type=int, default=2, help="TSLANet层数")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch大小")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout比例")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="标签平滑")
    
    # 预训练阶段（可选）
    parser.add_argument("--pretrain", action="store_true", help="是否进行掩码预训练")
    parser.add_argument("--pretrain_epochs", type=int, default=50, help="预训练轮数")
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="掩码比例")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/tslanet_ucr", help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例(从训练集划分)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TSLANetClassifier(nn.Module):
    """TSLANet分类器 = TSLANetEncoder + 分类头"""
    
    def __init__(
        self,
        encoder: TSLANetEncoder,
        num_classes: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.emb_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] 时间序列
        Returns:
            [B, num_classes] logits
        """
        # 编码
        features = self.encoder(x)  # [B, N, emb_dim]
        # 全局平均池化
        pooled = features.mean(dim=1)  # [B, emb_dim]
        # 分类
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_classes]
        return logits
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """获取全局embedding用于检索"""
        return self.encoder.get_embedding(x)


def create_data_loaders(args):
    """创建数据加载器"""
    ensure_ucr_data()
    
    # 加载数据
    train_df, test_df = load_ucr_dataset(args.dataset, raw_data_path=args.data_path)
    
    # 获取数据集信息
    all_labels = sorted(train_df["label"].unique().tolist())
    num_classes = len(all_labels)
    seq_len = train_df.shape[1] - 1  # 减去label列
    
    # 标签重映射到0-indexed
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
        val_df = test_df.copy()  # 如果训练集太小，用测试集作为验证集
    
    print(f"📊 Dataset: {args.dataset}")
    print(f"   Classes: {num_classes}")
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
    model: TSLANetClassifier,
    train_loader: DataLoader,
    optimizer,
    mask_ratio: float,
    epoch: int,
    num_epochs: int,
    device: str
) -> float:
    """预训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        features, _ = batch
        features = features.to(device)  # [B, L]
        
        # 掩码预训练
        preds, target, mask = model.encoder.pretrain_forward(features, mask_ratio=mask_ratio)
        
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


def train_one_epoch(
    model: TSLANetClassifier,
    train_loader: DataLoader,
    optimizer,
    criterion,
    epoch: int,
    num_epochs: int,
    device: str
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        features, labels = batch
        features = features.to(device)  # [B, L]
        labels = labels.to(device)  # [B]
        
        # 前向传播
        logits = model(features)  # [B, num_classes]
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: TSLANetClassifier,
    data_loader: DataLoader,
    criterion,
    device: str,
    desc: str = "Evaluating"
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(data_loader, desc=desc):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        num_batches += 1
        
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    # 计算指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()
    
    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TSLANet单数据集分类训练")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"预训练: {args.pretrain}")
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
    
    # 创建模型
    print("\n🔧 创建模型...")
    encoder = TSLANetEncoder(
        output_dim=args.emb_dim,
        dropout=args.dropout,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        max_seq_len=max(padded_seq_len, 512)  # 确保足够长
    )
    
    model = TSLANetClassifier(
        encoder=encoder,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)
    
    print(f"   Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 预训练阶段（可选）
    if args.pretrain:
        print("\n🔄 开始预训练阶段...")
        pretrain_optimizer = AdamW(model.encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        for epoch in range(1, args.pretrain_epochs + 1):
            pretrain_loss = pretrain_one_epoch(
                model, train_loader, pretrain_optimizer,
                args.mask_ratio, epoch, args.pretrain_epochs, device
            )
            print(f"Pretrain Epoch {epoch}: Loss = {pretrain_loss:.4f}")
        
        print("✅ 预训练完成")
    
    # 训练阶段
    print("\n🚀 开始分类训练...")
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    loss_history = []
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            epoch, args.epochs, device
        )
        
        # 验证
        val_results = evaluate(model, val_loader, criterion, device, "Validating")
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_results['loss']:.4f}, Val Acc = {val_results['accuracy']:.4f}")
        
        # 记录历史
        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_acc": val_results["accuracy"]
        })
        
        # 保存最佳模型
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            patience_counter = 0
            
            # 保存checkpoint
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "classifier_state": model.classifier.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "num_classes": num_classes,
                "seq_len": seq_len,
                "label_to_idx": label_to_idx,
                "config": vars(args)
            }
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pt"))
            print(f"💾 Saved best model (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ 早停! 验证准确率 {patience} 轮未改进")
                break
    
    # 保存训练历史
    with open(os.path.join(save_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f, indent=2)
    
    # 最终测试
    print("\n" + "=" * 60)
    print("📋 最终测试评估...")
    
    # 加载最佳模型
    best_ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device, weights_only=False)
    model.encoder.load_state_dict(best_ckpt["encoder_state"])
    model.classifier.load_state_dict(best_ckpt["classifier_state"])
    
    test_results = evaluate(model, test_loader, criterion, device, "Testing")
    
    print(f"\n✅ 测试结果:")
    print(f"   Test Loss: {test_results['loss']:.4f}")
    print(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    
    # 保存最终结果
    final_results = {
        "dataset": args.dataset,
        "best_val_acc": best_val_acc,
        "test_loss": test_results["loss"],
        "test_accuracy": test_results["accuracy"],
        "epochs_trained": epoch,
        "num_classes": num_classes,
        "seq_len": seq_len
    }
    
    with open(os.path.join(save_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("=" * 60)
    print(f"结果保存到: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
