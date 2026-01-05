#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet UEA多变量数据集预训练脚本 (修复版)

核心机制：Channel Independence (CI)
我们将所有变量(Channels)视为独立的单变量序列进行预训练。
输入形状变换: [Batch, Channel, Length] -> [Batch * Channel, Length]

使用方法：
    python scripts/pretrain_tslanet_uea.py --dataset Handwriting
    python scripts/pretrain_tslanet_uea.py --dataset_list src/opentslm/time_series_datasets/uea/uea_pretrain_datasets.txt
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import dataset
from tqdm.auto import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
dataset_list_path = (project_root / "data" / "Multivariate_ts" / "uea_datasets.txt").resolve()
uea_path = str(dataset_list_path.parent)

from opentslm import data
from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
# 复用加载器逻辑
from opentslm.time_series_datasets.uea.uea_pretrain_loader import (
    get_uea_pretrain_loader, 
    UEAPretrainDataset,
    collate_fn_pretrain,
)
from aeon.datasets import load_classification

def parse_args():
    parser = argparse.ArgumentParser(description="TSLANet UEA预训练")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, default=None, help="单个UEA数据集名称")
    parser.add_argument("--dataset_list", type=str, default=str(dataset_list_path), help="数据集列表文件")
    parser.add_argument("--save_path", type=str, default="pretrained/tslanet_uea.pt", help="保存路径")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    
    # 模型结构
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    
    # 系统
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例(仅当单数据集模式有效)")
    
    # 动态采样参数（解决OOM问题）
    parser.add_argument("--max_channels", type=int, default=32, help="最大通道数，超过则随机采样")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度，超过则随机裁剪")
    parser.add_argument("--skip_variable_length", action="store_true", help="跳过变长数据集")
    
    return parser.parse_args()

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, train_loader, optimizer, mask_ratio, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        # batch: [B, C, L]
        # Channel Independence: Flatten to [B*C, L]
        # 展平所有通道，视为独立的单变量样本
        B, C, L = batch.shape
        batch = batch.view(B * C, L).to(device)
        
        # 预训练前向传播
        # preds: [B*C, N, D], target: [B*C, N, D], mask: [B*C, N]
        preds, target, mask = model.pretrain_forward(batch, mask_ratio=mask_ratio)
        
        # 计算MSE损失 (只在掩码位置)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(num_batches, 1)

@torch.no_grad()
def validate(model, val_loader, mask_ratio, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        B, C, L = batch.shape
        batch = batch.view(B * C, L).to(device)
        
        preds, target, mask = model.pretrain_forward(batch, mask_ratio=mask_ratio)
        
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / max(num_batches, 1)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 60)
    print("TSLANet UEA多变量数据集预训练 (Channel Independence)")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"参数: {args}")
    
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = "cpu"
    device = args.device
    
    # 1. 准备数据 Loader
    if args.dataset:
        # 单数据集模式
        print(f"📂 Loading Single Dataset: {args.dataset}")
        X_train, _ = load_classification(args.dataset, split="train", extract_path=uea_path) # [N, C, L]
        # 注意：这里我们仅使用 train split 进行预训练，
        # 并从中划分出一部分作为 valid 监控 loss 变化
        
        val_size = int(len(X_train) * args.val_ratio)
        if val_size < 1: val_size = 1
        
        indices = np.random.permutation(len(X_train))
        X_val = X_train[indices[:val_size]]
        X_train = X_train[indices[val_size:]]
        
        print(f"   Train samples: {len(X_train)}")
        print(f"   Val samples:   {len(X_val)}")
        print(f"   Channels:      {X_train.shape[1]}")
        
        train_dataset = UEAPretrainDataset(X_train, max_channels=args.max_channels, max_length=args.max_length)
        val_dataset = UEAPretrainDataset(X_val, max_channels=args.max_channels, max_length=args.max_length)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, 
            collate_fn=lambda x: collate_fn_pretrain(x, args.patch_size)
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, 
            collate_fn=lambda x: collate_fn_pretrain(x, args.patch_size)
        )
        
    elif args.dataset_list:
        # 多数据集模式
        print(f"📂 Loading Multi Datasets list: {args.dataset_list}")
        # 使用 split='train'。如果是大规模预训练，通常不专门划分 valid，
        # 或者直接用 loader 的一部分数据。
        # 为简化，这里我们将 train_loader 视为 val_loader (仅用于打印 loss 趋势)
        # 实际生产中建议专门留出验证数据集
        train_loader = get_uea_pretrain_loader(
            args.dataset_list, 
            extract_path=uea_path,
            batch_size=args.batch_size, 
            patch_size=args.patch_size, 
            split="train", 
            num_workers=args.num_workers,
            max_channels=args.max_channels,
            max_length=args.max_length,
            skip_variable_length=args.skip_variable_length,
        )
        val_loader = train_loader 
    else:
        raise ValueError("Must specify --dataset or --dataset_list")

    # 2. 创建模型
    print("\n🔧 Creating TSLANetEncoder...")
    # 关键修改：移除 num_channels 参数
    # TSLANet (CI策略) 只接受单通道输入，我们通过 Reshape 将所有通道堆叠到 Batch 维
    model = TSLANetEncoder(
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        dropout=args.dropout,
        # num_channels=... (REMOVED)
    ).to(device)
    
    print(f"   Model Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 3. 训练循环
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    best_loss = float("inf")
    patience_counter = 0
    loss_history = []
    
    print("\n🚀 开始训练...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, args.mask_ratio, device)
        val_loss = validate(model, val_loader, args.mask_ratio, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        loss_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "args": vars(args)
            }, args.save_path)
            print(f"   💾 保存最佳模型: {args.save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f"\n⏹️ 早停! 验证损失 {args.early_stop} 轮未改进")
                break
                
    # 保存历史
    history_path = str(args.save_path).replace(".pt", "_history.json")
    with open(history_path, "w") as f:
        json.dump(loss_history, f, indent=2)

    print("\n✅ 训练完成!")

if __name__ == "__main__":
    main()
