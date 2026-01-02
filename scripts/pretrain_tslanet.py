#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet编码器在UCR数据集上的预训练脚本

使用掩码预训练方法（类似MAE/PatchTST）在UCR的98个训练数据集上预训练TSLANet编码器。

使用方法:
    python scripts/pretrain_tslanet.py \
        --dataset_list src/opentslm/time_series_datasets/ucr/ucr_train_98_datasets.txt \
        --save_path pretrained/tslanet_ucr98.pt \
        --epochs 50 \
        --batch_size 64 \
        --mask_ratio 0.4

参数说明:
    --dataset_list: 训练数据集列表文件路径
    --save_path: 预训练权重保存路径
    --epochs: 训练轮数 (默认50)
    --batch_size: 批次大小 (默认64)
    --mask_ratio: 掩码比例 (默认0.4)
    --lr: 学习率 (默认1e-3)
    --patch_size: patch大小 (默认8)
    --emb_dim: 嵌入维度 (默认128)
    --depth: 编码器深度 (默认2)
    --data_path: UCR数据根目录 (默认./data)
"""

import os
import sys
import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
from opentslm.time_series_datasets.ucr.ucr_pretrain_loader import (
    get_ucr_pretrain_loader,
    load_dataset_list,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TSLANet UCR预训练")
    
    # 数据相关
    parser.add_argument(
        "--dataset_list",
        type=str,
        default="src/opentslm/time_series_datasets/ucr/ucr_train_98_datasets.txt",
        help="训练数据集列表文件路径",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="UCR数据根目录",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="pretrained/tslanet_ucr98.pt",
        help="预训练权重保存路径",
    )
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--mask_ratio", type=float, default=0.4, help="掩码比例")
    
    # 模型结构
    parser.add_argument("--patch_size", type=int, default=8, help="patch大小")
    parser.add_argument("--emb_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--depth", type=int, default=2, help="编码器深度")
    parser.add_argument("--dropout", type=float, default=0.15, help="dropout比例")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--dry_run", action="store_true", help="干运行模式(只运行1个batch)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: TSLANetEncoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    mask_ratio: float,
    device: str,
    dry_run: bool = False,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        
        # 掩码预训练前向传播
        preds, target, mask = model.pretrain_forward(batch, mask_ratio=mask_ratio)
        
        # 计算损失 (只在被掩码的位置计算)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        if dry_run:
            break
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: TSLANetEncoder,
    val_loader: DataLoader,
    mask_ratio: float,
    device: str,
    dry_run: bool = False,
) -> float:
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        batch = batch.to(device)
        
        preds, target, mask = model.pretrain_forward(batch, mask_ratio=mask_ratio)
        
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss += loss.item()
        num_batches += 1
        
        if dry_run:
            break
    
    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("TSLANet UCR预训练")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"参数: {args}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = "cpu"
    device = args.device
    
    # 检查数据集列表文件
    if not os.path.exists(args.dataset_list):
        print(f"❌ 数据集列表文件不存在: {args.dataset_list}")
        sys.exit(1)
    
    # 加载数据
    print("\n📂 加载训练数据...")
    train_loader = get_ucr_pretrain_loader(
        dataset_list_file=args.dataset_list,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        raw_data_path=args.data_path,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
    
    # 加载验证数据 (使用测试集的一部分作为验证)
    print("\n📂 加载验证数据...")
    val_loader = get_ucr_pretrain_loader(
        dataset_list_file=args.dataset_list,
        split="test",  # 使用test split的数据作为验证
        batch_size=args.batch_size,
        shuffle=False,
        raw_data_path=args.data_path,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
    
    # 创建模型
    print("\n🔧 创建TSLANet编码器...")
    model = TSLANetEncoder(
        output_dim=args.emb_dim,
        patch_size=args.patch_size,
        emb_dim=args.emb_dim,
        depth=args.depth,
        dropout=args.dropout,
        use_icb=True,
        use_asb=True,
        adaptive_filter=True,
    ).to(device)
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数量: {num_params:,}")
    print(f"   可训练参数: {num_trainable:,}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    
    # 创建保存目录
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.mask_ratio, device, args.dry_run
        )
        
        # 验证
        val_loss = validate(model, val_loader, args.mask_ratio, device, args.dry_run)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "args": vars(args),
            }
            torch.save(checkpoint, args.save_path)
            print(f"💾 保存最佳模型到: {args.save_path}")
        else:
            patience_counter += 1
            print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
        
        # 早停
        if patience_counter >= args.early_stop:
            print(f"\n⏹️ 早停! 验证损失 {args.early_stop} 轮未改进")
            break
        
        if args.dry_run:
            print("\n🧪 干运行模式，提前退出")
            break
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"   最佳验证损失: {best_val_loss:.6f}")
    print(f"   模型保存路径: {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
