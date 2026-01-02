#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M1实验：UCR单数据集生成式分类训练脚本

使用约束解码的生成式分类方法，在单个UCR数据集上训练。
支持LoRA微调LLM + 训练编码器和投影层。

使用方法:
    python scripts/train_ucr_classification.py \
        --dataset ECG5000 \
        --encoder_type tslanet \
        --epochs 30 \
        --batch_size 8 \
        --use_lora

参数说明:
    --dataset: UCR数据集名称
    --encoder_type: 编码器类型 (transformer_cnn/tslanet)
    --encoder_pretrained: 编码器预训练权重路径
    --epochs: 训练轮数
    --batch_size: 批次大小
    --use_lora: 是否使用LoRA微调LLM
    --save_path: 模型保存路径
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.GenerativeClassifier import GenerativeClassifier
from opentslm.time_series_datasets.ucr.ucr_classification_dataset import (
    UCRClassificationDataset,
    collate_fn_classification,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UCR单数据集生成式分类训练")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 模型相关
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM模型ID")
    parser.add_argument("--encoder_type", type=str, default="tslanet", 
                       choices=["transformer_cnn", "tslanet"], help="编码器类型")
    parser.add_argument("--encoder_pretrained", type=str, default=None, help="编码器预训练权重")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch大小")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影层学习率")
    parser.add_argument("--lr_lora", type=float, default=1e-4, help="LoRA学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减")
    parser.add_argument("--warmup_frac", type=float, default=0.03, help="Warmup比例")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪")
    
    # LoRA相关
    parser.add_argument("--use_lora", action="store_true", help="是否使用LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # 其他
    parser.add_argument("--save_path", type=str, default=None, help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--early_stop", type=int, default=10, help="早停耐心值")
    parser.add_argument("--eval_every", type=int, default=1, help="每N个epoch评估一次")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: GenerativeClassifier,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # 计算loss
            loss = model.compute_loss(batch)
            total_loss += loss.item()
            
            # 约束解码预测
            predictions = model.predict(batch)
            
            # 计算准确率
            labels = [b["label_idx"] for b in batch]
            for pred, label in zip(predictions, labels):
                if pred == label:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader)
    
    return {"accuracy": accuracy, "loss": avg_loss}


def train_one_epoch(
    model: GenerativeClassifier,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler,
    grad_clip: float,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        
        clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}"
        })
    
    return total_loss / len(train_loader)


def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"M1实验: UCR单数据集生成式分类")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder_type}")
    print(f"LoRA: {args.use_lora}")
    print(f"时间: {datetime.datetime.now()}")
    print("=" * 60)
    
    set_seed(args.seed)
    
    # 设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        args.device = "cpu"
    device = args.device
    
    # 加载数据集
    print("\n📂 加载数据集...")
    train_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="train",
        raw_data_path=args.data_path,
        EOS_TOKEN="</s>",  # 临时值，后面会更新
    )
    
    test_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="test",
        raw_data_path=args.data_path,
        EOS_TOKEN="</s>",
    )
    
    num_classes = train_dataset.get_num_classes()
    print(f"   类别数: {num_classes}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    tslanet_config = {"patch_size": args.patch_size} if args.encoder_type == "tslanet" else None
    
    model = GenerativeClassifier(
        num_classes=num_classes,
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config,
    )
    
    # 更新数据集的EOS_TOKEN
    eos_token = model.get_eos_token()
    train_dataset.EOS_TOKEN = eos_token
    test_dataset.EOS_TOKEN = eos_token
    
    # 重新构建answer
    for i in range(len(train_dataset)):
        sample = train_dataset.df.iloc[i]
        label_idx = train_dataset.label_mapping[sample["label"]]
        train_dataset.df.at[i, "_answer_cache"] = f" {train_dataset.class_tokens[label_idx]}{eos_token}"
    
    # 启用LoRA
    if args.use_lora:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_classification(batch, patch_size=args.patch_size),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_classification(batch, patch_size=args.patch_size),
    )
    
    # 优化器 - 重要：包含embedding层用于训练类别token
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay},
        {"params": model.projector.parameters(), "lr": args.lr_projector, "weight_decay": args.weight_decay},
        # embedding层用于学习类别token
        {"params": [model.llm.get_input_embeddings().weight], "lr": args.lr_lora, "weight_decay": 0.0},
    ]
    
    if args.use_lora:
        lora_params = model.get_lora_parameters()
        if lora_params:
            param_groups.append({
                "params": lora_params, 
                "lr": args.lr_lora, 
                "weight_decay": args.weight_decay
            })
    
    optimizer = optim.AdamW(param_groups)
    
    # 学习率调度器
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    print(f"\n📈 训练步数: {total_steps}")
    print(f"🔥 Warmup步数: {warmup_steps}")
    
    # 保存路径
    if args.save_path is None:
        save_dir = f"results/m1_{args.dataset}_{args.encoder_type}"
        os.makedirs(save_dir, exist_ok=True)
        args.save_path = os.path.join(save_dir, "best_model.pt")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, args.grad_clip
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # 评估
        if epoch % args.eval_every == 0:
            metrics = evaluate(model, test_loader, device)
            test_acc = metrics["accuracy"]
            test_loss = metrics["loss"]
            
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                patience_counter = 0
                model.store_to_file(args.save_path)
                print(f"✔️ 新最佳模型! Acc: {best_acc:.4f}")
            else:
                patience_counter += 1
                print(f"   (无改进, patience: {patience_counter}/{args.early_stop})")
            
            # 早停
            if patience_counter >= args.early_stop:
                print(f"\n⏹️ 早停! 在epoch {epoch}")
                break
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"   数据集: {args.dataset}")
    print(f"   最佳准确率: {best_acc:.4f} (epoch {best_epoch})")
    print(f"   模型保存: {args.save_path}")
    print("=" * 60)
    
    return {"dataset": args.dataset, "best_accuracy": best_acc, "best_epoch": best_epoch}


if __name__ == "__main__":
    main()
