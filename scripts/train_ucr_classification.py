#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
M1实验：单UCR数据集生成式分类训练

使用生成式分类方法在单个UCR数据集上进行有监督分类训练。
- Prompt格式: Dataset=<name>. Classes: <cls_0>, <cls_1>, ... Predict label:
- 只对label token计算交叉熵损失
- 约束解码只允许类别token

使用方法:
    python scripts/train_ucr_classification.py \
        --dataset ECG5000 \
        --encoder_type tslanet \
        --epochs 20 \
        --batch_size 16

参数说明:
    --dataset: UCR数据集名称 (如 ECG5000)
    --encoder_type: 编码器类型 (tslanet 或 transformer_cnn)
    --encoder_pretrained: 预训练权重路径 (可选)
    --epochs: 训练轮数 (默认20)
    --batch_size: 批次大小 (默认16)
    --lr: 学习率 (默认1e-4)
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.llm.GenerativeClassifier import GenerativeClassifier
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data


class UCRClassificationDataset(Dataset):
    """
    UCR分类数据集
    
    为生成式分类准备数据：时间序列+标签
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        raw_data_path: str = "./data",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        
        # 加载数据
        ensure_ucr_data()
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        
        if split == "train":
            df = train_df
        else:
            df = test_df
        
        self.df = df.reset_index(drop=True)
        self.feature_cols = [c for c in df.columns if c != "label"]
        
        # 重新映射标签到0开始的整数
        unique_labels = sorted(df["label"].unique())
        self.label_map = {old: new for new, old in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
        print(f"📂 {split}: {len(self.df)} 样本, {self.num_classes} 类")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 时间序列
        feats = row[self.feature_cols].astype(float).values
        ts = torch.tensor(feats, dtype=torch.float32)
        ts = torch.nan_to_num(ts, nan=0.0)
        
        # Z-normalization
        mean = ts.mean()
        std = ts.std()
        if std > 1e-8:
            ts = (ts - mean) / std
        else:
            ts = ts - mean
        
        # 标签
        label = self.label_map[row["label"]]
        
        return ts, label


def collate_classification_batch(
    batch: List,
    dataset_name: str,
    classifier: GenerativeClassifier,
    patch_size: int = 8,
):
    """
    Collate函数：将batch转换为OpenTSLMSP格式
    """
    ts_list = []
    labels = []
    
    for ts, label in batch:
        ts_list.append(ts)
        labels.append(label)
    
    # 找到最大长度并填充
    max_len = max(ts.shape[0] for ts in ts_list)
    if max_len % patch_size != 0:
        max_len = max_len + (patch_size - max_len % patch_size)
    
    # Pad时间序列
    padded_ts = []
    for ts in ts_list:
        if ts.shape[0] < max_len:
            pad_len = max_len - ts.shape[0]
            ts = torch.nn.functional.pad(ts, (0, pad_len))
        padded_ts.append(ts)
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    # 构建OpenTSLMSP batch格式
    batch_dicts = []
    for ts in padded_ts:
        sample = classifier.build_classification_prompt(
            dataset_name=dataset_name,
            time_series=ts,
        )
        batch_dicts.append(sample)
    
    return batch_dicts, labels


def train_one_epoch(
    model: OpenTSLMSP,
    classifier: GenerativeClassifier,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_dicts, labels in pbar:
        labels = labels.to(device)
        
        # 前向传播 + 损失
        loss, logits = classifier.compute_classification_loss(batch_dicts, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 计算batch准确率
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.2%}"})
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: OpenTSLMSP,
    classifier: GenerativeClassifier,
    val_loader: DataLoader,
    device: str,
) -> Dict:
    """评估"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    for batch_dicts, labels in tqdm(val_loader, desc="Evaluating"):
        labels = labels.to(device)
        
        # 约束解码预测
        predictions, logits = classifier.predict(batch_dicts)
        
        all_preds.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        
        # 计算loss (可选)
        loss, _ = classifier.compute_classification_loss(batch_dicts, labels)
        total_loss += loss.item()
        num_batches += 1
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "predictions": all_preds,
        "labels": all_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="M1: UCR单数据集生成式分类")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, required=True, help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="数据路径")
    parser.add_argument("--save_dir", type=str, default="results/m1_classification", help="保存目录")
    
    # 模型相关
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B", help="LLM模型ID")
    parser.add_argument("--encoder_type", type=str, default="tslanet", choices=["tslanet", "transformer_cnn"])
    parser.add_argument("--encoder_pretrained", type=str, default=None, help="编码器预训练权重路径")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--lr_encoder", type=float, default=2e-4, help="编码器学习率")
    parser.add_argument("--lr_projector", type=float, default=1e-4, help="投影器学习率")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--patch_size", type=int, default=8, help="patch大小")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("M1实验: UCR单数据集生成式分类")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder_type}")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，使用CPU")
        device = "cpu"
    
    # 加载数据
    print("\n📂 加载数据集...")
    train_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="train",
        raw_data_path=args.data_path,
    )
    test_dataset = UCRClassificationDataset(
        dataset_name=args.dataset,
        split="test",
        raw_data_path=args.data_path,
    )
    
    num_classes = train_dataset.num_classes
    print(f"   类别数: {num_classes}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    tslanet_config = {"patch_size": args.patch_size}
    
    model = OpenTSLMSP(
        llm_id=args.llm_id,
        device=device,
        encoder_type=args.encoder_type,
        encoder_pretrained_path=args.encoder_pretrained,
        tslanet_config=tslanet_config if args.encoder_type == "tslanet" else None,
    )
    
    # 创建生成式分类器
    classifier = GenerativeClassifier(
        model=model,
        num_classes=num_classes,
    )
    
    # 创建DataLoader (使用lambda包装collate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_classification_batch(
            batch, args.dataset, classifier, args.patch_size
        ),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_classification_batch(
            batch, args.dataset, classifier, args.patch_size
        ),
    )
    
    # 优化器 - 只训练encoder和projector
    param_groups = [
        {"params": model.encoder.parameters(), "lr": args.lr_encoder},
        {"params": model.projector.parameters(), "lr": args.lr_projector},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 创建保存目录
    save_dir = Path(args.save_dir) / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    print("\n🚀 开始训练...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_one_epoch(model, classifier, train_loader, optimizer, device)
        
        # 评估
        eval_results = evaluate(model, classifier, test_loader, device)
        test_acc = eval_results["accuracy"]
        test_loss = eval_results["loss"]
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2%}")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                "epoch": epoch,
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "test_acc": test_acc,
                "num_classes": num_classes,
                "class_tokens": classifier.class_tokens,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"💾 保存最佳模型 (acc={test_acc:.2%})")
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成!")
    print(f"   数据集: {args.dataset}")
    print(f"   最佳准确率: {best_acc:.2%}")
    print(f"   模型保存: {save_dir / 'best_model.pt'}")
    print("=" * 60)
    
    # 保存结果
    results = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "best_accuracy": float(best_acc),
        "encoder_type": args.encoder_type,
        "epochs": args.epochs,
    }
    
    import json
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_acc


if __name__ == "__main__":
    main()
