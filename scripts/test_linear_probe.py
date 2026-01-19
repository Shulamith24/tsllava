#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Linear Probe 测试脚本

用于验证预训练 encoder/projector 的表征能力。
仅使用 encoder + projector + 线性分类头，不使用 LLM。

使用方法：
    python scripts/test_linear_probe.py \
        --local_checkpoint results/xxx/best_model.pt \
        --encoder_type tslanet \
        --dataset ECG5000 \
        --epochs 100 \
        --batch_size 32

输出：
    - 训练/测试准确率
    - 保存结果到 results/linear_probe/
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from opentslm.model_config import PATCH_SIZE, ENCODER_OUTPUT_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Linear Probe 测试 - 验证 encoder/projector 表征能力")
    
    # 模型相关
    parser.add_argument("--local_checkpoint", type=str, required=True,
                        help="本地checkpoint路径")
    parser.add_argument("--encoder_type", type=str, default="tslanet",
                        choices=["transformer_cnn", "tslanet"],
                        help="编码器类型")
    parser.add_argument("--tslanet_patch_size", type=int, default=8,
                        help="TSLANet的patch_size")
    
    # 数据相关
    parser.add_argument("--dataset", type=str, default="ECG5000", help="UCR数据集名称")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR数据根目录")
    
    # 训练相关
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    
    # 保存相关
    parser.add_argument("--save_dir", type=str, default="results/linear_probe", help="结果保存目录")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--freeze_encoder", action="store_true", help="冻结encoder和projector（纯线性探测）")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UCRDataset(Dataset):
    """简化的UCR数据集，直接返回时间序列和标签"""
    
    def __init__(self, split: str, dataset_name: str, data_path: str):
        from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data
        
        ensure_ucr_data()
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=data_path)
        
        if split == "train":
            df = train_df
        else:
            df = test_df
        
        # 获取所有唯一标签并排序
        all_labels = sorted(train_df["label"].unique().tolist())
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        self.num_classes = len(all_labels)
        
        # 转换为numpy
        self.labels = df["label"].values
        feature_cols = [col for col in df.columns if col != "label"]
        self.data = df[feature_cols].values.astype(np.float32)
        
        # 处理NaN
        self.data = np.nan_to_num(self.data, nan=0.0)
        
        print(f"📊 Loaded {split} set: {len(self.data)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ts = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # Per-sample z-normalization
        mean = ts.mean()
        std = ts.std()
        if std > 1e-8:
            ts = (ts - mean) / std
        else:
            ts = ts - mean
        
        label = self.label_to_idx[self.labels[idx]]
        return ts, label


def collate_fn(batch, patch_size: int):
    """Collate函数：填充到patch_size的整倍数"""
    time_series = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # 找到最大长度并填充到patch_size的倍数
    max_len = max(ts.shape[0] for ts in time_series)
    rem = max_len % patch_size
    if rem:
        max_len = max_len + patch_size - rem
    
    # 填充
    padded = torch.zeros(len(time_series), max_len)
    for i, ts in enumerate(time_series):
        padded[i, :ts.shape[0]] = ts
    
    return padded, labels


class LinearProbeModel(nn.Module):
    """线性探测模型：encoder + projector + 线性分类头"""
    
    def __init__(
        self,
        encoder_type: str,
        num_classes: int,
        device: str,
        tslanet_patch_size: int = 8,
        llm_hidden_size: int = 2048,  # default for Llama-3.2-1B
    ):
        super().__init__()
        self.device = device
        self.patch_size = tslanet_patch_size if encoder_type == "tslanet" else 4
        
        # 创建encoder
        if encoder_type == "tslanet":
            from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
            self.encoder = TSLANetEncoder(
                output_dim=ENCODER_OUTPUT_DIM,
                patch_size=tslanet_patch_size,
                emb_dim=128,
                depth=2,
                dropout=0.15,
            ).to(device)
        else:
            from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
            self.encoder = TransformerCNNEncoder().to(device)
        
        # 创建projector (与OpenTSLMSP一致)
        from opentslm.model.projector.MLPProjector import MLPProjector
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, llm_hidden_size, device=device
        ).to(device)
        
        # 线性分类头：使用pooling后的特征
        # encoder 输出 [B, N_patches, ENCODER_OUTPUT_DIM]
        # projector 输出 [B, N_patches, llm_hidden_size]
        # 使用mean pooling -> [B, llm_hidden_size]
        self.classifier = nn.Linear(llm_hidden_size, num_classes).to(device)
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: [B, T] 时间序列
        Returns:
            logits: [B, num_classes]
        """
        # Encoder: [B, T] -> [B, N_patches, ENCODER_OUTPUT_DIM]
        enc_out = self.encoder(x)
        
        # Projector: [B, N_patches, ENCODER_OUTPUT_DIM] -> [B, N_patches, llm_hidden_size]
        proj_out = self.projector(enc_out)
        
        # Mean pooling: [B, N_patches, llm_hidden_size] -> [B, llm_hidden_size]
        pooled = proj_out.mean(dim=1)
        
        # Classifier: [B, llm_hidden_size] -> [B, num_classes]
        logits = self.classifier(pooled)
        
        return logits
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """加载预训练的encoder和projector权重"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载encoder
        if "encoder_state" in checkpoint:
            self.encoder.load_state_dict(checkpoint["encoder_state"])
            print(f"✅ Loaded encoder weights")
        
        # 加载projector
        if "projector_state" in checkpoint:
            self.projector.load_state_dict(checkpoint["projector_state"])
            print(f"✅ Loaded projector weights")


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(train_loader), correct / total


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(data_loader), correct / total


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Linear Probe 测试 - 验证 encoder/projector 表征能力")
    print("=" * 60)
    print(f"时间: {datetime.datetime.now()}")
    print(f"数据集: {args.dataset}")
    print(f"编码器: {args.encoder_type}")
    print(f"冻结encoder: {args.freeze_encoder}")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 加载数据
    print("\n📂 加载数据...")
    train_dataset = UCRDataset("train", args.dataset, args.data_path)
    test_dataset = UCRDataset("test", args.dataset, args.data_path)
    
    patch_size = args.tslanet_patch_size if args.encoder_type == "tslanet" else 4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, patch_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, patch_size),
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # 创建模型
    print("\n🔧 创建模型...")
    model = LinearProbeModel(
        encoder_type=args.encoder_type,
        num_classes=train_dataset.num_classes,
        device=device,
        tslanet_patch_size=args.tslanet_patch_size,
    )
    
    # 加载预训练权重
    print(f"\n📂 加载预训练权重: {args.local_checkpoint}")
    model.load_pretrained_weights(args.local_checkpoint)
    
    # 冻结encoder和projector（如果需要）
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        for param in model.projector.parameters():
            param.requires_grad = False
        print("🧊 Encoder 和 Projector 已冻结")
        
        # 只训练分类头
        optimizer = AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        # 训练所有参数
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    print("\n🚀 开始训练...")
    best_test_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "classifier_state": model.classifier.state_dict(),
                "test_acc": test_acc,
                "epoch": epoch,
            }, os.path.join(save_dir, "best_model.pt"))
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        })
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")
    
    # 保存结果
    print("\n" + "=" * 60)
    print(f"✅ 最佳测试准确率: {best_test_acc:.4f}")
    print("=" * 60)
    
    final_results = {
        "dataset": args.dataset,
        "encoder_type": args.encoder_type,
        "freeze_encoder": args.freeze_encoder,
        "best_test_acc": best_test_acc,
        "final_train_acc": history[-1]["train_acc"],
        "final_test_acc": history[-1]["test_acc"],
    }
    
    with open(os.path.join(save_dir, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"结果保存到: {save_dir}")


if __name__ == "__main__":
    main()
