#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
TiViT UCR 时序分类训练脚本

使用 Linear Probing 方式训练:
1. 使用冻结的 ViT 提取所有样本的特征
2. 使用 sklearn LogisticRegression 进行分类
3. 评估测试集准确率

用法:
    python -m src.patchtst_ucr.train_tivit \
        --dataset ECG200 \
        --data_path ./data \
        --save_dir ./results/tivit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 添加 src 目录到路径
script_dir = Path(__file__).parent
src_dir = script_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from patchtst_ucr.tivit_model import TiViTFeatureExtractor, SUPPORTED_VIT_MODELS
from patchtst_ucr.ucr_dataset import UCRDatasetForPatchTST, get_dataset_info


# =============================================================================
# Arguments
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="TiViT Time Series Classification with Linear Probing"
    )
    
    # 数据集
    parser.add_argument("--dataset", type=str, required=True,
                        help="UCR 数据集名称")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="数据根目录")
    parser.add_argument("--save_dir", type=str, default="./results/tivit",
                        help="结果保存目录")
    
    # ViT 模型配置
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov2-base",
                        choices=SUPPORTED_VIT_MODELS,
                        help="预训练 ViT 模型名称")
    parser.add_argument("--vit_layer", type=int, default=None,
                        help="使用哪一层特征 (None=最后一层)")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "cls_token"],
                        help="特征聚合方式")
    
    # 时序转图像配置
    parser.add_argument("--patch_size_mode", type=str, default="sqrt",
                        choices=["sqrt", "linspace"],
                        help="Patch size 计算模式")
    parser.add_argument("--stride", type=float, default=0.1,
                        help="步长比例 (0-1)")
    
    # 分类器配置
    parser.add_argument("--classifier_type", type=str, default="logistic_regression",
                        choices=["logistic_regression", "nearest_centroid"],
                        help="分类器类型")
    parser.add_argument("--max_iter", type=int, default=500,
                        help="LogisticRegression 最大迭代次数")
    
    # 其他
    parser.add_argument("--batch_size", type=int, default=64,
                        help="特征提取 batch size")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (默认自动选择)")
    
    return parser.parse_args()


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_all_features(
    extractor: TiViTFeatureExtractor,
    dataset: UCRDatasetForPatchTST,
    batch_size: int,
    device: str,
    desc: str = "Extracting features",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 TiViT 提取数据集所有样本的特征
    
    Args:
        extractor: TiViT 特征提取器
        dataset: UCR 数据集
        batch_size: 批大小
        device: 设备
        desc: 进度条描述
    
    Returns:
        features: [N, hidden_size] 特征矩阵
        labels: [N] 标签数组
    """
    def collate_fn(batch):
        """将样本打包为 batch"""
        time_series = torch.stack([item["time_series"][0] for item in batch])
        labels = torch.tensor([item["int_label"] for item in batch])
        return time_series, labels
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 避免 Windows 多进程问题
        collate_fn=collate_fn,
    )
    
    all_features = []
    all_labels = []
    
    extractor.eval()
    
    with torch.no_grad():
        for time_series, labels in tqdm(dataloader, desc=desc):
            time_series = time_series.to(device)
            features = extractor(time_series)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return features, labels


# =============================================================================
# Classifier Training
# =============================================================================

def get_classifier(classifier_type: str, max_iter: int = 500, seed: int = 42):
    """创建分类器"""
    if classifier_type == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, random_state=seed),
        )
    elif classifier_type == "nearest_centroid":
        return make_pipeline(
            StandardScaler(),
            NearestCentroid(),
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


def train_and_evaluate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    classifier_type: str,
    max_iter: int = 500,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    训练分类器并评估
    
    Args:
        train_features/labels: 训练集
        test_features/labels: 测试集
        classifier_type: 分类器类型
        max_iter: 最大迭代次数
        seed: 随机种子
    
    Returns:
        train_accuracy, test_accuracy
    """
    clf = get_classifier(classifier_type, max_iter=max_iter, seed=seed)
    
    # 训练
    clf.fit(train_features, train_labels)
    
    # 评估
    train_acc = clf.score(train_features, train_labels)
    test_acc = clf.score(test_features, test_labels)
    
    return train_acc, test_acc


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 60)
    print("TiViT Linear Probing Classification")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"ViT Model: {args.vit_model_name}")
    print(f"ViT Layer: {args.vit_layer if args.vit_layer else 'last'}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Patch Size Mode: {args.patch_size_mode}")
    print(f"Stride: {args.stride}")
    print(f"Classifier: {args.classifier_type}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # 加载数据集
    print("\n[1/4] Loading dataset...")
    train_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="train",
        raw_data_path=args.data_path,
    )
    test_dataset = UCRDatasetForPatchTST(
        dataset_name=args.dataset,
        split="test",
        raw_data_path=args.data_path,
    )
    
    num_classes = train_dataset.get_num_classes()
    max_length = train_dataset.get_max_length()
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Num classes: {num_classes}")
    print(f"   Sequence length: {max_length}")
    
    # 加载 TiViT 特征提取器
    print("\n[2/4] Loading TiViT feature extractor...")
    extractor = TiViTFeatureExtractor(
        model_name=args.vit_model_name,
        layer_idx=args.vit_layer,
        aggregation=args.aggregation,
        patch_size_mode=args.patch_size_mode,
        stride=args.stride,
        freeze=True,
    )
    extractor = extractor.to(device)
    
    hidden_size = extractor.get_hidden_size()
    print(f"   Hidden size: {hidden_size}")
    
    # 提取特征
    print("\n[3/4] Extracting features...")
    train_features, train_labels = extract_all_features(
        extractor, train_dataset, args.batch_size, device, "   Train set"
    )
    test_features, test_labels = extract_all_features(
        extractor, test_dataset, args.batch_size, device, "   Test set"
    )
    
    print(f"   Train features shape: {train_features.shape}")
    print(f"   Test features shape: {test_features.shape}")
    
    # 训练分类器
    print(f"\n[4/4] Training {args.classifier_type}...")
    train_acc, test_acc = train_and_evaluate(
        train_features, train_labels,
        test_features, test_labels,
        args.classifier_type,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    
    print(f"   Train accuracy: {train_acc:.4f}")
    print(f"   Test accuracy: {test_acc:.4f}")
    
    # 保存结果
    vit_short = args.vit_model_name.split("/")[-1].replace("-", "_")
    result_dir = Path(args.save_dir) / args.dataset / f"{timestamp}_{vit_short}"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    result = {
        "dataset": args.dataset,
        "timestamp": timestamp,
        "vit_model_name": args.vit_model_name,
        "vit_layer": args.vit_layer,
        "aggregation": args.aggregation,
        "patch_size_mode": args.patch_size_mode,
        "stride": args.stride,
        "classifier_type": args.classifier_type,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "seed": args.seed,
    }
    
    result_path = result_dir / "final_results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Results saved to: {result_path}")
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {args.dataset} Test Accuracy = {test_acc:.4f}")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
