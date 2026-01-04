# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UEA多变量数据集预训练加载器

用于TSLANet在多个UEA数据集上的预训练。
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入aeon库
try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False


def load_dataset_list(file_path: str) -> List[str]:
    """从文件加载数据集名称列表"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    datasets = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            datasets.append(line)
    
    return datasets


class UEAPretrainDataset(Dataset):
    """
    UEA单数据集预训练Dataset
    
    只返回归一化后的时间序列数据（不包含标签）
    """
    def __init__(
        self,
        X: np.ndarray,  # [N, C, L]
    ):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取样本 [C, L]
        x = self.X[idx].astype(np.float32)
        
        # Per-sample normalization (在通道维度上)
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std = np.clip(std, a_min=1e-8, a_max=None)
        x = (x - mean) / std
        
        # 处理NaN
        x = np.nan_to_num(x, nan=0.0)
        
        return torch.tensor(x, dtype=torch.float32)


class UEAMultiDatasetForPretrain(Dataset):
    """
    合并多个UEA数据集用于预训练
    """
    def __init__(
        self,
        dataset_names: List[str],
        split: str = "train",
    ):
        """
        Args:
            dataset_names: UEA数据集名称列表
            split: "train" 或 "test"
        """
        if not AEON_AVAILABLE:
            raise ImportError("aeon库未安装。请运行: pip install aeon")
        
        all_samples = []
        
        print(f"📂 Loading {len(dataset_names)} UEA datasets for pretraining...")
        for name in dataset_names:
            try:
                X, _ = load_classification(name, split=split)
                # X: [N, C, L]
                all_samples.append(X)
                print(f"   ✓ {name}: {X.shape}")
            except Exception as e:
                print(f"   ✗ {name}: {e}")
        
        if not all_samples:
            raise ValueError("No datasets loaded successfully!")
        
        # 存储每个样本的信息（不拼接，因为通道数/长度可能不同）
        self.samples = []
        for X in all_samples:
            for i in range(len(X)):
                self.samples.append(X[i])  # [C, L]
        
        print(f"   Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 获取样本 [C, L]
        x = self.samples[idx].astype(np.float32)
        
        # Per-sample normalization
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std = np.clip(std, a_min=1e-8, a_max=None)
        x = (x - mean) / std
        
        # 处理NaN
        x = np.nan_to_num(x, nan=0.0)
        
        return torch.tensor(x, dtype=torch.float32)


def collate_fn_pretrain(batch: List[torch.Tensor], patch_size: int = 8) -> torch.Tensor:
    """
    预训练专用collate函数
    
    处理不同长度/通道数的时序，填充到patch_size的倍数
    """
    # 找到最大长度和最大通道数
    max_len = max(x.shape[1] for x in batch)
    max_channels = max(x.shape[0] for x in batch)
    
    # 填充到patch_size的倍数
    rem = max_len % patch_size
    if rem != 0:
        max_len = max_len + (patch_size - rem)
    
    # 创建填充后的batch
    batch_size = len(batch)
    padded_batch = torch.zeros(batch_size, max_channels, max_len)
    
    for i, x in enumerate(batch):
        c, l = x.shape
        padded_batch[i, :c, :l] = x
    
    return padded_batch


def get_uea_pretrain_loader(
    dataset_list_file: str,
    batch_size: int = 16,
    patch_size: int = 8,
    split: str = "train",
    num_workers: int = 0,
) -> DataLoader:
    """
    获取UEA多数据集预训练DataLoader
    """
    dataset_names = load_dataset_list(dataset_list_file)
    dataset = UEAMultiDatasetForPretrain(dataset_names, split=split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_pretrain(batch, patch_size),
    )
    
    return loader


# 常用UEA数据集列表
UEA_PRETRAIN_DATASETS = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "CharacterTrajectories",
    "Cricket",
    "DuckDuckGeese",
    "ERing",
    "EigenWorms",
    "Epilepsy",
    "EthanolConcentration",
    "FaceDetection",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Heartbeat",
    "InsectWingbeat",
    "JapaneseVowels",
    "Libras",
    "LSST",
    "MotorImagery",
    "NATOPS",
    "PenDigits",
    "PEMS-SF",
    "PhonemeSpectra",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "SpokenArabicDigits",
    "StandWalkJump",
    "UWaveGestureLibrary",
]


if __name__ == "__main__":
    # 测试单数据集加载
    from aeon.datasets import load_classification
    
    print("Testing UEA pretrain loader...")
    
    # 测试单个数据集
    X_train, y_train = load_classification("Handwriting", split="train")
    print(f"Handwriting: {X_train.shape}")
    
    dataset = UEAPretrainDataset(X_train)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample shape: {dataset[0].shape}")
