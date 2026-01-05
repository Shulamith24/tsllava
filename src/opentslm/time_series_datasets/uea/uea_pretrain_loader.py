# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UEA多变量数据集预训练加载器

修复版本：
1. 处理变长序列（跳过或截断）
2. 对大通道数/长度进行动态采样
3. Channel Independence策略
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


def is_variable_length(X) -> bool:
    """检查数据集是否为变长序列（返回的是list而非ndarray）"""
    return isinstance(X, list) or not hasattr(X, 'shape')


def convert_variable_length_to_fixed(X, max_len: int = 512) -> np.ndarray:
    """
    将变长序列转换为固定长度
    - 超长的截断
    - 短的填充
    """
    if not is_variable_length(X):
        return X
    
    samples = []
    for sample in X:
        # sample可能是 [C, L] 或 list of list
        if isinstance(sample, np.ndarray):
            c, l = sample.shape
        else:
            # 处理嵌套list
            try:
                sample = np.array(sample, dtype=np.float32)
                if sample.ndim == 1:
                    sample = sample.reshape(1, -1)
                c, l = sample.shape
            except:
                continue
        
        # 截断或填充到max_len
        if l > max_len:
            sample = sample[:, :max_len]
        elif l < max_len:
            pad = np.zeros((c, max_len - l), dtype=np.float32)
            sample = np.concatenate([sample, pad], axis=1)
        
        samples.append(sample)
    
    if not samples:
        return None
    
    # 确保所有样本通道数一致
    max_c = max(s.shape[0] for s in samples)
    result = []
    for s in samples:
        if s.shape[0] < max_c:
            pad = np.zeros((max_c - s.shape[0], s.shape[1]), dtype=np.float32)
            s = np.concatenate([s, pad], axis=0)
        result.append(s)
    
    return np.array(result, dtype=np.float32)


class UEAPretrainDataset(Dataset):
    """
    UEA单数据集预训练Dataset（带动态采样）
    
    Args:
        X: [N, C, L] 数据
        max_channels: 最大通道数，超过则随机采样
        max_length: 最大序列长度，超过则随机裁剪片段
    """
    def __init__(
        self,
        X: np.ndarray,
        max_channels: int = 32,
        max_length: int = 512,
    ):
        self.X = X
        self.max_channels = max_channels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # 获取样本 [C, L]
        x = self.X[idx].astype(np.float32)
        c, l = x.shape
        
        # 动态通道采样：如果通道数超限，随机选择max_channels个通道
        if c > self.max_channels:
            channel_indices = np.random.choice(c, self.max_channels, replace=False)
            channel_indices = np.sort(channel_indices)  # 保持顺序
            x = x[channel_indices, :]
            c = self.max_channels
        
        # 动态长度采样：如果长度超限，随机裁剪一段
        if l > self.max_length:
            start = np.random.randint(0, l - self.max_length + 1)
            x = x[:, start:start + self.max_length]
            l = self.max_length
        
        # Per-channel normalization
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std = np.clip(std, a_min=1e-8, a_max=None)
        x = (x - mean) / std
        
        # 处理NaN
        x = np.nan_to_num(x, nan=0.0)
        
        return torch.tensor(x, dtype=torch.float32)


class UEAMultiDatasetForPretrain(Dataset):
    """
    合并多个UEA数据集用于预训练（带动态采样）
    
    Args:
        dataset_names: 数据集名称列表
        split: "train" 或 "test"
        max_channels: 最大通道数
        max_length: 最大序列长度
        skip_variable_length: 是否跳过变长数据集
    """
    def __init__(
        self,
        dataset_names: List[str],
        split: str = "train",
        max_channels: int = 32,
        max_length: int = 512,
        skip_variable_length: bool = False,
    ):
        if not AEON_AVAILABLE:
            raise ImportError("aeon库未安装。请运行: pip install aeon")
        
        self.max_channels = max_channels
        self.max_length = max_length
        self.samples = []
        
        print(f"📂 Loading {len(dataset_names)} UEA datasets for pretraining...")
        print(f"   (max_channels={max_channels}, max_length={max_length})")
        
        for name in dataset_names:
            try:
                X, _ = load_classification(name, split=split)
                
                # 检查是否为变长序列
                if is_variable_length(X):
                    if skip_variable_length:
                        print(f"   ⏭ {name}: variable length, skipped")
                        continue
                    else:
                        # 尝试转换
                        X = convert_variable_length_to_fixed(X, max_length)
                        if X is None:
                            print(f"   ✗ {name}: failed to convert variable length")
                            continue
                        print(f"   ✓ {name}: {X.shape} (converted from variable length)")
                else:
                    print(f"   ✓ {name}: {X.shape}")
                
                # 添加样本
                for i in range(len(X)):
                    self.samples.append(X[i])  # [C, L]
                    
            except Exception as e:
                print(f"   ✗ {name}: {e}")
        
        if not self.samples:
            raise ValueError("No datasets loaded successfully!")
        
        print(f"   Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 获取样本 [C, L]
        x = self.samples[idx].astype(np.float32)
        c, l = x.shape
        
        # 动态通道采样
        if c > self.max_channels:
            channel_indices = np.random.choice(c, self.max_channels, replace=False)
            channel_indices = np.sort(channel_indices)
            x = x[channel_indices, :]
            c = self.max_channels
        
        # 动态长度采样
        if l > self.max_length:
            start = np.random.randint(0, l - self.max_length + 1)
            x = x[:, start:start + self.max_length]
            l = self.max_length
        
        # Per-channel normalization
        mean = np.nanmean(x, axis=1, keepdims=True)
        std = np.nanstd(x, axis=1, keepdims=True)
        std = np.clip(std, a_min=1e-8, a_max=None)
        x = (x - mean) / std
        
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
    max_channels: int = 32,
    max_length: int = 512,
    skip_variable_length: bool = True,
) -> DataLoader:
    """
    获取UEA多数据集预训练DataLoader
    
    Args:
        dataset_list_file: 数据集列表文件
        batch_size: 批次大小
        patch_size: patch大小
        split: 数据划分
        num_workers: 加载线程数
        max_channels: 最大通道数（超过则动态采样）
        max_length: 最大序列长度（超过则动态裁剪）
        skip_variable_length: 是否跳过变长数据集
    """
    dataset_names = load_dataset_list(dataset_list_file)
    dataset = UEAMultiDatasetForPretrain(
        dataset_names, 
        split=split,
        max_channels=max_channels,
        max_length=max_length,
        skip_variable_length=skip_variable_length,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=lambda batch: collate_fn_pretrain(batch, patch_size),
    )
    
    return loader


# 推荐的UEA数据集（排除变长和超大数据集）
UEA_PRETRAIN_DATASETS_SAFE = [
    "ArticularyWordRecognition",
    "AtrialFibrillation",
    "BasicMotions",
    "Cricket",
    "ERing",
    "Epilepsy",
    "EthanolConcentration",
    "FingerMovements",
    "HandMovementDirection",
    "Handwriting",
    "Libras",
    "LSST",
    "NATOPS",
    "PenDigits",
    "RacketSports",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "StandWalkJump",
    "UWaveGestureLibrary",
]


if __name__ == "__main__":
    print("Testing UEA pretrain loader with dynamic sampling...")
    
    if AEON_AVAILABLE:
        # 测试单个数据集
        X_train, _ = load_classification("Handwriting", split="train")
        print(f"Handwriting: {X_train.shape}")
        
        dataset = UEAPretrainDataset(X_train, max_channels=32, max_length=512)
        print(f"Dataset size: {len(dataset)}")
        print(f"Sample shape: {dataset[0].shape}")
        
        # 测试大通道数据集
        X_large, _ = load_classification("FaceDetection", split="train")
        print(f"\nFaceDetection (large channels): {X_large.shape}")
        
        dataset_large = UEAPretrainDataset(X_large, max_channels=32, max_length=512)
        sample = dataset_large[0]
        print(f"Sampled shape: {sample.shape}")
    else:
        print("请安装aeon: pip install aeon")

