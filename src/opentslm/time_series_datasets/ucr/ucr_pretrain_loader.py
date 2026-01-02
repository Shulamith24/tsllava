# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR多数据集预训练加载器

用于组合多个UCR数据集进行TSLANet编码器预训练。
仅加载时间序列数据（无监督预训练，不使用标签）。
"""

import os
from typing import List, Optional, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from opentslm.time_series_datasets.ucr.ucr_loader import (
    ensure_ucr_data,
    load_ucr_dataset,
    UCR_DIR,
)


def load_dataset_list(file_path: str) -> List[str]:
    """
    从文件加载数据集名称列表
    
    Args:
        file_path: 包含数据集名称的文本文件路径
    
    Returns:
        数据集名称列表
    """
    datasets = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                datasets.append(line)
    return datasets


class UCRPretrainDataset(Dataset):
    """
    单个UCR数据集的预训练Dataset
    
    仅返回归一化的时间序列数据（用于无监督预训练）
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "label",
        patch_size: int = 8,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols or [c for c in df.columns if c != label_col]
        self.patch_size = patch_size
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feats = row[self.feature_cols].astype(float).values
        tensor = torch.tensor(feats, dtype=torch.float32)
        
        # 处理NaN值
        tensor = torch.nan_to_num(tensor, nan=0.0)
        
        # Per-sample z-normalization
        mean = tensor.mean()
        std = tensor.std()
        if std > 1e-8:
            tensor = (tensor - mean) / std
        else:
            tensor = tensor - mean
        
        return tensor


def collate_fn_pretrain(batch: List[torch.Tensor], patch_size: int = 8):
    """
    预训练批次collate函数
    
    将不同长度的序列填充到相同长度（patch_size的倍数）
    """
    # 找到最大长度
    max_len = max(x.shape[0] for x in batch)
    
    # 填充到patch_size的倍数
    if max_len % patch_size != 0:
        max_len = max_len + (patch_size - max_len % patch_size)
    
    # 填充
    padded = []
    for x in batch:
        if x.shape[0] < max_len:
            pad_len = max_len - x.shape[0]
            x = torch.nn.functional.pad(x, (0, pad_len))
        padded.append(x)
    
    return torch.stack(padded)


class UCRMultiDatasetForPretrain(Dataset):
    """
    组合多个UCR数据集用于编码器预训练
    
    Args:
        dataset_names: 数据集名称列表
        split: "train", "test", 或 "all"
        raw_data_path: UCR数据路径
        patch_size: patch大小
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        split: Literal["train", "test", "all"] = "train",
        raw_data_path: str = "./data",
        patch_size: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dataset_names = dataset_names
        
        # 确保数据已下载
        ensure_ucr_data()
        
        # 加载所有数据集
        all_datasets = []
        total_samples = 0
        
        for name in dataset_names:
            try:
                train_df, test_df = load_ucr_dataset(name, raw_data_path=raw_data_path)
                
                if split == "train":
                    df = train_df
                elif split == "test":
                    df = test_df
                else:
                    df = pd.concat([train_df, test_df], ignore_index=True)
                
                dataset = UCRPretrainDataset(df, patch_size=patch_size)
                all_datasets.append(dataset)
                total_samples += len(dataset)
                
            except Exception as e:
                print(f"⚠️ 加载数据集 {name} 失败: {e}")
                continue
        
        if not all_datasets:
            raise ValueError("没有成功加载任何数据集！")
        
        self.combined_dataset = ConcatDataset(all_datasets)
        print(f"✅ 加载了 {len(all_datasets)} 个数据集，共 {total_samples} 个样本")
    
    def __len__(self):
        return len(self.combined_dataset)
    
    def __getitem__(self, idx):
        return self.combined_dataset[idx]


def get_ucr_pretrain_loader(
    dataset_list_file: str,
    split: Literal["train", "test", "all"] = "train",
    batch_size: int = 64,
    shuffle: bool = True,
    raw_data_path: str = "./data",
    patch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """
    获取UCR预训练DataLoader
    
    Args:
        dataset_list_file: 数据集列表文件路径
        split: 数据划分
        batch_size: 批次大小
        shuffle: 是否打乱
        raw_data_path: 数据路径
        patch_size: patch大小
        num_workers: 数据加载线程数
    
    Returns:
        DataLoader
    """
    # 加载数据集列表
    dataset_names = load_dataset_list(dataset_list_file)
    print(f"📂 从 {dataset_list_file} 加载 {len(dataset_names)} 个数据集")
    
    # 创建组合数据集
    dataset = UCRMultiDatasetForPretrain(
        dataset_names=dataset_names,
        split=split,
        raw_data_path=raw_data_path,
        patch_size=patch_size,
    )
    
    # 创建DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn_pretrain(batch, patch_size=patch_size),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return loader


# 测试
if __name__ == "__main__":
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_list = os.path.join(script_dir, "ucr_train_98_datasets.txt")
    
    if os.path.exists(train_list):
        loader = get_ucr_pretrain_loader(
            dataset_list_file=train_list,
            split="train",
            batch_size=32,
            patch_size=8,
        )
        
        batch = next(iter(loader))
        print(f"Batch shape: {batch.shape}")
        print(f"Batch mean: {batch.mean():.4f}, std: {batch.std():.4f}")
    else:
        print(f"数据集列表文件不存在: {train_list}")
