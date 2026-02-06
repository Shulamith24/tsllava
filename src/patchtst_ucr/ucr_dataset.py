# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
UCR Dataset for PatchTST 分类

简化的 Dataset 类，专门为 PatchTST 训练设计。
不依赖 opentslm 包。
"""

from typing import Dict, List, Literal, Optional, Tuple
import torch
from torch.utils.data import Dataset

from .ucr_loader import load_ucr_dataset


class UCRDatasetForPatchTST(Dataset):
    """
    UCR 数据集的 PyTorch Dataset 包装器
    
    专为 PatchTST 分类设计，返回:
    - time_series: 归一化的时间序列 tensor
    - int_label: 整数标签 (0 到 num_classes-1)
    
    Args:
        dataset_name: UCR数据集名称 (e.g. "ECG5000")
        split: 数据划分 ("train", "validation", "test")
        raw_data_path: 数据根目录
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Literal["train", "validation", "test"],
        raw_data_path: str = "./data",
    ):
        super().__init__()
        
        self.dataset_name = dataset_name
        self.split = split
        self.raw_data_path = raw_data_path
        
        # 加载数据
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path)
        
        # UCR没有官方验证集，validation 使用 test 数据
        if split == "train":
            self.df = train_df
        else:  # validation 或 test
            self.df = test_df
        
        # 获取特征列（除label外的所有列）
        self.feature_cols = [col for col in self.df.columns if col != "label"]
        
        # 创建标签映射 (原始标签 -> 0, 1, 2, ...)
        all_labels = sorted(train_df["label"].unique().tolist())
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(all_labels)
        
        # 转换为列表便于索引
        self.data = self.df.to_dict('records')
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        
        # 提取时间序列
        values = [row[col] for col in self.feature_cols]
        ts = torch.tensor(values, dtype=torch.float32)
        
        # 处理NaN
        ts = torch.nan_to_num(ts, nan=0.0)
        
        # Per-sample z-normalization
        mean = ts.mean()
        std = ts.std()
        if std > 1e-8:
            ts = (ts - mean) / std
        else:
            ts = ts - mean
        
        # 获取整数标签
        original_label = row["label"]
        int_label = self.label_to_idx[original_label]
        
        return {
            "time_series": [ts],  # 列表形式，与原代码兼容
            "int_label": int_label,
            "original_label": original_label,
        }
    
    def get_num_classes(self) -> int:
        """返回类别数量"""
        return self.num_classes
    
    def get_max_length(self) -> int:
        """返回时间序列最大长度"""
        return len(self.feature_cols)


def get_dataset_info(
    dataset_name: str,
    raw_data_path: str = "./data",
) -> Tuple[int, int]:
    """
    获取数据集统计信息
    
    Args:
        dataset_name: UCR数据集名称
        raw_data_path: 数据根目录
    
    Returns:
        (num_classes, max_length)
    """
    dataset = UCRDatasetForPatchTST(
        dataset_name=dataset_name,
        split="train",
        raw_data_path=raw_data_path,
    )
    return dataset.get_num_classes(), dataset.get_max_length()


# ---------------------------
# Test
# ---------------------------

if __name__ == "__main__":
    print("Testing UCRDatasetForPatchTST...")
    
    dataset = UCRDatasetForPatchTST(
        dataset_name="ECG200",
        split="train",
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Num classes: {dataset.get_num_classes()}")
    print(f"Max length: {dataset.get_max_length()}")
    print(f"Label mapping: {dataset.label_to_idx}")
    
    # 查看样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Time series shape: {sample['time_series'][0].shape}")
    print(f"Int label: {sample['int_label']}")
    print(f"Original label: {sample['original_label']}")
