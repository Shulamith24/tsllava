# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
多数据集注册中心 + 统一数据集包装器

用于多数据集统一训练：
1. MultiDatasetRegistry: 管理多个UCR数据集的元信息
2. UnifiedPrototypeDataset: 统一getitem输出格式
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from opentslm.time_series_datasets.ucr.UCRClassificationDataset import UCRClassificationDataset
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data


@dataclass
class DatasetInfo:
    """数据集元信息"""
    ds_id: int              # 数据集唯一ID (0-indexed)
    name: str               # 数据集名称
    num_classes: int        # 类别数
    num_train_samples: int  # 训练样本数
    num_test_samples: int   # 测试样本数
    label_to_idx: Dict      # 原始标签 → 类别索引 (0-indexed)
    idx_to_label: Dict      # 类别索引 → 原始标签


class MultiDatasetRegistry:
    """
    多数据集注册中心
    
    从配置文件读取数据集列表，加载每个数据集的元信息。
    
    使用方法:
        registry = MultiDatasetRegistry()
        registry.load_from_file("configs/multi_dataset_ucr.txt")
        print(registry.get_total_datasets())
    """
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = data_path
        self._datasets: Dict[int, DatasetInfo] = {}
        self._name_to_id: Dict[str, int] = {}
        self._next_id = 0
    
    def load_from_file(self, config_path: str) -> None:
        """
        从配置文件加载数据集列表
        
        配置文件格式：每行一个数据集名称，#开头为注释
        """
        ensure_ucr_data()
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            self.register(line)
        
        print(f"📚 Registered {self.get_total_datasets()} datasets from {config_path}")
    
    def register(self, dataset_name: str) -> DatasetInfo:
        """注册单个数据集"""
        if dataset_name in self._name_to_id:
            return self._datasets[self._name_to_id[dataset_name]]
        
        # 加载数据集获取元信息
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=self.data_path)
        
        # 获取标签映射
        all_labels = sorted(train_df["label"].unique().tolist())
        label_to_idx = {label: i for i, label in enumerate(all_labels)}
        idx_to_label = {i: label for label, i in label_to_idx.items()}
        
        info = DatasetInfo(
            ds_id=self._next_id,
            name=dataset_name,
            num_classes=len(all_labels),
            num_train_samples=len(train_df),
            num_test_samples=len(test_df),
            label_to_idx=label_to_idx,
            idx_to_label=idx_to_label,
        )
        
        self._datasets[self._next_id] = info
        self._name_to_id[dataset_name] = self._next_id
        self._next_id += 1
        
        print(f"   [{info.ds_id}] {dataset_name}: {info.num_classes} classes, "
              f"{info.num_train_samples} train / {info.num_test_samples} test")
        
        return info
    
    def get_dataset_info(self, ds_id: int) -> DatasetInfo:
        """获取数据集元信息"""
        return self._datasets[ds_id]
    
    def get_dataset_by_name(self, name: str) -> DatasetInfo:
        """按名称获取数据集"""
        return self._datasets[self._name_to_id[name]]
    
    def get_all_datasets(self) -> List[DatasetInfo]:
        """获取所有数据集信息"""
        return [self._datasets[i] for i in range(self._next_id)]
    
    def get_total_datasets(self) -> int:
        """获取数据集总数"""
        return self._next_id
    
    def get_class_counts(self) -> Dict[int, int]:
        """获取每个数据集的类别数 {ds_id: num_classes}"""
        return {ds_id: info.num_classes for ds_id, info in self._datasets.items()}
    
    def get_max_classes(self) -> int:
        """获取所有数据集中最大类别数"""
        return max(info.num_classes for info in self._datasets.values())
    
    def get_sample_counts(self) -> Dict[int, int]:
        """获取每个数据集的训练样本数 {ds_id: num_train_samples}"""
        return {ds_id: info.num_train_samples for ds_id, info in self._datasets.items()}


class UnifiedPrototypeDataset(Dataset):
    """
    统一Prototype数据集
    
    将多个UCR数据集合并为一个Dataset，每个样本包含 ds_id 标识。
    
    输出格式:
        {
            "time_series": [Tensor],  # 时间序列列表
            "label_index": int,       # 类别索引 (该数据集内部0-indexed)
            "ds_id": int,             # 数据集ID
            "ds_name": str,           # 数据集名称
            "_global_idx": int,       # 全局索引
        }
    """
    
    def __init__(
        self,
        registry: MultiDatasetRegistry,
        split: str = "train",
        eos_token: str = "<eos>",
    ):
        self.registry = registry
        self.split = split
        self.eos_token = eos_token
        
        # 加载所有数据集
        self._samples: List[Dict] = []
        self._ds_indices: Dict[int, List[int]] = {}  # ds_id → [sample indices]
        
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """加载所有数据集"""
        ensure_ucr_data()
        
        for ds_info in self.registry.get_all_datasets():
            start_idx = len(self._samples)
            
            # 加载数据
            train_df, test_df = load_ucr_dataset(
                ds_info.name, 
                raw_data_path=self.registry.data_path
            )
            
            # 选择split
            if self.split == "train":
                df = train_df
            else:  # validation/test 使用相同数据
                df = test_df
            
            # 转换为样本列表
            for _, row in df.iterrows():
                # 提取时间序列
                feature_cols = [col for col in row.index if col != "label"]
                values = [row[col] for col in feature_cols]
                ts_tensor = torch.tensor(values, dtype=torch.float32)
                ts_tensor = torch.nan_to_num(ts_tensor, nan=0.0)
                
                # Z-normalization
                mean = ts_tensor.mean()
                std = ts_tensor.std()
                if std > 1e-8:
                    ts_tensor = (ts_tensor - mean) / std
                else:
                    ts_tensor = ts_tensor - mean
                
                # 获取标签索引
                label_index = ds_info.label_to_idx[row["label"]]
                
                self._samples.append({
                    "time_series": [ts_tensor],
                    "label_index": label_index,
                    "ds_id": ds_info.ds_id,
                    "ds_name": ds_info.name,
                    "_original_label": row["label"],
                })
            
            end_idx = len(self._samples)
            self._ds_indices[ds_info.ds_id] = list(range(start_idx, end_idx))
        
        print(f"📦 UnifiedPrototypeDataset ({self.split}): {len(self._samples)} samples total")
    
    def __len__(self):
        return len(self._samples)
    
    def __getitem__(self, idx):
        sample = self._samples[idx].copy()
        sample["_global_idx"] = idx
        return sample
    
    def get_indices_for_dataset(self, ds_id: int) -> List[int]:
        """获取指定数据集的所有样本索引"""
        return self._ds_indices.get(ds_id, [])
    
    def get_all_ds_ids(self) -> List[int]:
        """获取所有数据集ID"""
        return list(self._ds_indices.keys())


# 测试
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    
    print("Testing MultiDatasetRegistry...")
    
    registry = MultiDatasetRegistry(data_path="./data")
    registry.register("ECG200")
    registry.register("Coffee")
    
    print(f"\nTotal datasets: {registry.get_total_datasets()}")
    print(f"Class counts: {registry.get_class_counts()}")
    
    print("\nTesting UnifiedPrototypeDataset...")
    dataset = UnifiedPrototypeDataset(registry, split="train")
    
    print(f"Total samples: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"ds_id: {sample['ds_id']}, ds_name: {sample['ds_name']}")
    print(f"label_index: {sample['label_index']}")
    print(f"time_series shape: {sample['time_series'][0].shape}")
