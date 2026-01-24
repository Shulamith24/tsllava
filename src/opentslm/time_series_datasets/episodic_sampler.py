# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
Episodic Batch Sampler

一个batch一个dataset，使用温度采样平衡大小差异：
p(i) ∝ n_i^α，其中 n_i 是数据集 i 的样本数

Args:
    alpha: 温度采样参数
           α=0: 完全均匀（小数据集不会被淹没）
           α=0.3~0.5: 常用折中
           α=1: 按数据量（大数据集主导）
"""

import math
import random
from typing import Iterator, List, Dict, Optional

from torch.utils.data import Sampler

from .multi_dataset import UnifiedPrototypeDataset


class EpisodicBatchSampler(Sampler):
    """
    Episodic采样器：一个batch一个dataset
    
    两层采样：
    1. 先采 dataset i ~ p(i)，其中 p(i) ∝ n_i^α
    2. 再从 dataset i 随机采一个 batch
    
    Args:
        dataset: UnifiedPrototypeDataset
        batch_size: 每个batch的样本数
        alpha: 温度采样参数，默认0.4
        num_episodes: 每个epoch的episode数（默认自动计算）
        shuffle: 是否在每个epoch开始时打乱
        drop_last: 是否丢弃最后不足batch_size的batch
    """
    
    def __init__(
        self,
        dataset: UnifiedPrototypeDataset,
        batch_size: int,
        alpha: float = 0.4,
        num_episodes: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # 获取每个数据集的样本索引
        self.ds_ids = dataset.get_all_ds_ids()
        self.ds_indices: Dict[int, List[int]] = {
            ds_id: dataset.get_indices_for_dataset(ds_id)
            for ds_id in self.ds_ids
        }
        
        # 计算采样概率
        sample_counts = {ds_id: len(indices) for ds_id, indices in self.ds_indices.items()}
        self._sample_counts = sample_counts
        
        # p(i) ∝ n_i^α
        weights = {ds_id: math.pow(count, alpha) for ds_id, count in sample_counts.items()}
        total_weight = sum(weights.values())
        self.ds_probs = {ds_id: w / total_weight for ds_id, w in weights.items()}
        
        # 打印采样概率
        print(f"📊 EpisodicBatchSampler (α={alpha}):")
        for ds_id in self.ds_ids:
            ds_name = dataset.registry.get_dataset_info(ds_id).name
            prob = self.ds_probs[ds_id]
            count = sample_counts[ds_id]
            print(f"   [{ds_id}] {ds_name}: {count} samples, p={prob:.3f}")
        
        # 计算episode数（默认：总样本数 / batch_size）
        if num_episodes is None:
            self.num_episodes = max(1, len(dataset) // batch_size)
        else:
            self.num_episodes = num_episodes
        
        print(f"   Episodes per epoch: {self.num_episodes}")
        
        # 内部状态
        self._ds_id_list = list(self.ds_ids)
        self._prob_list = [self.ds_probs[ds_id] for ds_id in self._ds_id_list]
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成batch索引"""
        # 每个数据集维护一个打乱的索引队列
        ds_queues: Dict[int, List[int]] = {}
        
        if self.shuffle:
            for ds_id, indices in self.ds_indices.items():
                shuffled = indices.copy()
                random.shuffle(shuffled)
                ds_queues[ds_id] = shuffled
        else:
            ds_queues = {ds_id: indices.copy() for ds_id, indices in self.ds_indices.items()}
        
        for _ in range(self.num_episodes):
            # 1. 采样数据集
            ds_id = random.choices(self._ds_id_list, weights=self._prob_list, k=1)[0]
            
            # 2. 从该数据集采样batch
            queue = ds_queues[ds_id]
            
            # 如果队列不足，重新填充
            if len(queue) < self.batch_size:
                new_indices = self.ds_indices[ds_id].copy()
                if self.shuffle:
                    random.shuffle(new_indices)
                queue.extend(new_indices)
            
            # 取batch
            batch = queue[:self.batch_size]
            ds_queues[ds_id] = queue[self.batch_size:]
            
            if self.drop_last and len(batch) < self.batch_size:
                continue
            
            yield batch
    
    def __len__(self) -> int:
        return self.num_episodes
    
    def get_dataset_sampling_stats(self) -> Dict[str, float]:
        """获取数据集采样统计（用于分析）"""
        return {
            self.dataset.registry.get_dataset_info(ds_id).name: prob
            for ds_id, prob in self.ds_probs.items()
        }


# 测试
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
    
    from opentslm.time_series_datasets.multi_dataset import (
        MultiDatasetRegistry, 
        UnifiedPrototypeDataset
    )
    
    print("Testing EpisodicBatchSampler...")
    
    registry = MultiDatasetRegistry(data_path="./data")
    registry.register("ECG200")
    registry.register("Coffee")
    
    dataset = UnifiedPrototypeDataset(registry, split="train")
    sampler = EpisodicBatchSampler(dataset, batch_size=8, alpha=0.4, num_episodes=10)
    
    print("\nSampling 5 episodes:")
    ds_count = {}
    for i, batch_indices in enumerate(sampler):
        if i >= 5:
            break
        # 检查同一batch是否来自同一数据集
        ds_ids = set(dataset[idx]["ds_id"] for idx in batch_indices)
        assert len(ds_ids) == 1, "Batch should contain samples from single dataset"
        ds_id = list(ds_ids)[0]
        ds_name = registry.get_dataset_info(ds_id).name
        ds_count[ds_name] = ds_count.get(ds_name, 0) + 1
        print(f"Episode {i}: {len(batch_indices)} samples from {ds_name}")
    
    print(f"\nDataset distribution: {ds_count}")
