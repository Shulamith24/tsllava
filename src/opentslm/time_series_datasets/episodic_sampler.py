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
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        
        # 获取每个数据集的样本索引
        self.ds_ids = dataset.get_all_ds_ids()
        self.ds_indices: Dict[int, List[int]] = {
            ds_id: dataset.get_indices_for_dataset(ds_id)
            for ds_id in self.ds_ids
        }
        
        # 将每个数据集的索引分片给当前rank
        if world_size > 1:
            for ds_id in self.ds_ids:
                indices = self.ds_indices[ds_id]
                # 简单的分片策略：按顺序分配
                # 更好的策略可能是先shuffle再分配，但为了确定性，这里保持简单
                # 确保每个rank分到的数据尽可能均匀
                total_size = len(indices)
                per_rank = int(math.ceil(total_size / world_size))
                start = rank * per_rank
                end = min(start + per_rank, total_size)
                self.ds_indices[ds_id] = indices[start:end]
        
        # 计算采样概率 (基于全局样本数还是本地样本数? 应该基于全局以保持分布一致)
        # 这里我们重新从registry获取全局样本数来计算概率
        registry = dataset.registry
        sample_counts = registry.get_sample_counts()
        # 只保留当前存在的ds_id
        sample_counts = {ds_id: sample_counts[ds_id] for ds_id in self.ds_ids}
        self._sample_counts = sample_counts
        
        # p(i) ∝ n_i^α
        weights = {ds_id: math.pow(count, alpha) for ds_id, count in sample_counts.items()}
        total_weight = sum(weights.values())
        self.ds_probs = {ds_id: w / total_weight for ds_id, w in weights.items()}
        
        # 打印采样概率 (只在rank 0打印)
        if rank == 0:
            print(f"📊 EpisodicBatchSampler (α={alpha}, world_size={world_size}):")
            for ds_id in self.ds_ids:
                ds_name = registry.get_dataset_info(ds_id).name
                prob = self.ds_probs[ds_id]
                count = sample_counts[ds_id]
                local_count = len(self.ds_indices[ds_id])
                print(f"   [{ds_id}] {ds_name}: {count} global samples ({local_count} local), p={prob:.3f}")
        
        # 计算episode数（默认：总样本数 / batch_size / world_size）
        if num_episodes is None:
            total_samples = len(dataset)
            self.num_episodes = max(1, total_samples // batch_size // world_size)
        else:
            self.num_episodes = num_episodes
        
        if rank == 0:
            print(f"   Episodes per epoch: {self.num_episodes}")
        
        # 内部状态
        self._ds_id_list = list(self.ds_ids)
        self._prob_list = [self.ds_probs[ds_id] for ds_id in self._ds_id_list]
    
    def set_epoch(self, epoch: int):
        """设置当前epoch，用于更新随机种子"""
        self.epoch = epoch
    
    def __iter__(self) -> Iterator[List[int]]:
        """生成batch索引"""
        # 1. 确定性地生成本epoch的数据集序列
        # 使用独立的RNG，种子为 seed + epoch
        # 这样所有rank生成的序列是一样的
        rng_ds = random.Random(self.seed + self.epoch)
        
        # 2. 准备本地数据的索引队列
        ds_queues: Dict[int, List[int]] = {}
        
        # 使用本地RNG（可以是全局random）来shuffle数据索引
        # 每个rank的shuffle应该是不同的（因为数据不同，且通常希望随机性）
        # 如果需要完全可复现，可以使用 random.Random(self.seed + self.rank + self.epoch)
        rng_data = random.Random(self.seed + self.rank + self.epoch)
        
        if self.shuffle:
            for ds_id, indices in self.ds_indices.items():
                shuffled = indices.copy()
                rng_data.shuffle(shuffled)
                ds_queues[ds_id] = shuffled
        else:
            ds_queues = {ds_id: indices.copy() for ds_id, indices in self.ds_indices.items()}
        
        for _ in range(self.num_episodes):
            # 1. 采样数据集 (所有rank相同)
            ds_id = rng_ds.choices(self._ds_id_list, weights=self._prob_list, k=1)[0]
            
            # 2. 从该数据集采样batch (各rank不同)
            queue = ds_queues[ds_id]
            
            # 如果队列不足，重新填充
            if len(queue) < self.batch_size:
                # 重新获取并shuffle
                new_indices = self.ds_indices[ds_id].copy()
                if not new_indices: # 防止空数据集死循环
                     yield [] 
                     continue

                if self.shuffle:
                    rng_data.shuffle(new_indices)
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
