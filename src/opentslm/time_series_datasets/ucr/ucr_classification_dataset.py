# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR分类数据集 - 用于M1生成式分类实验

Prompt格式: Dataset=<name>. Classes: <cls_0>, <cls_1>, ... Predict label:
Answer: <cls_i>
"""

import os
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import torch
from torch.utils.data import Dataset

from opentslm.time_series_datasets.ucr.ucr_loader import (
    ensure_ucr_data,
    load_ucr_dataset,
)


def get_class_tokens(num_classes: int) -> List[str]:
    """生成类别token列表"""
    return [f"<cls_{i}>" for i in range(num_classes)]


class UCRClassificationDataset(Dataset):
    """
    UCR单数据集分类Dataset
    
    用于生成式分类训练，prompt格式:
    Dataset=<name>. Classes: <cls_0>, <cls_1>, ... Predict label:
    
    Answer: <cls_i>
    
    Args:
        dataset_name: UCR数据集名称
        split: "train", "test"
        raw_data_path: UCR数据根目录
        EOS_TOKEN: 结束token
        normalize: 是否进行z-normalization
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Literal["train", "test"] = "train",
        raw_data_path: str = "./data",
        EOS_TOKEN: str = "</s>",
        normalize: bool = True,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.EOS_TOKEN = EOS_TOKEN
        self.normalize = normalize
        
        # 确保数据已下载
        ensure_ucr_data()
        
        # 加载数据
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        
        if split == "train":
            self.df = train_df
        else:
            self.df = test_df
        
        self.df = self.df.reset_index(drop=True)
        
        # 获取特征列
        self.feature_cols = [c for c in self.df.columns if c != "label"]
        
        # 获取类别信息
        all_labels = pd.concat([train_df["label"], test_df["label"]]).unique()
        self.label_mapping = {orig: idx for idx, orig in enumerate(sorted(all_labels))}
        self.num_classes = len(self.label_mapping)
        self.class_tokens = get_class_tokens(self.num_classes)
        
        # 生成类别token字符串
        self.class_tokens_str = ", ".join(self.class_tokens)
        
        print(f"📂 加载 {dataset_name} ({split}): {len(self.df)} 样本, {self.num_classes} 类别")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 获取时间序列
        feats = row[self.feature_cols].astype(float).values
        tensor = torch.tensor(feats, dtype=torch.float32)
        
        # 处理NaN
        tensor = torch.nan_to_num(tensor, nan=0.0)
        
        # Z-normalization
        if self.normalize:
            mean = tensor.mean()
            std = tensor.std()
            if std > 1e-8:
                tensor = (tensor - mean) / std
            else:
                tensor = tensor - mean
        
        # 获取标签
        orig_label = row["label"]
        label_idx = self.label_mapping[orig_label]
        label_token = self.class_tokens[label_idx]
        
        # 构建prompt
        pre_prompt = f"Dataset={self.dataset_name}. Classes: {self.class_tokens_str}. "
        time_series_text = ["Time series: "]
        post_prompt = "Predict label:"
        answer = f" {label_token}{self.EOS_TOKEN}"
        
        return {
            "pre_prompt": pre_prompt,
            "time_series_text": time_series_text,
            "time_series": [tensor],
            "post_prompt": post_prompt,
            "answer": answer,
            "label_idx": label_idx,  # 用于评估
        }
    
    def get_class_tokens(self) -> List[str]:
        """返回类别token列表"""
        return self.class_tokens
    
    def get_num_classes(self) -> int:
        """返回类别数量"""
        return self.num_classes


def collate_fn_classification(
    batch: List[Dict[str, Any]], 
    patch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    分类任务的collate函数
    
    将时间序列填充到patch_size的倍数
    """
    for sample in batch:
        for i, ts in enumerate(sample["time_series"]):
            L = ts.shape[0]
            if L % patch_size != 0:
                pad_len = patch_size - (L % patch_size)
                sample["time_series"][i] = torch.nn.functional.pad(ts, (0, pad_len))
    return batch


if __name__ == "__main__":
    # 测试
    dataset = UCRClassificationDataset("ECG5000", split="train")
    print(f"\n样本0:")
    sample = dataset[0]
    print(f"  pre_prompt: {sample['pre_prompt']}")
    print(f"  time_series shape: {sample['time_series'][0].shape}")
    print(f"  post_prompt: {sample['post_prompt']}")
    print(f"  answer: {sample['answer']}")
    print(f"  label_idx: {sample['label_idx']}")
