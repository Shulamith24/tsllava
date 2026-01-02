# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR单数据集分类任务Dataset

用于M1实验：验证时间序列→LLM通路的分类能力。
使用特殊标签token (<cls_0>, <cls_1>, ...) 进行分类。
"""

import os
from typing import List, Tuple, Literal, Optional, Dict, Any
import pandas as pd
import torch
from torch.utils.data import Dataset

from opentslm.time_series_datasets.ucr.ucr_loader import (
    ensure_ucr_data,
    load_ucr_dataset,
)


class UCRClassificationDataset(Dataset):
    """
    UCR单数据集分类任务Dataset
    
    特性：
    - 动态加载任意UCR数据集
    - 使用类别专用token（<cls_0>, <cls_1>, ...）作为标签
    - 返回格式与OpenTSLMSP兼容
    
    Args:
        dataset_name: UCR数据集名称，如"ECG5000"
        split: "train", "test", 或 "validation"
        EOS_TOKEN: 结束token
        cls_tokens: 类别token列表，如["<cls_0>", "<cls_1>", ...]
        raw_data_path: UCR数据根目录
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        cls_tokens: List[str],
        raw_data_path: str = "./data",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.EOS_TOKEN = EOS_TOKEN
        self.cls_tokens = cls_tokens
        self.raw_data_path = raw_data_path
        
        # 确保数据已下载
        ensure_ucr_data()
        
        # 加载数据
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        
        # 分割数据
        if split == "train":
            # 使用80%作为训练集
            n_train = int(len(train_df) * 0.8)
            self.df = train_df.iloc[:n_train].reset_index(drop=True)
        elif split == "validation":
            # 使用20%作为验证集
            n_train = int(len(train_df) * 0.8)
            self.df = train_df.iloc[n_train:].reset_index(drop=True)
        else:  # test
            self.df = test_df.reset_index(drop=True)
        
        # 获取特征列（排除label列）
        self.feature_cols = [c for c in self.df.columns if c != "label"]
        
        # 构建标签映射 (原始标签 -> 连续索引)
        all_labels = pd.concat([train_df["label"], test_df["label"]]).unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # 验证cls_tokens数量
        if len(cls_tokens) != self.num_classes:
            raise ValueError(
                f"cls_tokens数量({len(cls_tokens)})与数据集类别数({self.num_classes})不匹配"
            )
        
        print(f"📂 加载 {dataset_name} {split}集: {len(self.df)} 样本, {self.num_classes} 类别")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # 获取时间序列数据
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
        
        # 获取标签
        original_label = row["label"]
        label_idx = self.label_to_idx[original_label]
        cls_token = self.cls_tokens[label_idx]
        
        # 构建prompt
        pre_prompt = self._get_pre_prompt()
        post_prompt = self._get_post_prompt()
        answer = f"{cls_token}{self.EOS_TOKEN}"
        
        # 返回OpenTSLMSP兼容格式
        return {
            "pre_prompt": pre_prompt,
            "time_series_text": [f"The following is the time series data (mean={mean:.4f}, std={std:.4f}):"],
            "time_series": [tensor],
            "post_prompt": post_prompt,
            "answer": answer,
            "label_idx": label_idx,  # 用于评估
            "original_label": original_label,  # 原始标签
        }
    
    def _get_pre_prompt(self) -> str:
        """生成预提示"""
        label_list = ", ".join(self.cls_tokens)
        prompt = f"""You are a time series classifier for the {self.dataset_name} dataset.
This dataset has {self.num_classes} classes.

Your task is to analyze the time series pattern and classify it into one of the following labels:
{label_list}

"""
        return prompt.strip()
    
    def _get_post_prompt(self) -> str:
        """生成后提示"""
        return "Based on the patterns in the time series above, output ONLY the label token.\n\nLabel:"
    
    @staticmethod
    def get_class_tokens(num_classes: int) -> List[str]:
        """生成类别token列表"""
        return [f"<cls_{i}>" for i in range(num_classes)]
    
    @staticmethod
    def get_num_classes(dataset_name: str, raw_data_path: str = "./data") -> int:
        """获取数据集类别数"""
        ensure_ucr_data()
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        all_labels = pd.concat([train_df["label"], test_df["label"]]).unique()
        return len(all_labels)
    
    def get_label_mapping(self) -> Dict[str, int]:
        """返回原始标签到索引的映射"""
        return self.label_to_idx.copy()


def collate_fn_classification(batch: List[Dict[str, Any]], patch_size: int = 8):
    """
    分类任务的collate函数
    
    填充时间序列到patch_size的倍数
    """
    # 找到最大长度
    max_len = max(sample["time_series"][0].shape[0] for sample in batch)
    
    # 填充到patch_size的倍数
    if max_len % patch_size != 0:
        max_len = max_len + (patch_size - max_len % patch_size)
    
    # 填充每个样本的时间序列
    for sample in batch:
        ts = sample["time_series"][0]
        if ts.shape[0] < max_len:
            pad_len = max_len - ts.shape[0]
            sample["time_series"][0] = torch.nn.functional.pad(ts, (0, pad_len))
    
    return batch


# 测试
if __name__ == "__main__":
    # 测试数据集
    dataset_name = "ECG5000"
    num_classes = UCRClassificationDataset.get_num_classes(dataset_name)
    cls_tokens = UCRClassificationDataset.get_class_tokens(num_classes)
    
    print(f"\n数据集: {dataset_name}")
    print(f"类别数: {num_classes}")
    print(f"类别tokens: {cls_tokens}")
    
    # 创建数据集
    train_ds = UCRClassificationDataset(
        dataset_name=dataset_name,
        split="train",
        EOS_TOKEN="</s>",
        cls_tokens=cls_tokens,
    )
    
    val_ds = UCRClassificationDataset(
        dataset_name=dataset_name,
        split="validation",
        EOS_TOKEN="</s>",
        cls_tokens=cls_tokens,
    )
    
    test_ds = UCRClassificationDataset(
        dataset_name=dataset_name,
        split="test",
        EOS_TOKEN="</s>",
        cls_tokens=cls_tokens,
    )
    
    print(f"\n训练集: {len(train_ds)} 样本")
    print(f"验证集: {len(val_ds)} 样本")
    print(f"测试集: {len(test_ds)} 样本")
    
    # 查看一个样本
    sample = train_ds[0]
    print(f"\n样本keys: {sample.keys()}")
    print(f"时间序列形状: {sample['time_series'][0].shape}")
    print(f"标签索引: {sample['label_idx']}")
    print(f"原始标签: {sample['original_label']}")
    print(f"Answer: {sample['answer']}")
    print(f"\nPre-prompt:\n{sample['pre_prompt'][:200]}...")
    print(f"\nPost-prompt:\n{sample['post_prompt']}")
