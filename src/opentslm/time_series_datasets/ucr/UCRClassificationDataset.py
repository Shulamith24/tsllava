# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR单数据集分类Dataset

用于M1实验：验证时序-LLM通路的有监督分类能力。
使用LLaVA范式（Soft Prompt）进行指令式分类。
标签映射为特殊token格式: <c0>, <c1>, ...
"""

import os
import string
from typing import List, Tuple, Literal, Optional
import pandas as pd
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data


def index_to_class_token(index: int) -> str:
    """
    将整数索引转换为特殊类别token。
    
    映射规则：
    0 -> <c0>
    1 -> <c1>
    ...
    K-1 -> <cK-1>
    
    Args:
        index: 非负整数索引 (从0开始)
    
    Returns:
        对应的特殊类别token
    
    Examples:
        >>> index_to_class_token(0)
        '<c0>'
        >>> index_to_class_token(5)
        '<c5>'
        >>> index_to_class_token(25)
        '<c25>'
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    return f"<c{index}>"


class UCRClassificationDataset(QADataset):
    """
    UCR单数据集分类Dataset
    
    Prompt格式：
    ─────────────────────────────────────
    Classify the time series into one of {num_classes} classes.
    Output only the class token.

    Time series:
    <TS_TOKENS>
    
    Class:
    ─────────────────────────────────────
    
    Answer: <c0> (或 <c1>, <c2>, ...)
    
    Args:
        dataset_name: UCR数据集名称 (e.g. "ECG5000")
        split: 数据划分 ("train", "validation", "test")
        EOS_TOKEN: 结束token
        raw_data_path: 数据路径
        val_ratio: 从训练集划分验证集的比例 (UCR没有官方验证集)
    """
    
    # 类变量存储数据集信息
    _dataset_name: str = None
    _label_to_token: dict = None
    _token_to_label: dict = None
    _label_to_index: dict = None  # 原始标签到索引(0, 1, 2, ...)的映射
    _num_classes: int = None
    _class_tokens: List[str] = None
    
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        dataset_name: str = "ECG5000",
        raw_data_path: str = "./data",
        val_ratio: float = 0.1,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        # 存储实例变量
        self._instance_dataset_name = dataset_name
        self._instance_raw_data_path = raw_data_path
        self._instance_val_ratio = val_ratio
        
        # 调用父类初始化
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)
    
    def _load_splits(self) -> Tuple[List, List, List]:
        """
        加载UCR数据集
        
        UCR只有train和test，直接使用test作为validation（不从训练集划分）
        """
        ensure_ucr_data()
        
        dataset_name = self._instance_dataset_name
        raw_data_path = self._instance_raw_data_path
        
        # 加载数据
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=raw_data_path)
        
        # 获取所有唯一标签并排序
        all_labels = sorted(train_df["label"].unique().tolist())
        num_classes = len(all_labels)
        
        # 创建标签到特殊token的映射 (0-><c0>, 1-><c1>, ...)
        tokens = [index_to_class_token(i) for i in range(num_classes)]
        label_to_token = {label: tokens[i] for i, label in enumerate(all_labels)}
        token_to_label = {v: k for k, v in label_to_token.items()}
        # 创建标签到索引的映射（用于分类头）
        label_to_index = {label: i for i, label in enumerate(all_labels)}
        
        # 存储类变量
        UCRClassificationDataset._dataset_name = dataset_name
        UCRClassificationDataset._label_to_token = label_to_token
        UCRClassificationDataset._token_to_label = token_to_label
        UCRClassificationDataset._label_to_index = label_to_index
        UCRClassificationDataset._num_classes = num_classes
        UCRClassificationDataset._class_tokens = tokens
        
        print(f"📊 Dataset: {dataset_name}")
        print(f"   Classes: {num_classes}")
        print(f"   Label mapping: {label_to_token}")
        print(f"   Train samples: {len(train_df)}")
        print(f"   Test samples: {len(test_df)}")
        print(f"   (Validation = Test)")
        
        # 转换为列表形式
        train_list = train_df.to_dict('records')
        # validation和test使用相同的数据
        val_list = test_df.to_dict('records')
        test_list = test_df.to_dict('records')
        
        return train_list, val_list, test_list
    
    def _get_pre_prompt(self, row) -> str:
        """返回预提示文本"""
        num_classes = UCRClassificationDataset._num_classes
        
        prompt = f"""Classify the time series into one of {num_classes} classes.
Output only the class token.

Time series data:"""
        return prompt
    
    def _get_post_prompt(self, row) -> str:
        """返回后提示文本"""
        return "\nClass:"
    
    def _get_answer(self, row) -> str:
        """返回答案（特殊类别token）"""
        original_label = row["label"]
        class_token = UCRClassificationDataset._label_to_token[original_label]
        return class_token
    
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """将时间序列转换为TextTimeSeriesPrompt列表"""
        # 提取时间序列数据 (除了label列的所有列)
        feature_cols = [col for col in row.keys() if col != "label"]
        values = [row[col] for col in feature_cols]
        
        # 转换为tensor
        tensor = torch.tensor(values, dtype=torch.float32)
        
        # 处理NaN值
        tensor = torch.nan_to_num(tensor, nan=0.0)
        
        # Per-sample z-normalization
        mean = tensor.mean()
        std = tensor.std()
        if std > 1e-8:
            tensor = (tensor - mean) / std
        else:
            tensor = tensor - mean
        
        # 创建prompt (简单描述)
        # text_prompt = f"This is a univariate time series with {len(tensor)} data points, mean={mean:.4f}, std={std:.4f}:"
        text_prompt = f"This is a univariate time series with {len(tensor)} data points:"
        
        return [TextTimeSeriesPrompt(text_prompt, tensor.tolist())]
    
    def _format_sample(self, row):
        """格式化样本，添加额外信息"""
        sample = super()._format_sample(row)
        # 保存原始标签用于评估
        sample["original_label"] = row["label"]
        sample["label_index"] = UCRClassificationDataset._label_to_index[row["label"]]
        sample["class_token"] = UCRClassificationDataset._label_to_token[row["label"]]
        return sample
    
    @staticmethod
    def get_class_tokens() -> List[str]:
        """返回所有类别的特殊token"""
        return UCRClassificationDataset._class_tokens or []
    
    @staticmethod
    def get_num_classes() -> int:
        """返回类别数量"""
        return UCRClassificationDataset._num_classes or 0
    
    @staticmethod
    def get_label_mapping() -> dict:
        """返回原始标签到特殊token的映射"""
        return UCRClassificationDataset._label_to_token or {}
    
    @staticmethod
    def token_to_original(token: str) -> int:
        """将特殊token转换回原始标签"""
        return UCRClassificationDataset._token_to_label.get(token, -1)


# 测试
if __name__ == "__main__":
    # 测试数据集加载
    print("Testing UCRClassificationDataset...")
    
    dataset = UCRClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name="ECG200",
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Class tokens: {UCRClassificationDataset.get_class_tokens()}")
    print(f"Label mapping: {UCRClassificationDataset.get_label_mapping()}")
    
    # 查看样本
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n" + "="*50)
        print("Sample keys:", sample.keys())
        print("Pre-prompt:", sample["pre_prompt"])
        print("Post-prompt:", sample["post_prompt"])
        print("Answer:", sample["answer"])
        print("Class token:", sample.get("class_token", "N/A"))
        print("Original label:", sample.get("original_label", "N/A"))
        print("Time series text:", sample.get("time_series_text", ["N/A"])[0][:100] + "...")
