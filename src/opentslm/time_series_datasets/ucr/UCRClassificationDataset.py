# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR单数据集分类Dataset

用于M1实验：验证时序-LLM通路的有监督分类能力。
使用LLaVA范式（Soft Prompt）进行指令式分类。
标签映射为A, B, C, ...格式。
"""

import os
import string
from typing import List, Tuple, Literal, Optional
import pandas as pd
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.ucr.ucr_loader import load_ucr_dataset, ensure_ucr_data


def index_to_excel_label(index: int) -> str:
    """
    将整数索引转换为类似Excel列名的字母标签。
    
    映射规则：
    0-25: A, B, ..., Z
    26-51: AA, AB, ..., AZ
    52-77: BA, BB, ..., BZ
    ...
    
    Args:
        index: 非负整数索引 (从0开始)
    
    Returns:
        对应的字母标签
    
    Examples:
        >>> index_to_excel_label(0)
        'A'
        >>> index_to_excel_label(25)
        'Z'
        >>> index_to_excel_label(26)
        'AA'
        >>> index_to_excel_label(51)
        'AZ'
        >>> index_to_excel_label(52)
        'BA'
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    
    if index < 26:
        return chr(ord('A') + index)
    else:
        # 26-51 -> AA-AZ (prefix_idx=0, suffix_idx=0-25)
        # 52-77 -> BA-BZ (prefix_idx=1, suffix_idx=0-25)
        adjusted = index - 26
        prefix_idx = adjusted // 26
        suffix_idx = adjusted % 26
        return chr(ord('A') + prefix_idx) + chr(ord('A') + suffix_idx)


class UCRClassificationDataset(QADataset):
    """
    UCR单数据集分类Dataset
    
    Prompt格式：
    ─────────────────────────────────────
    You are a time series classifier for the {dataset_name} dataset.
    This dataset contains {num_classes} classes: A, B, C, ...
    Analyze the time series and output ONLY the single letter label.

    Time series:
    <TS_TOKENS>
    
    Label:
    ─────────────────────────────────────
    
    Answer: A (或B, C, ...)
    
    Args:
        dataset_name: UCR数据集名称 (e.g. "ECG5000")
        split: 数据划分 ("train", "validation", "test")
        EOS_TOKEN: 结束token
        raw_data_path: 数据路径
        val_ratio: 从训练集划分验证集的比例 (UCR没有官方验证集)
    """
    
    # 类变量存储数据集信息
    _dataset_name: str = None
    _label_to_letter: dict = None
    _letter_to_label: dict = None
    _num_classes: int = None
    _class_letters: List[str] = None
    
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
        
        # 创建标签到字母的映射 (0->A, 1->B, ... 26->AA, 27->AB, ...)
        letters = [index_to_excel_label(i) for i in range(num_classes)]
        label_to_letter = {label: letters[i] for i, label in enumerate(all_labels)}
        letter_to_label = {v: k for k, v in label_to_letter.items()}
        
        # 存储类变量
        UCRClassificationDataset._dataset_name = dataset_name
        UCRClassificationDataset._label_to_letter = label_to_letter
        UCRClassificationDataset._letter_to_label = letter_to_label
        UCRClassificationDataset._num_classes = num_classes
        UCRClassificationDataset._class_letters = letters
        
        print(f"📊 Dataset: {dataset_name}")
        print(f"   Classes: {num_classes}")
        print(f"   Label mapping: {label_to_letter}")
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
        dataset_name = UCRClassificationDataset._dataset_name
        num_classes = UCRClassificationDataset._num_classes
        class_letters = UCRClassificationDataset._class_letters
        
        classes_str = ", ".join(class_letters)
        
        prompt = f"""You are a time series classifier for the {dataset_name} dataset.
        This dataset contains {num_classes} classes: {classes_str}.
        Analyze the time series pattern and output ONLY the single letter label.

        Time series data:"""
        return prompt
    
    def _get_post_prompt(self, row) -> str:
        """返回后提示文本"""
        return "Label:"
    
    def _get_answer(self, row) -> str:
        """返回答案（字母标签）"""
        original_label = row["label"]
        letter_label = UCRClassificationDataset._label_to_letter[original_label]
        return letter_label
    
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
        sample["letter_label"] = UCRClassificationDataset._label_to_letter[row["label"]]
        return sample
    
    @staticmethod
    def get_labels() -> List[str]:
        """返回所有类别的字母标签"""
        return UCRClassificationDataset._class_letters or []
    
    @staticmethod
    def get_label_mapping() -> dict:
        """返回原始标签到字母的映射"""
        return UCRClassificationDataset._label_to_letter or {}
    
    @staticmethod
    def letter_to_original(letter: str) -> int:
        """将字母标签转换回原始标签"""
        return UCRClassificationDataset._letter_to_label.get(letter, -1)


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
    print(f"Labels: {UCRClassificationDataset.get_labels()}")
    print(f"Label mapping: {UCRClassificationDataset.get_label_mapping()}")
    
    # 查看样本
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n" + "="*50)
        print("Sample keys:", sample.keys())
        print("Pre-prompt:", sample["pre_prompt"])
        print("Post-prompt:", sample["post_prompt"])
        print("Answer:", sample["answer"])
        print("Letter label:", sample.get("letter_label", "N/A"))
        print("Original label:", sample.get("original_label", "N/A"))
        print("Time series text:", sample.get("time_series_text", ["N/A"])[0][:100] + "...")
