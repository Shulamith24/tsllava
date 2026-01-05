# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UEA多变量时间序列分类Dataset

用于M1实验：验证时序-LLM通路的有监督分类能力。
采用与HAR/PAMAP2相同的多通道处理方式。
"""

import string
from typing import List, Tuple, Literal
import numpy as np
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.time_series_datasets.uea.uea_loader import load_uea_dataset, ensure_uea_data


class UEAClassificationDataset(QADataset):
    """
    UEA多变量时间序列分类Dataset
    
    采用与HAR/PAMAP2相同的多通道处理方式：
    - 每个通道创建独立的TextTimeSeriesPrompt
    - 每个通道独立z-normalization
    - 标签映射为A, B, C, ...
    
    Args:
        dataset_name: UEA数据集名称 (e.g. "Handwriting")
        split: 数据划分 ("train", "validation", "test")
        EOS_TOKEN: 结束token
    """
    
    # 类变量存储数据集信息
    _dataset_name: str = None
    _label_to_letter: dict = None
    _letter_to_label: dict = None
    _num_classes: int = None
    _num_channels: int = None
    _class_letters: List[str] = None
    
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        dataset_name: str = "Handwriting",
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        # 存储实例变量
        self._instance_dataset_name = dataset_name
        
        # 调用父类初始化
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)
    
    def _load_splits(self) -> Tuple[List, List, List]:
        """
        加载UEA数据集
        
        UEA只有train和test，使用test作为validation
        """
        ensure_uea_data()
        
        dataset_name = self._instance_dataset_name
        
        # 加载数据
        X_train, y_train, X_test, y_test = load_uea_dataset(dataset_name)
        
        # 获取所有唯一标签并排序
        all_labels = sorted(np.unique(y_train).tolist())
        num_classes = len(all_labels)
        num_channels = X_train.shape[1]
        
        # 创建标签到字母的映射
        letters = list(string.ascii_uppercase)[:num_classes]
        label_to_letter = {label: letters[i] for i, label in enumerate(all_labels)}
        letter_to_label = {v: k for k, v in label_to_letter.items()}
        
        # 存储类变量
        UEAClassificationDataset._dataset_name = dataset_name
        UEAClassificationDataset._label_to_letter = label_to_letter
        UEAClassificationDataset._letter_to_label = letter_to_label
        UEAClassificationDataset._num_classes = num_classes
        UEAClassificationDataset._num_channels = num_channels
        UEAClassificationDataset._class_letters = letters
        
        print(f"📊 Dataset: {dataset_name}")
        print(f"   Classes: {num_classes}")
        print(f"   Channels: {num_channels}")
        print(f"   Label mapping: {label_to_letter}")
        print(f"   Train samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   (Validation = Test)")
        
        # 转换为列表形式（每个样本是一个dict）
        train_list = self._convert_to_list(X_train, y_train)
        # validation和test使用相同的数据
        val_list = self._convert_to_list(X_test, y_test)
        test_list = self._convert_to_list(X_test, y_test)
        
        return train_list, val_list, test_list
    
    def _convert_to_list(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        """将numpy数组转换为字典列表"""
        result = []
        for i in range(len(X)):
            result.append({
                "time_series": X[i],  # [C, L]
                "label": y[i],
            })
        return result
    
    def _get_pre_prompt(self, row) -> str:
        """返回预提示文本"""
        dataset_name = UEAClassificationDataset._dataset_name
        num_classes = UEAClassificationDataset._num_classes
        num_channels = UEAClassificationDataset._num_channels
        class_letters = UEAClassificationDataset._class_letters
        
        classes_str = ", ".join(class_letters)
        
        prompt = f"""You are a multivariate time series classifier for the {dataset_name} dataset.
        This dataset contains {num_classes} classes: {classes_str}.
        The time series has {num_channels} channels.
        Analyze all channels and output ONLY the single letter label.

        Time series data:"""
        return prompt
    
    def _get_post_prompt(self, row) -> str:
        """返回后提示文本"""
        return "Label:"
    
    def _get_answer(self, row) -> str:
        """返回答案（字母标签）"""
        original_label = row["label"]
        letter_label = UEAClassificationDataset._label_to_letter[original_label]
        return letter_label
    
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        """
        将多变量时间序列转换为TextTimeSeriesPrompt列表
        
        采用与HAR/PAMAP2相同的方式：每个通道独立处理
        """
        # time_series: [C, L]
        series = row["time_series"]
        
        # 转换为tensor [C, L]
        if isinstance(series, np.ndarray):
            series = torch.tensor(series, dtype=torch.float32)
        
        # 处理NaN值
        series = torch.nan_to_num(series, nan=0.0)
        
        num_channels = series.shape[0]
        
        # 每个通道独立归一化
        means = series.mean(dim=1, keepdim=True)  # [C, 1]
        stds = series.std(dim=1, keepdim=True)    # [C, 1]
        
        # 处理零或很小的标准差
        min_std = 1e-6
        stds = torch.clamp(stds, min=min_std)
        
        series_norm = (series - means) / stds  # [C, L]
        
        # 检查NaN/Inf
        if torch.isnan(series_norm).any() or torch.isinf(series_norm).any():
            print(f"⚠️ NaN/Inf detected after normalization, replacing with 0")
            series_norm = torch.nan_to_num(series_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 创建每个通道的prompt（与HAR/PAMAP2一致）
        prompts = []
        for i in range(num_channels):
            channel_data = series_norm[i].tolist()
            mean_val = means[i].item()
            std_val = stds[i].item()
            
            # text_prompt = f"Channel {i+1} data (mean={mean_val:.4f}, std={std_val:.4f}):"
            text_prompt = f"Channel {i+1} data:"
            prompts.append(TextTimeSeriesPrompt(text_prompt, channel_data))
        
        return prompts
    
    def _format_sample(self, row):
        """格式化样本，添加额外信息"""
        sample = super()._format_sample(row)
        # 保存原始标签用于评估
        sample["original_label"] = row["label"]
        sample["letter_label"] = UEAClassificationDataset._label_to_letter[row["label"]]
        return sample
    
    @staticmethod
    def get_labels() -> List[str]:
        """返回所有类别的字母标签"""
        return UEAClassificationDataset._class_letters or []
    
    @staticmethod
    def get_label_mapping() -> dict:
        """返回原始标签到字母的映射"""
        return UEAClassificationDataset._label_to_letter or {}


# 测试
if __name__ == "__main__":
    print("Testing UEAClassificationDataset...")
    
    dataset = UEAClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name="AtrialFibrillation",
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Labels: {UEAClassificationDataset.get_labels()}")
    print(f"Label mapping: {UEAClassificationDataset.get_label_mapping()}")
    
    # 查看样本
    if len(dataset) > 0:
        sample = dataset[0]
        print("\n" + "="*50)
        print("Sample keys:", sample.keys())
        print("Pre-prompt:", sample["pre_prompt"])
        print("Post-prompt:", sample["post_prompt"])
        print("Answer:", sample["answer"])
        print("Letter label:", sample.get("letter_label", "N/A"))
        print("Num time series:", len(sample.get("time_series_text", [])))
            
