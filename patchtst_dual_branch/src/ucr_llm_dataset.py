# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
UCR LLM 分类数据集

用于 LLM 指令式分类训练，使用类别 token 格式: <c0>, <c1>, ...
"""

from typing import List, Tuple, Literal
import torch

from .prompt.text_prompt import TextPrompt
from .prompt.text_time_series_prompt import TextTimeSeriesPrompt
from .prompt.prompt_with_answer import PromptWithAnswer
from .ucr_loader import load_ucr_dataset


def index_to_class_token(index: int) -> str:
    """
    将整数索引转换为特殊类别token。
    
    映射规则：
    0 -> <c0>
    1 -> <c1>
    ...
    K-1 -> <cK-1>
    """
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    return f"<c{index}>"


class UCRLLMClassificationDataset(torch.utils.data.Dataset):
    """
    UCR LLM 分类数据集
    
    Prompt格式：
    ─────────────────────────────────────
    Classify the time series into one of {num_classes} classes.
    Output only the class token.

    Time series data:
    <TS_EMBEDDINGS>
    
    Class:
    ─────────────────────────────────────
    
    Answer: <c0> (或 <c1>, <c2>, ...)
    """
    
    # 类变量存储数据集信息
    _dataset_name: str = None
    _label_to_token: dict = None
    _token_to_label: dict = None
    _num_classes: int = None
    _class_tokens: List[str] = None
    
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        dataset_name: str = "ECG5000",
        raw_data_path: str = "./data",
    ):
        """
        Args:
            split: 数据划分 ("train", "validation", "test")
            EOS_TOKEN: 结束token
            dataset_name: UCR数据集名称
            raw_data_path: 数据根目录
        """
        super().__init__()
        
        self.split = split
        self.EOS_TOKEN = EOS_TOKEN
        self.dataset_name = dataset_name
        self.raw_data_path = raw_data_path
        
        # 加载数据
        train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path)
        
        # 选择对应的划分
        if split == "train":
            self.df = train_df
        else:  # validation 或 test (UCR 没有官方验证集)
            self.df = test_df
        
        # 获取特征列（除label外的所有列）
        self.feature_cols = [col for col in self.df.columns if col != "label"]
        
        # 获取所有唯一标签并排序
        all_labels = sorted(train_df["label"].unique().tolist())
        self.num_classes = len(all_labels)
        
        # 创建标签到特殊token的映射
        tokens = [index_to_class_token(i) for i in range(self.num_classes)]
        self.label_to_token = {label: tokens[i] for i, label in enumerate(all_labels)}
        self.token_to_label = {v: k for k, v in self.label_to_token.items()}
        self.label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        self.class_tokens = tokens
        
        # 存储类变量
        UCRLLMClassificationDataset._dataset_name = dataset_name
        UCRLLMClassificationDataset._label_to_token = self.label_to_token
        UCRLLMClassificationDataset._token_to_label = self.token_to_label
        UCRLLMClassificationDataset._num_classes = self.num_classes
        UCRLLMClassificationDataset._class_tokens = self.class_tokens
        
        # 转换为列表
        self.data = self.df.to_dict('records')
        
        print(f"📊 UCRLLMClassificationDataset: {dataset_name}")
        print(f"   Split: {split}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Samples: {len(self.data)}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
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
        
        # 获取标签信息
        original_label = row["label"]
        class_token = self.label_to_token[original_label]
        int_label = self.label_to_idx[original_label]
        
        # 构造 Prompt
        pre_prompt = self._get_pre_prompt()
        post_prompt = self._get_post_prompt()
        time_series_text = self._get_time_series_text(ts)
        answer = class_token + self.EOS_TOKEN
        
        return {
            # LLM 训练所需字段
            "pre_prompt": pre_prompt,
            "time_series_text": [time_series_text],
            "time_series": [ts],  # List[Tensor]
            "post_prompt": post_prompt,
            "answer": answer,
            # 评估所需字段
            "int_label": int_label,
            "original_label": original_label,
            "class_token": class_token,
        }
    
    def _get_pre_prompt(self) -> str:
        """返回预提示文本"""
        prompt = f"""Classify the time series into one of {self.num_classes} classes.
Output only the class token.

Time series data:"""
        return prompt
    
    def _get_post_prompt(self) -> str:
        """返回后提示文本"""
        return "\nClass:"
    
    def _get_time_series_text(self, ts: torch.Tensor) -> str:
        """返回时间序列描述文本"""
        return f"This is a univariate time series with {len(ts)} data points:"
    
    def get_num_classes(self) -> int:
        """返回类别数量"""
        return self.num_classes
    
    def get_max_length(self) -> int:
        """返回时间序列最大长度"""
        return len(self.feature_cols)
    
    @staticmethod
    def get_class_tokens() -> List[str]:
        """返回所有类别的特殊token"""
        return UCRLLMClassificationDataset._class_tokens or []
    
    @staticmethod
    def get_label_mapping() -> dict:
        """返回原始标签到特殊token的映射"""
        return UCRLLMClassificationDataset._label_to_token or {}


# ---------------------------
# 测试
# ---------------------------

if __name__ == "__main__":
    print("Testing UCRLLMClassificationDataset...")
    
    dataset = UCRLLMClassificationDataset(
        split="train",
        EOS_TOKEN="<eos>",
        dataset_name="ECG200",
    )
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Class tokens: {dataset.get_class_tokens()}")
    
    # 查看样本
    sample = dataset[0]
    print("\n" + "="*50)
    print("Sample keys:", sample.keys())
    print("Pre-prompt:", sample["pre_prompt"])
    print("Post-prompt:", sample["post_prompt"])
    print("Answer:", sample["answer"])
    print("Time series shape:", sample["time_series"][0].shape)
