# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UCR ICL分类Dataset

用于In-Context Learning分类的Dataset，将检索到的支持样本和查询样本组合成ICL格式的prompt。

Prompt格式：
─────────────────────────────────────
You are a time series classifier for the {dataset_name} dataset.
This task has {num_classes} possible classes: A, B, C, ...

Based on the following labeled examples, classify the final query time series.
Only output the class label.

[Support Examples]
Example 1:
<TS_TOKENS>
Class: A

Example 2:
<TS_TOKENS>
Class: B

...

[Query]
Query time series:
<TS_TOKENS>

Class:
─────────────────────────────────────

Answer: A (或B, C, ...)
"""

import random
import torch
from typing import List, Dict, Tuple, Optional, Literal, Any
from torch.utils.data import Dataset

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import index_to_excel_label


class UCRICLClassificationDataset(Dataset):
    """
    UCR ICL分类Dataset
    
    将检索到的支持样本和查询样本组合成ICL格式的prompt。
    
    Args:
        time_series: [N, L] 所有样本的时间序列
        labels: [N] 对应的标签 (0-indexed)
        retriever: TSLANetRetriever实例
        dataset_name: 数据集名称
        k_shot: 每个类别的支持样本数
        top_m: 每个类别检索的候选数量
        eos_token: 结束token
        split: 数据划分 (train/test)
        exclude_query: 是否排除query自身 (训练时为True)
        max_episode_classes: 每个episode最多采样的类别数 (None表示使用全部类别)
        min_episode_classes: 每个episode最少采样的类别数 (默认2)
    """
    
    def __init__(
        self,
        time_series: torch.Tensor,
        labels: torch.Tensor,
        retriever,  # TSLANetRetriever
        dataset_name: str,
        k_shot: int = 1,
        top_m: int = 10,
        eos_token: str = "</s>",
        split: Literal["train", "test"] = "train",
        exclude_query: bool = True,
        max_episode_classes: Optional[int] = None,
        min_episode_classes: int = 2
    ):
        self.time_series = time_series
        self.labels = labels
        self.retriever = retriever
        self.dataset_name = dataset_name
        self.k_shot = k_shot
        self.top_m = top_m
        self.eos_token = eos_token
        self.split = split
        self.exclude_query = exclude_query if split == "train" else False
        
        # 动态类别采样参数
        self.max_episode_classes = max_episode_classes
        self.min_episode_classes = min_episode_classes
        
        # 获取类别信息
        self.unique_labels = sorted(torch.unique(labels).tolist())
        self.num_classes = len(self.unique_labels)
        
        # 创建标签到字母的映射 (全局映射，用于测试集或不启用动态采样时)
        self.label_to_letter = {
            label: index_to_excel_label(i) 
            for i, label in enumerate(self.unique_labels)
        }
        self.letter_to_label = {v: k for k, v in self.label_to_letter.items()}
        
        # 类别字母列表
        self.class_letters = [self.label_to_letter[l] for l in self.unique_labels]
        
        # 是否启用动态类别采样 (只在训练时启用)
        self.enable_dynamic_sampling = (
            split == "train" and 
            max_episode_classes is not None and 
            max_episode_classes < self.num_classes
        )
    
    def __len__(self):
        return len(self.time_series)
    
    def _sample_episode_classes(self, query_label: int) -> Tuple[List[int], Dict[int, str]]:
        """
        为当前episode采样类别并创建局部标签映射
        
        Args:
            query_label: 查询样本的标签（必须包含在采样类别中）
        
        Returns:
            episode_labels: 本episode采样的类别列表（已排序）
            episode_label_to_letter: 局部的标签到字母映射（从A开始）
        """
        # 确定采样的类别数量
        min_classes = max(2, self.min_episode_classes)
        max_classes = min(self.max_episode_classes, self.num_classes)
        
        if min_classes >= max_classes:
            n_classes = max_classes
        else:
            n_classes = random.randint(min_classes, max_classes)
        
        # 必须包含query_label
        other_labels = [l for l in self.unique_labels if l != query_label]
        
        # 从其他类别中随机采样
        n_sample_others = n_classes - 1
        if n_sample_others >= len(other_labels):
            sampled_others = other_labels
        else:
            sampled_others = random.sample(other_labels, n_sample_others)
        
        # 合并并排序
        episode_labels = sorted([query_label] + sampled_others)
        
        # 创建局部映射：从A开始重新编号
        episode_label_to_letter = {
            label: index_to_excel_label(i)
            for i, label in enumerate(episode_labels)
        }
        
        return episode_labels, episode_label_to_letter
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        返回ICL格式的样本
        
        Returns:
            Dict包含:
            - pre_prompt: 指令部分
            - time_series: 时间序列列表
            - time_series_text: 时间序列对应的文本描述列表
            - post_prompt: 查询提示部分
            - answer: 答案 (查询样本的标签)
            - letter_label: 字母标签
            - original_label: 原始标签
        """
        # 获取query
        query_ts = self.time_series[idx]
        query_label = int(self.labels[idx].item())
        
        # 动态类别采样 (仅训练时)
        if self.enable_dynamic_sampling:
            episode_labels, episode_label_to_letter = self._sample_episode_classes(query_label)
        else:
            episode_labels = self.unique_labels
            episode_label_to_letter = self.label_to_letter
        
        query_letter = episode_label_to_letter[query_label]
        
        # 检索支持样本 (只从episode包含的类别中检索)
        query_idx = idx if self.exclude_query else None
        support_indices, support_ts_list, support_labels = self.retriever.retrieve_for_query(
            query_ts,
            query_idx=query_idx,
            k_shot=self.k_shot,
            top_m=self.top_m,
            exclude_query=self.exclude_query,
            target_labels=episode_labels if self.enable_dynamic_sampling else None
        )
        
        # 构建prompt (使用局部映射)
        sample = self._build_icl_prompt(
            query_ts=query_ts,
            query_label=query_label,
            query_letter=query_letter,
            support_ts_list=support_ts_list,
            support_labels=support_labels,
            episode_labels=episode_labels,
            episode_label_to_letter=episode_label_to_letter
        )
        
        # 添加元信息
        sample["query_idx"] = idx
        sample["query_label"] = query_label
        sample["support_indices"] = support_indices
        sample["support_labels"] = support_labels
        sample["letter_label"] = query_letter
        sample["original_label"] = query_label
        sample["episode_labels"] = episode_labels  # 本episode的类别
        sample["episode_num_classes"] = len(episode_labels)  # 本episode的类别数
        
        return sample
    
    def _build_icl_prompt(
        self,
        query_ts: torch.Tensor,
        query_label: int,
        query_letter: str,
        support_ts_list: List[torch.Tensor],
        support_labels: List[int],
        episode_labels: Optional[List[int]] = None,
        episode_label_to_letter: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        构建ICL prompt
        
        Args:
            query_ts: [L] 查询样本的时间序列
            query_label: 查询样本的标签
            query_letter: 查询样本的字母标签
            support_ts_list: 支持样本的时间序列列表
            support_labels: 支持样本的标签列表
            episode_labels: 本episode的类别列表 (动态采样时使用)
            episode_label_to_letter: 本episode的标签映射 (动态采样时使用)
        
        Returns:
            PromptWithAnswer.to_dict()格式的字典
        """
        # 使用传入的映射或默认全局映射
        if episode_label_to_letter is None:
            episode_label_to_letter = self.label_to_letter
        if episode_labels is None:
            episode_labels = self.unique_labels
        
        episode_num_classes = len(episode_labels)
        episode_class_letters = [episode_label_to_letter[l] for l in episode_labels]
        
        # 预处理函数
        def preprocess_ts(ts: torch.Tensor) -> torch.Tensor:
            """z-normalization"""
            ts = ts.float()
            ts = torch.nan_to_num(ts, nan=0.0)
            mean = ts.mean()
            std = ts.std()
            if std > 1e-8:
                ts = (ts - mean) / std
            else:
                ts = ts - mean
            return ts
        
        # 构建类别字符串 (使用episode的局部类别)
        classes_str = ", ".join(episode_class_letters)
        
        # ===== Pre-prompt =====
        pre_prompt = f"""You are a time series classifier for the {self.dataset_name} dataset.
This task has {episode_num_classes} possible classes: {classes_str}.

Based on the following labeled examples, classify the final query time series.
Only output the class label.

[Support Examples]"""
        
        # ===== 时间序列prompt列表 =====
        text_time_series_prompt_list = []
        
        # 添加支持样本 - 每个样本包含文本描述(含标签)和时间序列
        for i, (support_ts, support_label) in enumerate(zip(support_ts_list, support_labels)):
            # 使用局部映射获取字母标签
            support_letter = episode_label_to_letter[support_label]
            support_ts_processed = preprocess_ts(support_ts)
            
            # 文本描述：包含示例编号和标签
            text = f"\nExample {i+1} (Class: {support_letter}):"
            
            # 创建TextTimeSeriesPrompt
            ts_prompt = TextTimeSeriesPrompt(
                text=text,
                time_series=support_ts_processed.tolist()
            )
            text_time_series_prompt_list.append(ts_prompt)
        
        # 添加Query
        query_ts_processed = preprocess_ts(query_ts)
        
        query_text = "\n\n[Query]\nQuery time series:"
        query_ts_prompt = TextTimeSeriesPrompt(
            text=query_text,
            time_series=query_ts_processed.tolist()
        )
        text_time_series_prompt_list.append(query_ts_prompt)
        
        # ===== Post-prompt =====
        post_prompt = "\n\nClass:"
        
        # ===== Answer =====
        answer = f" {query_letter}{self.eos_token}"
        
        # 使用PromptWithAnswer构建标准格式
        prompt_with_answer = PromptWithAnswer(
            pre_prompt=TextPrompt(pre_prompt),
            text_time_series_prompt_list=text_time_series_prompt_list,
            post_prompt=TextPrompt(post_prompt),
            answer=answer
        )
        
        return prompt_with_answer.to_dict()
    
    @staticmethod
    def get_label_from_letter(letter: str, letter_to_label: Dict[str, int]) -> int:
        """将字母标签转换回原始标签"""
        return letter_to_label.get(letter.strip().upper(), -1)


def create_icl_collate_fn(patch_size: int = 8):
    """
    创建ICL Dataset的collate函数
    
    Args:
        patch_size: 用于对齐时间序列的patch大小
    """
    from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
    
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        """
        Collate函数
        
        主要处理时间序列的对齐和批处理
        """
        return extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=patch_size
        )
    
    return collate_fn


# --- 测试代码 ---
if __name__ == "__main__":
    print("Testing UCRICLClassificationDataset...")
    
    # 模拟检索器
    class MockRetriever:
        def __init__(self, time_series, labels):
            self.time_series = time_series
            self.labels = labels
        
        def retrieve_for_query(self, query_ts, query_idx=None, k_shot=1, top_m=10, exclude_query=True):
            # 随机返回支持样本
            N = len(self.labels)
            indices = []
            ts_list = []
            label_list = []
            
            unique_labels = torch.unique(self.labels).tolist()
            for label in unique_labels:
                label_indices = (self.labels == label).nonzero().squeeze(-1).tolist()
                if isinstance(label_indices, int):
                    label_indices = [label_indices]
                
                for idx in label_indices[:k_shot]:
                    if exclude_query and idx == query_idx:
                        continue
                    indices.append(idx)
                    ts_list.append(self.time_series[idx])
                    label_list.append(label)
                    if len([l for l in label_list if l == label]) >= k_shot:
                        break
            
            return indices, ts_list, label_list
    
    # 创建测试数据
    N, L = 50, 100
    time_series = torch.randn(N, L)
    labels = torch.randint(0, 3, (N,))  # 3个类别
    
    retriever = MockRetriever(time_series, labels)
    
    # 创建Dataset
    dataset = UCRICLClassificationDataset(
        time_series=time_series,
        labels=labels,
        retriever=retriever,
        dataset_name="TestDataset",
        k_shot=2,
        top_m=10,
        eos_token="</s>",
        split="train",
        exclude_query=True
    )
    
    # 测试获取样本
    sample = dataset[0]
    
    print(f"Pre-prompt:\n{sample['pre_prompt'][:200]}...")
    print(f"\nNum TS prompts: {len(sample['text_time_series_prompt_list'])}")
    print(f"Post-prompt: {sample['post_prompt']}")
    print(f"Answer: {sample['answer']}")
    print(f"Query label: {sample['query_label']}")
    print(f"Letter label: {sample['letter_label']}")
    print(f"Support labels: {sample['support_labels']}")
    
    print("\n✅ 测试通过!")
