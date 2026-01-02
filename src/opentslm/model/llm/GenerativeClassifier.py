# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
生成式分类扩展模块

为OpenTSLMSP添加生成式分类能力，支持：
- 类别专用token (如 <cls_0>, <cls_1>, ...)
- 约束解码（只允许类别token）
- 只在label token上计算交叉熵损失

M1实验：单数据集有监督分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class GenerativeClassifier:
    """
    生成式分类器扩展
    
    为OpenTSLMSP模型添加分类能力：
    - 动态添加类别token到vocabulary
    - 约束解码只生成类别token
    - 只对类别token计算损失
    """
    
    def __init__(
        self, 
        model,  # OpenTSLMSP instance
        num_classes: int,
        class_names: Optional[List[str]] = None,
    ):
        """
        初始化生成式分类器
        
        Args:
            model: OpenTSLMSP模型实例
            num_classes: 类别数量
            class_names: 可选的类别名称列表
        """
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        
        # 生成类别token
        self.class_tokens = [f"<cls_{i}>" for i in range(num_classes)]
        
        # 添加类别token到tokenizer
        self._add_class_tokens()
        
        # 获取类别token的ID
        self.class_token_ids = torch.tensor(
            [self.model.tokenizer.convert_tokens_to_ids(t) for t in self.class_tokens],
            device=self.model.device
        )
        
        print(f"✅ 生成式分类器初始化完成:")
        print(f"   类别数: {num_classes}")
        print(f"   类别tokens: {self.class_tokens}")
        print(f"   Token IDs: {self.class_token_ids.tolist()}")
    
    def _add_class_tokens(self):
        """将类别token添加到tokenizer并resize模型embeddings"""
        tokenizer = self.model.tokenizer
        
        # 添加特殊token
        num_added = tokenizer.add_special_tokens({
            "additional_special_tokens": self.class_tokens
        })
        
        if num_added > 0:
            # Resize LLM embeddings
            self.model.llm.resize_token_embeddings(len(tokenizer))
            print(f"   添加了 {num_added} 个特殊token")
    
    def build_classification_prompt(
        self,
        dataset_name: str,
        time_series: torch.Tensor,
    ) -> Dict:
        """
        构建分类prompt格式
        
        格式: Dataset=<name>. Classes: <cls_0>, <cls_1>, ... Predict label:
        
        Args:
            dataset_name: 数据集名称
            time_series: 时间序列数据 [L] 或 [L, D]
        
        Returns:
            符合OpenTSLMSP batch格式的dict
        """
        # 构建类别列表字符串
        classes_str = ", ".join(self.class_tokens)
        
        # Pre-prompt
        pre_prompt = f"Dataset={dataset_name}. Classes: {classes_str}. "
        
        # Time series text (描述)
        time_series_text = ["Time series:"]
        
        # Post-prompt
        post_prompt = "Predict label:"
        
        return {
            "pre_prompt": pre_prompt,
            "time_series_text": time_series_text,
            "time_series": [time_series],
            "post_prompt": post_prompt,
        }
    
    def compute_classification_loss(
        self,
        batch: List[Dict],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算分类损失（只在label token上计算交叉熵）
        
        Args:
            batch: OpenTSLMSP格式的batch
            labels: 类别标签 [B]
        
        Returns:
            (loss, logits) - 损失和类别logits
        """
        device = self.model.device
        B = len(batch)
        
        # 获取prompt embeddings
        inputs_embeds, attention_mask = self.model.pad_and_apply_batch(batch)
        
        # 获取类别token embeddings (作为answer)
        # 每个样本的answer就是对应类别的token
        answer_ids = self.class_token_ids[labels].unsqueeze(1)  # [B, 1]
        answer_emb = self.model.llm.get_input_embeddings()(answer_ids)  # [B, 1, H]
        
        # 拼接prompt和answer
        inputs_embeds = torch.cat([inputs_embeds, answer_emb], dim=1)  # [B, L+1, H]
        attention_mask = torch.cat([
            attention_mask,
            torch.ones(B, 1, device=device, dtype=attention_mask.dtype)
        ], dim=1)  # [B, L+1]
        
        # Labels: 只在最后一个token（类别token）上计算loss
        L = inputs_embeds.size(1)
        target_labels = torch.full((B, L), -100, device=device, dtype=torch.long)
        target_labels[:, -1] = self.class_token_ids[labels]  # 最后一个位置是类别token
        
        # Forward + loss
        outputs = self.model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=target_labels,
            return_dict=True,
        )
        
        # 获取最后一个位置的logits用于评估
        # 注意：模型输出logits[i]对应预测token i+1的分布
        # 所以 logits[:, -2] 是预测最后一个token的logits
        last_logits = outputs.logits[:, -2, :]  # [B, vocab_size]
        
        # 只取类别token的logits
        class_logits = last_logits[:, self.class_token_ids]  # [B, num_classes]
        
        return outputs.loss, class_logits
    
    @torch.no_grad()
    def predict(
        self,
        batch: List[Dict],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        约束解码预测（单步forward + argmax）
        
        Args:
            batch: OpenTSLMSP格式的batch
        
        Returns:
            (predictions, class_logits) - 预测类别和logits
        """
        self.model.eval()
        device = self.model.device
        
        # 获取prompt embeddings
        inputs_embeds, attention_mask = self.model.pad_and_apply_batch(batch)
        
        # 单步forward获取next token logits
        outputs = self.model.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # 获取最后一个位置的logits (预测下一个token)
        next_logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        
        # 约束解码：mask所有非类别token
        # 将非类别token的logits设为-inf
        mask = torch.ones_like(next_logits) * float('-inf')
        mask[:, self.class_token_ids] = 0.0  # 只允许类别token
        masked_logits = next_logits + mask
        
        # Argmax得到预测类别
        predicted_token_ids = masked_logits.argmax(dim=-1)  # [B]
        
        # 将token ID转换回类别索引
        # 找到每个预测token ID在class_token_ids中的索引
        predictions = torch.zeros(len(batch), dtype=torch.long, device=device)
        for i, tid in enumerate(predicted_token_ids):
            match = (self.class_token_ids == tid).nonzero(as_tuple=True)[0]
            if len(match) > 0:
                predictions[i] = match[0]
            else:
                predictions[i] = 0  # fallback
        
        # 类别logits
        class_logits = next_logits[:, self.class_token_ids]  # [B, num_classes]
        
        return predictions, class_logits
    
    def label_to_token(self, label: int) -> str:
        """将类别索引转换为类别token"""
        return self.class_tokens[label]
    
    def token_to_label(self, token: str) -> int:
        """将类别token转换为类别索引"""
        return self.class_tokens.index(token)
