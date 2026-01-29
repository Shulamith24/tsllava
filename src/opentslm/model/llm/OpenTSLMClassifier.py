# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
OpenTSLMClassifier: LLM-based Time Series Classifier with [ANS] Token

实验 A: 验证 LLM 作为深序列聚合器的性能提升

与 OpenTSLMSP 的区别：
- 使用可学习的 [ANS] token 而非生成式分类
- 训练目标：K 类分类（交叉熵损失）而非文本生成
- 评估：直接基于 [ANS] 位置的 logits 进行分类

输入序列格式：
    [PrePrompt] + [TS tokens] + [PostPrompt] + [ANS]
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

from .TimeSeriesLLM import TimeSeriesLLM
from ..encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model_config import ENCODER_OUTPUT_DIM
from ..projector.MLPProjector import MLPProjector


class OpenTSLMClassifier(TimeSeriesLLM):
    """
    LLM-based Time Series Classifier using [ANS] token for classification
    
    核心组件：
    - encoder: 时间序列编码器
    - projector: 将编码器输出投影到 LLM hidden space
    - llm: Llama 预训练模型（支持 LoRA 微调）
    - ans_token: 可学习的 [ANS] 查询向量
    - classifier_head: 分类头 (hidden_size -> num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        encoder_pretrained_path: Optional[str] = None,
        tslanet_config: Optional[Dict] = None,
    ):
        """
        Args:
            num_classes: 分类类别数
            llm_id: LLM 模型 ID
            device: 设备
            encoder_type: 编码器类型 ("transformer_cnn" 或 "tslanet")
            encoder_pretrained_path: 编码器预训练权重路径（可选）
            tslanet_config: TSLANet 配置（仅当 encoder_type="tslanet" 时使用）
        """
        super().__init__(device)
        
        self.num_classes = num_classes

        # 1) tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2) LLM (使用原始预训练权重)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # 3) encoder + projector
        self.encoder_type = encoder_type
        if encoder_type == "tslanet":
            from ..encoder.TSLANetEncoder import TSLANetEncoder
            config = tslanet_config or {}
            default_config = {
                "output_dim": ENCODER_OUTPUT_DIM,
                "patch_size": 8,
                "emb_dim": 128,
                "depth": 2,
                "dropout": 0.15,
            }
            default_config.update(config)
            self.encoder = TSLANetEncoder(**default_config).to(device)
            self.patch_size = default_config.get("patch_size", 8)
            
            # 加载预训练权重
            if encoder_pretrained_path:
                self.encoder.load_pretrained(encoder_pretrained_path)
                print(f"✅ Loaded TSLANet pretrained weights from: {encoder_pretrained_path}")
        else:
            self.encoder = TransformerCNNEncoder().to(device)
            self.patch_size = 4
        
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)

        # 4) [ANS] token: 可学习的查询向量
        # 形状: [1, 1, hidden_size]
        self.ans_token = nn.Parameter(
            torch.randn(1, 1, self.llm.config.hidden_size, device=device, dtype=torch.bfloat16) * 0.02
        )

        # 5) 分类头
        self.classifier_head = nn.Linear(
            self.llm.config.hidden_size, num_classes, device=device, dtype=torch.bfloat16
        )

        # LoRA 相关
        self.lora_enabled = False
        self.original_llm = None

        # 冻结 LLM backbone（LoRA 会解冻部分参数）
        for p in self.llm.parameters():
            p.requires_grad = False

    def enable_gradient_checkpointing(self):
        """启用梯度检查点以减少内存使用"""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
        else:
            self.llm.config.use_cache = False
            if hasattr(self.llm, "model"):
                self.llm.model.gradient_checkpointing = True

    def forward(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        DDP 兼容的 forward 方法
        
        Args:
            batch: 批次数据
            
        Returns:
            损失值
        """
        return self.compute_loss(batch)

    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """启用 LoRA 微调"""
        if self.lora_enabled:
            print("⚠️ LoRA already enabled")
            return

        try:
            from peft import LoraConfig, get_peft_model

            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            # 保存原始模型引用
            self.original_llm = self.llm
            
            # 应用 LoRA
            self.llm = get_peft_model(self.llm, lora_config)
            self.lora_enabled = True
            
            print(f"✅ LoRA enabled: r={lora_r}, alpha={lora_alpha}")
            self.llm.print_trainable_parameters()

        except ImportError:
            raise ImportError("Please install peft: pip install peft")

    def get_lora_parameters(self):
        """获取 LoRA 参数用于优化器"""
        if not self.lora_enabled:
            return []
        
        lora_params = []
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_params.append(param)
        return lora_params

    def disable_lora(self):
        """禁用 LoRA"""
        if not self.lora_enabled:
            return
        
        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None
        
        self.lora_enabled = False
        print("✅ LoRA disabled")

    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理批次数据并添加 [ANS] token
        
        与 OpenTSLMSP 的区别：
        - 在每个样本序列末尾添加 [ANS] token
        - 返回 ans_positions 用于提取 [ANS] 的 hidden state
        
        Args:
            batch: 批次数据
            
        Returns:
            inputs_embeds: [B, L+1, H] (包含 [ANS])
            attention_mask: [B, L+1]
            ans_positions: [B] ([ANS] token 在每个样本中的位置)
        """
        device = self.device
        H = self.llm.config.hidden_size

        # 1) 收集所有文本
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        for sample in batch:
            start = len(all_texts)
            all_texts.append(sample["pre_prompt"])
            all_texts.extend(sample["time_series_text"])
            all_texts.append(sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))

        # 2) tokenize & embed 所有文本
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N_all, P_max, H]

        # 3) 批量编码时间序列
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(
                device, non_blocking=True
            )
            # 填充到 patch_size 的倍数
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)

            ts_enc = self.encoder(ts_padded.squeeze(-1))
            ts_proj = self.projector(ts_enc).to(text_embeds.dtype)
        else:
            ts_proj = torch.empty(0, 0, H, device=device, dtype=text_embeds.dtype)

        # 4) 重新组装每个样本
        all_seq_embeds, all_seq_masks = [], []
        ts_offset = 0
        for (start, end), n_ts in zip(text_ptrs, ts_counts):
            sample_embeds = text_embeds[start:end]
            sample_masks = attn_mask[start:end]
            seq_embeds, seq_masks = [], []

            # pre_prompt
            length = sample_masks[0].sum().item()
            seq_embeds.append(sample_embeds[0, :length, :])
            seq_masks.append(sample_masks[0, :length])

            # (text_i, ts_i) pairs
            for i in range(n_ts):
                idx = 1 + i
                length = sample_masks[idx].sum().item()
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(sample_masks[idx, :length])

                seq_embeds.append(ts_proj[ts_offset + i])
                seq_masks.append(
                    torch.ones(
                        ts_proj.size(1), device=device, dtype=sample_masks.dtype
                    )
                )

            ts_offset += n_ts

            # post_prompt
            idx = 1 + n_ts
            length = sample_masks[idx].sum().item()
            seq_embeds.append(sample_embeds[idx, :length, :])
            seq_masks.append(sample_masks[idx, :length])

            # 拼接为单个序列
            cat_embeds = torch.cat(seq_embeds, dim=0)  # [L_sample, H]
            cat_masks = torch.cat(seq_masks, dim=0)  # [L_sample]

            all_seq_embeds.append(cat_embeds)
            all_seq_masks.append(cat_masks)

        # 5) 添加 [ANS] token 到每个样本末尾
        ans_positions = []
        for i in range(len(all_seq_embeds)):
            # 记录 [ANS] 的位置（序列长度）
            ans_positions.append(all_seq_embeds[i].size(0))
            
            # 添加 [ANS] token embedding
            all_seq_embeds[i] = torch.cat(
                [all_seq_embeds[i], self.ans_token.squeeze(0)], dim=0
            )  # [L_sample+1, H]
            
            # 添加 attention mask
            all_seq_masks[i] = torch.cat(
                [all_seq_masks[i], torch.ones(1, device=device, dtype=all_seq_masks[i].dtype)], dim=0
            )  # [L_sample+1]

        # 6) 批量填充
        inputs_embeds = pad_sequence(
            all_seq_embeds, batch_first=True, padding_value=0.0
        )  # [B, L_max+1, H]
        attention_mask = pad_sequence(
            all_seq_masks, batch_first=True, padding_value=0
        )  # [B, L_max+1]

        ans_positions = torch.tensor(ans_positions, device=device, dtype=torch.long)  # [B]

        return inputs_embeds, attention_mask, ans_positions

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        计算分类损失
        
        Args:
            batch: 批次数据，每个样本包含 "int_label" 字段
            
        Returns:
            交叉熵损失
        """
        # 提取整数标签
        labels = torch.tensor(
            [b["int_label"] for b in batch], device=self.device, dtype=torch.long
        )

        # 获取输入 embeddings
        inputs_embeds, attention_mask, ans_positions = self.pad_and_apply_batch(batch)

        # LLM 前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # 提取 [ANS] 位置的 hidden states
        last_hidden_states = outputs.hidden_states[-1]  # [B, L, H]
        B = last_hidden_states.size(0)
        
        # 使用 ans_positions 提取每个样本的 [ANS] hidden state
        ans_hidden = last_hidden_states[torch.arange(B, device=self.device), ans_positions, :]  # [B, H]

        # 分类头
        logits = self.classifier_head(ans_hidden)  # [B, num_classes]

        # 计算交叉熵损失
        loss = nn.functional.cross_entropy(logits, labels)

        return loss

    @torch.no_grad()
    def predict(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        预测类别
        
        Args:
            batch: 批次数据
            
        Returns:
            预测的类别索引 [B]
        """
        # 获取输入 embeddings
        inputs_embeds, attention_mask, ans_positions = self.pad_and_apply_batch(batch)

        # LLM 前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

        # 提取 [ANS] 位置的 hidden states
        last_hidden_states = outputs.hidden_states[-1]
        B = last_hidden_states.size(0)
        ans_hidden = last_hidden_states[torch.arange(B, device=self.device), ans_positions, :]

        # 分类头
        logits = self.classifier_head(ans_hidden)

        # 预测
        predictions = torch.argmax(logits, dim=-1)  # [B]

        return predictions

    def get_eos_token(self):
        """返回 EOS token"""
        return self.tokenizer.eos_token

    def save_lora_state_to_checkpoint(self, checkpoint: dict) -> int:
        """保存 LoRA 状态到 checkpoint"""
        if not self.lora_enabled:
            checkpoint["lora_enabled"] = False
            return 0

        checkpoint["lora_enabled"] = True
        lora_state = {}
        num_params = 0
        
        for name, param in self.llm.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                lora_state[name] = param.data.cpu()
                num_params += param.numel()

        checkpoint["lora_state"] = lora_state
        return num_params

    def load_lora_state_from_checkpoint(
        self, checkpoint: dict, allow_missing: bool = False
    ):
        """从 checkpoint 加载 LoRA 状态"""
        ckpt_has_lora = checkpoint.get("lora_enabled", False)
        
        if not ckpt_has_lora and not self.lora_enabled:
            return
        
        if ckpt_has_lora and not self.lora_enabled:
            if not allow_missing:
                raise RuntimeError("Checkpoint has LoRA but current model doesn't")
            return
        
        if not ckpt_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError("Current model has LoRA but checkpoint doesn't")
            return

        # 加载 LoRA 权重
        lora_state = checkpoint["lora_state"]
        model_state = dict(self.llm.named_parameters())
        
        for name, param_data in lora_state.items():
            if name in model_state:
                model_state[name].data.copy_(param_data.to(self.device))
            else:
                print(f"⚠️ Warning: LoRA parameter {name} not found in current model")
