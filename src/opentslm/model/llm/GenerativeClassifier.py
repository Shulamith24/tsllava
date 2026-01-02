# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
生成式分类器 - 基于OpenTSLMSP的约束解码分类

特性:
- 专用类别token: <cls_0>, <cls_1>, ..., <cls_{K-1}>
- 约束解码: 只允许生成类别token
- 只对label token计算损失
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.utils.rnn import pad_sequence

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.llm.TimeSeriesLLM import TimeSeriesLLM
from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.encoder.TSLANetEncoder import TSLANetEncoder
from opentslm.model.projector.MLPProjector import MLPProjector


class GenerativeClassifier(TimeSeriesLLM):
    """
    生成式分类器
    
    基于OpenTSLMSP架构，支持:
    - 动态添加类别token
    - 约束解码（只生成类别token）
    - 只对label token计算损失
    
    Args:
        num_classes: 类别数量
        llm_id: LLM模型ID
        device: 设备
        encoder_type: 编码器类型 ("transformer_cnn" or "tslanet")
        encoder_pretrained_path: 编码器预训练权重路径
        tslanet_config: TSLANet配置
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
        super().__init__(device)
        
        self.num_classes = num_classes
        self.llm_id = llm_id
        
        # 1) 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 2) 添加类别token
        self.class_tokens = [f"<cls_{i}>" for i in range(num_classes)]
        num_added = self.tokenizer.add_tokens(self.class_tokens, special_tokens=True)
        print(f"📝 添加了 {num_added} 个类别token")
        
        # 获取类别token的id
        self.class_token_ids = self.tokenizer.convert_tokens_to_ids(self.class_tokens)
        print(f"📝 类别token IDs: {self.class_token_ids}")
        
        # 3) 加载LLM并resize embedding
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # 4) 初始化新添加的类别token embedding
        with torch.no_grad():
            embeddings = self.llm.get_input_embeddings()
            old_embeddings = embeddings.weight[:-num_classes, :]
            mean_emb = old_embeddings.mean(dim=0)
            std_emb = old_embeddings.std()
            for cls_id in self.class_token_ids:
                embeddings.weight[cls_id] = mean_emb + torch.randn_like(mean_emb) * std_emb * 0.02
        print(f"📝 类别token embedding已初始化")
        
        # 5) 编码器
        self.encoder_type = encoder_type
        if encoder_type == "tslanet":
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
            
            if encoder_pretrained_path:
                self.encoder.load_pretrained(encoder_pretrained_path)
                print(f"✅ 加载TSLANet预训练权重: {encoder_pretrained_path}")
        else:
            self.encoder = TransformerCNNEncoder().to(device)
            self.patch_size = 4
        
        # 6) Projector
        self.projector = MLPProjector(
            ENCODER_OUTPUT_DIM, self.llm.config.hidden_size, device=device
        ).to(device)
        
        # 7) LoRA相关
        self.lora_enabled = False
        self.original_llm = None
        
        # 8) 冻结LLM骨干，但让embedding层可训练
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.get_input_embeddings().weight.requires_grad = True
    
    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """启用LoRA微调"""
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft包未安装，无法使用LoRA")
        
        if self.lora_enabled:
            return
        
        self.original_llm = self.llm
        
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        self.lora_enabled = True
        
        lora_params = sum(
            p.numel() for n, p in self.llm.named_parameters()
            if p.requires_grad and "lora_" in n
        )
        print(f"✅ LoRA已启用: {lora_params:,} 可训练参数")
    
    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """获取LoRA参数"""
        if not self.lora_enabled:
            return []
        return [p for n, p in self.llm.named_parameters() if p.requires_grad and "lora_" in n]
    
    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量处理输入（与OpenTSLMSP相同逻辑）
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

        # 2) Tokenize & embed
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # 3) 批量编码时间序列
        ts_list: List[torch.Tensor] = []
        for sample in batch:
            for ts in sample["time_series"]:
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)

        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(device, non_blocking=True)
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

        # 4) 重新组装
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

            # time_series_text + ts
            for i in range(n_ts):
                idx = 1 + i
                length = sample_masks[idx].sum().item()
                seq_embeds.append(sample_embeds[idx, :length, :])
                seq_masks.append(sample_masks[idx, :length])

                proj = ts_proj[ts_offset + i]
                seq_embeds.append(proj)
                seq_masks.append(torch.ones(proj.size(0), device=device, dtype=torch.long))

            ts_offset += n_ts

            # post_prompt
            length = sample_masks[-1].sum().item()
            seq_embeds.append(sample_embeds[-1, :length, :])
            seq_masks.append(sample_masks[-1, :length])

            all_seq_embeds.append(torch.cat(seq_embeds, dim=0))
            all_seq_masks.append(torch.cat(seq_masks, dim=0))

        # 5) Padding
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)

        return inputs_embeds, attention_mask
    
    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        计算损失（只对label token计算交叉熵）
        """
        answers = [b["answer"] for b in batch]
        
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B, L, H = inputs_embeds.size()
        
        # Tokenize answers
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)
        
        # 拼接
        inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)
        attention_mask = torch.cat([attention_mask, ans_mask], dim=1)
        
        # Labels: 只对answer部分计算损失
        total_len = attention_mask.size(1)
        labels = torch.full((B, total_len), -100, device=self.device, dtype=torch.long)
        labels[:, L:] = ans_ids
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss
    
    @torch.no_grad()
    def predict(self, batch: List[Dict[str, any]]) -> List[int]:
        """
        约束解码预测
        
        只进行一次forward，对next_token_logits做mask，只允许类别token
        使用attention_mask找到每个样本的实际最后位置
        """
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B = inputs_embeds.size(0)
        
        # Forward得到logits
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # 获取每个样本的实际最后位置（attention_mask为1的最后一个位置）
        # attention_mask: [B, L], 1表示有效位置
        seq_lengths = attention_mask.sum(dim=1)  # [B] 每个样本的实际长度
        
        # 收集每个样本最后位置的logits
        next_token_logits_list = []
        for i in range(B):
            last_pos = seq_lengths[i].item() - 1  # 0-indexed
            next_token_logits_list.append(outputs.logits[i, last_pos, :])
        next_token_logits = torch.stack(next_token_logits_list, dim=0)  # [B, vocab_size]
        
        # 约束解码：只保留类别token的logits
        mask = torch.full_like(next_token_logits, float("-inf"))
        for cls_id in self.class_token_ids:
            mask[:, cls_id] = 0.0
        masked_logits = next_token_logits + mask
        
        # Argmax得到预测
        pred_token_ids = masked_logits.argmax(dim=-1)  # [B]
        
        # 转换为类别索引
        predictions = []
        for pred_id in pred_token_ids:
            pred_id = pred_id.item()
            if pred_id in self.class_token_ids:
                pred_cls = self.class_token_ids.index(pred_id)
            else:
                pred_cls = 0  # fallback
            predictions.append(pred_cls)
        
        return predictions
    
    def get_eos_token(self) -> str:
        return self.tokenizer.eos_token
    
    def store_to_file(self, path: str):
        """保存模型"""
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
            "num_classes": self.num_classes,
            "encoder_type": self.encoder_type,
            "class_tokens": self.class_tokens,
            "class_token_ids": self.class_token_ids,
        }
        
        if self.lora_enabled:
            checkpoint["lora_enabled"] = True
            lora_state = {}
            for n, p in self.llm.named_parameters():
                if p.requires_grad and "lora_" in n:
                    lora_state[n] = p.data.clone()
            checkpoint["lora_state"] = lora_state
        
        torch.save(checkpoint, path)
        print(f"💾 模型已保存: {path}")
    
    def load_from_file(self, path: str):
        """加载模型"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        
        if ckpt.get("lora_enabled", False) and "lora_state" in ckpt:
            if not self.lora_enabled:
                raise RuntimeError("检查点包含LoRA权重，但当前模型未启用LoRA")
            for n, p in self.llm.named_parameters():
                if n in ckpt["lora_state"]:
                    p.data.copy_(ckpt["lora_state"][n])
        
        print(f"📥 模型已加载: {path}")
