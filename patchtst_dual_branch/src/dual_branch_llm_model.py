# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
DualBranchLLMModel: 双分支 LLM 分类模型

核心设计：
- 复用 PatchTST 时序编码器
- 复用 VisionEncoder 图像编码器  
- 复用现有投影层结构，输出维度改为 LLM hidden_size
- 使用 LLM (Llama-3.2-1B) 进行分类
- 支持 LoRA 微调
- 支持 ts_only / vision_only / both 分支模式
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Literal, Any
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, PatchTSTConfig, PatchTSTModel

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. LoRA fine-tuning will be disabled.")

from .vision_encoder import VisionEncoder
from .projector import MLPProjector, LinearProjector
from .model_config import PATCH_SIZE


class DualBranchLLMModel(nn.Module):
    """
    双分支 LLM 分类模型
    
    架构:
    - 时序分支: PatchTST backbone → 投影层
    - 图像分支: VisionEncoder → 投影层
    - LLM: Llama-3.2-1B (+ LoRA)
    
    分类方式:
    - 使用 Soft Prompt (将时序/图像嵌入作为 prompt 的一部分)
    - LLM 自回归生成类别 token (<c0>, <c1>, ...)
    """
    
    def __init__(
        self,
        # LLM 配置
        llm_id: str = "meta-llama/Llama-3.2-1B",
        # 分支模式
        branch_mode: Literal["both", "ts_only", "vision_only"] = "both",
        # PatchTST 时序分支参数
        context_length: int = 512,
        patch_length: int = 16,
        stride: int = 8,
        d_model: int = 128,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 3,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        # Vision 分支参数
        vit_model_name: str = "facebook/dinov2-base",
        vit_layer_idx: int = -1,
        vit_patch_size: int = 16,
        vit_stride: float = 0.5,
        # 投影层参数
        projector_type: Literal["mlp", "linear"] = "mlp",
        projector_dropout: float = 0.1,
        # 冻结控制
        freeze_ts_backbone: bool = False,
        freeze_vision_backbone: bool = True,
        # 设备
        device: str = "cuda",
    ):
        super().__init__()
        
        self.device = device
        self.branch_mode = branch_mode
        self.context_length = context_length
        self.d_model = d_model
        self.ts_patch_size = PATCH_SIZE  # 用于 collate_fn
        
        # ============ 1) LLM + Tokenizer ============
        print(f"🔧 加载 LLM: {llm_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="eager",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        self.llm_hidden_size = self.llm.config.hidden_size
        
        # 冻结 LLM (后续通过 LoRA 微调)
        for p in self.llm.parameters():
            p.requires_grad = False
        
        # LoRA 状态
        self.lora_enabled = False
        self.original_llm = None
        
        # ============ 2) 时序分支: PatchTST ============
        if branch_mode in ["both", "ts_only"]:
            patchtst_config = PatchTSTConfig(
                num_input_channels=1,
                context_length=context_length,
                patch_length=patch_length,
                stride=stride,
                d_model=d_model,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                ffn_dim=ffn_dim,
                dropout=dropout,
                use_cls_token=False,
            )
            self.ts_backbone = PatchTSTModel(config=patchtst_config).to(device)
            self.ts_num_patches = (context_length - patch_length) // stride + 1
            
            # 时序分支投影层 -> LLM hidden_size
            if projector_type == "mlp":
                self.ts_projector = MLPProjector(d_model, self.llm_hidden_size, dropout=projector_dropout).to(device)
            else:
                self.ts_projector = LinearProjector(d_model, self.llm_hidden_size).to(device)
        else:
            self.ts_backbone = None
            self.ts_projector = None
            self.ts_num_patches = 0
        
        # ============ 3) 图像分支: VisionEncoder ============
        if branch_mode in ["both", "vision_only"]:
            self.vision_encoder = VisionEncoder(
                model_name=vit_model_name,
                layer_idx=vit_layer_idx,
                ts_patch_size=vit_patch_size,
                ts_stride=vit_stride,
                device=device,
            )
            self.vision_hidden_dim = self.vision_encoder.get_output_dim()
            self.vision_num_patches = self.vision_encoder.get_num_patches()
            
            # 图像分支投影层 -> LLM hidden_size
            if projector_type == "mlp":
                self.vision_projector = MLPProjector(self.vision_hidden_dim, self.llm_hidden_size, dropout=projector_dropout).to(device)
            else:
                self.vision_projector = LinearProjector(self.vision_hidden_dim, self.llm_hidden_size).to(device)
        else:
            self.vision_encoder = None
            self.vision_projector = None
            self.vision_hidden_dim = 0
            self.vision_num_patches = 0
        
        # ============ 4) 冻结控制 ============
        if freeze_ts_backbone and self.ts_backbone is not None:
            for param in self.ts_backbone.parameters():
                param.requires_grad = False
            print("🧊 PatchTST backbone 已冻结")
        
        if freeze_vision_backbone and self.vision_encoder is not None:
            self.vision_encoder.freeze()
        
        # 打印模型信息
        self._print_model_info()
    
    def _print_model_info(self):
        """打印模型信息"""
        ts_params = sum(p.numel() for p in self.ts_backbone.parameters()) if self.ts_backbone else 0
        vision_params = self.vision_encoder.count_parameters() if self.vision_encoder else 0
        llm_params = sum(p.numel() for p in self.llm.parameters())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*60}")
        print(f"DualBranchLLMModel 模型信息")
        print(f"{'='*60}")
        print(f"分支模式: {self.branch_mode}")
        print(f"LLM hidden size: {self.llm_hidden_size}")
        if self.ts_backbone:
            print(f"时序分支 (PatchTST):")
            print(f"  - context_length: {self.context_length}")
            print(f"  - num_patches: {self.ts_num_patches}")
            print(f"  - d_model: {self.d_model}")
            print(f"  - 参数量: {ts_params:,}")
        if self.vision_encoder:
            print(f"图像分支 (VisionEncoder):")
            print(f"  - num_patches: {self.vision_num_patches}")
            print(f"  - hidden_dim: {self.vision_hidden_dim}")
            print(f"  - 参数量: {vision_params:,}")
        print(f"LLM 参数量: {llm_params:,}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()
            print("✅ LLM 梯度检查点已启用")
        if self.ts_backbone and hasattr(self.ts_backbone, "gradient_checkpointing_enable"):
            self.ts_backbone.gradient_checkpointing_enable()
            print("✅ PatchTST 梯度检查点已启用")
    
    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """启用 LoRA 微调"""
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft package is required for LoRA. Install with: pip install peft")
        
        if self.lora_enabled:
            raise RuntimeError("LoRA already enabled. Call disable_lora() first.")
        
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
        
        lora_params = sum(p.numel() for n, p in self.llm.named_parameters() if p.requires_grad and "lora_" in n)
        print(f"✅ LoRA enabled: {lora_params:,} parameters")
    
    def get_lora_parameters(self):
        """获取 LoRA 参数"""
        if not self.lora_enabled:
            return []
        return [p for n, p in self.llm.named_parameters() if p.requires_grad and "lora_" in n]
    
    def disable_lora(self):
        """禁用 LoRA"""
        if not self.lora_enabled:
            return
        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None
        self.lora_enabled = False
        print("✅ LoRA disabled")
    
    def get_eos_token(self) -> str:
        """获取 EOS token"""
        return self.tokenizer.eos_token
    
    def _encode_time_series(self, past_values: torch.Tensor) -> torch.Tensor:
        """
        编码时间序列
        
        Args:
            past_values: [B, T, 1] 时间序列
            
        Returns:
            [B, num_patches, llm_hidden_size] 编码后的嵌入
        """
        embeddings_list = []
        
        # 时序分支
        if self.branch_mode in ["both", "ts_only"] and self.ts_backbone is not None:
            ts_output = self.ts_backbone(past_values=past_values)
            ts_embeddings = ts_output.last_hidden_state
            if ts_embeddings.dim() == 4:
                ts_embeddings = ts_embeddings.squeeze(1)
            ts_embeddings = self.ts_projector(ts_embeddings)
            embeddings_list.append(ts_embeddings)
        
        # 图像分支
        if self.branch_mode in ["both", "vision_only"] and self.vision_encoder is not None:
            vision_embeddings = self.vision_encoder(past_values)
            vision_embeddings = self.vision_projector(vision_embeddings)
            embeddings_list.append(vision_embeddings)
        
        # 拼接
        if len(embeddings_list) > 1:
            return torch.cat(embeddings_list, dim=1)
        else:
            return embeddings_list[0]
    
    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理批次数据，构建 LLM 输入
        
        将文本和时序嵌入拼接成完整的输入序列。
        """
        device = self.device
        H = self.llm_hidden_size
        
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
        
        # 2) Tokenize 并嵌入所有文本
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
            # Pad 到相同长度
            max_len = max(ts.size(0) for ts in ts_list)
            # 对齐到 context_length
            padded_len = max(max_len, self.context_length)
            
            ts_padded_list = []
            for ts in ts_list:
                if ts.size(0) < padded_len:
                    pad_amt = padded_len - ts.size(0)
                    ts = torch.nn.functional.pad(ts, (0, 0, 0, pad_amt), mode="constant", value=0.0)
                else:
                    ts = ts[:padded_len]
                ts_padded_list.append(ts)
            
            ts_padded = torch.stack(ts_padded_list, dim=0).to(device, non_blocking=True)
            ts_proj = self._encode_time_series(ts_padded).to(text_embeds.dtype)
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
            
            # 每个 (text_i, ts_i)
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
        
        # 5) Pad 最终序列
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)
        
        return inputs_embeds, attention_mask
    
    def forward(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """前向传播，返回损失"""
        return self.compute_loss(batch)
    
    def compute_loss(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """计算训练损失"""
        answers = [b["answer"] for b in batch]
        
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B, L, H = inputs_embeds.size()
        
        # Tokenize answers (不添加 bos)
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)
        
        # 拼接
        inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)
        attention_mask = torch.cat([attention_mask, ans_mask], dim=1)
        
        # 构建 labels (只在 answer 部分计算损失)
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
    
    def generate(
        self,
        batch: List[Dict[str, Any]],
        max_new_tokens: int = 10,
        **generate_kwargs
    ) -> List[str]:
        """生成预测"""
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    
    def count_parameters(self) -> int:
        """计算可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_lora_state_to_checkpoint(self, checkpoint: dict):
        """保存 LoRA 状态到 checkpoint"""
        checkpoint["lora_enabled"] = self.lora_enabled
        
        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            lora_state = {}
            for name, param in self.llm.named_parameters():
                if param.requires_grad and "lora_" in name:
                    lora_state[name] = param.data.clone()
            
            if lora_state:
                checkpoint["lora_state"] = lora_state
                checkpoint["lora_config"] = self.llm.peft_config
                print(f"💾 Saved LoRA adapters with {len(lora_state)} parameters")
    
    def load_lora_state_from_checkpoint(self, checkpoint: dict, allow_missing: bool = False):
        """从 checkpoint 加载 LoRA 状态"""
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)
        
        if checkpoint_has_lora and "lora_state" in checkpoint:
            if not self.lora_enabled:
                raise RuntimeError("Checkpoint contains LoRA but LoRA is not enabled.")
            
            lora_state = checkpoint["lora_state"]
            for name, param in self.llm.named_parameters():
                if name in lora_state and param.requires_grad and "lora_" in name:
                    param.data.copy_(lora_state[name])
            print(f"📥 Loaded LoRA adapters")
        elif not checkpoint_has_lora and self.lora_enabled and not allow_missing:
            raise RuntimeError("Checkpoint has no LoRA but LoRA is enabled.")
