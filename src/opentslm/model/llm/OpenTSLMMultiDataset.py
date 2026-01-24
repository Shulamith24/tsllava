# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
OpenTSLMMultiDataset: 多数据集统一Prototype分类模型

核心架构:
- 输入序列: [DS_PROMPT_{ds_id}] + [TS_TOKENS] + [CLS]
- 每数据集独立的 Prompt (from PromptBank) + Prototype (from PrototypeBank)
- 共享主干: Encoder + Projector + LLM (with LoRA)

与 OpenTSLMPrototype 的区别:
1. 使用 PromptBank 替代单份 prompt_embeds
2. 使用 PrototypeBank 替代单个 PrototypeClassificationHead
3. forward 接收同一个 ds_id 的 batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence

from .OpenTSLMSP import OpenTSLMSP
from .prototype_banks import PromptBank, PrototypeBank
from opentslm.time_series_datasets.multi_dataset import MultiDatasetRegistry


class OpenTSLMMultiDataset(OpenTSLMSP):
    """
    多数据集统一Prototype分类模型
    
    输入序列结构:
        [DS_PROMPT (prompt_len tokens)] + [TS_tokens] + [CLS (1 token)]
    
    每个数据集有独立的:
        - Prompt: 从 PromptBank 获取
        - Prototype + Temperature: 从 PrototypeBank 获取
    
    Args:
        registry: MultiDatasetRegistry 数据集注册表
        llm_id: LLM模型ID
        device: 设备
        encoder_type: 编码器类型
        prompt_len: 每个数据集的prompt长度
        init_temperature: Prototype温度初始值
        **kwargs: 其他传递给OpenTSLMSP的参数
    """
    
    def __init__(
        self,
        registry: MultiDatasetRegistry,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        prompt_len: int = 10,
        init_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            llm_id=llm_id,
            device=device,
            encoder_type=encoder_type,
            **kwargs
        )
        
        self.registry = registry
        self.hidden_size = self.llm.config.hidden_size
        self.prompt_len = prompt_len
        self.num_datasets = registry.get_total_datasets()
        
        # 获取LLM dtype和embedding统计信息
        llm_dtype = next(self.llm.parameters()).dtype
        with torch.no_grad():
            llm_embeddings = self.llm.get_input_embeddings().weight
            emb_mean = llm_embeddings.mean(dim=0)
            emb_std = llm_embeddings.std(dim=0)
        
        # PromptBank: 每数据集独立的prompt
        self.prompt_bank = PromptBank(
            num_datasets=self.num_datasets,
            prompt_len=prompt_len,
            hidden_size=self.hidden_size,
            init_mean=emb_mean,
            init_std=emb_std,
            dtype=llm_dtype,
            device=device,
        )
        
        # PrototypeBank: 每数据集独立的prototype + temperature
        class_counts = registry.get_class_counts()
        self.prototype_bank = PrototypeBank(
            class_counts=class_counts,
            hidden_size=self.hidden_size,
            init_temperature=init_temperature,
            init_mean=emb_mean,
            init_std=emb_std,
            dtype=llm_dtype,
            device=device,
        )
        
        # 共享的 CLS token
        cls_init = emb_mean + torch.randn(self.hidden_size, device=device, dtype=llm_dtype) * emb_std * 0.1
        self.cls_embed = nn.Parameter(cls_init)
        
        # 共享的 CLS 投影层 (MLP with residual)
        self.cls_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, dtype=llm_dtype),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=llm_dtype)
        ).to(device)
        
        # 近似恒等初始化
        with torch.no_grad():
            nn.init.eye_(self.cls_projector[0].weight)
            nn.init.zeros_(self.cls_projector[0].bias)
            nn.init.zeros_(self.cls_projector[2].weight)
            nn.init.zeros_(self.cls_projector[2].bias)
        
        print(f"✅ OpenTSLMMultiDataset initialized: {self.num_datasets} datasets")
    
    def freeze_backbone(self):
        """
        Stage 0: 冻结主干网络
        只训练 PromptBank + PrototypeBank + cls_embed + cls_projector
        """
        # 冻结 encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # 冻结 projector
        for param in self.projector.parameters():
            param.requires_grad = False
        
        # 冻结 LLM（包括LoRA如果有）
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # 确保可学习组件解冻
        for param in self.prompt_bank.parameters():
            param.requires_grad = True
        for param in self.prototype_bank.parameters():
            param.requires_grad = True
        self.cls_embed.requires_grad = True
        for param in self.cls_projector.parameters():
            param.requires_grad = True
        
        print("🧊 Stage 0: Backbone frozen (encoder + projector + LLM)")
        print("   训练参数: PromptBank, PrototypeBank, cls_embed, cls_projector")
    
    def unfreeze_for_stage1(self, unfreeze_encoder: bool = True):
        """
        Stage 1: 解冻组件进行联合训练
        
        Args:
            unfreeze_encoder: 是否解冻encoder（默认True）
        """
        # 解冻 encoder（可选）
        if unfreeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = True
            print("🔓 Encoder 已解冻")
        
        # 解冻 projector
        for param in self.projector.parameters():
            param.requires_grad = True
        print("🔓 Projector 已解冻")
        
        # 如果启用了LoRA，LoRA参数会自动可训练
        if self.lora_enabled:
            lora_params = self.get_lora_parameters()
            print(f"🔓 LoRA 参数: {len(lora_params)} 个")
        
        print("✅ Stage 1: 联合训练模式")
    
    def _build_multi_dataset_input_embeds(
        self,
        batch: List[Dict[str, any]],
        ds_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建多数据集Prototype模型的输入embedding
        
        输入格式: [DS_PROMPT_{ds_id}] + [TS_tokens] + [CLS]
        
        Args:
            batch: 批次数据，所有样本必须来自同一个 ds_id
            ds_id: 数据集ID
        
        Returns:
            inputs_embeds: [B, seq_len, H]
            attention_mask: [B, seq_len]
            cls_positions: [B] CLS token在每个样本中的位置
        """
        device = self.device
        B = len(batch)
        H = self.hidden_size
        
        # 1. 获取该数据集的prompt embeddings
        ds_prompt = self.prompt_bank.get(ds_id)  # [prompt_len, H]
        
        # 2. 处理时间序列
        ts_list = []
        for sample in batch:
            for ts in sample["time_series"]:
                if ts.dim() == 1:
                    ts = ts.unsqueeze(-1)
                ts_list.append(ts)
        
        # Pad时间序列并编码
        if ts_list:
            ts_padded = pad_sequence(ts_list, batch_first=True).to(device, non_blocking=True)
            T_max = ts_padded.size(1)
            rem = T_max % self.patch_size
            if rem:
                pad_len = self.patch_size - rem
                pad = ts_padded.new_zeros(ts_padded.size(0), pad_len, ts_padded.size(2))
                ts_padded = torch.cat([ts_padded, pad], dim=1)
            
            # Encode and project
            ts_enc = self.encoder(ts_padded.squeeze(-1))  # [B, N_patches, embed_dim]
            ts_proj = self.projector(ts_enc).to(ds_prompt.dtype)  # [B, N_patches, H]
        else:
            ts_proj = torch.empty(B, 0, H, device=device, dtype=ds_prompt.dtype)
        
        # 3. 构建每个样本的序列
        all_seq_embeds = []
        all_seq_masks = []
        cls_positions = []
        
        ts_offset = 0
        for i, sample in enumerate(batch):
            n_ts = len(sample["time_series"])
            
            # 获取这个样本的时序tokens
            sample_ts_embeds = ts_proj[ts_offset:ts_offset + n_ts]  # [n_ts, N_patches, H]
            ts_offset += n_ts
            
            # 合并所有时序的patches
            if n_ts > 0:
                ts_tokens = sample_ts_embeds.reshape(-1, H)  # [total_patches, H]
            else:
                ts_tokens = torch.empty(0, H, device=device, dtype=ds_prompt.dtype)
            
            # 构建序列: [DS_PROMPT] + [TS_tokens] + [CLS]
            seq_embeds = torch.cat([
                ds_prompt,                      # [prompt_len, H]
                ts_tokens,                      # [total_patches, H]
                self.cls_embed.unsqueeze(0)     # [1, H]
            ], dim=0)
            
            # 计算CLS位置
            cls_pos = self.prompt_len + ts_tokens.size(0)
            cls_positions.append(cls_pos)
            
            # Attention mask (全1)
            seq_mask = torch.ones(seq_embeds.size(0), device=device, dtype=torch.long)
            
            all_seq_embeds.append(seq_embeds)
            all_seq_masks.append(seq_mask)
        
        # 4. Pad到统一长度
        inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, max_len, H]
        attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, max_len]
        cls_positions = torch.tensor(cls_positions, device=device, dtype=torch.long)  # [B]
        
        return inputs_embeds, attention_mask, cls_positions
    
    def forward_prototype(
        self,
        batch: List[Dict[str, any]],
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prototype前向传播
        
        要求: batch 中所有样本必须来自同一个 ds_id
        
        Args:
            batch: 批次数据，每个样本需包含:
                - time_series: List[Tensor] 时间序列数据
                - label_index: int 类别索引 (该数据集内部索引)
                - ds_id: int 数据集ID
            return_hidden: 是否返回隐向量
        
        Returns:
            loss: 交叉熵损失
            logits: [B, num_classes_ds]
            (可选) hidden: [B, H] CLS隐向量
        """
        # 验证同一 ds_id
        ds_ids = set(sample["ds_id"] for sample in batch)
        assert len(ds_ids) == 1, f"Batch must contain samples from single dataset, got {ds_ids}"
        ds_id = batch[0]["ds_id"]
        
        # 1. 构建输入
        inputs_embeds, attention_mask, cls_positions = self._build_multi_dataset_input_embeds(batch, ds_id)
        
        # 2. LLM前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 3. 提取CLS位置的隐向量
        last_hidden = outputs.hidden_states[-1]
        
        B = len(batch)
        cls_hidden = torch.zeros(B, self.hidden_size, device=self.device, dtype=last_hidden.dtype)
        for i in range(B):
            cls_hidden[i] = last_hidden[i, cls_positions[i], :]
        
        # 4. 投影CLS隐向量（残差连接）
        cls_projected = cls_hidden + self.cls_projector(cls_hidden)
        
        # 5. 使用该数据集的Prototype计算logits
        logits = self.prototype_bank.logits(ds_id, cls_projected)  # [B, num_classes_ds]
        
        # 6. 计算损失
        labels = torch.tensor(
            [sample["label_index"] for sample in batch],
            device=self.device,
            dtype=torch.long
        )
        loss = F.cross_entropy(logits, labels)
        
        if return_hidden:
            return loss, logits, cls_hidden
        return loss, logits
    
    def forward(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """DDP兼容的forward方法"""
        loss, _ = self.forward_prototype(batch)
        return loss
    
    @torch.no_grad()
    def predict(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        预测类别
        
        Returns:
            predictions: [B] 预测的类别索引
        """
        self.eval()
        _, logits = self.forward_prototype(batch)
        return logits.argmax(dim=-1)
    
    def get_trainable_parameters_for_stage(self, stage: int) -> Dict[str, List[torch.nn.Parameter]]:
        """
        获取指定阶段的可训练参数分组
        
        Args:
            stage: 0 或 1
        
        Returns:
            参数组字典，key为组名，value为参数列表
        """
        param_groups = {}
        
        if stage == 0:
            # Stage 0: 只训练 banks + cls
            param_groups["prompt_bank"] = list(self.prompt_bank.parameters())
            param_groups["prototype_bank"] = list(self.prototype_bank.parameters())
            param_groups["cls"] = [self.cls_embed] + list(self.cls_projector.parameters())
        
        elif stage == 1:
            # Stage 1: 训练更多组件
            param_groups["encoder"] = list(self.encoder.parameters())
            param_groups["projector"] = list(self.projector.parameters())
            param_groups["prompt_bank"] = list(self.prompt_bank.parameters())
            param_groups["prototype_bank"] = list(self.prototype_bank.parameters())
            param_groups["cls"] = [self.cls_embed] + list(self.cls_projector.parameters())
            
            if self.lora_enabled:
                param_groups["lora"] = self.get_lora_parameters()
        
        return param_groups
    
    def store_to_file(self, path: str):
        """保存模型到文件"""
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
            "prompt_bank_state": self.prompt_bank.state_dict(),
            "prototype_bank_state": self.prototype_bank.state_dict(),
            "cls_embed": self.cls_embed.data,
            "cls_projector_state": self.cls_projector.state_dict(),
            "prompt_len": self.prompt_len,
            "num_datasets": self.num_datasets,
        }
        
        # LoRA状态
        self.save_lora_state_to_checkpoint(checkpoint)
        
        torch.save(checkpoint, path)
        print(f"💾 Saved OpenTSLMMultiDataset to: {path}")
    
    def load_from_file(self, path: str):
        """从文件加载模型"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        
        if "prompt_bank_state" in ckpt:
            self.prompt_bank.load_state_dict(ckpt["prompt_bank_state"])
        if "prototype_bank_state" in ckpt:
            self.prototype_bank.load_state_dict(ckpt["prototype_bank_state"])
        
        self.cls_embed.data = ckpt["cls_embed"].to(self.device)
        if "cls_projector_state" in ckpt:
            self.cls_projector.load_state_dict(ckpt["cls_projector_state"])
        
        # LoRA状态
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        
        print(f"📥 Loaded OpenTSLMMultiDataset from: {path}")
