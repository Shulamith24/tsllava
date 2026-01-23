# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
OpenTSLMPrototype: 基于Prototype的时间序列分类模型

核心架构:
- 输入序列: [Learnable Prompt] + [TS_TOKENS] + [CLS]
- 输出: CLS隐向量 → Prototype头 → logits

与原始OpenTSLMSP的区别:
1. 使用可学习的Prompt tokens替代自然语言prompt
2. 使用CLS token提取表征（放在TS tokens之后）
3. 使用Prototype + 余弦相似度 + 温度进行分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from torch.nn.utils.rnn import pad_sequence

from .OpenTSLMSP import OpenTSLMSP


class PrototypeClassificationHead(nn.Module):
    """
    Prototype分类头
    
    使用余弦相似度 + 可学习温度进行分类
    logits = cosine_similarity(z, prototypes) / temperature
    """
    
    def __init__(self, hidden_size: int, num_classes: int, init_temperature: float = 1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Prototype矩阵: [num_classes, hidden_size]
        self.prototypes = nn.Parameter(torch.randn(num_classes, hidden_size) * 0.02)
        
        # 可学习温度参数
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
    
    @property
    def temperature(self) -> torch.Tensor:
        """返回温度值（通过log确保始终为正）"""
        return self.log_temperature.exp().clamp(min=0.01, max=100.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算分类logits
        
        Args:
            z: CLS隐向量 [batch_size, hidden_size]
            
        Returns:
            logits: [batch_size, num_classes]
        """
        # L2归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # 余弦相似度
        similarity = torch.matmul(z_norm, proto_norm.T)  # [B, num_classes]
        
        # 温度缩放
        logits = similarity / self.temperature
        
        return logits


class OpenTSLMPrototype(OpenTSLMSP):
    """
    基于Prototype的时间序列分类模型
    
    输入序列结构:
        [Prompt (10 tokens)] + [TS_tokens (L tokens)] + [CLS (1 token)]
    
    输出:
        - 取CLS位置的隐向量
        - 通过Prototype头计算分类logits
    
    Args:
        llm_id: LLM模型ID
        device: 设备
        encoder_type: 编码器类型
        prompt_len: 可学习prompt的长度
        num_classes: 分类类别数
        init_temperature: 温度初始值
        **kwargs: 其他传递给OpenTSLMSP的参数
    """
    
    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        encoder_type: str = "transformer_cnn",
        prompt_len: int = 10,
        num_classes: int = 2,
        init_temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(
            llm_id=llm_id,
            device=device,
            encoder_type=encoder_type,
            **kwargs
        )
        
        # 获取LLM隐层维度
        self.hidden_size = self.llm.config.hidden_size
        self.prompt_len = prompt_len
        self.num_classes = num_classes
        
        # 可学习的Prompt tokens
        self.prompt_embeds = nn.Parameter(
            torch.randn(prompt_len, self.hidden_size, device=device) * 0.02
        )
        
        # CLS token
        self.cls_embed = nn.Parameter(
            torch.randn(self.hidden_size, device=device) * 0.02
        )
        
        # Prototype分类头
        self.cls_head = PrototypeClassificationHead(
            self.hidden_size,
            num_classes,
            init_temperature
        ).to(device)
    
    def freeze_backbone(self):
        """
        Stage 0: 冻结主干网络
        只训练 prompt_embeds + cls_embed + cls_head (prototypes + temperature)
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
        self.prompt_embeds.requires_grad = True
        self.cls_embed.requires_grad = True
        for param in self.cls_head.parameters():
            param.requires_grad = True
        
        print("🧊 Stage 0: Backbone frozen (encoder + projector + LLM)")
        print("   训练参数: prompt_embeds, cls_embed, cls_head (prototypes + temperature)")
    
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
    
    def _build_prototype_input_embeds(
        self,
        batch: List[Dict[str, any]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建Prototype模型的输入embedding
        
        输入格式: [Prompt] + [TS_tokens] + [CLS]
        
        Args:
            batch: 批次数据，每个样本需包含:
                - time_series: List[Tensor] 时间序列数据
                - label_index: int 类别索引（可选，用于计算loss）
        
        Returns:
            inputs_embeds: [B, seq_len, H]
            attention_mask: [B, seq_len]
            cls_positions: [B] CLS token在每个样本中的位置
        """
        device = self.device
        B = len(batch)
        H = self.hidden_size
        
        # 1. 处理时间序列
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
            ts_proj = self.projector(ts_enc).to(self.prompt_embeds.dtype)  # [B, N_patches, H]
        else:
            ts_proj = torch.empty(B, 0, H, device=device, dtype=self.prompt_embeds.dtype)
        
        # 2. 构建每个样本的序列
        all_seq_embeds = []
        all_seq_masks = []
        cls_positions = []
        
        # Prompt embedding (共享)
        prompt_embeds = self.prompt_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, prompt_len, H]
        
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
                ts_tokens = torch.empty(0, H, device=device, dtype=self.prompt_embeds.dtype)
            
            # 构建序列: [Prompt] + [TS_tokens] + [CLS]
            seq_embeds = torch.cat([
                self.prompt_embeds,  # [prompt_len, H]
                ts_tokens,          # [total_patches, H]
                self.cls_embed.unsqueeze(0)  # [1, H]
            ], dim=0)
            
            # 计算CLS位置
            cls_pos = self.prompt_len + ts_tokens.size(0)
            cls_positions.append(cls_pos)
            
            # Attention mask (全1)
            seq_mask = torch.ones(seq_embeds.size(0), device=device, dtype=torch.long)
            
            all_seq_embeds.append(seq_embeds)
            all_seq_masks.append(seq_mask)
        
        # 3. Pad到统一长度
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
        
        Args:
            batch: 批次数据，每个样本需包含:
                - time_series: List[Tensor] 时间序列数据
                - label_index: int 类别索引
            return_hidden: 是否返回隐向量
        
        Returns:
            loss: 交叉熵损失
            logits: [B, num_classes]
            (可选) hidden: [B, H] CLS隐向量
        """
        # 1. 构建输入
        inputs_embeds, attention_mask, cls_positions = self._build_prototype_input_embeds(batch)
        
        # 2. LLM前向传播
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # 3. 提取CLS位置的隐向量
        # outputs.hidden_states[-1]: [B, seq_len, H] (最后一层的隐状态)
        last_hidden = outputs.hidden_states[-1]
        
        B = len(batch)
        cls_hidden = torch.zeros(B, self.hidden_size, device=self.device, dtype=last_hidden.dtype)
        for i in range(B):
            cls_hidden[i] = last_hidden[i, cls_positions[i], :]
        
        # 4. Prototype分类
        logits = self.cls_head(cls_hidden)  # [B, num_classes]
        
        # 5. 计算损失
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
        """
        DDP兼容的forward方法
        """
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
            # Stage 0: 只训练 prompt + cls + cls_head
            param_groups["prompt_cls"] = [self.prompt_embeds, self.cls_embed]
            param_groups["cls_head"] = list(self.cls_head.parameters())
        
        elif stage == 1:
            # Stage 1: 训练更多组件
            param_groups["encoder"] = list(self.encoder.parameters())
            param_groups["projector"] = list(self.projector.parameters())
            param_groups["prompt_cls"] = [self.prompt_embeds, self.cls_embed]
            param_groups["cls_head"] = list(self.cls_head.parameters())
            
            if self.lora_enabled:
                param_groups["lora"] = self.get_lora_parameters()
        
        return param_groups
    
    def store_to_file(self, path: str):
        """保存模型到文件"""
        checkpoint = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
            "prompt_embeds": self.prompt_embeds.data,
            "cls_embed": self.cls_embed.data,
            "cls_head_state": self.cls_head.state_dict(),
            "prompt_len": self.prompt_len,
            "num_classes": self.num_classes,
        }
        
        # LoRA状态
        self.save_lora_state_to_checkpoint(checkpoint)
        
        torch.save(checkpoint, path)
        print(f"💾 Saved OpenTSLMPrototype to: {path}")
    
    def load_from_file(self, path: str):
        """从文件加载模型"""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])
        self.prompt_embeds.data = ckpt["prompt_embeds"].to(self.device)
        self.cls_embed.data = ckpt["cls_embed"].to(self.device)
        self.cls_head.load_state_dict(ckpt["cls_head_state"])
        
        # LoRA状态
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        
        print(f"📥 Loaded OpenTSLMPrototype from: {path}")
