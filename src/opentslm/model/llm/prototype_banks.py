# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PromptBank 和 PrototypeBank

多数据集统一训练的核心组件：
- PromptBank: 每数据集的可学习Prompt Tokens
- PrototypeBank: 每数据集的Prototype矩阵 + 可学习温度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PromptBank(nn.Module):
    """
    可学习 Prompt Token Bank
    
    每个数据集有独立的 prompt embeddings。
    
    参数形状: [num_datasets, prompt_len, hidden_size]
    
    Args:
        num_datasets: 数据集总数
        prompt_len: 每个数据集的prompt长度
        hidden_size: LLM隐层维度
        init_mean: 用于初始化的均值向量 (可选)
        init_std: 用于初始化的标准差向量 (可选)
        dtype: 数据类型
    """
    
    def __init__(
        self,
        num_datasets: int,
        prompt_len: int,
        hidden_size: int,
        init_mean: Optional[torch.Tensor] = None,
        init_std: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_datasets = num_datasets
        self.prompt_len = prompt_len
        self.hidden_size = hidden_size
        
        # 初始化 prompt embeddings
        if init_mean is not None and init_std is not None:
            # 使用LLM embedding统计信息初始化
            # [num_datasets, prompt_len, hidden_size]
            prompt_init = init_mean.view(1, 1, -1).expand(num_datasets, prompt_len, -1).clone()
            noise = torch.randn(num_datasets, prompt_len, hidden_size, device=device, dtype=dtype)
            # 增大扰动系数: 0.1 -> 0.5
            prompt_init = prompt_init + noise * init_std.view(1, 1, -1) * 0.5
            self.prompts = nn.Parameter(prompt_init)
        else:
            # 随机初始化
            self.prompts = nn.Parameter(
                torch.randn(num_datasets, prompt_len, hidden_size, device=device, dtype=dtype) * 0.02
            )
        
        print(f"📝 PromptBank: {num_datasets} datasets × {prompt_len} tokens × {hidden_size} dim")
    
    def get(self, ds_id: int) -> torch.Tensor:
        """
        获取指定数据集的prompt embeddings
        
        Args:
            ds_id: 数据集ID
            
        Returns:
            [prompt_len, hidden_size]
        """
        return self.prompts[ds_id]
    
    def get_batch(self, ds_ids: torch.Tensor) -> torch.Tensor:
        """
        获取一批数据集的prompt embeddings
        
        Args:
            ds_ids: [B] 数据集ID张量
            
        Returns:
            [B, prompt_len, hidden_size]
        """
        return self.prompts[ds_ids]


class PrototypeBankEntry(nn.Module):
    """
    单个数据集的Prototype + 温度
    
    包含：
    - prototypes: [num_classes, hidden_size]
    - log_temperature: 可学习温度（log空间）
    """
    
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        init_temperature: float = 1.0,
        init_mean: Optional[torch.Tensor] = None,
        init_std: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # Prototype矩阵
        if init_mean is not None and init_std is not None:
            # 使用LLM embedding统计信息初始化
            # [num_classes, hidden_size]
            proto_init = init_mean.unsqueeze(0).expand(num_classes, -1).clone()
            noise = torch.randn(num_classes, hidden_size, device=device, dtype=dtype)
            
            # 增大扰动系数: 0.1 -> 1.0，确保prototype之间有足够的区分度
            # 对于多类别数据集，如果prototype太接近，会导致很难训练
            proto_init = proto_init + noise * init_std * 1.0
            self.prototypes = nn.Parameter(proto_init)
        else:
            # 随机正交初始化 (如果类别数 <= hidden_size)
            if num_classes <= hidden_size:
                # 正交初始化能最大化初始区分度
                weight = torch.empty(num_classes, hidden_size, device=device, dtype=dtype)
                nn.init.orthogonal_(weight)
                self.prototypes = nn.Parameter(weight * 0.1)  # 缩放以匹配通常的embedding范数
            else:
                self.prototypes = nn.Parameter(
                    torch.randn(num_classes, hidden_size, device=device, dtype=dtype) * 0.02
                )
        
        # 可学习温度（log空间确保为正）
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(init_temperature, dtype=torch.float32, device=device))
        )
    
    @property
    def temperature(self) -> torch.Tensor:
        """返回温度值"""
        return self.log_temperature.exp().clamp(min=0.01, max=100.0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算logits
        
        Args:
            z: [B, hidden_size] CLS隐向量
            
        Returns:
            [B, num_classes] logits
        """
        # L2归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # 余弦相似度
        similarity = torch.matmul(z_norm, proto_norm.T)  # [B, num_classes]
        
        # 温度缩放
        logits = similarity / self.temperature
        
        return logits


class PrototypeBank(nn.Module):
    """
    Prototype矩阵 + 温度 Bank
    
    每个数据集有独立的：
    - P_i: [C_i, hidden_size] prototype矩阵
    - τ_i: 可学习温度标量
    
    Args:
        class_counts: {ds_id: num_classes} 每个数据集的类别数
        hidden_size: LLM隐层维度
        init_temperature: 温度初始值
        init_mean: 用于初始化的均值向量 (可选)
        init_std: 用于初始化的标准差向量 (可选)
        dtype: 数据类型
    """
    
    def __init__(
        self,
        class_counts: Dict[int, int],
        hidden_size: int,
        init_temperature: float = 1.0,
        init_mean: Optional[torch.Tensor] = None,
        init_std: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.class_counts = class_counts
        
        # 为每个数据集创建 PrototypeBankEntry
        self.entries = nn.ModuleDict()
        for ds_id, num_classes in class_counts.items():
            self.entries[str(ds_id)] = PrototypeBankEntry(
                num_classes=num_classes,
                hidden_size=hidden_size,
                init_temperature=init_temperature,
                init_mean=init_mean,
                init_std=init_std,
                dtype=dtype,
                device=device,
            )
        
        print(f"🎯 PrototypeBank: {len(class_counts)} datasets")
        for ds_id, num_classes in class_counts.items():
            print(f"   [{ds_id}]: {num_classes} classes")
    
    def logits(self, ds_id: int, z_cls: torch.Tensor) -> torch.Tensor:
        """
        计算指定数据集的分类logits
        
        Args:
            ds_id: 数据集ID
            z_cls: [B, hidden_size] CLS隐向量
            
        Returns:
            [B, num_classes] logits
        """
        return self.entries[str(ds_id)](z_cls)
    
    def get_temperature(self, ds_id: int) -> float:
        """获取指定数据集的温度值"""
        return self.entries[str(ds_id)].temperature.item()
    
    def get_prototypes(self, ds_id: int) -> torch.Tensor:
        """获取指定数据集的prototype矩阵"""
        return self.entries[str(ds_id)].prototypes
    
    def get_num_classes(self, ds_id: int) -> int:
        """获取指定数据集的类别数"""
        return self.class_counts[ds_id]


# 测试
if __name__ == "__main__":
    print("Testing PromptBank and PrototypeBank...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试 PromptBank
    prompt_bank = PromptBank(
        num_datasets=3,
        prompt_len=10,
        hidden_size=256,
        device=device,
    )
    
    prompt_0 = prompt_bank.get(0)
    print(f"Prompt 0 shape: {prompt_0.shape}")  # [10, 256]
    
    ds_ids = torch.tensor([0, 1, 2], device=device)
    prompts = prompt_bank.get_batch(ds_ids)
    print(f"Batch prompts shape: {prompts.shape}")  # [3, 10, 256]
    
    # 测试 PrototypeBank
    class_counts = {0: 2, 1: 5, 2: 10}
    proto_bank = PrototypeBank(
        class_counts=class_counts,
        hidden_size=256,
        device=device,
    )
    
    z = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    logits_0 = proto_bank.logits(0, z)
    logits_1 = proto_bank.logits(1, z)
    print(f"Logits for ds_id=0: {logits_0.shape}")  # [4, 2]
    print(f"Logits for ds_id=1: {logits_1.shape}")  # [4, 5]
    
    print(f"Temperature for ds_id=0: {proto_bank.get_temperature(0):.4f}")
    
    print("\n✅ All tests passed!")
