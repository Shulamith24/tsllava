# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
TSLANet检索器

基于TSLANet的相似样本检索器，用于ICL分类的支持样本检索。

特性：
- 按类别分组建索引：为每个类别单独存储样本索引
- 按类别检索：对每个类别检索top-m最近邻，选取k_shot个
- Query排除：训练时自动排除query自身

使用流程：
1. 加载训练好的TSLANet encoder
2. 对训练集构建索引 (build_index)
3. 对每个query检索支持样本 (retrieve)
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm.auto import tqdm


class TSLANetRetriever:
    """
    基于TSLANet的相似样本检索器
    
    支持按类别分组的检索策略：
    - 离线构建按类别分组的索引
    - 对每个类别单独检索top-m最近邻
    - 从top-m中选取k_shot个作为支持样本
    
    Args:
        encoder: TSLANetEncoder实例 (需要有get_embedding方法)
        device: 计算设备
    """
    
    def __init__(
        self,
        encoder,  # TSLANetEncoder or TSLANetClassifier
        device: str = "cuda"
    ):
        self.encoder = encoder
        self.device = device
        
        # 索引数据
        self.embeddings: Optional[torch.Tensor] = None  # [N, emb_dim]
        self.labels: Optional[torch.Tensor] = None  # [N]
        self.time_series: Optional[torch.Tensor] = None  # [N, L]
        self.class_indices: Dict[int, List[int]] = {}  # 类别 -> 样本索引列表
        self.num_classes: int = 0
        
        # 确保encoder在正确设备上
        self.encoder = self.encoder.to(device)
        self.encoder.eval()
    
    @torch.no_grad()
    def build_index(
        self,
        time_series: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 64,
        show_progress: bool = True
    ):
        """
        离线构建按类别分组的索引
        
        Args:
            time_series: [N, L] 所有训练样本的时间序列
            labels: [N] 对应的标签 (0-indexed)
            batch_size: 计算embedding时的批次大小
            show_progress: 是否显示进度条
        """
        N = time_series.shape[0]
        
        # 存储原始数据
        self.time_series = time_series.cpu()
        self.labels = labels.cpu()
        
        # 计算所有embedding
        all_embeddings = []
        
        iterator = range(0, N, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Building index")
        
        for i in iterator:
            batch = time_series[i:i+batch_size].to(self.device)
            
            # 获取embedding
            if hasattr(self.encoder, 'get_embedding'):
                emb = self.encoder.get_embedding(batch)
            else:
                # 如果没有get_embedding方法，使用forward后平均池化
                features = self.encoder(batch)  # [B, N_patches, dim]
                emb = features.mean(dim=1)  # [B, dim]
            
            all_embeddings.append(emb.cpu())
        
        self.embeddings = torch.cat(all_embeddings, dim=0)  # [N, emb_dim]
        
        # L2归一化用于余弦相似度
        self.embeddings = F.normalize(self.embeddings, p=2, dim=-1)
        
        # 按类别分组建索引
        self.class_indices = {}
        unique_labels = torch.unique(self.labels).tolist()
        self.num_classes = len(unique_labels)
        
        for cls in unique_labels:
            mask = (self.labels == cls)
            indices = torch.where(mask)[0].tolist()
            self.class_indices[cls] = indices
        
        print(f"✅ 索引构建完成: {N} 样本, {self.num_classes} 类别")
        for cls, indices in self.class_indices.items():
            print(f"   类别 {cls}: {len(indices)} 样本")
    
    @torch.no_grad()
    def retrieve(
        self,
        query_emb: torch.Tensor,
        query_idx: Optional[int] = None,
        k_shot: int = 1,
        top_m: int = 10,
        exclude_query: bool = True,
        target_labels: Optional[List[int]] = None
    ) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        """
        按类别检索支持样本
        
        对每个类别：
        1. 计算query与该类所有样本的相似度
        2. 取top-m最相似的样本
        3. 从top-m中选取前k_shot个（排除query自身）
        
        Args:
            query_emb: [emb_dim] query的embedding (已归一化)
            query_idx: query在索引中的全局索引 (用于排除自身)
            k_shot: 每个类别选取的支持样本数
            top_m: 每个类别检索的候选数量
            exclude_query: 是否排除query自身
            target_labels: 目标类别列表 (只从这些类别中检索, None表示检索所有类别)
        
        Returns:
            support_indices: 支持样本的全局索引列表
            support_ts: 支持样本的时间序列列表
            support_labels: 支持样本的标签列表
        """
        if self.embeddings is None:
            raise RuntimeError("请先调用 build_index() 构建索引")
        
        # 确定query_emb归一化
        query_emb = F.normalize(query_emb.cpu().unsqueeze(0), p=2, dim=-1).squeeze(0)
        
        support_indices = []
        support_ts = []
        support_labels = []
        
        # 确定要检索的类别
        if target_labels is not None:
            classes_to_search = [cls for cls in target_labels if cls in self.class_indices]
        else:
            classes_to_search = sorted(self.class_indices.keys())
        
        # 对每个类别检索
        for cls in classes_to_search:
            cls_global_indices = self.class_indices[cls]
            
            if len(cls_global_indices) == 0:
                continue
            
            # 获取该类别所有样本的embedding
            cls_embs = self.embeddings[cls_global_indices]  # [N_cls, emb_dim]
            
            # 计算余弦相似度 (由于已归一化，点积=余弦相似度)
            similarities = torch.matmul(cls_embs, query_emb)  # [N_cls]
            
            # 排序取top-m
            sorted_local_indices = similarities.argsort(descending=True)
            
            # 选取k_shot个（排除query自身）
            count = 0
            for local_idx in sorted_local_indices:
                if count >= k_shot:
                    break
                
                global_idx = cls_global_indices[local_idx.item()]
                
                # 排除query自身
                if exclude_query and query_idx is not None and global_idx == query_idx:
                    continue
                
                support_indices.append(global_idx)
                support_ts.append(self.time_series[global_idx])
                support_labels.append(cls)
                count += 1
                
                # 已达到top_m限制 (但优先保证k_shot)
                if local_idx.item() >= top_m - 1 and count < k_shot:
                    # 如果在top_m内还没凑够k_shot，继续找
                    pass
        
        return support_indices, support_ts, support_labels
    
    @torch.no_grad()
    def retrieve_for_query(
        self,
        query_ts: torch.Tensor,
        query_idx: Optional[int] = None,
        k_shot: int = 1,
        top_m: int = 10,
        exclude_query: bool = True,
        target_labels: Optional[List[int]] = None
    ) -> Tuple[List[int], List[torch.Tensor], List[int]]:
        """
        给定query时间序列，检索支持样本
        
        Args:
            query_ts: [L] query时间序列
            query_idx: query在索引中的全局索引
            k_shot: 每个类别选取的支持样本数
            top_m: 每个类别检索的候选数量
            exclude_query: 是否排除query自身
            target_labels: 目标类别列表 (只从这些类别中检索, None表示检索所有类别)
        
        Returns:
            同 retrieve()
        """
        # 计算query的embedding
        query_ts = query_ts.unsqueeze(0).to(self.device)  # [1, L]
        
        if hasattr(self.encoder, 'get_embedding'):
            query_emb = self.encoder.get_embedding(query_ts)  # [1, emb_dim]
        else:
            features = self.encoder(query_ts)
            query_emb = features.mean(dim=1)
        
        query_emb = query_emb.squeeze(0).cpu()  # [emb_dim]
        
        return self.retrieve(query_emb, query_idx, k_shot, top_m, exclude_query, target_labels)
    
    def save_index(self, path: str):
        """保存索引到文件"""
        torch.save({
            "embeddings": self.embeddings,
            "labels": self.labels,
            "time_series": self.time_series,
            "class_indices": self.class_indices,
            "num_classes": self.num_classes
        }, path)
        print(f"💾 索引已保存到: {path}")
    
    def load_index(self, path: str):
        """从文件加载索引"""
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.time_series = data["time_series"]
        self.class_indices = data["class_indices"]
        self.num_classes = data["num_classes"]
        print(f"📂 索引已加载: {len(self.labels)} 样本, {self.num_classes} 类别")
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        return {cls: len(indices) for cls, indices in self.class_indices.items()}


# --- 测试代码 ---
if __name__ == "__main__":
    print("Testing TSLANetRetriever...")
    
    # 模拟一个简单的encoder
    class MockEncoder:
        def __init__(self, emb_dim=128):
            self.emb_dim = emb_dim
        
        def to(self, device):
            return self
        
        def eval(self):
            pass
        
        def get_embedding(self, x):
            # 随机返回embedding
            B = x.shape[0]
            return torch.randn(B, self.emb_dim)
    
    # 创建检索器
    encoder = MockEncoder()
    retriever = TSLANetRetriever(encoder, device="cpu")
    
    # 构建索引
    N, L = 100, 50
    time_series = torch.randn(N, L)
    labels = torch.randint(0, 5, (N,))  # 5个类别
    
    retriever.build_index(time_series, labels, batch_size=32)
    
    # 测试检索
    query_ts = torch.randn(L)
    indices, ts_list, label_list = retriever.retrieve_for_query(
        query_ts, query_idx=None, k_shot=2, top_m=10
    )
    
    print(f"检索到 {len(indices)} 个支持样本")
    print(f"标签分布: {sorted(label_list)}")
    
    # 测试保存/加载
    retriever.save_index("test_index.pt")
    retriever.load_index("test_index.pt")
    os.remove("test_index.pt")
    
    print("✅ 测试通过!")
