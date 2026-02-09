# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + VisionEncoder 双分支时序分类模型

独立项目包，包含：
- dual_branch_model: 双分支模型主体 (TransformerAggregator 分类)
- dual_branch_llm_model: 双分支 LLM 模型 (LLM 生成分类)
- vision_encoder: 视觉编码器（TiViT风格）
- aggregator: Transformer聚合器
- projector: 投影层模块
- ucr_dataset: UCR数据集加载
- ucr_llm_dataset: UCR LLM 分类数据集
- train_dual_branch_tivit: TransformerAggregator 训练脚本
- train_dual_branch_llm: LLM 分类训练脚本
"""

from .dual_branch_model import PatchTSTWithVisionBranch
from .vision_encoder import VisionEncoder
from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector
from .ucr_dataset import UCRDatasetForPatchTST, get_dataset_info

# LLM 分类模块（延迟导入以避免不需要时的依赖问题）
# from .dual_branch_llm_model import DualBranchLLMModel
# from .ucr_llm_dataset import UCRLLMClassificationDataset

__all__ = [
    # 原有模块
    "PatchTSTWithVisionBranch",
    "VisionEncoder",
    "SmallTransformerAggregator",
    "MLPProjector",
    "LinearProjector",
    "UCRDatasetForPatchTST",
    "get_dataset_info",
    # LLM 模块 (按需导入)
    # "DualBranchLLMModel",
    # "UCRLLMClassificationDataset",
]

