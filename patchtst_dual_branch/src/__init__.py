# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
PatchTST + VisionEncoder 双分支时序分类模型

独立项目包，包含：
- dual_branch_model: 双分支模型主体
- vision_encoder: 视觉编码器（TiViT风格）
- aggregator: Transformer聚合器
- projector: 投影层模块
- ucr_dataset: UCR数据集加载
- train_dual_branch_tivit: 训练脚本
"""

from .dual_branch_model import PatchTSTWithVisionBranch
from .vision_encoder import VisionEncoder
from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector
from .ucr_dataset import UCRDatasetForPatchTST, get_dataset_info

__all__ = [
    "PatchTSTWithVisionBranch",
    "VisionEncoder",
    "SmallTransformerAggregator",
    "MLPProjector",
    "LinearProjector",
    "UCRDatasetForPatchTST",
    "get_dataset_info",
]
