# PatchTST UCR 分类模块
"""
独立的PatchTST在UCR数据集上的分类模块，不依赖opentslm包。

包含两个主要模型：
1. PatchTSTWithAggregator: 纯PatchTST + Transformer聚合头
2. PatchTSTWithVisionBranch: PatchTST + VLM图像分支双分支融合模型
"""

from .ucr_loader import ensure_ucr_data, load_ucr_dataset, get_all_ucr_datasets
from .ucr_dataset import UCRDatasetForPatchTST, get_dataset_info
from .aggregator import SmallTransformerAggregator
from .projector import MLPProjector, LinearProjector
from .model import PatchTSTWithAggregator
from .vision_encoder import VisionEncoder, ViTEncoder, ResNetEncoder, LightweightCNNEncoder
from .ts_to_image import TimeSeriesToImage, LearnableTimeSeriesToImage, SimpleTimeSeriesToImage
from .dual_branch_model import PatchTSTWithVisionBranch, CrossAttentionFusion

__all__ = [
    # 数据加载
    "ensure_ucr_data",
    "load_ucr_dataset",
    "get_all_ucr_datasets",
    "UCRDatasetForPatchTST",
    "get_dataset_info",
    # 核心组件
    "SmallTransformerAggregator",
    "MLPProjector",
    "LinearProjector",
    # 原始模型
    "PatchTSTWithAggregator",
    # 图像分支组件
    "VisionEncoder",
    "ViTEncoder",
    "ResNetEncoder",
    "LightweightCNNEncoder",
    "TimeSeriesToImage",
    "LearnableTimeSeriesToImage",
    "SimpleTimeSeriesToImage",
    # 双分支融合模型
    "PatchTSTWithVisionBranch",
    "CrossAttentionFusion",
]
