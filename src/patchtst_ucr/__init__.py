# PatchTST UCR 分类模块
"""
独立的PatchTST在UCR数据集上的分类模块，不依赖opentslm包。
"""

from .ucr_loader import ensure_ucr_data, load_ucr_dataset, get_all_ucr_datasets
from .ucr_dataset import UCRDatasetForPatchTST, get_dataset_info
from .aggregator import SmallTransformerAggregator
from .model import PatchTSTWithAggregator

__all__ = [
    "ensure_ucr_data",
    "load_ucr_dataset",
    "get_all_ucr_datasets",
    "UCRDatasetForPatchTST",
    "get_dataset_info",
    "SmallTransformerAggregator",
    "PatchTSTWithAggregator",
]
