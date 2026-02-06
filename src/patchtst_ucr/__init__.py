# PatchTST UCR 分类模块
"""
独立的PatchTST在UCR数据集上的分类模块，不依赖opentslm包。
"""

from .ucr_loader import ensure_ucr_data, load_ucr_dataset
from .ucr_dataset import UCRDatasetForPatchTST

__all__ = [
    "ensure_ucr_data",
    "load_ucr_dataset",
    "UCRDatasetForPatchTST",
]
