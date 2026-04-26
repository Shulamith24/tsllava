# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
UEA多变量时间序列数据集加载器,在 train_uea_classification.py时使用

使用aeon库加载UEA数据集。
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

DEFAULT_UEA_EXTRACT_PATH = Path(__file__).resolve().parents[4] / "data" / "Multivariate_ts"

# 尝试导入aeon库
try:
    from aeon.datasets import load_classification
    AEON_AVAILABLE = True
except ImportError:
    AEON_AVAILABLE = False
    print("⚠️ aeon库未安装。请运行: pip install aeon")


def ensure_uea_data():
    """确保aeon库可用"""
    if not AEON_AVAILABLE:
        raise ImportError(
            "aeon库未安装。请运行: pip install aeon\n"
            "或: uv add aeon"
        )


def load_uea_dataset(
    dataset_name: str,
    extract_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载UEA多变量时间序列数据集
    
    Args:
        dataset_name: 数据集名称 (e.g. "Handwriting", "BasicMotions")
        extract_path: 数据解压路径（可选）。默认使用仓库下的 data/Multivariate_ts，
            也可通过环境变量 OPENTSLM_UEA_EXTRACT_PATH 覆盖。
    
    Returns:
        X_train: [N_train, C, L] 训练数据
        y_train: [N_train] 训练标签
        X_test: [N_test, C, L] 测试数据
        y_test: [N_test] 测试标签
    """
    ensure_uea_data()
    resolved_extract_path = extract_path or os.environ.get("OPENTSLM_UEA_EXTRACT_PATH") or str(DEFAULT_UEA_EXTRACT_PATH)
    
    print(f"📂 Loading UEA dataset: {dataset_name}")
    
    # 加载训练集
    X_train, y_train = load_classification(
        name=dataset_name,
        split="train",
        extract_path=resolved_extract_path,
    )
    
    # 加载测试集
    X_test, y_test = load_classification(
        name=dataset_name,
        split="test",
        extract_path=resolved_extract_path,
    )
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Channels: {X_train.shape[1]}, Length: {X_train.shape[2]}")
    print(f"   Classes: {len(np.unique(y_train))}")
    
    return X_train, y_train, X_test, y_test


def get_uea_dataset_info(dataset_name: str) -> dict:
    """获取UEA数据集的基本信息"""
    ensure_uea_data()
    
    X_train, y_train, X_test, y_test = load_uea_dataset(dataset_name)
    
    return {
        "name": dataset_name,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_channels": X_train.shape[1],
        "length": X_train.shape[2],
        "n_classes": len(np.unique(y_train)),
        "classes": sorted(np.unique(y_train).tolist()),
    }


# 常用UEA数据集列表
UEA_DATASETS = [
    "Handwriting",
    "BasicMotions",
    "Epilepsy",
    "NATOPS",
    "RacketSports",
    "FingerMovements",
    "HandMovementDirection",
    "SelfRegulationSCP1",
    "SelfRegulationSCP2",
    "UWaveGestureLibrary",
]


if __name__ == "__main__":
    # 测试加载
    if AEON_AVAILABLE:
        info = get_uea_dataset_info("Epilepsy")
        print(f"\n📊 Dataset info: {info}")
    else:
        print("请安装aeon: pip install aeon")
