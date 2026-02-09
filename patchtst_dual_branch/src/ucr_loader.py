# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""
UCR 数据集加载器

从 opentslm.time_series_datasets.ucr.ucr_loader 复制并简化，
移除对 opentslm 包的依赖。
"""

import os
import zipfile
import requests
from typing import Tuple
from pathlib import Path

import pandas as pd

# ---------------------------
# Constants
# ---------------------------

DEFAULT_RAW_DATA_PATH = "./data"
UCR_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"


def ensure_ucr_data(
    raw_data_path: str = DEFAULT_RAW_DATA_PATH,
    url: str = UCR_URL,
):
    """
    1) Download the UCRArchive_2018.zip if missing.
    2) Extract it to `raw_data_path/UCRArchive_2018`.
    """
    ucr_zip = os.path.join(raw_data_path, "UCRArchive_2018.zip")
    ucr_dir = os.path.join(raw_data_path, "UCRArchive_2018")
    
    # Create data directory
    os.makedirs(raw_data_path, exist_ok=True)

    # If already extracted, skip
    if os.path.isdir(ucr_dir):
        return

    # 1) Download ZIP if needed
    if not os.path.isfile(ucr_zip):
        print(f"Downloading UCR Archive from {url} …")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with open(ucr_zip, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    # 2) Extract outer ZIP (password is 'someone')
    print(f"Extracting {ucr_zip} …")
    with zipfile.ZipFile(ucr_zip, "r") as z:
        z.setpassword(b'someone')
        z.extractall(raw_data_path)


def load_ucr_dataset(
    dataset_name: str,
    raw_data_path: str = DEFAULT_RAW_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the TRAIN and TEST TSVs for a given UCR dataset.

    Args:
        dataset_name: Name of the dataset folder (e.g. "ECG5000").
        raw_data_path: Base path where the archive is extracted.

    Returns:
        train_df, test_df: DataFrames with columns ["label", "t1", "t2", …].
    """
    ensure_ucr_data(raw_data_path=raw_data_path)

    base = os.path.join(raw_data_path, "UCRArchive_2018", dataset_name)
    train_path = os.path.join(base, f"{dataset_name}_TRAIN.tsv")
    test_path = os.path.join(base, f"{dataset_name}_TEST.tsv")

    # Load using pandas; first column is label, rest are the series values
    train_df = pd.read_csv(train_path, sep="\t", header=None)
    test_df = pd.read_csv(test_path, sep="\t", header=None)

    # Rename columns: 0 → "label", 1...N → "t1","t2",…
    n_cols = train_df.shape[1] - 1
    col_names = ["label"] + [f"t{i}" for i in range(1, n_cols + 1)]
    train_df.columns = col_names
    test_df.columns = col_names

    return train_df, test_df


def get_all_ucr_datasets(raw_data_path: str = DEFAULT_RAW_DATA_PATH) -> list:
    """
    获取所有可用的UCR数据集名称列表
    
    Args:
        raw_data_path: 数据根目录
    
    Returns:
        数据集名称列表（按字母排序）
    """
    ensure_ucr_data(raw_data_path=raw_data_path)
    
    ucr_dir = os.path.join(raw_data_path, "UCRArchive_2018")
    datasets = []
    
    for name in sorted(os.listdir(ucr_dir)):
        full_path = os.path.join(ucr_dir, name)
        if os.path.isdir(full_path):
            # 检查是否包含必要的tsv文件
            train_file = os.path.join(full_path, f"{name}_TRAIN.tsv")
            test_file = os.path.join(full_path, f"{name}_TEST.tsv")
            if os.path.exists(train_file) and os.path.exists(test_file):
                datasets.append(name)
    
    return datasets


# ---------------------------
# Test
# ---------------------------

if __name__ == "__main__":
    datasets = get_all_ucr_datasets()
    print(f"Found {len(datasets)} UCR datasets")
    print(f"First 5: {datasets[:5]}")
    
    # Test loading one dataset
    train_df, test_df = load_ucr_dataset("ECG200")
    print(f"\nECG200:")
    print(f"  Train shape: {train_df.shape}")
    print(f"  Test shape: {test_df.shape}")
    print(f"  Labels: {sorted(train_df['label'].unique())}")
