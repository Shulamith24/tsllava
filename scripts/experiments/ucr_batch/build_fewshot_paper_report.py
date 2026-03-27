#!/usr/bin/env python3
"""
读入json配置文件，示范位于scripts/experiments/ucr_batch/reporting/examples/fewshot_report_config.json
根据配置文件中的模型列表和数据集路径，生成一个包含所有模型在指定数据集上的few-shot性能比较的报告。
json配置文件中包含以下字段:
    "report_name": 结果文件夹名称
    "dataset_source": 标准数据集全集的来源，之后用于构造结果数据集交集
    "coverage_mode": "strict 每个模型必须全覆盖指定数据集和shots"或"intersection 取所有模型结果数据集的交集"
    "shots": [1, 5, 10] 需要包含的shot数量列表
    "models": {"key":模型内部表示，
    "label": 结果里展示的模型名称，
    "results_txt": 结果文件路径，
    "color": 结果里展示的模型颜色,
    "primary": true/false 是否是主模型，主模型会在结果里突出显示
    "marker": 趋势图里面的marker样式
    }


生成内容包括：
1. 

"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reporting.pipeline import generate_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build paper-ready few-shot UCR report artifacts from one or more "
            "results.txt ledgers."
        )
    )
    parser.add_argument("--report-config", required=True, help="Path to the report JSON config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_report(args.report_config)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Output dir: {Path(manifest['generated_files'][0]).resolve().parent}")
    print(f"Datasets: {manifest['dataset_count']} | shots: {','.join(manifest['shots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
