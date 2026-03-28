#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reporting.pipeline import generate_report  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-ready UCR report artifacts from one or more batch result ledgers."
    )
    parser.add_argument("--report-config", required=True, help="Path to the report JSON config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_report(args.report_config)
    print(f"Report ready: {manifest['report_name']}")
    print(f"Kind/stage: {manifest['report_kind']} / {manifest['report_stage']}")
    print(f"Output dir: {Path(manifest['generated_files'][0]).resolve().parent}")
    print(f"Datasets: {manifest['dataset_count']} | shots: {','.join(manifest['shots'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
