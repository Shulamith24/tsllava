#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Ablation wrapper for the standalone PatchTST UCR few-shot script."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ORIGINAL_SCRIPT = PROJECT_ROOT / "scripts" / "train_ucr_patchtst_classification_fewshot.py"
DEFAULT_FEWSHOT_SAVE_DIR = "results/ablations/patchtst_ucr_fewshot"
DEFAULT_FULL_SAVE_DIR = "results/ablations/patchtst_ucr_full"
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data")


def load_original_module():
    spec = importlib.util.spec_from_file_location(
        "patchtst_ucr_fewshot_original",
        ORIGINAL_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load original script from {ORIGINAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to a UCR dataset dir or a UCRArchive_2018 dir for ablation-friendly loading.",
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--save_dir", type=str, default=None)
    args, remaining = parser.parse_known_args(argv)
    args.save_dir_explicit = any(
        token == "--save_dir" or token.startswith("--save_dir=") for token in provided_argv
    )
    return args, remaining


def parse_protocol_from_argv(argv: List[str]) -> str:
    for idx, token in enumerate(argv):
        if token == "--protocol" and idx + 1 < len(argv):
            return argv[idx + 1]
        if token.startswith("--protocol="):
            return token.split("=", 1)[1]
    return "fewshot"


def resolve_ucr_source(
    *,
    data_dir: Optional[str],
    dataset: Optional[str],
    data_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    effective_dataset = dataset
    effective_data_path = data_path or DEFAULT_DATA_PATH

    if not data_dir:
        if effective_dataset is None:
            raise ValueError("--dataset is required when running the ablation wrapper without --data_dir.")
        return effective_dataset, effective_data_path

    path = Path(data_dir).resolve()

    if path.is_dir() and list(path.glob("*_TRAIN.tsv")):
        effective_dataset = effective_dataset or path.name
        if path.parent.name == "UCRArchive_2018":
            effective_data_path = str(path.parent.parent)
        else:
            effective_data_path = str(path.parent)
        return effective_dataset, effective_data_path

    if path.is_dir() and path.name == "UCRArchive_2018":
        if effective_dataset is None:
            raise ValueError("--dataset is required when --data_dir points to UCRArchive_2018.")
        effective_data_path = str(path.parent)
        return effective_dataset, effective_data_path

    if path.is_dir() and effective_dataset and (path / "UCRArchive_2018" / effective_dataset).is_dir():
        effective_data_path = str(path)
        return effective_dataset, effective_data_path

    raise ValueError(
        "Unsupported --data_dir layout. Expected either "
        "/.../UCRArchive_2018/<dataset>, /.../UCRArchive_2018 with --dataset, "
        "or the base data dir with --dataset."
    )


def build_forwarded_argv(
    *,
    wrapper_args: argparse.Namespace,
    remaining_argv: List[str],
) -> List[str]:
    protocol = parse_protocol_from_argv(remaining_argv)
    effective_dataset, effective_data_path = resolve_ucr_source(
        data_dir=wrapper_args.data_dir,
        dataset=wrapper_args.dataset,
        data_path=wrapper_args.data_path,
    )

    forwarded = list(remaining_argv)
    if effective_dataset is not None:
        forwarded.extend(["--dataset", effective_dataset])
    if effective_data_path is not None:
        forwarded.extend(["--data_path", effective_data_path])
    effective_save_dir = wrapper_args.save_dir
    if not wrapper_args.save_dir_explicit:
        effective_save_dir = (
            DEFAULT_FULL_SAVE_DIR if protocol == "full" else DEFAULT_FEWSHOT_SAVE_DIR
        )
    if effective_save_dir:
        forwarded.extend(["--save_dir", effective_save_dir])
    return forwarded


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if any(flag in argv for flag in ("-h", "--help")):
        print(
            "Ablation wrapper extras:\n"
            "  --dataset: dataset name under ./data/UCRArchive_2018, e.g. AllGestureWiimoteX\n"
            "  --data_dir: accepts either /.../UCRArchive_2018/<dataset>,\n"
            "              /.../UCRArchive_2018 with --dataset, or the base data dir with --dataset.\n"
            f"  --data_path: defaults to {DEFAULT_DATA_PATH}\n"
            f"  --save_dir: defaults to {DEFAULT_FEWSHOT_SAVE_DIR} or {DEFAULT_FULL_SAVE_DIR} based on --protocol\n"
        )

    wrapper_args, remaining_argv = parse_wrapper_args(argv)
    forwarded_argv = build_forwarded_argv(
        wrapper_args=wrapper_args,
        remaining_argv=remaining_argv,
    )

    original_module = load_original_module()
    original_argv = sys.argv
    try:
        sys.argv = [str(ORIGINAL_SCRIPT)] + forwarded_argv
        original_module.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
