#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Export grayscale pseudo-images for the NewTS vision branch from a UCR sample.

The script saves both:
- the raw normalized 2D grid before square padding / resizing
- the final grayscale image that is replicated to RGB before entering the ViT
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opentslm.model.encoder.NewTSVisionEncoder import NewTSPseudoImageTransform
from opentslm.time_series_datasets.ucr.ucr_loader import UCRDataset, load_ucr_dataset


def cli_flag_was_provided(argv: list[str], flag_name: str) -> bool:
    return any(token == flag_name or token.startswith(f"{flag_name}=") for token in argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    provided_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Export grayscale pseudo-images for the NewTS vision branch")
    parser.add_argument("--dataset", type=str, default="CricketZ", help="UCR dataset name")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--sample_index", type=int, default=0, help="Zero-based sample index inside the split")
    parser.add_argument("--data_path", type=str, default="./data", help="UCR data root")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/visualizations/newts_pseudo_images",
        help="Base directory for generated outputs",
    )
    parser.add_argument(
        "--local_checkpoint",
        type=str,
        default=None,
        help="Optional local checkpoint used to hydrate NewTS image-branch config",
    )
    parser.add_argument("--vit_patch_size", type=int, default=16, help="Patch size used by the 1D->2D transform")
    parser.add_argument("--vit_stride", type=float, default=0.5, help="Stride ratio used by unfold-based modes")
    parser.add_argument(
        "--vision_2d_mode",
        type=str,
        default="reshape_serpentine",
        choices=["reshape_serpentine", "adaptive_unfold", "legacy_unfold"],
        help="1D->2D pseudo-image construction mode",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target square size for the grayscale image consumed by the ViT",
    )
    parser.add_argument(
        "--min_render_size",
        type=int,
        default=256,
        help="Nearest-neighbor upscale target for saving very small grids as readable PNGs",
    )

    args = parser.parse_args(argv)
    args.vit_patch_size_explicit = cli_flag_was_provided(provided_argv, "--vit_patch_size")
    args.vit_stride_explicit = cli_flag_was_provided(provided_argv, "--vit_stride")
    args.vision_2d_mode_explicit = cli_flag_was_provided(provided_argv, "--vision_2d_mode")
    return args


def load_newts_config_from_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config") or {}
    encoder_type = model_config.get("encoder_type")
    if encoder_type != "newts_dual_branch":
        raise ValueError(
            f"Checkpoint encoder_type must be 'newts_dual_branch' for pseudo-image export, got {encoder_type!r}"
        )

    encoder_config = dict(model_config.get("encoder_config") or {})
    return {
        "encoder_type": encoder_type,
        "llm_id": model_config.get("llm_id"),
        "encoder_config": encoder_config,
    }


def resolve_visual_config(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_metadata = None
    if args.local_checkpoint:
        checkpoint_metadata = load_newts_config_from_checkpoint(args.local_checkpoint)
        encoder_config = checkpoint_metadata["encoder_config"]

        if not args.vit_patch_size_explicit and "vit_patch_size" in encoder_config:
            args.vit_patch_size = int(encoder_config["vit_patch_size"])
        if not args.vit_stride_explicit and "vit_stride" in encoder_config:
            args.vit_stride = float(encoder_config["vit_stride"])
        if not args.vision_2d_mode_explicit and "vision_2d_mode" in encoder_config:
            args.vision_2d_mode = str(encoder_config["vision_2d_mode"])

    return {
        "vit_patch_size": int(args.vit_patch_size),
        "vit_stride": float(args.vit_stride),
        "vision_2d_mode": args.vision_2d_mode,
        "image_size": int(args.image_size),
        "checkpoint_metadata": checkpoint_metadata,
    }


def load_ucr_sample(
    *,
    dataset_name: str,
    split: str,
    data_path: str,
    sample_index: int,
) -> tuple[torch.Tensor, int, int]:
    train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=data_path)
    split_df = train_df if split == "train" else test_df
    dataset = UCRDataset(split_df)

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is out of range for {dataset_name}/{split} with {len(dataset)} samples"
        )

    time_series, label = dataset[sample_index]
    return time_series, int(label), len(dataset)


def _safe_label_token(label: int) -> str:
    return str(label).replace("-", "neg_")


def _save_grayscale_png(
    image_array: np.ndarray,
    output_path: Path,
    *,
    min_render_size: int,
) -> tuple[int, int]:
    array_uint8 = np.clip(np.round(image_array * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(array_uint8, mode="L")

    width, height = image.size
    min_side = min(width, height)
    scale = 1 if min_side >= min_render_size else int(math.ceil(min_render_size / max(1, min_side)))

    if scale > 1:
        resampling = getattr(Image, "Resampling", Image).NEAREST
        image = image.resize((width * scale, height * scale), resample=resampling)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return image.size


def export_newts_pseudo_image(args: argparse.Namespace) -> dict[str, Any]:
    resolved = resolve_visual_config(args)
    time_series, label, split_size = load_ucr_sample(
        dataset_name=args.dataset,
        split=args.split,
        data_path=args.data_path,
        sample_index=args.sample_index,
    )

    transform = NewTSPseudoImageTransform(
        ts_patch_size=resolved["vit_patch_size"],
        ts_stride=resolved["vit_stride"],
        vision_2d_mode=resolved["vision_2d_mode"],
        image_size=resolved["image_size"],
    )

    ts_batch = time_series.view(1, -1, 1)
    raw_grid = transform.ts2grid(ts_batch)[0, 0].detach().cpu().numpy()
    resized_grid = transform.ts2grayscale_image(ts_batch)[0, 0].detach().cpu().numpy()

    sample_dir = (
        Path(args.output_dir)
        / args.dataset
        / args.split
        / f"sample_{args.sample_index:04d}_label_{_safe_label_token(label)}"
    )
    raw_grid_path = sample_dir / "pseudo_image_grid.png"
    resized_grid_path = sample_dir / "pseudo_image_resized.png"
    metadata_path = sample_dir / "metadata.json"

    raw_render_size = _save_grayscale_png(raw_grid, raw_grid_path, min_render_size=args.min_render_size)
    resized_render_size = _save_grayscale_png(
        resized_grid,
        resized_grid_path,
        min_render_size=args.min_render_size,
    )

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "split_size": split_size,
        "sample_index": args.sample_index,
        "label": label,
        "time_series_length": int(time_series.numel()),
        "transform_config": {
            "vit_patch_size": resolved["vit_patch_size"],
            "vit_stride": resolved["vit_stride"],
            "vision_2d_mode": resolved["vision_2d_mode"],
            "image_size": resolved["image_size"],
        },
        "raw_grid_shape": list(raw_grid.shape),
        "resized_grid_shape": list(resized_grid.shape),
        "saved_render_size": {
            "raw_grid": list(raw_render_size),
            "resized_grid": list(resized_render_size),
        },
        "checkpoint_metadata": resolved["checkpoint_metadata"],
        "artifacts": {
            "pseudo_image_grid_png": str(raw_grid_path.resolve()),
            "pseudo_image_resized_png": str(resized_grid_path.resolve()),
            "metadata_json": str(metadata_path.resolve()),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    metadata = export_newts_pseudo_image(args)
    print("Saved NewTS pseudo-image artifacts:")
    for key, value in metadata["artifacts"].items():
        print(f"  {key}: {value}")
    return metadata


if __name__ == "__main__":
    main()
