from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

try:
    from .common import PAPER_OUTPUT_DIR, apply_paper_style, save_pdf_png
except ImportError:  # pragma: no cover
    from scripts.paper_plots.common import PAPER_OUTPUT_DIR, apply_paper_style, save_pdf_png


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opentslm.model.encoder.NewTSVisionEncoder import NewTSPseudoImageTransform
from opentslm.time_series_datasets.ucr.ucr_loader import UCRDataset, load_ucr_dataset


_CACHE_ROOT = Path("/tmp") / "tsllava_visualization_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a dual-view motivation figure for TimeMorph")
    parser.add_argument("--dataset", type=str, default="TwoLeadECG", help="UCR dataset used for the illustrative example")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="UCR split")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index inside the split")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the UCR archive root")
    parser.add_argument("--patch_size", type=int, default=16, help="Temporal patch length")
    parser.add_argument("--stride", type=int, default=8, help="Temporal patch stride")
    parser.add_argument(
        "--vision_2d_mode",
        type=str,
        default="legacy_unfold",
        choices=["legacy_unfold", "tivit_sqrt_overlap"],
        help="Pseudo-image construction mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PAPER_OUTPUT_DIR),
        help="Directory where the figure PDF/PNG will be written",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="dual_view_motivation",
        help="Base file name without extension",
    )
    parser.add_argument(
        "--focus_patch_index",
        type=int,
        default=-1,
        help="Patch index to highlight; -1 selects a centered patch automatically",
    )
    parser.add_argument(
        "--num_patch_rows",
        type=int,
        default=4,
        help="Number of consecutive temporal patches to show in the middle panel",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Target image size for the morphology panel",
    )
    return parser.parse_args(argv)


def load_sample(dataset_name: str, split: str, data_path: str, sample_index: int) -> tuple[torch.Tensor, int, int]:
    train_df, test_df = load_ucr_dataset(dataset_name, raw_data_path=data_path)
    split_df = train_df if split == "train" else test_df
    dataset = UCRDataset(split_df)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index={sample_index} is out of range for {dataset_name}/{split} with {len(dataset)} samples")
    series, label = dataset[sample_index]
    return series, int(label), len(dataset)


def prepare_patches(series: torch.Tensor, patch_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    transform = NewTSPseudoImageTransform(ts_patch_size=patch_size, ts_stride=0.5, vision_2d_mode="legacy_unfold", image_size=224)
    normalized = transform._normalize_time_series(series.view(1, -1, 1)).squeeze(0).squeeze(-1)
    if stride <= 0:
        raise ValueError("stride must be positive")

    pad_right = (stride - (normalized.numel() - patch_size) % stride) % stride if normalized.numel() >= patch_size else patch_size - normalized.numel()
    if stride == patch_size:
        pad_right = (patch_size - normalized.numel() % patch_size) % patch_size
    padded = torch.cat([normalized, normalized[-1:].expand(pad_right)]) if pad_right > 0 else normalized
    if stride == patch_size:
        patches = padded.unfold(0, patch_size, patch_size)
    else:
        patches = padded.unfold(0, patch_size, stride)
    patch_array = patches.detach().cpu().numpy()
    raw_series = normalized.detach().cpu().numpy()
    return raw_series, patch_array


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    row_min = x.min(axis=1, keepdims=True)
    row_max = x.max(axis=1, keepdims=True)
    return (x - row_min) / np.maximum(row_max - row_min, 1e-6)


def plot_dual_view_motivation(args: argparse.Namespace) -> dict[str, Any]:
    apply_paper_style()

    series, label, split_size = load_sample(args.dataset, args.split, args.data_path, args.sample_index)
    raw_series, patches = prepare_patches(series, args.patch_size, args.stride)
    transform = NewTSPseudoImageTransform(
        ts_patch_size=args.patch_size,
        ts_stride=0.5,
        vision_2d_mode=args.vision_2d_mode,
        image_size=args.image_size,
    )
    morphology = transform.ts2grayscale_image(series.view(1, -1, 1))[0, 0].detach().cpu().numpy()

    patch_count = patches.shape[0]
    if patch_count == 0:
        raise ValueError("No patches could be extracted from the selected series")

    focus_patch = args.focus_patch_index
    if focus_patch < 0:
        focus_patch = max(0, min(patch_count // 2, patch_count - args.num_patch_rows))
    focus_patch = max(0, min(focus_patch, max(0, patch_count - args.num_patch_rows)))
    end_patch = min(patch_count, focus_patch + args.num_patch_rows)
    shown_patches = normalize_rows(patches[focus_patch:end_patch])

    start_t = focus_patch * args.stride
    end_t = min(len(raw_series), start_t + args.patch_size + (args.num_patch_rows - 1) * args.stride)

    fig = plt.figure(figsize=(13.8, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.05], wspace=0.22)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_patch = fig.add_subplot(gs[0, 1])
    ax_morph = fig.add_subplot(gs[0, 2])

    x = np.arange(raw_series.shape[0])
    ax_raw.plot(x, raw_series, color="#1f1f1f", lw=1.2)
    ax_raw.axvspan(start_t, end_t, color="#e58a2f", alpha=0.16, lw=0)
    for offset in range(args.num_patch_rows + 1):
        boundary = start_t + offset * args.stride
        if boundary <= raw_series.shape[0]:
            ax_raw.axvline(boundary, color="#e58a2f", lw=0.8, alpha=0.55, linestyle="--")
    ax_raw.set_title("Raw waveform")
    ax_raw.set_xlabel("Time")
    ax_raw.set_ylabel("Normalized value")
    ax_raw.text(
        0.02,
        0.96,
        f"Dataset: {args.dataset}  |  label: {label}",
        transform=ax_raw.transAxes,
        va="top",
        ha="left",
        fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    )
    ax_raw.text(
        0.02,
        0.08,
        "Temporal branch preserves ordered dynamics",
        transform=ax_raw.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        color="#8a4d15",
    )
    ax_raw.set_xlim(0, raw_series.shape[0] - 1)

    ax_patch.imshow(
        shown_patches,
        aspect="auto",
        cmap="Blues",
        interpolation="nearest",
        origin="lower",
    )
    ax_patch.set_title("Temporal patches")
    ax_patch.set_xlabel("Within-patch time")
    ax_patch.set_ylabel("Patch index")
    ax_patch.set_yticks(np.arange(shown_patches.shape[0]))
    ax_patch.set_yticklabels([f"p{focus_patch + i}" for i in range(shown_patches.shape[0])])
    ax_patch.set_xticks([0, args.patch_size // 2, args.patch_size - 1])
    ax_patch.set_xticklabels(["0", f"{args.patch_size // 2}", f"{args.patch_size - 1}"])
    ax_patch.text(
        0.02,
        0.97,
        "Local trend / phase shift / amplitude change",
        transform=ax_patch.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#103a5f",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
    )
    focus_row = min(shown_patches.shape[0] - 1, max(0, args.num_patch_rows // 2))
    ax_patch.add_patch(
        Rectangle(
            (-0.5, focus_row - 0.5),
            args.patch_size - 0.02,
            1.0,
            fill=False,
            edgecolor="#c23b22",
            linewidth=1.6,
        )
    )

    ax_morph.imshow(
        morphology,
        aspect="auto",
        cmap="gray_r",
        interpolation="nearest",
        origin="upper",
    )
    ax_morph.set_title("Morphology map")
    ax_morph.set_xlabel("Within-window position")
    ax_morph.set_ylabel("Window index")
    ax_morph.text(
        0.02,
        0.97,
        "Peaks / valleys / motifs / cross-window patterns",
        transform=ax_morph.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#2f2f2f",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.72),
    )
    for ax in (ax_raw, ax_patch, ax_morph):
        ax.tick_params(axis="both", which="both", length=2.5, width=0.7)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.98, wspace=0.26)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_pdf_png(fig, output_dir, args.output_name)
    plt.close(fig)

    return {
        "dataset": args.dataset,
        "split": args.split,
        "sample_index": args.sample_index,
        "label": label,
        "split_size": split_size,
        "patch_count": patch_count,
        "focus_patch": focus_patch,
        "artifacts": {kind: str(path.resolve()) for kind, path in artifacts.items()},
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = plot_dual_view_motivation(args)
    print(metadata["artifacts"]["pdf"])
    print(metadata["artifacts"]["png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
