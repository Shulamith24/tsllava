from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from .common import apply_paper_style, save_pdf_png
except ImportError:  # pragma: no cover
    from scripts.paper_plots.common import apply_paper_style, save_pdf_png


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opentslm.model.encoder.NewTSVisionEncoder import NewTSPseudoImageTransform


_CACHE_ROOT = Path("/tmp") / "tsllava_visualization_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


BLUE = "#1f4e79"
RED = "#c43c4e"
BOX_EDGE = "#c9d7e8"
GRID_COLOR = "#e8e8e8"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the dual-view motivation figure for TimeMorph")
    parser.add_argument("--data_path", type=str, default="./data", help="Path containing UCRArchive_2018")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "latex_all" / "figures"),
        help="Directory where the figure PDF/PNG will be written",
    )
    parser.add_argument("--output_name", type=str, default="dual_view_motivation")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=96)
    return parser.parse_args(argv)


def load_train_array(dataset: str, data_path: str) -> tuple[np.ndarray, np.ndarray]:
    base = Path(data_path)
    archive = base if base.name == "UCRArchive_2018" else base / "UCRArchive_2018"
    path = archive / dataset / f"{dataset}_TRAIN.tsv"
    if not path.exists():
        raise FileNotFoundError(f"UCR TRAIN file not found: {path}")
    df = pd.read_csv(path, sep="\t", header=None)
    labels = df.iloc[:, 0].to_numpy()
    values = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return labels, values


def z_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def nearest_centroid_example(labels: np.ndarray, values: np.ndarray, target_label: int) -> np.ndarray:
    subset = np.stack([z_normalize(row) for row in values[labels == target_label]], axis=0)
    centroid = subset.mean(axis=0, keepdims=True)
    idx = np.argmin(np.linalg.norm(subset - centroid, axis=1))
    return subset[idx]


def extract_patches(series: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    if len(series) < patch_size:
        pad = np.full(patch_size - len(series), series[-1], dtype=series.dtype)
        series = np.concatenate([series, pad])
    rem = (len(series) - patch_size) % stride
    if rem:
        pad = np.full(stride - rem, series[-1], dtype=series.dtype)
        series = np.concatenate([series, pad])
    starts = range(0, len(series) - patch_size + 1, stride)
    return np.stack([series[start : start + patch_size] for start in starts], axis=0)


def morphology_map(series: np.ndarray, patch_size: int, image_size: int) -> np.ndarray:
    transform = NewTSPseudoImageTransform(
        ts_patch_size=patch_size,
        ts_stride=0.5,
        vision_2d_mode="legacy_unfold",
        image_size=image_size,
    )
    tensor = torch.tensor(series, dtype=torch.float32).view(1, -1, 1)
    image = transform.ts2grayscale_image(tensor)[0, 0].detach().cpu().numpy()
    return image.astype(np.float32)


def plot_raw_axis(
    ax: plt.Axes,
    dataset: str,
    pair: tuple[int, int],
    series_pair: tuple[np.ndarray, np.ndarray],
    ylabel: str,
    legend_loc: str,
    *,
    show_xlabel: bool,
) -> None:
    x = np.arange(len(series_pair[0]))
    ax.plot(x, series_pair[0], color=BLUE, lw=1.45, label=f"class {pair[0]}")
    ax.plot(x, series_pair[1], color=RED, lw=1.45, label=f"class {pair[1]}")
    ax.set_title("Raw class contrast", fontsize=11, pad=5)
    ax.set_ylabel(ylabel, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("Time", fontsize=10)
    ax.text(
        0.02,
        0.92,
        dataset,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc=legend_loc, fontsize=7.2, frameon=False, handlelength=1.7, borderpad=0.2)
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, linestyle="--", alpha=0.8)
    ax.tick_params(labelsize=8, length=2.5)
    ax.set_xlim(0, len(series_pair[0]) - 1)


def add_patch_box(
    ax: plt.Axes,
    patch: np.ndarray,
    left: float,
    bottom: float,
    width: float,
    height: float,
    color: str,
) -> None:
    inset = ax.inset_axes([left, bottom, width, height])
    x = np.linspace(0, 1, len(patch))
    y = z_normalize(patch)
    ymin, ymax = y.min(), y.max()
    denom = max(float(ymax - ymin), 1e-6)
    y = (y - ymin) / denom
    inset.plot(x, y, color=color, lw=1.25)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.08, 1.08)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor(BOX_EDGE)
        spine.set_linewidth(0.8)


def plot_patch_axis(
    ax: plt.Axes,
    pair: tuple[int, int],
    series_pair: tuple[np.ndarray, np.ndarray],
    patch_start: int,
    patch_size: int,
    stride: int,
    *,
    show_xlabel: bool,
) -> None:
    ax.set_title("Ordered waveform patches", fontsize=11, pad=5)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    patches_a = extract_patches(series_pair[0], patch_size, stride)
    patches_b = extract_patches(series_pair[1], patch_size, stride)
    selected = list(range(patch_start, patch_start + 4))

    box_w = 0.15
    box_h = 0.24
    x0 = 0.23
    dx = 0.18
    y_top = 0.60
    y_bot = 0.15

    ax.text(0.08, y_top + box_h * 0.5, f"class {pair[0]}", ha="right", va="center", fontsize=8.5)
    ax.text(0.08, y_bot + box_h * 0.5, f"class {pair[1]}", ha="right", va="center", fontsize=8.5)

    for col, patch_idx in enumerate(selected):
        left = x0 + col * dx
        if patch_idx < len(patches_a):
            add_patch_box(ax, patches_a[patch_idx], left, y_top, box_w, box_h, BLUE)
        if patch_idx < len(patches_b):
            add_patch_box(ax, patches_b[patch_idx], left, y_bot, box_w, box_h, RED)
        ax.text(left + box_w / 2, y_top + box_h + 0.055, f"p{patch_idx}", ha="center", va="bottom", fontsize=7.8)
        ax.text(left + box_w / 2, y_bot + box_h + 0.035, f"p{patch_idx}", ha="center", va="bottom", fontsize=7.8)
        if col < len(selected) - 1:
            for y in (y_top, y_bot):
                ax.annotate(
                    "",
                    xy=(left + box_w + 0.035, y + box_h / 2),
                    xytext=(left + box_w + 0.005, y + box_h / 2),
                    arrowprops=dict(arrowstyle="->", color="#4b5563", lw=0.7),
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                )

    if show_xlabel:
        ax.text(x0, 0.03, "first", ha="left", va="center", fontsize=8.5)
        ax.text(x0 + 3 * dx + box_w, 0.03, "last", ha="right", va="center", fontsize=8.5)
        ax.text(0.5, -0.055, "Within-patch time", ha="center", va="center", fontsize=10, transform=ax.transAxes)
    else:
        ax.text(x0, 0.04, "first", ha="left", va="center", fontsize=8.5)
        ax.text(x0 + 3 * dx + box_w, 0.04, "last", ha="right", va="center", fontsize=8.5)


def plot_morph_axis(
    ax: plt.Axes,
    pair: tuple[int, int],
    series_pair: tuple[np.ndarray, np.ndarray],
    patch_size: int,
    image_size: int,
    *,
    show_xlabel: bool,
) -> None:
    ax.set_title("Window-shape morphology", fontsize=11, pad=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    for row, series in enumerate(series_pair):
        image = morphology_map(series, patch_size, image_size)
        y0 = 1 - row
        ax.imshow(image, cmap="gray", aspect="auto", interpolation="nearest", extent=(0, 1, y0, y0 + 0.92))
    ax.axhline(1.0, color="white", lw=3.0)
    ax.set_yticks([1.46, 0.46])
    ax.set_yticklabels([f"class {pair[0]}", f"class {pair[1]}"], fontsize=8.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["start", "end"], fontsize=8.5)
    if show_xlabel:
        ax.set_xlabel("2D waveform map", fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#d0d0d0")


def build_cases(data_path: str) -> list[dict[str, Any]]:
    cases = [
        {"dataset": "CBF", "pair": (2, 3), "ylabel": "A. Temporal cue", "patch_start": 7, "legend_loc": "upper right"},
        {"dataset": "ECG200", "pair": (-1, 1), "ylabel": "B. Morphology cue", "patch_start": 2, "legend_loc": "lower right"},
    ]
    for case in cases:
        labels, values = load_train_array(case["dataset"], data_path)
        pair = case["pair"]
        case["series_pair"] = (
            nearest_centroid_example(labels, values, pair[0]),
            nearest_centroid_example(labels, values, pair[1]),
        )
    return cases


def plot_dual_view_motivation(args: argparse.Namespace) -> dict[str, Any]:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.grid": False,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    cases = build_cases(args.data_path)

    fig = plt.figure(figsize=(8.9, 5.25))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.18, 1.05, 1.0],
        height_ratios=[1, 1],
        left=0.075,
        right=0.985,
        top=0.92,
        bottom=0.16,
        wspace=0.34,
        hspace=0.42,
    )

    for row, case in enumerate(cases):
        show_xlabel = row == 1
        ax_raw = fig.add_subplot(gs[row, 0])
        ax_patch = fig.add_subplot(gs[row, 1])
        ax_morph = fig.add_subplot(gs[row, 2])
        plot_raw_axis(
            ax_raw,
            case["dataset"],
            case["pair"],
            case["series_pair"],
            case["ylabel"],
            case["legend_loc"],
            show_xlabel=show_xlabel,
        )
        plot_patch_axis(
            ax_patch,
            case["pair"],
            case["series_pair"],
            case["patch_start"],
            args.patch_size,
            args.stride,
            show_xlabel=show_xlabel,
        )
        plot_morph_axis(
            ax_morph,
            case["pair"],
            case["series_pair"],
            args.patch_size,
            args.image_size,
            show_xlabel=show_xlabel,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_pdf_png(fig, output_dir, args.output_name)
    plt.close(fig)

    return {
        "artifacts": {kind: str(path.resolve()) for kind, path in artifacts.items()},
        "cases": [{"dataset": c["dataset"], "pair": c["pair"], "legend_loc": c["legend_loc"]} for c in cases],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata = plot_dual_view_motivation(args)
    print(metadata["artifacts"]["pdf"])
    print(metadata["artifacts"]["png"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
