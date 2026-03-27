from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path("/tmp") / "tsllava_reporting_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from .common import DEFAULT_COLORS, DEFAULT_MARKERS


def plot_fewshot_trend(
    *,
    summary_csv: Path,
    output_dir: Path,
    model_order: list[dict[str, str]],
    report_name: str,
) -> list[Path]:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"summary_by_shot.csv is empty: {summary_csv}")

    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 0.8,
        }
    )

    ordered_shots = df["shot"].drop_duplicates().tolist()
    shot_map = {shot: idx for idx, shot in enumerate(ordered_shots)}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for idx, model in enumerate(model_order):
        model_df = df[df["model_key"] == model["key"]].copy()
        model_df["shot_order"] = model_df["shot"].map(shot_map)
        model_df = model_df.sort_values("shot_order")
        color = model.get("color") or DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
        marker = model.get("marker") or DEFAULT_MARKERS[idx % len(DEFAULT_MARKERS)]
        linewidth = 2.8 if model.get("primary") else 1.8
        markersize = 7 if model.get("primary") else 5.5
        alpha = 1.0 if model.get("primary") else 0.9
        zorder = 4 if model.get("primary") else 3

        ax.plot(
            model_df["shot_order"],
            model_df["accuracy_mean_pct"],
            label=model["label"],
            color=color,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
            zorder=zorder,
        )

    ax.set_xticks(list(shot_map.values()), [str(shot) for shot in ordered_shots])
    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_title(f"{report_name} few-shot trend")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=min(4, len(model_order)), frameon=False)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    png_path = output_dir / "fewshot_trend.png"
    pdf_path = output_dir / "fewshot_trend.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]
