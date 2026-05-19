from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from .common import PAPER_OUTPUT_DIR, apply_paper_style, save_pdf_png


DEFAULT_WTL_PATH = Path(__file__).resolve().parents[2] / "results" / "ucr_batches" / "reports" / "ucr_fewshot_paper_current" / "paper_table_wtl.csv"


def load_wtl_matrix(wtl_csv: str | Path = DEFAULT_WTL_PATH) -> pd.DataFrame:
    df = pd.read_csv(wtl_csv)
    if df.empty:
        raise ValueError(f"empty W/T/L table: {wtl_csv}")
    rows: list[dict[str, object]] = []
    shot_specs = [
        ("1-shot", "shot_1_wins", "shot_1_ties", "shot_1_losses", "shot_1_comparisons"),
        ("2-shot", "shot_2_wins", "shot_2_ties", "shot_2_losses", "shot_2_comparisons"),
        ("5-shot", "shot_5_wins", "shot_5_ties", "shot_5_losses", "shot_5_comparisons"),
        ("10-shot", "shot_10_wins", "shot_10_ties", "shot_10_losses", "shot_10_comparisons"),
    ]
    for row in df.itertuples(index=False):
        baseline = str(row.baseline_label)
        for shot, wins_field, ties_field, losses_field, comps_field in shot_specs:
            wins = int(getattr(row, wins_field))
            ties = int(getattr(row, ties_field))
            losses = int(getattr(row, losses_field))
            comps = int(getattr(row, comps_field))
            margin = (wins - losses) / comps if comps else 0.0
            rows.append(
                {
                    "baseline": baseline,
                    "shot": shot,
                    "margin": margin,
                    "wtl": f"{wins}/{ties}/{losses}",
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "comparisons": comps,
                }
            )
    return pd.DataFrame(rows)


def plot_wtl_heatmap(
    *,
    wtl_csv: str | Path = DEFAULT_WTL_PATH,
    output_dir: str | Path = PAPER_OUTPUT_DIR,
    output_name: str = "ucr_wtl_heatmap",
) -> dict[str, str]:
    apply_paper_style()
    long_df = load_wtl_matrix(wtl_csv)
    if long_df.empty:
        raise ValueError("no W/T/L rows could be assembled")

    baseline_order = ["COSCO-ResNet", "ResNet", "TapNet", "PatchTST", "TimesNet", "GPT4TS"]
    shot_order = ["1-shot", "2-shot", "5-shot", "10-shot"]
    pivot_margin = (
        long_df.pivot(index="baseline", columns="shot", values="margin")
        .reindex(index=baseline_order, columns=shot_order)
    )
    pivot_text = (
        long_df.pivot(index="baseline", columns="shot", values="wtl")
        .reindex(index=baseline_order, columns=shot_order)
    )

    fig, ax = plt.subplots(figsize=(7.7, 3.65))
    values = pivot_margin.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(values)))
    norm = TwoSlopeNorm(vmin=-max(0.15, vmax), vcenter=0.0, vmax=max(0.15, vmax))
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(shot_order)), shot_order)
    ax.set_yticks(np.arange(len(baseline_order)), baseline_order)
    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Baseline")
    ax.set_title("TimeMorph win margin against representative baselines")

    for i, baseline in enumerate(baseline_order):
        for j, shot in enumerate(shot_order):
            text = pivot_text.loc[baseline, shot]
            if pd.isna(text):
                continue
            value = values[i, j]
            ax.text(j, i, text, ha="center", va="center", fontsize=8.0, color="#1f1f1f" if abs(value) < vmax * 0.45 else "white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cbar.set_label("Win margin (W - L) / N")
    ax.tick_params(axis="both", which="both", length=0)
    fig.tight_layout()

    artifacts = save_pdf_png(fig, Path(output_dir), output_name)
    plt.close(fig)
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}
