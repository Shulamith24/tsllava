from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import PAPER_COLORS, PAPER_OUTPUT_DIR, apply_paper_style, save_pdf_png


MODEL_ORDER = [
    ("m2_pretrained", "TimeMorph", PAPER_COLORS["primary"], "o", True),
    ("cosco_resnet", "COSCO-ResNet", PAPER_COLORS["cosco"], "D", False),
    ("resnet", "ResNet", PAPER_COLORS["resnet"], "s", False),
    ("tapnet", "TapNet", PAPER_COLORS["tapnet"], "^", False),
    ("patchtst", "PatchTST", PAPER_COLORS["patchtst"], "X", False),
    ("tslib_informer", "Informer", PAPER_COLORS["informer"], "h", False),
    ("onefitsall", "GPT4TS", PAPER_COLORS["gpt4ts"], "x", False),
]


def load_fewshot_curve_data(summary_csv: str | Path, merged_results_csv: str | Path) -> pd.DataFrame:
    summary = pd.read_csv(summary_csv)
    merged = pd.read_csv(merged_results_csv)
    if summary.empty:
        raise ValueError(f"empty summary table: {summary_csv}")
    if merged.empty:
        raise ValueError(f"empty merged results table: {merged_results_csv}")

    rows: list[dict[str, object]] = []
    if "shot" in summary.columns:
        shot_values = summary["shot"].astype(str).tolist()
    else:
        shot_values = [
            column[len("shot_") : -len("_accuracy_pct")]
            for column in summary.columns
            if column.startswith("shot_") and column.endswith("_accuracy_pct")
        ]
    shot_order = [str(shot) for shot in sorted({int(shot) for shot in shot_values if str(shot).isdigit()})]
    order_lookup = {shot: idx for idx, shot in enumerate(shot_order)}
    for model_key, model_label, color, marker, primary in MODEL_ORDER:
        model_summary = summary[summary["model_key"] == model_key] if "model_key" in summary.columns else pd.DataFrame()
        if model_summary.empty and "model_key" not in summary.columns:
            model_row = summary[summary["paper_label"].astype(str) == model_label] if "paper_label" in summary.columns else pd.DataFrame()
            if model_row.empty:
                continue
            model_summary = model_row
        if model_summary.empty:
            continue
        model_merged = merged[merged["model_key"] == model_key].copy()
        for shot in shot_order:
            if "shot" in model_summary.columns:
                shot_summary = model_summary[model_summary["shot"].astype(str) == shot]
                if shot_summary.empty:
                    continue
                row = shot_summary.iloc[0]
                accuracy_mean_pct = float(row["accuracy_mean_pct"])
            else:
                col = f"shot_{shot}_accuracy_pct"
                if col not in model_summary.columns:
                    continue
                row = model_summary.iloc[0]
                accuracy_mean_pct = float(row[col])
            shot_merged = model_merged[model_merged["shot"].astype(str) == shot]
            rows.append(
                {
                    "model_key": model_key,
                    "model_label": model_label,
                    "shot": shot,
                    "shot_index": order_lookup[shot],
                    "accuracy_mean_pct": accuracy_mean_pct,
                    "uncertainty_pct": float(shot_merged["accuracy_std"].astype(float).mean() * 100.0)
                    if not shot_merged.empty and "accuracy_std" in shot_merged.columns
                    else float("nan"),
                    "color": color,
                    "marker": marker,
                    "primary": primary,
                }
            )

    curve = pd.DataFrame(rows)
    if curve.empty:
        raise ValueError("no few-shot curve rows could be assembled")
    return curve.sort_values(["shot_index", "primary", "model_key"], ascending=[True, False, True]).reset_index(drop=True)


def plot_fewshot_scaling(
    *,
    summary_csv: str | Path,
    merged_results_csv: str | Path,
    output_dir: str | Path = PAPER_OUTPUT_DIR,
    output_name: str = "fewshot_trend",
) -> dict[str, str]:
    apply_paper_style()
    curve = load_fewshot_curve_data(summary_csv, merged_results_csv)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for model_key, model_df in curve.groupby("model_key", sort=False):
        row0 = model_df.iloc[0]
        linewidth = 2.8 if bool(row0["primary"]) else 1.8
        markersize = 7 if bool(row0["primary"]) else 5.5
        alpha = 1.0 if bool(row0["primary"]) else 0.92
        zorder = 5 if bool(row0["primary"]) else 3
        label = row0["model_label"] + (" (ours)" if bool(row0["primary"]) else "")
        x = model_df["shot_index"].to_numpy(dtype=float)
        y = model_df["accuracy_mean_pct"].to_numpy(dtype=float)
        band = model_df["uncertainty_pct"].fillna(0.0).to_numpy(dtype=float)
        ax.fill_between(x, y - band, y + band, color=row0["color"], alpha=0.09 if bool(row0["primary"]) else 0.04, linewidth=0)
        ax.errorbar(
            x,
            y,
            yerr=band,
            label=label,
            color=row0["color"],
            marker=row0["marker"],
            linewidth=linewidth,
            markersize=markersize,
            capsize=2.4,
            elinewidth=0.95 if bool(row0["primary"]) else 0.8,
            alpha=alpha,
            zorder=zorder,
        )

    shot_order = [str(shot) for shot in sorted(curve["shot"].unique().tolist(), key=int)]
    ax.set_xticks(range(len(shot_order)), [f"{shot}-shot" for shot in shot_order])
    ax.set_xlabel("Shots per class")
    ax.set_ylabel("Macro accuracy (%)")
    ax.set_ylim(30, 75.5)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, columnspacing=1.0, handlelength=2.0)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    artifacts = save_pdf_png(fig, Path(output_dir), output_name)
    plt.close(fig)
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}
