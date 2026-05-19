from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import (
    LATEX_ROOT,
    PAPER_COLORS,
    PAPER_OUTPUT_DIR,
    apply_paper_style,
    extract_latex_numbers,
    normalize_latex_label,
    read_latex_rows,
    save_pdf_png,
)


UCR_ABLATION_TABLE = LATEX_ROOT / "tables" / "main" / "ucr_fewshot_ablation_summary.tex"
LLM_ABLATION_TABLE = LATEX_ROOT / "tables" / "main" / "m2_llm_backbone_ablation_summary.tex"
SEMANTIC_PRIOR_TABLE = LATEX_ROOT / "tables" / "main" / "semantic_prior_summary.tex"


def _load_ablation_avg_map(path: Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for cells in read_latex_rows(path):
        label = normalize_latex_label(cells[0])
        numbers = extract_latex_numbers(" ".join(cells[1:]))
        if not numbers:
            continue
        if "LLM Backbone" in label or "Backbone" in label:
            mapping[label] = numbers[4] if len(numbers) >= 5 else numbers[-1]
        else:
            mapping[label] = numbers[-1]
    if not mapping:
        raise ValueError(f"no table rows parsed from {path}")
    return mapping


def _load_semantic_prior_gains(path: Path) -> dict[str, float]:
    rows = read_latex_rows(path)
    gains: dict[str, float] = {}
    for cells in rows:
        label = normalize_latex_label(cells[0])
        if "Delta" not in label and "d" not in label.lower():
            continue
        values = extract_latex_numbers(" ".join(cells[1:]))
        if len(values) >= 3:
            gains["MIT-BIH"] = values[0]
            gains["CinC2017"] = values[1]
            gains["Avg"] = values[2]
            break
    if not gains:
        raise ValueError(f"semantic-prior delta row was not found in {path}")
    return gains


def load_ablation_drop_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ucr = _load_ablation_avg_map(UCR_ABLATION_TABLE)
    llm = _load_ablation_avg_map(LLM_ABLATION_TABLE)
    semantic = _load_semantic_prior_gains(SEMANTIC_PRIOR_TABLE)

    panel_a_rows = [
        {"component": "No staged pretraining", "drop_pp": ucr["TimeMorph"] - ucr["w/o Curriculum Pretraining"], "color": PAPER_COLORS["negative"]},
        {"component": "Temporal-only / removing morphology", "drop_pp": ucr["TimeMorph"] - ucr["Temporal only"], "color": PAPER_COLORS["negative"]},
        {"component": "Morphology-only / removing temporal", "drop_pp": ucr["TimeMorph"] - ucr["Visual only"], "color": PAPER_COLORS["negative"]},
        {"component": "No class-token scoring", "drop_pp": ucr["TimeMorph"] - ucr["w/o Class-Token Label-Space Scoring"], "color": PAPER_COLORS["negative"]},
        {"component": "No LLM backbone", "drop_pp": ucr["TimeMorph"] - llm["w/o LLM Backbone"], "color": PAPER_COLORS["negative"]},
    ]
    panel_a = pd.DataFrame(panel_a_rows).sort_values("drop_pp", ascending=False).reset_index(drop=True)
    panel_b = pd.DataFrame(
        [
            {"dataset": "MIT-BIH", "gain_pp": semantic["MIT-BIH"], "color": PAPER_COLORS["panel_b"]},
            {"dataset": "CinC2017", "gain_pp": semantic["CinC2017"], "color": PAPER_COLORS["panel_b"]},
            {"dataset": "Avg", "gain_pp": semantic["Avg"], "color": PAPER_COLORS["panel_b_light"]},
        ]
    )
    return panel_a, panel_b


def plot_ablation_drops(
    *,
    output_dir: str | Path = PAPER_OUTPUT_DIR,
    output_name: str = "ablation_drops",
) -> dict[str, str]:
    apply_paper_style()
    panel_a, panel_b = load_ablation_drop_data()

    fig = plt.figure(figsize=(9.25, 5.7))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 0.9], hspace=0.35)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    y = range(len(panel_a))
    ax_a.axvline(0, color="#444444", linewidth=0.8)
    ax_a.barh(y, panel_a["drop_pp"], color=panel_a["color"], height=0.62, alpha=0.96)
    ax_a.set_yticks(list(y), panel_a["component"].tolist())
    ax_a.set_xlabel("Avg accuracy drop (pp)")
    ax_a.set_xlim(0, max(panel_a["drop_pp"]) + 0.9)
    ax_a.set_title("Panel A. UCR ablation drops")
    ax_a.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.4)
    ax_a.set_axisbelow(True)
    for idx, value in enumerate(panel_a["drop_pp"].tolist()):
        ax_a.text(value + 0.12, idx, f"{value:.2f}", va="center", ha="left", fontsize=8.5, color="#5a1d1d")

    y2 = range(len(panel_b))
    ax_b.axvline(0, color="#444444", linewidth=0.8)
    ax_b.barh(y2, panel_b["gain_pp"], color=panel_b["color"], height=0.6, alpha=0.95)
    ax_b.set_yticks(list(y2), panel_b["dataset"].tolist())
    ax_b.set_xlabel("Avg accuracy gain (pp)")
    ax_b.set_xlim(0, max(panel_b["gain_pp"]) + 0.9)
    ax_b.set_title("Panel B. Semantic-prior gain on external datasets")
    ax_b.grid(axis="x", linestyle="--", linewidth=0.55, alpha=0.4)
    ax_b.set_axisbelow(True)
    for idx, value in enumerate(panel_b["gain_pp"].tolist()):
        ax_b.text(value + 0.12, idx, f"+{value:.2f}", va="center", ha="left", fontsize=8.5, color="#12384b")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    artifacts = save_pdf_png(fig, Path(output_dir), output_name)
    plt.close(fig)
    return {kind: str(path.resolve()) for kind, path in artifacts.items()}
