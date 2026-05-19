#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "results" / "ucr_batches" / "reports"
NIPS_ROOT = REPO_ROOT / "latex_all" / "Formatting_Instructions_For_NeurIPS_2026"
PAPER_PLOTS_ROOT = NIPS_ROOT / "paper_plots" / "outputs"

ASSET_MAP = {
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "main_table.tex":
        NIPS_ROOT / "tables" / "main" / "ucr_fewshot_main.tex",
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "paper_table_wtl.tex":
        NIPS_ROOT / "tables" / "main" / "ucr_fewshot_wtl.tex",
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "paper_appendix_shot_1.tex":
        NIPS_ROOT / "tables" / "appendix" / "ucr_fewshot_shot_1.tex",
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "paper_appendix_shot_2.tex":
        NIPS_ROOT / "tables" / "appendix" / "ucr_fewshot_shot_2.tex",
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "paper_appendix_shot_5.tex":
        NIPS_ROOT / "tables" / "appendix" / "ucr_fewshot_shot_5.tex",
    REPORTS_ROOT / "ucr_fewshot_paper_current" / "paper_appendix_shot_10.tex":
        NIPS_ROOT / "tables" / "appendix" / "ucr_fewshot_shot_10.tex",
    REPORTS_ROOT / "m2_pretrain_ablation_partial_preview" / "main_table.tex":
        NIPS_ROOT / "tables" / "ablations" / "m2_pretrain_preview.tex",
    REPORTS_ROOT / "m2_dual_view_ablation_partial_preview" / "main_table.tex":
        NIPS_ROOT / "tables" / "ablations" / "m2_dual_view_preview.tex",
    REPORTS_ROOT / "m2_constrained_decoding_ablation_partial_preview" / "main_table.tex":
        NIPS_ROOT / "tables" / "ablations" / "m2_constrained_decoding_preview.tex",
    PAPER_PLOTS_ROOT / "dual_view_motivation.pdf":
        NIPS_ROOT / "figures" / "dual_view_motivation.pdf",
    PAPER_PLOTS_ROOT / "dual_view_motivation.png":
        NIPS_ROOT / "figures" / "dual_view_motivation.png",
    PAPER_PLOTS_ROOT / "fewshot_trend.pdf":
        NIPS_ROOT / "figures" / "fewshot_trend.pdf",
    PAPER_PLOTS_ROOT / "fewshot_trend.png":
        NIPS_ROOT / "figures" / "fewshot_trend.png",
    PAPER_PLOTS_ROOT / "ablation_drops.pdf":
        NIPS_ROOT / "figures" / "ablation_drops.pdf",
    PAPER_PLOTS_ROOT / "ablation_drops.png":
        NIPS_ROOT / "figures" / "ablation_drops.png",
    PAPER_PLOTS_ROOT / "ucr_wtl_heatmap.pdf":
        NIPS_ROOT / "figures" / "ucr_wtl_heatmap.pdf",
    PAPER_PLOTS_ROOT / "ucr_wtl_heatmap.png":
        NIPS_ROOT / "figures" / "ucr_wtl_heatmap.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the paper-ready tables and figures from results/ into nips/."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned copies without modifying files.",
    )
    return parser.parse_args()


def relative_to_repo(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> int:
    args = parse_args()

    missing = [src for src in ASSET_MAP if not src.exists()]
    if missing:
        missing_str = "\n".join(f"- {relative_to_repo(path)}" for path in missing)
        raise FileNotFoundError(f"Missing source assets:\n{missing_str}")

    for src, dst in ASSET_MAP.items():
        print(f"{relative_to_repo(src)} -> {relative_to_repo(dst)}")
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if not args.dry_run:
        print("Sync complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
