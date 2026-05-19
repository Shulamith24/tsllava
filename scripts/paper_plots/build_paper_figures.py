from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    SCRIPT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(SCRIPT_DIR.parent.parent))
    from scripts.paper_plots.ablation_drops import plot_ablation_drops
    from scripts.paper_plots.common import PAPER_OUTPUT_DIR
    from scripts.paper_plots.dual_view_motivation import parse_args as parse_dual_view_args, plot_dual_view_motivation
    from scripts.paper_plots.fewshot_scaling import plot_fewshot_scaling
    from scripts.paper_plots.wtl_heatmap import plot_wtl_heatmap
else:
    from .ablation_drops import plot_ablation_drops
    from .common import PAPER_OUTPUT_DIR
    from .dual_view_motivation import parse_args as parse_dual_view_args, plot_dual_view_motivation
    from .fewshot_scaling import plot_fewshot_scaling
    from .wtl_heatmap import plot_wtl_heatmap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the NeurIPS paper figures for TimeMorph.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PAPER_OUTPUT_DIR),
        help="Directory where PDF and PNG outputs are written.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to the UCR archive root used by the dual-view example.",
    )
    parser.add_argument(
        "--dual-view-dataset",
        type=str,
        default="TwoLeadECG",
        help="Dataset for the illustrative dual-view figure.",
    )
    parser.add_argument(
        "--dual-view-split",
        type=str,
        default="train",
        choices=["train", "test"],
    )
    parser.add_argument(
        "--dual-view-sample-index",
        type=int,
        default=0,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    dual_args = parse_dual_view_args(
        [
            "--dataset",
            args.dual_view_dataset,
            "--split",
            args.dual_view_split,
            "--sample_index",
            str(args.dual_view_sample_index),
            "--data_path",
            args.data_path,
            "--output_dir",
            str(output_dir),
        ]
    )
    plot_dual_view_motivation(dual_args)
    plot_fewshot_scaling(
        summary_csv=Path(__file__).resolve().parents[2] / "results" / "ucr_batches" / "reports" / "ucr_fewshot_paper_current" / "summary_by_shot.csv",
        merged_results_csv=Path(__file__).resolve().parents[2] / "results" / "ucr_batches" / "reports" / "ucr_fewshot_paper_current" / "merged_results.csv",
        output_dir=output_dir,
    )
    plot_ablation_drops(output_dir=output_dir)
    plot_wtl_heatmap(output_dir=output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
