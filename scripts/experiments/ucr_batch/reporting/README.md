# UCR Reporting

This module builds paper-ready UCR benchmark artifacts from the batch runner
outputs under `results/ucr_batches/...`.

Preferred entry point:

```bash
python scripts/experiments/ucr_batch/build_ucr_report.py \
  --report-config scripts/experiments/ucr_batch/reporting/examples/current_results_preview.json
```

The legacy `build_fewshot_paper_report.py` script still works as a compatibility
shim and forwards to the unified entry point.

## Report kinds

- `leaderboard`: multi-model few-shot comparison with summary table, appendix
  tables, and trend plots.
- `ablation`: variant-vs-reference comparison with preview/final semantics,
  compact main table, and signed-delta appendix tables.

For new configs, prefer `items` with `job_dir`. The loader derives
`results.txt` and `batch_config.json` automatically and records provenance in
`report_manifest.json`. Legacy leaderboard configs using `models` and
`results_txt` are still supported.

## Example configs

- `examples/current_results_preview.json`: leaderboard preview over current
  model baselines.
- `examples/pretrain_ablation_preview.json`: pretraining ablation preview using
  intersection coverage over the currently shared datasets.
- `examples/pretrain_ablation_final.json`: strict final version of the same
  ablation; it fails after writing `coverage_report.csv` until coverage is
  complete.

## Outputs

All outputs are written to `results/ucr_batches/reports/<report_name>/`.

Leaderboard reports include:

- `coverage_report.csv`
- `merged_results.csv`
- `summary_by_shot.csv`
- `rank_summary.csv`
- `main_table.tex`
- `appendix_shot_*.tex`
- `appendix_tables.tex`
- `fewshot_trend.png`
- `fewshot_trend.pdf`
- `report_manifest.json`

Ablation reports include:

- `coverage_report.csv`
- `merged_results.csv`
- `ablation_summary.csv`
- `cell_deltas.csv`
- `main_table.tex`
- `appendix_shot_*.tex`
- `appendix_tables.tex`
- `report_manifest.json`
