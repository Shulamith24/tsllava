# UCR Few-Shot Reporting

This module builds paper-ready few-shot comparison artifacts from one or more
`results.txt` ledgers produced by the UCR batch runner.

Entry point:

```bash
python scripts/experiments/ucr_batch/build_fewshot_paper_report.py \
  --report-config scripts/experiments/ucr_batch/reporting/examples/current_results_preview.json
```

Outputs are written under `results/ucr_batches/reports/<report_name>/` and include:

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

The report config is model-agnostic. As long as a new baseline exports the same
`results.txt` schema, it can be added through JSON without code changes.
