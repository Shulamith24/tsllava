from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from ..ucr_datasets import discover_datasets, resolve_ucr_archive
except ImportError:
    from ucr_datasets import discover_datasets, resolve_ucr_archive
from .common import REPORTS_ROOT, ReportConfig, slugify
from .config import load_report_config
from .data import analyze_coverage, build_rank_summary, build_summary_by_shot, infer_shots, load_results_frame
from .latex import render_appendix_table, render_main_table, write_appendix_wrapper
from .plotting import plot_fewshot_trend


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _manifest_model_entry(model) -> dict[str, object]:
    return {
        "key": model.key,
        "label": model.label,
        "results_txt": str(model.results_txt),
        "primary": model.primary,
        "color": model.color,
        "marker": model.marker,
    }


def _build_model_order(config: ReportConfig) -> list[dict[str, object]]:
    return [
        {
            "key": model.key,
            "label": model.label,
            "primary": model.primary,
            "color": model.color,
            "marker": model.marker,
        }
        for model in config.models
    ]


def generate_report(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config = load_report_config(config_path)
    report_dir = Path(output_root).resolve() / slugify(config.report_name) if output_root else REPORTS_ROOT / slugify(config.report_name)
    report_dir.mkdir(parents=True, exist_ok=True)

    frame = load_results_frame(config.models)
    shots = list(config.shots) if config.shots else infer_shots(frame)

    archive_dir = resolve_ucr_archive(config.dataset_source)
    expected_datasets = discover_datasets(archive_dir)

    _deduped_frame, selected_frame, issues, selected_datasets, has_fatal_coverage = analyze_coverage(
        frame=frame,
        models=config.models,
        expected_datasets=expected_datasets,
        shots=shots,
        coverage_mode=config.coverage_mode,
    )

    coverage_df = pd.DataFrame(
        [
            {
                "severity": issue.severity,
                "issue_type": issue.issue_type,
                "model_key": issue.model_key,
                "model_label": issue.model_label,
                "dataset": issue.dataset,
                "shot": issue.shot,
                "details": issue.details,
            }
            for issue in issues
        ]
    )
    if coverage_df.empty:
        coverage_df = pd.DataFrame(
            columns=["severity", "issue_type", "model_key", "model_label", "dataset", "shot", "details"]
        )
    else:
        coverage_df = coverage_df.sort_values(["severity", "issue_type", "model_key", "dataset", "shot"]).reset_index(
            drop=True
        )
    coverage_report_path = _write_csv(coverage_df, report_dir / "coverage_report.csv")

    if has_fatal_coverage:
        raise ValueError(
            f"Coverage validation failed for report '{config.report_name}'. "
            f"Inspect {coverage_report_path} for missing or duplicate entries."
        )

    merged_results_path = _write_csv(selected_frame, report_dir / "merged_results.csv")
    summary_by_shot = build_summary_by_shot(selected_frame, shots)
    summary_by_shot_path = _write_csv(summary_by_shot, report_dir / "summary_by_shot.csv")

    rank_summary = build_rank_summary(selected_frame)
    rank_summary_path = _write_csv(rank_summary, report_dir / "rank_summary.csv")

    model_order = _build_model_order(config)
    main_table_path = report_dir / "main_table.tex"
    main_table_path.write_text(
        render_main_table(
            report_name=config.report_name,
            model_order=model_order,
            summary_by_shot=summary_by_shot,
            rank_summary=rank_summary,
            shots=shots,
        ),
        encoding="utf-8",
    )

    appendix_shot_files: list[str] = []
    generated_files = [
        str(coverage_report_path),
        str(merged_results_path),
        str(summary_by_shot_path),
        str(rank_summary_path),
        str(main_table_path),
    ]

    for shot in shots:
        appendix_path = report_dir / f"appendix_shot_{shot}.tex"
        appendix_path.write_text(
            render_appendix_table(
                report_name=config.report_name,
                shot=shot,
                datasets=selected_datasets,
                model_order=model_order,
                selected_frame=selected_frame,
            ),
            encoding="utf-8",
        )
        appendix_shot_files.append(appendix_path.name)
        generated_files.append(str(appendix_path))

    appendix_wrapper_path = write_appendix_wrapper(report_dir, appendix_shot_files)
    generated_files.append(str(appendix_wrapper_path))

    generated_plot_files = plot_fewshot_trend(
        summary_csv=summary_by_shot_path,
        output_dir=report_dir,
        model_order=model_order,
        report_name=config.report_name,
    )
    generated_files.extend(str(path) for path in generated_plot_files)

    manifest = {
        "report_name": config.report_name,
        "report_slug": slugify(config.report_name),
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config_path": str(config.config_path),
        "dataset_source": str(config.dataset_source),
        "coverage_mode": config.coverage_mode,
        "shots": shots,
        "dataset_count": len(selected_datasets),
        "datasets": selected_datasets,
        "models": [_manifest_model_entry(model) for model in config.models],
        "generated_files": generated_files,
    }
    manifest_path = report_dir / "report_manifest.json"
    manifest["generated_files"].append(str(manifest_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
