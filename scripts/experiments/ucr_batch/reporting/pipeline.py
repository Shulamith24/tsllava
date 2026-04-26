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
from .data import (
    analyze_coverage,
    build_ablation_cell_deltas,
    build_ablation_summary,
    build_paper_appendix_csv,
    build_paper_appendix_shot_bundle,
    build_paper_overall_summary,
    build_paper_rank_table,
    build_paper_wtl_table,
    build_rank_summary,
    build_summary_by_shot,
    infer_shots,
    load_result_bundles,
    resolve_primary_item,
)
from .latex import (
    render_ablation_appendix_table,
    render_ablation_main_table,
    render_appendix_table,
    render_main_table,
    render_paper_appendix_table,
    render_paper_overall_table,
    render_paper_rank_table,
    render_paper_wtl_table,
    write_appendix_wrapper,
)
from .plotting import plot_ablation_trend, plot_fewshot_trend


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _manifest_item_entry(bundle) -> dict[str, object]:
    return {
        "key": bundle.spec.key,
        "label": bundle.spec.label,
        "results_txt": str(bundle.run_ref.results_txt),
        "job_dir": str(bundle.run_ref.job_dir) if bundle.run_ref.job_dir else None,
        "batch_config_path": str(bundle.run_ref.batch_config_path) if bundle.run_ref.batch_config_path else None,
        "experiment": bundle.run_ref.experiment,
        "protocol": bundle.run_ref.protocol,
        "summary_kind": bundle.run_ref.summary_kind,
        "primary": bundle.spec.primary,
        "color": bundle.spec.color,
        "marker": bundle.spec.marker,
        "variant_tags": list(bundle.spec.variant_tags),
        "family": bundle.spec.family,
        "paper_label": bundle.spec.paper_label,
        "available_datasets": list(bundle.available_datasets),
        "available_shots": list(bundle.available_shots),
    }


def _build_model_order(config: ReportConfig) -> list[dict[str, object]]:
    return [
        {
            "key": item.key,
            "label": item.label,
            "paper_label": item.paper_label or item.label,
            "family": item.family or "",
            "primary": item.primary,
            "color": item.color,
            "marker": item.marker,
        }
        for item in config.items
    ]


def _build_coverage_frame(issues) -> pd.DataFrame:
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
    return coverage_df


def _write_leaderboard_outputs(
    *,
    config: ReportConfig,
    report_dir: Path,
    selected_frame: pd.DataFrame,
    selected_datasets: list[str],
    shots: list[str],
    generated_files: list[str],
) -> None:
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
            dataset_count=len(selected_datasets),
            num_runs=config.override_num_runs,
        ),
        encoding="utf-8",
    )

    generated_files.extend([str(summary_by_shot_path), str(rank_summary_path), str(main_table_path)])

    if config.appendix_tables_enabled:
        appendix_shot_files: list[str] = []
        for shot in shots:
            appendix_path = report_dir / f"appendix_shot_{shot}.tex"
            appendix_path.write_text(
                render_appendix_table(
                    report_name=config.report_name,
                    shot=shot,
                    datasets=selected_datasets,
                    model_order=model_order,
                    selected_frame=selected_frame,
                    dataset_count=len(selected_datasets),
                    num_runs=config.override_num_runs,
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

    if config.paper_tables_enabled:
        _write_paper_outputs(
            config=config,
            report_dir=report_dir,
            selected_frame=selected_frame,
            selected_datasets=selected_datasets,
            shots=shots,
            generated_files=generated_files,
        )


def _write_paper_outputs(
    *,
    config: ReportConfig,
    report_dir: Path,
    selected_frame: pd.DataFrame,
    selected_datasets: list[str],
    shots: list[str],
    generated_files: list[str],
) -> None:
    model_order = _build_model_order(config)
    overall_summary = build_paper_overall_summary(
        selected_frame=selected_frame,
        items=config.items,
        shots=shots,
    )
    overall_csv_path = _write_csv(overall_summary, report_dir / "paper_table_overall.csv")

    rank_table = build_paper_rank_table(
        selected_frame=selected_frame,
        items=config.items,
        shots=shots,
    )
    rank_csv_path = _write_csv(rank_table, report_dir / "paper_table_rank.csv")

    primary_item = resolve_primary_item(config.items)
    baseline_keys = config.wtl_baselines or tuple(item.key for item in config.items if item.key != primary_item.key)
    wtl_table = build_paper_wtl_table(
        selected_frame=selected_frame,
        items=config.items,
        shots=shots,
        primary_key=primary_item.key,
        baseline_keys=baseline_keys,
    )
    wtl_csv_path = _write_csv(wtl_table, report_dir / "paper_table_wtl.csv")

    overall_tex_path = report_dir / "paper_table_overall.tex"
    overall_tex_path.write_text(
        render_paper_overall_table(
            report_name=config.report_name,
            model_order=model_order,
            overall_summary=overall_summary,
            shots=shots,
            dataset_count=len(selected_datasets),
            num_runs=config.override_num_runs,
        ),
        encoding="utf-8",
    )

    rank_tex_path = report_dir / "paper_table_rank.tex"
    rank_tex_path.write_text(
        render_paper_rank_table(
            report_name=config.report_name,
            model_order=model_order,
            rank_table=rank_table,
            shots=shots,
            dataset_count=len(selected_datasets),
        ),
        encoding="utf-8",
    )

    wtl_tex_path = report_dir / "paper_table_wtl.tex"
    wtl_tex_path.write_text(
        render_paper_wtl_table(
            report_name=config.report_name,
            primary_label=primary_item.paper_label or primary_item.label,
            wtl_table=wtl_table,
            shots=shots,
            dataset_count=len(selected_datasets),
        ),
        encoding="utf-8",
    )

    generated_files.extend(
        [
            str(overall_csv_path),
            str(rank_csv_path),
            str(wtl_csv_path),
            str(overall_tex_path),
            str(rank_tex_path),
            str(wtl_tex_path),
        ]
    )

    paper_appendix_files: list[str] = []
    for shot in shots:
        bundle = build_paper_appendix_shot_bundle(
            selected_frame=selected_frame,
            items=config.items,
            shot=shot,
            datasets=selected_datasets,
        )
        appendix_csv_path = _write_csv(
            build_paper_appendix_csv(bundle=bundle, items=config.items, show_std=config.appendix_show_std),
            report_dir / f"paper_appendix_shot_{shot}.csv",
        )
        appendix_tex_path = report_dir / f"paper_appendix_shot_{shot}.tex"
        appendix_tex_path.write_text(
            render_paper_appendix_table(
                report_name=config.report_name,
                shot=shot,
                model_order=model_order,
                appendix_bundle=bundle,
                show_std=config.appendix_show_std,
                dataset_count=len(selected_datasets),
                num_runs=config.override_num_runs,
            ),
            encoding="utf-8",
        )
        generated_files.extend([str(appendix_csv_path), str(appendix_tex_path)])
        paper_appendix_files.append(appendix_tex_path.name)

    paper_wrapper_path = report_dir / "paper_appendix_tables.tex"
    wrapper_lines = ["% Wrapper file for paper appendix few-shot tables."]
    for idx, shot_file in enumerate(paper_appendix_files):
        if idx:
            wrapper_lines.append(r"\clearpage")
        wrapper_lines.append(rf"\input{{{shot_file}}}")
    paper_wrapper_path.write_text("\n".join(wrapper_lines) + "\n", encoding="utf-8")
    generated_files.append(str(paper_wrapper_path))


def _write_ablation_outputs(
    *,
    config: ReportConfig,
    report_dir: Path,
    selected_frame: pd.DataFrame,
    selected_datasets: list[str],
    shots: list[str],
    generated_files: list[str],
) -> None:
    model_order = _build_model_order(config)
    ablation_summary = build_ablation_summary(
        selected_frame=selected_frame,
        items=config.items,
        shots=shots,
        reference_key=config.reference_key or "",
    )
    ablation_summary_path = _write_csv(ablation_summary, report_dir / "ablation_summary.csv")
    cell_deltas = build_ablation_cell_deltas(
        selected_frame=selected_frame,
        items=config.items,
        reference_key=config.reference_key or "",
        datasets=selected_datasets,
        shots=shots,
    )
    cell_deltas_path = _write_csv(cell_deltas, report_dir / "cell_deltas.csv")

    main_table_path = report_dir / "main_table.tex"
    main_table_path.write_text(
        render_ablation_main_table(
            report_name=config.report_name,
            family_label=config.family_label or config.report_name,
            model_order=model_order,
            ablation_summary=ablation_summary,
            shots=shots,
            reference_key=config.reference_key or "",
            report_stage=config.report_stage,
            dataset_count=len(selected_datasets),
            coverage_mode=config.coverage_mode,
        ),
        encoding="utf-8",
    )

    generated_files.extend([str(ablation_summary_path), str(cell_deltas_path), str(main_table_path)])

    if config.appendix_tables_enabled:
        appendix_shot_files: list[str] = []
        for shot in shots:
            appendix_path = report_dir / f"appendix_shot_{shot}.tex"
            appendix_path.write_text(
                render_ablation_appendix_table(
                    report_name=config.report_name,
                    family_label=config.family_label or config.report_name,
                    shot=shot,
                    datasets=selected_datasets,
                    model_order=model_order,
                    reference_key=config.reference_key or "",
                    cell_deltas=cell_deltas,
                ),
                encoding="utf-8",
            )
            appendix_shot_files.append(appendix_path.name)
            generated_files.append(str(appendix_path))

        appendix_wrapper_path = write_appendix_wrapper(report_dir, appendix_shot_files)
        generated_files.append(str(appendix_wrapper_path))

    generated_plot_files = plot_ablation_trend(
        summary_csv=ablation_summary_path,
        output_dir=report_dir,
        model_order=model_order,
        report_name=config.report_name,
        reference_key=config.reference_key or "",
    )
    generated_files.extend(str(path) for path in generated_plot_files)


def generate_report(
    config_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    config = load_report_config(config_path)
    report_dir = (
        Path(output_root).resolve() / slugify(config.report_name)
        if output_root
        else REPORTS_ROOT / slugify(config.report_name)
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_result_bundles(config.items, override_num_runs=config.override_num_runs)
    frame = pd.concat([bundle.frame for bundle in bundles if not bundle.frame.empty], ignore_index=True)
    shots = list(config.shots) if config.shots else infer_shots(frame)

    archive_dir = resolve_ucr_archive(config.dataset_source)
    discovered_datasets = discover_datasets(archive_dir)
    if config.dataset_allowlist is None:
        expected_datasets = discovered_datasets
    else:
        discovered_set = set(discovered_datasets)
        missing = [dataset for dataset in config.dataset_allowlist if dataset not in discovered_set]
        if missing:
            raise ValueError(
                "dataset_allowlist contains datasets not found in dataset_source: "
                + ",".join(missing)
            )
        expected_datasets = list(config.dataset_allowlist)

    _deduped_frame, selected_frame, issues, selected_datasets, has_fatal_coverage = analyze_coverage(
        frame=frame,
        items=config.items,
        expected_datasets=expected_datasets,
        shots=shots,
        coverage_mode=config.coverage_mode,
    )

    coverage_df = _build_coverage_frame(issues)
    coverage_report_path = _write_csv(coverage_df, report_dir / "coverage_report.csv")

    if has_fatal_coverage:
        raise ValueError(
            f"Coverage validation failed for report '{config.report_name}'. "
            f"Inspect {coverage_report_path} for missing or duplicate entries."
        )

    merged_results_path = _write_csv(selected_frame, report_dir / "merged_results.csv")
    generated_files = [str(coverage_report_path), str(merged_results_path)]

    if config.report_kind == "leaderboard":
        _write_leaderboard_outputs(
            config=config,
            report_dir=report_dir,
            selected_frame=selected_frame,
            selected_datasets=selected_datasets,
            shots=shots,
            generated_files=generated_files,
        )
    elif config.report_kind == "ablation":
        _write_ablation_outputs(
            config=config,
            report_dir=report_dir,
            selected_frame=selected_frame,
            selected_datasets=selected_datasets,
            shots=shots,
            generated_files=generated_files,
        )
    else:
        raise ValueError(f"Unsupported report_kind: {config.report_kind}")

    manifest = {
        "report_name": config.report_name,
        "report_slug": slugify(config.report_name),
        "report_kind": config.report_kind,
        "report_stage": config.report_stage,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config_path": str(config.config_path),
        "dataset_source": str(config.dataset_source),
        "dataset_allowlist": list(config.dataset_allowlist) if config.dataset_allowlist is not None else None,
        "coverage_mode": config.coverage_mode,
        "paper_tables_enabled": config.paper_tables_enabled,
        "appendix_show_std": config.appendix_show_std,
        "appendix_tables_enabled": config.appendix_tables_enabled,
        "override_num_runs": config.override_num_runs,
        "wtl_baselines": list(config.wtl_baselines),
        "shots": shots,
        "expected_dataset_count": len(expected_datasets),
        "dataset_count": len(selected_datasets),
        "datasets": selected_datasets,
        "items": [_manifest_item_entry(bundle) for bundle in bundles],
        "generated_files": generated_files,
    }
    manifest["models"] = manifest["items"]
    if config.reference_key:
        manifest["reference_key"] = config.reference_key
    if config.family_label:
        manifest["family_label"] = config.family_label
    if config.report_kind == "ablation":
        manifest["shared_dataset_count"] = len(selected_datasets)

    manifest_path = report_dir / "report_manifest.json"
    manifest["generated_files"].append(str(manifest_path))
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
