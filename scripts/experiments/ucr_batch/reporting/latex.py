from __future__ import annotations

from pathlib import Path

import pandas as pd

from .common import latex_escape, slugify


def _highlight_groups(values: dict[str, float], *, higher_is_better: bool) -> tuple[set[str], set[str]]:
    if not values:
        return set(), set()

    ranked = sorted(values.items(), key=lambda item: item[1], reverse=higher_is_better)
    distinct_values: list[float] = []
    for _key, value in ranked:
        if not distinct_values or abs(value - distinct_values[-1]) > 1e-12:
            distinct_values.append(value)

    best_value = distinct_values[0]
    second_value = distinct_values[1] if len(distinct_values) > 1 else None
    best = {key for key, value in values.items() if abs(value - best_value) <= 1e-12}
    second = set()
    if second_value is not None:
        second = {key for key, value in values.items() if abs(value - second_value) <= 1e-12}
    return best, second


def _format_metric(
    *,
    model_key: str,
    value: float,
    best: set[str],
    second: set[str],
) -> str:
    rendered = f"{value:.2f}"
    if model_key in best:
        return rf"\textbf{{{rendered}}}"
    if model_key in second:
        return rf"\underline{{{rendered}}}"
    return rendered


def _format_metric_text(
    *,
    model_key: str,
    rendered: str,
    best: set[str],
    second: set[str],
) -> str:
    if model_key in best:
        return rf"\textbf{{{rendered}}}"
    if model_key in second:
        return rf"\underline{{{rendered}}}"
    return rendered


def _format_signed(value: float | object) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):+.2f}"


def _lookup_value(frame: pd.DataFrame, row_key: str, column_key: str) -> float | object:
    if row_key not in frame.index or column_key not in frame.columns:
        return pd.NA
    return frame.loc[row_key, column_key]


def _paper_label(model: dict[str, object]) -> str:
    return str(model.get("paper_label") or model["label"])


def _format_optional_metric(
    *,
    model_key: str,
    value: float | object,
    best: set[str],
    second: set[str],
    missing_text: str = "--",
) -> str:
    if pd.isna(value):
        return missing_text
    return _format_metric(model_key=model_key, value=float(value), best=best, second=second)


def _format_appendix_metric(
    *,
    model_key: str,
    value: float | object,
    std_value: float | object,
    best: set[str],
    second: set[str],
    show_std: bool,
) -> str:
    if pd.isna(value):
        return ""
    rendered = f"{float(value):.2f}"
    if show_std and not pd.isna(std_value):
        rendered = f"{rendered} $\\pm$ {float(std_value):.2f}"
    return _format_metric_text(model_key=model_key, rendered=rendered, best=best, second=second)


def render_main_table(
    *,
    report_name: str,
    model_order: list[dict[str, str]],
    summary_by_shot: pd.DataFrame,
    rank_summary: pd.DataFrame,
    shots: list[str],
    dataset_count: int | None = None,
    num_runs: int | None = None,
) -> str:
    pivot = summary_by_shot.pivot(index="model_key", columns="shot", values="accuracy_mean_pct")
    avg_scores = pivot.mean(axis=1)
    rank_lookup = rank_summary.set_index("model_key")["mean_rank"].to_dict()

    columns = [f"{shot}-shot" for shot in shots] + ["Avg", "AvgRank"]
    body_rows: list[str] = []

    shot_highlights: dict[str, tuple[set[str], set[str]]] = {}
    for shot in shots:
        if shot not in pivot.columns:
            values = {}
        else:
            values = {key: float(value) for key, value in pivot[shot].dropna().items()}
        shot_highlights[shot] = _highlight_groups(values, higher_is_better=True)

    avg_highlights = _highlight_groups({key: float(value) for key, value in avg_scores.items()}, higher_is_better=True)
    rank_highlights = _highlight_groups(rank_lookup, higher_is_better=False)

    for model in model_order:
        key = model["key"]
        label = latex_escape(model["label"])
        values = [label]
        for shot in shots:
            best, second = shot_highlights[shot]
            values.append(
                _format_optional_metric(
                    model_key=key,
                    value=_lookup_value(pivot, key, shot),
                    best=best,
                    second=second,
                )
            )
        values.append(
            _format_optional_metric(
                model_key=key,
                value=avg_scores.loc[key] if key in avg_scores.index else pd.NA,
                best=avg_highlights[0],
                second=avg_highlights[1],
            )
        )
        values.append(
            _format_optional_metric(
                model_key=key,
                value=rank_lookup.get(key, pd.NA),
                best=rank_highlights[0],
                second=rank_highlights[1],
            )
        )
        body_rows.append(" & ".join(values) + r" \\")

    column_spec = "l" + "r" * len(columns)
    label = slugify(report_name)
    if dataset_count is None:
        caption = "Few-shot UCR classification accuracy (\\%) across the evaluated benchmark. Best is bold and second-best is underlined."
    else:
        run_clause = f" We report the mean over {num_runs} runs." if num_runs is not None else ""
        caption = (
            f"Few-shot classification accuracy (\\%) on {dataset_count} shared UCR datasets."
            f"{run_clause} Best is bold and second-best is underlined."
        )
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}-fewshot-main}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Model & " + " & ".join(columns) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_appendix_table(
    *,
    report_name: str,
    shot: str,
    datasets: list[str],
    model_order: list[dict[str, str]],
    selected_frame: pd.DataFrame,
    dataset_count: int | None = None,
    num_runs: int | None = None,
) -> str:
    shot_df = selected_frame[selected_frame["shot"].astype(str) == shot].copy()
    pivot = shot_df.pivot(index="dataset", columns="model_key", values="accuracy").reindex(
        index=datasets, columns=[model["key"] for model in model_order]
    )

    body_rows: list[str] = []
    for dataset in datasets:
        row_values = {
            model_key: float(pivot.loc[dataset, model_key]) * 100.0
            for model_key in pivot.columns
            if not pd.isna(pivot.loc[dataset, model_key])
        }
        best, second = _highlight_groups(row_values, higher_is_better=True)
        rendered = [latex_escape(dataset)]
        for model in model_order:
            key = model["key"]
            value = pivot.loc[dataset, key]
            if pd.isna(value):
                rendered.append("")
            else:
                rendered.append(
                    _format_metric(
                        model_key=key,
                        value=float(value) * 100.0,
                        best=best,
                        second=second,
                    )
                )
        body_rows.append(" & ".join(rendered) + r" \\")

    label = slugify(report_name)
    column_spec = "l" + "r" * len(model_order)
    header = "Dataset & " + " & ".join(latex_escape(model["label"]) for model in model_order) + r" \\"
    lines = [
        "% Requires \\usepackage{booktabs,longtable,pdflscape}",
        r"\begin{landscape}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{1pt}",
        r"\tiny",
        rf"\begin{{longtable}}{{{column_spec}}}",
        rf"\caption{{Per-dataset {shot}-shot classification accuracy (\%) on {dataset_count or len(datasets)} shared UCR datasets. Values are means{f' over {num_runs} runs' if num_runs is not None else ''}.}}\label{{tab:{label}-shot-{shot}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{Per-dataset {shot}-shot classification accuracy (\%) on {dataset_count or len(datasets)} shared UCR datasets (continued).}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{len(model_order) + 1}}}{{r}}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body_rows,
        r"\end{longtable}",
        r"\endgroup",
        r"\end{landscape}",
    ]
    return "\n".join(lines) + "\n"


def render_ablation_main_table(
    *,
    report_name: str,
    family_label: str,
    model_order: list[dict[str, str]],
    ablation_summary: pd.DataFrame,
    shots: list[str],
    reference_key: str,
    report_stage: str,
    dataset_count: int,
) -> str:
    summary = ablation_summary.set_index("model_key")
    shot_highlights = {
        shot: _highlight_groups(
            {key: float(summary.loc[key, f"shot_{shot}_accuracy_pct"]) for key in summary.index},
            higher_is_better=True,
        )
        for shot in shots
    }
    avg_highlights = _highlight_groups(
        {key: float(summary.loc[key, "avg_accuracy_pct"]) for key in summary.index},
        higher_is_better=True,
    )

    columns = [f"{shot}-shot" for shot in shots] + ["Avg", "Delta vs Ref", "W/T/L"]
    body_rows: list[str] = []
    for model in model_order:
        key = model["key"]
        rendered = [latex_escape(model["label"])]
        for shot in shots:
            best, second = shot_highlights[shot]
            rendered.append(
                _format_metric(
                    model_key=key,
                    value=float(summary.loc[key, f"shot_{shot}_accuracy_pct"]),
                    best=best,
                    second=second,
                )
            )
        rendered.append(
            _format_metric(
                model_key=key,
                value=float(summary.loc[key, "avg_accuracy_pct"]),
                best=avg_highlights[0],
                second=avg_highlights[1],
            )
        )
        if key == reference_key:
            rendered.extend(["--", "--"])
        else:
            rendered.append(_format_signed(summary.loc[key, "delta_vs_reference_pct"]))
            rendered.append(str(summary.loc[key, "win_tie_loss"]))
        body_rows.append(" & ".join(rendered) + r" \\")

    stage_note = ""
    if report_stage == "preview":
        stage_note = f" Preview uses the {dataset_count} shared datasets currently available."
    label = slugify(report_name)
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Ablation results for {latex_escape(family_label)} on UCR few-shot classification (\%). Positive delta means improvement over the reference.{stage_note}}}",
        rf"\label{{tab:{label}-ablation-main}}",
        rf"\begin{{tabular}}{{l{'r' * len(columns)}}}",
        r"\toprule",
        "Variant & " + " & ".join(columns) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_ablation_appendix_table(
    *,
    report_name: str,
    family_label: str,
    shot: str,
    datasets: list[str],
    model_order: list[dict[str, str]],
    reference_key: str,
    cell_deltas: pd.DataFrame,
) -> str:
    reference_label = next(model["label"] for model in model_order if model["key"] == reference_key)
    variant_models = [model for model in model_order if model["key"] != reference_key]
    shot_df = cell_deltas[cell_deltas["shot"].astype(str) == shot].copy()
    shot_df = shot_df.set_index("dataset").loc[datasets].reset_index()

    body_rows: list[str] = []
    for dataset in datasets:
        dataset_row = shot_df[shot_df["dataset"].astype(str) == dataset].iloc[0]
        accuracy_values = {
            reference_key: float(dataset_row["reference_accuracy_pct"]),
            **{
                model["key"]: float(dataset_row[f"{model['key']}_accuracy_pct"])
                for model in variant_models
            },
        }
        best, second = _highlight_groups(accuracy_values, higher_is_better=True)
        rendered = [
            latex_escape(dataset),
            _format_metric(
                model_key=reference_key,
                value=float(dataset_row["reference_accuracy_pct"]),
                best=best,
                second=second,
            ),
        ]
        for model in variant_models:
            key = model["key"]
            rendered.append(
                _format_metric(
                    model_key=key,
                    value=float(dataset_row[f"{key}_accuracy_pct"]),
                    best=best,
                    second=second,
                )
            )
            rendered.append(_format_signed(dataset_row[f"{key}_delta_vs_reference_pct"]))
        body_rows.append(" & ".join(rendered) + r" \\")

    columns = ["Dataset", reference_label]
    for model in variant_models:
        columns.extend([model["label"], f"Delta vs {reference_label}"])

    label = slugify(report_name)
    caption = latex_escape(family_label)
    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        rf"\begin{{longtable}}{{l{'r' * (len(columns) - 1)}}}",
        rf"\caption{{Per-dataset {shot}-shot ablation results (\%) for {caption}, with signed deltas against {latex_escape(reference_label)}.}}\label{{tab:{label}-ablation-shot-{shot}}}\\",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{Per-dataset {shot}-shot ablation results (\%) for {caption} (continued).}}\\",
        r"\toprule",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{len(columns)}}}{{r}}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body_rows,
        r"\end{longtable}",
        r"\endgroup",
    ]
    return "\n".join(lines) + "\n"


def render_paper_overall_table(
    *,
    report_name: str,
    model_order: list[dict[str, object]],
    overall_summary: pd.DataFrame,
    shots: list[str],
    dataset_count: int | None = None,
    num_runs: int | None = None,
) -> str:
    summary = overall_summary.set_index("model_key")
    shot_highlights = {
        shot: _highlight_groups(
            {
                key: float(summary.loc[key, f"shot_{shot}_accuracy_pct"])
                for key in summary.index
                if not pd.isna(summary.loc[key, f"shot_{shot}_accuracy_pct"])
            },
            higher_is_better=True,
        )
        for shot in shots
    }

    body_rows: list[str] = []
    for model in model_order:
        key = str(model["key"])
        rendered = [latex_escape(_paper_label(model)), latex_escape(str(model.get("family") or ""))]
        for shot in shots:
            best, second = shot_highlights[shot]
            rendered.append(
                _format_optional_metric(
                    model_key=key,
                    value=summary.loc[key, f"shot_{shot}_accuracy_pct"] if key in summary.index else pd.NA,
                    best=best,
                    second=second,
                )
            )
        body_rows.append(" & ".join(rendered) + r" \\")

    columns = ["Method", "Family"] + [f"{shot}-shot Acc. $\\uparrow$" for shot in shots]
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Overall few-shot classification performance on {f'{dataset_count} shared' if dataset_count is not None else 'the shared'} UCR datasets{f', averaged over {num_runs} runs' if num_runs is not None else ''}.}}",
        rf"\label{{tab:{slugify(report_name)}-paper-overall}}",
        rf"\begin{{tabular}}{{ll{'r' * len(shots)}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_paper_rank_table(
    *,
    report_name: str,
    model_order: list[dict[str, object]],
    rank_table: pd.DataFrame,
    shots: list[str],
    dataset_count: int | None = None,
) -> str:
    summary = rank_table.set_index("model_key")
    shot_highlights = {
        shot: _highlight_groups(
            {
                key: float(summary.loc[key, f"shot_{shot}_rank"])
                for key in summary.index
                if not pd.isna(summary.loc[key, f"shot_{shot}_rank"])
            },
            higher_is_better=False,
        )
        for shot in shots
    }
    avg_highlights = _highlight_groups(
        {key: float(summary.loc[key, "avg_rank"]) for key in summary.index if not pd.isna(summary.loc[key, "avg_rank"])},
        higher_is_better=False,
    )

    body_rows: list[str] = []
    for model in model_order:
        key = str(model["key"])
        rendered = [latex_escape(_paper_label(model))]
        for shot in shots:
            best, second = shot_highlights[shot]
            rendered.append(
                _format_optional_metric(
                    model_key=key,
                    value=summary.loc[key, f"shot_{shot}_rank"] if key in summary.index else pd.NA,
                    best=best,
                    second=second,
                )
            )
        rendered.append(
            _format_optional_metric(
                model_key=key,
                value=summary.loc[key, "avg_rank"] if key in summary.index else pd.NA,
                best=avg_highlights[0],
                second=avg_highlights[1],
            )
        )
        body_rows.append(" & ".join(rendered) + r" \\")

    columns = ["Method"] + [f"{shot}-shot Rank $\\downarrow$" for shot in shots] + ["Avg. Rank $\\downarrow$"]
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Average rank on {f'{dataset_count} shared' if dataset_count is not None else 'the shared'} UCR datasets. Lower is better.}}",
        rf"\label{{tab:{slugify(report_name)}-paper-rank}}",
        rf"\begin{{tabular}}{{l{'r' * (len(shots) + 1)}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_paper_wtl_table(
    *,
    report_name: str,
    primary_label: str,
    wtl_table: pd.DataFrame,
    shots: list[str],
    dataset_count: int | None = None,
) -> str:
    body_rows: list[str] = []
    for row in wtl_table.itertuples(index=False):
        rendered = [latex_escape(str(row.baseline_label))]
        for shot in shots:
            rendered.append(latex_escape(str(getattr(row, f"shot_{shot}_wtl"))))
        body_rows.append(" & ".join(rendered) + r" \\")

    columns = ["Baseline"] + [f"{shot}-shot" for shot in shots]
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Win/Tie/Loss of {latex_escape(primary_label)} against selected baselines on {f'{dataset_count} shared' if dataset_count is not None else 'the shared'} UCR datasets.}}",
        rf"\label{{tab:{slugify(report_name)}-paper-wtl}}",
        rf"\begin{{tabular}}{{l{'r' * len(shots)}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def render_paper_appendix_table(
    *,
    report_name: str,
    shot: str,
    model_order: list[dict[str, object]],
    appendix_bundle,
    show_std: bool,
    dataset_count: int | None = None,
    num_runs: int | None = None,
) -> str:
    datasets = appendix_bundle.accuracy_pct.index.tolist()
    model_keys = [str(model["key"]) for model in model_order]

    body_rows: list[str] = []
    for dataset in datasets:
        row_values = {
            key: float(appendix_bundle.accuracy_pct.loc[dataset, key])
            for key in model_keys
            if not pd.isna(appendix_bundle.accuracy_pct.loc[dataset, key])
        }
        best, second = _highlight_groups(row_values, higher_is_better=True)
        rendered = [latex_escape(str(dataset))]
        for model in model_order:
            key = str(model["key"])
            rendered.append(
                _format_appendix_metric(
                    model_key=key,
                    value=appendix_bundle.accuracy_pct.loc[dataset, key],
                    std_value=appendix_bundle.std_pct.loc[dataset, key],
                    best=best,
                    second=second,
                    show_std=show_std,
                )
            )
        body_rows.append(" & ".join(rendered) + r" \\")

    summary_rows: list[str] = []
    summary_specs = [
        ("Avg. Acc.", "avg_accuracy_pct", True),
        ("Avg. Rank", "avg_rank", False),
        ("#Best", "num_best", True),
    ]
    for label, column_name, higher_is_better in summary_specs:
        values = {
            key: float(appendix_bundle.summary.loc[key, column_name])
            for key in model_keys
            if not pd.isna(appendix_bundle.summary.loc[key, column_name])
        }
        best, second = _highlight_groups(values, higher_is_better=higher_is_better)
        rendered = [latex_escape(label)]
        for model in model_order:
            key = str(model["key"])
            value = appendix_bundle.summary.loc[key, column_name]
            if pd.isna(value):
                rendered.append("")
            elif column_name == "num_best":
                rendered.append(
                    _format_metric_text(
                        model_key=key,
                        rendered=str(int(value)),
                        best=best,
                        second=second,
                    )
                )
            else:
                rendered.append(
                    _format_metric(
                        model_key=key,
                        value=float(value),
                        best=best,
                        second=second,
                    )
                )
        summary_rows.append(" & ".join(rendered) + r" \\")

    header = "Dataset & " + " & ".join(latex_escape(_paper_label(model)) for model in model_order) + r" \\"
    lines = [
        "% Requires \\usepackage{booktabs,longtable,pdflscape}",
        r"\begin{landscape}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{1pt}",
        r"\tiny",
        rf"\begin{{longtable}}{{l{'r' * len(model_order)}}}",
        rf"\caption{{Per-dataset {shot}-shot classification results on {dataset_count or len(datasets)} shared UCR datasets. Each value is reported as mean $\pm$ standard deviation{f' over {num_runs} runs' if num_runs is not None else ''}.}}\label{{tab:{slugify(report_name)}-paper-shot-{shot}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{Per-dataset {shot}-shot classification results on {dataset_count or len(datasets)} shared UCR datasets (continued).}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        rf"\midrule \multicolumn{{{len(model_order) + 1}}}{{r}}{{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        *body_rows,
        r"\midrule",
        *summary_rows,
        r"\end{longtable}",
        r"\endgroup",
        r"\end{landscape}",
    ]
    return "\n".join(lines) + "\n"


def write_appendix_wrapper(output_dir: Path, shot_files: list[str]) -> Path:
    wrapper_path = output_dir / "appendix_tables.tex"
    lines = ["% Wrapper file for all appendix few-shot tables."]
    for idx, shot_file in enumerate(shot_files):
        if idx:
            lines.append(r"\clearpage")
        lines.append(rf"\input{{{shot_file}}}")
    wrapper_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return wrapper_path
