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
    higher_is_better: bool,
) -> str:
    rendered = f"{value:.2f}"
    if model_key in best:
        return rf"\textbf{{{rendered}}}"
    if model_key in second:
        return rf"\underline{{{rendered}}}"
    return rendered


def render_main_table(
    *,
    report_name: str,
    model_order: list[dict[str, str]],
    summary_by_shot: pd.DataFrame,
    rank_summary: pd.DataFrame,
    shots: list[str],
) -> str:
    pivot = summary_by_shot.pivot(index="model_key", columns="shot", values="accuracy_mean_pct")
    avg_scores = pivot.mean(axis=1)
    rank_lookup = rank_summary.set_index("model_key")["mean_rank"].to_dict()

    columns = [f"{shot}-shot" for shot in shots] + ["Avg", "AvgRank"]
    body_rows: list[str] = []

    shot_highlights: dict[str, tuple[set[str], set[str]]] = {}
    for shot in shots:
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
                _format_metric(
                    model_key=key,
                    value=float(pivot.loc[key, shot]),
                    best=best,
                    second=second,
                    higher_is_better=True,
                )
            )
        values.append(
            _format_metric(
                model_key=key,
                value=float(avg_scores.loc[key]),
                best=avg_highlights[0],
                second=avg_highlights[1],
                higher_is_better=True,
            )
        )
        values.append(
            _format_metric(
                model_key=key,
                value=float(rank_lookup[key]),
                best=rank_highlights[0],
                second=rank_highlights[1],
                higher_is_better=False,
            )
        )
        body_rows.append(" & ".join(values) + r" \\")

    column_spec = "l" + "r" * len(columns)
    caption_name = latex_escape(report_name)
    label = slugify(report_name)
    lines = [
        "% Requires \\usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{Few-shot UCR classification accuracy (\%) for {caption_name}. Best is bold and second-best is underlined.}}",
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
) -> str:
    shot_df = selected_frame[selected_frame["shot"].astype(str) == shot].copy()
    pivot = shot_df.pivot(index="dataset", columns="model_key", values="accuracy")
    pivot = pivot.loc[datasets, [model["key"] for model in model_order]]

    body_rows: list[str] = []
    for dataset in datasets:
        row_values = {model_key: float(pivot.loc[dataset, model_key]) * 100.0 for model_key in pivot.columns}
        best, second = _highlight_groups(row_values, higher_is_better=True)
        rendered = [latex_escape(dataset)]
        for model in model_order:
            key = model["key"]
            rendered.append(
                _format_metric(
                    model_key=key,
                    value=row_values[key],
                    best=best,
                    second=second,
                    higher_is_better=True,
                )
            )
        body_rows.append(" & ".join(rendered) + r" \\")

    label = slugify(report_name)
    column_spec = "l" + "r" * len(model_order)
    header = "Dataset & " + " & ".join(latex_escape(model["label"]) for model in model_order) + r" \\"
    caption = latex_escape(report_name)

    lines = [
        "% Requires \\usepackage{booktabs,longtable}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        rf"\begin{{longtable}}{{{column_spec}}}",
        rf"\caption{{Per-dataset {shot}-shot accuracy (\%) for {caption}.}}\label{{tab:{label}-shot-{shot}}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{Per-dataset {shot}-shot accuracy (\%) for {caption} (continued).}}\\",
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
