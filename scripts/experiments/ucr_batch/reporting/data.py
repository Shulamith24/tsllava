from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import ReportItemSpec, RunRef, sort_shots


@dataclass(frozen=True)
class CoverageIssue:
    severity: str
    issue_type: str
    model_key: str
    model_label: str
    dataset: str
    shot: str
    details: str


@dataclass(frozen=True)
class ResultBundle:
    spec: ReportItemSpec
    run_ref: RunRef
    frame: pd.DataFrame
    available_datasets: tuple[str, ...]
    available_shots: tuple[str, ...]


@dataclass(frozen=True)
class PaperAppendixShotBundle:
    shot: str
    accuracy_pct: pd.DataFrame
    std_pct: pd.DataFrame
    rank: pd.DataFrame
    summary: pd.DataFrame


def _load_batch_config(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"batch config must be a JSON object: {path}")
    return payload


def _build_run_ref(item: ReportItemSpec) -> RunRef:
    batch_config = _load_batch_config(item.batch_config_path)
    experiment = None
    protocol = None
    summary_kind = None
    if batch_config:
        experiment = str(batch_config.get("experiment", "")).strip() or None
        protocol = str(batch_config.get("protocol", "")).strip() or None
        summary_kind = str(batch_config.get("summary_kind", "")).strip() or None
    return RunRef(
        key=item.key,
        results_txt=item.results_txt,
        job_dir=item.job_dir,
        batch_config_path=item.batch_config_path if item.batch_config_path and item.batch_config_path.exists() else None,
        batch_config=batch_config,
        experiment=experiment,
        protocol=protocol,
        summary_kind=summary_kind,
    )


def _read_ledger_rows(item: ReportItemSpec, run_ref: RunRef) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(item.results_txt, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            shot = str(row.get("shot", "")).strip()
            status = str(row.get("status", "")).strip()
            if not dataset or not shot or shot == "__dataset__" or status != "success":
                continue
            rows.append(
                {
                    "model_key": item.key,
                    "model_label": item.label,
                    "family": item.family or "",
                    "paper_label": item.paper_label or item.label,
                    "results_txt": str(item.results_txt),
                    "job_dir": str(item.job_dir) if item.job_dir else "",
                    "batch_config_path": str(run_ref.batch_config_path) if run_ref.batch_config_path else "",
                    "experiment": run_ref.experiment or "",
                    "protocol": run_ref.protocol or "",
                    "summary_kind": run_ref.summary_kind or "",
                    "primary": item.primary,
                    "color": item.color or "",
                    "marker": item.marker or "",
                    "variant_tags": ",".join(item.variant_tags),
                    "dataset": dataset,
                    "shot": shot,
                    "accuracy": float(row.get("accuracy", "")),
                    "accuracy_std": (
                        float(row.get("accuracy_std", ""))
                        if str(row.get("accuracy_std", "")).strip()
                        else pd.NA
                    ),
                    "num_runs": str(row.get("num_runs", "")).strip(),
                }
            )
    return rows


def load_result_bundles(
    items: tuple[ReportItemSpec, ...],
    *,
    override_num_runs: int | None = None,
) -> tuple[ResultBundle, ...]:
    bundles: list[ResultBundle] = []
    total_rows = 0
    for item in items:
        if not item.results_txt.exists():
            raise FileNotFoundError(f"Missing results.txt for item {item.key}: {item.results_txt}")
        run_ref = _build_run_ref(item)
        rows = _read_ledger_rows(item, run_ref)
        frame = pd.DataFrame.from_records(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    "model_key",
                    "model_label",
                    "family",
                    "paper_label",
                    "results_txt",
                    "job_dir",
                    "batch_config_path",
                    "experiment",
                    "protocol",
                    "summary_kind",
                    "primary",
                    "color",
                    "marker",
                    "variant_tags",
                    "dataset",
                    "shot",
                    "accuracy",
                    "accuracy_std",
                    "num_runs",
                ]
            )
            available_datasets: tuple[str, ...] = tuple()
            available_shots: tuple[str, ...] = tuple()
        else:
            frame["shot"] = frame["shot"].astype(str)
            frame["dataset"] = frame["dataset"].astype(str)
            if override_num_runs is not None:
                frame["num_runs"] = str(override_num_runs)
            else:
                frame["num_runs"] = frame["num_runs"].astype(str)
            available_datasets = tuple(sorted(frame["dataset"].unique().tolist()))
            available_shots = tuple(sort_shots(frame["shot"].unique().tolist()))
            total_rows += len(frame)

        bundles.append(
            ResultBundle(
                spec=item,
                run_ref=run_ref,
                frame=frame,
                available_datasets=available_datasets,
                available_shots=available_shots,
            )
        )

    if total_rows == 0:
        raise ValueError("No successful few-shot records were found in the configured results files")
    return tuple(bundles)


def load_results_frame(items: tuple[ReportItemSpec, ...]) -> pd.DataFrame:
    bundles = load_result_bundles(items)
    frames = [bundle.frame for bundle in bundles if not bundle.frame.empty]
    frame = pd.concat(frames, ignore_index=True)
    frame["shot"] = frame["shot"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["num_runs"] = frame["num_runs"].astype(str)
    return frame


def infer_shots(frame: pd.DataFrame) -> list[str]:
    per_model: list[set[str]] = []
    for model_key, model_df in frame.groupby("model_key"):
        numeric_shots = {shot for shot in model_df["shot"].unique().tolist() if shot.isdigit()}
        if not numeric_shots:
            raise ValueError(f"Unable to infer numeric few-shot set for item {model_key}")
        per_model.append(numeric_shots)

    shared = set.intersection(*per_model)
    if not shared:
        raise ValueError("Unable to infer a shared numeric shot set across the configured items")
    return sort_shots(list(shared))


def analyze_coverage(
    *,
    frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    expected_datasets: list[str],
    shots: list[str],
    coverage_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[CoverageIssue], list[str], bool]:
    issues: list[CoverageIssue] = []
    duplicate_groups = frame.groupby(["model_key", "dataset", "shot"]).size().reset_index(name="count")
    duplicate_groups = duplicate_groups[duplicate_groups["count"] > 1]
    item_by_key = {item.key: item for item in items}

    for row in duplicate_groups.itertuples(index=False):
        item = item_by_key[row.model_key]
        issues.append(
            CoverageIssue(
                severity="error",
                issue_type="duplicate_success",
                model_key=item.key,
                model_label=item.label,
                dataset=row.dataset,
                shot=row.shot,
                details=f"{row.count} success rows share the same dataset/shot key",
            )
        )

    deduped = frame.drop_duplicates(subset=["model_key", "dataset", "shot"], keep="first").copy()
    deduped = deduped[deduped["shot"].isin(shots)].copy()

    expected_dataset_set = set(expected_datasets)
    unexpected = deduped.loc[~deduped["dataset"].isin(expected_dataset_set), ["model_key", "dataset"]].drop_duplicates()
    for row in unexpected.itertuples(index=False):
        item = item_by_key[row.model_key]
        issues.append(
            CoverageIssue(
                severity="warning",
                issue_type="unexpected_dataset",
                model_key=item.key,
                model_label=item.label,
                dataset=row.dataset,
                shot="",
                details="dataset is present in results.txt but not in dataset_source",
            )
        )

    deduped = deduped[deduped["dataset"].isin(expected_dataset_set)].copy()

    for item in items:
        item_df = deduped[deduped["model_key"] == item.key]
        num_runs_sets = (
            item_df.groupby("shot")["num_runs"]
            .apply(lambda series: sorted({value for value in series.tolist() if value}))
            .to_dict()
        )
        for shot in shots:
            values = num_runs_sets.get(shot, [])
            if len(values) > 1:
                issues.append(
                    CoverageIssue(
                        severity="warning",
                        issue_type="mixed_num_runs",
                        model_key=item.key,
                        model_label=item.label,
                        dataset="",
                        shot=shot,
                        details="num_runs varies across datasets: " + ",".join(values),
                    )
                )

    complete_by_model: dict[str, set[str]] = {}
    for item in items:
        complete_datasets: set[str] = set()
        item_df = deduped[deduped["model_key"] == item.key]
        for dataset, dataset_df in item_df.groupby("dataset"):
            dataset_shots = set(dataset_df["shot"].tolist())
            if all(shot in dataset_shots for shot in shots):
                complete_datasets.add(dataset)
        complete_by_model[item.key] = complete_datasets

    if coverage_mode == "strict":
        selected_datasets = list(expected_datasets)
    elif coverage_mode == "intersection":
        common = set(expected_datasets)
        for datasets in complete_by_model.values():
            common &= datasets
        selected_datasets = [dataset for dataset in expected_datasets if dataset in common]
    else:
        selected_datasets = list(expected_datasets)

    fatal_issues = bool(not duplicate_groups.empty)
    if coverage_mode == "strict":
        for item in items:
            available_keys = {
                (row.dataset, row.shot)
                for row in deduped.loc[deduped["model_key"] == item.key, ["dataset", "shot"]].itertuples(index=False)
            }
            for dataset in expected_datasets:
                for shot in shots:
                    if (dataset, shot) not in available_keys:
                        issues.append(
                            CoverageIssue(
                                severity="error",
                                issue_type="missing_result",
                                model_key=item.key,
                                model_label=item.label,
                                dataset=dataset,
                                shot=shot,
                                details="missing success record required by strict coverage",
                            )
                        )
                        fatal_issues = True
    elif coverage_mode == "intersection":
        for item in items:
            for dataset in expected_datasets:
                if dataset not in complete_by_model[item.key]:
                    missing_shots = [
                        shot
                        for shot in shots
                        if (
                            (
                                (deduped["model_key"] == item.key)
                                & (deduped["dataset"] == dataset)
                                & (deduped["shot"] == shot)
                            ).sum()
                            == 0
                        )
                    ]
                    if missing_shots:
                        issues.append(
                            CoverageIssue(
                                severity="warning",
                                issue_type="excluded_from_intersection",
                                model_key=item.key,
                                model_label=item.label,
                                dataset=dataset,
                                shot=",".join(missing_shots),
                                details="dataset excluded because not all shots are present for this item",
                            )
                        )

        if not selected_datasets:
            issues.append(
                CoverageIssue(
                    severity="error",
                    issue_type="empty_intersection",
                    model_key="",
                    model_label="",
                    dataset="",
                    shot="",
                    details="no common dataset remains after applying intersection coverage",
                )
            )
            fatal_issues = True
    else:
        for item in items:
            available_keys = {
                (row.dataset, row.shot)
                for row in deduped.loc[deduped["model_key"] == item.key, ["dataset", "shot"]].itertuples(index=False)
            }
            for dataset in expected_datasets:
                missing_shots = [shot for shot in shots if (dataset, shot) not in available_keys]
                if missing_shots:
                    issues.append(
                        CoverageIssue(
                            severity="warning",
                            issue_type="missing_result_sparse",
                            model_key=item.key,
                            model_label=item.label,
                            dataset=dataset,
                            shot=",".join(missing_shots),
                            details="missing success record retained as blank cells under sparse coverage",
                        )
                    )

    selected_frame = deduped[deduped["dataset"].isin(selected_datasets)].copy()
    selected_frame["shot"] = pd.Categorical(selected_frame["shot"], categories=shots, ordered=True)
    selected_frame["dataset"] = pd.Categorical(selected_frame["dataset"], categories=selected_datasets, ordered=True)
    selected_frame = selected_frame.sort_values(["dataset", "shot", "model_key"]).reset_index(drop=True)

    issues_df = pd.DataFrame(
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
    if issues_df.empty:
        issues_df = pd.DataFrame(
            columns=["severity", "issue_type", "model_key", "model_label", "dataset", "shot", "details"]
        )
    return deduped, selected_frame, issues, selected_datasets, fatal_issues


def build_summary_by_shot(selected_frame: pd.DataFrame, shots: list[str]) -> pd.DataFrame:
    summary = (
        selected_frame.groupby(["model_key", "model_label", "primary", "color", "marker", "shot"], observed=True)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_mean_pct=("accuracy", lambda values: values.mean() * 100.0),
            num_datasets=("dataset", "nunique"),
            num_runs_values=("num_runs", lambda values: ",".join(sorted({value for value in values.tolist() if value}))),
        )
        .reset_index()
    )
    summary["shot"] = summary["shot"].astype(str)
    summary["shot_index"] = summary["shot"].map({shot: idx for idx, shot in enumerate(shots)})
    return summary.sort_values(["shot_index", "model_key"]).drop(columns=["shot_index"]).reset_index(drop=True)


def _average_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=False)


def build_rank_summary(selected_frame: pd.DataFrame) -> pd.DataFrame:
    ranked = selected_frame.copy()
    ranked["rank"] = ranked.groupby(["dataset", "shot"], observed=True)["accuracy"].transform(_average_rank)
    summary = (
        ranked.groupby(["model_key", "model_label"], observed=True)
        .agg(
            mean_rank=("rank", "mean"),
            wins=("rank", lambda values: int((values == 1).sum())),
            num_cells=("rank", "size"),
        )
        .reset_index()
        .sort_values(["mean_rank", "model_key"])
        .reset_index(drop=True)
    )
    return summary


def _paper_label(item: ReportItemSpec) -> str:
    return item.paper_label or item.label


def _paper_family(item: ReportItemSpec) -> str:
    return item.family or ""


def resolve_primary_item(items: tuple[ReportItemSpec, ...]) -> ReportItemSpec:
    for item in items:
        if item.primary:
            return item
    return items[0]


def build_paper_overall_summary(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    shots: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        item_df = selected_frame[selected_frame["model_key"] == item.key].copy()
        row: dict[str, object] = {
            "model_key": item.key,
            "model_label": item.label,
            "paper_label": _paper_label(item),
            "family": _paper_family(item),
            "primary": item.primary,
        }
        shot_metrics: list[float] = []
        for shot in shots:
            shot_df = item_df[item_df["shot"].astype(str) == shot]
            row[f"shot_{shot}_accuracy_pct"] = (
                float(shot_df["accuracy"].mean() * 100.0) if not shot_df.empty else pd.NA
            )
            row[f"shot_{shot}_coverage_count"] = int(shot_df["dataset"].astype(str).nunique())
            if not shot_df.empty:
                shot_metrics.append(float(shot_df["accuracy"].mean() * 100.0))
        row["avg_accuracy_pct"] = float(sum(shot_metrics) / len(shot_metrics)) if shot_metrics else pd.NA
        rows.append(row)

    return pd.DataFrame(rows)


def build_paper_rank_table(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    shots: list[str],
) -> pd.DataFrame:
    ranked = selected_frame.copy()
    ranked["rank"] = ranked.groupby(["dataset", "shot"], observed=True)["accuracy"].transform(_average_rank)

    rows: list[dict[str, object]] = []
    for item in items:
        item_df = ranked[ranked["model_key"] == item.key].copy()
        row: dict[str, object] = {
            "model_key": item.key,
            "model_label": item.label,
            "paper_label": _paper_label(item),
            "family": _paper_family(item),
            "primary": item.primary,
        }
        shot_ranks: list[float] = []
        for shot in shots:
            shot_df = item_df[item_df["shot"].astype(str) == shot]
            row[f"shot_{shot}_rank"] = float(shot_df["rank"].mean()) if not shot_df.empty else pd.NA
            row[f"shot_{shot}_coverage_count"] = int(shot_df["dataset"].astype(str).nunique())
            if not shot_df.empty:
                shot_ranks.append(float(shot_df["rank"].mean()))
        row["avg_rank"] = float(sum(shot_ranks) / len(shot_ranks)) if shot_ranks else pd.NA
        rows.append(row)

    return pd.DataFrame(rows)


def build_paper_wtl_table(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    shots: list[str],
    primary_key: str,
    baseline_keys: tuple[str, ...],
) -> pd.DataFrame:
    item_lookup = {item.key: item for item in items}
    primary_item = item_lookup[primary_key]
    ours = selected_frame[selected_frame["model_key"] == primary_key][["dataset", "shot", "accuracy"]].rename(
        columns={"accuracy": "primary_accuracy"}
    )

    rows: list[dict[str, object]] = []
    for baseline_key in baseline_keys:
        baseline_item = item_lookup[baseline_key]
        baseline_df = selected_frame[selected_frame["model_key"] == baseline_key][["dataset", "shot", "accuracy"]].rename(
            columns={"accuracy": "baseline_accuracy"}
        )
        compared = ours.merge(baseline_df, on=["dataset", "shot"], how="inner")
        row: dict[str, object] = {
            "primary_key": primary_item.key,
            "primary_label": _paper_label(primary_item),
            "baseline_key": baseline_item.key,
            "baseline_label": _paper_label(baseline_item),
            "baseline_family": _paper_family(baseline_item),
        }
        total_compared = 0
        total_wins = 0
        total_ties = 0
        total_losses = 0
        for shot in shots:
            shot_df = compared[compared["shot"].astype(str) == shot].copy()
            deltas = shot_df["primary_accuracy"] - shot_df["baseline_accuracy"]
            wins = int((deltas > 1e-12).sum())
            ties = int((deltas.abs() <= 1e-12).sum())
            losses = int((deltas < -1e-12).sum())
            comparisons = int(len(shot_df))
            row[f"shot_{shot}_wins"] = wins
            row[f"shot_{shot}_ties"] = ties
            row[f"shot_{shot}_losses"] = losses
            row[f"shot_{shot}_comparisons"] = comparisons
            row[f"shot_{shot}_wtl"] = f"{wins} / {ties} / {losses}"
            total_compared += comparisons
            total_wins += wins
            total_ties += ties
            total_losses += losses
        row["total_comparisons"] = total_compared
        row["total_wins"] = total_wins
        row["total_ties"] = total_ties
        row["total_losses"] = total_losses
        rows.append(row)

    return pd.DataFrame(rows)


def build_paper_appendix_shot_bundle(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    shot: str,
    datasets: list[str],
) -> PaperAppendixShotBundle:
    model_keys = [item.key for item in items]
    shot_df = selected_frame[selected_frame["shot"].astype(str) == shot].copy()

    accuracy = (
        shot_df.pivot(index="dataset", columns="model_key", values="accuracy")
        .reindex(index=datasets, columns=model_keys)
        .astype(float)
        .mul(100.0)
    )
    std = (
        shot_df.pivot(index="dataset", columns="model_key", values="accuracy_std")
        .reindex(index=datasets, columns=model_keys)
        .apply(pd.to_numeric, errors="coerce")
        .mul(100.0)
    )
    rank = accuracy.rank(axis=1, method="average", ascending=False)
    summary = pd.DataFrame(index=model_keys)
    summary["avg_accuracy_pct"] = accuracy.mean(axis=0)
    summary["avg_rank"] = rank.mean(axis=0)
    best_mask = accuracy.eq(accuracy.max(axis=1), axis=0) & accuracy.notna()
    summary["num_best"] = best_mask.sum(axis=0).astype(int)
    summary.index.name = "model_key"

    return PaperAppendixShotBundle(
        shot=shot,
        accuracy_pct=accuracy,
        std_pct=std,
        rank=rank,
        summary=summary,
    )


def _format_appendix_value(value: object, std_value: object, *, show_std: bool) -> str:
    if pd.isna(value):
        return ""
    rendered = f"{float(value):.2f}"
    if show_std and not pd.isna(std_value):
        return f"{rendered} \u00b1 {float(std_value):.2f}"
    return rendered


def build_paper_appendix_csv(
    *,
    bundle: PaperAppendixShotBundle,
    items: tuple[ReportItemSpec, ...],
    show_std: bool,
) -> pd.DataFrame:
    columns = ["Dataset"] + [_paper_label(item) for item in items]
    rows: list[dict[str, object]] = []

    for dataset in bundle.accuracy_pct.index.tolist():
        row: dict[str, object] = {"Dataset": dataset}
        for item in items:
            row[_paper_label(item)] = _format_appendix_value(
                bundle.accuracy_pct.loc[dataset, item.key],
                bundle.std_pct.loc[dataset, item.key],
                show_std=show_std,
            )
        rows.append(row)

    for summary_label, column_name in (
        ("Avg. Acc.", "avg_accuracy_pct"),
        ("Avg. Rank", "avg_rank"),
        ("#Best", "num_best"),
    ):
        row = {"Dataset": summary_label}
        for item in items:
            value = bundle.summary.loc[item.key, column_name]
            if summary_label == "#Best" and not pd.isna(value):
                row[_paper_label(item)] = str(int(value))
            elif pd.isna(value):
                row[_paper_label(item)] = ""
            else:
                row[_paper_label(item)] = f"{float(value):.2f}"
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_ablation_summary(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    shots: list[str],
    reference_key: str,
) -> pd.DataFrame:
    shot_means = (
        selected_frame.groupby(["model_key", "shot"], observed=True)["accuracy"]
        .mean()
        .mul(100.0)
        .reset_index(name="accuracy_mean_pct")
    )
    overall_means = (
        selected_frame.groupby(["model_key"], observed=True)["accuracy"]
        .mean()
        .mul(100.0)
        .reset_index(name="avg_accuracy_pct")
    )

    reference_frame = selected_frame[selected_frame["model_key"] == reference_key][["dataset", "shot", "accuracy"]].rename(
        columns={"accuracy": "reference_accuracy"}
    )
    compared = selected_frame.merge(reference_frame, on=["dataset", "shot"], how="left")
    compared["delta_vs_reference_pct"] = (compared["accuracy"] - compared["reference_accuracy"]) * 100.0

    shot_lookup = {
        (row.model_key, str(row.shot)): float(row.accuracy_mean_pct) for row in shot_means.itertuples(index=False)
    }
    overall_lookup = {row.model_key: float(row.avg_accuracy_pct) for row in overall_means.itertuples(index=False)}

    rows: list[dict[str, object]] = []
    for item in items:
        row: dict[str, object] = {
            "model_key": item.key,
            "model_label": item.label,
            "primary": item.primary,
            "color": item.color or "",
            "marker": item.marker or "",
            "is_reference": item.key == reference_key,
            "avg_accuracy_pct": overall_lookup[item.key],
            "delta_vs_reference_pct": pd.NA,
            "wins": pd.NA,
            "ties": pd.NA,
            "losses": pd.NA,
            "win_tie_loss": "--",
            "num_cells": int((selected_frame["model_key"] == item.key).sum()),
            "num_datasets": int(
                selected_frame.loc[selected_frame["model_key"] == item.key, "dataset"].astype(str).nunique()
            ),
        }
        for shot in shots:
            row[f"shot_{shot}_accuracy_pct"] = shot_lookup[(item.key, shot)]

        if item.key != reference_key:
            item_compared = compared[compared["model_key"] == item.key].copy()
            deltas = item_compared["delta_vs_reference_pct"]
            wins = int((deltas > 1e-12).sum())
            ties = int((deltas.abs() <= 1e-12).sum())
            losses = int((deltas < -1e-12).sum())
            row["delta_vs_reference_pct"] = float(deltas.mean())
            row["wins"] = wins
            row["ties"] = ties
            row["losses"] = losses
            row["win_tie_loss"] = f"{wins}/{ties}/{losses}"

        rows.append(row)

    return pd.DataFrame(rows)


def build_ablation_cell_deltas(
    *,
    selected_frame: pd.DataFrame,
    items: tuple[ReportItemSpec, ...],
    reference_key: str,
    datasets: list[str],
    shots: list[str],
) -> pd.DataFrame:
    reference_label = next(item.label for item in items if item.key == reference_key)
    pivot = selected_frame.pivot(index=["dataset", "shot"], columns="model_key", values="accuracy")

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for shot in shots:
            row_key = (dataset, shot)
            if row_key not in pivot.index:
                continue
            row: dict[str, object] = {
                "dataset": dataset,
                "shot": shot,
                "reference_key": reference_key,
                "reference_label": reference_label,
                "reference_accuracy_pct": float(pivot.loc[row_key, reference_key]) * 100.0,
            }
            reference_accuracy_pct = row["reference_accuracy_pct"]
            for item in items:
                accuracy_pct = float(pivot.loc[row_key, item.key]) * 100.0
                row[f"{item.key}_accuracy_pct"] = accuracy_pct
                if item.key != reference_key:
                    row[f"{item.key}_delta_vs_reference_pct"] = accuracy_pct - reference_accuracy_pct
            rows.append(row)

    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    cells["shot"] = pd.Categorical(cells["shot"], categories=shots, ordered=True)
    cells["dataset"] = pd.Categorical(cells["dataset"], categories=datasets, ordered=True)
    return cells.sort_values(["dataset", "shot"]).reset_index(drop=True)
