from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .common import ModelSpec, sort_shots


@dataclass(frozen=True)
class CoverageIssue:
    severity: str
    issue_type: str
    model_key: str
    model_label: str
    dataset: str
    shot: str
    details: str


def _read_ledger_rows(model: ModelSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(model.results_txt, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dataset = str(row.get("dataset", "")).strip()
            shot = str(row.get("shot", "")).strip()
            status = str(row.get("status", "")).strip()
            if not dataset or not shot or shot == "__dataset__" or status != "success":
                continue
            rows.append(
                {
                    "model_key": model.key,
                    "model_label": model.label,
                    "results_txt": str(model.results_txt),
                    "primary": model.primary,
                    "color": model.color or "",
                    "marker": model.marker or "",
                    "dataset": dataset,
                    "shot": shot,
                    "accuracy": float(row.get("accuracy", "")),
                    "accuracy_std": float(row.get("accuracy_std", "")) if str(row.get("accuracy_std", "")).strip() else pd.NA,
                    "num_runs": str(row.get("num_runs", "")).strip(),
                }
            )
    return rows


def load_results_frame(models: tuple[ModelSpec, ...]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for model in models:
        if not model.results_txt.exists():
            raise FileNotFoundError(f"Missing results.txt for model {model.key}: {model.results_txt}")
        records.extend(_read_ledger_rows(model))

    if not records:
        raise ValueError("No successful few-shot records were found in the configured results files")

    frame = pd.DataFrame.from_records(records)
    frame["shot"] = frame["shot"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["num_runs"] = frame["num_runs"].astype(str)
    return frame


def infer_shots(frame: pd.DataFrame) -> list[str]:
    per_model: list[set[str]] = []
    for model_key, model_df in frame.groupby("model_key"):
        numeric_shots = {shot for shot in model_df["shot"].unique().tolist() if shot.isdigit()}
        if not numeric_shots:
            raise ValueError(f"Unable to infer numeric few-shot set for model {model_key}")
        per_model.append(numeric_shots)

    shared = set.intersection(*per_model)
    if not shared:
        raise ValueError("Unable to infer a shared numeric shot set across the configured models")
    return sort_shots(list(shared))


def analyze_coverage(
    *,
    frame: pd.DataFrame,
    models: tuple[ModelSpec, ...],
    expected_datasets: list[str],
    shots: list[str],
    coverage_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[CoverageIssue], list[str], bool]:
    issues: list[CoverageIssue] = []
    duplicate_groups = frame.groupby(["model_key", "dataset", "shot"]).size().reset_index(name="count")
    duplicate_groups = duplicate_groups[duplicate_groups["count"] > 1]
    model_by_key = {model.key: model for model in models}

    for row in duplicate_groups.itertuples(index=False):
        model = model_by_key[row.model_key]
        issues.append(
            CoverageIssue(
                severity="error",
                issue_type="duplicate_success",
                model_key=model.key,
                model_label=model.label,
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
        model = model_by_key[row.model_key]
        issues.append(
            CoverageIssue(
                severity="warning",
                issue_type="unexpected_dataset",
                model_key=model.key,
                model_label=model.label,
                dataset=row.dataset,
                shot="",
                details="dataset is present in results.txt but not in dataset_source",
            )
        )

    deduped = deduped[deduped["dataset"].isin(expected_dataset_set)].copy()

    for model in models:
        model_df = deduped[deduped["model_key"] == model.key]
        num_runs_sets = (
            model_df.groupby("shot")["num_runs"]
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
                        model_key=model.key,
                        model_label=model.label,
                        dataset="",
                        shot=shot,
                        details="num_runs varies across datasets: " + ",".join(values),
                    )
                )

    complete_by_model: dict[str, set[str]] = {}
    for model in models:
        complete_datasets: set[str] = set()
        model_df = deduped[deduped["model_key"] == model.key]
        for dataset, dataset_df in model_df.groupby("dataset"):
            dataset_shots = set(dataset_df["shot"].tolist())
            if all(shot in dataset_shots for shot in shots):
                complete_datasets.add(dataset)
        complete_by_model[model.key] = complete_datasets

    if coverage_mode == "strict":
        selected_datasets = list(expected_datasets)
    else:
        common = set(expected_datasets)
        for datasets in complete_by_model.values():
            common &= datasets
        selected_datasets = [dataset for dataset in expected_datasets if dataset in common]

    fatal_issues = bool(not duplicate_groups.empty)
    if coverage_mode == "strict":
        for model in models:
            available_keys = {
                (row.dataset, row.shot)
                for row in deduped.loc[deduped["model_key"] == model.key, ["dataset", "shot"]].itertuples(index=False)
            }
            for dataset in expected_datasets:
                for shot in shots:
                    if (dataset, shot) not in available_keys:
                        issues.append(
                            CoverageIssue(
                                severity="error",
                                issue_type="missing_result",
                                model_key=model.key,
                                model_label=model.label,
                                dataset=dataset,
                                shot=shot,
                                details="missing success record required by strict coverage",
                            )
                        )
                        fatal_issues = True
    else:
        for model in models:
            for dataset in expected_datasets:
                if dataset not in complete_by_model[model.key]:
                    missing_shots = [shot for shot in shots if ((deduped["model_key"] == model.key) & (deduped["dataset"] == dataset) & (deduped["shot"] == shot)).sum() == 0]
                    if missing_shots:
                        issues.append(
                            CoverageIssue(
                                severity="warning",
                                issue_type="excluded_from_intersection",
                                model_key=model.key,
                                model_label=model.label,
                                dataset=dataset,
                                shot=",".join(missing_shots),
                                details="dataset excluded because not all shots are present for this model",
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
