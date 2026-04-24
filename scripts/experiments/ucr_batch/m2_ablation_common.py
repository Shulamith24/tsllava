from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .reporting.common import REPORTS_ROOT, slugify
    from .ucr_datasets import discover_datasets, resolve_ucr_archive
except ImportError:
    from reporting.common import REPORTS_ROOT, slugify
    from ucr_datasets import discover_datasets, resolve_ucr_archive


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHOTS = ("1", "2", "5", "10")
DEFAULT_DATA_PATH = REPO_ROOT / "data"
M2_FEWSHOT_ROOT = REPO_ROOT / "results" / "ucr_batches" / "m2_pretrained" / "fewshot"
DEFAULT_REFERENCE_JOB_DIR = M2_FEWSHOT_ROOT / "m2_fewshot_6gpu"
DEFAULT_FEWSHOT_SECOND_JOB_DIR = M2_FEWSHOT_ROOT / "fewshot_second"
DEFAULT_WITHOUT_PRETRAIN_JOB_DIR = M2_FEWSHOT_ROOT / "without_pretrain"
DEFAULT_STAGE012_JOB_DIR = M2_FEWSHOT_ROOT / "stage012"

_FORWARD_FLAGS_WITH_VALUES = {
    "--local_checkpoint",
    "--shots",
    "--num_runs",
    "--runtime_branch_mode",
}
_FORWARD_BOOL_FLAGS = {
    "--disable_constrained_decoding",
    "--resume",
}


@dataclass(frozen=True)
class UCRDatasetMetadata:
    dataset: str
    num_classes: int
    train_size: int
    test_size: int
    series_length: int


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_csv_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_success_rows(results_txt: Path, *, shots: Sequence[str] = DEFAULT_SHOTS) -> list[dict[str, str]]:
    with open(results_txt, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [
            row
            for row in reader
            if row.get("status") == "success" and str(row.get("shot", "")).strip() in set(shots)
        ]


def complete_dataset_set(results_txt: Path, *, shots: Sequence[str] = DEFAULT_SHOTS) -> set[str]:
    shots_set = {str(shot) for shot in shots}
    completed: dict[str, set[str]] = {}
    for row in read_success_rows(results_txt, shots=shots):
        dataset = str(row.get("dataset", "")).strip()
        shot = str(row.get("shot", "")).strip()
        if not dataset or not shot:
            continue
        completed.setdefault(dataset, set()).add(shot)
    return {dataset for dataset, available_shots in completed.items() if available_shots >= shots_set}


def shared_complete_datasets(
    results_files: Sequence[Path],
    *,
    dataset_source: str | Path = DEFAULT_DATA_PATH,
    shots: Sequence[str] = DEFAULT_SHOTS,
) -> list[str]:
    if not results_files:
        return []
    shared = complete_dataset_set(results_files[0], shots=shots)
    for path in results_files[1:]:
        shared &= complete_dataset_set(path, shots=shots)
    archive_dir = resolve_ucr_archive(dataset_source)
    return [dataset for dataset in discover_datasets(archive_dir) if dataset in shared]


def _read_tsv_stats(path: Path) -> tuple[int, int, set[str]]:
    row_count = 0
    series_length = 0
    labels: set[str] = set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            row_count += 1
            labels.add(str(row[0]))
            series_length = max(series_length, max(0, len(row) - 1))
    return row_count, series_length, labels


def load_ucr_dataset_metadata(
    dataset_source: str | Path,
    datasets: Sequence[str],
) -> dict[str, UCRDatasetMetadata]:
    archive_dir = resolve_ucr_archive(dataset_source)
    metadata: dict[str, UCRDatasetMetadata] = {}
    for dataset in datasets:
        dataset_dir = archive_dir / dataset
        train_path = dataset_dir / f"{dataset}_TRAIN.tsv"
        test_path = dataset_dir / f"{dataset}_TEST.tsv"
        train_size, train_length, train_labels = _read_tsv_stats(train_path)
        test_size, test_length, test_labels = _read_tsv_stats(test_path)
        metadata[dataset] = UCRDatasetMetadata(
            dataset=dataset,
            num_classes=len(train_labels | test_labels),
            train_size=train_size,
            test_size=test_size,
            series_length=max(train_length, test_length),
        )
    return metadata


def _rank_bins(values_by_dataset: Mapping[str, int], *, num_bins: int = 3) -> dict[str, int]:
    ordered = sorted(values_by_dataset.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    if total == 0:
        return {}
    bins: dict[str, int] = {}
    for index, (dataset, _value) in enumerate(ordered):
        bins[dataset] = min(num_bins - 1, (index * num_bins) // total)
    return bins


def build_strata(metadata_by_dataset: Mapping[str, UCRDatasetMetadata]) -> dict[str, dict[str, Any]]:
    class_bins = _rank_bins({dataset: meta.num_classes for dataset, meta in metadata_by_dataset.items()})
    train_bins = _rank_bins({dataset: meta.train_size for dataset, meta in metadata_by_dataset.items()})
    length_bins = _rank_bins({dataset: meta.series_length for dataset, meta in metadata_by_dataset.items()})

    strata: dict[str, dict[str, Any]] = {}
    for dataset, meta in metadata_by_dataset.items():
        class_bin = class_bins[dataset]
        train_bin = train_bins[dataset]
        length_bin = length_bins[dataset]
        strata[dataset] = {
            "dataset": dataset,
            "num_classes": meta.num_classes,
            "train_size": meta.train_size,
            "test_size": meta.test_size,
            "series_length": meta.series_length,
            "class_bin": class_bin,
            "train_bin": train_bin,
            "length_bin": length_bin,
            "stratum_key": f"c{class_bin}_t{train_bin}_l{length_bin}",
        }
    return strata


def stratified_sample_datasets(
    metadata_by_dataset: Mapping[str, UCRDatasetMetadata],
    *,
    num_datasets: int,
    sampling_seed: int,
) -> list[str]:
    candidate_datasets = list(metadata_by_dataset.keys())
    if num_datasets <= 0:
        raise ValueError("num_datasets must be positive")
    if num_datasets >= len(candidate_datasets):
        return candidate_datasets

    strata = build_strata(metadata_by_dataset)
    grouped: dict[tuple[int, int, int], list[str]] = {}
    for dataset in candidate_datasets:
        info = strata[dataset]
        key = (info["class_bin"], info["train_bin"], info["length_bin"])
        grouped.setdefault(key, []).append(dataset)

    rng = random.Random(sampling_seed)
    for datasets in grouped.values():
        datasets.sort()
        rng.shuffle(datasets)

    selected: list[str] = []
    while len(selected) < num_datasets:
        stratum_order = sorted(grouped.keys())
        rng.shuffle(stratum_order)
        made_progress = False
        for stratum_key in stratum_order:
            bucket = grouped[stratum_key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            made_progress = True
            if len(selected) >= num_datasets:
                break
        if not made_progress:
            break
    return selected


def subset_id_from_datasets(datasets: Sequence[str]) -> str:
    joined = ",".join(datasets).encode("utf-8")
    return hashlib.sha1(joined).hexdigest()[:10]


def resolve_selected_datasets(
    *,
    candidate_datasets: Sequence[str],
    dataset_source: str | Path,
    num_datasets: int | None,
    sampling_seed: int,
    dataset_list: Sequence[str] | None = None,
) -> tuple[list[str], str, dict[str, dict[str, Any]]]:
    candidate = list(candidate_datasets)
    candidate_set = set(candidate)
    raw_metadata = load_ucr_dataset_metadata(dataset_source, candidate)
    metadata = build_strata(raw_metadata)

    if dataset_list:
        missing = [dataset for dataset in dataset_list if dataset not in candidate_set]
        if missing:
            raise ValueError(
                "dataset_list contains datasets outside the candidate pool: " + ",".join(missing)
            )
        selected = list(dataset_list)
        selection_mode = "explicit_list"
    elif num_datasets is None or num_datasets >= len(candidate):
        selected = candidate
        selection_mode = "full_pool"
    else:
        sampled = stratified_sample_datasets(
            {dataset: raw_metadata[dataset] for dataset in candidate},
            num_datasets=num_datasets,
            sampling_seed=sampling_seed,
        )
        selected = sampled
        selection_mode = "stratified"

    return selected, selection_mode, metadata


def load_subset_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    selected = payload.get("selected_datasets")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"subset_manifest missing a non-empty selected_datasets list: {path}")
    return payload


def build_subset_manifest(
    *,
    candidate_pool_name: str,
    dataset_source: str | Path,
    shots: Sequence[str],
    candidate_datasets: Sequence[str],
    selected_datasets: Sequence[str],
    selection_mode: str,
    sampling_seed: int | None,
    num_datasets_requested: int | None,
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    subset_id = subset_id_from_datasets(selected_datasets)
    return {
        "created_at": utc_timestamp(),
        "subset_id": subset_id,
        "candidate_pool_name": candidate_pool_name,
        "dataset_source": str(resolve_ucr_archive(dataset_source)),
        "shots": list(shots),
        "candidate_dataset_count": len(candidate_datasets),
        "candidate_datasets": list(candidate_datasets),
        "selected_dataset_count": len(selected_datasets),
        "selected_datasets": list(selected_datasets),
        "selection_mode": selection_mode,
        "sampling_seed": sampling_seed,
        "num_datasets_requested": num_datasets_requested,
        "metadata": {dataset: dict(metadata[dataset]) for dataset in candidate_datasets if dataset in metadata},
    }


def load_batch_config(batch_config_path: Path) -> dict[str, Any]:
    payload = read_json(batch_config_path)
    if not isinstance(payload.get("forward_args"), list):
        raise ValueError(f"batch_config forward_args must be a list: {batch_config_path}")
    return payload


def strip_managed_forward_args(forward_args: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(forward_args):
        token = str(forward_args[index])
        base = token.split("=", 1)[0]
        if base in _FORWARD_FLAGS_WITH_VALUES:
            if "=" in token:
                index += 1
            else:
                index += 2
            continue
        if base in _FORWARD_BOOL_FLAGS:
            index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def build_forward_args_from_reference(
    *,
    reference_batch_config_path: Path,
    local_checkpoint: str,
    shots: Sequence[str] = DEFAULT_SHOTS,
    num_runs: int = 3,
    runtime_branch_mode: str | None = None,
    disable_constrained_decoding: bool = False,
) -> list[str]:
    batch_config = load_batch_config(reference_batch_config_path)
    forward_args = strip_managed_forward_args(batch_config.get("forward_args", []))
    forward_args.extend(["--local_checkpoint", str(local_checkpoint)])
    forward_args.extend(["--shots", ",".join(str(shot) for shot in shots)])
    forward_args.extend(["--num_runs", str(num_runs)])
    if runtime_branch_mode is not None:
        forward_args.extend(["--runtime_branch_mode", runtime_branch_mode])
    if disable_constrained_decoding:
        forward_args.append("--disable_constrained_decoding")
    return forward_args


def build_ablation_report_config(
    *,
    report_name: str,
    family_label: str,
    reference_key: str,
    dataset_source: str | Path,
    dataset_allowlist: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    shots: Sequence[str] = DEFAULT_SHOTS,
    appendix_tables_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "report_name": report_name,
        "report_kind": "ablation",
        "report_stage": "final",
        "family_label": family_label,
        "reference_key": reference_key,
        "dataset_source": str(resolve_ucr_archive(dataset_source)),
        "dataset_allowlist": list(dataset_allowlist),
        "shots": [str(shot) for shot in shots],
        "appendix_tables_enabled": appendix_tables_enabled,
        "items": [dict(item) for item in items],
    }


def default_report_dir(report_name: str) -> Path:
    return REPORTS_ROOT / slugify(report_name)


def build_run_request_payload(
    *,
    script_name: str,
    report_name: str,
    subset_manifest: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "created_at": utc_timestamp(),
        "script_name": script_name,
        "report_name": report_name,
        "subset_id": subset_manifest.get("subset_id"),
        "selected_datasets": list(subset_manifest.get("selected_datasets", [])),
    }
    payload.update(extra)
    return payload
