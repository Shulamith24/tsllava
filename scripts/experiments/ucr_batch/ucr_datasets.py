from __future__ import annotations

from pathlib import Path


def resolve_ucr_archive(data_path: str | Path) -> Path:
    candidate = Path(data_path).resolve()
    if candidate.name == "UCRArchive_2018" and candidate.is_dir():
        return candidate
    archive_dir = candidate / "UCRArchive_2018"
    if archive_dir.is_dir():
        return archive_dir
    raise FileNotFoundError(f"Unable to locate UCRArchive_2018 under {candidate}")


def discover_datasets(ucr_archive_dir: Path) -> list[str]:
    datasets: list[str] = []
    for dataset_dir in sorted(path for path in ucr_archive_dir.iterdir() if path.is_dir()):
        dataset = dataset_dir.name
        if (dataset_dir / f"{dataset}_TRAIN.tsv").exists() and (dataset_dir / f"{dataset}_TEST.tsv").exists():
            datasets.append(dataset)
    return datasets
