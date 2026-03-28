from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


REPORTS_ROOT = Path(__file__).resolve().parents[4] / "results" / "ucr_batches" / "reports"

DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
DEFAULT_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]

VALID_REPORT_KINDS = {"leaderboard", "ablation"}
VALID_REPORT_STAGES = {"preview", "final"}
VALID_COVERAGE_MODES = {"strict", "intersection"}


@dataclass(frozen=True)
class ReportItemSpec:
    key: str
    label: str
    results_txt: Path
    job_dir: Path | None = None
    batch_config_path: Path | None = None
    primary: bool = False
    color: str | None = None
    marker: str | None = None
    variant_tags: tuple[str, ...] = tuple()


ModelSpec = ReportItemSpec


@dataclass(frozen=True)
class RunRef:
    key: str
    results_txt: Path
    job_dir: Path | None
    batch_config_path: Path | None
    batch_config: Mapping[str, Any] | None
    experiment: str | None
    protocol: str | None
    summary_kind: str | None


@dataclass(frozen=True)
class ReportConfig:
    report_name: str
    config_path: Path
    report_kind: str
    report_stage: str
    dataset_source: Path
    coverage_mode: str
    shots: tuple[str, ...] | None
    items: tuple[ReportItemSpec, ...]
    reference_key: str | None = None
    family_label: str | None = None

    @property
    def models(self) -> tuple[ReportItemSpec, ...]:
        return self.items


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return slug or "report"


def sort_shots(shots: list[str] | tuple[str, ...]) -> list[str]:
    def shot_key(raw: str) -> tuple[int, int | str]:
        try:
            return (0, int(raw))
        except ValueError:
            return (1, raw)

    return sorted(shots, key=shot_key)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)
