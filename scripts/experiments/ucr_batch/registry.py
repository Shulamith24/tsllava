from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEWSHOT_SHOTS = "1,2,5,10,full"
BLOCKED_FORWARD_ARGS = ["--dataset", "--data_dir", "--data_path", "--save_dir", "--protocol"]


@dataclass(frozen=True)
class ExperimentEntry:
    experiment: str
    protocol: str
    script_path: Path
    summary_kind: str
    add_protocol_flag: bool
    supports_inner_resume: bool
    default_shots: str | None = None
    blocked_forward_args: Tuple[str, ...] = tuple(BLOCKED_FORWARD_ARGS)


def _entry(
    experiment: str,
    protocol: str,
    rel_script_path: str,
    *,
    summary_kind: str,
    add_protocol_flag: bool,
    supports_inner_resume: bool,
    default_shots: str | None = None,
) -> ExperimentEntry:
    return ExperimentEntry(
        experiment=experiment,
        protocol=protocol,
        script_path=REPO_ROOT / rel_script_path,
        summary_kind=summary_kind,
        add_protocol_flag=add_protocol_flag,
        supports_inner_resume=supports_inner_resume,
        default_shots=default_shots,
    )


REGISTRY: Dict[Tuple[str, str], ExperimentEntry] = {
    ("m2_pretrained", "full"): _entry(
        "m2_pretrained",
        "full",
        "scripts/train_ucr_classification_pretrained_full.py",
        summary_kind="full",
        add_protocol_flag=False,
        supports_inner_resume=True,
    ),
    ("m2_pretrained", "fewshot"): _entry(
        "m2_pretrained",
        "fewshot",
        "scripts/train_ucr_classification_pretrained_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
    ),
    ("onefitsall", "full"): _entry(
        "onefitsall",
        "full",
        "scripts/ablations/train_onefitsall_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
    ),
    ("onefitsall", "fewshot"): _entry(
        "onefitsall",
        "fewshot",
        "scripts/ablations/train_onefitsall_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
    ),
    ("patchtst", "full"): _entry(
        "patchtst",
        "full",
        "scripts/ablations/train_ucr_patchtst_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
    ),
    ("patchtst", "fewshot"): _entry(
        "patchtst",
        "fewshot",
        "scripts/ablations/train_ucr_patchtst_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
    ),
}


def get_entry(experiment: str, protocol: str) -> ExperimentEntry:
    key = (experiment, protocol)
    if key not in REGISTRY:
        raise KeyError(f"Unsupported experiment/protocol: {experiment}/{protocol}")
    return REGISTRY[key]


def list_experiments() -> List[str]:
    return sorted({experiment for experiment, _protocol in REGISTRY})
