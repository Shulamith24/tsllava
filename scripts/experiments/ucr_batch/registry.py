from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEWSHOT_SHOTS = "1,2,5,10,full"
DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS = "1,2,5,10"
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
    fixed_args: Tuple[str, ...] = tuple()


def _entry(
    experiment: str,
    protocol: str,
    rel_script_path: str,
    *,
    summary_kind: str,
    add_protocol_flag: bool,
    supports_inner_resume: bool,
    default_shots: str | None = None,
    fixed_args: Tuple[str, ...] = tuple(),
) -> ExperimentEntry:
    return ExperimentEntry(
        experiment=experiment,
        protocol=protocol,
        script_path=REPO_ROOT / rel_script_path,
        summary_kind=summary_kind,
        add_protocol_flag=add_protocol_flag,
        supports_inner_resume=supports_inner_resume,
        default_shots=default_shots,
        fixed_args=fixed_args,
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
    ("tslib_autoformer", "full"): _entry(
        "tslib_autoformer",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "autoformer"),
    ),
    ("tslib_autoformer", "fewshot"): _entry(
        "tslib_autoformer",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "autoformer"),
    ),
    ("tslib_crossformer", "full"): _entry(
        "tslib_crossformer",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "crossformer"),
    ),
    ("tslib_crossformer", "fewshot"): _entry(
        "tslib_crossformer",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "crossformer"),
    ),
    ("tslib_dlinear", "full"): _entry(
        "tslib_dlinear",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "dlinear"),
    ),
    ("tslib_dlinear", "fewshot"): _entry(
        "tslib_dlinear",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "dlinear"),
    ),
    ("tslib_fedformer", "full"): _entry(
        "tslib_fedformer",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "fedformer"),
    ),
    ("tslib_fedformer", "fewshot"): _entry(
        "tslib_fedformer",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "fedformer"),
    ),
    ("tslib_informer", "full"): _entry(
        "tslib_informer",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "informer"),
    ),
    ("tslib_informer", "fewshot"): _entry(
        "tslib_informer",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "informer"),
    ),
    ("tslib_timesnet", "full"): _entry(
        "tslib_timesnet",
        "full",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="full",
        add_protocol_flag=True,
        supports_inner_resume=True,
        fixed_args=("--model", "timesnet"),
    ),
    ("tslib_timesnet", "fewshot"): _entry(
        "tslib_timesnet",
        "fewshot",
        "scripts/ablations/train_tslib_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=True,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
        fixed_args=("--model", "timesnet"),
    ),
    ("cosco_resnet", "fewshot"): _entry(
        "cosco_resnet",
        "fewshot",
        "scripts/ablations/train_cosco_resnet_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_FEWSHOT_SHOTS,
    ),
    ("1nn_ed", "fewshot"): _entry(
        "1nn_ed",
        "fewshot",
        "scripts/ablations/train_ucr_distance_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS,
        fixed_args=("--metric", "ed"),
    ),
    ("1nn_dtw", "fewshot"): _entry(
        "1nn_dtw",
        "fewshot",
        "scripts/ablations/train_ucr_distance_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS,
        fixed_args=("--metric", "dtw"),
    ),
    ("resnet", "fewshot"): _entry(
        "resnet",
        "fewshot",
        "scripts/ablations/train_ucr_simple_backbone_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS,
        fixed_args=("--model", "resnet"),
    ),
    ("tapnet", "fewshot"): _entry(
        "tapnet",
        "fewshot",
        "scripts/ablations/train_ucr_simple_backbone_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS,
        fixed_args=("--model", "tapnet"),
    ),
    ("inceptiontime", "fewshot"): _entry(
        "inceptiontime",
        "fewshot",
        "scripts/ablations/train_ucr_simple_backbone_classification_fewshot.py",
        summary_kind="fewshot",
        add_protocol_flag=False,
        supports_inner_resume=True,
        default_shots=DEFAULT_LIGHTWEIGHT_FEWSHOT_SHOTS,
        fixed_args=("--model", "inceptiontime"),
    ),
}


def get_entry(experiment: str, protocol: str) -> ExperimentEntry:
    key = (experiment, protocol)
    if key not in REGISTRY:
        raise KeyError(f"Unsupported experiment/protocol: {experiment}/{protocol}")
    return REGISTRY[key]


def list_experiments() -> List[str]:
    return sorted({experiment for experiment, _protocol in REGISTRY})
