# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import random
import re
from copy import deepcopy
from typing import Any, Iterable, Literal, Sequence


CLASS_TOKEN_PATTERN = re.compile(r"<c(\d+)>")
LabelInterface = Literal["anonymous", "semantic"]
SemanticTargetMode = Literal["class_token", "phrase"]


CANONICAL_LABEL_CARDS: dict[str, dict[Any, dict[str, Any]]] = {
    "mitbih": {
        "N": {
            "short_code": "N",
            "canonical_name": "normal beat",
            "description": "A heartbeat segment labeled as a normal beat.",
            "contrastive_description": (
                "A normal beat, not a supraventricular ectopic beat, "
                "ventricular ectopic beat, fusion beat, or unknown beat type."
            ),
        },
        "S": {
            "short_code": "S",
            "canonical_name": "supraventricular ectopic beat",
            "description": "A heartbeat segment labeled as a supraventricular ectopic beat.",
            "contrastive_description": (
                "A supraventricular ectopic beat, not a normal beat, "
                "ventricular ectopic beat, fusion beat, or unknown beat type."
            ),
        },
        "V": {
            "short_code": "V",
            "canonical_name": "ventricular ectopic beat",
            "description": "A heartbeat segment labeled as a ventricular ectopic beat.",
            "contrastive_description": (
                "A ventricular ectopic beat, not a normal beat, "
                "supraventricular ectopic beat, fusion beat, or unknown beat type."
            ),
        },
        "F": {
            "short_code": "F",
            "canonical_name": "fusion beat",
            "description": "A heartbeat segment labeled as a fusion beat.",
            "contrastive_description": (
                "A fusion beat, not a normal beat, supraventricular ectopic beat, "
                "ventricular ectopic beat, or unknown beat type."
            ),
        },
        "Q": {
            "short_code": "Q",
            "canonical_name": "unknown beat type",
            "description": "A heartbeat segment labeled as an unknown beat type.",
            "contrastive_description": (
                "An unknown beat type, not a normal beat, supraventricular ectopic beat, "
                "ventricular ectopic beat, or fusion beat."
            ),
        },
    },
    "sleepedf": {
        "W": {
            "short_code": "W",
            "canonical_name": "wake stage",
            "description": "A sleep epoch labeled as wake stage.",
            "contrastive_description": "A wake-stage sleep epoch, not N1, N2, N3, or REM stage.",
        },
        "N1": {
            "short_code": "N1",
            "canonical_name": "N1 stage",
            "description": "A sleep epoch labeled as N1 stage.",
            "contrastive_description": "An N1 sleep epoch, not wake, N2, N3, or REM stage.",
        },
        "N2": {
            "short_code": "N2",
            "canonical_name": "N2 stage",
            "description": "A sleep epoch labeled as N2 stage.",
            "contrastive_description": "An N2 sleep epoch, not wake, N1, N3, or REM stage.",
        },
        "N3": {
            "short_code": "N3",
            "canonical_name": "N3 stage",
            "description": "A sleep epoch labeled as N3 stage.",
            "contrastive_description": "An N3 sleep epoch, not wake, N1, N2, or REM stage.",
        },
        "REM": {
            "short_code": "REM",
            "canonical_name": "REM stage",
            "description": "A sleep epoch labeled as REM stage.",
            "contrastive_description": "A REM sleep epoch, not wake, N1, N2, or N3 stage.",
        },
    },
    "cinc2017af": {
        "N": {
            "short_code": "N",
            "canonical_name": "normal rhythm",
            "description": "An ECG recording labeled as normal rhythm.",
            "contrastive_description": (
                "A normal rhythm ECG recording, not atrial fibrillation, other rhythm, or noisy signal."
            ),
        },
        "A": {
            "short_code": "AF",
            "canonical_name": "atrial fibrillation",
            "description": "An ECG recording labeled as atrial fibrillation.",
            "contrastive_description": (
                "An atrial fibrillation ECG recording, not normal rhythm, other rhythm, or noisy signal."
            ),
        },
        "O": {
            "short_code": "O",
            "canonical_name": "other rhythm",
            "description": "An ECG recording labeled as other rhythm.",
            "contrastive_description": (
                "An ECG recording with other rhythm, neither normal rhythm, atrial fibrillation, nor noisy signal."
            ),
        },
        "~": {
            "short_code": "~",
            "canonical_name": "noisy signal",
            "description": "An ECG recording labeled as noisy signal.",
            "contrastive_description": (
                "A noisy ECG recording, not normal rhythm, atrial fibrillation, or other rhythm."
            ),
        },
    },
}


CANONICAL_LABEL_VERBALIZERS: dict[str, dict[Any, tuple[str, ...]]] = {
    "mitbih": {
        "N": ("normal beat",),
        "S": ("supraventricular ectopic beat",),
        "V": ("ventricular ectopic beat",),
        "F": ("fusion beat",),
        "Q": ("unknown beat type",),
    },
    "sleepedf": {
        "W": ("wake stage",),
        "N1": ("N1 stage",),
        "N2": ("N2 stage",),
        "N3": ("N3 stage",),
        "REM": ("REM stage",),
    },
    "cinc2017af": {
        "N": ("normal rhythm",),
        "A": ("atrial fibrillation",),
        "O": ("other rhythm",),
        "~": ("noisy signal",),
    },
}

MULTI_LABEL_VERBALIZERS: dict[str, dict[Any, tuple[str, ...]]] = {
    "mitbih": {
        "N": (
            "ECG heartbeat segment: normal beat",
            "heartbeat class: normal beat",
            "ECG beat type: normal beat",
        ),
        "S": (
            "ECG heartbeat segment: supraventricular ectopic beat",
            "heartbeat class: supraventricular ectopic beat",
            "ECG beat type: supraventricular ectopic beat",
        ),
        "V": (
            "ECG heartbeat segment: ventricular ectopic beat",
            "heartbeat class: ventricular ectopic beat",
            "ECG beat type: ventricular ectopic beat",
        ),
        "F": (
            "ECG heartbeat segment: fusion beat",
            "heartbeat class: fusion beat",
            "ECG beat type: fusion beat",
        ),
        "Q": (
            "ECG heartbeat segment: unknown beat type",
            "heartbeat class: unknown beat type",
            "ECG beat type: unknown beat type",
        ),
    },
    "sleepedf": {
        "W": (
            "sleep epoch: wake stage",
            "sleep stage: wake stage",
            "EEG sleep epoch: wake stage",
        ),
        "N1": (
            "sleep epoch: N1 stage",
            "sleep stage: N1 stage",
            "EEG sleep epoch: N1 stage",
        ),
        "N2": (
            "sleep epoch: N2 stage",
            "sleep stage: N2 stage",
            "EEG sleep epoch: N2 stage",
        ),
        "N3": (
            "sleep epoch: N3 stage",
            "sleep stage: N3 stage",
            "EEG sleep epoch: N3 stage",
        ),
        "REM": (
            "sleep epoch: REM stage",
            "sleep stage: REM stage",
            "EEG sleep epoch: REM stage",
        ),
    },
    "cinc2017af": {
        "N": (
            "ECG recording: normal rhythm",
            "cardiac rhythm class: normal rhythm",
            "single-lead ECG label: normal rhythm",
        ),
        "A": (
            "ECG recording: atrial fibrillation",
            "cardiac rhythm class: atrial fibrillation",
            "single-lead ECG label: atrial fibrillation",
        ),
        "O": (
            "ECG recording: other rhythm",
            "cardiac rhythm class: other rhythm",
            "single-lead ECG label: other rhythm",
        ),
        "~": (
            "ECG recording: noisy signal",
            "cardiac rhythm class: noisy signal",
            "single-lead ECG label: noisy signal",
        ),
    },
}


def index_to_class_token(index: int) -> str:
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    return f"<c{index}>"


def normalize_label_interface(label_interface: str | None) -> LabelInterface:
    normalized = (label_interface or "anonymous").strip().lower()
    if normalized not in {"anonymous", "semantic"}:
        raise ValueError(
            f"Unsupported label_interface: {label_interface!r}. "
            "Expected 'anonymous' or 'semantic'."
        )
    return normalized  # type: ignore[return-value]


def get_label_verbalizers(
    dataset_family: str,
    ordered_labels: Sequence[Any],
    *,
    verbalizer_set: str = "canonical",
    verbalizer_mode: str = "canonical",
) -> dict[Any, list[str]]:
    normalized_family = str(dataset_family).strip().lower()
    normalized_set = str(verbalizer_set).strip().lower()
    normalized_mode = str(verbalizer_mode).strip().lower()
    if normalized_set != "canonical":
        raise ValueError(f"Unsupported verbalizer_set: {verbalizer_set!r}")
    verbalizer_bank = {
        "canonical": CANONICAL_LABEL_VERBALIZERS,
        "multi": MULTI_LABEL_VERBALIZERS,
    }.get(normalized_mode)
    if verbalizer_bank is None:
        raise ValueError(f"Unsupported verbalizer_mode: {verbalizer_mode!r}")
    if normalized_family not in verbalizer_bank:
        raise ValueError(
            f"No canonical label verbalizers are defined for dataset_family={dataset_family!r}"
        )

    family_verbalizers = verbalizer_bank[normalized_family]
    missing = [label for label in ordered_labels if label not in family_verbalizers]
    if missing:
        raise ValueError(
            f"Missing canonical verbalizers for {normalized_family} labels: {missing}"
        )
    return {label: list(family_verbalizers[label]) for label in ordered_labels}


def get_label_cards(
    dataset_family: str,
    ordered_labels: Sequence[Any],
    *,
    verbalizer_set: str = "canonical",
    verbalizer_mode: str = "canonical",
    class_tokens: Sequence[str] | None = None,
) -> dict[Any, dict[str, Any]]:
    normalized_family = str(dataset_family).strip().lower()
    if normalized_family not in CANONICAL_LABEL_CARDS:
        raise ValueError(
            f"No label cards are defined for dataset_family={dataset_family!r}"
        )
    label_verbalizers = get_label_verbalizers(
        dataset_family,
        ordered_labels,
        verbalizer_set=verbalizer_set,
        verbalizer_mode=verbalizer_mode,
    )
    family_cards = CANONICAL_LABEL_CARDS[normalized_family]
    missing = [label for label in ordered_labels if label not in family_cards]
    if missing:
        raise ValueError(f"Missing label cards for {normalized_family} labels: {missing}")

    cards: dict[Any, dict[str, Any]] = {}
    for index, label in enumerate(ordered_labels):
        card = deepcopy(family_cards[label])
        card["label"] = label
        card["class_id"] = index
        card["class_token"] = (
            str(class_tokens[index]) if class_tokens is not None and index < len(class_tokens) else index_to_class_token(index)
        )
        card["verbalizers"] = list(label_verbalizers[label])
        cards[label] = card
    return cards


def build_label_interface_mapping(
    *,
    dataset_family: str,
    ordered_labels: Sequence[Any],
    label_to_token: dict[Any, str],
    class_tokens: Sequence[str],
    label_interface: str | None = "anonymous",
    verbalizer_set: str = "canonical",
    verbalizer_mode: str = "canonical",
    semantic_target_mode: str = "class_token",
) -> tuple[LabelInterface, dict[Any, str], dict[Any, list[str]], list[str], dict[Any, dict[str, Any]]]:
    interface = normalize_label_interface(label_interface)
    target_mode = str(semantic_target_mode).strip().lower()
    if target_mode not in {"class_token", "phrase"}:
        raise ValueError(f"Unsupported semantic_target_mode: {semantic_target_mode!r}")

    if interface == "anonymous":
        try:
            label_verbalizers = get_label_verbalizers(
                dataset_family,
                ordered_labels,
                verbalizer_set=verbalizer_set,
                verbalizer_mode=verbalizer_mode,
            )
            label_cards = get_label_cards(
                dataset_family,
                ordered_labels,
                verbalizer_set=verbalizer_set,
                verbalizer_mode=verbalizer_mode,
                class_tokens=class_tokens,
            )
        except ValueError:
            label_verbalizers = {}
            label_cards = {}
        return interface, dict(label_to_token), label_verbalizers, list(class_tokens), label_cards

    label_verbalizers = get_label_verbalizers(
        dataset_family,
        ordered_labels,
        verbalizer_set=verbalizer_set,
        verbalizer_mode=verbalizer_mode,
    )
    label_cards = get_label_cards(
        dataset_family,
        ordered_labels,
        verbalizer_set=verbalizer_set,
        verbalizer_mode=verbalizer_mode,
        class_tokens=class_tokens,
    )
    if target_mode == "phrase":
        label_mapping = {label: label_verbalizers[label][0] for label in ordered_labels}
        return interface, label_mapping, label_verbalizers, [], label_cards
    return interface, dict(label_to_token), label_verbalizers, list(class_tokens), label_cards


def format_label_verbalizer_options(
    ordered_labels: Sequence[Any],
    label_verbalizers: dict[Any, list[str]],
) -> str:
    return "\n".join(label_verbalizers[label][0] for label in ordered_labels)


def format_label_card_options(
    ordered_labels: Sequence[Any],
    label_cards: dict[Any, dict[str, Any]],
) -> str:
    return "\n".join(
        f"{label_cards[label]['class_token']} = {label_cards[label]['canonical_name']}"
        for label in ordered_labels
    )


def build_label_token_mapping(
    labels: Sequence[Any],
    *,
    ordered_labels: Sequence[Any] | None = None,
) -> tuple[list[Any], dict[Any, str], dict[str, Any], list[str]]:
    if ordered_labels is None:
        ordered = sorted(set(labels))
    else:
        ordered = list(ordered_labels)

    if not ordered:
        raise ValueError("ordered labels must not be empty")

    tokens = [index_to_class_token(index) for index in range(len(ordered))]
    label_to_token = {label: tokens[index] for index, label in enumerate(ordered)}
    token_to_label = {token: label for label, token in label_to_token.items()}
    return ordered, label_to_token, token_to_label, tokens


def extract_class_token(text: str) -> str | None:
    match = CLASS_TOKEN_PATTERN.search(text.strip())
    return match.group(0) if match else None


def class_token_to_index(token: str) -> int | None:
    match = CLASS_TOKEN_PATTERN.fullmatch(token.strip())
    if match is None:
        return None
    return int(match.group(1))


def summarize_label_counts(labels: Iterable[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(int(label))
        counts[key] = counts.get(key, 0) + 1
    return counts


def split_rows_stratified(
    rows: Sequence[dict[str, Any]],
    *,
    label_key: str = "label",
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("Cannot split an empty row collection")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Expected non-negative validation/test fractions summing to less than 1")

    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row[label_key], []).append(dict(row))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    for label in sorted(groups, key=lambda value: str(value)):
        label_rows = groups[label]
        rng.shuffle(label_rows)
        n_rows = len(label_rows)
        if n_rows == 1:
            train_count, val_count, test_count = 1, 0, 0
        elif n_rows == 2:
            train_count, val_count, test_count = 1, 0, 1
        else:
            test_count = max(1, int(round(n_rows * test_fraction)))
            val_count = max(1, int(round(n_rows * val_fraction)))
            if val_count + test_count >= n_rows:
                overflow = val_count + test_count - (n_rows - 1)
                val_count = max(0, val_count - overflow)
            train_count = n_rows - val_count - test_count

        train_rows.extend(label_rows[:train_count])
        val_rows.extend(label_rows[train_count : train_count + val_count])
        test_rows.extend(label_rows[train_count + val_count : train_count + val_count + test_count])

    for split_rows in (train_rows, val_rows, test_rows):
        split_rows.sort(key=lambda row: str(row.get("record_name", row.get("id", ""))))

    return train_rows, val_rows, test_rows
