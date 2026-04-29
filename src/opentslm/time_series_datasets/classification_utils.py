# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


CLASS_TOKEN_PATTERN = re.compile(r"<c(\d+)>")


def index_to_class_token(index: int) -> str:
    if index < 0:
        raise ValueError(f"Index must be non-negative, got {index}")
    return f"<c{index}>"


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
