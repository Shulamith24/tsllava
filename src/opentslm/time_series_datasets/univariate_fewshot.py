# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from opentslm.time_series_datasets.mitbih.MITBIHClassificationDataset import (
    MITBIHClassificationDataset,
)
from opentslm.time_series_datasets.sleep.SleepEDFClassificationDataset import (
    SleepEDFClassificationDataset,
)
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import (
    UCRClassificationDataset,
)


DatasetFamily = Literal["ucr", "mitbih", "sleepedf"]


@dataclass
class UnivariateFewShotBundle:
    dataset_family: DatasetFamily
    dataset_name: str
    split_protocol: str
    train_dataset: object
    val_dataset: object
    test_dataset: object
    num_classes: int
    class_tokens: list[str]
    label_mapping: dict


def resolve_univariate_dataset_name(dataset_family: str, dataset_name: str | None) -> str:
    normalized_family = str(dataset_family).strip().lower()
    normalized_name = (dataset_name or "").strip()
    if normalized_family == "ucr":
        return normalized_name or "CricketZ"
    if normalized_family == "mitbih":
        return normalized_name or "MITBIHArrhythmia"
    if normalized_family == "sleepedf":
        return normalized_name or "SleepEDFCassette"
    raise ValueError(f"Unsupported dataset_family: {dataset_family}")


def resolve_univariate_split_protocol(dataset_family: str, split_protocol: str | None) -> str:
    normalized_family = str(dataset_family).strip().lower()
    normalized_protocol = (split_protocol or "").strip().lower()
    if normalized_protocol in {"", "default"}:
        if normalized_family == "ucr":
            return "official"
        if normalized_family == "mitbih":
            return "de_chazal"
        if normalized_family == "sleepedf":
            return "subject"
    if normalized_family == "ucr" and normalized_protocol == "official":
        return normalized_protocol
    if normalized_family == "mitbih" and normalized_protocol == "de_chazal":
        return normalized_protocol
    if normalized_family == "sleepedf" and normalized_protocol == "subject":
        return normalized_protocol
    raise ValueError(
        f"Unsupported split_protocol {split_protocol!r} for dataset_family={dataset_family!r}"
    )


def load_univariate_fewshot_bundle(args, *, eos_token: str) -> UnivariateFewShotBundle:
    dataset_family = str(args.dataset_family).strip().lower()
    dataset_name = resolve_univariate_dataset_name(dataset_family, getattr(args, "dataset", None))
    split_protocol = resolve_univariate_split_protocol(
        dataset_family,
        getattr(args, "split_protocol", None),
    )

    if dataset_family == "ucr":
        train_dataset = UCRClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
        )
        val_dataset = UCRClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
        )
        test_dataset = UCRClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
        )
        num_classes = UCRClassificationDataset.get_num_classes()
        class_tokens = UCRClassificationDataset.get_class_tokens()
        label_mapping = UCRClassificationDataset.get_label_mapping()
    elif dataset_family == "mitbih":
        train_dataset = MITBIHClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
        )
        val_dataset = MITBIHClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
        )
        test_dataset = MITBIHClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
        )
        num_classes = MITBIHClassificationDataset.get_num_classes()
        class_tokens = MITBIHClassificationDataset.get_class_tokens()
        label_mapping = MITBIHClassificationDataset.get_label_mapping()
    elif dataset_family == "sleepedf":
        train_dataset = SleepEDFClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        val_dataset = SleepEDFClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        test_dataset = SleepEDFClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        num_classes = SleepEDFClassificationDataset.get_num_classes()
        class_tokens = SleepEDFClassificationDataset.get_class_tokens()
        label_mapping = SleepEDFClassificationDataset.get_label_mapping()
    else:
        raise ValueError(f"Unsupported dataset_family: {dataset_family}")

    return UnivariateFewShotBundle(
        dataset_family=dataset_family,  # type: ignore[arg-type]
        dataset_name=dataset_name,
        split_protocol=split_protocol,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
        class_tokens=class_tokens,
        label_mapping=label_mapping,
    )
