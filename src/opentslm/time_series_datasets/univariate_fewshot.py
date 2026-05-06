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
from opentslm.time_series_datasets.cinc2017af.CinC2017AFClassificationDataset import (
    CinC2017AFClassificationDataset,
)
from opentslm.time_series_datasets.heart_sound.HeartSoundClassificationDataset import (
    HeartSoundClassificationDataset,
)
from opentslm.time_series_datasets.ucr.UCRClassificationDataset import (
    UCRClassificationDataset,
)


DatasetFamily = Literal[
    "ucr",
    "mitbih",
    "sleepedf",
    "cinc2017af",
    "cinc2016heart",
]


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
    ordered_labels: list
    label_verbalizers: dict[int, list[str]]
    label_to_class_id: dict[str, int]
    label_cards: dict[int, dict]


def resolve_univariate_dataset_name(dataset_family: str, dataset_name: str | None) -> str:
    normalized_family = str(dataset_family).strip().lower()
    normalized_name = (dataset_name or "").strip()
    if normalized_family == "ucr":
        return normalized_name or "CricketZ"
    if normalized_family == "mitbih":
        return normalized_name or "MITBIHArrhythmia"
    if normalized_family == "sleepedf":
        return normalized_name or "SleepEDFCassette"
    if normalized_family == "cinc2017af":
        return normalized_name or "CinC2017AF"
    if normalized_family == "cinc2016heart":
        return normalized_name or "CinC2016HeartSound"
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
        if normalized_family in {"cinc2017af", "cinc2016heart"}:
            return "stratified"
    if normalized_family == "ucr" and normalized_protocol == "official":
        return normalized_protocol
    if normalized_family == "mitbih" and normalized_protocol == "de_chazal":
        return normalized_protocol
    if normalized_family == "sleepedf" and normalized_protocol == "subject":
        return normalized_protocol
    if normalized_family in {"cinc2017af", "cinc2016heart"} and normalized_protocol == "stratified":
        return normalized_protocol
    raise ValueError(
        f"Unsupported split_protocol {split_protocol!r} for dataset_family={dataset_family!r}"
    )


def _build_label_metadata(
    *,
    ordered_labels: list,
    class_tokens: list[str],
    label_mapping: dict,
    raw_label_verbalizers: dict,
    raw_label_cards: dict | None = None,
) -> tuple[dict[int, list[str]], dict[str, int]]:
    label_verbalizers: dict[int, list[str]] = {}
    label_to_class_id: dict[str, int] = {}

    for class_id, label in enumerate(ordered_labels):
        answer = label_mapping.get(label)
        if answer is not None:
            label_to_class_id[str(answer).strip()] = class_id
        if class_id < len(class_tokens):
            label_to_class_id[str(class_tokens[class_id]).strip()] = class_id

        verbalizers = [str(item).strip() for item in raw_label_verbalizers.get(label, [])]
        if verbalizers:
            label_verbalizers[class_id] = verbalizers
            for verbalizer in verbalizers:
                label_to_class_id[verbalizer] = class_id

    return label_verbalizers, label_to_class_id


def _build_label_cards_metadata(
    *,
    ordered_labels: list,
    raw_label_cards: dict | None,
) -> dict[int, dict]:
    if not raw_label_cards:
        return {}
    label_cards: dict[int, dict] = {}
    for class_id, label in enumerate(ordered_labels):
        card = raw_label_cards.get(label)
        if card:
            label_cards[class_id] = dict(card)
    return label_cards


def load_univariate_fewshot_bundle(args, *, eos_token: str) -> UnivariateFewShotBundle:
    dataset_family = str(args.dataset_family).strip().lower()
    dataset_name = resolve_univariate_dataset_name(dataset_family, getattr(args, "dataset", None))
    split_protocol = resolve_univariate_split_protocol(
        dataset_family,
        getattr(args, "split_protocol", None),
    )
    label_interface = getattr(args, "label_interface", "anonymous")
    verbalizer_set = getattr(args, "verbalizer_set", "canonical")
    verbalizer_mode = getattr(args, "verbalizer_mode", "canonical")
    semantic_target_mode = getattr(args, "semantic_target_mode", "class_token")
    if str(label_interface).strip().lower() == "semantic" and dataset_family not in {
        "mitbih",
        "sleepedf",
        "cinc2017af",
    }:
        raise ValueError(
            f"label_interface='semantic' is only supported for MIT-BIH, SleepEDF, and CinC2017AF; "
            f"got dataset_family={dataset_family!r}"
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
        ordered_labels = sorted(label_mapping.keys())
        raw_label_verbalizers = {}
        raw_label_cards = {}
    elif dataset_family == "mitbih":
        train_dataset = MITBIHClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        val_dataset = MITBIHClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        test_dataset = MITBIHClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        num_classes = MITBIHClassificationDataset.get_num_classes()
        class_tokens = MITBIHClassificationDataset.get_class_tokens()
        label_mapping = MITBIHClassificationDataset.get_label_mapping()
        ordered_labels = MITBIHClassificationDataset.get_ordered_labels()
        raw_label_verbalizers = MITBIHClassificationDataset.get_label_verbalizers()
        raw_label_cards = MITBIHClassificationDataset.get_label_cards()
    elif dataset_family == "sleepedf":
        train_dataset = SleepEDFClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        val_dataset = SleepEDFClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        test_dataset = SleepEDFClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        num_classes = SleepEDFClassificationDataset.get_num_classes()
        class_tokens = SleepEDFClassificationDataset.get_class_tokens()
        label_mapping = SleepEDFClassificationDataset.get_label_mapping()
        ordered_labels = SleepEDFClassificationDataset.get_ordered_labels()
        raw_label_verbalizers = SleepEDFClassificationDataset.get_label_verbalizers()
        raw_label_cards = SleepEDFClassificationDataset.get_label_cards()
    elif dataset_family == "cinc2017af":
        train_dataset = CinC2017AFClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        val_dataset = CinC2017AFClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        test_dataset = CinC2017AFClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
            label_interface=label_interface,
            verbalizer_set=verbalizer_set,
            verbalizer_mode=verbalizer_mode,
            semantic_target_mode=semantic_target_mode,
        )
        num_classes = CinC2017AFClassificationDataset.get_num_classes()
        class_tokens = CinC2017AFClassificationDataset.get_class_tokens()
        label_mapping = CinC2017AFClassificationDataset.get_label_mapping()
        ordered_labels = CinC2017AFClassificationDataset.get_ordered_labels()
        raw_label_verbalizers = CinC2017AFClassificationDataset.get_label_verbalizers()
        raw_label_cards = CinC2017AFClassificationDataset.get_label_cards()
    elif dataset_family == "cinc2016heart":
        train_dataset = HeartSoundClassificationDataset(
            split="train",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        val_dataset = HeartSoundClassificationDataset(
            split="validation",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        test_dataset = HeartSoundClassificationDataset(
            split="test",
            EOS_TOKEN=eos_token,
            dataset_name=dataset_name,
            raw_data_path=args.data_path,
            split_protocol=split_protocol,
            split_seed=42,
        )
        num_classes = HeartSoundClassificationDataset.get_num_classes()
        class_tokens = HeartSoundClassificationDataset.get_class_tokens()
        label_mapping = HeartSoundClassificationDataset.get_label_mapping()
        ordered_labels = HeartSoundClassificationDataset.get_ordered_labels()
        raw_label_verbalizers = {}
        raw_label_cards = {}
    else:
        raise ValueError(f"Unsupported dataset_family: {dataset_family}")

    label_verbalizers, label_to_class_id = _build_label_metadata(
        ordered_labels=list(ordered_labels),
        class_tokens=list(class_tokens),
        label_mapping=label_mapping,
        raw_label_verbalizers=raw_label_verbalizers,
    )
    label_cards = _build_label_cards_metadata(
        ordered_labels=list(ordered_labels),
        raw_label_cards=raw_label_cards,
    )

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
        ordered_labels=list(ordered_labels),
        label_verbalizers=label_verbalizers,
        label_to_class_id=label_to_class_id,
        label_cards=label_cards,
    )
