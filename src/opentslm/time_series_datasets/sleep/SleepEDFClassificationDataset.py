# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, ClassVar, List, Literal, Tuple

import numpy as np
import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.LazyQADataset import LazyQADataset
from opentslm.time_series_datasets.classification_utils import (
    build_label_interface_mapping,
    build_label_token_mapping,
    format_label_card_options,
    format_label_verbalizer_options,
)
from opentslm.time_series_datasets.sleep.sleepedf_classification_loader import (
    SLEEPEDF_DEFAULT_CHANNEL,
    SLEEPEDF_DEFAULT_EPOCH_SECONDS,
    SLEEPEDF_LABEL_ORDER,
    load_sleepedf_classification_splits,
)


class SleepEDFClassificationDataset(LazyQADataset):
    _cache: ClassVar[dict[tuple[str, str, int, str, int], dict[str, Any]]] = {}
    _signal_cache: ClassVar[dict[str, np.ndarray]] = {}
    _dataset_name: ClassVar[str | None] = None
    _label_to_token: ClassVar[dict[str, str] | None] = None
    _token_to_label: ClassVar[dict[str, str] | None] = None
    _label_mapping: ClassVar[dict[str, str] | None] = None
    _label_verbalizers: ClassVar[dict[str, list[str]] | None] = None
    _label_cards: ClassVar[dict[str, dict[str, Any]] | None] = None
    _label_interface: ClassVar[str] = "anonymous"
    _num_classes: ClassVar[int | None] = None
    _class_tokens: ClassVar[List[str] | None] = None
    _ordered_labels: ClassVar[List[str] | None] = None

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        dataset_name: str = "SleepEDFCassette",
        raw_data_path: str = "./data",
        split_protocol: str = "subject",
        split_seed: int = 42,
        channel: str = SLEEPEDF_DEFAULT_CHANNEL,
        epoch_seconds: int = SLEEPEDF_DEFAULT_EPOCH_SECONDS,
        label_interface: str = "anonymous",
        verbalizer_set: str = "canonical",
        verbalizer_mode: str = "canonical",
        semantic_target_mode: str = "class_token",
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        self._instance_dataset_name = dataset_name
        self._instance_raw_data_path = raw_data_path
        self._instance_split_protocol = split_protocol
        self._instance_split_seed = int(split_seed)
        self._instance_channel = channel
        self._instance_epoch_seconds = int(epoch_seconds)
        self._instance_label_interface = label_interface
        self._instance_verbalizer_set = verbalizer_set
        self._instance_verbalizer_mode = verbalizer_mode
        self._instance_semantic_target_mode = semantic_target_mode
        super().__init__(
            split,
            EOS_TOKEN,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
        )

    def _load_splits(self) -> Tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        cache_key = (
            self._instance_raw_data_path,
            self._instance_split_protocol,
            self._instance_split_seed,
            self._instance_channel,
            self._instance_epoch_seconds,
        )
        payload = self.__class__._cache.get(cache_key)
        if payload is None:
            train_rows, val_rows, test_rows = load_sleepedf_classification_splits(
                raw_data_path=self._instance_raw_data_path,
                split_protocol=self._instance_split_protocol,
                seed=self._instance_split_seed,
                channel=self._instance_channel,
                epoch_seconds=self._instance_epoch_seconds,
            )
            ordered_labels, label_to_token, token_to_label, class_tokens = build_label_token_mapping(
                SLEEPEDF_LABEL_ORDER,
                ordered_labels=SLEEPEDF_LABEL_ORDER,
            )
            label_to_index = {label: index for index, label in enumerate(ordered_labels)}
            for row_group in (train_rows, val_rows, test_rows):
                for row in row_group:
                    row["int_label"] = label_to_index[row["label"]]
            payload = {
                "train": train_rows,
                "validation": val_rows,
                "test": test_rows,
                "ordered_labels": ordered_labels,
                "label_to_token": label_to_token,
                "token_to_label": token_to_label,
                "class_tokens": class_tokens,
            }
            self.__class__._cache[cache_key] = payload

        (
            interface,
            label_mapping,
            label_verbalizers,
            class_tokens,
            label_cards,
        ) = build_label_interface_mapping(
            dataset_family="sleepedf",
            ordered_labels=payload["ordered_labels"],
            label_to_token=payload["label_to_token"],
            class_tokens=payload["class_tokens"],
            label_interface=self._instance_label_interface,
            verbalizer_set=self._instance_verbalizer_set,
            verbalizer_mode=self._instance_verbalizer_mode,
            semantic_target_mode=self._instance_semantic_target_mode,
        )

        self._instance_ordered_labels = list(payload["ordered_labels"])
        self._instance_label_to_token = dict(payload["label_to_token"])
        self._instance_label_mapping = dict(label_mapping)
        self._instance_label_verbalizers = dict(label_verbalizers)
        self._instance_label_cards = dict(label_cards)
        self._instance_class_tokens = list(class_tokens)
        self._instance_label_interface = interface

        self.__class__._dataset_name = self._instance_dataset_name
        self.__class__._ordered_labels = list(self._instance_ordered_labels)
        self.__class__._label_to_token = dict(payload["label_to_token"])
        self.__class__._token_to_label = dict(payload["token_to_label"])
        self.__class__._label_mapping = dict(label_mapping)
        self.__class__._label_verbalizers = dict(label_verbalizers)
        self.__class__._label_cards = dict(label_cards)
        self.__class__._label_interface = interface
        self.__class__._class_tokens = list(class_tokens)
        self.__class__._num_classes = len(self._instance_ordered_labels)
        return payload["train"], payload["validation"], payload["test"]

    @classmethod
    def _get_signal_array(cls, signal_path: str) -> np.ndarray:
        cached = cls._signal_cache.get(signal_path)
        if cached is None:
            cached = np.load(signal_path, mmap_mode="r")
            cls._signal_cache[signal_path] = cached
        return cached

    def _slice_epoch(self, row) -> np.ndarray:
        signal = self._get_signal_array(row["signal_path"])
        start = int(row["start_sample"])
        num_samples = int(row["num_samples"])
        stop = start + num_samples
        segment = np.asarray(signal[start:stop], dtype=np.float32)
        if segment.shape[0] == num_samples:
            return segment
        padded = np.zeros((num_samples,), dtype=np.float32)
        padded[: segment.shape[0]] = segment
        return padded

    def _get_pre_prompt(self, row) -> str:
        del row
        num_classes = len(self._instance_ordered_labels)
        if self._instance_label_interface == "semantic":
            if self._instance_semantic_target_mode == "phrase":
                options = format_label_verbalizer_options(
                    self._instance_ordered_labels,
                    self._instance_label_verbalizers,
                )
                output_instruction = "Output exactly one label phrase from the list."
            else:
                options = format_label_card_options(
                    self._instance_ordered_labels,
                    self._instance_label_cards,
                )
                output_instruction = "Output exactly one class token from the list."
            return (
                f"Classify the sleep EEG epoch into one of the following sleep stage labels:\n"
                f"{options}\n\n"
                f"{output_instruction}\n\n"
                "EEG time series:"
            )
        class_list = ", ".join(self._instance_class_tokens)
        return (
            f"Classify the sleep EEG epoch into one of {num_classes} sleep stages.\n"
            f"Choose exactly one label from: {class_list}.\n"
            "Output only the class token.\n\n"
            "EEG time series:"
        )

    def _get_post_prompt(self, row) -> str:
        del row
        if self._instance_label_interface == "semantic" and self._instance_semantic_target_mode == "phrase":
            return "\nLabel:"
        return "\nClass:"

    def _get_answer(self, row) -> str:
        return self._instance_label_mapping[row["label"]]

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        tensor = torch.as_tensor(self._slice_epoch(row), dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        if std > 1e-6:
            tensor = (tensor - mean) / std
        else:
            tensor = tensor - mean
        prompt_text = (
            f"This is a univariate sleep EEG epoch from the {self._instance_channel} channel with {tensor.numel()} points:"
        )
        return [TextTimeSeriesPrompt(prompt_text, tensor.tolist())]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["original_label"] = row["label"]
        sample["class_token"] = self._instance_label_to_token[row["label"]]
        sample["label_answer"] = self._instance_label_mapping[row["label"]]
        sample["label_interface"] = self._instance_label_interface
        sample["semantic_target_mode"] = self._instance_semantic_target_mode
        if self._instance_label_interface == "semantic" and self._instance_semantic_target_mode == "phrase":
            sample["answer_loss_normalization"] = "sample"
        if row["label"] in self._instance_label_verbalizers:
            sample["label_verbalizers"] = self._instance_label_verbalizers[row["label"]]
        if row["label"] in self._instance_label_cards:
            sample["label_card"] = dict(self._instance_label_cards[row["label"]])
        sample["int_label"] = int(row["int_label"])
        sample["record_name"] = row["record_name"]
        sample["subject_id"] = row["subject_id"]
        sample["signal_path"] = row["signal_path"]
        sample["start_sample"] = int(row["start_sample"])
        sample["num_samples"] = int(row["num_samples"])
        sample["sample_rate"] = float(row["sample_rate"])
        sample["channel"] = self._instance_channel
        return sample

    def get_int_labels(self) -> list[int]:
        return [int(row["int_label"]) for row in self.dataset]

    @classmethod
    def get_class_tokens(cls) -> List[str]:
        return list(cls._class_tokens or [])

    @classmethod
    def get_num_classes(cls) -> int:
        return int(cls._num_classes or 0)

    @classmethod
    def get_label_mapping(cls) -> dict[str, str]:
        return dict(cls._label_mapping or cls._label_to_token or {})

    @classmethod
    def get_ordered_labels(cls) -> list[str]:
        return list(cls._ordered_labels or [])

    @classmethod
    def get_label_verbalizers(cls) -> dict[str, list[str]]:
        return dict(cls._label_verbalizers or {})

    @classmethod
    def get_label_cards(cls) -> dict[str, dict[str, Any]]:
        return dict(cls._label_cards or {})
