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
from opentslm.time_series_datasets.classification_utils import build_label_token_mapping
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
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        self._instance_dataset_name = dataset_name
        self._instance_raw_data_path = raw_data_path
        self._instance_split_protocol = split_protocol
        self._instance_split_seed = int(split_seed)
        self._instance_channel = channel
        self._instance_epoch_seconds = int(epoch_seconds)
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

        self.__class__._dataset_name = self._instance_dataset_name
        self.__class__._ordered_labels = list(payload["ordered_labels"])
        self.__class__._label_to_token = dict(payload["label_to_token"])
        self.__class__._token_to_label = dict(payload["token_to_label"])
        self.__class__._class_tokens = list(payload["class_tokens"])
        self.__class__._num_classes = len(payload["class_tokens"])
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
        num_classes = self.get_num_classes()
        return (
            f"Classify the sleep EEG epoch into one of {num_classes} sleep stages.\n"
            "Output only the class token.\n\n"
            "EEG time series:"
        )

    def _get_post_prompt(self, row) -> str:
        del row
        return "\nClass:"

    def _get_answer(self, row) -> str:
        return self.get_label_mapping()[row["label"]]

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
        sample["class_token"] = self.get_label_mapping()[row["label"]]
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
        return dict(cls._label_to_token or {})

    @classmethod
    def get_ordered_labels(cls) -> list[str]:
        return list(cls._ordered_labels or [])
