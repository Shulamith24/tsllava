# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, ClassVar, List, Literal, Tuple

import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.LazyQADataset import LazyQADataset
from opentslm.time_series_datasets.classification_utils import build_label_token_mapping
from opentslm.time_series_datasets.heart_sound.heart_sound_loader import (
    HEART_SOUND_LABEL_ORDER,
    HEART_SOUND_TARGET_LENGTH,
    load_heart_sound_splits,
)


class HeartSoundClassificationDataset(LazyQADataset):
    _cache: ClassVar[dict[tuple[str, str, int, int], dict[str, Any]]] = {}
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
        dataset_name: str = "CinC2016HeartSound",
        raw_data_path: str = "./data",
        split_protocol: str = "stratified",
        split_seed: int = 42,
        target_length: int = HEART_SOUND_TARGET_LENGTH,
        format_sample_str: bool = False,
        time_series_format_function=None,
    ):
        self._instance_dataset_name = dataset_name
        self._instance_raw_data_path = raw_data_path
        self._instance_split_protocol = split_protocol
        self._instance_split_seed = int(split_seed)
        self._instance_target_length = int(target_length)
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
            self._instance_target_length,
        )
        payload = self.__class__._cache.get(cache_key)
        if payload is None:
            train_rows, val_rows, test_rows = load_heart_sound_splits(
                raw_data_path=self._instance_raw_data_path,
                split_protocol=self._instance_split_protocol,
                seed=self._instance_split_seed,
                target_length=self._instance_target_length,
            )
            ordered_labels, label_to_token, token_to_label, class_tokens = build_label_token_mapping(
                HEART_SOUND_LABEL_ORDER,
                ordered_labels=HEART_SOUND_LABEL_ORDER,
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

    def _get_pre_prompt(self, row) -> str:
        del row
        num_classes = self.get_num_classes()
        return (
            f"Classify the heart sound recording into one of {num_classes} classes.\n"
            "Output only the class token.\n\n"
            "Phonocardiogram time series:"
        )

    def _get_post_prompt(self, row) -> str:
        del row
        return "\nClass:"

    def _get_answer(self, row) -> str:
        return self.get_label_mapping()[row["label"]]

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        tensor = torch.as_tensor(row["time_series"], dtype=torch.float32)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        mean = tensor.mean()
        std = tensor.std(unbiased=False)
        if std > 1e-6:
            tensor = (tensor - mean) / std
        else:
            tensor = tensor - mean
        prompt_text = f"This is a univariate phonocardiogram recording with {tensor.numel()} points:"
        return [TextTimeSeriesPrompt(prompt_text, tensor.tolist())]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["original_label"] = row["label"]
        sample["class_token"] = self.get_label_mapping()[row["label"]]
        sample["int_label"] = int(row["int_label"])
        sample["record_name"] = row["record_name"]
        sample["source_database"] = row["source_database"]
        sample["sample_rate"] = float(row["sample_rate"])
        sample["source_sample_rate"] = float(row["source_sample_rate"])
        sample["original_length"] = int(row["original_length"])
        sample["wav_path"] = row["wav_path"]
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
