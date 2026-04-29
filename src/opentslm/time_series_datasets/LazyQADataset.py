# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, List, Literal, Tuple

import numpy as np
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from torch.utils.data import Dataset


class LazyQADataset(Dataset, ABC):
    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function: Callable[[np.ndarray], str] | None = None,
    ):
        self.EOS_TOKEN = EOS_TOKEN
        self._format_sample_str_enabled = format_sample_str
        self._time_series_format_function = time_series_format_function

        train, val, test = self._load_splits()
        match split:
            case "train":
                self.dataset = train
            case "validation":
                self.dataset = val
            case "test":
                self.dataset = test
            case _:
                raise RuntimeError(
                    "Split must be a literal of 'train', 'test', or 'validation'"
                )

    @abstractmethod
    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        pass

    @abstractmethod
    def _get_answer(self, row) -> str:
        pass

    @abstractmethod
    def _get_pre_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_post_prompt(self, row) -> str:
        pass

    @abstractmethod
    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        pass

    def _format_sample(self, row):
        answer = self._get_answer(row)
        if not answer.endswith(self.EOS_TOKEN):
            answer += self.EOS_TOKEN

        return PromptWithAnswer(
            TextPrompt(self._get_pre_prompt(row).strip()),
            self._get_text_time_series_prompt_list(row),
            TextPrompt(self._get_post_prompt(row).strip()),
            answer.strip(),
        ).to_dict()

    def _format_sample_str(
        self, time_series_format_function: Callable[[np.ndarray], str] | None, row
    ):
        def fallback_timeseries_formatter(time_series: np.ndarray) -> str:
            return np.array2string(
                time_series,
                separator=" ",
                formatter={"all": lambda x: f'"{x:.2f}"'.replace(".", "")},
                threshold=sys.maxsize,
                max_line_width=sys.maxsize,
            ).removeprefix("[").removesuffix("]")

        if not time_series_format_function:
            time_series_format_function = fallback_timeseries_formatter

        prompt_chunks = [self._get_pre_prompt(row).strip()]
        for text_time_series_prompt in self._get_text_time_series_prompt_list(row):
            prompt_chunks.append(text_time_series_prompt.get_text())
            time_series = time_series_format_function(
                text_time_series_prompt.get_time_series()
            )
            prompt_chunks.append(time_series)
        prompt_chunks.append(self._get_post_prompt(row).strip())
        return {"prompt": "\n".join(prompt_chunks), "answer": self._get_answer(row)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if self._format_sample_str_enabled:
            formatter = partial(
                self._format_sample_str,
                self._time_series_format_function,
            )
            return formatter(row)
        return self._format_sample(row)
