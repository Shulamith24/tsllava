# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class Prompt(ABC):
    @abstractmethod
    def get_text(self):
        pass
