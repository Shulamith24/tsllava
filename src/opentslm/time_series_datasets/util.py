# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import math
from opentslm.model_config import PATCH_SIZE

import torch.nn.functional as F

import torch

MAX_VALUE = 50_000
MIN_VALUE = -MAX_VALUE


def extend_time_series_to_match_patch_size_and_aggregate(
    batch,
    *,
    patch_size: int = PATCH_SIZE,
    normalize: bool = False,
    normalize_eps: float = 1e-5,
    pad_mode: str = "zero",
    augment: bool = False,
    jitter_std: float = 0.02,
    scaling_range: tuple[float, float] = (0.9, 1.1),
    time_mask_ratio: float = 0.05,
    time_mask_prob: float = 0.3,
    freq_dropout_ratio: float = 0.05,
    freq_dropout_prob: float = 0.2,
    enable_freq_dropout: bool = False,
):
    """
    Pad variable-length series so each sample length is a multiple of *patch_size*.
    Optionally normalize each time series to have zero mean and unit variance.
    """

    if pad_mode not in {"zero", "last", "repeat"}:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    def _apply_augmentations(ts: torch.Tensor) -> torch.Tensor:
        x = ts

        if jitter_std > 0:
            x = x + torch.randn_like(x) * jitter_std

        scaling_min, scaling_max = scaling_range
        if scaling_max > 0 and scaling_min > 0 and scaling_max != scaling_min:
            scale = torch.empty(1, device=x.device).uniform_(scaling_min, scaling_max).item()
            x = x * scale

        if time_mask_prob > 0 and time_mask_ratio > 0 and torch.rand(1).item() < time_mask_prob:
            L = x.numel()
            mask_len = max(1, int(round(L * time_mask_ratio)))
            mask_len = min(mask_len, L)
            start = torch.randint(0, L - mask_len + 1, (1,)).item()
            x = x.clone()
            x[start:start + mask_len] = 0.0

        if (
            enable_freq_dropout
            and freq_dropout_prob > 0
            and freq_dropout_ratio > 0
            and torch.rand(1).item() < freq_dropout_prob
        ):
            L = x.numel()
            x_fft = torch.fft.rfft(x)
            n_bins = x_fft.size(0)
            if n_bins > 1:
                n_drop = max(1, int(round((n_bins - 1) * freq_dropout_ratio)))
                n_drop = min(n_drop, n_bins - 1)
                drop_idx = torch.randperm(n_bins - 1)[:n_drop] + 1  # keep DC component
                x_fft[drop_idx] = 0
                x = torch.fft.irfft(x_fft, n=L)

        return x

    processed_batch = []

    for element in batch:
        out_element = dict(element)

        # 1) pull out the list of (1D) time‑series
        ts_list = element["time_series"]

        # 2) convert each to a torch.Tensor (float)
        ts_tensors = []
        for ts in ts_list:
            ts_tensor = torch.as_tensor(ts, dtype=torch.float32).flatten()
            if ts_tensor.numel() == 0:
                raise ValueError("Encountered empty time series in batch.")
            ts_tensors.append(ts_tensor)

        # 3) normalize each time series if requested
        if normalize:
            normalized_tensors = []
            for ts in ts_tensors:
                mean = ts.mean()
                std = ts.std(unbiased=False)
                if std > normalize_eps:
                    ts_normalized = (ts - mean) / (std + normalize_eps)
                else:
                    ts_normalized = ts - mean
                normalized_tensors.append(ts_normalized)
            ts_tensors = normalized_tensors

        # 3.5) lightweight augmentation (train-only, controlled by caller)
        if augment:
            ts_tensors = [_apply_augmentations(ts) for ts in ts_tensors]

        # 4) find the longest series length
        max_len = max([ts.size(0) for ts in ts_tensors])

        # 5) round up to nearest multiple of patch_size
        padded_len = ((max_len + patch_size - 1) // patch_size) * patch_size

        # 6) pad (or trim) each series to padded_len
        padded = []
        for ts in ts_tensors:
            L = ts.size(0)
            if L < padded_len:
                pad_amt = padded_len - L
                if pad_mode == "zero":
                    ts = F.pad(ts, (0, pad_amt), mode="constant", value=0.0)
                elif pad_mode == "last":
                    last_val = ts[-1]
                    ts = torch.cat(
                        [ts, torch.full((pad_amt,), last_val, dtype=ts.dtype, device=ts.device)],
                        dim=0,
                    )
                else:  # repeat
                    repeat_times = math.ceil(pad_amt / L)
                    extension = ts.repeat(repeat_times)[:pad_amt]
                    ts = torch.cat([ts, extension], dim=0)
            else:
                ts = ts[:padded_len]
            padded.append(ts)

        # 7) stack into a single 2D tensor: (num_series, padded_len)
        out_element["time_series"] = torch.stack(padded, dim=0)
        processed_batch.append(out_element)

    return processed_batch
