# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TSLIB_ROOT = PROJECT_ROOT / "temp" / "Time-Series-Library"
TSLIB_PACKAGE_NAMES = ("layers", "models", "utils")

MODEL_NAME_MAP = {
    "autoformer": "Autoformer",
    "crossformer": "Crossformer",
    "dlinear": "DLinear",
    "fedformer": "FEDformer",
    "informer": "Informer",
    "timesnet": "TimesNet",
}

MODEL_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Autoformer": {
        "train_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "e_layers": 3,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 1,
        "moving_avg": 25,
    },
    "Crossformer": {
        "train_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "e_layers": 3,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 1,
        "moving_avg": 25,
    },
    "FEDformer": {
        "train_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "e_layers": 3,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 1,
        "moving_avg": 25,
    },
    "DLinear": {
        "train_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "moving_avg": 25,
    },
    "Informer": {
        "train_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "e_layers": 3,
        "d_model": 128,
        "d_ff": 256,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 1,
        "moving_avg": 25,
    },
    "TimesNet": {
        "train_epochs": 30,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "e_layers": 2,
        "d_model": 32,
        "d_ff": 64,
        "n_heads": 8,
        "d_layers": 1,
        "dropout": 0.1,
        "factor": 1,
        "moving_avg": 25,
        "top_k": 3,
        "num_kernels": 6,
    },
}


def normalize_model_name(model_name: str) -> str:
    if not model_name:
        raise ValueError("Model name must be provided.")

    normalized = MODEL_NAME_MAP.get(model_name.strip().lower())
    if normalized is None:
        valid = ", ".join(sorted(MODEL_NAME_MAP))
        raise ValueError(f"Unsupported model: {model_name}. Expected one of: {valid}")
    return normalized


def _load_package(package_name: str, package_dir: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        package_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to bootstrap TSLib package: {package_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def bootstrap_tslib_packages() -> Path:
    if not TSLIB_ROOT.exists():
        raise FileNotFoundError(f"TSLib root not found: {TSLIB_ROOT}")

    root_str = str(TSLIB_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    for package_name in TSLIB_PACKAGE_NAMES:
        package_dir = TSLIB_ROOT / package_name
        for module_name in list(sys.modules.keys()):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                del sys.modules[module_name]
        _load_package(package_name, package_dir)

    return TSLIB_ROOT


@dataclass
class TSLibModelProfile:
    model_name: str
    train_epochs: int
    batch_size: int
    learning_rate: float
    task_name: str = "classification"
    seq_len: int = 1
    label_len: int = 0
    pred_len: int = 0
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    num_class: int = 2
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 3
    d_layers: int = 1
    d_ff: int = 256
    factor: int = 1
    moving_avg: int = 25
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    activation: str = "gelu"
    distil: bool = True
    top_k: int = 3
    num_kernels: int = 6

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_model_profile(
    model_name: str,
    *,
    context_length: int,
    num_classes: int,
    overrides: Optional[Dict[str, Any]] = None,
) -> TSLibModelProfile:
    canonical_name = normalize_model_name(model_name)
    resolved = dict(MODEL_DEFAULTS[canonical_name])
    resolved.update(
        {
            "model_name": canonical_name,
            "seq_len": context_length,
            "num_class": num_classes,
        }
    )

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                resolved[key] = value

    return TSLibModelProfile(**resolved)


def prepare_tslib_classification_batch(
    batch: List[Dict[str, Any]],
    *,
    context_length: int,
    device: str,
    pad_mode: str = "zero",
) -> Dict[str, torch.Tensor]:
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if pad_mode not in {"zero", "last", "repeat"}:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")

    x_enc_list: List[torch.Tensor] = []
    padding_masks: List[torch.Tensor] = []
    labels: List[int] = []

    for sample in batch:
        ts_container = sample["time_series"]
        ts_raw = ts_container[0] if isinstance(ts_container, list) else ts_container
        ts = torch.as_tensor(ts_raw, dtype=torch.float32).flatten()
        if ts.numel() == 0:
            raise ValueError("Encountered empty time series.")

        observed_len = min(ts.numel(), context_length)
        observed = ts[:observed_len]

        if observed_len < context_length:
            pad_len = context_length - observed_len
            if pad_mode == "zero":
                pad = observed.new_zeros(pad_len)
            elif pad_mode == "last":
                pad = observed.new_full((pad_len,), float(observed[-1]))
            else:
                repeat_times = math.ceil(pad_len / max(1, observed.numel()))
                pad = observed.repeat(repeat_times)[:pad_len]
            padded = torch.cat([observed, pad], dim=0)
        else:
            padded = observed

        padding_mask = torch.zeros(context_length, dtype=torch.float32)
        padding_mask[:observed_len] = 1.0

        x_enc_list.append(padded.unsqueeze(-1))
        padding_masks.append(padding_mask)
        labels.append(int(sample["int_label"]))

    return {
        "x_enc": torch.stack(x_enc_list, dim=0).to(device),
        "padding_mask": torch.stack(padding_masks, dim=0).to(device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


class TSLibClassifierAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, model_name: str, profile: TSLibModelProfile):
        super().__init__()
        self.model = model
        self.model_name = normalize_model_name(model_name)
        self.profile = profile
        self.loss_fn = nn.CrossEntropyLoss()

    @classmethod
    def build_model(
        cls,
        *,
        model_name: str,
        num_classes: int,
        context_length: int,
        device: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "TSLibClassifierAdapter":
        bootstrap_tslib_packages()
        canonical_name = normalize_model_name(model_name)
        profile = resolve_model_profile(
            canonical_name,
            context_length=context_length,
            num_classes=num_classes,
            overrides=overrides,
        )
        module = importlib.import_module(f"models.{canonical_name}")
        model_cls = getattr(module, "Model")
        model = model_cls(profile.to_namespace()).float().to(device)
        return cls(model=model, model_name=canonical_name, profile=profile)

    @property
    def config(self) -> TSLibModelProfile:
        return self.profile

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def forward_logits(
        self,
        *,
        x_enc: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(x_enc, padding_mask, None, None)

    def forward_loss(self, batch_inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(
            x_enc=batch_inputs["x_enc"],
            padding_mask=batch_inputs["padding_mask"],
        )
        loss = self.loss_fn(logits, batch_inputs["labels"])
        return loss, logits

    @staticmethod
    def mask_logits_for_selected_classes(
        logits: torch.Tensor,
        selected_class_ids: Optional[Iterable[int]],
    ) -> torch.Tensor:
        if selected_class_ids is None:
            return logits

        selected = list(selected_class_ids)
        if not selected:
            raise ValueError("selected_class_ids must not be empty when provided.")

        masked_logits = logits.clone()
        allow_mask = torch.zeros(masked_logits.shape[-1], dtype=torch.bool, device=masked_logits.device)
        allow_mask[selected] = True
        masked_logits[..., ~allow_mask] = float("-inf")
        return masked_logits

    @torch.no_grad()
    def predict(
        self,
        batch_inputs: Dict[str, torch.Tensor],
        *,
        selected_class_ids: Optional[Iterable[int]] = None,
    ) -> torch.Tensor:
        logits = self.forward_logits(
            x_enc=batch_inputs["x_enc"],
            padding_mask=batch_inputs["padding_mask"],
        )
        masked_logits = self.mask_logits_for_selected_classes(logits, selected_class_ids)
        return torch.argmax(masked_logits, dim=-1)

    def save_checkpoint(
        self,
        *,
        save_path: str,
        optimizer,
        scheduler,
        epoch: int,
        args: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None,
        rank: int = 0,
    ) -> None:
        if rank != 0:
            return

        checkpoint = {
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "args": args,
            "model_name": self.model_name,
            "profile": self.profile.to_dict(),
            "context_length": self.profile.seq_len,
            "num_classes": self.profile.num_class,
        }
        if extra_state:
            checkpoint.update(extra_state)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)

    def load_checkpoint(
        self,
        *,
        checkpoint_path: str,
        device: str,
        optimizer=None,
        scheduler=None,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])

        if optimizer is not None and checkpoint.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        return checkpoint
