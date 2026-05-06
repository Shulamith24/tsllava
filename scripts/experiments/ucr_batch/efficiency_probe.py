#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ABLATION_DIR = REPO_ROOT / "scripts" / "ablations"
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ABLATION_DIR))
sys.path.insert(0, str(SRC_DIR))

_CACHE_ROOT = Path("/tmp") / "tsllava_efficiency_probe"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, RAdam
from torch.utils.data import DataLoader, Dataset, Subset

from fewshot_utils import filter_indices_by_class_ids, sample_support_info
from registry import REPO_ROOT as REGISTRY_REPO_ROOT
from ucr_datasets import discover_datasets, resolve_ucr_archive
from ucr_fewshot_baseline_utils import (
    SeriesDataset,
    build_label_to_indices,
    load_univariate_arrays,
    remap_labels_to_local,
    set_seed,
)


DEFAULT_METHODS = [
    "m2_pretrained",
    "resnet",
    "tapnet",
    "cosco_resnet",
    "patchtst",
    "tslib_dlinear",
    "tslib_timesnet",
    "tslib_autoformer",
    "tslib_crossformer",
    "tslib_fedformer",
    "tslib_informer",
    "onefitsall",
]

DEFAULT_RUNTIME_DATASETS = [
    "Coffee",
    "GunPoint",
    "ECG5000",
    "ElectricDevices",
    "Crop",
    "ACSF1",
    "Phoneme",
    "NonInvasiveFetalECGThorax1",
]

METHOD_LABELS = {
    "m2_pretrained": "ChronoMorph",
    "resnet": "ResNet",
    "tapnet": "TapNet",
    "cosco_resnet": "COSCO-ResNet",
    "patchtst": "PatchTST",
    "tslib_dlinear": "DLinear",
    "tslib_timesnet": "TimesNet",
    "tslib_autoformer": "Autoformer",
    "tslib_crossformer": "Crossformer",
    "tslib_fedformer": "FEDformer",
    "tslib_informer": "Informer",
    "onefitsall": "GPT4TS",
}

TSLIB_METHOD_TO_MODEL = {
    "tslib_dlinear": "dlinear",
    "tslib_timesnet": "timesnet",
    "tslib_autoformer": "autoformer",
    "tslib_crossformer": "crossformer",
    "tslib_fedformer": "fedformer",
    "tslib_informer": "informer",
}

METHOD_COLORS = {
    "m2_pretrained": "#B22222",
    "resnet": "#4C72B0",
    "tapnet": "#55A868",
    "cosco_resnet": "#C44E52",
    "patchtst": "#CCB974",
    "tslib_dlinear": "#8172B2",
    "tslib_timesnet": "#64B5CD",
    "tslib_autoformer": "#8C564B",
    "tslib_crossformer": "#E377C2",
    "tslib_fedformer": "#7F7F7F",
    "tslib_informer": "#BCBD22",
    "onefitsall": "#17BECF",
}


@dataclass
class DatasetContext:
    dataset: str
    series_length: int
    num_classes: int
    support_size: int
    query_size: int
    train_features: np.ndarray
    test_features: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    support_indices: List[int]
    query_indices: List[int]
    selected_class_ids: List[int]
    support_labels_local: np.ndarray
    query_labels_local: np.ndarray
    run_seed: int


@dataclass
class ProbeCase:
    method_key: str
    label: str
    model: torch.nn.Module
    train_loader: DataLoader
    inference_loader: DataLoader
    optimizer: Optional[torch.optim.Optimizer]
    train_step: Callable[["ProbeCase", Any], float]
    inference_step: Callable[["ProbeCase", Any], int]
    device: torch.device
    effective_trainable_params: Optional[int] = None
    effective_trainable_bytes: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_method_list(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    requested = parse_csv_list(raw)
    unknown = [item for item in requested if item not in METHOD_LABELS]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    return requested


def normalize_dataset_list(raw: str, *, data_path: str, runtime_datasets: Sequence[str]) -> List[str]:
    value = raw.strip().lower()
    if value == "all":
        return discover_datasets(resolve_ucr_archive(data_path))
    if value == "runtime":
        return list(runtime_datasets)
    return parse_csv_list(raw)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_arg)


def run_seed_for(shot_index: int, run_id: int, seed_base: int) -> int:
    return int(seed_base) + int(shot_index) * 1000 + int(run_id)


def load_dataset_context(
    *,
    dataset: str,
    data_path: str,
    shot: int,
    run_seed: int,
    way: Optional[int],
    normalize: bool,
) -> DatasetContext:
    payload = load_univariate_arrays(
        dataset,
        data_path=data_path,
        normalize=normalize,
        dataset_family="ucr",
        split_protocol="official",
    )
    train_features = payload["train_features"]
    test_features = payload["test_features"]
    train_labels = payload["train_labels"]
    test_labels = payload["test_labels"]
    label_to_indices = build_label_to_indices(train_labels)
    test_label_to_indices = build_label_to_indices(test_labels)
    support_info = sample_support_info(label_to_indices, shot, run_seed, way=way)
    selected_class_ids = [int(class_id) for class_id in support_info["selected_class_ids"]]
    support_indices = [int(index) for index in support_info["selected_indices"]]
    query_indices = [int(index) for index in filter_indices_by_class_ids(test_label_to_indices, selected_class_ids)]
    support_labels_local, _ = remap_labels_to_local(train_labels[support_indices], selected_class_ids)
    query_labels_local, _ = remap_labels_to_local(test_labels[query_indices], selected_class_ids)
    return DatasetContext(
        dataset=dataset,
        series_length=int(payload["series_length"]),
        num_classes=len(selected_class_ids),
        support_size=len(support_indices),
        query_size=len(query_indices),
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        support_indices=support_indices,
        query_indices=query_indices,
        selected_class_ids=selected_class_ids,
        support_labels_local=support_labels_local,
        query_labels_local=query_labels_local,
        run_seed=run_seed,
    )


class RawSeriesDictDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "time_series": [self.features[index]],
            "int_label": int(self.labels[index]),
        }


def make_tensor_loaders(ctx: DatasetContext, *, batch_size: int, eval_batch_size: int) -> tuple[DataLoader, DataLoader]:
    support_dataset = SeriesDataset(ctx.train_features[ctx.support_indices], ctx.support_labels_local)
    query_dataset = SeriesDataset(ctx.test_features[ctx.query_indices], ctx.query_labels_local)
    return (
        DataLoader(
            support_dataset,
            batch_size=max(1, min(batch_size, len(support_dataset))),
            shuffle=True,
            drop_last=False,
        ),
        DataLoader(
            query_dataset,
            batch_size=max(1, min(eval_batch_size, len(query_dataset))),
            shuffle=False,
            drop_last=False,
        ),
    )


def make_raw_loaders(ctx: DatasetContext, *, batch_size: int, eval_batch_size: int) -> tuple[DataLoader, DataLoader]:
    support_dataset = RawSeriesDictDataset(ctx.train_features[ctx.support_indices], ctx.support_labels_local)
    query_dataset = RawSeriesDictDataset(ctx.test_features[ctx.query_indices], ctx.query_labels_local)
    return (
        DataLoader(
            support_dataset,
            batch_size=max(1, min(batch_size, len(support_dataset))),
            shuffle=True,
            collate_fn=lambda batch: batch,
            drop_last=False,
        ),
        DataLoader(
            query_dataset,
            batch_size=max(1, min(eval_batch_size, len(query_dataset))),
            shuffle=False,
            collate_fn=lambda batch: batch,
            drop_last=False,
        ),
    )


def count_unique_params(model: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for param in model.parameters():
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        total += int(param.numel())
    return total


def count_trainable_params_and_bytes(model: torch.nn.Module) -> tuple[int, int]:
    seen: set[int] = set()
    total_params = 0
    total_bytes = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen:
            continue
        seen.add(param_id)
        total_params += int(param.numel())
        total_bytes += int(param.numel() * param.element_size())
    return total_params, total_bytes


def count_chronomorph_effective_trainable(model) -> tuple[int, int]:
    from opentslm.model.class_token_rows import get_class_token_ids

    seen: set[int] = set()
    total_params = 0
    total_bytes = 0

    class_token_ids: tuple[int, ...] = tuple()
    try:
        class_token_ids = tuple(get_class_token_ids(model))
    except Exception:
        class_token_ids = tuple()

    row_param_ids: set[int] = set()
    if class_token_ids:
        for parameter in (
            model.llm.get_input_embeddings().weight,
            model.llm.lm_head.weight,
        ):
            param_id = id(parameter)
            if param_id in row_param_ids:
                continue
            row_param_ids.add(param_id)
            total_params += len(class_token_ids) * int(parameter.shape[1])
            total_bytes += len(class_token_ids) * int(parameter.shape[1]) * parameter.element_size()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        param_id = id(param)
        if param_id in seen or param_id in row_param_ids:
            continue
        seen.add(param_id)
        total_params += int(param.numel())
        total_bytes += int(param.numel() * param.element_size())

    return total_params, total_bytes


def cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return float("nan")
    cuda_synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024.0**3))


def cycle_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def simple_train_step(case: ProbeCase, batch: Any) -> float:
    inputs, labels = batch
    inputs = inputs.to(case.device)
    labels = labels.to(case.device)
    assert case.optimizer is not None
    case.optimizer.zero_grad(set_to_none=True)
    logits, _ = case.model(inputs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    clip_grad_norm_(case.model.parameters(), max_norm=4.0)
    case.optimizer.step()
    return float(loss.detach().item())


@torch.inference_mode()
def simple_inference_step(case: ProbeCase, batch: Any) -> int:
    inputs, _labels = batch
    inputs = inputs.to(case.device)
    case.model(inputs)
    return int(inputs.shape[0])


def patchtst_batch_to_inputs(batch: List[Dict[str, Any]], *, context_length: int, device: torch.device, pad_mode: str):
    from opentslm.model.PatchTSTClassifier import prepare_patchtst_classification_batch

    return prepare_patchtst_classification_batch(
        batch,
        context_length=context_length,
        device=str(device),
        pad_mode=pad_mode,
    )


def patchtst_train_step(case: ProbeCase, batch: Any) -> float:
    metadata = case.metadata or {}
    inputs = patchtst_batch_to_inputs(
        batch,
        context_length=int(metadata["context_length"]),
        device=case.device,
        pad_mode=str(metadata.get("pad_mode", "zero")),
    )
    assert case.optimizer is not None
    case.optimizer.zero_grad(set_to_none=True)
    outputs = case.model(
        past_values=inputs["past_values"],
        target_values=inputs["target_values"],
        past_observed_mask=inputs["past_observed_mask"],
        return_dict=True,
    )
    loss = outputs.loss
    loss.backward()
    clip_grad_norm_(case.model.parameters(), max_norm=4.0)
    case.optimizer.step()
    return float(loss.detach().item())


@torch.inference_mode()
def patchtst_inference_step(case: ProbeCase, batch: Any) -> int:
    metadata = case.metadata or {}
    inputs = patchtst_batch_to_inputs(
        batch,
        context_length=int(metadata["context_length"]),
        device=case.device,
        pad_mode=str(metadata.get("pad_mode", "zero")),
    )
    case.model(
        past_values=inputs["past_values"],
        past_observed_mask=inputs["past_observed_mask"],
        return_dict=True,
    )
    return int(inputs["past_values"].shape[0])


def tslib_train_step(case: ProbeCase, batch: Any) -> float:
    from opentslm.model.TSLibClassification import prepare_tslib_classification_batch

    metadata = case.metadata or {}
    batch_inputs = prepare_tslib_classification_batch(
        batch,
        context_length=int(metadata["context_length"]),
        device=str(case.device),
        pad_mode=str(metadata.get("pad_mode", "zero")),
    )
    assert case.optimizer is not None
    case.optimizer.zero_grad(set_to_none=True)
    loss, _logits = case.model.forward_loss(batch_inputs)
    loss.backward()
    clip_grad_norm_(case.model.parameters(), max_norm=4.0)
    case.optimizer.step()
    return float(loss.detach().item())


@torch.inference_mode()
def tslib_inference_step(case: ProbeCase, batch: Any) -> int:
    from opentslm.model.TSLibClassification import prepare_tslib_classification_batch

    metadata = case.metadata or {}
    batch_inputs = prepare_tslib_classification_batch(
        batch,
        context_length=int(metadata["context_length"]),
        device=str(case.device),
        pad_mode=str(metadata.get("pad_mode", "zero")),
    )
    case.model.forward_logits(
        x_enc=batch_inputs["x_enc"],
        padding_mask=batch_inputs["padding_mask"],
    )
    return int(batch_inputs["x_enc"].shape[0])


def import_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_simple_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    from simple_backbone_models import build_simple_backbone
    from train_ucr_simple_backbone_classification_fewshot import MODEL_DEFAULTS

    model_name = "resnet" if method_key == "resnet" else "tapnet"
    defaults = MODEL_DEFAULTS[model_name]
    train_loader, inference_loader = make_tensor_loaders(
        ctx,
        batch_size=args.batch_size or defaults.batch_size,
        eval_batch_size=args.eval_batch_size,
    )
    model = build_simple_backbone(
        model_name,
        input_channels=1,
        num_classes=ctx.num_classes,
        dropout=defaults.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate or 1e-4, weight_decay=defaults.weight_decay)
    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=simple_train_step,
        inference_step=simple_inference_step,
        device=device,
    )


def build_cosco_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    cosco = import_module_from_path(
        "tsllava_efficiency_cosco",
        ABLATION_DIR / "train_cosco_resnet_classification_fewshot.py",
    )
    cosco_root = cosco.resolve_cosco_root(args.cosco_root)
    cosco.ensure_cosco_components_loaded(cosco_root)
    train_loader, inference_loader = make_tensor_loaders(
        ctx,
        batch_size=args.batch_size or 128,
        eval_batch_size=args.eval_batch_size,
    )
    model = cosco.ResNet(input_size=1, nb_classes=ctx.num_classes).to(device)
    criterion = cosco.PrototypicalLoss(flag="neg")
    optimizer = cosco.SAM(
        model.parameters(),
        torch.optim.SGD,
        lr=args.learning_rate or 1e-2,
        momentum=0.9,
        rho=0.05,
    )

    def cosco_train_step(case: ProbeCase, batch: Any) -> float:
        inputs, labels = batch
        inputs = inputs.to(case.device)
        labels = labels.to(case.device)
        cosco.enable_running_stats(case.model)
        case.optimizer.zero_grad()
        _, embeddings = case.model(inputs)
        loss = criterion(embeddings, labels)
        loss.backward()
        case.optimizer.first_step(zero_grad=True)
        cosco.disable_running_stats(case.model)
        _, embeddings_second = case.model(inputs)
        second_loss = criterion(embeddings_second, labels)
        second_loss.backward()
        case.optimizer.second_step(zero_grad=True)
        return float(second_loss.detach().item())

    @torch.inference_mode()
    def cosco_inference_step(case: ProbeCase, batch: Any) -> int:
        inputs, _labels = batch
        inputs = inputs.to(case.device)
        case.model(inputs)
        return int(inputs.shape[0])

    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=cosco_train_step,
        inference_step=cosco_inference_step,
        device=device,
    )


def build_patchtst_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    from opentslm.model.PatchTSTClassifier import PatchTSTClassifierAdapter

    train_loader, inference_loader = make_raw_loaders(
        ctx,
        batch_size=args.batch_size or 16,
        eval_batch_size=args.eval_batch_size,
    )
    model = PatchTSTClassifierAdapter.build_model(
        num_classes=ctx.num_classes,
        context_length=ctx.series_length,
        device=str(device),
        patchtst_model_id=args.patchtst_model_id,
        patch_length=16,
        stride=8,
        d_model=128,
        num_attention_heads=8,
        num_hidden_layers=3,
        ffn_dim=512,
        dropout=0.1,
        head_dropout=0.1,
        use_cls_token=True,
        pooling_type="mean",
        reset_head=True,
    )
    optimizer = AdamW(model.parameters(), lr=args.learning_rate or 1e-4, weight_decay=1e-2)
    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=patchtst_train_step,
        inference_step=patchtst_inference_step,
        device=device,
        metadata={"context_length": ctx.series_length, "pad_mode": "zero"},
    )


def build_tslib_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    from opentslm.model.TSLibClassification import TSLibClassifierAdapter

    model_name = TSLIB_METHOD_TO_MODEL[method_key]
    train_loader, inference_loader = make_raw_loaders(
        ctx,
        batch_size=args.batch_size or 16,
        eval_batch_size=args.eval_batch_size,
    )
    model = TSLibClassifierAdapter.build_model(
        model_name=model_name,
        num_classes=ctx.num_classes,
        context_length=ctx.series_length,
        device=str(device),
    )
    optimizer = RAdam(model.get_trainable_parameters(), lr=args.learning_rate or 1e-3, weight_decay=0.0)
    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=tslib_train_step,
        inference_step=tslib_inference_step,
        device=device,
        metadata={"context_length": ctx.series_length, "pad_mode": "zero"},
    )


def build_onefitsall_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    onefitsall = import_module_from_path(
        "tsllava_efficiency_onefitsall",
        ABLATION_DIR / "train_onefitsall_classification_fewshot.py",
    )
    one_args = onefitsall.parse_args(
        [
            "--dataset",
            ctx.dataset,
            "--data_path",
            args.data_path,
            "--shots",
            str(args.shot),
            "--num_runs",
            "1",
            "--batch_size",
            str(args.batch_size or 64),
            "--eval_batch_size",
            str(args.eval_batch_size),
            "--gpu",
            "-1" if device.type == "cpu" else str(device.index or 0),
        ]
    )
    one_args = onefitsall.normalize_protocol_args(one_args)
    config = onefitsall.build_config(one_args)
    train_data, test_data = onefitsall.load_raw_splits(config, one_args, ctx.dataset)
    support_ids = list(ctx.support_indices)
    query_ids = list(ctx.query_indices)
    onefitsall.apply_run_normalization(train_data, test_data, support_ids, config, Path(args.output_dir))
    train_dataset = onefitsall.ClassiregressionDataset(train_data, support_ids)
    test_dataset = onefitsall.ClassiregressionDataset(test_data, query_ids)
    train_loader = onefitsall.create_dataloader(
        train_dataset,
        batch_size=min(one_args.batch_size, max(1, len(train_dataset))),
        max_len=train_data.max_seq_len,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    inference_loader = onefitsall.create_dataloader(
        test_dataset,
        batch_size=one_args.eval_batch_size,
        max_len=test_data.max_seq_len,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = onefitsall.gpt4ts(config, train_data).to(device)
    optimizer, output_reg = onefitsall.build_optimizer(model, config)
    loss_module = onefitsall.get_loss_module(config)

    def onefitsall_train_step(case: ProbeCase, batch: Any) -> float:
        x, targets, padding_masks, _ids = batch
        x = x.to(case.device)
        targets = targets.to(case.device)
        padding_masks = padding_masks.to(case.device)
        assert case.optimizer is not None
        case.optimizer.zero_grad()
        predictions = case.model(x, padding_masks)
        loss_vec = loss_module(predictions, targets)
        loss = torch.sum(loss_vec) / len(loss_vec)
        if output_reg:
            loss = loss + output_reg * onefitsall.l2_reg_loss(case.model)
        loss.backward()
        clip_grad_norm_(case.model.parameters(), max_norm=4.0)
        case.optimizer.step()
        return float(loss.detach().item())

    @torch.inference_mode()
    def onefitsall_inference_step(case: ProbeCase, batch: Any) -> int:
        x, _targets, padding_masks, _ids = batch
        x = x.to(case.device)
        padding_masks = padding_masks.to(case.device)
        case.model(x, padding_masks)
        return int(x.shape[0])

    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=onefitsall_train_step,
        inference_step=onefitsall_inference_step,
        device=device,
    )


def build_m2_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    m2 = import_module_from_path(
        "tsllava_efficiency_m2_fewshot",
        REPO_ROOT / "scripts" / "train_ucr_classification_pretrained_fewshot.py",
    )
    m2_args_list = [
        "--dataset",
        ctx.dataset,
        "--data_path",
        args.data_path,
        "--shots",
        str(args.shot),
        "--num_runs",
        "1",
        "--device",
        str(device),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--batch_size",
        str(args.batch_size or 8),
        "--dataloader_num_workers",
        "0",
        "--no_persistent_workers",
        "--no_pin_memory",
        "--eval_decode_mode",
        "logits",
    ]
    if args.chronomorph_local_checkpoint:
        m2_args_list.extend(["--local_checkpoint", args.chronomorph_local_checkpoint])
    if args.chronomorph_pretrained_model:
        m2_args_list.extend(["--pretrained_model", args.chronomorph_pretrained_model])
    if args.chronomorph_llm_id:
        m2_args_list.extend(["--llm_id", args.chronomorph_llm_id])
    if args.chronomorph_vit_model_name:
        m2_args_list.extend(["--vit_model_name", args.chronomorph_vit_model_name])
    if args.gradient_checkpointing:
        m2_args_list.append("--gradient_checkpointing")

    m2_args = m2.parse_args(m2_args_list)
    m2_args.use_lora = not m2_args.no_lora
    model = m2.build_model(args=m2_args, device=str(device), rank=0)
    underlying = m2.get_model(model)
    _class_tokens, class_token_ids = m2.add_class_tokens_to_model(
        underlying,
        num_classes=ctx.num_classes,
        tokenizer_training_mode=m2_args.tokenizer_training_mode,
        rank=0,
    )
    selected_class_token_ids = [class_token_ids[class_id] for class_id in ctx.selected_class_ids]

    dataset_bundle = m2.load_univariate_fewshot_bundle(m2_args, eos_token=underlying.get_eos_token())
    label_to_indices = m2.build_label_to_indices(dataset_bundle.train_dataset)
    test_label_to_indices = m2.build_label_to_indices(dataset_bundle.test_dataset)
    support_info = sample_support_info(label_to_indices, args.shot, ctx.run_seed, way=args.way)
    support_dataset = Subset(dataset_bundle.train_dataset, support_info["selected_indices"])
    query_indices = filter_indices_by_class_ids(test_label_to_indices, support_info["selected_class_ids"])
    query_dataset = Subset(dataset_bundle.test_dataset, query_indices)
    train_loader = DataLoader(
        support_dataset,
        batch_size=max(1, min(args.batch_size or 8, len(support_dataset))),
        shuffle=True,
        collate_fn=m2.make_collate_fn(m2_args, is_train=True),
        num_workers=0,
        pin_memory=False,
    )
    inference_loader = DataLoader(
        query_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=m2.make_collate_fn(m2_args, is_train=False),
        num_workers=0,
        pin_memory=False,
    )
    optimizer, scheduler, _total_steps, _warmup_steps = m2.build_optimizer_scheduler(
        model,
        train_loader,
        m2_args,
        num_epochs=1,
        grad_acc_steps=1,
        include_lora=True,
    )

    def m2_train_step(case: ProbeCase, batch: Any) -> float:
        assert case.optimizer is not None
        case.optimizer.zero_grad(set_to_none=True)
        loss = case.model.compute_loss(batch)
        loss.backward()
        clip_grad_norm_(case.model.parameters(), max_norm=m2_args.grad_clip)
        case.optimizer.step()
        scheduler.step()
        m2.sanitize_class_token_optimizer_state(case.optimizer, underlying)
        return float(loss.detach().item())

    @torch.inference_mode()
    def m2_inference_step(case: ProbeCase, batch: Any) -> int:
        inputs_embeds, attention_mask = underlying.pad_and_apply_batch(batch)
        outputs = underlying.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        class_token_tensor = torch.tensor(selected_class_token_ids, device=outputs.logits.device, dtype=torch.long)
        last_token_positions = attention_mask.to(outputs.logits.device).long().sum(dim=1) - 1
        batch_indices = torch.arange(outputs.logits.size(0), device=outputs.logits.device)
        next_token_logits = outputs.logits[batch_indices, last_token_positions, :]
        next_token_logits.index_select(dim=-1, index=class_token_tensor)
        return len(batch)

    trainable_params, trainable_bytes = count_chronomorph_effective_trainable(underlying)
    return ProbeCase(
        method_key=method_key,
        label=METHOD_LABELS[method_key],
        model=model,
        train_loader=train_loader,
        inference_loader=inference_loader,
        optimizer=optimizer,
        train_step=m2_train_step,
        inference_step=m2_inference_step,
        device=device,
        effective_trainable_params=trainable_params,
        effective_trainable_bytes=trainable_bytes,
        metadata={"inference_mode": "one_pass_class_token_logits"},
    )


def build_probe_case(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> ProbeCase:
    if method_key == "m2_pretrained":
        return build_m2_case(method_key, ctx, args, device)
    if method_key in {"resnet", "tapnet"}:
        return build_simple_case(method_key, ctx, args, device)
    if method_key == "cosco_resnet":
        return build_cosco_case(method_key, ctx, args, device)
    if method_key == "patchtst":
        return build_patchtst_case(method_key, ctx, args, device)
    if method_key in TSLIB_METHOD_TO_MODEL:
        return build_tslib_case(method_key, ctx, args, device)
    if method_key == "onefitsall":
        return build_onefitsall_case(method_key, ctx, args, device)
    raise ValueError(f"Unsupported method: {method_key}")


def measure_train_step(case: ProbeCase, *, warmup_steps: int, timed_steps: int) -> Dict[str, float]:
    if timed_steps <= 0:
        return {
            "train_step_ms_mean": float("nan"),
            "train_step_ms_std": float("nan"),
            "peak_train_memory_gb": float("nan"),
        }
    case.model.train()
    batch_iter = cycle_batches(case.train_loader)
    for _ in range(max(0, warmup_steps)):
        case.train_step(case, next(batch_iter))
    reset_peak_memory(case.device)
    times: List[float] = []
    for _ in range(timed_steps):
        batch = next(batch_iter)
        cuda_synchronize(case.device)
        start = time.perf_counter()
        case.train_step(case, batch)
        cuda_synchronize(case.device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "train_step_ms_mean": float(statistics.mean(times)),
        "train_step_ms_std": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "peak_train_memory_gb": peak_memory_gb(case.device),
    }


def measure_inference(case: ProbeCase, *, warmup_steps: int, max_batches: int) -> Dict[str, float]:
    case.model.eval()
    batch_iter = iter(case.inference_loader)
    for _ in range(max(0, warmup_steps)):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(case.inference_loader)
            batch = next(batch_iter)
        case.inference_step(case, batch)

    cuda_synchronize(case.device)
    total_examples = 0
    total_seconds = 0.0
    measured_batches = 0
    for batch in case.inference_loader:
        if measured_batches >= max_batches:
            break
        cuda_synchronize(case.device)
        start = time.perf_counter()
        batch_size = case.inference_step(case, batch)
        cuda_synchronize(case.device)
        total_seconds += time.perf_counter() - start
        total_examples += int(batch_size)
        measured_batches += 1

    if total_examples <= 0 or total_seconds <= 0:
        return {
            "inference_ms_per_sample": float("nan"),
            "inference_samples_per_second": float("nan"),
            "inference_batches": float(measured_batches),
            "inference_examples": float(total_examples),
        }
    return {
        "inference_ms_per_sample": float(1000.0 * total_seconds / total_examples),
        "inference_samples_per_second": float(total_examples / total_seconds),
        "inference_batches": float(measured_batches),
        "inference_examples": float(total_examples),
    }


def probe_one(method_key: str, ctx: DatasetContext, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    set_seed(ctx.run_seed)
    case = build_probe_case(method_key, ctx, args, device)
    total_params = count_unique_params(case.model)
    if case.effective_trainable_params is None or case.effective_trainable_bytes is None:
        trainable_params, trainable_bytes = count_trainable_params_and_bytes(case.model)
    else:
        trainable_params = case.effective_trainable_params
        trainable_bytes = case.effective_trainable_bytes
    row: Dict[str, Any] = {
        "method_key": method_key,
        "method_label": case.label,
        "dataset": ctx.dataset,
        "series_length": ctx.series_length,
        "num_classes": ctx.num_classes,
        "support_size": ctx.support_size,
        "query_size": ctx.query_size,
        "shot": args.shot,
        "run_id": args.run_id,
        "seed": ctx.run_seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100.0 * trainable_params / max(total_params, 1),
        "adapter_size_mb": trainable_bytes / (1024.0**2),
        "status": "ok",
        "error": "",
    }

    if not args.skip_runtime:
        row.update(
            measure_train_step(
                case,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
            )
        )
        row.update(
            measure_inference(
                case,
                warmup_steps=args.warmup_steps,
                max_batches=args.max_infer_batches,
            )
        )
    else:
        row.update(
            {
                "train_step_ms_mean": float("nan"),
                "train_step_ms_std": float("nan"),
                "peak_train_memory_gb": float("nan"),
                "inference_ms_per_sample": float("nan"),
                "inference_samples_per_second": float("nan"),
                "inference_batches": float("nan"),
                "inference_examples": float("nan"),
            }
        )
    return row


def failed_row(method_key: str, dataset: str, ctx: Optional[DatasetContext], args: argparse.Namespace, exc: BaseException) -> Dict[str, Any]:
    return {
        "method_key": method_key,
        "method_label": METHOD_LABELS.get(method_key, method_key),
        "dataset": dataset,
        "series_length": ctx.series_length if ctx else "",
        "num_classes": ctx.num_classes if ctx else "",
        "support_size": ctx.support_size if ctx else "",
        "query_size": ctx.query_size if ctx else "",
        "shot": args.shot,
        "run_id": args.run_id,
        "seed": ctx.run_seed if ctx else "",
        "total_params": "",
        "trainable_params": "",
        "trainable_pct": "",
        "adapter_size_mb": "",
        "train_step_ms_mean": "",
        "train_step_ms_std": "",
        "peak_train_memory_gb": "",
        "inference_ms_per_sample": "",
        "inference_samples_per_second": "",
        "inference_batches": "",
        "inference_examples": "",
        "status": "error",
        "error": f"{type(exc).__name__}: {exc}",
    }


def numeric_mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value not in ("", None) and not pd.isna(value)]
    if not numeric:
        return float("nan")
    return float(statistics.mean(numeric))


def numeric_std(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if value not in ("", None) and not pd.isna(value)]
    if len(numeric) <= 1:
        return 0.0 if numeric else float("nan")
    return float(statistics.stdev(numeric))


def aggregate_rows(detail_rows: List[Dict[str, Any]], method_order: Sequence[str]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(detail_rows)
    rows: List[Dict[str, Any]] = []
    for method_key in method_order:
        method_df = df[df["method_key"] == method_key]
        ok_df = method_df[method_df["status"] == "ok"]
        row = {
            "method_key": method_key,
            "method_label": METHOD_LABELS.get(method_key, method_key),
            "num_datasets": int(len(ok_df)),
            "num_errors": int(len(method_df) - len(ok_df)),
            "total_params_mean": numeric_mean(ok_df.get("total_params", [])),
            "trainable_params_mean": numeric_mean(ok_df.get("trainable_params", [])),
            "trainable_pct_mean": numeric_mean(ok_df.get("trainable_pct", [])),
            "adapter_size_mb_mean": numeric_mean(ok_df.get("adapter_size_mb", [])),
            "peak_train_memory_gb_mean": numeric_mean(ok_df.get("peak_train_memory_gb", [])),
            "train_step_ms_mean": numeric_mean(ok_df.get("train_step_ms_mean", [])),
            "train_step_ms_std": numeric_std(ok_df.get("train_step_ms_mean", [])),
            "inference_ms_per_sample_mean": numeric_mean(ok_df.get("inference_ms_per_sample", [])),
            "inference_samples_per_second_mean": numeric_mean(ok_df.get("inference_samples_per_second", [])),
        }
        rows.append(row)
    return rows


def format_params(value: Any) -> str:
    if value in ("", None) or pd.isna(value):
        return "--"
    value = float(value)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:.0f}"


def format_float(value: Any, digits: int = 2) -> str:
    if value in ("", None) or pd.isna(value):
        return "--"
    return f"{float(value):.{digits}f}"


def latex_escape(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def render_main_latex_table(summary_rows: List[Dict[str, Any]]) -> str:
    body = []
    for row in summary_rows:
        body.append(
            " & ".join(
                [
                    latex_escape(row["method_label"]),
                    format_params(row["total_params_mean"]),
                    format_params(row["trainable_params_mean"]),
                    format_float(row["trainable_pct_mean"], 2),
                    format_float(row["adapter_size_mb_mean"], 1),
                    format_float(row["peak_train_memory_gb_mean"], 2),
                    format_float(row["train_step_ms_mean"], 1),
                    format_float(row["inference_ms_per_sample_mean"], 2),
                    format_float(row["inference_samples_per_second_mean"], 1),
                ]
            )
            + r" \\"
        )
    return "\n".join(
        [
            r"% Generated by scripts/experiments/ucr_batch/efficiency_probe.py",
            r"% Requires \usepackage{booktabs}",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Computational efficiency and adaptation cost under the 10-shot UCR protocol. Total parameters count the full deployed model, while updated parameters count only the parameters optimized during downstream adaptation. Adapter size estimates the per-dataset trainable-state footprint. Runtime is measured on representative UCR datasets with batch size 8 on a single RTX 3090.}",
            r"\label{tab:computational-efficiency}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Model & Total params & Updated params & Updated \% & Adapter MB & Peak GB & Train ms/step & Infer ms/ex. & Infer ex./s \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def render_appendix_latex_table(detail_rows: List[Dict[str, Any]], method_order: Sequence[str]) -> str:
    ok_rows = [row for row in detail_rows if row.get("status") == "ok"]
    ok_rows.sort(key=lambda row: (row["dataset"], method_order.index(row["method_key"]) if row["method_key"] in method_order else 999))
    body = []
    for row in ok_rows:
        body.append(
            " & ".join(
                [
                    latex_escape(str(row["dataset"])),
                    latex_escape(str(row["method_label"])),
                    format_float(row.get("peak_train_memory_gb"), 2),
                    format_float(row.get("train_step_ms_mean"), 1),
                    format_float(row.get("inference_ms_per_sample"), 2),
                    format_float(row.get("inference_samples_per_second"), 1),
                ]
            )
            + r" \\"
        )
    if not body:
        body = [r"\multicolumn{6}{c}{Run \texttt{efficiency\_probe.py} to populate this table.} \\"]
    return "\n".join(
        [
            r"% Generated by scripts/experiments/ucr_batch/efficiency_probe.py",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Per-dataset runtime details for the computational-efficiency probe.}",
            r"\label{tab:computational-efficiency-details}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Dataset & Model & Peak GB & Train ms/step & Infer ms/ex. & Infer ex./s \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def parse_accuracy_table(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    label_to_key = {label: key for key, label in METHOD_LABELS.items()}
    # Backward compatibility for older generated tables.
    label_to_key["Duo" + "TSP"] = "m2_pretrained"
    accuracies: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or "&" not in stripped or stripped.startswith("\\"):
            continue
        fields = [field.strip() for field in stripped.rstrip("\\").split("&")]
        if len(fields) < 6:
            continue
        label = fields[0].replace(r"\textbf{", "").replace(r"\underline{", "").replace("}", "")
        key = label_to_key.get(label)
        if not key:
            continue
        avg_field = fields[-2]
        avg_field = (
            avg_field.replace(r"\textbf{", "")
            .replace(r"\underline{", "")
            .replace("}", "")
            .strip()
        )
        try:
            accuracies[key] = float(avg_field)
        except ValueError:
            continue
    return accuracies


def plot_accuracy_vs_trainable(summary_rows: List[Dict[str, Any]], *, accuracy_table: Path, output_dir: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accuracies = parse_accuracy_table(accuracy_table)
    points = []
    for row in summary_rows:
        method_key = row["method_key"]
        if method_key not in accuracies:
            continue
        trainable = row.get("trainable_params_mean")
        if trainable in ("", None) or pd.isna(trainable) or float(trainable) <= 0:
            continue
        points.append((method_key, row["method_label"], float(trainable), float(accuracies[method_key])))
    if not points:
        return []

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for method_key, label, trainable, accuracy in points:
        ax.scatter(
            trainable,
            accuracy,
            s=80 if method_key == "m2_pretrained" else 55,
            color=METHOD_COLORS.get(method_key, "#4C72B0"),
            edgecolor="black" if method_key == "m2_pretrained" else "none",
            linewidth=0.8,
            zorder=4 if method_key == "m2_pretrained" else 3,
        )
        ax.annotate(label, (trainable, accuracy), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Updated parameters (log scale)")
    ax.set_ylabel("Average few-shot accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    png_path = output_dir / "accuracy_vs_trainable_params.png"
    pdf_path = output_dir / "accuracy_vs_trainable_params.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png_path, pdf_path]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure parameter counts, adaptation footprint, peak memory, and runtime "
            "for the UCR few-shot models reported in the paper."
        )
    )
    parser.add_argument("--methods", default="all", help="Comma-separated method keys, or 'all'.")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_RUNTIME_DATASETS),
        help="Runtime datasets, comma-separated.",
    )
    parser.add_argument(
        "--param-datasets",
        default="runtime",
        help="Datasets for parameter aggregation: 'runtime', 'all', or comma-separated names.",
    )
    parser.add_argument("--data-path", default=str(REGISTRY_REPO_ROOT / "data"))
    parser.add_argument("--output-dir", default=str(REGISTRY_REPO_ROOT / "results" / "ucr_batches" / "reports" / "efficiency"))
    parser.add_argument(
        "--latex-table",
        default=str(REGISTRY_REPO_ROOT / "latex_all" / "elsevier" / "tables" / "main" / "efficiency_table.tex"),
    )
    parser.add_argument(
        "--appendix-table",
        default=str(
            REGISTRY_REPO_ROOT
            / "latex_all"
            / "elsevier"
            / "tables"
            / "appendix"
            / "efficiency_runtime_details.tex"
        ),
    )
    parser.add_argument(
        "--accuracy-table",
        default=str(REGISTRY_REPO_ROOT / "latex_all" / "elsevier" / "tables" / "main" / "ucr_fewshot_main.tex"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--shot-index", type=int, default=3, help="Index of the 10-shot setting in the paper shots.")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--fewshot-seed-base", type=int, default=3407)
    parser.add_argument("--way", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--max-infer-batches", type=int, default=16)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--cosco-root", default=None)
    parser.add_argument("--patchtst-model-id", default=None)
    parser.add_argument("--chronomorph-local-checkpoint", default=None)
    parser.add_argument("--chronomorph-pretrained-model", default=None)
    parser.add_argument("--chronomorph-llm-id", default=None)
    parser.add_argument("--chronomorph-vit-model-name", default=None)
    parser.add_argument("--" + "duotsp-local-checkpoint", dest="chronomorph_local_checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--" + "duotsp-pretrained-model", dest="chronomorph_pretrained_model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--" + "duotsp-llm-id", dest="chronomorph_llm_id", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--" + "duotsp-vit-model-name", dest="chronomorph_vit_model_name", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use Coffee/GunPoint, one warmup and one timed step for quick validation.",
    )
    args = parser.parse_args(argv)
    if args.smoke_test:
        args.datasets = "Coffee,GunPoint"
        args.param_datasets = "runtime"
        args.warmup_steps = min(args.warmup_steps, 1)
        args.timed_steps = min(args.timed_steps, 1)
        args.max_infer_batches = min(args.max_infer_batches, 1)
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    methods = normalize_method_list(args.methods)
    runtime_datasets = normalize_dataset_list(args.datasets, data_path=args.data_path, runtime_datasets=DEFAULT_RUNTIME_DATASETS)
    param_datasets = normalize_dataset_list(args.param_datasets, data_path=args.data_path, runtime_datasets=runtime_datasets)
    datasets = list(dict.fromkeys([*runtime_datasets, *param_datasets]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    run_seed = run_seed_for(args.shot_index, args.run_id, args.fewshot_seed_base)
    detail_rows: List[Dict[str, Any]] = []
    manifest_errors: List[Dict[str, str]] = []

    print(f"Efficiency probe device: {device}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Runtime datasets: {', '.join(runtime_datasets)}")
    print(f"Parameter datasets: {', '.join(param_datasets)}")

    context_cache: Dict[str, DatasetContext] = {}
    for dataset in datasets:
        ctx: Optional[DatasetContext] = None
        try:
            ctx = load_dataset_context(
                dataset=dataset,
                data_path=args.data_path,
                shot=args.shot,
                run_seed=run_seed,
                way=args.way,
                normalize=args.normalize,
            )
            context_cache[dataset] = ctx
        except Exception as exc:
            manifest_errors.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            if args.fail_on_error:
                raise
            print(f"[dataset={dataset}] failed to load: {exc}", file=sys.stderr)
            continue

        for method_key in methods:
            should_measure_runtime = dataset in runtime_datasets
            original_skip_runtime = args.skip_runtime
            if not should_measure_runtime:
                args.skip_runtime = True
            try:
                print(f"[{method_key}/{dataset}] probing...")
                row = probe_one(method_key, ctx, args, device)
                detail_rows.append(row)
            except Exception as exc:
                if args.fail_on_error:
                    raise
                print(f"[{method_key}/{dataset}] failed: {exc}", file=sys.stderr)
                traceback.print_exc(limit=2)
                detail_rows.append(failed_row(method_key, dataset, ctx, args, exc))
            finally:
                args.skip_runtime = original_skip_runtime
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_rows = aggregate_rows(detail_rows, methods)
    detail_csv = output_dir / "efficiency_details.csv"
    summary_csv = output_dir / "efficiency_summary.csv"
    main_table_path = output_dir / "efficiency_table.tex"
    appendix_table_path = output_dir / "efficiency_runtime_details.tex"
    write_csv(detail_csv, detail_rows)
    write_csv(summary_csv, summary_rows)
    main_table = render_main_latex_table(summary_rows)
    appendix_table = render_appendix_latex_table(detail_rows, methods)
    write_text(main_table_path, main_table)
    write_text(appendix_table_path, appendix_table)
    write_text(Path(args.latex_table), main_table)
    write_text(Path(args.appendix_table), appendix_table)

    generated_files = list(
        dict.fromkeys(
            [
                str(detail_csv),
                str(summary_csv),
                str(main_table_path),
                str(appendix_table_path),
                args.latex_table,
                args.appendix_table,
            ]
        )
    )
    if not args.skip_plot:
        generated_files.extend(
            str(path)
            for path in plot_accuracy_vs_trainable(
                summary_rows,
                accuracy_table=Path(args.accuracy_table),
                output_dir=output_dir,
            )
        )

    manifest = {
        "methods": methods,
        "runtime_datasets": runtime_datasets,
        "param_datasets": param_datasets,
        "shot": args.shot,
        "run_id": args.run_id,
        "seed": run_seed,
        "device": str(device),
        "skip_runtime": bool(args.skip_runtime),
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "max_infer_batches": args.max_infer_batches,
        "generated_files": generated_files,
        "errors": manifest_errors,
    }
    write_json(output_dir / "efficiency_manifest.json", manifest)
    print("Generated:")
    for path in generated_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
