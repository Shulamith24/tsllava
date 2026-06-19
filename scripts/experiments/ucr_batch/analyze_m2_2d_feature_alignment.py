#!/usr/bin/env python3

"""Compute feature-level diagnostics for TimeMorph 1D-to-2D morphology ablations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
ABLATION_SCRIPT_DIR = REPO_ROOT / "scripts" / "ablations"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ABLATION_SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from m2_ablation_common import DEFAULT_DATA_PATH, parse_csv_list  # noqa: E402
from train_m2_no_llm_linear_classification_fewshot import (  # noqa: E402
    build_label_to_indices,
    extract_encoder_state_from_checkpoint,
    make_collate_fn,
    resolve_encoder_config_from_checkpoint,
)
from ucr_datasets import discover_datasets, resolve_ucr_archive  # noqa: E402
from ucr_fewshot_baseline_utils import resolve_device  # noqa: E402
from opentslm.model.encoder.NewTSDualBranchEncoder import NewTSDualBranchEncoder  # noqa: E402
from opentslm.model.encoder.NewTSVisionEncoder import LEGACY_VISION_2D_MODE, SUPPORTED_VISION_2D_MODES  # noqa: E402
from opentslm.time_series_datasets.univariate_fewshot import load_univariate_fewshot_bundle  # noqa: E402


DEFAULT_METHODS = (
    "legacy_unfold",
    "line_plot",
    "gaf",
    "recurrence_plot",
    "stft_spectrogram",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export mKNN/label-kNN diagnostics for M2 2D morphology methods.")
    parser.add_argument("--local_checkpoint", required=True)
    parser.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--split_protocol", default="default")
    parser.add_argument("--output_csv", default="results/analysis/m2_2d_feature_alignment.csv")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vit_patch_size", type=int, default=16)
    parser.add_argument("--vit_stride", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0, help="Optional deterministic cap per dataset split.")
    return parser.parse_args(argv)


def _make_encoder_args(args: argparse.Namespace, method: str) -> SimpleNamespace:
    return SimpleNamespace(
        vision_2d_mode=method,
        vision_2d_mode_explicit=True,
        vit_patch_size=args.vit_patch_size,
        vit_patch_size_explicit=True,
        vit_stride=args.vit_stride,
        vit_stride_explicit=True,
        freeze_ts_backbone=False,
        freeze_vision_backbone=True,
    )


def build_encoder(args: argparse.Namespace, method: str, device: torch.device) -> NewTSDualBranchEncoder:
    checkpoint = torch.load(args.local_checkpoint, map_location="cpu", weights_only=False)
    encoder_args = _make_encoder_args(args, method)
    encoder_config = resolve_encoder_config_from_checkpoint(checkpoint, encoder_args)
    encoder = NewTSDualBranchEncoder(**encoder_config, device=str(device)).to(device)
    encoder.load_state_dict(extract_encoder_state_from_checkpoint(checkpoint), strict=True)
    encoder.eval()
    return encoder


def _dataset_args(args: argparse.Namespace, dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_family="ucr",
        dataset=dataset,
        split_protocol=args.split_protocol,
        data_path=args.data_path,
        label_interface="anonymous",
        verbalizer_set="canonical",
        verbalizer_mode="multi",
        semantic_target_mode="class_token",
    )


def _resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.datasets:
        return parse_csv_list(args.datasets)
    archive = resolve_ucr_archive(args.data_path)
    return discover_datasets(archive)


def _collate_args() -> SimpleNamespace:
    return SimpleNamespace(
        pad_mode="last",
        enable_augmentation=False,
        aug_jitter_std=0.0,
        aug_scaling_min=1.0,
        aug_scaling_max=1.0,
        aug_time_mask_ratio=0.0,
        aug_time_mask_prob=0.0,
        aug_freq_dropout_ratio=0.0,
        aug_freq_dropout_prob=0.0,
    )


@torch.no_grad()
def extract_features(
    *,
    encoder: NewTSDualBranchEncoder,
    dataset: Any,
    labels: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    indices = list(range(len(dataset)))
    if args.max_samples and len(indices) > args.max_samples:
        indices = indices[: int(args.max_samples)]
    subset = torch.utils.data.Subset(dataset, indices)
    selected_labels = [int(labels[index]) for index in indices]
    global_to_local = {label: label for label in sorted(set(selected_labels))}
    loader = DataLoader(
        subset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        collate_fn=make_collate_fn(_collate_args(), is_train=False, global_to_local=global_to_local),
    )

    ts_features: List[torch.Tensor] = []
    vision_features: List[torch.Tensor] = []
    label_values: List[torch.Tensor] = []
    for inputs, _local_labels, global_labels in loader:
        outputs = encoder(
            inputs.to(device),
            runtime_branch_mode="both",
            return_intermediates=True,
        )
        if outputs.get("pooled_ts") is None or outputs.get("pooled_vision") is None:
            raise RuntimeError("Feature diagnostics require both TS and vision branches.")
        ts_features.append(outputs["pooled_ts"].detach().float().cpu())
        vision_features.append(outputs["pooled_vision"].detach().float().cpu())
        label_values.append(global_labels.detach().long().cpu())

    return {
        "ts": torch.cat(ts_features, dim=0),
        "vision": torch.cat(vision_features, dim=0),
        "labels": torch.cat(label_values, dim=0),
    }


def _knn_indices(features: torch.Tensor, k: int) -> torch.Tensor:
    n = int(features.size(0))
    effective_k = max(1, min(int(k), n - 1))
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    similarity = normalized @ normalized.t()
    similarity.fill_diagonal_(-float("inf"))
    return torch.topk(similarity, k=effective_k, dim=1).indices


def mutual_knn_score(a: torch.Tensor, b: torch.Tensor, k: int) -> tuple[float, float, int]:
    n = int(a.size(0))
    if n <= 1:
        return 0.0, 0.0, 0
    knn_a = _knn_indices(a, k)
    knn_b = _knn_indices(b, k)
    effective_k = int(knn_a.size(1))
    overlaps = []
    for row_a, row_b in zip(knn_a, knn_b):
        overlaps.append(len(set(row_a.tolist()) & set(row_b.tolist())) / effective_k)
    raw = float(sum(overlaps) / max(len(overlaps), 1))
    chance = effective_k / max(n - 1, 1)
    adjusted = (raw - chance) / max(1.0 - chance, 1e-12)
    return raw, float(adjusted), effective_k


def label_knn_score(features: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if int(features.size(0)) <= 1:
        return 0.0
    knn = _knn_indices(features, k)
    neighbor_labels = labels[knn]
    matches = neighbor_labels.eq(labels.unsqueeze(1))
    return float(matches.float().mean().item())


def linear_cka(a: torch.Tensor, b: torch.Tensor) -> float:
    if int(a.size(0)) <= 1:
        return 0.0
    x = a.float() - a.float().mean(dim=0, keepdim=True)
    y = b.float() - b.float().mean(dim=0, keepdim=True)
    numerator = torch.linalg.matrix_norm(x.t() @ y, ord="fro").pow(2)
    denominator = torch.linalg.matrix_norm(x.t() @ x, ord="fro") * torch.linalg.matrix_norm(y.t() @ y, ord="fro")
    if float(denominator.item()) <= 0.0:
        return 0.0
    return float((numerator / denominator).item())


def write_rows(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "method",
        "num_samples",
        "k",
        "mknn",
        "adjusted_mknn",
        "label_knn_vision",
        "cka_ts_vision",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    methods = parse_csv_list(args.methods)
    if not methods:
        raise ValueError("--methods must contain at least one method")
    unsupported = [method for method in methods if method not in SUPPORTED_VISION_2D_MODES]
    if unsupported:
        raise ValueError(f"Unsupported methods: {','.join(unsupported)}")

    device = resolve_device(args.device)
    datasets = _resolve_datasets(args)
    rows: list[dict[str, object]] = []
    for method in methods:
        encoder = build_encoder(args, method, device)
        for dataset_name in datasets:
            bundle = load_univariate_fewshot_bundle(_dataset_args(args, dataset_name), eos_token="<eos>")
            label_to_indices = build_label_to_indices(bundle.test_dataset)
            labels = []
            for label, indices in label_to_indices.items():
                labels.extend((index, int(label)) for index in indices)
            ordered_labels = [label for _index, label in sorted(labels)]
            features = extract_features(
                encoder=encoder,
                dataset=bundle.test_dataset,
                labels=ordered_labels,
                args=args,
                device=device,
            )
            mknn, adjusted_mknn, effective_k = mutual_knn_score(features["ts"], features["vision"], args.k)
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "num_samples": int(features["labels"].numel()),
                    "k": effective_k,
                    "mknn": mknn,
                    "adjusted_mknn": adjusted_mknn,
                    "label_knn_vision": label_knn_score(features["vision"], features["labels"], args.k),
                    "cka_ts_vision": linear_cka(features["ts"], features["vision"]),
                }
            )
            print(f"[{method}] {dataset_name}: mKNN={mknn:.4f}, labelKNN={rows[-1]['label_knn_vision']:.4f}")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_rows(Path(args.output_csv), rows)
    print(f"Wrote feature diagnostics to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
