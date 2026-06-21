#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


_CACHE_ROOT = Path("/tmp") / "tsllava_timemorph_visual_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


DEFAULT_FEATURES_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
    / "timemorph_visual_analysis"
    / "features"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "latex_all" / "figures"
DEFAULT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
    / "timemorph_visual_analysis"
    / "visualization_manifest.json"
)
DEFAULT_TSNE_CACHE_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
    / "timemorph_visual_analysis"
    / "tsne_cache"
)


COLORS = [
    "#8FB9A8",
    "#F2A7A5",
    "#AFCBFF",
    "#F6D186",
    "#CDB4DB",
    "#9AD0EC",
    "#B8E0D2",
    "#F7C6C7",
    "#D6EADF",
    "#BFD7EA",
]
TSNE_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]
BRANCH_BAR_COLORS = [
    "#4E79A7",
    "#59A14F",
    "#F28E2B",
    "#B07AA1",
    "#BAB0AC",
    "#E15759",
    "#76B7B2",
]
WRONG_MARKER_COLOR = "#6E6E6E"


def resolve_requested_datasets(
    requested: list[str],
    available: list[str],
    *,
    context: str,
    fallback: list[str],
) -> list[str]:
    if not requested:
        return fallback
    kept = [dataset for dataset in requested if dataset in available]
    missing = [dataset for dataset in requested if dataset not in available]
    if missing:
        tqdm.write(
            f"Warning: skipping {len(missing)} {context} dataset(s) without exported features: "
            + ", ".join(missing)
        )
    if not kept:
        raise ValueError(
            f"None of the requested {context} datasets have exported features under --features_dir. "
            f"Available datasets: {', '.join(available)}"
        )
    return kept


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": "#B8B8B8",
            "axes.prop_cycle": matplotlib.cycler(color=COLORS),
            "grid.color": "#E9E4DF",
            "grid.alpha": 0.55,
            "legend.edgecolor": "#D0D0D0",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


FIGURE_ALIASES = {
    "all": ["attention_rollout", "representation_analysis", "tsne_appendix", "branch_alignment_heatmap"],
    "attention": ["attention_rollout"],
    "attention_rollout": ["attention_rollout"],
    "representation": ["representation_analysis"],
    "representation_analysis": ["representation_analysis"],
    "appendix": ["tsne_appendix"],
    "tsne_appendix": ["tsne_appendix"],
    "alignment": ["branch_alignment_heatmap"],
    "branch_alignment_heatmap": ["branch_alignment_heatmap"],
}


def parse_figures(value: str | None) -> list[str]:
    requested = parse_csv(value or "all")
    figures: list[str] = []
    for item in requested:
        key = item.lower().replace("-", "_")
        if key not in FIGURE_ALIASES:
            valid = ", ".join(sorted(FIGURE_ALIASES))
            raise ValueError(f"Unknown figure '{item}'. Valid values: {valid}")
        for figure in FIGURE_ALIASES[key]:
            if figure not in figures:
                figures.append(figure)
    return figures


def resolve_required_dataset(name: str, available: list[str], *, context: str) -> str:
    if name in available:
        return name
    raise ValueError(
        f"Requested {context} dataset '{name}' does not have exported features under --features_dir. "
        f"Available datasets: {', '.join(available)}"
    )


def merge_with_existing_manifest(manifest_path: Path, artifacts: dict[str, Any]) -> dict[str, Any]:
    if not manifest_path.exists():
        return artifacts
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        tqdm.write(f"Warning: could not read existing manifest, rewriting it: {type(exc).__name__}: {exc}")
        return artifacts
    if not isinstance(existing, dict):
        return artifacts

    merged = dict(existing)
    for key, value in artifacts.items():
        if key == "dataset_metadata" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            dataset_metadata = dict(merged[key])
            dataset_metadata.update(value)
            merged[key] = dataset_metadata
        else:
            merged[key] = value
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TimeMorph qualitative and representation analysis figures.")
    parser.add_argument("--features_dir", type=str, default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument(
        "--figures",
        type=str,
        default="all",
        help=(
            "Comma-separated figures to generate. Valid values: all, attention_rollout, "
            "representation_analysis, tsne_appendix, branch_alignment_heatmap. "
            "Short aliases are also accepted: attention, representation, appendix, alignment."
        ),
    )
    parser.add_argument("--main_dataset", type=str, default="ECG200")
    parser.add_argument("--attention_datasets", type=str, default="ECG200,TwoLeadECG")
    parser.add_argument("--appendix_datasets", type=str, default="")
    parser.add_argument("--attention_samples_per_dataset", type=int, default=2)
    parser.add_argument("--attention_max_samples", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=3407)
    parser.add_argument("--tsne_max_iter", type=int, default=1000)
    parser.add_argument("--tsne_cache_dir", type=str, default=str(DEFAULT_TSNE_CACHE_DIR))
    parser.add_argument("--refresh_tsne_cache", action="store_true")
    parser.add_argument("--disable_tsne_cache", action="store_true")
    return parser.parse_args(argv)


def load_feature_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Feature artifact not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        payload: dict[str, Any] = {key: data[key] for key in data.files if key != "metadata_json"}
        if "metadata_json" in data.files:
            payload["metadata"] = json.loads(str(data["metadata_json"].item()))
    return payload


def load_dataset_metadata(features_dir: Path, dataset: str) -> dict[str, Any]:
    path = features_dir / dataset / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def infer_datasets(features_dir: Path) -> list[str]:
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features directory does not exist: {features_dir}. "
            "Run export_timemorph_visual_features.py before plotting."
        )
    return sorted(path.name for path in features_dir.iterdir() if path.is_dir() and (path / "metadata.json").exists())


def save_pdf_png(fig, output_dir: Path, base_name: str, *, dpi: int = 320) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"pdf": str(pdf_path.resolve()), "png": str(png_path.resolve())}


def normalize_series(series: np.ndarray) -> np.ndarray:
    series = np.asarray(series, dtype=np.float32)
    std = float(series.std())
    if std <= 1e-6:
        return series - float(series.mean())
    return (series - float(series.mean())) / std


def class_color_map(labels: np.ndarray) -> dict[int, str]:
    unique = sorted(set(int(label) for label in labels.tolist()))
    return {label: TSNE_COLORS[index % len(TSNE_COLORS)] for index, label in enumerate(unique)}


def compute_tsne(features: np.ndarray, *, random_state: int, max_iter: int) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] < 3 or features.shape[1] == 0:
        raise ValueError(f"t-SNE requires a non-empty 2D feature matrix with at least 3 samples, got {features.shape}")
    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("t-SNE feature matrix contains NaN or Inf")
    scaled = StandardScaler().fit_transform(features)
    n_samples = scaled.shape[0]
    perplexity = min(30, max(5, (n_samples - 1) // 3))
    perplexity = min(perplexity, n_samples - 1)
    return TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
        perplexity=perplexity,
        max_iter=max(250, int(max_iter)),
    ).fit_transform(scaled)


def tsne_feature_signature(features: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    digest = hashlib.sha1(array.view(np.uint8)).hexdigest()
    return array, {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha1": digest,
    }


def tsne_cache_path(cache_dir: Path, dataset: str, feature_key: str, *, random_state: int, max_iter: int) -> Path:
    safe_dataset = dataset.replace("/", "_")
    safe_feature = feature_key.replace("/", "_")
    return cache_dir / safe_dataset / f"{safe_feature}_rs{int(random_state)}_iter{max(250, int(max_iter))}.npz"


def load_cached_tsne(path: Path, expected_metadata: dict[str, Any]) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            if metadata != expected_metadata:
                return None
            return data["embedding"].astype(np.float32)
    except Exception as exc:
        tqdm.write(f"Warning: ignoring unreadable t-SNE cache {path}: {type(exc).__name__}: {exc}")
        return None


def compute_or_load_tsne(
    features: np.ndarray,
    *,
    dataset: str,
    feature_key: str,
    cache_dir: Path | None,
    random_state: int,
    max_iter: int,
    refresh_cache: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    features, signature = tsne_feature_signature(features)
    effective_max_iter = max(250, int(max_iter))
    metadata = {
        "cache_version": 1,
        "dataset": dataset,
        "feature_key": feature_key,
        "random_state": int(random_state),
        "max_iter": effective_max_iter,
        "feature_signature": signature,
    }
    if cache_dir is None:
        embedding = compute_tsne(features, random_state=random_state, max_iter=effective_max_iter)
        return embedding, {"status": "disabled"}

    path = tsne_cache_path(cache_dir, dataset, feature_key, random_state=random_state, max_iter=effective_max_iter)
    if not refresh_cache:
        cached = load_cached_tsne(path, metadata)
        if cached is not None:
            return cached, {"status": "hit", "path": str(path.resolve())}

    embedding = compute_tsne(features, random_state=random_state, max_iter=effective_max_iter).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embedding=embedding,
        metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    return embedding, {"status": "refreshed" if refresh_cache else "miss", "path": str(path.resolve())}


def validate_tsne_labels(labels: np.ndarray, *, context: str) -> None:
    unique, counts = np.unique(labels.astype(np.int64), return_counts=True)
    if unique.size < 2:
        raise ValueError(f"{context} t-SNE requires at least two classes, got labels {unique.tolist()}")
    if np.any(counts <= 1):
        sparse = {int(label): int(count) for label, count in zip(unique, counts) if count <= 1}
        raise ValueError(f"{context} t-SNE requires more than one sample per class, sparse classes: {sparse}")


def scatter_tsne(ax, embedding: np.ndarray, labels: np.ndarray, correct: np.ndarray, *, title: str) -> None:
    cmap = class_color_map(labels)
    for label in sorted(cmap):
        mask = labels == label
        correct_mask = mask & correct.astype(bool)
        wrong_mask = mask & ~correct.astype(bool)
        if correct_mask.any():
            ax.scatter(
                embedding[correct_mask, 0],
                embedding[correct_mask, 1],
                s=17,
                c=cmap[label],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.25,
            )
        if wrong_mask.any():
            ax.scatter(
                embedding[wrong_mask, 0],
                embedding[wrong_mask, 1],
                s=29,
                c=cmap[label],
                alpha=0.92,
                marker="x",
                linewidths=0.8,
            )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


def branch_complement_categories(
    both: dict[str, Any],
    ts_only: dict[str, Any],
    vision_only: dict[str, Any],
) -> tuple[list[str], list[int]]:
    both_correct = both["correct"].astype(bool)
    ts_correct = ts_only["correct"].astype(bool)
    vision_correct = vision_only["correct"].astype(bool)
    if not (both_correct.shape == ts_correct.shape == vision_correct.shape):
        raise ValueError("Branch correctness arrays must have identical shapes.")
    for key in ("sample_indices", "labels"):
        if key in both and key in ts_only and key in vision_only:
            if not (np.array_equal(both[key], ts_only[key]) and np.array_equal(both[key], vision_only[key])):
                raise ValueError(f"Branch artifacts must be aligned by {key}.")

    categories = [
        "Fusion recovers\nsingle-view errors",
        "All three\ncorrect",
        "Temporal helps\nfusion",
        "Morphology helps\nfusion",
        "All wrong",
        "Fusion loses\nsingle-view hit",
        "Other",
    ]
    fusion_recovers = both_correct & ~ts_correct & ~vision_correct
    all_three_correct = both_correct & ts_correct & vision_correct
    temporal_helps = both_correct & ts_correct & ~vision_correct
    morphology_helps = both_correct & ~ts_correct & vision_correct
    all_wrong = ~both_correct & ~ts_correct & ~vision_correct
    fusion_loses = ~both_correct & (ts_correct | vision_correct)
    assigned = (
        fusion_recovers
        | all_three_correct
        | temporal_helps
        | morphology_helps
        | all_wrong
        | fusion_loses
    )
    counts = [
        int(fusion_recovers.sum()),
        int(all_three_correct.sum()),
        int(temporal_helps.sum()),
        int(morphology_helps.sum()),
        int(all_wrong.sum()),
        int(fusion_loses.sum()),
        int((~assigned).sum()),
    ]
    return categories, counts


def plot_branch_bars(ax, datasets: list[str], features_dir: Path) -> dict[str, Any]:
    all_counts: list[list[int]] = []
    labels: list[str] | None = None
    details: dict[str, Any] = {}
    for dataset in tqdm(datasets, desc="Branch overlap", leave=False):
        both = load_feature_npz(features_dir / dataset / "test_both.npz")
        ts_only = load_feature_npz(features_dir / dataset / "test_ts_only.npz")
        vision_only = load_feature_npz(features_dir / dataset / "test_vision_only.npz")
        labels, counts = branch_complement_categories(both, ts_only, vision_only)
        all_counts.append(counts)
        total = max(1, int(np.sum(counts)))
        details[dataset] = {
            "counts": {label: int(count) for label, count in zip(labels, counts)},
            "total": total,
            "both_accuracy": float(both["correct"].mean()),
            "ts_only_accuracy": float(ts_only["correct"].mean()),
            "vision_only_accuracy": float(vision_only["correct"].mean()),
        }

    counts_np = np.asarray(all_counts, dtype=np.float32)
    proportions = counts_np / counts_np.sum(axis=1, keepdims=True).clip(min=1)
    left = np.zeros(len(datasets), dtype=np.float32)
    y = np.arange(len(datasets))
    for idx in range(proportions.shape[1]):
        ax.barh(
            y,
            proportions[:, idx],
            left=left,
            color=BRANCH_BAR_COLORS[idx],
            edgecolor="white",
            linewidth=0.5,
            height=0.58,
            label=labels[idx] if labels else None,
        )
        left += proportions[:, idx]
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of query samples", labelpad=7)
    ax.set_title("Branch correctness overlap")
    legend_columns = len(labels) if labels else 1
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.0, -0.33, 1.0, 0.12),
        ncol=legend_columns,
        mode="expand",
        frameon=False,
        columnspacing=0.9,
        handlelength=1.35,
        handletextpad=0.35,
        borderaxespad=0.0,
    )
    return details


def select_attention_samples(
    attention_payloads: list[tuple[str, dict[str, np.ndarray]]],
    *,
    samples_per_dataset: int,
    max_samples: int,
) -> list[tuple[str, dict[str, np.ndarray], int]]:
    selected: list[tuple[str, dict[str, np.ndarray], int]] = []
    max_samples = max(1, int(max_samples))
    samples_per_dataset = max(1, int(samples_per_dataset))
    for dataset, payload in attention_payloads:
        seen_labels: set[int] = set()
        labels = payload["labels"].astype(np.int64)
        for sample_idx, label_value in enumerate(labels.tolist()):
            label = int(label_value)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            selected.append((dataset, payload, sample_idx))
            if len(seen_labels) >= samples_per_dataset or len(selected) >= max_samples:
                break
        if len(selected) >= max_samples:
            break
    return selected


def plot_attention_rollout(
    features_dir: Path,
    output_dir: Path,
    datasets: list[str],
    *,
    samples_per_dataset: int,
    max_samples: int,
) -> dict[str, Any]:
    attention_payloads: list[tuple[str, dict[str, np.ndarray]]] = []
    missing_reasons: dict[str, Any] = {}
    for dataset in tqdm(datasets, desc="Loading attention rollouts", leave=False):
        path = features_dir / dataset / "attention_rollout.npz"
        if not path.exists():
            metadata_path = features_dir / dataset / "metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    missing_reasons[dataset] = metadata.get("attention_rollout", "missing artifact")
                except Exception as exc:
                    missing_reasons[dataset] = f"metadata read failed: {type(exc).__name__}: {exc}"
            else:
                missing_reasons[dataset] = "metadata.json not found"
            continue
        with np.load(path, allow_pickle=False) as data:
            if data["raw_series"].shape[0] == 0:
                continue
            attention_payloads.append((dataset, {key: data[key] for key in data.files}))
    if not attention_payloads:
        raise FileNotFoundError(
            "No attention_rollout.npz artifacts were found for the requested datasets. "
            f"Export metadata reports: {missing_reasons}"
        )

    selected_samples = select_attention_samples(
        attention_payloads,
        samples_per_dataset=samples_per_dataset,
        max_samples=max_samples,
    )
    if not selected_samples:
        raise ValueError(f"No attention samples could be selected from datasets: {datasets}")

    rows = len(selected_samples)
    fig, axes = plt.subplots(rows, 3, figsize=(10.8, 2.35 * rows), squeeze=False)
    selected_details: list[dict[str, Any]] = []
    for row_idx, (dataset, payload, sample_idx) in enumerate(
        tqdm(selected_samples, desc="Plotting attention samples", leave=False)
    ):
        raw = normalize_series(payload["raw_series"][sample_idx])
        morphology = payload["morphology"][sample_idx]
        saliency = payload["saliency"][sample_idx]
        label = int(payload["labels"][sample_idx])
        pred = int(payload["predictions"][sample_idx])
        correct = bool(payload["correct"][sample_idx])
        sample_index = int(payload["sample_indices"][sample_idx]) if "sample_indices" in payload else sample_idx
        selected_details.append(
            {
                "dataset": dataset,
                "artifact_row": int(sample_idx),
                "sample_index": sample_index,
                "label": label,
                "prediction": pred,
                "correct": correct,
            }
        )

        ax_raw, ax_morph, ax_overlay = axes[row_idx]
        ax_raw.plot(np.arange(raw.size), raw, color="#222222", linewidth=0.9)
        ax_raw.set_title(f"{dataset}: y={label}, pred={pred}, {'correct' if correct else 'wrong'}")
        ax_raw.set_xlabel("Time")
        ax_raw.set_ylabel("Value")
        ax_raw.grid(alpha=0.22)

        ax_morph.imshow(morphology, cmap="gray_r", aspect="auto", interpolation="nearest")
        ax_morph.set_title("Morphology map")
        ax_morph.set_xticks([])
        ax_morph.set_yticks([])

        ax_overlay.imshow(morphology, cmap="gray_r", aspect="auto", interpolation="nearest")
        ax_overlay.imshow(saliency, cmap="magma", aspect="auto", alpha=0.46, interpolation="bilinear")
        ax_overlay.set_title("Attention rollout overlay")
        ax_overlay.set_xticks([])
        ax_overlay.set_yticks([])
    fig.subplots_adjust(hspace=0.42, wspace=0.2)
    artifacts = save_pdf_png(fig, output_dir, "timemorph_attention_rollout")
    artifacts["selected_samples"] = selected_details
    return artifacts


def plot_representation_analysis(
    features_dir: Path,
    output_dir: Path,
    *,
    main_dataset: str,
    branch_datasets: list[str],
    random_state: int,
    tsne_max_iter: int,
    tsne_cache_dir: Path | None,
    refresh_tsne_cache: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    both = load_feature_npz(features_dir / main_dataset / "test_both.npz")
    validate_tsne_labels(both["labels"], context=main_dataset)
    panel_features = [
        ("Temporal", "pooled_ts"),
        ("Morphology", "pooled_vision"),
        ("Fused decision", "decision_state"),
    ]
    embeddings = []
    tsne_cache_details: dict[str, Any] = {}
    for title, feature_key in tqdm(panel_features, desc=f"{main_dataset} representation t-SNE", leave=False):
        embedding, cache_detail = compute_or_load_tsne(
            both[feature_key],
            dataset=main_dataset,
            feature_key=feature_key,
            cache_dir=tsne_cache_dir,
            random_state=random_state,
            max_iter=tsne_max_iter,
            refresh_cache=refresh_tsne_cache,
        )
        embeddings.append((title, embedding))
        tsne_cache_details[feature_key] = cache_detail

    branch_count = max(1, len(branch_datasets))
    fig_height = max(6.2, 4.4 + 0.28 * branch_count)
    branch_height_ratio = max(0.95, 0.11 * branch_count)
    fig = plt.figure(figsize=(13.4, fig_height))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, branch_height_ratio], hspace=0.38, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    labels = both["labels"].astype(np.int64)
    correct = both["correct"].astype(bool)
    scatter_tsne(axes[0], embeddings[0][1], labels, correct, title=f"{main_dataset}: temporal tokens")
    scatter_tsne(axes[1], embeddings[1][1], labels, correct, title=f"{main_dataset}: morphology tokens")
    scatter_tsne(axes[2], embeddings[2][1], labels, correct, title=f"{main_dataset}: fused LLM decision state")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Correct", markerfacecolor=TSNE_COLORS[0], markersize=6),
        Line2D([0], [0], marker="x", color=WRONG_MARKER_COLOR, label="Wrong", markersize=6, linestyle="None"),
    ]
    axes[2].legend(handles=legend_handles, loc="lower right", frameon=True, borderpad=0.3)

    ax_bar = fig.add_subplot(gs[1, :])
    branch_details = plot_branch_bars(ax_bar, branch_datasets, features_dir)
    artifacts = save_pdf_png(fig, output_dir, "timemorph_representation_analysis")
    return artifacts, {
        "main_dataset": main_dataset,
        "branch_datasets": branch_datasets,
        "branch_details": branch_details,
        "tsne_cache": tsne_cache_details,
    }


def plot_tsne_appendix(
    features_dir: Path,
    output_dir: Path,
    datasets: list[str],
    *,
    random_state: int,
    tsne_max_iter: int,
    tsne_cache_dir: Path | None,
    refresh_tsne_cache: bool,
) -> dict[str, Any]:
    if not datasets:
        raise ValueError("Appendix t-SNE requires at least one dataset.")
    rows = len(datasets)
    fig, axes = plt.subplots(rows, 3, figsize=(11.3, 3.1 * rows), squeeze=False)
    total_panels = len(datasets) * 3
    progress = tqdm(total=total_panels, desc="Appendix t-SNE", leave=False)
    tsne_cache_details: dict[str, dict[str, Any]] = {}
    for row, dataset in enumerate(datasets):
        both = load_feature_npz(features_dir / dataset / "test_both.npz")
        panels = [
            ("Temporal", "pooled_ts"),
            ("Morphology", "pooled_vision"),
            ("Fused decision", "decision_state"),
        ]
        tsne_cache_details[dataset] = {}
        for col, (title, feature_key) in enumerate(panels):
            progress.set_postfix_str(f"{dataset}: {title}")
            validate_tsne_labels(both["labels"], context=dataset)
            embedding, cache_detail = compute_or_load_tsne(
                both[feature_key],
                dataset=dataset,
                feature_key=feature_key,
                cache_dir=tsne_cache_dir,
                random_state=random_state,
                max_iter=tsne_max_iter,
                refresh_cache=refresh_tsne_cache,
            )
            tsne_cache_details[dataset][feature_key] = cache_detail
            scatter_tsne(
                axes[row, col],
                embedding,
                both["labels"].astype(np.int64),
                both["correct"].astype(bool),
                title=f"{dataset}: {title}",
            )
            progress.update(1)
    progress.close()
    fig.subplots_adjust(hspace=0.42, wspace=0.18)
    artifacts = save_pdf_png(fig, output_dir, "timemorph_tsne_appendix")
    artifacts["tsne_cache"] = tsne_cache_details
    return artifacts


def mutual_knn_alignment(a: np.ndarray, b: np.ndarray, *, k: int = 10) -> float:
    if a.shape[0] != b.shape[0]:
        raise ValueError("Feature matrices must have the same number of samples for mutual kNN alignment.")
    n_samples = int(a.shape[0])
    if n_samples <= 1 or a.shape[1] == 0 or b.shape[1] == 0:
        return float("nan")
    k_eff = min(int(k), n_samples - 1)
    a_scaled = StandardScaler().fit_transform(a.astype(np.float32))
    b_scaled = StandardScaler().fit_transform(b.astype(np.float32))
    neigh_a = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(a_scaled).kneighbors(return_distance=False)[:, 1:]
    neigh_b = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine").fit(b_scaled).kneighbors(return_distance=False)[:, 1:]
    overlaps = [
        len(set(neigh_a[idx].tolist()) & set(neigh_b[idx].tolist())) / float(k_eff)
        for idx in range(n_samples)
    ]
    return float(np.mean(overlaps))


def plot_alignment_heatmap(features_dir: Path, output_dir: Path, datasets: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    views = [
        ("Temporal", "pooled_ts"),
        ("Morphology", "pooled_vision"),
        ("Decision", "decision_state"),
    ]
    matrices: list[np.ndarray] = []
    details: dict[str, Any] = {}
    for dataset in tqdm(datasets, desc="Computing alignment heatmap", leave=False):
        both = load_feature_npz(features_dir / dataset / "test_both.npz")
        matrix = np.eye(len(views), dtype=np.float32)
        for i in range(len(views)):
            for j in range(i + 1, len(views)):
                score = mutual_knn_alignment(both[views[i][1]], both[views[j][1]], k=10)
                matrix[i, j] = matrix[j, i] = score
        matrices.append(matrix)
        details[dataset] = {
            f"{views[i][0]}-{views[j][0]}": float(matrix[i, j])
            for i in range(len(views))
            for j in range(i + 1, len(views))
        }
    mean_matrix = np.nanmean(np.stack(matrices, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(mean_matrix, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(views)))
    ax.set_yticks(range(len(views)))
    ax.set_xticklabels([view[0] for view in views], rotation=25, ha="right")
    ax.set_yticklabels([view[0] for view in views])
    for i in range(len(views)):
        for j in range(len(views)):
            ax.text(j, i, f"{mean_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Mutual kNN alignment across views")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Alignment score")
    artifacts = save_pdf_png(fig, output_dir, "timemorph_branch_alignment_heatmap")
    return artifacts, {"datasets": datasets, "per_dataset": details, "mean_matrix": mean_matrix.tolist()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_style()

    features_dir = Path(args.features_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    datasets = infer_datasets(features_dir)
    if not datasets:
        raise FileNotFoundError(f"No dataset feature directories found under {features_dir}")

    selected_figures = parse_figures(args.figures)
    requested_attention_datasets = parse_csv(args.attention_datasets)
    requested_appendix_datasets = parse_csv(args.appendix_datasets)
    tsne_cache_dir = None if args.disable_tsne_cache else Path(args.tsne_cache_dir).resolve()
    appendix_datasets: list[str] | None = None
    used_datasets: set[str] = set()
    artifacts: dict[str, Any] = {
        "datasets": datasets,
        "features_dir": str(features_dir),
        "tsne_cache_dir": str(tsne_cache_dir) if tsne_cache_dir is not None else None,
        "refresh_tsne_cache": bool(args.refresh_tsne_cache),
        "requested_attention_datasets": requested_attention_datasets,
        "requested_appendix_datasets": requested_appendix_datasets,
        "selected_figures": selected_figures,
    }
    with tqdm(total=len(selected_figures) + 1, desc="Plotting TimeMorph figures") as progress:
        if "attention_rollout" in selected_figures:
            attention_fallback = [args.main_dataset] if args.main_dataset in datasets else [datasets[0]]
            attention_datasets = resolve_requested_datasets(
                requested_attention_datasets,
                datasets,
                context="attention",
                fallback=attention_fallback,
            )
            used_datasets.update(attention_datasets)
            artifacts["attention_rollout"] = plot_attention_rollout(
                features_dir,
                output_dir,
                attention_datasets,
                samples_per_dataset=int(args.attention_samples_per_dataset),
                max_samples=int(args.attention_max_samples),
            )
            progress.update(1)

        if "representation_analysis" in selected_figures:
            main_dataset = resolve_required_dataset(args.main_dataset, datasets, context="main")
            if appendix_datasets is None:
                appendix_datasets = resolve_requested_datasets(
                    requested_appendix_datasets,
                    datasets,
                    context="appendix/overlap",
                    fallback=datasets,
                )
            used_datasets.add(main_dataset)
            used_datasets.update(appendix_datasets)
            rep_artifacts, rep_details = plot_representation_analysis(
                features_dir,
                output_dir,
                main_dataset=main_dataset,
                branch_datasets=appendix_datasets,
                random_state=int(args.random_state),
                tsne_max_iter=int(args.tsne_max_iter),
                tsne_cache_dir=tsne_cache_dir,
                refresh_tsne_cache=bool(args.refresh_tsne_cache),
            )
            artifacts["representation_analysis"] = rep_artifacts
            artifacts["representation_details"] = rep_details
            progress.update(1)

        if "tsne_appendix" in selected_figures:
            if appendix_datasets is None:
                appendix_datasets = resolve_requested_datasets(
                    requested_appendix_datasets,
                    datasets,
                    context="appendix/overlap",
                    fallback=datasets,
                )
            used_datasets.update(appendix_datasets)
            artifacts["tsne_appendix"] = plot_tsne_appendix(
                features_dir,
                output_dir,
                appendix_datasets,
                random_state=int(args.random_state),
                tsne_max_iter=int(args.tsne_max_iter),
                tsne_cache_dir=tsne_cache_dir,
                refresh_tsne_cache=bool(args.refresh_tsne_cache),
            )
            progress.update(1)

        if "branch_alignment_heatmap" in selected_figures:
            if appendix_datasets is None:
                appendix_datasets = resolve_requested_datasets(
                    requested_appendix_datasets,
                    datasets,
                    context="appendix/overlap",
                    fallback=datasets,
                )
            used_datasets.update(appendix_datasets)
            align_artifacts, align_details = plot_alignment_heatmap(features_dir, output_dir, appendix_datasets)
            artifacts["branch_alignment_heatmap"] = align_artifacts
            artifacts["branch_alignment_details"] = align_details
            progress.update(1)

        metadata_datasets = sorted(used_datasets) if used_datasets else datasets
        artifacts["dataset_metadata"] = {
            dataset: load_dataset_metadata(features_dir, dataset)
            for dataset in tqdm(metadata_datasets, desc="Loading metadata", leave=False)
        }
        progress.update(1)

    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = merge_with_existing_manifest(manifest_path, artifacts)
    manifest_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote visualization manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
