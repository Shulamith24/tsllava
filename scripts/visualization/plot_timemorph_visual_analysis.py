#!/usr/bin/env python3

from __future__ import annotations

import argparse
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


COLORS = [
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#17BECF",
]


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TimeMorph qualitative and representation analysis figures.")
    parser.add_argument("--features_dir", type=str, default=str(DEFAULT_FEATURES_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest_path", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--main_dataset", type=str, default="ECG200")
    parser.add_argument("--attention_datasets", type=str, default="ECG200,TwoLeadECG")
    parser.add_argument("--appendix_datasets", type=str, default="")
    parser.add_argument("--random_state", type=int, default=3407)
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
    return {label: COLORS[index % len(COLORS)] for index, label in enumerate(unique)}


def compute_tsne(features: np.ndarray, *, random_state: int) -> np.ndarray:
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
    ).fit_transform(scaled)


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
    bar_colors = ["#B22222", "#4C72B0", "#7A7A7A", "#55A868", "#D0D0D0", "#C44E52", "#F0F0F0"]
    all_counts: list[list[int]] = []
    labels: list[str] | None = None
    details: dict[str, Any] = {}
    for dataset in datasets:
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
            color=bar_colors[idx],
            edgecolor="white",
            linewidth=0.5,
            height=0.58,
            label=labels[idx] if labels else None,
        )
        left += proportions[:, idx]
    ax.set_yticks(y)
    ax.set_yticklabels(datasets)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of query samples")
    ax.set_title("Branch correctness overlap")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    return details


def plot_attention_rollout(features_dir: Path, output_dir: Path, datasets: list[str]) -> dict[str, str]:
    attention_payloads = []
    missing_reasons: dict[str, Any] = {}
    for dataset in datasets:
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

    rows = min(4, sum(payload["raw_series"].shape[0] for _, payload in attention_payloads))
    fig, axes = plt.subplots(rows, 3, figsize=(10.8, 2.35 * rows), squeeze=False)
    row_idx = 0
    for dataset, payload in attention_payloads:
        for sample_idx in range(payload["raw_series"].shape[0]):
            if row_idx >= rows:
                break
            raw = normalize_series(payload["raw_series"][sample_idx])
            morphology = payload["morphology"][sample_idx]
            saliency = payload["saliency"][sample_idx]
            label = int(payload["labels"][sample_idx])
            pred = int(payload["predictions"][sample_idx])
            correct = bool(payload["correct"][sample_idx])

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
            row_idx += 1
        if row_idx >= rows:
            break
    for idx in range(row_idx, rows):
        for ax in axes[idx]:
            ax.axis("off")
    fig.subplots_adjust(hspace=0.42, wspace=0.2)
    return save_pdf_png(fig, output_dir, "timemorph_attention_rollout")


def plot_representation_analysis(
    features_dir: Path,
    output_dir: Path,
    *,
    main_dataset: str,
    branch_datasets: list[str],
    random_state: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    both = load_feature_npz(features_dir / main_dataset / "test_both.npz")
    validate_tsne_labels(both["labels"], context=main_dataset)
    ts_embedding = compute_tsne(both["pooled_ts"], random_state=random_state)
    vision_embedding = compute_tsne(both["pooled_vision"], random_state=random_state)
    decision_embedding = compute_tsne(both["decision_state"], random_state=random_state)

    fig = plt.figure(figsize=(13.4, 6.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.38, wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    labels = both["labels"].astype(np.int64)
    correct = both["correct"].astype(bool)
    scatter_tsne(axes[0], ts_embedding, labels, correct, title=f"{main_dataset}: temporal tokens")
    scatter_tsne(axes[1], vision_embedding, labels, correct, title=f"{main_dataset}: morphology tokens")
    scatter_tsne(axes[2], decision_embedding, labels, correct, title=f"{main_dataset}: fused LLM decision state")

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="Correct", markerfacecolor="#4C72B0", markersize=6),
        Line2D([0], [0], marker="x", color="#222222", label="Wrong", markersize=6, linestyle="None"),
    ]
    axes[2].legend(handles=legend_handles, loc="lower right", frameon=True, borderpad=0.3)

    ax_bar = fig.add_subplot(gs[1, :])
    branch_details = plot_branch_bars(ax_bar, branch_datasets, features_dir)
    artifacts = save_pdf_png(fig, output_dir, "timemorph_representation_analysis")
    return artifacts, {
        "main_dataset": main_dataset,
        "branch_datasets": branch_datasets,
        "branch_details": branch_details,
    }


def plot_tsne_appendix(features_dir: Path, output_dir: Path, datasets: list[str], *, random_state: int) -> dict[str, str]:
    if not datasets:
        raise ValueError("Appendix t-SNE requires at least one dataset.")
    rows = len(datasets)
    fig, axes = plt.subplots(rows, 3, figsize=(11.3, 3.1 * rows), squeeze=False)
    for row, dataset in enumerate(datasets):
        both = load_feature_npz(features_dir / dataset / "test_both.npz")
        panels = [
            ("Temporal", both["pooled_ts"]),
            ("Morphology", both["pooled_vision"]),
            ("Fused decision", both["decision_state"]),
        ]
        for col, (title, features) in enumerate(panels):
            validate_tsne_labels(both["labels"], context=dataset)
            embedding = compute_tsne(features, random_state=random_state)
            scatter_tsne(
                axes[row, col],
                embedding,
                both["labels"].astype(np.int64),
                both["correct"].astype(bool),
                title=f"{dataset}: {title}",
            )
    fig.subplots_adjust(hspace=0.42, wspace=0.18)
    return save_pdf_png(fig, output_dir, "timemorph_tsne_appendix")


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
    for dataset in datasets:
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

    main_dataset = args.main_dataset if args.main_dataset in datasets else datasets[0]
    attention_datasets = [dataset for dataset in parse_csv(args.attention_datasets) if dataset in datasets]
    if not attention_datasets:
        attention_datasets = [main_dataset]
    appendix_datasets = [dataset for dataset in parse_csv(args.appendix_datasets) if dataset in datasets] or datasets

    artifacts: dict[str, Any] = {"datasets": datasets, "features_dir": str(features_dir)}
    artifacts["attention_rollout"] = plot_attention_rollout(features_dir, output_dir, attention_datasets)
    rep_artifacts, rep_details = plot_representation_analysis(
        features_dir,
        output_dir,
        main_dataset=main_dataset,
        branch_datasets=appendix_datasets,
        random_state=int(args.random_state),
    )
    artifacts["representation_analysis"] = rep_artifacts
    artifacts["representation_details"] = rep_details
    artifacts["tsne_appendix"] = plot_tsne_appendix(
        features_dir,
        output_dir,
        appendix_datasets,
        random_state=int(args.random_state),
    )
    align_artifacts, align_details = plot_alignment_heatmap(features_dir, output_dir, appendix_datasets)
    artifacts["branch_alignment_heatmap"] = align_artifacts
    artifacts["branch_alignment_details"] = align_details
    artifacts["dataset_metadata"] = {
        dataset: load_dataset_metadata(features_dir, dataset)
        for dataset in datasets
    }

    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote visualization manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
