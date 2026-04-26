from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def _candidate_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _find_snapshot_in_cache(model_id: str) -> str | None:
    repo_dir_name = f"models--{model_id.replace('/', '--')}"
    for cache_root in _candidate_cache_roots():
        repo_dir = cache_root / repo_dir_name
        if not repo_dir.exists():
            continue

        ref_path = repo_dir / "refs" / "main"
        if ref_path.exists():
            snapshot_id = ref_path.read_text(encoding="utf-8").strip()
            if snapshot_id:
                snapshot_dir = repo_dir / "snapshots" / snapshot_id
                if snapshot_dir.exists() and (snapshot_dir / "config.json").exists():
                    return str(snapshot_dir)

        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        candidates = sorted(
            (
                path for path in snapshots_dir.iterdir()
                if path.is_dir() and (path / "config.json").exists()
            ),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])
    return None


def resolve_local_hf_snapshot(model_id_or_path: str) -> str:
    """Return a local directory when the model already exists in the HF cache."""
    if not model_id_or_path:
        return model_id_or_path
    if Path(model_id_or_path).exists():
        return str(Path(model_id_or_path))
    cached_snapshot = _find_snapshot_in_cache(model_id_or_path)
    if cached_snapshot is not None:
        return cached_snapshot
    try:
        return snapshot_download(model_id_or_path, local_files_only=True)
    except Exception:
        return model_id_or_path
