from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
LATEX_ROOT = REPO_ROOT / "latex_all" / "Formatting_Instructions_For_NeurIPS_2026"
PAPER_OUTPUT_DIR = LATEX_ROOT / "paper_plots" / "outputs"
FIGURES_DIR = LATEX_ROOT / "figures"
REPORTS_ROOT = REPO_ROOT / "results" / "ucr_batches" / "reports"

_CACHE_ROOT = Path("/tmp") / "tsllava_paper_plots_cache"
_MPL_CONFIG_DIR = _CACHE_ROOT / "matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


PAPER_COLORS = {
    "primary": "#B22222",
    "cosco": "#C44E52",
    "resnet": "#4C72B0",
    "tapnet": "#55A868",
    "patchtst": "#CCB974",
    "informer": "#BCBD22",
    "gpt4ts": "#17BECF",
    "neutral": "#6B7280",
    "positive": "#2E7D32",
    "negative": "#B23A48",
    "panel_b": "#2F6B8A",
    "panel_b_light": "#9FC5D9",
}


def apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pdf_png(fig, output_dir: Path, base_name: str, *, dpi: int = 320) -> dict[str, Path]:
    ensure_dir(output_dir)
    pdf_path = output_dir / f"{base_name}.pdf"
    png_path = output_dir / f"{base_name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return {"pdf": pdf_path, "png": png_path}


def extract_latex_numbers(text: str) -> list[float]:
    matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    return [float(match) for match in matches]


def read_latex_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%") or line.startswith("\\"):
            continue
        if "&" not in line:
            continue
        line = line.removesuffix(r"\\").strip()
        rows.append([cell.strip() for cell in line.split("&")])
    return rows


def normalize_latex_label(label: str) -> str:
    cleaned = label.replace("$", "")
    cleaned = cleaned.replace(r"\textbf{", "").replace(r"\underline{", "")
    cleaned = cleaned.replace("}", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.strip()
    return re.sub(r"\s+", " ", cleaned)
