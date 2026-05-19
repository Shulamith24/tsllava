#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from scripts.paper_plots.dual_view_motivation import main, parse_args, plot_dual_view_motivation  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())

