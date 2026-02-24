#!/usr/bin/env python3
"""
Diagonal inverse-variance synthesis from per-scan artifacts. Thin CLI around parallel_solve.
Usage: run_synthesis.py <layout.npz> <scan_npz_dir> <out_combined.npz>
"""

from __future__ import annotations

import pathlib
import sys

BASE_DIR = pathlib.Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent

if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

from cad.parallel_solve import load_layout, run_synthesis


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) < 3:
        print(
            "Usage: run_synthesis.py <layout.npz> <scan_npz_dir> <out_combined.npz>",
            file=sys.stderr,
        )
        sys.exit(1)
    layout_path = pathlib.Path(argv[0])
    scan_npz_dir = pathlib.Path(argv[1])
    out_path = pathlib.Path(argv[2])

    layout = load_layout(layout_path)
    run_synthesis(layout, scan_npz_dir, out_path)


if __name__ == "__main__":
    main()
