#!/usr/bin/env python3
"""
Benchmark parallel-path core functions: load_layout, load_scan_artifact, run_synthesis.
Writes benchmark_speed.txt and benchmark_memory.txt to this directory.

Run on compute node nid001097: ssh nid001097, then
  module load conda; module load gpu/1.0; conda activate jax;
  cd <repo_root>; PYTHONPATH=cad/src python cad/analysis_parallel/benchmark/benchmark_core.py
"""
from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CAD_SRC = BASE.parent.parent / "src"
if str(CAD_SRC) not in sys.path:
    sys.path.insert(0, str(CAD_SRC))

import numpy as np

from cad.parallel_solve.layout import load_layout
from cad.parallel_solve.reconstruct_scan import load_scan_artifact
from cad.parallel_solve.synthesize_scan import run_synthesis

LAYOUT_NPZ = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data/ra0hdec-59.75/101706388/layout.npz")
SCAN_ML_DIR = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data/ra0hdec-59.75/101706388/scan_ml")
OUT_DIR = BASE
N_REP = 5


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    speed_lines: list[str] = ["Parallel core benchmark: speed", "=" * 50]
    mem_lines: list[str] = ["Parallel core benchmark: memory", "=" * 50]

    # --- load_layout ---
    if not LAYOUT_NPZ.exists():
        speed_lines.append(f"load_layout: SKIP (missing {LAYOUT_NPZ})")
        mem_lines.append(f"load_layout: SKIP (missing {LAYOUT_NPZ})")
        layout = None
    else:
        rss_before = _rss_mb()
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            layout = load_layout(LAYOUT_NPZ)
            timings.append(time.perf_counter() - t0)
        rss_after = _rss_mb()
        speed_lines.append(f"load_layout: {np.mean(timings):.6f} s (mean over {N_REP})")
        mem_lines.append(f"load_layout: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        if layout is not None:
            mem_lines.append(f"  n_scans={layout.n_scans} n_obs={layout.n_obs} n_pix={layout.n_pix}")

    # --- load_scan_artifact ---
    scan_npz: Path | None = None
    if SCAN_ML_DIR.exists():
        candidates = sorted(SCAN_ML_DIR.glob("scan_*_ml.npz"))
        if candidates:
            scan_npz = candidates[0]
    if scan_npz is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            scan_npz = Path(f.name)
        try:
            n = 100
            np.savez_compressed(
                scan_npz,
                obs_pix_global_scan=np.arange(n, dtype=np.int64),
                c_hat_scan_obs=np.zeros((n,), dtype=np.float64),
                cov_inv=np.eye(n, dtype=np.float64),
                Pt_Ninv_d=np.zeros((n,), dtype=np.float64),
            )
            rss_before = _rss_mb()
            timings = []
            for _ in range(N_REP):
                t0 = time.perf_counter()
                art = load_scan_artifact(scan_npz)
                timings.append(time.perf_counter() - t0)
            rss_after = _rss_mb()
            speed_lines.append(f"load_scan_artifact (n={n} synthetic): {np.mean(timings):.6f} s (mean over {N_REP})")
            mem_lines.append(f"load_scan_artifact: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        finally:
            scan_npz.unlink(missing_ok=True)
            scan_npz = None
    else:
        rss_before = _rss_mb()
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            art = load_scan_artifact(scan_npz)
            timings.append(time.perf_counter() - t0)
        rss_after = _rss_mb()
        n_s = art["cov_inv"].shape[0]
        speed_lines.append(f"load_scan_artifact (scan {scan_npz.name}, n_obs_scan={n_s}): {np.mean(timings):.6f} s (mean over {N_REP})")
        mem_lines.append(f"load_scan_artifact: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        mem_lines.append(f"  cov_inv {n_s}x{n_s} = {n_s * n_s * 8 / 1024**2:.2f} MB float64")

    # --- run_synthesis ---
    if layout is None or not SCAN_ML_DIR.exists():
        speed_lines.append("run_synthesis: SKIP (no layout or scan_ml dir)")
        mem_lines.append("run_synthesis: SKIP (no layout or scan_ml dir)")
    else:
        out_npz = OUT_DIR / "benchmark_synthesis_out.npz"
        rss_before = _rss_mb()
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            run_synthesis(layout, SCAN_ML_DIR, out_npz)
            timings.append(time.perf_counter() - t0)
        rss_after = _rss_mb()
        speed_lines.append(f"run_synthesis (n_scans={layout.n_scans}): {np.mean(timings):.6f} s (mean over {N_REP})")
        mem_lines.append(f"run_synthesis: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        n_obs = layout.n_obs
        mem_lines.append(f"  cov_inv_tot {n_obs}x{n_obs} = {n_obs * n_obs * 8 / 1024**2:.2f} MB float64")
        if out_npz.exists():
            out_npz.unlink(missing_ok=True)

    speed_lines.append("")
    mem_lines.append("")

    out_speed = OUT_DIR / "benchmark_speed.txt"
    out_mem = OUT_DIR / "benchmark_memory.txt"
    out_speed.write_text("\n".join(speed_lines))
    out_mem.write_text("\n".join(mem_lines))
    print("\n".join(speed_lines))
    print("\n".join(mem_lines))
    print("Wrote", out_speed, out_mem)


if __name__ == "__main__":
    main()
