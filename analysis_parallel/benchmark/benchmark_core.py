#!/usr/bin/env python3
"""
Benchmark parallel path on real data from benchmark_data.

Single-scan estimate uses one scan from benchmark_data (short run, not full
computation). Writes benchmark_speed.txt and benchmark_memory.txt.

Run on compute node: ssh <gpu_node>, then
  module load conda; module load gpu/1.0; conda activate jax;
  cd <repo_root>; python cad/analysis_parallel/benchmark/benchmark_core.py
"""
from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CAD_DIR = BASE.parent.parent
if str(CAD_DIR / "src") not in sys.path:
    sys.path.insert(0, str(CAD_DIR / "src"))

import numpy as np

from cad.parallel_solve.layout import GlobalLayout, build_layout, load_layout
from cad.parallel_solve.reconstruct_scan import load_scan_artifact, run_one_scan
from cad.parallel_solve.synthesize_scan import run_synthesis

# Real data: one scan from benchmark_data (same as benchmark_cg_convergence)
BENCHMARK_DATA_DIR = BASE / "benchmark_data" / "ra0hdec-59.75" / "101706388"
SCAN_PATH = BENCHMARK_DATA_DIR / "0000_calibrated_scan000.npz"

# Two reconstructed scans for synthesis (hardcoded)
RECONSTRUCTED_DIR = BASE / "benchmark_data" / "ra0hdec-59.75_reconstructed" / "101706388"
SYNTHESIS_SCAN_PATHS = (
    RECONSTRUCTED_DIR / "scan_0000_ml.npz",
    RECONSTRUCTED_DIR / "scan_0001_ml.npz",
)

OUT_BASE = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
DATASET_NAME = "ra0hdec-59.75"
FIELD_ID = "101706388"
LAYOUT_NPZ = OUT_BASE / DATASET_NAME / FIELD_ID / "layout.npz"
SCAN_NPZ_DIR = OUT_BASE / DATASET_NAME / FIELD_ID / "scans"
OUT_DIR = BASE
N_REP = 5

# Short run uses this many iters (cheap); scale to production for time estimate
CG_SAMPLE = 20
CG_FULL = 512
SCALE = CG_FULL / CG_SAMPLE


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _layout_from_two_scan_artifacts(
    path0: Path,
    path1: Path,
    field_id: str = "101706388",
) -> GlobalLayout:
    """Build GlobalLayout from two scan artifact NPZs (union of obs pixels)."""
    art0 = load_scan_artifact(path0)
    art1 = load_scan_artifact(path1)
    obs0 = np.asarray(art0["obs_pix_global_scan"], dtype=np.int64)
    obs1 = np.asarray(art1["obs_pix_global_scan"], dtype=np.int64)
    obs_pix_global = np.unique(np.concatenate([obs0, obs1]))
    with np.load(path0, allow_pickle=True) as z:
        bbox_ix0 = int(z["bbox_ix0"])
        bbox_iy0 = int(z["bbox_iy0"])
        nx = int(z["nx"])
        ny = int(z["ny"])
        pixel_size_deg = float(z["pixel_size_deg"])
    n_pix = nx * ny
    global_to_obs = np.full(n_pix, -1, dtype=np.int64)
    global_to_obs[obs_pix_global] = np.arange(obs_pix_global.size, dtype=np.int64)
    return GlobalLayout(
        bbox_ix0=bbox_ix0,
        bbox_iy0=bbox_iy0,
        nx=nx,
        ny=ny,
        obs_pix_global=obs_pix_global,
        global_to_obs=global_to_obs,
        scan_paths=(path0, path1),
        pixel_size_deg=pixel_size_deg,
        field_id=field_id,
    )


def main() -> None:
    speed_lines: list[str] = ["Parallel benchmark: speed (real data where noted)", "=" * 50]
    mem_lines: list[str] = ["Parallel benchmark: memory", "=" * 50]

    # --- load_layout (optional: from scratch if present) ---
    layout = None
    if LAYOUT_NPZ.exists():
        rss_before = _rss_mb()
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            layout = load_layout(LAYOUT_NPZ)
            timings.append(time.perf_counter() - t0)
        rss_after = _rss_mb()
        speed_lines.append(f"load_layout (scratch): {np.mean(timings):.6f} s (mean over {N_REP})")
        mem_lines.append(f"load_layout: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        if layout is not None:
            mem_lines.append(f"  n_scans={layout.n_scans} n_obs={layout.n_obs} n_pix={layout.n_pix}")
    else:
        speed_lines.append(f"load_layout: SKIP (missing {LAYOUT_NPZ})")
        mem_lines.append(f"load_layout: SKIP (missing {LAYOUT_NPZ})")

    # --- single-scan on real benchmark_data (short run), then load_scan_artifact on that output ---
    if not SCAN_PATH.exists():
        speed_lines.append("")
        speed_lines.append(f"single_scan_estimate + load_scan_artifact: SKIP (missing {SCAN_PATH})")
        mem_lines.append("single_scan_estimate + load_scan_artifact: SKIP (benchmark_data scan missing)")
    else:
        layout_bench = build_layout(field_id=FIELD_ID, scan_paths=[SCAN_PATH])
        timings_dict: dict[str, float] = {}
        artifact_path: Path | None = None
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            t0 = time.perf_counter()
            run_one_scan(
                layout_bench,
                0,
                out_dir,
                cg_maxiter=CG_SAMPLE,
                cg_tol=1.0,
                timings=timings_dict,
            )
            t_total = time.perf_counter() - t0
            artifact_path = out_dir / "scan_0000_ml.npz"
            if artifact_path.exists():
                rss_before = _rss_mb()
                load_timings = []
                for _ in range(N_REP):
                    t0 = time.perf_counter()
                    art = load_scan_artifact(artifact_path)
                    load_timings.append(time.perf_counter() - t0)
                rss_after = _rss_mb()
                n_s = art["cov_inv"].shape[0]
                speed_lines.append("")
                speed_lines.append(f"load_scan_artifact (real, from single-scan run, n_obs_scan={n_s}): {np.mean(load_timings):.6f} s (mean over {N_REP})")
                mem_lines.append(f"load_scan_artifact: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
                mem_lines.append(f"  cov_inv {n_s}x{n_s} = {n_s * n_s * 8 / 1024**2:.2f} MB float64")

        t_solve = timings_dict.get("solve_single_scan", 0.0)
        t_fisher = timings_dict.get("build_scan_information", 0.0)
        t_setup = timings_dict.get("setup", 0.0)
        t_write = timings_dict.get("write", 0.0)
        est_total = t_setup + t_write + (t_solve + t_fisher) * SCALE
        speed_lines.append("")
        speed_lines.append("Single-scan estimate (real data: benchmark_data/.../0000_calibrated_scan000.npz):")
        speed_lines.append(f"  Short run ({CG_SAMPLE} iters): {t_total:.2f} s total")
        for stage, t in timings_dict.items():
            if stage in ("solve_single_scan", "build_scan_information"):
                speed_lines.append(f"    {stage}: {t:.2f} s (est. full: {t * SCALE:.1f} s)")
            else:
                speed_lines.append(f"    {stage}: {t:.2f} s")
        speed_lines.append(f"  Estimated single scan ({CG_FULL} iters): {est_total:.1f} s")
        mem_lines.append("")
        mem_lines.append("single_scan_estimate: (see speed for stages)")

    # --- run_synthesis (two hardcoded reconstructed scans in benchmark_data) ---
    both_present = all(p.exists() for p in SYNTHESIS_SCAN_PATHS)
    if not both_present:
        speed_lines.append("")
        speed_lines.append(f"run_synthesis: SKIP (missing {RECONSTRUCTED_DIR}/scan_0000_ml.npz or scan_0001_ml.npz)")
        mem_lines.append("run_synthesis: SKIP (two-scan benchmark_data not present)")
    else:
        layout_synth = _layout_from_two_scan_artifacts(
            SYNTHESIS_SCAN_PATHS[0],
            SYNTHESIS_SCAN_PATHS[1],
            field_id=FIELD_ID,
        )
        out_npz = OUT_DIR / "benchmark_synthesis_out.npz"
        rss_before = _rss_mb()
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            run_synthesis(layout_synth, RECONSTRUCTED_DIR, out_npz)
            timings.append(time.perf_counter() - t0)
        rss_after = _rss_mb()
        speed_lines.append("")
        speed_lines.append(
            f"run_synthesis (2 scans, benchmark_data/ra0hdec-59.75_reconstructed/...): {np.mean(timings):.6f} s (mean over {N_REP})"
        )
        mem_lines.append(f"run_synthesis: RSS {rss_before:.1f} -> {rss_after:.1f} MB")
        n_obs = layout_synth.n_obs
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
