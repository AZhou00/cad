#!/usr/bin/env python3
"""
Benchmark parallel path only: layout, per-scan (CPU + GPU), load artifact, synthesis.

All data from benchmark_data (no scratch). Output format per step:
  equation | implemented by xxx | ___ benchmark results | description [single-scan time if applicable].

Run on compute node: ssh <gpu_node>, module load conda; module load gpu/1.0; conda activate jax;
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

from cad.parallel_solve.layout import GlobalLayout, build_layout
from cad.parallel_solve.reconstruct_scan import load_scan_artifact, run_one_scan
from cad.parallel_solve.synthesize_scan import run_synthesis

FIELD_ID = "ra0hdec-59.75"
OBSERVATION_ID = "101706388"
BENCHMARK_DATA_DIR = BASE / "benchmark_data" / FIELD_ID / OBSERVATION_ID
SCAN_PATH = BENCHMARK_DATA_DIR / "0000_calibrated_scan000.npz"
RECONSTRUCTED_DIR = BASE / "benchmark_data" / f"{FIELD_ID}_reconstructed" / OBSERVATION_ID
SYNTHESIS_SCAN_PATHS = (
    RECONSTRUCTED_DIR / "scan_0000_ml.npz",
    RECONSTRUCTED_DIR / "scan_0001_ml.npz",
)
OUT_DIR = BASE
N_REP = 5
CG_SAMPLE = 20
CG_FULL = 512
SCALE = CG_FULL / CG_SAMPLE


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _layout_from_two_scan_artifacts(
    path0: Path,
    path1: Path,
    observation_id: str = OBSERVATION_ID,
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
        field_id=observation_id,
    )


def main() -> None:
    lines: list[str] = [
        "Parallel path benchmark (benchmark_data only)",
        "=" * 60,
        "",
    ]

    # --- 1. build_layout: obs_pix_global, global_to_obs from hit counts ---
    if not SCAN_PATH.exists():
        lines.append("build_layout: SKIP (missing benchmark_data scan)")
    else:
        timings = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            build_layout(field_id=OBSERVATION_ID, scan_paths=[SCAN_PATH])
            timings.append(time.perf_counter() - t0)
        t_mean = float(np.mean(timings))
        rss = _rss_mb()
        lines.extend([
            "Layout (bbox union, observed pixel set)",
            "   obs_pix_global, global_to_obs from hit_count >= min_hits",
            "   Implemented by: cad.parallel_solve.layout.build_layout",
            f"   ___ Benchmark: {t_mean:.6f} s (mean over {N_REP}), RSS {rss:.1f} MB",
            "   Builds global CMB bbox and observed-pixel index set from scan paths.",
            "",
        ])

    # --- Single-scan: joint CG (CPU) + Fisher cov_inv/Pt_Ninv_d (GPU) ---
    if not SCAN_PATH.exists():
        lines.append("run_one_scan (single scan): SKIP (missing benchmark_data scan)")
    else:
        layout_bench = build_layout(field_id=OBSERVATION_ID, scan_paths=[SCAN_PATH])
        timings_dict: dict[str, float] = {}
        t_load = None
        n_s = None
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
                load_timings = []
                for _ in range(N_REP):
                    t0 = time.perf_counter()
                    art = load_scan_artifact(artifact_path)
                    load_timings.append(time.perf_counter() - t0)
                t_load = float(np.mean(load_timings))
                n_s = art["cov_inv"].shape[0]

        t_setup = timings_dict.get("setup", 0.0)
        t_solve = timings_dict.get("solve_single_scan", 0.0)
        t_fisher = timings_dict.get("build_scan_information", 0.0)
        t_write = timings_dict.get("write", 0.0)
        est_single_scan = t_setup + t_write + (t_solve + t_fisher) * SCALE
        with tempfile.TemporaryDirectory() as tmpdir2:
            t0 = time.perf_counter()
            run_one_scan(layout_bench, 0, Path(tmpdir2), cg_maxiter=CG_FULL, cg_tol=1e-3)
            t_full_512 = time.perf_counter() - t0
        lines.extend([
            "Joint solve (CPU): [P' N^{-1} P, P' N^{-1} W; W' N^{-1} P, M] [c; a0] = [P' N^{-1} d; W' N^{-1} d]",
            "   Implemented by: cad.direct_solve.reconstruct_scan.solve_single_scan (CG)",
            f"   ___ Benchmark (short {CG_SAMPLE} iters): solve_single_scan {t_solve:.3f} s.",
            f"   ___ Benchmark (full {CG_FULL} iters, cg_tol=1e-3): one-scan pipeline {t_full_512:.1f} s.",
            f"   Scaled to {CG_FULL} iters: {t_solve * SCALE:.1f} s per scan.",
            "",
            "Per-scan precision and RHS (GPU): Cov(hat c_s)^{-1} = P' tilde N_s^{-1} P, P' tilde N_s^{-1} d (Woodbury)",
            "   Implemented by: cad.parallel_solve.fisher.build_scan_information",
            f"   ___ Benchmark (short {CG_SAMPLE} iters): build_scan_information {t_fisher:.3f} s.",
            f"   Scaled to {CG_FULL} iters: {t_fisher * SCALE:.1f} s per scan.",
            "",
            "Full single-scan pipeline (setup + joint solve + Fisher + write)",
            "   Implemented by: cad.parallel_solve.reconstruct_scan.run_one_scan",
            f"   ___ Benchmark (short run {CG_SAMPLE} iters): {t_total:.2f} s total.",
            f"   Estimated time for one scan at production ({CG_FULL} iters): {est_single_scan:.1f} s.",
            "",
        ])

        if t_load is not None and n_s is not None:
            lines.extend([
                "Load per-scan artifact (I/O)",
                "   Read cov_inv, Pt_Ninv_d, obs_pix_global_scan from scan_XXXX_ml.npz",
                "   Implemented by: cad.parallel_solve.reconstruct_scan.load_scan_artifact",
                f"   ___ Benchmark: {t_load:.6f} s (mean over {N_REP}), n_obs_scan={n_s}, cov_inv {n_s}x{n_s}.",
                "   Used once per scan during synthesis.",
                "",
            ])

    # --- run_synthesis: sum precision, sum RHS, solve (CPU-only; micro-benchmark) ---
    both_present = all(p.exists() for p in SYNTHESIS_SCAN_PATHS)
    if not both_present:
        lines.append(f"run_synthesis: SKIP (missing scan_0000_ml.npz or scan_0001_ml.npz in benchmark_data/{FIELD_ID}_reconstructed/...)")
    else:
        layout_synth = _layout_from_two_scan_artifacts(
            SYNTHESIS_SCAN_PATHS[0],
            SYNTHESIS_SCAN_PATHS[1],
            observation_id=OBSERVATION_ID,
        )
        out_npz = OUT_DIR / "benchmark_synthesis_out.npz"
        timings_synth: list[dict[str, float]] = []
        for _ in range(N_REP):
            td: dict[str, float] = {}
            run_synthesis(layout_synth, RECONSTRUCTED_DIR, out_npz, timings=td)
            timings_synth.append(td)
        t_synth = float(np.mean([sum(td.values()) for td in timings_synth]))
        t_load_s = float(np.mean([td["load_s"] for td in timings_synth]))
        t_accum_s = float(np.mean([td["accumulate_s"] for td in timings_synth]))
        t_solve_s = float(np.mean([td["solve_s"] for td in timings_synth]))
        rss = _rss_mb()
        n_obs = layout_synth.n_obs
        lines.extend([
            "Synthesis: (sum_s Cov(hat c_s)^{-1}) c_hat = sum_s P' tilde N_s^{-1} d_s (CPU only)",
            "   Implemented by: cad.parallel_solve.synthesize_scan.run_synthesis",
            f"   ___ Benchmark (2 scans, n_obs={n_obs}): total {t_synth:.6f} s (mean over {N_REP}), RSS {rss:.1f} MB.",
            f"   Micro: load_s={t_load_s:.4f} s, accumulate_s={t_accum_s:.4f} s, solve_s={t_solve_s:.4f} s.",
            "   Solve is O(n_obs^3); for large n_obs synthesis is dominated by solve and I/O.",
            "",
        ])
        if out_npz.exists():
            out_npz.unlink(missing_ok=True)

    out_path = OUT_DIR / "benchmark_results.txt"
    out_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
