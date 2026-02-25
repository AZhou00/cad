#!/usr/bin/env python3
"""
Reconstruction: build layout, then run scans in parallel on 4 GPUs.
Skips scans that already have output. For use on a single node with 4 GPUs (e.g. after ssh to a GPU node).

Input data: cad/data/<field_id>/<observation_id>/ (e.g. cad/data/ra0hdec-59.75/101706388).
Output: OUT_BASE / field_id / observation_id / (layout.npz, scans/*.npz).

Usage:
  module load conda; module load gpu/1.0; conda activate jax
  cd <repo_root>
  python cad/analysis_parallel/run_reconstruction.py [field_id] [observation_id]

Example (from scratch for cad/data/ra0hdec-59.75/101706388):
  python cad/analysis_parallel/run_reconstruction.py ra0hdec-59.75 101706388

If args omitted, uses FIELD_ID and OBSERVATION_ID below. After all scans: python run_synthesis.py (paths in that script must match).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Parameters (override via argv or edit)
BASE_DIR = Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
OUT_BASE = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
N_GPUS = 4

# Default data; argv can override
FIELD_ID = "ra0hdec-59.75"
OBSERVATION_ID = "101724132"

# run_one_scan kwargs (optional)
N_ELL_BINS = 128
CL_FLOOR_MK2 = 1e-12
NOISE_MK = 10.0
CG_TOL = 1e-3
CG_MAXITER = 512


def _worker(
    gpu_id: int,
    scan_indices: list[int],
    layout_path: Path,
    out_dir: Path,
) -> int:
    """Run reconstruction for assigned scan indices on one GPU. Set CUDA_VISIBLE_DEVICES before importing JAX."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if str(CAD_DIR / "src") not in sys.path:
        sys.path.insert(0, str(CAD_DIR / "src"))
    from cad.parallel_solve import load_layout, run_one_scan

    layout = load_layout(layout_path)
    done = 0
    for idx in tqdm(scan_indices, desc=("gpu%i" % gpu_id), leave=True):
        run_one_scan(
            layout,
            idx,
            out_dir,
            n_ell_bins=N_ELL_BINS,
            cl_floor_mk2=CL_FLOOR_MK2,
            noise_per_raw_detector_per_153hz_sample_mk=NOISE_MK,
            cg_tol=CG_TOL,
            cg_maxiter=CG_MAXITER,
        )
        done += 1
        print(f"[gpu{gpu_id}] scan {idx} done ({done}/{len(scan_indices)})", flush=True)
    return done


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) >= 2:
        field_id, observation_id = str(argv[0]), str(argv[1])
    else:
        field_id, observation_id = FIELD_ID, OBSERVATION_ID

    layout_path = OUT_BASE / field_id / observation_id / "layout.npz"
    out_dir = OUT_BASE / field_id / observation_id / "scans"

    # Build layout (subprocess so this process does not import JAX)
    build_cmd = [sys.executable, str(BASE_DIR / "build_layout.py"), field_id, observation_id]
    print("[build_layout]", " ".join(build_cmd), flush=True)
    subprocess.run(build_cmd, cwd=str(CAD_DIR.parent), check=True)

    # Which scans are left to do
    with np.load(layout_path, allow_pickle=True) as z:
        n_scans = len(z["scan_paths"])
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(n_scans) if not (out_dir / f"scan_{i:04d}_ml.npz").exists()]
    if not todo:
        print("All scans already done.", flush=True)
        return
    print(f"Scans to do: {len(todo)} / {n_scans}", flush=True)

    # Partition by GPU: worker gpu_id gets indices where index % N_GPUS == gpu_id
    by_gpu: list[list[int]] = [[] for _ in range(N_GPUS)]
    for i in todo:
        by_gpu[i % N_GPUS].append(i)
    for g in range(N_GPUS):
        by_gpu[g].sort()
        print(f"  GPU {g}: {len(by_gpu[g])} scans", flush=True)

    # Run N_GPUS workers
    with mp.Pool(N_GPUS) as pool:
        results = pool.starmap(
            _worker,
            [(g, by_gpu[g], layout_path, out_dir) for g in range(N_GPUS)],
        )
    print("Done.", sum(results), "scans completed.", flush=True)


if __name__ == "__main__":
    main()
