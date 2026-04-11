#!/usr/bin/env python3
"""
Run reconstruction for every observation under a field scratch root where work remains.

Completion rule (same as checking whether reconstruction finished):
  len(binned_tod_10arcmin/*.npz) == len(scans/scan_*_ml.npz)

Observations without a binned_tod_10arcmin directory on scratch are skipped (logged); there is
nothing to align with scans/ until that data exists there.

Note: build_layout discovers observations and binned_tod_10arcmin under OUT_BASE/<field_id>/ on
scratch (see build_layout.py), matching this script.

Usage:
  module load conda; module load gpu/1.0; conda activate jax
  cd <repo_root>
  python cad/analysis_parallel/run_reconstruction_field.py

Optional argv:
  list              -- print status table for all observations, do not run
  list <obs_id>     -- print one row for that observation only
  <obs_id>          -- only that observation: if incomplete, run run_reconstruction.py once
                       (same as: python cad/analysis_parallel/run_reconstruction.py <field> <obs_id>)

Fastest path for one observation and remaining scans only: call run_reconstruction.py directly;
it skips existing scan_*_ml.npz after rebuild_layout.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CAD_DIR = BASE_DIR.parent
CAD_ROOT = CAD_DIR.parent

OUT_BASE = Path("/pscratch/sd/j/junzhez/cmb-atmosphere-data")
FIELD_ID = "ra0hdec-59.75"
FIELD_SCRATCH_ROOT = OUT_BASE / FIELD_ID


def _count_binned_npz(obs_dir: Path) -> int | None:
    binned = obs_dir / "binned_tod_10arcmin"
    if not binned.is_dir():
        return None
    return sum(1 for _ in binned.glob("*.npz"))


def _count_scan_ml_npz(obs_dir: Path) -> int:
    scans = obs_dir / "scans"
    if not scans.is_dir():
        return 0
    return sum(1 for _ in scans.glob("scan_*_ml.npz"))


def _observation_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(p)
    out.sort(key=lambda x: int(x.name))
    return out


def main() -> None:
    argv = sys.argv[1:]
    list_only = False
    single_obs: str | None = None
    if not argv:
        pass
    elif argv[0] == "list":
        list_only = True
        if len(argv) >= 2 and argv[1].isdigit():
            single_obs = argv[1]
    elif argv[0].isdigit():
        single_obs = argv[0]
    else:
        print(
            "Usage: run_reconstruction_field.py [list [obs_id] | <obs_id>]",
            file=sys.stderr,
        )
        sys.exit(2)

    if single_obs is not None:
        obs_path = FIELD_SCRATCH_ROOT / single_obs
        if not obs_path.is_dir():
            print(f"No observation directory: {obs_path}", flush=True)
            sys.exit(1)
        obs_dirs = [obs_path]
    else:
        obs_dirs = _observation_dirs(FIELD_SCRATCH_ROOT)
    if not obs_dirs:
        print(f"No numeric observation subdirs under {FIELD_SCRATCH_ROOT}", flush=True)
        return

    rows: list[tuple[str, str, int | str, int, str]] = []
    for obs in obs_dirs:
        oid = obs.name
        n_b = _count_binned_npz(obs)
        n_s = _count_scan_ml_npz(obs)
        if n_b is None:
            rows.append((oid, "no_binned", "-", n_s, "skip"))
        elif n_b == 0:
            rows.append((oid, "empty_binned", 0, n_s, "skip"))
        elif n_s == n_b:
            rows.append((oid, "ok", n_b, n_s, "done"))
        else:
            rows.append((oid, "incomplete", n_b, n_s, "run" if not list_only else "would_run"))

    w_obs = max(len(r[0]) for r in rows)
    print(f"field={FIELD_ID} root={FIELD_SCRATCH_ROOT}", flush=True)
    print(
        f"{'obs':>{w_obs}}  {'status':<14}  binned_npz  scan_ml_npz  action",
        flush=True,
    )
    for oid, st, nb, ns, act in rows:
        nb_s = str(nb)
        print(f"{oid:>{w_obs}}  {st:<14}  {nb_s:>10}  {ns:>11}  {act}", flush=True)

    if list_only:
        return

    to_run = [oid for oid, st, _, _, act in rows if act == "run"]
    if not to_run:
        print("Nothing to run.", flush=True)
        return

    recon = BASE_DIR / "run_reconstruction.py"
    for oid in to_run:
        cmd = [sys.executable, str(recon), FIELD_ID, oid]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(CAD_ROOT), check=True)


if __name__ == "__main__":
    main()
