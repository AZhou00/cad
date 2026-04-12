#!/usr/bin/env python3
"""
Run reconstruction for every incomplete observation under the field scratch root, or list their ids.

Completion rule:
  len(binned_tod_10arcmin/*.npz) == len(scans/scan_*_ml.npz)

Uses OUT_BASE/<field_id>/ on scratch (same tree as build_layout.py). Subdirs without
binned_tod_10arcmin or with empty binned are skipped.

Usage:
  module load conda; module load gpu/1.0; conda activate jax
  cd <repo_root>

  python cad/analysis_parallel/run_reconstruction_field.py
      Print a status table (binned vs scan_ml counts per obs), then run
      run_reconstruction.py once per incomplete observation (sequential).

  python cad/analysis_parallel/run_reconstruction_field.py list-ids
      Print one incomplete observation id per line (stdout). Used by
      nersc_pm_submit_reconstruction_incomplete.sh. No reconstruction.

Single-observation runs: use run_reconstruction.py <field_id> <obs_id> directly.
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


def _status_rows(
    obs_dirs: list[Path],
) -> list[tuple[str, str, int | str, int, str]]:
    """One row per obs: (obs_id, status, binned_npz or '-', scan_ml_npz, action)."""
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
            rows.append((oid, "incomplete", n_b, n_s, "run"))
    return rows


def _print_status_table(rows: list[tuple[str, str, int | str, int, str]]) -> None:
    w_obs = max(len(r[0]) for r in rows)
    print(f"field={FIELD_ID} root={FIELD_SCRATCH_ROOT}", flush=True)
    print(
        f"{'obs':>{w_obs}}  {'status':<14}  binned_npz  scan_ml_npz  action",
        flush=True,
    )
    for oid, st, nb, ns, act in rows:
        nb_s = str(nb)
        print(f"{oid:>{w_obs}}  {st:<14}  {nb_s:>10}  {ns:>11}  {act}", flush=True)
    print(flush=True)


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) > 1:
        print("Usage: run_reconstruction_field.py [list-ids]", file=sys.stderr)
        sys.exit(2)
    list_ids = len(argv) == 1
    if list_ids and argv[0] != "list-ids":
        print("Usage: run_reconstruction_field.py [list-ids]", file=sys.stderr)
        sys.exit(2)

    obs_dirs = _observation_dirs(FIELD_SCRATCH_ROOT)
    if not obs_dirs:
        print(
            f"No numeric observation subdirs under {FIELD_SCRATCH_ROOT}",
            flush=True,
            file=sys.stderr,
        )
        return

    rows = _status_rows(obs_dirs)
    incomplete = [oid for oid, st, _, _, _ in rows if st == "incomplete"]

    if list_ids:
        for oid in incomplete:
            print(oid, flush=True)
        return

    _print_status_table(rows)

    if not incomplete:
        print("Nothing to run.", flush=True)
        return

    recon = BASE_DIR / "run_reconstruction.py"
    for oid in incomplete:
        cmd = [sys.executable, str(recon), FIELD_ID, oid]
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(CAD_ROOT), check=True)


if __name__ == "__main__":
    main()
