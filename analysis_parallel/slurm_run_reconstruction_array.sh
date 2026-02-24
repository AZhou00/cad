#!/bin/bash
# SLURM array job: one task per scan. Build global layout first, then submit this script.
# Usage: sbatch --array=0-N slurm_run_reconstruction_array.sh <layout.npz> <out_dir>
# Optional env: CAD_ENV_SH path to source before run (e.g. env.sh with conda activate).

set -e
LAYOUT_NPZ="${1:?layout.npz}"
OUT_DIR="${2:?out_dir}"
CAD_ENV="${CAD_ENV_SH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -n "$CAD_ENV" ]; then
  source "$CAD_ENV"
fi
SCAN_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
python run_reconstruction_single.py "$LAYOUT_NPZ" "$OUT_DIR" "$SCAN_INDEX"
