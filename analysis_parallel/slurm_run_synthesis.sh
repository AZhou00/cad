#!/bin/bash
# Run synthesis after all scan artifacts exist. Submit with afterok of array job if desired.
# Usage: sbatch slurm_run_synthesis.sh <layout.npz> <scan_npz_dir> <out_combined.npz>
# Example: sbatch --dependency=afterok:JOBID slurm_run_synthesis.sh layout.npz ./scans recon_combined_ml.npz

set -e
LAYOUT_NPZ="${1:?layout.npz}"
SCAN_NPZ_DIR="${2:?scan_npz_dir}"
OUT_NPZ="${3:?out_combined.npz}"
CAD_ENV="${CAD_ENV_SH:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -n "$CAD_ENV" ]; then
  source "$CAD_ENV"
fi
python run_synthesis.py "$LAYOUT_NPZ" "$SCAN_NPZ_DIR" "$OUT_NPZ"
