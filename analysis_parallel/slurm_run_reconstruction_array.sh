#!/bin/bash
#SBATCH --account=des
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=0:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/global/homes/j/junzhez/cmb-atmosphere/cad/analysis_parallel/logs/%x_%A_%a.log
#SBATCH --array=0-71
# One job per scan; 0-71 = 72 scans for FIELD_ID 101706388. Override if layout has different n_scans: --array=0-<n_scans-1>

# Hardcoded paths (edit FIELD_ID for another observation)
SCRIPT_DIR="/global/homes/j/junzhez/cmb-atmosphere/cad/analysis_parallel"
DATA_ROOT="/global/homes/j/junzhez/cmb-atmosphere/cad/data/ra0hdec-59.75"
DATASET_NAME="ra0hdec-59.75"
FIELD_ID="101706388"
OUT_BASE="/pscratch/sd/j/junzhez/cmb-atmosphere-data"
LAYOUT_NPZ="${OUT_BASE}/${DATASET_NAME}/${FIELD_ID}/layout.npz"
SCAN_OUT_DIR="${OUT_BASE}/${DATASET_NAME}/${FIELD_ID}/scans"

# Usage: sbatch --array=0-71 slurm_run_reconstruction_array.sh
# Paths follow OUT_BASE / DATASET_NAME / FIELD_ID (same as build_layout output). Build layout first.

set -e
if [ ! -f "$LAYOUT_NPZ" ]; then
  echo "Layout file not found: $LAYOUT_NPZ" >&2
  exit 1
fi
cd "$SCRIPT_DIR"
if [ -n "${CAD_ENV_SH:-}" ]; then
  source "$CAD_ENV_SH"
fi
module load conda
module load gpu/1.0
conda activate jax
SCAN_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
python /global/homes/j/junzhez/cmb-atmosphere/cad/analysis_parallel/run_reconstruction_single.py "$LAYOUT_NPZ" "$SCAN_OUT_DIR" "$SCAN_INDEX"
