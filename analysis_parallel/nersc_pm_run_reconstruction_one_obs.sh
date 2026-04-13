#!/bin/bash
# Perlmutter: one observation, one node, 4 GPUs, 4 hours.
# Usage (from repo root, after sbatch copies this script):
#   sbatch cad/analysis_parallel/nersc_pm_run_reconstruction_one_obs.sh <observation_id>
# The observation id is argument $1 to this script (standard Slurm behavior).
#
# Runs: python cad/analysis_parallel/run_reconstruction.py <FIELD_ID> <obs_id>
# See run_reconstruction.py for pipeline steps (build_layout -> missing scan_*_ml only).
#
#SBATCH --account=des
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --job-name=cmb-recon-one
#SBATCH --output=cmb-recon-one-%j.out
#SBATCH --error=cmb-recon-one-%j.err

# sbatch passes arguments after the script path as $1, $2, ... to the batch script at run time.
# Fallback: OBS_ID=... sbatch --export=ALL,OBS_ID (or include in submit wrapper).
FIELD_ID="ra0hdec-59.75"
OBS_ID="${1:-${OBS_ID:-}}"
if [[ -z "${OBS_ID}" ]]; then
  echo "Missing observation id. Submit as: sbatch ${0} <observation_id>" >&2
  exit 1
fi

module load conda
module load gpu/1.0
conda activate jax

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

echo "OBS_ID=${OBS_ID} FIELD_ID=${FIELD_ID} JOBID=${SLURM_JOB_ID:-local}" >&2
srun python cad/analysis_parallel/run_reconstruction.py "${FIELD_ID}" "${OBS_ID}"
