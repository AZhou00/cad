#!/bin/bash
# Perlmutter: margined synthesis via run_synthesis_margined.py (no args).
# Python discovers all reconstruction-complete observations under the field root and runs multi-obs
# synthesis on a union plate grid (remap per layout.npz; common pixel_size_deg); see discover_synthesis_ready_observation_ids.
# 1 node, 4 GPUs, 4 hours.
#
# After recon jobs (example):
#   sbatch --dependency=afterok:JOB1,JOB2 cad/analysis_parallel/nersc_pm_run_synthesis_margined.sh
#
#SBATCH --account=des
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --job-name=cmb-syn-marg
#SBATCH --output=cmb-syn-marg-%j.out
#SBATCH --error=cmb-syn-marg-%j.err

module load conda
module load gpu/1.0
conda activate jax

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

echo "JOBID=${SLURM_JOB_ID:-local} run_synthesis_margined.py (all discovered obs, union plate)" >&2
srun python cad/analysis_parallel/run_synthesis_margined.py
