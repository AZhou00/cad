#!/bin/bash
# Perlmutter: margined synthesis on all reconstruction-complete observations (see run_synthesis_margined.py).
# 1 node, 4 GPUs, 4 hours. No script arguments; Python discovers obs ids on scratch.
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

export SLURM_CPU_BIND="cores"

set -euo pipefail

module load conda
module load gpu/1.0
conda activate jax

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

echo "JOBID=${SLURM_JOB_ID:-local} run_synthesis_margined.py (all processed obs)" >&2
srun python cad/analysis_parallel/run_synthesis_margined.py
