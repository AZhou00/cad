#!/bin/bash
# List incomplete observations (field-level file-count check), then submit one Slurm job per obs.
#
# Incomplete = len(binned_tod_10arcmin/*.npz) != len(scans/scan_*_ml.npz) under
#   OUT_BASE/ra0hdec-59.75/<obs_id>/ (see run_reconstruction_field.py).
#
# Each job: 1 Perlmutter GPU node, 4 GPUs, 4 hours (nersc_pm_run_reconstruction_one_obs.sh).
#
# Usage (from repo root on Perlmutter login):
#   module load conda; module load gpu/1.0; conda activate jax
#   ./cad/analysis_parallel/nersc_pm_submit_reconstruction_incomplete.sh
# If you see "Permission denied", run: chmod +x cad/analysis_parallel/nersc_pm_submit_reconstruction_incomplete.sh
# or: bash cad/analysis_parallel/nersc_pm_submit_reconstruction_incomplete.sh
#
# Dry-run (print sbatch lines only):
#   DRY_RUN=1 ./cad/analysis_parallel/nersc_pm_submit_reconstruction_incomplete.sh
#
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

JOB_SCRIPT="${SCRIPT_DIR}/nersc_pm_run_reconstruction_one_obs.sh"

echo "Incomplete observation ids (machine list):" >&2
IDS=$(python cad/analysis_parallel/run_reconstruction_field.py list-ids)
if [[ -z "${IDS}" ]]; then
  echo "(none)" >&2
  exit 0
fi
echo "${IDS}" >&2
echo >&2

while IFS= read -r oid; do
  [[ -z "${oid}" ]] && continue
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "sbatch ${JOB_SCRIPT} ${oid}"
  else
    sbatch "${JOB_SCRIPT}" "${oid}"
  fi
done <<< "${IDS}"
