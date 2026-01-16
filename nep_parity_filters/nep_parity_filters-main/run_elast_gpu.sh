#!/bin/bash
#SBATCH --job-name=O-Mo-S_565
#SBATCH --output=j_%x_%A_%a.out
#SBATCH --error=j_%x_%A_%a.err
#SBATCH --account=hennig
#SBATCH --qos=hennig
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem=32gb
#SBATCH --time=24:00:00

###############################################################################
# USAGE:
#   sbatch --array=0-3 run_elast_gpu.sh
###############################################################################

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_TASKS=${SLURM_ARRAY_TASK_COUNT}
HOST=$(hostname -s)
START_TS=$(date +"%Y-%m-%d %H:%M:%S")

echo "=================================================="
echo "Array task   : ${TASK_ID} / $((N_TASKS-1))"
echo "Host         : ${HOST}"
echo "Start time   : ${START_TS}"
echo "Working dir  : $(pwd)"
echo "=================================================="
echo ""

# =========================
# MODULES
# =========================
module purge
module load nvhpc/25.3 openmpi/5.0.7 mkl/2025.1.0 fftw/3.3.10
export LD_LIBRARY_PATH=${HPC_NVHPC_DIR}/compilers/extras/qd/lib:${LD_LIBRARY_PATH}
export OMPI_MCA_pml=ucx
export UCX_TLS=^ib

VASP_GPU=/blue/ypchen/ntaormina/bin/vasp/vasp_std_gpu

# =========================
# COLLECT *ALL* STRAIN DIRS
# =========================
mapfile -t ALL_DIRS < <(
  find . -mindepth 2 -maxdepth 2 -type d -name "strain_*" | sort
)

TOTAL=${#ALL_DIRS[@]}

if (( TOTAL == 0 )); then
  echo "No strain_* directories found. Exiting."
  exit 1
fi

# =========================
# STRIDE ASSIGNMENT
# =========================
MY_DIRS=()
for (( i=TASK_ID; i<TOTAL; i+=N_TASKS )); do
  MY_DIRS+=( "${ALL_DIRS[$i]}" )
done

MY_TOTAL=${#MY_DIRS[@]}

echo "Total strain directories     : ${TOTAL}"
echo "Directories assigned to task : ${MY_TOTAL}"
echo "Assigned list:"
printf "  %s\n" "${MY_DIRS[@]}"
echo ""

# =========================
# PROGRESS LOG
# =========================
PROGRESS_LOG="progress_task_${TASK_ID}.log"
echo "TASK ${TASK_ID} START @ ${START_TS} on ${HOST}" > "${PROGRESS_LOG}"

# =========================
# MAIN LOOP
# =========================
COUNT=0
for subdir in "${MY_DIRS[@]}"; do
  COUNT=$((COUNT + 1))
  STEP_START=$(date +"%Y-%m-%d %H:%M:%S")

  echo ""
  echo "--------------------------------------------------"
  echo "[Task ${TASK_ID}] (${COUNT}/${MY_TOTAL})"
  echo "Directory : ${subdir}"
  echo "Start     : ${STEP_START}"
  echo "--------------------------------------------------"

  pushd "$subdir" >/dev/null

  # -------- SKIP COMPLETED RUNS --------
  if [[ -f ".DONE" ]]; then
    echo "[Task ${TASK_ID}] SKIP (already DONE): ${subdir}"
    echo "[${STEP_START}] SKIP ${subdir}" >> "${PROGRESS_LOG}"
    popd >/dev/null
    continue
  fi

  echo "[${STEP_START}] START ${subdir}" >> "${PROGRESS_LOG}"

  # GPU sanity check
  nvidia-smi 1>&2

  # Run VASP
  srun $VASP_GPU 2>&1 | tee log.vasp

  STEP_END=$(date +"%Y-%m-%d %H:%M:%S")

  # -------- SUCCESS CHECK --------
  if [[ -f OUTCAR ]] && grep -q "Voluntary context switches" OUTCAR; then
    touch .DONE
    echo "[${STEP_END}] DONE  ${subdir}" >> "${PROGRESS_LOG}"
  else
    echo "[${STEP_END}] FAIL  ${subdir}" >> "${PROGRESS_LOG}"
  fi

  popd >/dev/null
done

END_TS=$(date +"%Y-%m-%d %H:%M:%S")

echo ""
echo "=================================================="
echo "Task ${TASK_ID} COMPLETE"
echo "End time : ${END_TS}"
echo "=================================================="

echo "TASK ${TASK_ID} END @ ${END_TS}" >> "${PROGRESS_LOG}"
