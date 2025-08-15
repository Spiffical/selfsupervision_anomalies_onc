#!/usr/bin/env bash
# Linked job submitter that auto-picks GPU flags by cluster and forwards
# named args to your job script.
#
# Usage:
#   ./submit_linked_jobs.sh [TRAINING_DATA_PATH] [WANDB_PROJECT] [WANDB_GROUP] [TRAIN_RATIO] [PROJECT_PATH] [JOB_NAME] [TASK] [EXP_DIR] [VENV_PATH]
#
# Env overrides:
#   NUM_JOBS               - number of linked jobs (default 5)
#   AMBA_GPU               - e.g., "h100:1" to force a specific GPU
#   EXTRA_SBATCH_FLAGS     - extra flags passed to sbatch (e.g., "--partition=gpubase_bygpu_b2")
#
# Notes:
# - If EXTRA_SBATCH_FLAGS includes --gpus/--gres/--partition/-p, we won't add our own GPU flag.

set -euo pipefail

NUM_JOBS=${NUM_JOBS:-5}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/submit_amba_spectrogram.sh"

TRAINING_DATA_PATH=${1:-$HOME/projects/def-kmoran/merileo/ssl_hydrophones/data/h5/different_locations_incl_backgroundpipelinenormals_multilabel.h5}
WANDB_PROJECT=${2:-"amba_spectrogram"}
WANDB_GROUP=${3:-"default_experiment"}
TRAIN_RATIO=${4:-0.8}
PROJECT_PATH=${5:-$HOME/selfsupervision_anomalies_onc}
JOB_NAME=${6:-amba_spectrogram}
TASK=${7:-"pretrain_joint"}
EXP_DIR=${8:-"/exp"}
VENV_PATH=${9:-$HOME/selfsupervision_anomalies_onc/myenv}

mkdir -p out err

detect_cluster() {
  # Prefer SLURM_CLUSTER_NAME if present
  if [[ -n "${SLURM_CLUSTER_NAME:-}" ]]; then
    echo "${SLURM_CLUSTER_NAME,,}"
    return
  fi
  # Fallback: scontrol show config
  local n
  n=$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{gsub(/[[:space:]]/,"",$2); print tolower($2); exit}') || true
  if [[ -n "$n" ]]; then
    echo "$n"
    return
  fi
  # Last resort: hostname heuristic
  if hostname | grep -qi 'fir'; then
    echo "fir"
  else
    echo "unknown"
  fi
}

CLUSTER="$(detect_cluster)"

# Decide GPU flag unless user already supplies one
GPU_FLAG=()
if [[ -n "${AMBA_GPU:-}" ]]; then
  GPU_FLAG=(--gpus="$AMBA_GPU")
else
  case "$CLUSTER" in
    fir) GPU_FLAG=(--gpus=h100:1) ;;  # Fir: H100
    *)   GPU_FLAG=(--gpus=1) ;;       # Generic GPU elsewhere
  esac
fi

# If user provided their own GPU/partition flags, don't add ours
if [[ -n "${EXTRA_SBATCH_FLAGS:-}" ]]; then
  if grep -Eq -- '--gpus=|--gres=|--partition=|-p([[:space:]]|=)' <<<"${EXTRA_SBATCH_FLAGS}"; then
    GPU_FLAG=()
  fi
fi

echo "Cluster detected: ${CLUSTER}"
echo "Using GPU flags: ${GPU_FLAG[*]:-(none; provided by user)}"
[[ -n "${EXTRA_SBATCH_FLAGS:-}" ]] && echo "Extra sbatch flags: ${EXTRA_SBATCH_FLAGS}"

# Helper to submit one job
submit_one() {
  local dep_flag=()
  if [[ $# -ge 1 && -n "${1:-}" ]]; then
    dep_flag=(--dependency=afterany:"$1")
  fi

  sbatch --parsable \
    ${GPU_FLAG+"${GPU_FLAG[@]}"} \
    ${EXTRA_SBATCH_FLAGS:-} \
    "${dep_flag[@]}" \
    --job-name="${JOB_NAME}" \
    --output="out/${JOB_NAME}_job${JOB_INDEX}_%j.out" \
    --error="err/${JOB_NAME}_job${JOB_INDEX}_%j.err" \
    "$JOB_SCRIPT" \
      --dataset "${TRAINING_DATA_PATH}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-group "${WANDB_GROUP}" \
      --train-ratio "${TRAIN_RATIO}" \
      --project-path "${PROJECT_PATH}" \
      --resume "true" \
      --task "${TASK}" \
      --exp-dir "${EXP_DIR}" \
      --venv-path "${VENV_PATH}"
}

# Submit the first job
JOB_INDEX=1
prev_job_id=$(submit_one "")
echo "Submitted job ${JOB_INDEX} with Job ID: ${prev_job_id}"

# Submit dependent jobs
for i in $(seq 2 "${NUM_JOBS}"); do
  JOB_INDEX=$i
  prev_job_id=$(submit_one "${prev_job_id}")
  echo "Submitted job ${JOB_INDEX} with dependency on Job ID: ${prev_job_id}"
done
