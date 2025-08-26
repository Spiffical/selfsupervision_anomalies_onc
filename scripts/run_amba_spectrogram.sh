#!/bin/bash
set -euo pipefail
# Debug: print failing line on any error
trap 'echo "[ERROR] Bash exited at line $LINENO" >&2' ERR
# Enable xtrace early to trace commands
PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '
set -x

# -------------------------------
# Cluster detection (DRAC vs local)
# -------------------------------
detect_cluster() {
  if [[ -n "${SLURM_CLUSTER_NAME:-}" ]]; then
    echo "${SLURM_CLUSTER_NAME,,}"
    return
  fi
  local n
  n=$(scontrol show config 2>/dev/null | awk -F= '/^ClusterName/{gsub(/[[:space:]]/,"",$2); print tolower($2); exit}') || true
  [[ -n "$n" ]] && { echo "$n"; return; }
  if hostname | egrep -qi 'cedar|graham|narval|beluga|fir'; then
    echo "drac"
  else
    echo "local"
  fi
}

CLUSTER="$(detect_cluster)"
IS_DRAC=false
[[ "$CLUSTER" != "local" ]] && IS_DRAC=true

# -------------------------------
# Defaults & CLI parsing
# -------------------------------
PYTHON_SCRIPT=""
DATA_TRAIN_PATH=""
WANDB_PROJECT="amba_spectrogram"
WANDB_GROUP="default_experiment"
TRAIN_RATIO=0.8
RESUME="true"              # default: resume if checkpoint exists
EXP_DIR="/exp"
TASK="pretrain_joint"
MULTICLASS="false"
NUM_CLASSES=""
WANDB_ENTITY=""
declare -a EXCLUDE_LABELS=()
PRETRAINED_PATH=""
DRY_RUN="false"

# Virtualenv (optional). If not set and not already in a venv, we will:
# - on DRAC: default to $HOME/selfsupervision_anomalies_onc/myenv
# - local: try ./.venv or ./venv
VENV_PATH="${VENV_PATH:-}"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python-script)     PYTHON_SCRIPT="$2"; shift 2;;
    --dataset)           DATA_TRAIN_PATH="$2"; shift 2;;
    --wandb-project)     WANDB_PROJECT="$2"; shift 2;;
    --wandb-group)       WANDB_GROUP="$2"; shift 2;;
    --train-ratio)       TRAIN_RATIO="$2"; shift 2;;
    --resume)            if [[ "${2,,}" == "false" ]]; then RESUME="false"; fi; shift 2;;
    --exp-dir)           EXP_DIR="$2"; shift 2;;
    --task)              TASK="$2"; shift 2;;
    --wandb-entity)      WANDB_ENTITY="$2"; shift 2;;
    --exclude-label)     EXCLUDE_LABELS+=("$2"); shift 2;;
    --pretrained-path)   PRETRAINED_PATH="$2"; shift 2;;
    --dry-run)           DRY_RUN="true"; shift;;
    --multiclass)        MULTICLASS="true"; shift;;
    --num-classes|--num_classes)
                         NUM_CLASSES="$2"; shift 2;;
    --venv|--venv-path)  VENV_PATH="$2"; shift 2;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# -------------------------------
# Validation
# -------------------------------
if [ -z "$PYTHON_SCRIPT" ]; then
  echo "Error: --python-script is required"
  exit 1
fi
if [ -z "$DATA_TRAIN_PATH" ]; then
  echo "Error: --dataset is required"
  exit 1
fi

# -------------------------------
# Environment setup (modules/venv)
# Only activate if we're not already in a venv
# -------------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if $IS_DRAC; then
    echo "🔧 Detected DRAC cluster environment"
    # Load python module if available
    if type module &>/dev/null; then
      module load python/3.10
    fi
    # Choose venv path: CLI > env > default
    VENV_PATH="${VENV_PATH:-$HOME/selfsupervision_anomalies_onc/myenv}"
    if [[ -f "$VENV_PATH/bin/activate" ]]; then
      echo "Activating venv: $VENV_PATH"
      # shellcheck disable=SC1090
      source "$VENV_PATH/bin/activate"
    else
      echo "Warning: venv not found at $VENV_PATH/bin/activate (continuing with current Python)"
    fi

    # DRAC-specific env
    export TORCH_HOME="${TORCH_HOME:-../../pretrained_models}"
    export PYTHONPATH="${PYTHONPATH:-}:$HOME/selfsupervision_anomalies_onc"
    export PYTHONPATH="$PYTHONPATH:$SCRATCH/ssamba_project/src"
    export PYTHONPATH="$PYTHONPATH:$SLURM_TMPDIR/ssamba_project/src"

    # Load .env from DRAC home if present
    if [ -f "$HOME/selfsupervision_anomalies_onc/.env" ]; then
      # shellcheck disable=SC2046
      export $(grep -v '^#' "$HOME/selfsupervision_anomalies_onc/.env" | xargs)
    fi
  else
    echo "💻 Detected local environment"
    # Local venv logic
    if [[ -n "$VENV_PATH" && -f "$VENV_PATH/bin/activate" ]]; then
      echo "Activating venv: $VENV_PATH"
      # shellcheck disable=SC1090
      source "$VENV_PATH/bin/activate"
    elif [[ -f ".venv/bin/activate" ]]; then
      echo "Activating venv: .venv"
      # shellcheck disable=SC1091
      source ".venv/bin/activate"
    elif [[ -f "venv/bin/activate" ]]; then
      echo "Activating venv: ./venv"
      # shellcheck disable=SC1091
      source "venv/bin/activate"
    else
      echo "Note: no virtual environment found/activated (continuing with current Python)"
    fi

    # Load .env from project root if present
    if [ -f .env ]; then
      # shellcheck disable=SC2046
      export $(grep -v '^#' .env | xargs)
    fi
  fi
else
  echo "✅ A virtual environment is already active: $VIRTUAL_ENV (leaving it as-is)"
fi

# -------------------------------
# Fixed parts of experiment naming
# -------------------------------
folder_mask_patch=300
folder_batch_size=16
folder_lr=1e-4
folder_fstride=16
folder_tstride=16

# -------------------------------
# Task-specific training params
# -------------------------------
echo "Excluded labels: ${EXCLUDE_LABELS[*]}"
echo "[MARK] TASK=$TASK, EXP_DIR=$EXP_DIR, PRETRAINED_PATH=$PRETRAINED_PATH, TRAIN_RATIO=$TRAIN_RATIO"

if [[ $TASK == *"pretrain"* ]]; then
  # Pretraining parameters
  mask_patch=300
  batch_size=16
  lr=1e-4
  lr_patience=2
  epoch=200
  freqm=0
  timem=0
  mixup=0
  bal=none
  fstride=16
  tstride=16
  main_metric="acc"
else
  # Finetuning parameters
  mask_patch=0
  batch_size=16
  lr=5e-5
  lr_patience=3
  epoch=200
  freqm=48
  timem=192
  mixup=0.5
  bal=balanced
  fstride=10
  tstride=10
  main_metric="auc"
fi

# -------------------------------
# Dataset/model parameters
# -------------------------------
dataset=custom
dataset_mean=51.506817
dataset_std=13.638703
target_length=512
num_mel_bins=512

train_ratio=$TRAIN_RATIO
val_ratio=0.1
split_seed=42

model_size=base
patch_size=16
embed_dim=768
depth=24

fshape=16
tshape=16

rms_norm='false'
residual_in_fp32='false'
fused_add_norm='false'
if_rope='false'
if_rope_residual='false'
bimamba_type="v2"
drop_path_rate=0.1
stride=16
channels=1
drop_rate=0.
norm_epsilon=1e-5
if_bidirectional='true'
final_pool_type='none'
if_abs_pos_embed='true'
if_bimamba='false'
if_cls_token='true'
if_devide_out='true'
use_double_cls_token='false'
use_middle_cls_token='false'

# -------------------------------
# Experiment directory naming
# -------------------------------
exclude_labels_str=""
if (( ${#EXCLUDE_LABELS[@]} > 0 )); then
  labels_joined=""
  for label in "${EXCLUDE_LABELS[@]}"; do
    if [ -z "$labels_joined" ]; then
      labels_joined="${label// /_}"
    else
      labels_joined="${labels_joined}_${label// /_}"
    fi
  done
  exclude_labels_str="-excl${labels_joined}"
fi

base_folder=amba-${model_size}-f${fshape}-t${tshape}-b${folder_batch_size}-lr${folder_lr}-m${folder_mask_patch}-custom-tr$(printf "%.1f" ${TRAIN_RATIO})-${WANDB_GROUP}${exclude_labels_str}
echo "Base folder: $base_folder"

# Compute and create experiment dirs
if [[ $TASK == *"pretrain"* ]]; then
  exp_dir=${EXP_DIR}/pretrain/${base_folder}
else
  exp_dir=${EXP_DIR}/finetune/${base_folder}
fi
echo "[MARK] Creating exp_dir: $exp_dir"
mkdir -p "${exp_dir}/models" || { echo "[FATAL] mkdir failed for $exp_dir"; exit 91; }
ls -ld "$exp_dir" "$exp_dir/models" || true

# -------------------------------
# Build Python command (avoid set -e pitfalls from command substitution)
# -------------------------------
PY_APPEND_PRETRAINED=""
if [ -n "$PRETRAINED_PATH" ]; then
  PY_APPEND_PRETRAINED=" --pretrained_path \"$PRETRAINED_PATH\""
fi
PY_APPEND_NUM_CLASSES=""
if [ -n "$NUM_CLASSES" ]; then
  PY_APPEND_NUM_CLASSES=" --num_classes $NUM_CLASSES"
fi

PYTHON_CMD="python -u -W ignore \"$PYTHON_SCRIPT\" --use_wandb --wandb_entity \"${WANDB_ENTITY:-spencer-bialek}\" \
--wandb_project ${WANDB_PROJECT} \
--wandb_group ${WANDB_GROUP} \
--dataset custom \
--data-train \"$DATA_TRAIN_PATH\" \
--exp-dir \"$exp_dir\"${PY_APPEND_PRETRAINED} \
--dataset_mean ${dataset_mean} \
--dataset_std ${dataset_std} \
--train_ratio ${train_ratio} \
--val_ratio ${val_ratio} \
--split_seed ${split_seed} \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model False \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--tstride $tstride --fstride $fstride --fshape ${fshape} --tshape ${tshape} \
--target_length ${target_length} --num_mel_bins ${num_mel_bins} \
--model_size ${model_size} --mask_patch ${mask_patch} --n-print-steps 100 \
--task ${TASK} --lr_patience ${lr_patience} --epoch_iter 1 \
--patch_size ${patch_size} --embed_dim ${embed_dim} --depth ${depth} \
--rms_norm ${rms_norm} --residual_in_fp32 ${residual_in_fp32} \
--fused_add_norm ${fused_add_norm} --if_rope ${if_rope} --if_rope_residual ${if_rope_residual} \
--bimamba_type ${bimamba_type} --use_middle_cls_token ${use_middle_cls_token} \
--drop_path_rate ${drop_path_rate} --stride ${stride} --channels ${channels}${PY_APPEND_NUM_CLASSES} \
--drop_rate ${drop_rate} --norm_epsilon ${norm_epsilon} \
--if_bidirectional ${if_bidirectional} --final_pool_type ${final_pool_type} \
--if_abs_pos_embed ${if_abs_pos_embed} --if_bimamba ${if_bimamba} \
--if_cls_token ${if_cls_token} --if_devide_out ${if_devide_out} \
--use_double_cls_token ${use_double_cls_token} --use_middle_cls_token ${use_middle_cls_token} \
--main_metric ${main_metric}"

echo "[MARK] Built PYTHON_CMD (len=${#PYTHON_CMD})"

# Append exclude labels as a multi-value flag
if [ ${#EXCLUDE_LABELS[@]} -gt 0 ]; then
  PYTHON_CMD+=" --exclude_labels"
  for label in "${EXCLUDE_LABELS[@]}"; do
    PYTHON_CMD+=" \"${label}\""
  done
fi

# Resume flag
if [ "$RESUME" != "false" ]; then
  PYTHON_CMD+=" --resume"
fi

# Multiclass flag
if [ "$MULTICLASS" = "true" ]; then
  PYTHON_CMD+=" --multiclass"
fi

# -------------------------------
# Execute
# -------------------------------
echo "Python command that will be executed:"
echo "$PYTHON_CMD"
echo

if [ "$DRY_RUN" = "true" ]; then
  echo "Dry run completed. Exiting without executing."
  exit 0
fi

# Extra debug on DRAC
$IS_DRAC && set -x

# Pre-flight: show Python and torch versions, confirm data script path
python --version || true
python -c 'import torch, sys; print("torch:", torch.__version__)' || true
echo "Exists PYTHON_SCRIPT? $( [ -f "$PYTHON_SCRIPT" ] && echo yes || echo no ) at $PYTHON_SCRIPT"
echo "Exists DATA? $( [ -f "$DATA_TRAIN_PATH" ] && echo yes || echo no ) at $DATA_TRAIN_PATH"
echo "which python: $(which python)"
echo "PYTHONPATH=${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

# Execute via a temporary script to preserve exact quoting and capture stderr
echo "$PYTHON_CMD" > run_cmd.sh
chmod +x run_cmd.sh
echo "[MARK] Executing run_cmd.sh under bash -x"
bash -x ./run_cmd.sh 2>&1 || true
status=$?
echo "[MARK] Python exit status: $status"
exit $status
