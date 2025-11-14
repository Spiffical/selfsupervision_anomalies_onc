#!/usr/bin/env bash
# Common config for all setup scripts

# Where the data should live INSIDE THE STUDENT PROJECT/COMPUTE SERVER.
DATA_DIR="$HOME/data"

# Hugging Face direct URLs
FINETUNE_MODEL_URL="https://huggingface.co/merileo/finetune-amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones-noexclude/resolve/main/ft-cls_best_checkpoint.pth"
PRETRAIN_MODEL_URL="https://huggingface.co/merileo/amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones_FINAL/resolve/main/pretrain-joint_best_checkpoint.pth"
DATASET_URL="https://huggingface.co/merileo/different_locations_incl_backgroundpipelinenormals_multilabel/resolve/main/different_locations_incl_backgroundpipelinenormals_multilabel.h5"

# Repo + venv / kernel settings (unchanged)
REPO_DIR="$HOME/selfsupervision_anomalies_onc"
VENV_DIR="$HOME/.venvs/onc-tutorial"
KERNEL_NAME="onc-tutorial"
KERNEL_DISPLAY_NAME="ONC Tutorial (PyTorch system)"
PYTHON_BIN="python3"

mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$VENV_DIR")"