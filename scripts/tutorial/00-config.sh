#!/usr/bin/env bash
# Common config for all setup scripts

# Where the data should live INSIDE THE STUDENT PROJECT/COMPUTE SERVER.
DATA_DIR="$HOME/data"

# Subdirectories
TRAINED_MODELS_DIR="$DATA_DIR/trained_models"
PRETRAIN_DIR="$TRAINED_MODELS_DIR/pretrain"
FINETUNE_DIR="$TRAINED_MODELS_DIR/finetune"
DATASETS_DIR="$DATA_DIR/datasets"

# Base HF URL
HF_BASE_URL="https://huggingface.co/merileo/onc-ssl-tutorial/resolve/main"

# Long run-name subdirs on HF
FT_RUN_SUBDIR="finetune/amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones-noexclude"
PT_RUN_SUBDIR="pretrain/amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones_FINAL"

# Finetune model + args
FINETUNE_CKPT_URL="$HF_BASE_URL/$FT_RUN_SUBDIR/models/ft-cls_best_checkpoint.pth"
FINETUNE_ARGS_URL="$HF_BASE_URL/$FT_RUN_SUBDIR/args.pkl"

# Pretrain model + args
PRETRAIN_CKPT_URL="$HF_BASE_URL/$PT_RUN_SUBDIR/models/pretrain-joint_best_checkpoint.pth"
PRETRAIN_ARGS_URL="$HF_BASE_URL/$PT_RUN_SUBDIR/args.pkl"

# Datasets in repo
DATASET_FULL_URL="$HF_BASE_URL/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
DATASET_SMALL_URL="$HF_BASE_URL/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"

# Repo + venv / kernel settings
REPO_DIR="$HOME/selfsupervision_anomalies_onc"
VENV_DIR="$HOME/.venvs/onc-tutorial"
KERNEL_NAME="onc-tutorial"
KERNEL_DISPLAY_NAME="ONC Tutorial (PyTorch system)"
PYTHON_BIN="python3"

mkdir -p "$TRAINED_MODELS_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$DATASETS_DIR"
mkdir -p "$(dirname "$VENV_DIR")"