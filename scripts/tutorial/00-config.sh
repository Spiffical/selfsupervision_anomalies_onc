#!/usr/bin/env bash

###############################################################################
# Tutorial configuration: paths + download URLs
###############################################################################

# Root data dir on each compute server (change if you like)
DATA_DIR="/data"

# Local subdirectories
TRAINED_MODELS_DIR="$DATA_DIR/trained_models"
PRETRAIN_DIR="$TRAINED_MODELS_DIR/pretrain"
FINETUNE_DIR="$TRAINED_MODELS_DIR/finetune"
DATASETS_DIR="$DATA_DIR/datasets"

# Base public GCS URL
GCS_BASE="https://storage.googleapis.com/onc-ssl-tutorial-data"

# ---------------------------------------------------------------------------
# Models we actually use in the tutorial
# ---------------------------------------------------------------------------

# GCS subdirs for the chosen runs
FT_RUN_SUBDIR="trained-models/finetune/amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones-noexclude"
PT_RUN_SUBDIR="trained-models/pretrain/amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones_FINAL"

# Finetune model + args (classification model)
FINETUNE_CKPT_URL="$GCS_BASE/$FT_RUN_SUBDIR/models/ft-cls_best_checkpoint.pth"
FINETUNE_ARGS_URL="$GCS_BASE/$FT_RUN_SUBDIR/args.pkl"

# Pretrain model + args (self-supervised backbone)
PRETRAIN_CKPT_URL="$GCS_BASE/$PT_RUN_SUBDIR/models/pretrain-joint_best_checkpoint.pth"
PRETRAIN_ARGS_URL="$GCS_BASE/$PT_RUN_SUBDIR/args.pkl"

# ---------------------------------------------------------------------------
# Datasets (full + small)
# ---------------------------------------------------------------------------

DATASET_FULL_URL="$GCS_BASE/datasets/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
DATASET_SMALL_URL="$GCS_BASE/datasets/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"

# ---------------------------------------------------------------------------
# Repo + venv / kernel settings (as before)
# ---------------------------------------------------------------------------

REPO_DIR="$HOME/selfsupervision_anomalies_onc"
VENV_DIR="$HOME/.venvs/onc-tutorial"
KERNEL_NAME="onc-tutorial"
KERNEL_DISPLAY_NAME="ONC Tutorial (PyTorch system)"
PYTHON_BIN="python3"

# Ensure directories exist
mkdir -p "$TRAINED_MODELS_DIR" "$PRETRAIN_DIR" "$FINETUNE_DIR" "$DATASETS_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
