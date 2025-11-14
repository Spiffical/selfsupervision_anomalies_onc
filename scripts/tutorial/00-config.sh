#!/usr/bin/env bash
# Common config for all setup scripts

# Where the data should live INSIDE THE STUDENT PROJECT/COMPUTE SERVER.
DATA_DIR="$HOME/data"

# URL to your dataset/model (publish from your instructor project OR Drive gdown link)
# This can be a Google Drive “uc?export=download&id=...” link or a published CoCalc link.
DATA_URL="https://drive.google.com/drive/folders/1FyKnK__lNM4-LcZnfr48JQZ_gIUK7uZ9?usp=drive_link"

# Repo directory (the repo they just cloned)
REPO_DIR="$HOME/selfsupervision_anomalies_onc"

# Virtual env location
VENV_DIR="$HOME/.venvs/onc-tutorial"

# Jupyter kernel name
KERNEL_NAME="onc-tutorial"
KERNEL_DISPLAY_NAME="ONC Tutorial (PyTorch system)"

# Python to use (the one on the compute server)
PYTHON_BIN="python3"

# If your CFS needs creating:
mkdir -p "$DATA_DIR"
mkdir -p "$(dirname "$VENV_DIR")"
