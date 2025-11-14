#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Data download script starting..."
echo "[+] Target data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[+] Using wget from: $(command -v wget)"

# Helper to download if missing
download_if_missing () {
    local url="$1"
    local outname="$2"
    if [ -f "$outname" ]; then
        echo "[+] $outname already exists, skipping download."
    else
        echo "[+] Downloading $outname ..."
        wget -O "$outname" "$url"
    fi
}

# 1) Finetuned model
download_if_missing "$FINETUNE_MODEL_URL" "ft-cls_best_checkpoint.pth"

# 2) Pretrained model
download_if_missing "$PRETRAIN_MODEL_URL" "pretrain-joint_best_checkpoint.pth"

# 3) Dataset
download_if_missing "$DATASET_URL" "different_locations_incl_backgroundpipelinenormals_multilabel.h5"

echo "[+] Final contents of $DATA_DIR:"
ls -lah

echo "[+] Data download script finished successfully."
