#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Data download script starting..."
echo "[+] Root data directory: $DATA_DIR"
echo "[+] Models dir:          $TRAINED_MODELS_DIR"
echo "[+] Pretrain dir:        $PRETRAIN_DIR"
echo "[+] Finetune dir:        $FINETUNE_DIR"
echo "[+] Datasets dir:        $DATASETS_DIR"

mkdir -p "$PRETRAIN_DIR" "$FINETUNE_DIR" "$DATASETS_DIR"

if ! command -v wget >/dev/null 2>&1; then
    echo "[!] ERROR: wget not found."
    exit 1
fi
echo "[+] Using wget from: $(command -v wget)"

download_if_missing () {
    local url="$1"
    local outpath="$2"
    local outdir
    outdir="$(dirname "$outpath")"
    mkdir -p "$outdir"

    if [ -f "$outpath" ]; then
        echo "[+] $(basename "$outpath") already exists in $(dirname "$outpath"), skipping."
    else
        echo "[+] Downloading $(basename "$outpath") to $outdir ..."
        wget -O "$outpath" "$url"
    fi
}

# --- Finetune model + args ---
download_if_missing "$FINETUNE_CKPT_URL" "$FINETUNE_DIR/ft-cls_best_checkpoint.pth"
download_if_missing "$FINETUNE_ARGS_URL" "$FINETUNE_DIR/args.pkl"

# --- Pretrain model + args ---
download_if_missing "$PRETRAIN_CKPT_URL" "$PRETRAIN_DIR/pretrain-joint_best_checkpoint.pth"
download_if_missing "$PRETRAIN_ARGS_URL" "$PRETRAIN_DIR/args.pkl"

# --- Datasets (full + small) ---
download_if_missing "$DATASET_FULL_URL"  "$DATASETS_DIR/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
download_if_missing "$DATASET_SMALL_URL" "$DATASETS_DIR/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"

echo
echo "[+] Final layout under $DATA_DIR:"
find "$DATA_DIR" -maxdepth 3 -type f -printf "    %P (%k KB)\n"

echo "[+] Data download script finished successfully."
