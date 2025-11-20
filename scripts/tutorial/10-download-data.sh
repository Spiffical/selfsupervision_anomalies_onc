#!/usr/bin/env bash
set -euo pipefail

# ---------------- Logging setup ----------------
LOG_DIR="$HOME"
LOG_FILE="$LOG_DIR/onc-tutorial-download-$(date +'%Y%m%d-%H%M%S').log"

mkdir -p "$LOG_DIR"

# Send all stdout/stderr to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[+] Logging download output to: $LOG_FILE"
echo

# ---------------- Script body ------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for optional data directory argument
if [ "$#" -ge 1 ]; then
    export DATA_DIR="$1"
    echo "[+] Using provided data directory: $DATA_DIR"
fi

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
        echo "[+] $(basename "$outpath") already exists, overwriting..."
    fi

    echo "[+] Downloading $(basename "$outpath") to $outdir ..."
    wget -nv -O "$outpath" "$url"
}

# ---------------------------------------------------------------------------
# Finetune model + args
# ---------------------------------------------------------------------------
download_if_missing "$FINETUNE_CKPT_URL" "$FINETUNE_DIR/ft-cls_best_checkpoint.pth"
download_if_missing "$FINETUNE_ARGS_URL" "$FINETUNE_DIR/args.pkl"

# ---------------------------------------------------------------------------
# Pretrain model + args
# ---------------------------------------------------------------------------
download_if_missing "$PRETRAIN_CKPT_URL" "$PRETRAIN_DIR/pretrain-joint_best_checkpoint.pth"
download_if_missing "$PRETRAIN_ARGS_URL" "$PRETRAIN_DIR/args.pkl"

# ---------------------------------------------------------------------------
# Datasets (full + small + split)
# ---------------------------------------------------------------------------
download_if_missing "$DATASET_FULL_URL"  "$DATASETS_DIR/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
download_if_missing "$DATASET_SMALL_URL" "$DATASETS_DIR/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"
download_if_missing "$DATASET_SPLIT_URL" "$DATASETS_DIR/full_split_seed42.npz"

# ---------------------------------------------------------------------------
# Additional Models
# ---------------------------------------------------------------------------
download_if_missing "$CNN_BASELINE_URL" "$TRAINED_MODELS_DIR/cnn_baseline/cnn_best.pt"
download_if_missing "$SSAMBA_SMALL_FT_URL" "$FINETUNE_DIR/ssamba_finetune_small/ft-avgtok_best_checkpoint.pth"

echo
echo "[+] Final layout under $DATA_DIR:"
find "$DATA_DIR" -maxdepth 3 -type f -printf "    %P (%k KB)\n"

echo
echo "[+] Data download script finished successfully."
echo "[+] Full log saved at: $LOG_FILE"
