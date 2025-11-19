#!/usr/bin/env bash
set -e

# Configuration
BUCKET="gs://onc-ssl-tutorial-data"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Local source files
LOCAL_SMALL_H5="$REPO_DIR/data/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"
LOCAL_SPLIT="$REPO_DIR/data/full_split_seed42.npz"
LOCAL_CNN="$REPO_DIR/cnn_experiments/cnn_best.pt"
LOCAL_SSAMBA="$REPO_DIR/ssamba_experiments_small/models/ft-avgtok_best_checkpoint.pth"

# Remote targets
REMOTE_SMALL_H5="$BUCKET/datasets/different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"
REMOTE_SPLIT="$BUCKET/datasets/full_split_seed42.npz"
REMOTE_CNN="$BUCKET/trained-models/cnn_baseline/cnn_best.pt"
REMOTE_SSAMBA="$BUCKET/trained-models/finetune/ssamba_finetune_small/ft-avgtok_best_checkpoint.pth"

echo "[+] Checking for gsutil..."
if ! command -v gsutil >/dev/null 2>&1; then
    echo "[!] gsutil not found. Please install Google Cloud SDK."
    exit 1
fi

echo "[+] Listing current bucket contents..."
gsutil ls -r "$BUCKET" || echo "[!] Could not list bucket. Check permissions."

upload_file() {
    local local_path="$1"
    local remote_path="$2"

    if [ -f "$local_path" ]; then
        echo "[+] Uploading $local_path to $remote_path..."
        gsutil cp "$local_path" "$remote_path"
    else
        echo "[!] Local file not found: $local_path"
    fi
}

echo
echo "[+] Starting uploads..."

# 1. Small Dataset
upload_file "$LOCAL_SMALL_H5" "$REMOTE_SMALL_H5"

# 2. Seed Split
upload_file "$LOCAL_SPLIT" "$REMOTE_SPLIT"

# 3. CNN Baseline
upload_file "$LOCAL_CNN" "$REMOTE_CNN"

# 4. SSAMBA Small Finetune
upload_file "$LOCAL_SSAMBA" "$REMOTE_SSAMBA"

echo
echo "[+] Upload process finished."
