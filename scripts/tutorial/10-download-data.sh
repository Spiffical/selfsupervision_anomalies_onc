#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Data download script starting..."
echo "[+] Target data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Use the compute server's Python if not set
: "${PYTHON_BIN:=python3}"

echo "[+] Ensuring gdown is installed for this user..."
# Install gdown to user site-packages
"$PYTHON_BIN" -m pip install --user --no-cache-dir "gdown>=5.0.0" >/dev/null

# Make sure ~/.local/bin is on PATH (where user-level gdown lives)
export PATH="$HOME/.local/bin:$PATH"

if ! command -v gdown >/dev/null 2>&1; then
    echo "[!] ERROR: gdown still not found after installation."
    echo "    Check that ~/.local/bin exists and contains 'gdown'."
    ls -l "$HOME/.local/bin" || true
    exit 1
fi

echo "[+] Using gdown at: $(which gdown)"
echo "[+] Downloading Google Drive folder (contents) from:"
echo "    $DATA_URL"

cd "$DATA_DIR"

# Optional: clean out any prior contents in this directory
echo "[+] Cleaning existing contents of $DATA_DIR"
rm -rf ./*

# Download the folder contents.
# DATA_URL should be a Google Drive *folder* share link, e.g.:
# https://drive.google.com/drive/folders/1VW-MHFmiYba282hey1MAQUyYp4Q1OZVf?usp=sharing
gdown --folder "$DATA_URL" --remaining-ok

echo "[+] Initial download complete. Current structure:"
ls -lah

# If gdown created a single top-level directory (e.g., ./Data),
# move its contents up and remove the extra folder so that
# $DATA_DIR directly contains the files.
subdirs=($(find . -mindepth 1 -maxdepth 1 -type d -printf '%P\n'))

if [ "${#subdirs[@]}" -eq 1 ]; then
    SUBDIR="${subdirs[0]}"
    echo "[+] Detected single subfolder '$SUBDIR'. Flattening into $DATA_DIR..."
    shopt -s dotglob
    mv "$SUBDIR"/* .
    rmdir "$SUBDIR"
    shopt -u dotglob
fi

echo "[+] Final contents of $DATA_DIR:"
ls -lah

echo "[+] Data download script finished successfully."
