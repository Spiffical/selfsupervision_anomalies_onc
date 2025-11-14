#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Downloading Drive folder into: $DATA_DIR"
cd "$DATA_DIR"

# ensure gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
    echo "[+] Installing gdown..."
    $PYTHON_BIN -m pip install --user gdown
fi

# download entire folder
gdown --folder "$DATA_URL"

echo "[+] Folder download complete."
