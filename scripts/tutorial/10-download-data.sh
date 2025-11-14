#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Ensuring gdown is installed..."

# Install gdown using the compute server's python and user site-packages
$PYTHON_BIN -m pip install --user --no-cache-dir gdown

# Ensure ~/.local/bin is in PATH for this script (CoCalc doesn't always add it)
export PATH="$HOME/.local/bin:$PATH"

if ! command -v gdown >/dev/null 2>&1; then
    echo "[!] gdown still not found. Checking installation..."
    ls ~/.local/bin
    echo "[!] ERROR: gdown not available even after installation."
    exit 1
fi

echo "[+] gdown located at: $(which gdown)"
echo "[+] Downloading Google Drive folder…"

cd "$DATA_DIR"

gdown --folder "$DATA_URL"

echo "[+] Download complete."
