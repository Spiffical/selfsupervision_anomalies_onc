#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Activating venv to register kernel..."
source "$VENV_DIR/bin/activate"

python -m ipykernel install --user \
    --name="$KERNEL_NAME" \
    --display-name="$KERNEL_DISPLAY_NAME"

echo "[+] Kernel '$KERNEL_DISPLAY_NAME' installed for this user."
