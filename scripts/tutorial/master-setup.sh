#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== ONC tutorial setup starting ==="

bash "${SCRIPT_DIR}/10-download-data.sh"
bash "${SCRIPT_DIR}/20-create-venv.sh"
bash "${SCRIPT_DIR}/30-register-kernel.sh"

echo "=== ONC tutorial setup finished successfully ==="
