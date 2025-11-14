#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Creating virtualenv at: $VENV_DIR"
$PYTHON_BIN -m venv --system-site-packages "$VENV_DIR"

# activate
source "$VENV_DIR/bin/activate"

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Downgrading numpy to stay compatible with system PyTorch..."
pip install --no-cache-dir "numpy<2"

echo "[+] Installing tutorial packages..."
pip install --no-cache-dir \
    "mamba-ssm" \
    "causal_conv1d>=1.5.0" \
    "s3prl==0.4.15" \
    "onc>=2.3.0" \
    "h5py>=3.8.0" \
    "matplotlib>=3.7.0" \
    "pandas>=2.0.0" \
    "scikit-learn>=1.0.0" \
    "tqdm>=4.65.0" \
    "scipy>=1.7.3" \
    "soundfile>=0.13.1" \
    "librosa>=0.11.0" \
    "einops==0.8.0" \
    "timm>=0.9.0" \
    "ipython==8.24.0" \
    "python-dotenv>=1.0.0" \
    "wandb>=0.15.0"

echo "[+] Virtualenv ready."
