#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00-config.sh"

echo "[+] Creating virtualenv at: $VENV_DIR"
$PYTHON_BIN -m venv --system-site-packages "$VENV_DIR"

# activate
source "$VENV_DIR/bin/activate"

# CoCalc sets PIP_CONSTRAINT to include a custom torch build.
# We do NOT want that to interfere with our venv installs.
unset PIP_CONSTRAINT

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Installing numpy<2 (to match your working setup)..."
pip install --no-cache-dir "numpy<2"

echo "[+] Installing base packages..."
pip install --no-cache-dir \
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
    "wandb>=0.15.0" \
    "causal_conv1d>=1.5.0" \
    "onc>=2.3.0" \
    "onc-hydrophone-data>=0.1.0" \
    "dash>=3.3.0" \
    "seaborn>=0.12.0"

echo "[+] Installing labeling app (standalone repo)..."
pip install --no-cache-dir "hydrophone-verification-app @ git+https://github.com/Spiffical/hydrophone-labeling-verification-app.git"

echo "[+] Verifying system torch is visible inside the venv..."
python -c "import torch; print('torch version:', torch.__version__)" || {
    echo '[!] torch is not visible from system-site-packages; aborting.'
    exit 1
}

echo "[+] Installing mamba-ssm without touching torch..."
# --no-build-isolation: use current env (with system torch), don't create a fresh build env
# --no-deps: do not try to (re)install torch or torchaudio as dependencies
pip install --no-cache-dir --no-build-isolation --no-deps mamba-ssm

echo "[+] Installing s3prl pinned, without deps (to avoid torch/torchaudio conflicts)..."
pip install --no-cache-dir --no-deps "s3prl==0.4.15"

echo "[+] Virtualenv ready. Final key package versions:"
python - << 'EOF'
import pkg_resources
for name in ["torch", "torchaudio", "numpy", "mamba-ssm", "s3prl", "timm"]:
    try:
        print(name, pkg_resources.get_distribution(name).version)
    except Exception:
        print(name, "NOT INSTALLED")
EOF
