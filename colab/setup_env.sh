#!/usr/bin/env bash
set -euo pipefail

# Colab environment setup for this repo.
# - Pins Torch/TorchVision/Torchaudio to a CUDA 12.1 wheel combo
# - Applies protobuf/s3prl compatibility fix
#
# Usage in Colab:
#   %%bash
#   bash selfsupervision_anomalies_onc/colab/setup_env.sh

TORCH_VER="${TORCH_VER:-2.4.1+cu121}"
TVISION_VER="${TVISION_VER:-0.19.1+cu121}"
TAUDIO_VER="${TAUDIO_VER:-2.4.1+cu121}"

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Uninstalling conflicting packages (if present)..."
python -m pip uninstall -y torch torchvision torchaudio xformers triton || true

echo "Installing pinned Torch stack: torch=${TORCH_VER} torchvision=${TVISION_VER} torchaudio=${TAUDIO_VER}"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==${TORCH_VER}" "torchvision==${TVISION_VER}" "torchaudio==${TAUDIO_VER}"

echo "Applying protobuf / s3prl fix for Colab..."
python -m pip install "protobuf>=5.29.1,<7"
python -m pip install --no-deps "s3prl==0.4.15"

python - << 'PY'
import sys, pkg_resources
try:
    import torch
except Exception as e:
    torch = None
    print('Torch import failed:', e)

def ver(name):
    try:
        return pkg_resources.get_distribution(name).version
    except Exception:
        return 'not-installed'

print('google-colab:', ver('google-colab'))
print('protobuf    :', ver('protobuf'))
print('s3prl       :', ver('s3prl'))
if torch is not None:
    print('Torch       :', torch.__version__)
    print('CUDA ver    :', getattr(torch.version, 'cuda', None))
    print('CUDA avail  :', torch.cuda.is_available())
print('Python      :', sys.version)
PY

cat << 'MSG'

============================================================
Environment setup complete. Next step (in a Python cell):

    import os, signal; os.kill(os.getpid(), signal.SIGKILL)

This restarts the Colab runtime so the new Torch stack is active.
============================================================
MSG

