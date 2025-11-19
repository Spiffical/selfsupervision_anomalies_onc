#!/usr/bin/env bash
set -euo pipefail

# Install a pinned PyTorch stack suitable for Colab GPU sessions.
# Default pins target CUDA 12.1 wheels; override via env vars if desired.

TORCH_VER="${TORCH_VER:-2.4.1+cu121}"
TVISION_VER="${TVISION_VER:-0.19.1+cu121}"
TAUDIO_VER="${TAUDIO_VER:-2.4.1+cu121}"

echo "[install_torch] Upgrading pip..."
python -m pip install --upgrade pip -q

echo "[install_torch] Uninstalling conflicting packages (torch/vision/audio/xformers/triton)..."
python -m pip uninstall -y torch torchvision torchaudio xformers triton >/dev/null 2>&1 || true

echo "[install_torch] Installing PyTorch stack: torch=${TORCH_VER} torchvision=${TVISION_VER} torchaudio=${TAUDIO_VER}"
python -m pip install -q --index-url https://download.pytorch.org/whl/cu121 \
  "torch==${TORCH_VER}" "torchvision==${TVISION_VER}" "torchaudio==${TAUDIO_VER}"

echo "[install_torch] Applying protobuf/s3prl compatibility fix..."
python -m pip install -q "protobuf>=5.29.1,<7"
python -m pip install -q --no-deps "s3prl==0.4.15"

python - << 'PY'
import sys, pkg_resources
try:
    import torch
    tv = getattr(torch, '__version__', 'unknown')
    cv = getattr(torch.version, 'cuda', None)
    ca = torch.cuda.is_available()
except Exception as e:
    torch, tv, cv, ca = None, f'failed: {e}', None, False

def ver(name):
    try:
        return pkg_resources.get_distribution(name).version
    except Exception:
        return 'not-installed'

print('[install_torch] Versions:')
print('  torch      :', tv)
print('  torch.cuda :', cv)
print('  cuda avail :', ca)
print('  protobuf   :', ver('protobuf'))
print('  s3prl      :', ver('s3prl'))
print('  python     :', sys.version)
PY

cat << 'MSG'
[install_torch] Done. Now restart the runtime to activate the new torch stack.
Use the notebook cell that calls:

    import os, signal; os.kill(os.getpid(), signal.SIGKILL)

MSG

