#!/bin/bash
set -euo pipefail

echo "🧩 DRAC installer (cluster-safe)"

# Expect modules to be preloaded by user (don’t change site env silently)
# Recommended:
# module --force purge
# module load StdEnv/2023 python/3.10 gcc/12.3 cuda/12.2 cudnn/8.9.5.29

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "❌ Please activate your venv first (e.g., source .env_drac/bin/activate)"
  exit 1
fi

echo "📦 Installing base deps (Rust-free constraints)…"
pip install -r requirements-base.txt -c drac/constraints-drac.txt

echo "🧱 Locking PyTorch trio to 2.6.0/2.6.0/0.21.0…"
pip install --no-deps --force-reinstall \
  "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0"

echo "🐍 Installing mamba-related packages (no dependency resolver)…"
pip install --no-deps "mamba_ssm==2.2.4" "causal_conv1d>=1.5.0" "s3prl==0.4.15"

echo "🌊 ONC utilities…"
pip install "onc>=2.3.0"

echo "🔧 Editable install of this repo…"
pip install .

echo "✅ Done. Reminder: import mamba_ssm only on GPU nodes."
