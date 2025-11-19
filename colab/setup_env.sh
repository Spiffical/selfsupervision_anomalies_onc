#!/usr/bin/env bash
set -euo pipefail

# Colab environment setup wrapper for this repo.
# Delegates to dedicated installers to keep responsibilities separate.
# Usage in Colab:
#   %%bash
#   bash selfsupervision_anomalies_onc/colab/setup_env.sh

echo "[setup_env] Running requirements patcher (optional)..."
python selfsupervision_anomalies_onc/colab/patch_requirements_for_colab.py selfsupervision_anomalies_onc || true

echo "[setup_env] Installing PyTorch stack..."
bash selfsupervision_anomalies_onc/colab/install_torch.sh

cat << 'MSG'
[setup_env] Done installing PyTorch and compatibility fixes.
Now restart the runtime to activate the new environment, then install mamba-ssm:

  1) Restart cell (in notebook): os.kill(os.getpid(), signal.SIGKILL)
  2) Install mamba: %%bash\n     python selfsupervision_anomalies_onc/colab/install_mamba.py

MSG
