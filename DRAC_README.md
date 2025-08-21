# DRAC Cluster Usage Guide

This guide explains how to run the Self-Supervised Anomaly Detection project on the Digital Research Alliance of Canada (DRAC) clusters (e.g., Fir/Cedar/Graham/Narval/Beluga).

> **Key rules (TL;DR)**
> - Use a single, consistent CUDA stack: **StdEnv/2023 + python/3.10 + gcc/12.3 + cuda/12.2 + cudnn/8.9.5.29**.
> - Keep the PyTorch trio pinned: **torch 2.6.0, torchaudio 2.6.0, torchvision 0.21.0**.
> - Install `mamba_ssm`, `causal_conv1d`, and `s3prl` **with `--no-deps`** so pip doesn’t downgrade Torch.
> - Import `mamba_ssm` **only on GPU nodes** (Triton needs an active CUDA backend at import time).

---

## Prerequisites

- DRAC account and allocations.
- Familiarity with SLURM.
- Access to a GPU partition on your site (A100/H100/V100 etc.).

---

## Preferred: one-shot install script

1) Load modules (keep this stack)
```bash
module --force purge
module load StdEnv/2023 python/3.10 gcc/12.3 cuda/12.2 cudnn/8.9.5.29
````

2. Create & activate a virtual environment

```bash
python -m venv .env_drac
source .env_drac/bin/activate
```

3. Run the installer

```bash
bash drac/scripts/install_deps_drac.sh
```

> The script does:
>
> * Base deps with Rust-free constraints (`-c drac/constraints-drac.txt`)
> * Locks `torch/torchaudio/torchvision` to 2.6.0/2.6.0/0.21.0
> * Installs `mamba_ssm`, `causal_conv1d`, `s3prl` with `--no-deps`
> * Installs `onc` and this repo

**Version check (quick):**

```bash
python -c "import torch, torchaudio, torchvision; \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('torchaudio', torchaudio.__version__); \
print('torchvision', torchvision.__version__); \
print('cuda_available', torch.cuda.is_available())"
```

---

<details>
<summary><strong>Manual install (if you can’t run the script)</strong></summary>

### 1) Load modules

```bash
module --force purge
module load StdEnv/2023 python/3.10 gcc/12.3 cuda/12.2 cudnn/8.9.5.29
```

### 2) Virtual environment

```bash
python -m venv .env_drac
source .env_drac/bin/activate
```

### 3) Base dependencies (avoid Rust builds)

```bash
pip install -r requirements-base.txt -c drac/constraints-drac.txt
```

### 4) Lock PyTorch to 2.6.0 trio

```bash
pip install --no-deps --force-reinstall \
  "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0"
```

### 5) Mamba-related packages (no dependency resolver)

```bash
pip install --no-deps "mamba_ssm==2.2.4" "causal_conv1d>=1.5.0" "s3prl==0.4.15"
pip install "onc>=2.3.0"
```

### 6) Install this repo

```bash
pip install .
```

**Note:** The Compute Canada `mamba_ssm` wheel declares a dependency on `torch~=2.5.0`. We stay on 2.6.0; pip may warn, but runtime is typically fine. If you do hit runtime issues, see **Alternative Torch stack (2.5.1)** below.

</details>

---

## Quick GPU Sanity Check

Run this **inside a GPU allocation** (see “Interactive GPU” below):

```bash
python - <<'PY'
import torch, triton
from triton.runtime import driver
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("triton backend:", getattr(driver.active, "name", "NONE"))
# Import mamba_ssm only on GPU nodes
import mamba_ssm
print("mamba_ssm import OK")
PY
```

Expected:

* `cuda_available: True`
* `triton backend: cuda`
* `mamba_ssm import OK`

---

## Running Jobs

### Interactive GPU (debugging)

```bash
salloc --time=02:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=24G
module --force purge
module load StdEnv/2023 python/3.10 gcc/12.3 cuda/12.2 cudnn/8.9.5.29
source ~/selfsupervision_anomalies_onc/.env_drac/bin/activate
```

### Linked jobs / multi-ratio / supervised (examples)

**Single linked job submission:**

```bash
python drac/scripts/submit_jobs.py \
    /path/to/your_dataset.h5 \
    --job-name "ssamba_experiment" \
    --num-jobs 3 \
    --wandb-project "ssamba_drac" \
    --wandb-group "experiment_v1" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/ssamba_experiments \
    --mode single \
    --train-ratio 0.8 \
    --training-type pretrain_finetune \
    --task ft_avgtok
```

**Training size experiments:**

```bash
python drac/scripts/submit_jobs.py \
    /path/to/your_dataset.h5 \
    --job-name "ssamba_size_exp" \
    --num-jobs 2 \
    --wandb-project "ssamba_drac" \
    --wandb-group "size_experiments" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/ssamba_experiments \
    --mode multi \
    --train-ratios 0.2 0.4 0.6 0.8 \
    --training-type pretrain_finetune
```

**Supervised only:**

```bash
python drac/scripts/submit_jobs.py \
    /path/to/your_dataset.h5 \
    --job-name "supervised_baseline" \
    --num-jobs 1 \
    --wandb-project "ssamba_drac" \
    --wandb-group "supervised" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/ssamba_experiments \
    --training-type supervised \
    --train-ratio 0.8
```

---

## Storage & Structure

**Use appropriate storage:**

* `~` (home): code & small configs only (file-count quota \~500k).
* `/project`: shared data, long-term storage.
* `/scratch/$USER`: experiments, checkpoints, caches (preferred for heavy I/O).

**Example layout:**

```
/scratch/$USER/ssamba_experiments/
├── pretrain/...
└── finetune/...
```

---

## SLURM Templates

**Typical resources:**

```bash
#SBATCH --account=<your_allocation>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
```

**Larger:**

```bash
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
```

---

## Monitoring

```bash
squeue -u $USER
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
tail -f out/job_name_*.out
tail -f err/job_name_*.err
scancel <job_id>       # cancel one
scancel -u $USER       # cancel all
```

---

## Troubleshooting

### 1) Pip tries to downgrade Torch

`mamba_ssm` / `s3prl` may push Torch → 2.5.1. Fix:

```bash
pip install --no-deps --force-reinstall "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0"
pip install --no-deps "mamba_ssm==2.2.4" "causal_conv1d>=1.5.0" "s3prl==0.4.15"
```

Or always install with the preferred constraints and `--no-deps`.

### 2) Triton backend = NONE

* Run on a **GPU node** (not a login node).
* Load `cuda/12.2` module.
* Check:

  ```bash
  python - <<'PY'
  import torch; print(torch.cuda.is_available())
  from triton.runtime import driver; print(getattr(driver.active,"name","NONE"))
  PY
  ```
* If needed: `pip install "cuda-python>=12.2,<13"` and ensure `libcuda.so.1` is on `LD_LIBRARY_PATH`.

### 3) “Disk quota exceeded (os error 122)”

Likely **inode** quota in `$HOME`. Free files and move caches:

```bash
du --inodes -d1 ~ | sort -n | tail -20
pip cache purge
rm -rf ~/.cache/{pip,wandb,huggingface} ~/.cargo ~/.rustup ~/wandb
```

Prefer using `/scratch/$USER` for caches and experiment output.

### 4) Optional: silence “Ignoring invalid distribution -orch”

If you see that warning, remove the stray broken dist:

```bash
rm -rf "$VIRTUAL_ENV/lib/python3.10/site-packages"/-orch*
```

### 5) W\&B optional

If cluster runs don’t need W\&B, set `WANDB_MODE=disabled` or gate imports behind a flag.

---

<details>
<summary><strong>Alternative Torch stack (align to mamba_ssm’s metadata)</strong></summary>

The Compute Canada `mamba_ssm==2.2.4` wheel declares `torch~=2.5.0`. If you prefer zero resolver warnings:

```bash
pip install --no-deps --force-reinstall \
  "torch==2.5.1" "torchaudio==2.5.1" "torchvision==0.20.1"

# Reinstall mamba bits without deps so Torch stays put
pip install --no-deps "mamba_ssm==2.2.4" "causal_conv1d>=1.5.0" "s3prl==0.4.15"
```

</details>

---

## Links

* DRAC docs: [https://docs.alliancecan.ca](https://docs.alliancecan.ca)
* SLURM docs: [https://slurm.schedmd.com](https://slurm.schedmd.com)