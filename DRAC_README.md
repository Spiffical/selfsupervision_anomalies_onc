# DRAC Cluster Usage Guide

This guide provides instructions for running the Self-Supervised Anomaly Detection project on DRAC (Digital Research Alliance of Canada) clusters.

## Prerequisites

- Access to a DRAC cluster (Cedar, Graham, Narval, or Beluga)
- Familiarity with SLURM job submission
- Valid DRAC account with appropriate resource allocations

## Setup

### 1. Environment Setup

**Load Required Modules:**
```bash
module load python/3.10
module load cuda/11.8
module load cudnn/8.7
```

**Create Virtual Environment:**
```bash
python -m venv .env_drac
source .env_drac/bin/activate
```

**Install Dependencies:**
```bash
pip install -r requirements.txt
pip install -r drac/requirements_drac.txt
pip install .
```

### 2. Data Location

The primary ONC dataset is located at:
```
/lustre03/project/6003287/shared/ssamba_data/
```

Ensure your scripts point to this path when running on DRAC.

## Usage

The DRAC scripts support multiple job submission modes:

### 1. Single Linked Job Submission

Submit a series of linked jobs where each depends on the previous one:

```bash
python drac/scripts/submit_jobs.py \
    /lustre03/project/6003287/shared/ssamba_data/your_dataset.h5 \
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

### 2. Training Size Experiments

Run experiments across multiple training set sizes:

```bash
python drac/scripts/submit_jobs.py \
    /lustre03/project/6003287/shared/ssamba_data/your_dataset.h5 \
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

### 3. Supervised Training Only

Run supervised training without pre-training:

```bash
python drac/scripts/submit_jobs.py \
    /lustre03/project/6003287/shared/ssamba_data/your_dataset.h5 \
    --job-name "supervised_baseline" \
    --num-jobs 1 \
    --wandb-project "ssamba_drac" \
    --wandb-group "supervised" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/ssamba_experiments \
    --training-type supervised \
    --train-ratio 0.8
```

## Available Scripts

### Main Submission Script

- **`submit_jobs.py`**: Primary script for job submission with multiple modes
  - Single linked jobs (pre-training → fine-tuning)
  - Multi-ratio experiments
  - Supervised training
  - Dry-run capability for testing

### Individual Submission Scripts

- **`submit_amba_spectrogram.sh`**: SLURM script for SSAMBA training
- **`submit_supervised.sh`**: SLURM script for supervised training
- **`submit_amba_finetune.sh`**: SLURM script for fine-tuning only

### Experiment Scripts

- **`submit_linked_jobs.sh`**: Submit linked pre-training and fine-tuning jobs
- **`submit_training_size_experiments.sh`**: Run experiments across training sizes
- **`test_job.sh`**: Test job for validating setup

## Key Parameters

### Job Configuration
- `--job-name`: Base name for SLURM jobs
- `--num-jobs`: Number of sequential jobs to submit
- `--time-limit`: Maximum runtime per job (default: 12 hours)
- `--project-path`: Path to the project directory
- `--exp-dir`: Directory for experiment outputs (use `/scratch/$USER/`)

### Training Configuration
- `--training-type`: Choose `pretrain_finetune` or `supervised`
- `--task`: Fine-tuning task (`ft_avgtok`, `ft_cls`, `ft_avgtok_1sec`)
- `--train-ratio`: Fraction of data for training (single mode)
- `--train-ratios`: Multiple ratios for experiments (multi mode)
- `--exclude-labels`: Labels to exclude from training

### Weights & Biases
- `--wandb-project`: W&B project name
- `--wandb-group`: W&B group for organizing runs
- `--wandb-entity`: W&B team/entity name

### Advanced Options
- `--resume`: Resume training from checkpoints
- `--pretrained-path`: Path to pre-trained model
- `--dry-run`: Print commands without executing

## Resource Management

### Storage Guidelines

**Use appropriate storage locations:**
- **Home directory (`~`)**: Code, small configs (limited quota)
- **Project space (`/project/`)**: Shared datasets, long-term storage
- **Scratch space (`/scratch/$USER/`)**: Experiment outputs, temporary files

**Example directory structure:**
```
/scratch/$USER/ssamba_experiments/
├── pretrain/
│   └── amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/
└── finetune/
    └── amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/
```

### SLURM Resource Allocation

**Typical resource requirements:**
```bash
#SBATCH --account=def-username        # Your allocation
#SBATCH --gres=gpu:v100:1            # 1 V100 GPU
#SBATCH --cpus-per-task=4            # 4 CPU cores
#SBATCH --mem=32G                    # 32GB RAM
#SBATCH --time=0-12:00:00            # 12 hours max
```

**For larger models or datasets:**
```bash
#SBATCH --gres=gpu:a100:1            # A100 GPU (if available)
#SBATCH --cpus-per-task=8            # More CPU cores
#SBATCH --mem=64G                    # More RAM
#SBATCH --time=1-00:00:00            # 24 hours max
```

## Monitoring Jobs

### Check Job Status
```bash
# View your running jobs
squeue -u $USER

# View detailed job information
scontrol show job <job_id>

# View job history
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Start,End,Elapsed
```

### View Job Outputs
```bash
# Check output logs
tail -f out/job_name_*.out

# Check error logs
tail -f err/job_name_*.err
```

### Cancel Jobs
```bash
# Cancel a specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

## Troubleshooting

### Common Issues

1. **Module Loading Errors**
   ```bash
   # Check available modules
   module avail python
   module avail cuda
   
   # Load compatible versions
   module load python/3.10 cuda/11.8
   ```

2. **Storage Quota Exceeded**
   ```bash
   # Check disk usage
   diskusage_report
   
   # Clean up old experiments
   rm -rf /scratch/$USER/old_experiments/
   ```

3. **GPU Memory Issues**
   - Reduce batch size in your configuration
   - Use gradient checkpointing
   - Monitor GPU usage with `nvidia-smi`

4. **Job Dependencies**
   ```bash
   # Check job dependencies
   squeue -u $USER --format="%.10i %.20j %.8T %.15D"
   ```

### Getting Help

- **DRAC Documentation**: [docs.alliancecan.ca](https://docs.alliancecan.ca)
- **SLURM Documentation**: [slurm.schedmd.com](https://slurm.schedmd.com)
- **Support**: Contact DRAC support through their ticketing system

## Examples

### Complete Workflow Example

```bash
# 1. Load modules and activate environment
module load python/3.10 cuda/11.8 cudnn/8.7
source .env_drac/bin/activate

# 2. Submit pre-training and fine-tuning jobs
python drac/scripts/submit_jobs.py \
    /lustre03/project/6003287/shared/ssamba_data/onc_dataset.h5 \
    --job-name "ssamba_full_exp" \
    --num-jobs 2 \
    --wandb-project "ssamba_production" \
    --wandb-group "full_pipeline" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/ssamba_production \
    --mode single \
    --train-ratio 0.8 \
    --training-type pretrain_finetune \
    --task ft_avgtok \
    --time-limit "1-00:00:00"

# 3. Monitor progress
watch -n 30 'squeue -u $USER'
```

### Dry Run Testing

Always test your job submission with `--dry-run` first:

```bash
python drac/scripts/submit_jobs.py \
    /path/to/dataset.h5 \
    --job-name "test_run" \
    --num-jobs 1 \
    --wandb-project "test" \
    --wandb-group "test" \
    --project-path $PWD \
    --exp-dir /scratch/$USER/test \
    --dry-run
``` 