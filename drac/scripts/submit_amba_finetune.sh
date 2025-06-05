#!/bin/bash
#SBATCH --account=def-kmoran                # DRAC project account
#SBATCH --job-name=amba_finetune            # Job name
#SBATCH --output=out/amba_finetune_%j.out   # Standard output log (%j adds job ID)
#SBATCH --error=err/amba_finetune_%j.err    # Standard error log
#SBATCH --time=3:00:00                     # Maximum runtime (HH:MM:SS)
#SBATCH --gres=gpu:v100l:1                  # Request 1 V100 GPU
#SBATCH --cpus-per-task=4                   # Number of CPU cores
#SBATCH --mem=32G                           # Memory per node

# Input arguments
TRAINING_DATA_PATH=${1:-$HOME/projects/def-kmoran/merileo/ssl_hydrophones/data/h5/training_data.h5}
PRETRAINED_MODEL_PATH=${2:-""}  # Path to pretrained model
WANDB_GROUP=${3:-"default_finetune"}  # Default group if not provided
TRAIN_RATIO=${4:-0.8}  # Default to 0.8 if not provided
PROJECT_PATH=${5:-$HOME/ssamba}

# Extract the filename from the training data path
TRAINING_DATA_FILENAME=$(basename "$TRAINING_DATA_PATH")

# Load required modules
module load python/3.10

# Activate your virtual environment
source $HOME/ssamba/myenv/bin/activate

# Load W&B API key from .env file if needed
if [ -f $PROJECT_PATH/.env ]; then
    export $(grep -v '^#' $PROJECT_PATH/.env | xargs)
fi

# Copy training data to SLURM temporary directory while preserving filename
echo "Copying training data to temporary directory..."
cp "$TRAINING_DATA_PATH" "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME"

# Copy project files to SLURM temporary directory
echo "Copying project files to temporary directory..."
cp -ru "$PROJECT_PATH" "$SLURM_TMPDIR/ssamba_project"

# Copy pretrained model to temporary directory if provided
if [ ! -z "$PRETRAINED_MODEL_PATH" ]; then
    echo "Copying pretrained model to temporary directory..."
    mkdir -p "$SLURM_TMPDIR/pretrained"
    cp "$PRETRAINED_MODEL_PATH" "$SLURM_TMPDIR/pretrained/$(basename "$PRETRAINED_MODEL_PATH")"
    PRETRAINED_MODEL_PATH="$SLURM_TMPDIR/pretrained/$(basename "$PRETRAINED_MODEL_PATH")"
fi

# Navigate to the copied project directory
cd "$SLURM_TMPDIR/ssamba_project"

# Run the finetuning script, passing all necessary arguments
bash src/run_amba_finetune.sh "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME" "$PRETRAINED_MODEL_PATH" "$TRAIN_RATIO" "$WANDB_GROUP" 