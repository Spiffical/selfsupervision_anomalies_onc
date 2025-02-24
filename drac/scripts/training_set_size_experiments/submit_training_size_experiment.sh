#!/bin/bash
#SBATCH --account=def-kmoran               # Replace with your Compute Canada project account
#SBATCH --job-name=amba_train_size         # Job name
#SBATCH --output=out/amba_train_size_%j.out    # Standard output log (%j adds job ID)
#SBATCH --error=err/amba_train_size_%j.err     # Standard error log
#SBATCH --time=08:00:00                    # Maximum runtime (HH:MM:SS)
#SBATCH --gres=gpu:v100l:1                 # Request 1 V100 GPU
#SBATCH --cpus-per-task=4                  # Number of CPU cores
#SBATCH --mem=32G                          # Memory per node

# Input arguments
TRAINING_DATA_PATH=${1:-$HOME/projects/def-kmoran/merileo/ssl_hydrophones/data/h5/training_data.h5}
TRAIN_RATIO=${2:-0.8}  # Default to 80% if not provided
PROJECT_PATH=${3:-$HOME/ssamba}

# Extract the filename from the training data path
TRAINING_DATA_FILENAME=$(basename "$TRAINING_DATA_PATH")

# W&B Group Name (ensures all runs are grouped together in the W&B dashboard)
WANDB_GROUP="training_size_experiment"

# Load required modules
module load python/3.10

# Activate your virtual environment
source $HOME/ssamba/myenv/bin/activate

# Load W&B API key from .env file if available
if [ -f $PROJECT_PATH/.env ]; then
    export $(grep -v '^#' $PROJECT_PATH/.env | xargs)
fi

# Copy training data to SLURM temporary directory
echo "Copying training data to temporary directory..."
cp "$TRAINING_DATA_PATH" "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME"

# Copy project files to SLURM temporary directory
echo "Copying project files to temporary directory..."
cp -ru "$PROJECT_PATH" "$SLURM_TMPDIR/ssamba_project"

# Navigate to the copied project directory
cd "$SLURM_TMPDIR/ssamba_project"

# Run the training script with dynamic training ratio and W&B group
bash src/run_amba_spectrogram.sh "$SLURM_TMPDIR/$TRAINING_DATA_FILENAME" "$TRAIN_RATIO" "$WANDB_GROUP"
