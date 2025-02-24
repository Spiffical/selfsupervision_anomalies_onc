#!/bin/bash

# Check if dataset path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_path>"
  exit 1
fi

DATASET_PATH=$1
WANDB_GROUP=${2:-"training_size_experiment"}
PROJECT_PATH=${3:-$HOME/ssamba}
JOB_NAME=${4:-amba_spectrogram}

# Submit jobs with different training ratios
sbatch --job-name=$JOB_NAME \
       --output=out/${JOB_NAME}_ratio0.1_%j.out \
       --error=err/${JOB_NAME}_ratio0.1_%j.err \
       submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_GROUP" 0.1 $PROJECT_PATH

sbatch --job-name=$JOB_NAME \
       --output=out/${JOB_NAME}_ratio0.2_%j.out \
       --error=err/${JOB_NAME}_ratio0.2_%j.err \
       submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_GROUP" 0.2 $PROJECT_PATH

sbatch --job-name=$JOB_NAME \
       --output=out/${JOB_NAME}_ratio0.4_%j.out \
       --error=err/${JOB_NAME}_ratio0.4_%j.err \
       submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_GROUP" 0.4 $PROJECT_PATH

sbatch --job-name=$JOB_NAME \
       --output=out/${JOB_NAME}_ratio0.6_%j.out \
       --error=err/${JOB_NAME}_ratio0.6_%j.err \
       submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_GROUP" 0.6 $PROJECT_PATH

sbatch --job-name=$JOB_NAME \
       --output=out/${JOB_NAME}_ratio0.8_%j.out \
       --error=err/${JOB_NAME}_ratio0.8_%j.err \
       submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_GROUP" 0.8 $PROJECT_PATH
