#!/bin/bash

NUM_JOBS=5  # Number of linked jobs
JOB_SCRIPT="submit_amba_spectrogram.sh"

TRAINING_DATA_PATH=${1:-$HOME/projects/def-kmoran/merileo/ssl_hydrophones/data/h5/different_locations_incl_backgroundpipelinenormals_multilabel.h5}
WANDB_PROJECT=${2:-"amba_spectrogram"}
WANDB_GROUP=${3:-"default_experiment"}  # Default group if not provided
TRAIN_RATIO=${4:-0.8}  # Default to 0.8 if not provided
PROJECT_PATH=${5:-$HOME/ssamba}
JOB_NAME=${6:-amba_spectrogram}
TASK=${7:-"pretrain_joint"}  # Default to pretrain_joint if not provided
EXP_DIR=${8:-"/exp"}  # Default to /exp if not provided

# Submit the first job
prev_job_id=$(sbatch --parsable \
              --job-name=$JOB_NAME \
              --output=out/${JOB_NAME}_job1_%j.out \
              --error=err/${JOB_NAME}_job1_%j.err \
              $JOB_SCRIPT $TRAINING_DATA_PATH $WANDB_PROJECT $WANDB_GROUP $TRAIN_RATIO $PROJECT_PATH "true" "$TASK" "$EXP_DIR")
echo "Submitted job 1 with Job ID: $prev_job_id"

# Submit dependent jobs
for i in $(seq 2 $NUM_JOBS); do
    prev_job_id=$(sbatch --parsable \
                  --dependency=afterany:$prev_job_id \
                  --job-name=$JOB_NAME \
                  --output=out/${JOB_NAME}_job${i}_%j.out \
                  --error=err/${JOB_NAME}_job${i}_%j.err \
                  $JOB_SCRIPT $TRAINING_DATA_PATH $WANDB_PROJECT $WANDB_GROUP $TRAIN_RATIO $PROJECT_PATH "true" "$TASK" "$EXP_DIR")
    echo "Submitted job $i with dependency on Job ID: $prev_job_id"
done
