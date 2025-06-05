#!/bin/bash

# Check if dataset path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_path> <wandb_project> <wandb_group> <num_jobs> <train_ratio> <project_path> <resume> <task> <exp_dir> <time_limit>"
  echo "  dataset_path: Path to the training dataset"
  echo "  wandb_project: W&B project name (default: amba_spectrogram)"
  echo "  wandb_group: W&B group name (default: training_size_experiment)"
  echo "  num_jobs: Number of linked jobs per training size (default: 5)"
  echo "  project_path: Path to project directory (default: $HOME/ssamba)"
  echo "  resume: Whether to resume from checkpoint (default: true)"
  echo "  task: Training task (default: pretrain_joint)"
  echo "  exp_dir: Experiment directory (default: /exp)"
  echo "  time_limit: Job time limit in HH:MM:SS format (default: 08:00:00)"
  exit 1
fi

DATASET_PATH=$1
WANDB_PROJECT=${2:-"amba_spectrogram"}
WANDB_GROUP=${3:-"training_size_experiment"}
NUM_JOBS=${4:-5}  # Default to 5 linked jobs per training size
PROJECT_PATH=${5:-$HOME/ssamba}
RESUME=${6:-"true"}  # Default to true - will automatically resume if checkpoint exists
TASK=${7:-"pretrain_joint"}  # Default to pretrain_joint if not provided
EXP_DIR=${8:-"/exp"}  # Default to /exp if not provided
TIME_LIMIT=${9:-"08:00:00"}  # Default to 8 hours if not provided

# Create output and error directories if they don't exist
mkdir -p out err

# Function to submit linked jobs for a given training ratio
submit_linked_jobs() {
    local ratio=$1
    local prev_job_id

    # Submit the first job for this ratio
    prev_job_id=$(sbatch --parsable \
                  --job-name="${JOB_NAME}_${ratio}" \
                  --time=$TIME_LIMIT \
                  --output=out/${JOB_NAME}_ratio${ratio}_job1_%j.out \
                  --error=err/${JOB_NAME}_ratio${ratio}_job1_%j.err \
                  submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_PROJECT" "$WANDB_GROUP" $ratio $PROJECT_PATH $RESUME $TASK $EXP_DIR)
    echo "Submitted first job for ratio $ratio with Job ID: $prev_job_id"

    # Submit dependent jobs
    for i in $(seq 2 $NUM_JOBS); do
        prev_job_id=$(sbatch --parsable \
                      --dependency=afterany:$prev_job_id \
                      --job-name="${JOB_NAME}_${ratio}" \
                      --time=$TIME_LIMIT \
                      --output=out/${JOB_NAME}_ratio${ratio}_job${i}_%j.out \
                      --error=err/${JOB_NAME}_ratio${ratio}_job${i}_%j.err \
                      submit_amba_spectrogram.sh "$DATASET_PATH" "$WANDB_PROJECT" "$WANDB_GROUP" $ratio $PROJECT_PATH $RESUME $TASK $EXP_DIR)
        echo "Submitted job $i for ratio $ratio with dependency on Job ID: $prev_job_id"
    done
}

# Submit jobs for each training ratio
for ratio in 0.1 0.2 0.4 0.6 0.8; do
    echo "Submitting $NUM_JOBS linked jobs for training ratio $ratio"
    submit_linked_jobs $ratio
done 