#!/bin/bash

# Check if dataset path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_path> [wandb_group] [num_jobs] [project_path] [job_name]"
  exit 1
fi

DATASET_PATH=$1
WANDB_GROUP=${2:-"training_size_experiment"}
NUM_JOBS=${3:-5}  # Default to 5 linked jobs per training size
PROJECT_PATH=${4:-$HOME/ssamba}
JOB_NAME=${5:-amba_spectrogram}

# Create output and error directories if they don't exist
mkdir -p out err

# Function to submit linked jobs for a given training ratio
submit_linked_jobs() {
    local ratio=$1
    local prev_job_id

    # Submit the first job for this ratio
    prev_job_id=$(sbatch --parsable \
                  --job-name="${JOB_NAME}_${ratio}" \
                  --output=out/${JOB_NAME}_ratio${ratio}_job1_%j.out \
                  --error=err/${JOB_NAME}_ratio${ratio}_job1_%j.err \
                  submit_amba_spectrogram.sh "$DATASET_PATH" "${WANDB_GROUP}" $ratio $PROJECT_PATH)
    echo "Submitted first job for ratio $ratio with Job ID: $prev_job_id"

    # Submit dependent jobs
    for i in $(seq 2 $NUM_JOBS); do
        prev_job_id=$(sbatch --parsable \
                      --dependency=afterany:$prev_job_id \
                      --job-name="${JOB_NAME}_${ratio}" \
                      --output=out/${JOB_NAME}_ratio${ratio}_job${i}_%j.out \
                      --error=err/${JOB_NAME}_ratio${ratio}_job${i}_%j.err \
                      submit_amba_spectrogram.sh "$DATASET_PATH" "${WANDB_GROUP}" $ratio $PROJECT_PATH)
        echo "Submitted job $i for ratio $ratio with dependency on Job ID: $prev_job_id"
    done
}

# Submit jobs for each training ratio
for ratio in 0.1 0.2 0.4 0.6 0.8; do
    echo "Submitting $NUM_JOBS linked jobs for training ratio $ratio"
    submit_linked_jobs $ratio
done 