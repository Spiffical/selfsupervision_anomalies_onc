#!/bin/bash

NUM_JOBS=5  # Number of linked jobs
JOB_SCRIPT="submit_amba_spectrogram.sh"

TRAINING_DATA_PATH="/home/merileo/projects/def-kmoran/merileo/ssl_hydrophones/data/h5/different_locations_incl_backgroundpipelinenormals_multilabel.h5"
PROJECT_PATH="$HOME/ssamba"

# Submit the first job
prev_job_id=$(sbatch --parsable $JOB_SCRIPT $TRAINING_DATA_PATH $PROJECT_PATH)
echo "Submitted job 1 with Job ID: $prev_job_id"

# Submit dependent jobs
for i in $(seq 2 $NUM_JOBS); do
    prev_job_id=$(sbatch --parsable --dependency=afterany:$prev_job_id $JOB_SCRIPT $TRAINING_DATA_PATH $PROJECT_PATH)
    echo "Submitted job $i with dependency on Job ID: $prev_job_id"
done
