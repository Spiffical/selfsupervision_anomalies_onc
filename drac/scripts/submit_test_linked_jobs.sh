#!/bin/bash

NUM_JOBS=5  # Number of linked jobs to submit
JOB_SCRIPT="test_job.sh"

# Submit the first job
prev_job_id=$(sbatch --parsable $JOB_SCRIPT)
echo "Submitted job 1 with Job ID: $prev_job_id"

# Submit subsequent jobs with dependencies
for i in $(seq 2 $NUM_JOBS); do
    prev_job_id=$(sbatch --parsable --dependency=afterany:$prev_job_id $JOB_SCRIPT)
    echo "Submitted job $i with dependency on Job ID: $prev_job_id"
done
