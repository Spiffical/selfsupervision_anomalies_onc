#!/bin/bash
#SBATCH --account=def-kmoran               # Replace with your Compute Canada project account
#SBATCH --job-name=test_linked_jobs        # Job name
#SBATCH --output=test_job_%j.out           # Standard output log (%j adds job ID)
#SBATCH --error=test_job_%j.err            # Standard error log
#SBATCH --time=00:02:00                    # Maximum runtime (2 minutes)
#SBATCH --cpus-per-task=1                  # Number of CPU cores
#SBATCH --mem=1G                           # Memory per node

# Load required modules (for consistency)
module load python/3.10

# Display job information
echo "Running Job ID: $SLURM_JOB_ID"
echo "This job started at: $(date)"

# Simulate some work
sleep 200  # Sleep for 200 seconds to simulate processing

# Write to an output file to confirm execution
echo "Job $SLURM_JOB_ID completed successfully at $(date)" >> $SLURM_TMPDIR/job_output.txt

# Copy the output to your home directory to persist after job ends
cp $SLURM_TMPDIR/job_output.txt ~/test_job_output_$SLURM_JOB_ID.txt

echo "Job $SLURM_JOB_ID has finished and output is saved."
