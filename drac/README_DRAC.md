# DRAC Cluster Usage Guide for SSAMBA Anomaly Detection

This guide provides specific instructions for running the Self-Supervised Anomaly Detection project on the DRAC (Digital Research Alliance of Canada) cluster.

## Prerequisites

*   Access to a DRAC cluster (e.g., Cedar, Graham, Narval, Beluga).
*   Familiarity with SLURM job submission.

## Setup

1.  **Clone the Repository (if not already done):
    ```bash
    git clone https://github.com/Spiffical/selfsupervision_anomalies_onc.git
    cd selfsupervision_anomalies_onc
    git checkout drac # Ensure you are on the drac branch
    ```

2.  **Load Required Modules:**
    Before setting up the Python environment, you may need to load specific modules. Common modules include:
    ```bash
    module load python/3.9  # Or your preferred/project-compatible Python version
    module load cuda/11.8   # Or the CUDA version compatible with your PyTorch build
    module load cudnn/8.7   # Or the CUDNN version compatible with CUDA & PyTorch
    # Add any other necessary modules (e.g., for specific compilers or libraries)
    ```
    Consult the DRAC documentation for available modules.

3.  **Create and Activate Python Environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv .env_drac
    source .env_drac/bin/activate
    ```

4.  **Install DRAC-specific Requirements:**
    The `drac/` directory contains a `requirements_drac.txt` file optimized for the cluster environment.
    ```bash
    pip install -r drac/requirements_drac.txt
    ```
    *Note: You might also need to install the base requirements from `requirements.txt` if `requirements_drac.txt` doesn't include them all, or if it only contains overrides/additions.*
    ```bash
    pip install -r requirements.txt # If needed
    pip install -r drac/requirements_drac.txt
    ```

5.  **Install the Project Package:**
    ```bash
    pip install .
    ```

## Data

As mentioned in the main README, the primary ONC dataset is expected to be located at:
`/lustre03/project/6003287/shared/ssamba_data/`

Ensure your scripts and configurations point to this path when running on DRAC.

## Running Jobs

The `drac/scripts/` directory contains scripts tailored for job submission on DRAC using SLURM.

### Main Submission Script: `submit_jobs.py`

*(Detailed instructions on how to use `submit_jobs.py` will go here. This should cover:)*
*   *Purpose of the script.*
*   *Key command-line arguments.*
*   *How to specify different models, datasets, or experiment configurations.*
*   *Example usage.*
*   *How it generates SLURM batch scripts.*

### Other Submission Scripts

*(Details about other scripts like `submit_supervised.sh`, `submit_amba_spectrogram.sh`, etc., if they are meant to be used directly on DRAC or are called by `submit_jobs.py`)*

### Example SLURM Script Structure (Illustrative)

Your `submit_jobs.py` likely generates SLURM scripts similar to this:

```bash
#!/bin/bash
#SBATCH --account=<your_account_id>  # e.g., def-username or rrg-username
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=4             # Number of CPUs
#SBATCH --mem=32G                     # Memory allocation
#SBATCH --time=0-24:00:00             # Maximum job run time (D-HH:MM:SS)
#SBATCH --output=logs/job_%x_%j.out   # Standard output and error log

# Load modules (if not loaded in login environment)
module load python/3.9 cuda/11.8 cudnn/8.7

# Activate environment
source .env_drac/bin/activate

# Navigate to project directory (if needed, often current dir by default)
# cd /path/to/your/selfsupervision_anomalies_onc

# Run your experiment script
python src/run_amba_spectrogram.py --config drac_configs/my_experiment.yaml --exp-dir /scratch/username/ssamba_runs/exp1
# (Adjust command as per your actual script and arguments)
```

## Important Notes for DRAC Usage

*   **Storage:** Be mindful of storage quotas on `/project`, `/scratch`, and home directories. Intermediate results and large datasets should ideally be on `/scratch` or project spaces.
*   **Output Directories:** Ensure your experiment scripts write outputs (models, logs, results) to appropriate directories, preferably within a `/scratch` space or your allocated project directory, to avoid filling up your home directory.
*   **Resource Allocation:** Adjust SLURM parameters (`--gres`, `--cpus-per-task`, `--mem`, `--time`) based on your job's requirements and available cluster resources.
*   **Monitoring Jobs:** Use `squeue -u <your_username>` to monitor your jobs and `sacct -j <jobid>` for details on completed jobs.

*(Further DRAC-specific tips and troubleshooting can be added here.)* 