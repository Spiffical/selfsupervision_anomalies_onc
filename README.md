# Self-Supervised Learning for Anomaly Detection in Underwater Acoustics

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
  - [Dependencies](#dependencies)
  - [Installation Steps](#installation-steps)
- [Data](#data)
  - [ONC Data Download and Preparation](#onc-data-download-and-preparation)
- [Usage](#usage)
  - [Running Experiments Locally](#running-experiments-locally)
  - [Self-Supervised Pre-training and Fine-tuning Example](#self-supervised-pre-training-and-fine-tuning-example)
    - [Available Tasks](#available-tasks)
    - [1. Pre-training Phase](#1-pre-training-phase)
    - [2. Fine-tuning Phase](#2-fine-tuning-phase)
    - [Key Parameters](#key-parameters)
    - [Task Selection Guidelines](#task-selection-guidelines)
  - [Running on DRAC Cluster](#running-on-drac-cluster)
    - [Quick Start for DRAC](#quick-start-for-drac)
    - [Available DRAC Scripts](#available-drac-scripts)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

This project focuses on leveraging self-supervised learning techniques for anomaly detection in underwater acoustic data. It utilizes the SSAMBA (Self-Supervised Audio Mamba) model as a core component to learn robust audio representations, which are then applied to identify anomalous sound events in recordings from the Ocean Networks Canada (ONC) dataset.

The primary goal is to develop and evaluate methods for detecting unusual underwater sounds that may indicate equipment malfunction, unique biological events, or other significant occurrences that deviate from normal ambient noise.

This repository is a personalized fork and significant modification of the original [SSAMBA project](https://github.com/SiavashShams/ssamba) (see also their paper: [SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model](https://arxiv.org/abs/2405.11831)). While the core SSAMBA architecture is used, this project includes custom data handling, experiment scripts, and analysis tools tailored for the ONC dataset and the specific task of anomaly detection.

## Repository Structure

The repository is organized as follows:

*   `src/`: Contains the main source code.
    *   `ssamba/`: The core Python package for this project, including dataset handling (`dataset.py`), model definitions (within `models/`), training and evaluation logic (`traintest.py`, `traintest_mask.py`), and various utilities (`utilities/`).
    *   `run_supervised.py`: Script for running supervised training/evaluation.
    *   `run_amba_spectrogram.py`: Script for running self-supervised pre-training or fine-tuning experiments with the SSAMBA model on spectrograms.
*   `scripts/`: Contains utility and experiment execution scripts.
    *   Data download: `download_spectrograms.py` (see [DATA_DOWNLOAD_AND_PREPARATION.md](DATA_DOWNLOAD_AND_PREPARATION.md))
    *   Data preparation: `create_h5_dataset.py`
    *   Analysis: `analyze_labels.py`, `analyze_val_set.py`
    *   Local experiment runners: `run_supervised.sh`, `run_amba_spectrogram.sh`, etc.
*   `notebooks/`: Jupyter notebooks for data exploration, results analysis, and experimentation.
*   `tools/`: Utility applications and tools.
    *   `labeling/`: Interactive Dash app for data labeling and annotation.
*   `utils/`: Utility modules and helper functions.
    *   `data/`: Data processing utilities including spectrogram downloading, deployment checking, and data manipulation tools.
*   `data/`: (Gitignored) Intended for storing local data
*   `logs/`: (Gitignored) Default directory for log files generated during experiments.
*   `eval/`: Scripts related to model evaluation.
*   `tests/`: Unit tests for the project.
*   `exp/`: (Gitignored) Default directory where experiment outputs (models, results, configs) are saved.
*   `drac/`: Contains DRAC cluster-specific job submission scripts and configurations for running experiments on Digital Research Alliance of Canada clusters.

## Setup & Installation

### Dependencies

*   Python 3.8+
*   PyTorch (see `requirements-base.txt` for version, compatible with CUDA if available)
*   Other Python packages are listed in `requirements-base.txt` and `requirements-mamba.txt`.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc.git
    cd selfsupervision_anomalies_onc
    ```

2.  **Set up a Python environment:**
    It is highly recommended to use a virtual environment (e.g., `conda` or `venv`).

    *Using conda:*
    ```bash
    conda create -n ssamba_env python=3.9  # Or your preferred Python version
    conda activate ssamba_env
    ```

    *Using venv:*
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Python packages:**

    **Option A: Automatic Installation (Recommended)**
    ```bash
    ./install_deps.sh
    ```
    
    **Option B: Conda-based Installation (Alternative)**
    ```bash
    ./install_deps_conda.sh
    ```
    
    **Option C: Manual Step-by-Step Installation**
    ```bash
    # Step 1: Install base dependencies (including PyTorch)
    pip install -r requirements-base.txt
    
    # Step 2: Install mamba-related packages that depend on PyTorch
    pip install -r requirements-mamba.txt
    
    # Step 3: Install additional packages for evaluation
    pip install seaborn
    ```

    > **Note**: Dependencies are split into separate files because some packages (`causal_conv1d`, `mamba_ssm`) require PyTorch to be installed first during their build process. The automatic installation script (Option A) will detect your system and install appropriate packages.

4.  **Install the project package:**
    This step makes the `ssamba` module importable in your environment.
    ```bash
    pip install .
    ```

### Platform Compatibility & Troubleshooting

**⚠️ Important: SSAMBA Model Requirements**
The self-supervised SSAMBA model (`run_amba_spectrogram.py`) requires `causal_conv1d` and `mamba_ssm` packages, which have strict requirements:
- **Linux OS only** (Windows/macOS not supported)
- **NVIDIA GPU** with CUDA support
- **CUDA toolkit** (nvcc) installed

**✅ What Works on All Platforms (including macOS)**
- Supervised learning (`run_supervised.py`)
- Model evaluation (`eval/evaluate_model.py`) 
- Data download and processing tools
- All analysis and visualization scripts

<details>
<summary><strong>📱 Platform-Specific Installation Instructions</strong></summary>

**macOS Users:**
```bash
# Use the standard installation - CUDA packages will be automatically skipped
./install_deps.sh
```
You'll get a CPU-only installation that supports most functionality except the SSAMBA model.

**Linux with NVIDIA GPU:**
```bash
# Install CUDA toolkit first (if not already installed)
conda install nvidia::cuda-toolkit=12.1

# Then run the installation
./install_deps.sh
```

**Linux without GPU:**
The installation script will automatically detect the absence of CUDA and install CPU-compatible packages only.

</details>

<details>
<summary><strong>🔧 Troubleshooting CUDA Installation Issues</strong></summary>

If you encounter `bare_metal_version` errors or "NVCC not found":

**Install CUDA Toolkit:**
```bash
# Option 1: Using conda (recommended)
conda install nvidia::cuda-toolkit=12.1

# Option 2: Manual installation
# Follow CUDA installation guide for your Linux distribution
# Then retry: pip install -r requirements-mamba.txt
```

**Alternative Installation Methods:**
```bash
# Try installing without build isolation
pip install causal_conv1d --no-build-isolation
pip install mamba_ssm --no-build-isolation

# Or force CPU-only installation
pip install -r requirements-cpu.txt
```

**Check Your Setup:**
```bash
# Verify CUDA is available
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

</details>

## Data

### ONC Data Download and Preparation

This repository includes comprehensive tools for downloading Ocean Networks Canada (ONC) hydrophone data and preparing it for machine learning workflows. The system supports:

- **🤖 Interactive Downloads**: Smart guidance through device selection and date ranges
- **🚀 Deployment Validation**: Ensures hydrophones were active during requested periods
- **📊 Flexible Batch Sizes**: Control spectrograms per request (1-36 spectrograms)
- **📁 Universal Folder Support**: Works with enhanced, legacy, flat, and nested structures
- **🏷️ Interactive Labeling Tool**: Dash-based app for visual annotation with audio playback
- **🗂️ HDF5 Dataset Creation**: Convert spectrograms into ML-ready datasets with flexible labeling

**For complete setup, usage examples, and workflow documentation, see: [DATA_DOWNLOAD_AND_PREPARATION.md](DATA_DOWNLOAD_AND_PREPARATION.md)**

#### Quick Start

```bash
# Interactive mode - guides you through the entire process
python scripts/download_spectrograms.py

# Direct download with custom batch size
python scripts/download_spectrograms.py --mode sampling --device ICLISTENHF6020 --start-date 2021 1 1 --threshold 500 --spectrograms-per-batch 12 --check-deployments

# Label your spectrograms using the interactive tool (recommended)
cd tools/labeling && python run.py

# Create HDF5 dataset from downloaded spectrograms
python scripts/create_h5_dataset.py data/mat/ICLISTENHF6020/ --output datasets/hydrophone_data.h5
```

## Usage

### Running Experiments Locally

The main scripts for running experiments are `src/run_supervised.py` and `src/run_amba_spectrogram.py`. These scripts accept various command-line arguments to configure the dataset, model, training parameters, etc.

Example shell scripts are provided in the `scripts/` directory to demonstrate how to run these Python scripts:
*   `scripts/run_supervised.sh` - Automatically detects DRAC vs local environment
*   `scripts/run_amba_spectrogram.sh` - Automatically detects DRAC vs local environment

Both scripts now automatically detect whether you're running on DRAC or locally and adjust the environment setup accordingly:

```bash
# Supervised training
bash scripts/run_supervised.sh --dataset data/your_dataset.h5

# Self-supervised training
bash scripts/run_amba_spectrogram.sh --dataset data/your_dataset.h5 --task pretrain_joint
```

Examine these shell scripts and modify them as needed (e.g., update paths, hyperparameters).

### Self-Supervised Pre-training and Fine-tuning Example

The `run_amba_spectrogram.sh` script is designed for self-supervised learning workflows. Here's how to use it for both pre-training and fine-tuning:

#### Available Tasks

**Pre-training Tasks:**
- `pretrain_mpc`: Masked Patch Classification (discriminative objective)
- `pretrain_mpg`: Masked Patch Generation/Reconstruction (generative objective)  
- `pretrain_joint`: Combined MPC + MPG training (recommended)

**Fine-tuning Tasks:**
- `ft_cls`: Fine-tuning using [CLS] token for classification
- `ft_avgtok`: Fine-tuning using average of all patch tokens (default for SSAMBA)
- `ft_avgtok_1sec`: Fine-tuning with 1-second segment averaging

#### 1. Pre-training Phase

First, train the model using self-supervised learning on your dataset:

```bash
bash scripts/run_amba_spectrogram.sh \
    --python-script src/run_amba_spectrogram.py \
    --dataset data/your_dataset.h5 \
    --task pretrain_joint \
    --wandb-project "ssamba_pretraining" \
    --wandb-group "experiment_v1" \
    --train-ratio 0.8 \
    --exp-dir ./exp
```

This will:
- Use both masked patch classification and reconstruction for robust self-supervised learning
- Save the pre-trained model to `./exp/pretrain/amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/`
- Log training progress to Weights & Biases

#### 2. Fine-tuning Phase

After pre-training completes, fine-tune the model for your specific anomaly detection task. You can choose different fine-tuning strategies:

**Option A: Average Token Fine-tuning (Recommended)**
```bash
bash scripts/run_amba_spectrogram.sh \
    --python-script src/run_amba_spectrogram.py \
    --dataset data/your_dataset.h5 \
    --task ft_avgtok \
    --pretrained-path ./exp/pretrain/amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/models/best_audio_model.pth \
    --wandb-project "ssamba_finetuning" \
    --wandb-group "experiment_v1" \
    --train-ratio 0.8 \
    --exp-dir ./exp
```

**Option B: CLS Token Fine-tuning**
```bash
bash scripts/run_amba_spectrogram.sh \
    --python-script src/run_amba_spectrogram.py \
    --dataset data/your_dataset.h5 \
    --task ft_cls \
    --pretrained-path ./exp/pretrain/amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/models/best_audio_model.pth \
    --wandb-project "ssamba_finetuning" \
    --wandb-group "experiment_v1" \
    --train-ratio 0.8 \
    --exp-dir ./exp
```

Both will:
- Load the pre-trained weights and fine-tune for classification
- Use data augmentation and balanced sampling for better performance
- Save the fine-tuned model to `./exp/finetune/amba-base-f16-t16-b16-lr0.0001-m300-custom-tr0.8-experiment_v1/`

#### Key Parameters

- `--task`: Choose from the available tasks above based on your training phase and strategy
- `--pretrained-path`: Path to the pre-trained model checkpoint (only needed for fine-tuning)
- `--dataset`: Path to your HDF5 dataset file
- `--wandb-project` / `--wandb-group`: For experiment tracking and organization
- `--train-ratio`: Fraction of data to use for training (rest split between validation and test)
- `--exp-dir`: Directory where models and logs will be saved

#### Task Selection Guidelines

- **For pre-training**: Use `pretrain_joint` for the most robust self-supervised learning
- **For fine-tuning**: Use `ft_avgtok` (SSAMBA default) or `ft_cls` depending on your preference
  - `ft_avgtok`: Uses average of all patch representations (typically better for SSAMBA)
  - `ft_cls`: Uses the [CLS] token representation (more traditional approach)

The script automatically adjusts hyperparameters based on the task (e.g., learning rate, data augmentation, masking strategy).

### Running on DRAC Cluster

For users with access to DRAC (Digital Research Alliance of Canada) clusters, comprehensive job submission scripts and documentation are available in the `drac/` directory.

**For detailed DRAC setup and usage instructions**, see: **[DRAC_README.md](DRAC_README.md)**

#### Quick Start for DRAC

1. **Setup environment:**
   ```bash
   module load StdEnv/2023 python/3.10 gcc/12.3 cuda/12.2 cudnn/8.9.5.29
   python -m venv .env_drac
   source .env_drac/bin/activate
   bash drac/scripts/install_deps_drac.sh
   ```

2. **Submit linked pre-training and fine-tuning jobs:**
   ```bash
   python drac/scripts/submit_jobs.py \
       /path/to/your_dataset.h5 \
       --job-name "ssamba_experiment" \
       --num-jobs 2 \
       --wandb-project "ssamba_drac" \
       --wandb-group "experiment_v1" \
       --project-path $PWD \
       --exp-dir /scratch/$USER/ssamba_experiments \
       --training-type pretrain_finetune \
       --task ft_avgtok
   ```

#### Available DRAC Scripts

- **`submit_jobs.py`**: Main job submission script with multiple modes
- **`submit_amba_spectrogram.sh`**: SLURM script for SSAMBA training
- **`submit_supervised.sh`**: SLURM script for supervised training
- **Training size experiments**: Scripts for running experiments across multiple training set sizes

The DRAC scripts support:
- **Linked job submission**: Automatic dependency management between pre-training and fine-tuning
- **Training size experiments**: Run experiments across multiple data ratios
- **Resource management**: Optimized SLURM configurations for different cluster types
- **Dry-run testing**: Test job configurations before submission

### Jupyter Notebooks

Exploratory data analysis, model evaluation, and experimental results can be found in or performed using the Jupyter notebooks in the `notebooks/` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

*   This work builds upon the original [SSAMBA: Self-Supervised Audio Representation Learning with Mamba State Space Model](https://github.com/SiavashShams/ssamba) project and its accompanying [paper](https://arxiv.org/abs/2405.11831).
*   The underlying Mamba architecture is based on the work presented in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
*   Dataset provided by [Ocean Networks Canada](https://www.oceannetworks.ca/).

---
