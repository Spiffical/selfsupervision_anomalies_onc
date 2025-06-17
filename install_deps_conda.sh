#!/bin/bash

echo "Installing dependencies using conda (alternative to pip)..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please install conda/miniconda first or use the pip-based install_deps.sh"
    exit 1
fi

# Install PyTorch and basic scientific packages from conda-forge
echo "Installing base packages with conda..."
conda install -y pytorch torchaudio numpy scipy matplotlib pandas scikit-learn h5py tqdm ipython -c pytorch -c conda-forge

# Install packages that are easier with pip
echo "Installing additional packages with pip..."
pip install wandb einops timm python-dotenv interfaces model_factory wget

# Try conda for causal_conv1d (often works better than pip)
echo "Attempting to install causal_conv1d with conda..."
conda install -y causal-conv1d -c conda-forge || {
    echo "Warning: causal_conv1d not available in conda-forge. Trying pip..."
    pip install "causal_conv1d>=1.5.0" --no-build-isolation || {
        echo "Warning: causal_conv1d installation failed with both conda and pip."
    }
}

# Try mamba_ssm
echo "Installing mamba_ssm..."
pip install mamba_ssm || echo "Warning: mamba_ssm installation failed"

# Install remaining packages
pip install s3prl==0.4.15 || echo "Warning: s3prl installation failed"
pip install "onc>=2.3.0" || echo "Warning: onc installation failed"
pip install seaborn

echo ""
echo "Conda-based installation complete!"
echo "This approach often works better for compiled packages like causal_conv1d." 