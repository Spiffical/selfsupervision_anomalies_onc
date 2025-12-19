#!/bin/bash

echo "Installing ONC SSAMBA dependencies using conda..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please install conda/miniconda first or use ./install_deps.sh"
    exit 1
fi

# Install PyTorch and basic scientific packages from conda-forge
echo "Installing base packages with conda..."
conda install -y pytorch torchaudio numpy scipy matplotlib pandas scikit-learn h5py tqdm ipython -c pytorch -c conda-forge

# Install the package in editable mode
echo ""
echo "Installing onc_ssamba package..."
pip install -e .

# Try installing CUDA packages
echo ""
echo "Attempting to install Mamba/CUDA dependencies..."

# Try conda for causal_conv1d (often works better than pip on conda)
conda install -y causal-conv1d -c conda-forge 2>/dev/null || {
    echo "causal_conv1d not in conda-forge, trying pip..."
    pip install "causal_conv1d>=1.5.0" --no-build-isolation || {
        echo "⚠️  causal_conv1d installation failed."
    }
}

# Try mamba_ssm
pip install mamba_ssm || echo "⚠️  mamba_ssm installation failed"

echo ""
echo "🎉 Conda-based installation complete!"
echo "📦 Package import: from onc_ssamba import ONCSpectrogramDataset, create_model"