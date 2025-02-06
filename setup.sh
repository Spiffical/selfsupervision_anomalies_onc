#!/bin/bash

# Exit on error
set -e

# Install NVIDIA runtime libraries if not present
if ! dpkg -l | grep -q "nvidia-cuda-toolkit"; then
    echo "Installing NVIDIA CUDA runtime libraries..."
    sudo apt-get update
    sudo apt-get install -y nvidia-cuda-toolkit
fi

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and install build tools
pip install --upgrade pip
pip install --upgrade setuptools wheel build

# Uninstall existing PyTorch and TorchAudio if present
pip uninstall -y torch torchvision torchaudio

# Install PyTorch and TorchAudio with CUDA 12.4 support
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA versions match
python -c "import torch; import torchaudio; print(f'PyTorch version: {torch.__version__}'); print(f'PyTorch CUDA version: {torch.version.cuda}'); print(f'TorchAudio version: {torchaudio.__version__}')"

# Install numpy before causal_conv1d
pip install numpy

# Try installing causal_conv1d with build isolation disabled
TORCH_CUDA_ARCH_LIST="7.5" pip install causal_conv1d --no-build-isolation

# Install other core dependencies
pip install h5py wandb

# Install model dependencies
pip install timm

# Install the rest of the requirements
pip install -r requirements.txt

# Update LD_LIBRARY_PATH if needed
CUDA_PATH=$(dirname $(dirname $(which nvcc)))
if [ -d "${CUDA_PATH}/lib64" ]; then
    echo "export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "export PATH=${CUDA_PATH}/bin:$PATH" >> ~/.bashrc
fi

echo "Setup complete! Please run the following commands to update your environment:"
echo "source ~/.bashrc"
echo "source .venv/bin/activate"

# Print PyTorch device information for verification
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA version:', torch.version.cuda)"

# Print CUDA information for debugging
echo "Checking CUDA installation..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
fi
if command -v nvcc &> /dev/null; then
    nvcc --version
fi 