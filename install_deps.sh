#!/bin/bash

echo "Installing dependencies in correct order..."

# Detect operating system
OS=$(uname -s)
echo "Detected OS: $OS"

# Step 1: Install base dependencies (including PyTorch)
echo "Step 1: Installing base dependencies..."
pip install -r requirements-base.txt

# Step 2: Check if we should try to install CUDA-dependent packages
if [[ "$OS" == "Darwin" ]]; then
    echo ""
    echo "⚠️  WARNING: You are on macOS!"
    echo "   CUDA is not supported on macOS, so causal_conv1d and mamba_ssm cannot be installed."
    echo "   Installing CPU-compatible packages only..."
    echo ""
    
    # Install CPU-only packages
    pip install -r requirements-cpu.txt
    
    echo ""
    echo "✅ Installation complete (CPU-only mode)"
    echo "   You can use:"
    echo "   - Supervised learning scripts (run_supervised.py)"
    echo "   - Evaluation tools (eval/evaluate_model.py)"
    echo "   - Data download and processing tools"
    echo ""
    echo "   ❌ SSAMBA model (run_amba_spectrogram.py) will NOT work"
    echo "      as it requires causal_conv1d and mamba_ssm packages."
    
elif command -v nvcc &> /dev/null; then
    echo ""
    echo "✅ NVCC found! Attempting to install CUDA-dependent packages..."
    
    # Set environment variables that can help with compilation
    export TORCH_CUDA_ARCH_LIST=""
    export FORCE_CUDA=0
    
    # Try installing CUDA-dependent packages
    echo "Installing CUDA-dependent packages..."
    # Skip building CUDA extensions for mamba to avoid long compiles in some envs
    export MAMBA_SKIP_CUDA_BUILD=TRUE
    pip install -r requirements-mamba.txt || {
        echo ""
        echo "⚠️  CUDA packages failed to install."
        echo "   Falling back to CPU-compatible packages..."
        pip install -r requirements-cpu.txt
        echo ""
        echo "   You may need to install CUDA toolkit:"
        echo "   conda install nvidia::cuda-toolkit=12.1"
        echo "   or follow CUDA installation guide for your system."
    }
    
else
    echo ""
    echo "⚠️  WARNING: NVCC not found!"
    echo "   This usually means CUDA toolkit is not installed."
    echo "   Installing CPU-compatible packages only..."
    echo ""
    
    # Install CPU-only packages
    pip install -r requirements-cpu.txt
    
    echo ""
    echo "💡 To install CUDA packages later:"
    echo "   1. Install CUDA toolkit: conda install nvidia::cuda-toolkit=12.1"
    echo "   2. Re-run: pip install -r requirements-mamba.txt"
fi

echo ""
echo "Installing additional evaluation packages..."
pip install seaborn

echo ""
echo "🎉 Installation summary:"
echo "✅ Base dependencies installed successfully"
if [[ "$OS" == "Darwin" ]] || ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA packages skipped (not available on this system)"
    echo "✅ CPU-compatible packages installed"
else
    echo "✅ CUDA packages installation attempted"
fi
echo "✅ Additional evaluation packages installed" 