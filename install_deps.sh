#!/bin/bash

echo "Installing ONC SSAMBA dependencies..."

# Detect operating system
OS=$(uname -s)
echo "Detected OS: $OS"

# Step 1: Install the package with base dependencies
echo ""
echo "Step 1: Installing base package..."
pip install -e .

# Step 2: Check if we should try to install CUDA-dependent packages
if [[ "$OS" == "Darwin" ]]; then
    echo ""
    echo "⚠️  WARNING: You are on macOS!"
    echo "   CUDA is not supported on macOS, so causal_conv1d and mamba_ssm cannot be installed."
    echo ""
    echo "✅ Installation complete (CPU-only mode)"
    echo "   You can use:"
    echo "   - Supervised learning scripts (run_supervised.py)"
    echo "   - Evaluation tools (eval/evaluate_model.py)"
    echo "   - Data processing tools"
    echo ""
    echo "   ❌ SSAMBA model (run_amba_spectrogram.py) will NOT work"
    echo "      as it requires causal_conv1d and mamba_ssm packages."
    
elif command -v nvcc &> /dev/null; then
    echo ""
    echo "✅ NVCC found! Building CUDA-dependent packages from source..."
    echo "   This may take 10-30 minutes (compiling CUDA kernels)..."
    echo ""
    
    # Build causal_conv1d from source
    echo "Building causal_conv1d from source..."
    pip install --no-cache-dir --no-binary causal_conv1d causal_conv1d || {
        echo "⚠️  causal_conv1d build failed."
    }
    
    # Build mamba_ssm from source (this is the slow part)
    echo ""
    echo "Building mamba_ssm from source (this takes a while)..."
    pip install --no-cache-dir --no-binary mamba_ssm mamba_ssm==2.2.5 || {
        echo ""
        echo "⚠️  mamba_ssm build failed."
        echo "   You may need to install CUDA toolkit:"
        echo "   conda install nvidia::cuda-toolkit=12.1"
        echo "   or ensure nvcc version matches your torch CUDA version."
    }
    
else
    echo ""
    echo "⚠️  WARNING: NVCC not found!"
    echo "   This usually means CUDA toolkit is not installed."
    echo "   Base package installed without Mamba support."
    echo ""
    echo "💡 To install CUDA packages later:"
    echo "   1. Install CUDA toolkit (must match your torch CUDA version)"
    echo "   2. Re-run: pip install --no-cache-dir --no-binary mamba_ssm mamba_ssm==2.2.5"
fi

echo ""
echo "🎉 Installation summary:"
echo "✅ onc_ssamba package installed"
if [[ "$OS" == "Darwin" ]] || ! command -v nvcc &> /dev/null; then
    echo "⚠️  CUDA/Mamba packages skipped (not available on this system)"
else
    echo "✅ CUDA/Mamba packages built from source"
fi
echo ""
echo "📦 Package import: from onc_ssamba import ONCSpectrogramDataset, create_model"