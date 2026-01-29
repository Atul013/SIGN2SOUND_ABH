#!/bin/bash

# Setup Environment Script for Sign2Sound Phase 2
# This script sets up the development environment on Linux/Mac
# For Windows, use PowerShell equivalent or WSL

echo "=========================================="
echo "Sign2Sound Phase 2 - Environment Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Check if Python 3.8+ is installed
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required"
    echo "   Current version: $python_version"
    exit 1
fi

echo "✅ Python version OK"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "✅ Virtual environment recreated"
    else
        echo "Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip
echo "✅ pip upgraded"

# Install dependencies
echo ""
echo "Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating project directories..."
directories=(
    "data/raw"
    "data/processed"
    "data/splits"
    "results/sample_outputs"
    "checkpoints"
    "logs"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  Created: $dir"
    else
        echo "  Exists: $dir"
    fi
done

echo "✅ Directories created"

# Download MediaPipe models (if needed)
echo ""
echo "Checking MediaPipe models..."
if [ ! -f "models/hand_landmarker.task" ]; then
    echo "⚠️  Hand landmarker model not found"
    echo "   Please download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    echo "   Save to: models/hand_landmarker.task"
else
    echo "✅ Hand landmarker model found"
fi

# Check GPU availability (optional)
echo ""
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✅ GPU available"
else
    echo "⚠️  No NVIDIA GPU detected (CPU mode)"
fi

# Run tests to verify installation
echo ""
echo "Running verification tests..."
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python -c "import mediapipe as mp; print(f'MediaPipe version: {mp.__version__}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Download datasets (run scripts/download_datasets.sh)"
echo "  2. Preprocess data (python preprocessing/preprocess.py)"
echo "  3. Train model (python training/train.py)"
echo "  4. Run inference (python inference/realtime_demo.py)"
echo ""
