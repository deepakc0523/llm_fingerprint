#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_colab.sh
# ==============================================================================
# Automatically sets up Google Colab environment:
#   1. Installs pinned python dependencies
#   2. Installs compatible vLLM release
#   3. Verifies CUDA/GPU availability
#   4. Performs environmental validation check
# ==============================================================================

set -eo pipefail

echo "============================================================"
echo " Starting Google Colab Setup for Fingerprint Pipeline"
echo "============================================================"

# 1. Verify CUDA and GPU existence
echo "Checking CUDA GPU device..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: No NVIDIA GPU detected or nvidia-smi not available!"
    echo "Please change your Colab runtime type to T4 GPU, L4 GPU, or A100."
    exit 1
fi
nvidia-smi -L

# 2. Install pinned dependencies from requirements.txt
echo "Installing pinned system dependencies..."
pip install -r requirements.txt --quiet

# 3. Install compatible vLLM release
# vllm==0.10.1.1 is fully compatible with torch>=2.4.0 and transformers>=4.45.2 on Colab
echo "Installing vLLM engine..."
pip install vllm==0.10.1.1 --quiet

# 4. Verify environment status
echo "Verifying environment compatibility..."
python scripts/verify_environment.py

echo "============================================================"
echo " 🎉 Environment Ready!"
echo "============================================================"
