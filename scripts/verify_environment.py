#!/usr/bin/env python3
"""
scripts/verify_environment.py
==============================
Verifies the local or cloud runtime environment.
Prints details about Python, CUDA, GPU VRAM, PyTorch, Transformers, and vLLM.
Performs checks and fails fast (exit code 1) if requirements aren't met.
"""

import sys
import os
import platform

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def main():
    print_section("Fingerprint Environment Verification")
    
    # 1. Python version
    python_ver = platform.python_version()
    print(f"Python Version:         {python_ver}")
    
    # 2. PyTorch & CUDA Info
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda if cuda_avail else "N/A"
        bf16_supported = False
        if cuda_avail:
            try:
                bf16_supported = torch.cuda.is_bf16_supported()
            except Exception:
                pass
        
        print(f"PyTorch Version:        {torch_ver}")
        print(f"CUDA Available:         {cuda_avail}")
        print(f"CUDA Version:           {cuda_ver}")
        print(f"BF16 Supported:         {bf16_supported}")
    except ImportError:
        print("[ERROR] PyTorch is NOT installed!")
        sys.exit(1)
        
    # 3. GPU/VRAM Details
    gpu_name = "N/A"
    vram_total_gib = 0.0
    if cuda_avail:
        try:
            device_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_id)
            total_mem = torch.cuda.get_device_properties(device_id).total_memory
            vram_total_gib = round(total_mem / (1024 ** 3), 2)
            print(f"GPU Device Name:        {gpu_name}")
            print(f"Total VRAM:             {vram_total_gib} GiB")
        except Exception as e:
            print(f"[WARNING] Could not fetch GPU/VRAM details: {e}")
    else:
        print("GPU Device Name:        N/A (CPU Execution Mode)")
        
    # 4. Hugging Face Transformers version
    try:
        import transformers
        print(f"Transformers Version:   {transformers.__version__}")
    except ImportError:
        print("[ERROR] Transformers is NOT installed!")
        sys.exit(1)
        
    # 5. vLLM Version
    vllm_installed = False
    vllm_ver = "N/A"
    try:
        import vllm
        vllm_installed = True
        vllm_ver = vllm.__version__
        print(f"vLLM Version:           {vllm_ver}")
    except ImportError:
        print("vLLM Version:           Not Installed")

    # ── VALIDATION CHECKS ──
    print_section("Status Summary & Validation")
    
    errors = []
    
    # Parse Python version
    py_major, py_minor, _ = map(int, python_ver.split(".")[:3])
    if py_major < 3 or (py_major == 3 and py_minor < 8):
        errors.append("Python version must be 3.8 or higher.")
        
    # PyTorch validation
    if not cuda_avail:
        print("[WARNING] Warning: No CUDA-capable GPU detected. vLLM backend will NOT function.")
        print("   Only 'backend: transformers' with device='cpu' or 'mps' can be used.")
    else:
        if vram_total_gib < 10.0:
            print("[WARNING] Warning: GPU VRAM is below 10 GiB. Running models larger than 3B might trigger OOMs.")
            
    # Print status
    if errors:
        print("[ERROR] ENVIRONMENT INCOMPATIBLE:")
        for err in errors:
            print(f"   - {err}")
        sys.exit(1)
    else:
        print("[SUCCESS] Core environment checks passed.")
        if vllm_installed and cuda_avail:
            print("[SUCCESS] Production ready: vLLM backend is supported.")
        else:
            print("[SUCCESS] Development mode: Transformers backend fallback is supported.")
        print("=" * 60)

if __name__ == "__main__":
    main()
