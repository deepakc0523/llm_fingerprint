#!/usr/bin/env python3
"""
scripts/smoke_test.py
=====================
Automated end-to-end smoke test for the Fingerprint text generation pipeline.
Runs a 10-sample generation, verifies check-pointing, Parquet writing, metadata,
and resume capability. Works for both "vllm" and "transformers" backends.
"""

import sys
import os
import shutil
import time
import pandas as pd
import pyarrow.parquet as pq

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset.utils import load_yaml_config
from src.generation.generator import LLMGeneratorPipeline

def cleanup_test_dir(test_dir: str):
    if os.path.exists(test_dir):
        try:
            shutil.rmtree(test_dir)
            print(f"Cleaned up temporary test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {test_dir}: {e}")

def create_mock_prefixes(prefix_path: str, count: int = 15):
    os.makedirs(os.path.dirname(prefix_path), exist_ok=True)
    records = []
    for i in range(count):
        records.append({
            "prefix_id": f"prefix_{i:04d}",
            "dataset_name": "smoke_test_dataset",
            "category": "testing",
            "prefix_text": f"This is prefix prompt number {i}. The story continues from here:"
        })
    df = pd.DataFrame(records)
    df.to_parquet(prefix_path, index=False)
    print(f"Created mock prefix cache at {prefix_path} with {count} samples.")

def run_test():
    print("=" * 70)
    print(" Running Fingerprint Synthetic Generation Smoke Test")
    print("=" * 70)

    # 1. Setup temporary configurations and directories
    test_dir = os.path.join(os.getcwd(), "data_smoke_test")
    cleanup_test_dir(test_dir)
    
    mock_input_parquet = os.path.join(test_dir, "prefixes", "prefix_cache.parquet")
    create_mock_prefixes(mock_input_parquet, count=15)

    # Load original config and patch it for smoke test
    try:
        orig_config = load_yaml_config("configs/generation.yaml")
    except Exception as e:
        print(f"[ERROR] Failed to load configs/generation.yaml: {e}")
        sys.exit(1)

    # Run on CPU/MPS fallback if CUDA is not available
    import torch
    backend = orig_config.get("backend", "vllm")
    if not torch.cuda.is_available():
        print("[WARNING] No CUDA GPU detected. Forcing backend to 'transformers' and device to 'cpu' for smoke test.")
        backend = "transformers"
        device = "cpu"
    else:
        device = orig_config.get("device", "auto")

    # Create patched generation config dict
    test_config = orig_config.copy()
    test_config.update({
        "backend": backend,
        "device": device,
        "input_file": mock_input_parquet,
        "output_directory": os.path.join(test_dir, "synthetic"),
        "checkpoint_frequency": 3,
        "max_samples": 10,
        "batch_size": 4 if backend == "transformers" else 4,  # Keep batch sizes low for quick test
    })

    model_alias = test_config.get("model", "qwen2_5")
    dest_parquet = os.path.join(test_config["output_directory"], model_alias, "generated.parquet")
    dest_checkpoint = os.path.join(test_config["output_directory"], model_alias, "checkpoint.json")
    dest_metadata = os.path.join(test_config["output_directory"], model_alias, "metadata.json")

    # 2. RUN PART 1: Generate first batch of samples (up to 10 max_samples limit)
    print(f"\n--- Part 1: Generating 10 samples (Backend: {backend}) ---")
    try:
        pipeline = LLMGeneratorPipeline(test_config)
        pipeline.run()
    except Exception as e:
        print(f"[ERROR] Execution failed during generation: {e}")
        cleanup_test_dir(test_dir)
        sys.exit(1)

    # 3. VERIFY GENERATION OUTPUTS
    print("\n--- Verifying Part 1 Outputs ---")
    if not os.path.exists(dest_parquet):
        print("[ERROR] Parquet output file was not created!")
        sys.exit(1)
    
    # Read Parquet
    df_out = pd.read_parquet(dest_parquet)
    print(f"Parquet file contains {len(df_out)} rows.")
    if len(df_out) != 10:
        print(f"[ERROR] Expected 10 rows, got {len(df_out)}")
        sys.exit(1)
        
    expected_cols = [
        "prefix_id", "dataset_name", "category", "human_prefix", "generated_text",
        "model_name", "temperature", "top_p", "max_new_tokens", "prompt_length",
        "completion_length", "generation_time", "timestamp"
    ]
    for col in expected_cols:
        if col not in df_out.columns:
            print(f"[ERROR] Missing column '{col}' in output schema!")
            sys.exit(1)
    print("[SUCCESS] Schema columns verified successfully.")

    if not os.path.exists(dest_checkpoint):
        print("[ERROR] Checkpoint file was not created!")
        sys.exit(1)
    print("[SUCCESS] Checkpoint file exists.")

    if not os.path.exists(dest_metadata):
        print("[ERROR] Metadata file was not created!")
        sys.exit(1)
    print("[SUCCESS] Metadata file exists.")

    # 4. RUN PART 2: Resume capability test
    # Increase max_samples to 12. Since 10 are completed, it should only generate 2 more.
    print(f"\n--- Part 2: Resume verification (Scaling max_samples to 12) ---")
    test_config["max_samples"] = 12
    
    try:
        pipeline2 = LLMGeneratorPipeline(test_config)
        pipeline2.run()
    except Exception as e:
        print(f"[ERROR] Execution failed during resume generation: {e}")
        cleanup_test_dir(test_dir)
        sys.exit(1)

    # Verify resume output
    df_resume = pd.read_parquet(dest_parquet)
    print(f"After resume run, parquet contains {len(df_resume)} rows.")
    if len(df_resume) != 12:
        print(f"[ERROR] Expected 12 rows in total after resuming, got {len(df_resume)}")
        sys.exit(1)
        
    # Check that we didn't duplicate prompt completions
    unique_ids = df_resume["prefix_id"].nunique()
    if unique_ids != 12:
        print(f"[ERROR] Duplicate prefix_ids found: {unique_ids} unique out of {len(df_resume)}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print(" [SUCCESS] SMOKE TEST COMPLETED SUCCESSFULLY!")
    print(" All systems operational (Parquet, Checkpoint, Metadata, Resume).")
    print("=" * 70)
    cleanup_test_dir(test_dir)

if __name__ == "__main__":
    run_test()
