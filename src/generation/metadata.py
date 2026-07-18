"""
src.generation.metadata
=======================
Handles collecting and writing metadata summaries for LLM generation runs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict
import torch

_log = logging.getLogger(__name__)


def collect_environment_info() -> Dict[str, Any]:
    """
    Collects reproducibility and environment telemetry metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary of environment details.
    """
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

    try:
        import transformers
        env_info["transformers_version"] = transformers.__version__
    except ImportError:
        env_info["transformers_version"] = None

    if torch.cuda.is_available():
        try:
            device_id = torch.cuda.current_device()
            env_info["gpu_name"] = torch.cuda.get_device_name(device_id)
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            # Convert bytes to MiB/GiB
            env_info["total_vram_gib"] = round(total_memory / (1024 ** 3), 2)
            # Fetch currently available VRAM
            allocated = torch.cuda.memory_allocated(device_id)
            env_info["free_vram_gib"] = round((total_memory - allocated) / (1024 ** 3), 2)
        except Exception as e:
            _log.warning("Could not extract GPU metrics: %s", e)
            env_info["gpu_name"] = "Unknown CUDA Device"
            env_info["total_vram_gib"] = None
            env_info["free_vram_gib"] = None
    elif hasattr(torch, "mps") and torch.mps.is_available():
        env_info["gpu_name"] = "Apple Silicon MPS"
        env_info["total_vram_gib"] = None
        env_info["free_vram_gib"] = None
    else:
        env_info["gpu_name"] = "CPU Execution"
        env_info["total_vram_gib"] = 0.0
        env_info["free_vram_gib"] = 0.0

    return env_info


class MetadataTracker:
    """Tracks generation performance, metrics, and parameters, saving reports to metadata.json."""

    def __init__(self, metadata_path: str) -> None:
        """
        Parameters
        ----------
        metadata_path:
            The output path for metadata (e.g., 'data/synthetic/llama3/metadata.json').
        """
        self.metadata_path = metadata_path
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.total_records_generated: int = 0
        self.total_generation_time_sec: float = 0.0

    def start_run(self) -> None:
        """Records the start of a generation run."""
        self.start_time = time.time()
        self.total_records_generated = 0
        self.total_generation_time_sec = 0.0

    def record_batch(self, batch_size: int, generation_time: float) -> None:
        """Accumulates run statistics for a completed batch."""
        self.total_records_generated += batch_size
        self.total_generation_time_sec += generation_time

    def end_run(self) -> None:
        """Records the end of a generation run."""
        self.end_time = time.time()

    def save_metadata(self, run_config: Dict[str, Any], extra_stats: Dict[str, Any] | None = None) -> None:
        """
        Saves run summary details to metadata.json.

        Parameters
        ----------
        run_config:
            The parsed generation configuration parameters.
        extra_stats:
            Additional telemetry data.
        """
        extra_stats = extra_stats or {}
        elapsed_realtime = (self.end_time - self.start_time) if self.end_time > 0 else (time.time() - self.start_time)
        
        # Calculate throughput
        records_per_sec = (self.total_records_generated / elapsed_realtime) if elapsed_realtime > 0 else 0.0

        # Collect platform reproducibility variables
        env_info = collect_environment_info()

        metadata = {
            "model_name": run_config.get("model"),
            "model_path": run_config.get("model_paths", {}).get(run_config.get("model")),
            "prompt_mode": run_config.get("prompt_mode", "raw"),
            "parameters": {
                "temperature": run_config.get("temperature"),
                "top_p": run_config.get("top_p"),
                "max_new_tokens": run_config.get("max_new_tokens"),
                "batch_size": run_config.get("batch_size"),
            },
            "metrics": {
                "total_records_generated": self.total_records_generated,
                "total_generation_time_seconds": self.total_generation_time_sec,
                "elapsed_realtime_seconds": elapsed_realtime,
                "records_per_second": records_per_sec,
            },
            "system_info": {
                "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.start_time)),
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.end_time if self.end_time > 0 else time.time())),
                "gpu_name": env_info.get("gpu_name"),
                "available_vram_gib": env_info.get("free_vram_gib"),
                "total_vram_gib": env_info.get("total_vram_gib"),
                "pytorch_version": env_info.get("pytorch_version"),
                "transformers_version": env_info.get("transformers_version"),
                "cuda_version": env_info.get("cuda_version"),
            },
        }

        # Merge in extra stats
        metadata.update(extra_stats)

        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        try:
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            _log.info("Saved metadata report to %s", self.metadata_path)
        except Exception as e:
            _log.error("Failed to write metadata file %s: %s", self.metadata_path, e)
