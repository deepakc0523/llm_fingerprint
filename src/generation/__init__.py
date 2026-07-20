"""
src.generation
==============
Stage 2: LLM Synthetic Text Generation Package.

Supports two interchangeable inference backends, selectable via generation.yaml:
  backend: vllm          → high-throughput production inference (vLLM)
  backend: transformers  → fallback / debug (HuggingFace Transformers)
"""

from src.generation.generator import LLMGeneratorPipeline
from src.generation.model_loader import ModelLoaderRegistry
from src.generation.prompt_formatter import PromptFormatter
from src.generation.parquet_writer import ParquetBatchWriter
from src.generation.checkpoint import CheckpointManager
from src.generation.metadata import MetadataTracker
from src.generation.vllm_loader import load_vllm_model, resolve_quantization, resolve_dtype
from src.generation.vllm_generator import VLLMInferenceBackend, get_gpu_metrics, format_gpu_metrics_str

__all__ = [
    "LLMGeneratorPipeline",
    "ModelLoaderRegistry",
    "PromptFormatter",
    "ParquetBatchWriter",
    "CheckpointManager",
    "MetadataTracker",
    "load_vllm_model",
    "resolve_quantization",
    "resolve_dtype",
    "VLLMInferenceBackend",
    "get_gpu_metrics",
    "format_gpu_metrics_str",
]
