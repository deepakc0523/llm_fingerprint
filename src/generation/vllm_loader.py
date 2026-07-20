"""
src.generation.vllm_loader
==========================
vLLM model loader with automatic quantization and dtype resolution.

Supports:
  - AWQ, GPTQ, FP16 quantization (auto-detected or explicit via generation.yaml)
  - BF16 / FP16 dtype auto-selection based on GPU capability
  - Graceful prefix-caching fallback when not supported by the model/vLLM version

This module is intentionally separate from model_loader.py (Transformers backend)
so both loaders remain fully independent and the backend switch requires zero code edits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

_log = logging.getLogger(__name__)


def resolve_quantization(model_path: str, config: Dict[str, Any]) -> Optional[str]:
    """
    Resolves the quantization mode from config and model path hints.

    Priority
    --------
    1. Explicit ``"awq"`` or ``"gptq"``  → used directly.
    2. ``"auto"``                         → prefer AWQ, then GPTQ, detected from
                                            model_path substrings.  Falls back to
                                            ``None`` (FP16) if neither is found.
    3. ``null`` / ``None``               → ``None`` (no quantization, FP16 / default dtype).

    Parameters
    ----------
    model_path:
        HuggingFace model ID or absolute local path string.
    config:
        Parsed generation.yaml dictionary.

    Returns
    -------
    Optional[str]
        One of ``"awq"``, ``"gptq"``, or ``None``.
    """
    quant = config.get("quantization", "auto")

    if quant is None:
        _log.info("Quantization: None (FP16 / default dtype)")
        return None

    quant_str = str(quant).lower().strip()

    if quant_str in ("awq", "gptq"):
        _log.info("Quantization: explicit %s", quant_str.upper())
        return quant_str

    if quant_str == "auto":
        model_lower = model_path.lower()
        if "awq" in model_lower:
            _log.info("Quantization: auto → AWQ detected in model path")
            return "awq"
        if "gptq" in model_lower:
            _log.info("Quantization: auto → GPTQ detected in model path")
            return "gptq"
        _log.info(
            "Quantization: auto → no quantized checkpoint detected in '%s'. "
            "Using FP16 / default dtype.",
            model_path,
        )
        return None

    _log.warning(
        "Unknown quantization value '%s'. Falling back to None (FP16).", quant
    )
    return None


def resolve_dtype(config: Dict[str, Any]) -> str:
    """
    Resolves the compute dtype for the vLLM engine.

    - ``"auto"``      → ``"bfloat16"`` if GPU supports BF16, otherwise ``"float16"``.
    - Explicit        → passed through as-is (``"float16"``, ``"bfloat16"``, ``"float32"``).

    Parameters
    ----------
    config:
        Parsed generation.yaml dictionary.

    Returns
    -------
    str
        dtype string accepted by vllm.LLM (e.g. ``"float16"``, ``"bfloat16"``).
    """
    dtype = config.get("dtype", "auto")

    if dtype is None or str(dtype).lower() == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            _log.info("Dtype: auto → bfloat16 (GPU supports BF16)")
            return "bfloat16"
        _log.info("Dtype: auto → float16 (BF16 not supported or no CUDA device)")
        return "float16"

    dtype_str = str(dtype).lower().strip()
    if dtype_str in ("float16", "bfloat16", "float32"):
        _log.info("Dtype: explicit %s", dtype_str)
        return dtype_str

    _log.warning("Unknown dtype '%s'. Defaulting to float16.", dtype)
    return "float16"


def check_prefix_caching_supported() -> bool:
    """
    Checks if vLLM prefix caching is supported in the current environment.

    Prefix caching requires:
      1. A CUDA-capable GPU (not supported on CPU or MPS backends).
      2. Support in the installed vLLM version (i.e. 'enable_prefix_caching'
         is a valid argument of EngineArgs / LLM).
    """
    if not torch.cuda.is_available():
        _log.warning("CUDA is not available. Prefix caching is unsupported.")
        return False

    try:
        from vllm.engine.arg_utils import EngineArgs
        import inspect
        sig = inspect.signature(EngineArgs.__init__)
        if "enable_prefix_caching" in sig.parameters:
            return True
        _log.warning("enable_prefix_caching parameter not found in EngineArgs signature.")
    except Exception as exc:
        _log.warning("Could not verify prefix caching support via vLLM signature: %s", exc)

    return False


def load_vllm_model(model_path: str, config: Dict[str, Any]):
    """
    Creates and returns a configured vLLM ``LLM`` engine instance.

    Attempt order
    -------------
    1. With prefix caching enabled (if ``use_prefix_caching: true`` in config).
    2. Without prefix caching (graceful fallback if the model / vLLM version
       does not support it — logs a warning and retries silently).

    Parameters
    ----------
    model_path:
        HuggingFace model ID or absolute local path.
    config:
        Parsed generation.yaml dictionary.

    Returns
    -------
    vllm.LLM
        The initialized vLLM engine.

    Raises
    ------
    ImportError
        If vLLM is not installed.
    RuntimeError
        If the engine fails to initialize even without prefix caching.
    """
    # 1. Apply a robust monkey patch for the Hugging Face Transformers tokenizer.
    # Newer transformers versions (>= 4.54.0 / 5.x) removed `all_special_tokens_extended`.
    # Older vLLM versions (e.g. 0.10.x/0.11.x) access this attribute during tokenizer loading.
    # Dynamically patching it on PreTrainedTokenizerBase prevents AttributeError.
    try:
        import transformers
        if hasattr(transformers.tokenization_utils_base, "PreTrainedTokenizerBase"):
            if not hasattr(transformers.tokenization_utils_base.PreTrainedTokenizerBase, "all_special_tokens_extended"):
                _log.info(
                    "Monkey-patching PreTrainedTokenizerBase to add missing "
                    "'all_special_tokens_extended' attribute."
                )
                transformers.tokenization_utils_base.PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens
                )
    except Exception as exc:
        _log.warning("Failed to apply transformers tokenizer monkey-patch: %s", exc)

    try:
        from vllm import LLM
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed.  Install it with:\n"
            "    pip install vllm\n"
            "Or switch to the Transformers backend in generation.yaml:\n"
            "    backend: transformers"
        ) from exc

    quantization = resolve_quantization(model_path, config)
    dtype = resolve_dtype(config)
    gpu_memory_utilization = float(config.get("gpu_memory_utilization", 0.90))
    tensor_parallel_size = int(config.get("tensor_parallel_size", 1))
    trust_remote_code = bool(config.get("trust_remote_code", True))
    use_prefix_caching = bool(config.get("use_prefix_caching", True))

    # Pre-validate prefix caching support before engine initialization.
    if use_prefix_caching:
        if not check_prefix_caching_supported():
            _log.warning(
                "Prefix caching is unsupported in this environment. "
                "Disabling prefix caching completely before engine initialization."
            )
            use_prefix_caching = False

    _log.info(
        "Initializing vLLM engine | model=%s | quantization=%s | dtype=%s | "
        "gpu_memory_utilization=%.2f | tensor_parallel_size=%d | "
        "trust_remote_code=%s | prefix_caching=%s",
        model_path,
        quantization if quantization else "None (FP16)",
        dtype,
        gpu_memory_utilization,
        tensor_parallel_size,
        trust_remote_code,
        use_prefix_caching,
    )

    base_kwargs: Dict[str, Any] = {
        "model": model_path,
        "trust_remote_code": trust_remote_code,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
    }
    if quantization is not None:
        base_kwargs["quantization"] = quantization

    # Attempt 1: with prefix caching (if requested by config and verified supported)
    if use_prefix_caching:
        try:
            llm = LLM(**base_kwargs, enable_prefix_caching=True)
            _log.info("vLLM engine loaded successfully — prefix caching ENABLED.")
            return llm
        except Exception as exc:
            _log.warning(
                "Prefix caching not supported for this model/vLLM version: %s. "
                "Retrying without prefix caching.",
                exc,
            )

    # Attempt 2: without prefix caching
    llm = LLM(**base_kwargs)
    _log.info("vLLM engine loaded successfully — prefix caching DISABLED.")
    return llm
