"""
src.generation.vllm_generator
==============================
vLLM inference backend for the Stage 2 Fingerprint generation pipeline.

Wraps ``vllm.LLM`` + ``SamplingParams`` into the same ``GeneratedRecord``
interface used by the Transformers backend, enabling fully transparent backend
switching via a single line in ``generation.yaml``.

Key features
------------
Continuous batching
    vLLM handles all internal GPU micro-batching and KV-cache management.
    The ``batch_size`` config key controls the *submission chunk size* — how
    many prompts are submitted per ``llm.generate()`` call for progress
    tracking granularity; it does not affect GPU saturation.

Adaptive OOM recovery
    On ``CUDA out of memory``, the chunk size is automatically halved:
        256 → 128 → 64 → 32 → 16
    The adjustment is logged and persisted across batches so subsequent
    calls also use the reduced size, without any manual intervention.

GPU telemetry
    ``get_gpu_metrics()`` exposes VRAM usage and GPU utilisation (%) per
    checkpoint interval for the enhanced logging in ``generator.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.generation.parquet_writer import GeneratedRecord

_log = logging.getLogger(__name__)

# Minimum allowed chunk size before OOM is re-raised
_MIN_CHUNK_SIZE: int = 16


# ──────────────────────────────────────────────────────────────────────────────
# GPU telemetry helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_gpu_metrics() -> Dict[str, Any]:
    """
    Collects GPU VRAM and utilisation metrics for telemetry logging.

    Returns
    -------
    Dict[str, Any]
        Keys: ``vram_used_gib``, ``vram_total_gib``, ``vram_used_pct``,
        ``gpu_utilization_pct``.  Values are ``None`` when not available.
    """
    metrics: Dict[str, Any] = {
        "vram_used_gib": None,
        "vram_total_gib": None,
        "vram_used_pct": None,
        "gpu_utilization_pct": None,
    }

    if not torch.cuda.is_available():
        return metrics

    try:
        device_id = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_id)
        total_bytes = props.total_memory
        allocated_bytes = torch.cuda.memory_allocated(device_id)

        metrics["vram_total_gib"] = round(total_bytes / (1024 ** 3), 2)
        metrics["vram_used_gib"] = round(allocated_bytes / (1024 ** 3), 2)
        if total_bytes > 0:
            metrics["vram_used_pct"] = round(allocated_bytes / total_bytes * 100, 1)
    except Exception as exc:
        _log.debug("Could not collect VRAM metrics: %s", exc)

    try:
        # torch.cuda.utilization() returns GPU SM utilisation %
        metrics["gpu_utilization_pct"] = torch.cuda.utilization()
    except Exception as exc:
        _log.debug("Could not collect GPU utilisation: %s", exc)

    return metrics


def format_gpu_metrics_str(gpu_info: Dict[str, Any]) -> str:
    """Formats GPU metrics into a compact, log-friendly single line."""
    vram_used = gpu_info.get("vram_used_gib")
    vram_total = gpu_info.get("vram_total_gib")
    vram_pct = gpu_info.get("vram_used_pct")
    util_pct = gpu_info.get("gpu_utilization_pct")

    vram_str = (
        f"{vram_used:.2f}/{vram_total:.2f} GiB ({vram_pct:.1f}%)"
        if (vram_used is not None and vram_total is not None and vram_pct is not None)
        else "N/A"
    )
    util_str = f"{util_pct}%" if util_pct is not None else "N/A"
    return f"VRAM: {vram_str} | GPU util: {util_str}"


# ──────────────────────────────────────────────────────────────────────────────
# vLLM inference backend
# ──────────────────────────────────────────────────────────────────────────────

class VLLMInferenceBackend:
    """
    Wraps a vLLM ``LLM`` engine for batch inference, producing
    ``GeneratedRecord`` objects that are 100% compatible with the existing
    ``ParquetBatchWriter`` schema.

    Adaptive chunk size
    -------------------
    On ``CUDA out of memory``, the chunk size is automatically halved and
    generation is retried transparently:

        256  →  128  →  64  →  32  →  16  →  RuntimeError (re-raised)

    The effective chunk size is persisted across batches so all subsequent
    calls benefit from the already-reduced size, avoiding repeated OOM probing.
    """

    def __init__(
        self,
        llm,
        config: Dict[str, Any],
        model_name: str,
        quantization: Optional[str],
        dtype: str,
        tokenizer_for_chat=None,
    ) -> None:
        """
        Parameters
        ----------
        llm:
            Initialised ``vllm.LLM`` engine instance.
        config:
            Parsed ``generation.yaml`` dictionary.
        model_name:
            Model alias used for record metadata (e.g. ``"qwen2_5"``).
        quantization:
            Resolved quantisation string (``"awq"``, ``"gptq"``, or ``None``).
        dtype:
            Resolved dtype string (``"float16"`` or ``"bfloat16"``).
        tokenizer_for_chat:
            Optional HuggingFace ``PreTrainedTokenizer`` loaded **without model
            weights** — used solely for ``apply_chat_template()`` when
            ``prompt_mode: chat``.  Unused for actual inference.
        """
        from vllm import SamplingParams

        self.llm = llm
        self.config = config
        self.model_name = model_name
        self.quantization = quantization
        self.dtype = dtype
        self.tokenizer_for_chat = tokenizer_for_chat

        # Sampling hyperparameters
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.9))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.seed = int(config.get("seed", 42))

        self.sampling_params = SamplingParams(
            temperature=self.temperature if self.temperature > 0 else 0.0,
            top_p=self.top_p if self.temperature > 0 else 1.0,
            max_tokens=self.max_new_tokens,
            seed=self.seed,
        )

        # Adaptive chunk size — starts at configured batch_size, halves on OOM
        configured_chunk = int(config.get("batch_size", 256))
        self._chunk_size: int = max(configured_chunk, _MIN_CHUNK_SIZE)

        _log.info(
            "VLLMInferenceBackend ready | model=%s | quantization=%s | dtype=%s | "
            "temp=%.2f | top_p=%.2f | max_tokens=%d | seed=%d | chunk_size=%d",
            model_name,
            quantization if quantization else "None (FP16)",
            dtype,
            self.temperature,
            self.top_p,
            self.max_new_tokens,
            self.seed,
            self._chunk_size,
        )

    @property
    def effective_chunk_size(self) -> int:
        """Current adaptive chunk size, updated after any OOM reductions."""
        return self._chunk_size

    def generate_batch(
        self,
        batch_rows: List[Dict[str, Any]],
        formatted_prompts: List[str],
    ) -> Tuple[List[GeneratedRecord], int]:
        """
        Generates completions for a batch of prompts via vLLM.

        Implements adaptive OOM recovery: on ``CUDA out of memory``, the chunk
        size is halved and generation retried automatically.

        Parameters
        ----------
        batch_rows:
            List of prefix row dicts with keys ``prefix_id``, ``dataset_name``,
            ``category``, and ``prefix_text``.
        formatted_prompts:
            Formatted prompt strings aligned index-for-index with ``batch_rows``.

        Returns
        -------
        Tuple[List[GeneratedRecord], int]
            ``(generated_records, total_tokens_generated)``

        Raises
        ------
        RuntimeError
            If generation fails even at the minimum chunk size.
        """
        current_chunk = self._chunk_size

        while current_chunk >= _MIN_CHUNK_SIZE:
            try:
                records, total_tokens = self._run_inference(
                    batch_rows, formatted_prompts, current_chunk
                )
                # Persist successful chunk size for all future batches
                self._chunk_size = current_chunk
                return records, total_tokens

            except RuntimeError as exc:
                err_lower = str(exc).lower()
                if "out of memory" in err_lower or "cuda out of memory" in err_lower:
                    reduced = current_chunk // 2
                    if reduced < _MIN_CHUNK_SIZE:
                        _log.error(
                            "GPU OOM at minimum chunk size %d. Cannot recover further. "
                            "Consider reducing gpu_memory_utilization in generation.yaml.",
                            _MIN_CHUNK_SIZE,
                        )
                        raise
                    _log.warning(
                        "⚠  GPU OOM at chunk_size=%d — reducing to chunk_size=%d and retrying. "
                        "This setting will persist for all remaining batches.",
                        current_chunk,
                        reduced,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    current_chunk = reduced
                else:
                    raise

        raise RuntimeError(
            f"vLLM generation failed: exhausted all chunk sizes down to {_MIN_CHUNK_SIZE}."
        )

    def _run_inference(
        self,
        batch_rows: List[Dict[str, Any]],
        formatted_prompts: List[str],
        chunk_size: int,
    ) -> Tuple[List[GeneratedRecord], int]:
        """
        Submits prompts to the vLLM engine in sub-chunks and assembles records.

        vLLM handles all internal micro-batching and KV-cache management.
        Sub-chunking here provides OOM-recovery granularity only.

        Parameters
        ----------
        batch_rows:
            Aligned list of prefix row dicts.
        formatted_prompts:
            Aligned list of formatted prompt strings.
        chunk_size:
            Maximum prompts per ``llm.generate()`` call.

        Returns
        -------
        Tuple[List[GeneratedRecord], int]
            ``(records, total_tokens_generated)``
        """
        all_records: List[GeneratedRecord] = []
        total_tokens: int = 0
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        n = len(formatted_prompts)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            sub_prompts = formatted_prompts[start:end]
            sub_rows = batch_rows[start:end]

            sub_start = time.perf_counter()
            outputs = self.llm.generate(sub_prompts, self.sampling_params)
            sub_elapsed = time.perf_counter() - sub_start

            per_record_time = sub_elapsed / len(sub_rows) if sub_rows else 0.0

            for i, (row, output) in enumerate(zip(sub_rows, outputs)):
                completion_text = output.outputs[0].text if output.outputs else ""
                completion_tokens = (
                    len(output.outputs[0].token_ids) if output.outputs else 0
                )
                total_tokens += completion_tokens

                record = GeneratedRecord(
                    prefix_id=str(row["prefix_id"]),
                    dataset_name=str(row["dataset_name"]),
                    category=str(row["category"]),
                    human_prefix=str(row["prefix_text"]),
                    generated_text=completion_text,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_new_tokens=self.max_new_tokens,
                    prompt_length=len(sub_prompts[i]),
                    completion_length=len(completion_text),
                    generation_time=per_record_time,
                    timestamp=timestamp,
                )
                all_records.append(record)

        return all_records, total_tokens
