"""
src.generation.generator
========================
The main orchestrator pipeline for Stage 2 LLM text generation.
Reads prefix caches, feeds prompts to causal models, structures outputs
according to a strict schema, writes logs, and supports resuming via checkpoints.
Optimized with robust batch recovery, environment telemetry, and real-time speed metrics.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from src.dataset.utils import get_logger, load_yaml_config
from src.generation.checkpoint import CheckpointManager
from src.generation.generation_utils import set_seed, Timer
from src.generation.metadata import MetadataTracker
from src.generation.model_loader import ModelLoaderRegistry, cleanup_gpu, detect_device
from src.generation.parquet_writer import GeneratedRecord, ParquetBatchWriter
from src.generation.prompt_formatter import PromptFormatter
from src.generation.vllm_loader import load_vllm_model, resolve_quantization, resolve_dtype
from src.generation.vllm_generator import VLLMInferenceBackend, get_gpu_metrics, format_gpu_metrics_str


class LLMGeneratorPipeline:
    """Orchestrates the entire Stage 2 synthetic text generation pipeline."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Parameters
        ----------
        config:
            Dictionary containing generation settings, file paths, and hyperparameters.
        """
        self.config = config
        self.model_name = config.get("model", "llama3")
        self.model_path = config.get("model_paths", {}).get(self.model_name)
        if not self.model_path:
            raise ValueError(f"Model path not specified for model key '{self.model_name}'")

        # Resolve execution device
        self.device = config.get("device", "auto")
        if self.device == "auto":
            self.device = detect_device()

        # Configurable output filename (default: "generated.parquet")
        output_filename = config.get("output_filename", "generated.parquet")

        # Setup paths based on requested output directory design
        self.output_dir = os.path.join(config.get("output_directory", "data/synthetic"), self.model_name)
        self.output_parquet = os.path.join(self.output_dir, output_filename)
        self.checkpoint_json = os.path.join(self.output_dir, "checkpoint.json")
        self.metadata_json = os.path.join(self.output_dir, "metadata.json")
        self.log_file = os.path.join(self.output_dir, "generation.log")

        # Initialize structured logger for this run
        self.log = get_logger(
            name=f"fingerprint.generation.{self.model_name}",
            level=config.get("log_level", "INFO"),
            log_file=self.log_file,
        )

        self.log.info("Initialized LLMGeneratorPipeline for model: %s", self.model_name)
        self.log.info("Execution device: %s", self.device)
        self.log.info("Outputs will be written to: %s", self.output_parquet)

        # Seeding
        set_seed(config.get("seed", 42))

        # Hyperparameters
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.9))
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.batch_size = int(config.get("batch_size", 8))
        self.checkpoint_frequency = int(config.get("checkpoint_frequency", 100))
        self.max_samples = config.get("max_samples")  # Maximum prefixes to process

        # Backend selection — controlled entirely by generation.yaml, no code change needed.
        # "vllm"         → high-throughput production inference
        # "transformers" → fallback / debug path
        self.backend = config.get("backend", "transformers").lower()
        if self.backend not in ("vllm", "transformers"):
            _log = logging.getLogger(__name__)
            _log.warning(
                "Unknown backend '%s'. Falling back to 'transformers'.", self.backend
            )
            self.backend = "transformers"
        self.log.info("Inference backend: %s", self.backend)

        # Initialize managers
        self.checkpoint_mgr = CheckpointManager(self.checkpoint_json)
        self.metadata_tracker = MetadataTracker(self.metadata_json)
        self.writer = ParquetBatchWriter(self.output_parquet)

    def validate_setup(self) -> None:
        """
        Validates the configuration, dependencies, model setup, and environment compatibility.
        Fails fast with descriptive errors before loading heavy model weights.
        """
        self.log.info("Running pre-generation validation checks...")

        # 1. Model Path check
        if not self.model_path:
            raise ValueError(f"Model path not specified for alias '{self.model_name}'")

        # 2. Backend dependency and environment compatibility checks
        if self.backend == "vllm":
            try:
                import vllm
            except ImportError:
                raise ImportError(
                    f"vLLM is required when backend='vllm', but it is not installed. "
                    f"Install vllm or switch backend to 'transformers'."
                )
            
            # Verify CUDA for vLLM
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA-capable GPU is required for the vLLM backend, but CUDA is not available. "
                    "Use backend='transformers' with device='cpu' or 'mps' on this environment."
                )

            # Check quantization config
            quant = self.config.get("quantization", "auto")
            if quant is not None:
                quant_str = str(quant).lower().strip()
                if quant_str not in ("auto", "awq", "gptq"):
                    raise ValueError(
                        f"Unsupported vLLM quantization: '{quant}'. "
                        f"Supported values: 'auto', 'awq', 'gptq', or null."
                    )

            # Check dtype config
            dtype = self.config.get("dtype", "auto")
            if dtype is not None:
                dtype_str = str(dtype).lower().strip()
                if dtype_str not in ("auto", "float16", "bfloat16", "float32"):
                    raise ValueError(
                        f"Unsupported vLLM dtype: '{dtype}'. "
                        f"Supported values: 'auto', 'float16', 'bfloat16', 'float32'."
                    )

        elif self.backend == "transformers":
            # Check Transformers quantization options compatibility
            if self.config.get("load_in_8bit") and self.config.get("load_in_4bit"):
                raise ValueError("Cannot load model in 8-bit and 4-bit simultaneously.")
            
            # Check device placement
            if self.device not in ("cuda", "mps", "cpu") and self.device != "auto":
                try:
                    torch.device(self.device)
                except Exception as e:
                    raise ValueError(f"Invalid device specification: '{self.device}'. Error: {e}")

        # 3. Fast tokenizer verification (if path exists locally or reachable online)
        try:
            from transformers import AutoTokenizer
            self.log.info("Verifying tokenizer metadata loading for %s...", self.model_path)
            # Try loading fast tokenizer config only (local/cached check only, doesn't load model weights)
            tokenizer_check = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=self.config.get("trust_remote_code", True),
                local_files_only=bool(os.path.exists(self.model_path))
            )
            self.log.info("✓ Tokenizer metadata verified.")
        except Exception as e:
            self.log.warning(
                "Could not perform pre-run tokenizer check for '%s': %s. "
                "The pipeline will proceed but might fail during full load.",
                self.model_path, e
            )

        self.log.info("✓ Pre-generation verification complete. Setup is valid.")

    def load_prefixes_chunked(self) -> Generator[Dict[str, Any], None, None]:
        """
        Loads prefixes from the input Parquet file in a memory-efficient chunked manner.
        Yields individual prefix dictionaries.

        Yields
        ------
        Dict[str, Any]
            Dictionary representing a single prefix record.
        """
        input_file = self.config.get("input_file", "data/prefixes/prefix_cache.parquet")
        self.log.info("Accessing prefix cache file: %s (chunked mode)", input_file)

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input prefix cache parquet file not found at: {input_file}")

        parquet_file = pq.ParquetFile(input_file)
        
        # Read file row-group by row-group to minimize RAM usage
        for i in range(parquet_file.num_row_groups):
            self.log.debug("Reading row group %d/%d from cache", i + 1, parquet_file.num_row_groups)
            table = parquet_file.read_row_group(i, columns=["prefix_id", "dataset_name", "category", "prefix_text"])
            df = table.to_pandas()
            for record in df.to_dict(orient="records"):
                yield record

    def run(self) -> None:
        """
        Executes the generation loop: loading inputs, preparing the model,
        applying prompt formatting, and generating/checkpointing completions.
        Features robust batch-level exception recovery and real-time throughput metrics.
        """
        # Run pre-generation validation checks to fail fast if config or imports are broken
        self.validate_setup()

        self.log.info("Starting generation pipeline run...")
        
        # Load Checkpoint to check which prefixes are already completed
        self.checkpoint_mgr.load_checkpoint()
        
        # Collect and filter prefixes
        raw_prefixes = []
        try:
            for row in self.load_prefixes_chunked():
                prefix_id = row.get("prefix_id")
                if prefix_id and not self.checkpoint_mgr.is_completed(prefix_id):
                    raw_prefixes.append(row)
                
                # Apply max_samples limit if specified
                if self.max_samples is not None and len(raw_prefixes) >= self.max_samples:
                    self.log.info("Reached maximum requested sample limit of %d", self.max_samples)
                    break
        except Exception as e:
            self.log.error("Failed to load prefix cache: %s", e)
            raise e

        total_to_generate = len(raw_prefixes)
        self.log.info("Total prefixes remaining to generate: %d", total_to_generate)

        if total_to_generate == 0:
            self.log.info("All records already generated or max_samples limit met. Nothing to do!")
            return

        # ── Model Loading (backend-dependent) ────────────────────────────────
        prompt_mode = self.config.get("prompt_mode", "raw")
        vllm_backend_obj: VLLMInferenceBackend | None = None
        model = None
        tokenizer = None

        if self.backend == "vllm":
            self.log.info("Loading vLLM engine...")
            llm = load_vllm_model(self.model_path, self.config)
            self.log.info("vLLM engine loaded successfully.")

            # For chat mode: load the HuggingFace tokenizer (weights NOT loaded)
            # solely for apply_chat_template(). All other modes need no tokenizer.
            if prompt_mode == "chat":
                self.log.info(
                    "prompt_mode='chat' with vLLM backend — loading HF tokenizer "
                    "for chat template application (no model weights loaded)."
                )
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

            _quant = resolve_quantization(self.model_path, self.config)
            _dtype = resolve_dtype(self.config)

            vllm_backend_obj = VLLMInferenceBackend(
                llm=llm,
                config=self.config,
                model_name=self.model_name,
                quantization=_quant,
                dtype=_dtype,
                tokenizer_for_chat=tokenizer,
            )

        else:  # backend: transformers
            self.log.info("Loading Transformers model and tokenizer...")
            model, tokenizer = ModelLoaderRegistry.load_model_and_tokenizer(
                model_name=self.model_name,
                model_path=self.model_path,
                device=self.device,
                use_bf16=self.config.get("use_bf16", True),
                load_in_8bit=self.config.get("load_in_8bit", False),
                load_in_4bit=self.config.get("load_in_4bit", False),
            )
            self.log.info("Transformers model and tokenizer loaded successfully.")

        # ── Prompt Formatter (same interface for both backends) ───────────────
        # tokenizer is None for vLLM + non-chat modes; PromptFormatter handles this gracefully.
        formatter = PromptFormatter(
            tokenizer=tokenizer,
            mode=prompt_mode,
        )

        self.metadata_tracker.start_run()
        
        # Split prefixes into batches
        batches = [raw_prefixes[i : i + self.batch_size] for i in range(0, len(raw_prefixes), self.batch_size)]
        total_batches = len(batches)

        self.log.info(
            "Starting inference loop: processing %d prefixes in %d batches (size=%d)",
            total_to_generate,
            total_batches,
            self.batch_size,
        )

        session_completed_ids = set()
        start_time = time.time()
        processed_records = 0

        # Progress bar configuration
        pbar = tqdm(total=total_to_generate, desc=f"Generating ({self.model_name})")

        for batch_idx, batch_rows in enumerate(batches):
            batch_size_actual = len(batch_rows)
            
            try:
                # Format prompts — identical for both backends
                prefixes_text = [row["prefix_text"] for row in batch_rows]
                formatted_prompts = formatter.format_batch(prefixes_text)

                # ── Inference (backend-dependent) ─────────────────────────────
                with Timer() as timer:
                    if self.backend == "vllm":
                        # vLLM: continuous batching handled internally.
                        # generate_batch() includes adaptive OOM recovery.
                        generated_records, total_tokens_generated = (
                            vllm_backend_obj.generate_batch(batch_rows, formatted_prompts)
                        )

                    else:
                        # ── Transformers path (unchanged) ─────────────────────
                        inputs = tokenizer(
                            formatted_prompts,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                        )
                        input_ids = inputs["input_ids"].to(self.device)
                        attention_mask = inputs["attention_mask"].to(self.device)

                        with torch.no_grad():
                            outputs = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=self.max_new_tokens,
                                temperature=self.temperature if self.temperature > 0 else None,
                                top_p=self.top_p if self.temperature > 0 else None,
                                do_sample=self.temperature > 0,
                                pad_token_id=tokenizer.pad_token_id,
                            )

                        generated_records: List[GeneratedRecord] = []
                        total_tokens_generated = 0

                        for i, row in enumerate(batch_rows):
                            prompt_len_tokens = int(input_ids[i].shape[0])
                            total_len_tokens = int(outputs[i].shape[0])
                            completion_len_tokens = total_len_tokens - prompt_len_tokens
                            total_tokens_generated += completion_len_tokens

                            completion_tokens = outputs[i][prompt_len_tokens:]
                            completion_text = tokenizer.decode(
                                completion_tokens, skip_special_tokens=True
                            )

                            record = GeneratedRecord(
                                prefix_id=row["prefix_id"],
                                dataset_name=row["dataset_name"],
                                category=row["category"],
                                human_prefix=row["prefix_text"],
                                generated_text=completion_text,
                                model_name=self.model_name,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                max_new_tokens=self.max_new_tokens,
                                prompt_length=len(formatted_prompts[i]),
                                completion_length=len(completion_text),
                                generation_time=timer.elapsed / batch_size_actual,
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            )
                            generated_records.append(record)

                # Track completed prefix IDs — unified path for both backends
                for row in batch_rows:
                    session_completed_ids.add(row["prefix_id"])

                # Save batch data
                self.writer.write_batch(generated_records)
                processed_records += batch_size_actual
                
                # Telemetry and logging calculations
                self.metadata_tracker.record_batch(batch_size_actual, timer.elapsed)
                
                elapsed_runtime = time.time() - start_time
                records_per_sec = processed_records / elapsed_runtime if elapsed_runtime > 0 else 0.0
                tokens_per_sec = total_tokens_generated / timer.elapsed if timer.elapsed > 0 else 0.0
                remaining_records = total_to_generate - processed_records
                eta_seconds = remaining_records / records_per_sec if records_per_sec > 0 else 0.0

                self.log.info(
                    "Batch %d/%d completed | Time: %.2fs | Speed: %.2f rec/s | Avg Speed: %.2f tok/s | "
                    "Remaining: %d | Elapsed: %.1fs | ETA: %.1fs",
                    batch_idx + 1,
                    total_batches,
                    timer.elapsed,
                    batch_size_actual / timer.elapsed if timer.elapsed > 0 else 0.0,
                    tokens_per_sec,
                    remaining_records,
                    elapsed_runtime,
                    eta_seconds,
                )

                # Update progress bar description with telemetry
                pbar.set_postfix({
                    "rec/s": f"{records_per_sec:.2f}",
                    "elapsed": f"{elapsed_runtime:.1f}s",
                    "eta": f"{eta_seconds:.1f}s"
                })
                pbar.update(batch_size_actual)

                # Checkpoint saving + enhanced telemetry
                if (batch_idx + 1) % self.checkpoint_frequency == 0 or (batch_idx + 1) == total_batches:
                    self.checkpoint_mgr.save_checkpoint(
                        last_processed_index=self.checkpoint_mgr.last_processed_index + batch_size_actual,
                        new_completed_ids=session_completed_ids,
                        processed_batches=batch_idx + 1,
                        processed_records=processed_records,
                        remaining_records=remaining_records,
                    )
                    session_completed_ids.clear()

                    # ── Checkpoint telemetry (GPU + backend info) ─────────────
                    gpu_info = get_gpu_metrics()
                    gpu_str = format_gpu_metrics_str(gpu_info)
                    if self.backend == "vllm" and vllm_backend_obj is not None:
                        _quant_display = (
                            vllm_backend_obj.quantization
                            if vllm_backend_obj.quantization
                            else "None (FP16)"
                        )
                        _dtype_display = vllm_backend_obj.dtype
                        _chunk_display = vllm_backend_obj.effective_chunk_size
                    else:
                        _quant_display = (
                            "8-bit" if self.config.get("load_in_8bit")
                            else "4-bit" if self.config.get("load_in_4bit")
                            else "None (FP16/BF16)"
                        )
                        _dtype_display = "bfloat16" if self.config.get("use_bf16") else "float16"
                        _chunk_display = self.batch_size
                    self.log.info(
                        "\n"
                        "  ┌─ Checkpoint ──────────────────────────────────────────\n"
                        "  │  Backend:      %s\n"
                        "  │  Quantization: %s\n"
                        "  │  Dtype:        %s\n"
                        "  │  Chunk size:   %d\n"
                        "  │  %s\n"
                        "  │  Completed: %d / %d records | Elapsed: %.1fs\n"
                        "  └───────────────────────────────────────────────────────",
                        self.backend,
                        _quant_display,
                        _dtype_display,
                        _chunk_display,
                        gpu_str,
                        processed_records,
                        total_to_generate,
                        time.time() - start_time,
                    )

            except Exception as e:
                self.log.error(
                    "Batch recovery mechanism: Exception encountered processing batch %d/%d: %s. "
                    "Skipping current batch and continuing pipeline.",
                    batch_idx + 1,
                    total_batches,
                    e,
                    exc_info=True,
                )
                cleanup_gpu()
                pbar.update(batch_size_actual)
                continue

        pbar.close()
        self.writer.close()
        self.metadata_tracker.end_run()

        # Pass backend/vLLM metadata to save_metadata without modifying MetadataTracker
        extra_stats: dict = {"backend": self.backend}
        if self.backend == "vllm" and vllm_backend_obj is not None:
            extra_stats.update({
                "vllm_quantization": (
                    vllm_backend_obj.quantization
                    if vllm_backend_obj.quantization
                    else "None (FP16)"
                ),
                "vllm_dtype": vllm_backend_obj.dtype,
                "vllm_chunk_size_final": vllm_backend_obj.effective_chunk_size,
            })

        self.metadata_tracker.save_metadata(self.config, extra_stats=extra_stats)
        self.log.info("Generation run completed successfully.")
        cleanup_gpu()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.generation.generator",
        description="Fingerprint Stage 2 — LLM Text Generation Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/generation.yaml",
        help="Path to generation.yaml (default: configs/generation.yaml)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    config = load_yaml_config(args.config)
    pipeline = LLMGeneratorPipeline(config)
    pipeline.run()
