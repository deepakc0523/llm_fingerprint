# CHANGELOG

All notable changes to the Fingerprint LLM Synthetic Text Generation pipeline are documented here.

---

## [2.0.0] — 2026-07-20

### Overview
Migration of the Stage 2 synthetic text generation backend from HuggingFace Transformers to vLLM,
with a dual-backend architecture that allows transparent switching via a single config line.

---

### Added

#### `src/generation/vllm_loader.py` *(new)*
- `resolve_quantization(model_path, config)` — Automatic quantization detection.
  Priority: AWQ (preferred) → GPTQ → FP16 (None). Detects from model path substrings
  when `quantization: auto`.
- `resolve_dtype(config)` — Automatic dtype resolution. Returns `bfloat16` if GPU
  supports BF16, otherwise `float16`. Falls through to explicit value if set.
- `load_vllm_model(model_path, config)` — vLLM `LLM` engine factory.
  Attempts prefix caching first; falls back gracefully with a warning if unsupported.

#### `src/generation/vllm_generator.py` *(new)*
- `VLLMInferenceBackend` class — wraps `vllm.LLM` + `SamplingParams` into the same
  `GeneratedRecord` interface as the Transformers backend.
  - Adaptive OOM recovery: `256 → 128 → 64 → 32 → 16` chunk size ladder.
  - Effective chunk size persists across batches after first OOM reduction.
  - `generate_batch()` returns `(List[GeneratedRecord], int)` identical to Transformers path.
- `get_gpu_metrics()` — Collects VRAM used/total (GiB), VRAM %, GPU utilisation %.
- `format_gpu_metrics_str()` — Compact single-line telemetry string for logging.

#### `configs/generation.yaml` — new keys
| Key | Default | Description |
|-----|---------|-------------|
| `backend` | `"vllm"` | Inference backend: `"vllm"` or `"transformers"` |
| `gpu_memory_utilization` | `0.90` | Fraction of VRAM reserved for model + KV cache |
| `tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism |
| `quantization` | `"auto"` | AWQ / GPTQ / auto-detect / null (FP16) |
| `dtype` | `"auto"` | `bfloat16` / `float16` / auto |
| `use_prefix_caching` | `true` | vLLM prefix caching (graceful fallback) |
| `trust_remote_code` | `true` | Trust remote code for custom architectures |

#### `configs/generation.yaml` — updated keys
| Key | Old Value | New Value | Reason |
|-----|-----------|-----------|--------|
| `batch_size` | `8` | `256` | Repurposed as vLLM submission chunk size |
| `max_samples` | `null` | `1000` | Set benchmark default; change to scale |

---

### Modified

#### `src/generation/generator.py`
- Added `self.backend` attribute in `__init__` from `config.get("backend", "transformers")`.
- Model loading in `run()` branches on `self.backend`:
  - `vllm` → `load_vllm_model()` + optional tokenizer for chat mode only.
  - `transformers` → existing `ModelLoaderRegistry.load_model_and_tokenizer()` (unchanged).
- Inference loop branches on `self.backend`:
  - `vllm` → `vllm_backend_obj.generate_batch()`.
  - `transformers` → existing `model.generate()` + decode loop (unchanged).
- `session_completed_ids` tracking unified into a single post-inference block.
- Checkpoint saves now emit a structured telemetry block: backend, quantization, dtype,
  chunk size, VRAM used/total, GPU utilisation, completed/total records, elapsed time.
- `save_metadata()` receives `extra_stats` with backend / vLLM runtime details without
  modifying `MetadataTracker`.

#### `src/generation/prompt_formatter.py`
- `tokenizer` parameter made optional (`default=None`).
- Added guard: if `mode="chat"` and `tokenizer=None`, falls back to `"instruction"` format
  with a logged warning (avoids silent crash on vLLM non-chat runs).

#### `src/generation/__init__.py`
- Exported `load_vllm_model`, `resolve_quantization`, `resolve_dtype`,
  `VLLMInferenceBackend`, `get_gpu_metrics`, `format_gpu_metrics_str`.

#### `requirements.txt`
- Added commented entry for `vllm>=0.4.0` with install instructions.
  Not a hard dependency; the Transformers backend remains self-contained.

---

### Unchanged (preserved by design)

| Component | File |
|-----------|------|
| Checkpoint manager | `src/generation/checkpoint.py` |
| Parquet writer + schema | `src/generation/parquet_writer.py` |
| Metadata tracker | `src/generation/metadata.py` |
| Generation utilities | `src/generation/generation_utils.py` |
| Transformers model loader | `src/generation/model_loader.py` |
| Output schema | `GeneratedRecord` dataclass — all fields identical |
| Downstream ML compatibility | Parquet files: same column names, types, compression |

---

### Scalability

No code changes are required to scale the pipeline:

| Target | Change |
|--------|--------|
| 1,000 prompts | `max_samples: 1000` |
| 10,000 prompts | `max_samples: 10000` |
| 50,000 prompts | `max_samples: 50000` |
| Full dataset | `max_samples: null` |

---

### Performance (expected)

vLLM typically delivers **3–10× higher throughput** compared to HuggingFace Transformers
on the same hardware due to PagedAttention, continuous batching, and CUDA graph optimisation.

---

## [1.0.0] — Initial Release

- Stage 2 generation pipeline using HuggingFace Transformers.
- `ModelLoaderRegistry` with Llama, Qwen, Gemma, Mistral, Phi loaders.
- `PromptFormatter` with raw / chat / instruction / completion modes.
- `CheckpointManager` with JSON state persistence.
- `ParquetBatchWriter` with strict schema enforcement.
- `MetadataTracker` for run telemetry and environment capture.
