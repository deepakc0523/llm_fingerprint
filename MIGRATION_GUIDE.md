# Migration Guide — Transformers → vLLM Backend

This guide documents all changes required to migrate from the HuggingFace Transformers
inference backend (v1.0.0) to the vLLM dual-backend architecture (v2.0.0).

---

## 1. Install vLLM

vLLM is **not** a hard dependency of this project (the Transformers backend runs without it).
Install it separately in your environment:

```bash
pip install vllm
```

> **Important:** vLLM requires Python 3.8+ and a CUDA-capable GPU.
> It does **not** run on CPU or Apple Silicon (MPS).  For those environments, use
> `backend: transformers` in `generation.yaml`.

### Colab / cloud GPU setup

```bash
# Colab T4 / L4 / A100
pip install vllm --quiet
```

---

## 2. Update `configs/generation.yaml`

### Add the new vLLM keys

```yaml
# ── Backend Selection ─────────────────────────────────────────────────────────
backend: "vllm"                  # Switch to "transformers" for fallback at any time

# ── vLLM Engine Settings ──────────────────────────────────────────────────────
gpu_memory_utilization: 0.90    # Fraction of VRAM for model + KV cache
tensor_parallel_size: 1         # Number of GPUs (1 = single GPU)
quantization: "auto"            # "auto" | "awq" | "gptq" | null (FP16)
dtype: "auto"                   # "auto" | "float16" | "bfloat16"
use_prefix_caching: true        # Graceful fallback if unsupported
trust_remote_code: true
```

### Update `batch_size` and `max_samples`

```yaml
# vLLM backend: submission chunk size per llm.generate() call (not GPU batch size)
# Adaptive OOM recovery halves this automatically: 256 → 128 → 64 → 32 → 16
batch_size: 256

# Benchmark run (change to 10000, 50000, or null — NO code changes needed)
max_samples: 1000
```

---

## 3. Switching Between Backends

Change **one line** in `generation.yaml`:

```yaml
backend: "vllm"          # production — high throughput
backend: "transformers"  # fallback — any GPU, CPU, or MPS
```

When switching to `backend: transformers`, also set a smaller `batch_size`:
```yaml
batch_size: 8    # recommended for 7–9B models on a single GPU with Transformers
```

No source code changes are needed for either switch.

---

## 4. Quantization Selection Guide

| Config | Effect | When to use |
|--------|--------|-------------|
| `quantization: "auto"` | Detects AWQ/GPTQ from model path, else FP16 | **Recommended default** |
| `quantization: "awq"` | Forces AWQ | You have an AWQ checkpoint |
| `quantization: "gptq"` | Forces GPTQ | You have a GPTQ checkpoint |
| `quantization: null` | No quantization (FP16/BF16) | Maximum accuracy |

### Auto-detection rules

`"auto"` inspects the `model_paths` value:
- Model path contains `"awq"` → use AWQ
- Model path contains `"gptq"` → use GPTQ
- Neither → use FP16 / `dtype: auto`

**Example:** `Qwen/Qwen2.5-7B-Instruct-AWQ` → auto-detected as AWQ.

---

## 5. Dtype Selection Guide

| Config | Effect |
|--------|--------|
| `dtype: "auto"` | BF16 if GPU supports it (A100, L4, H100), else FP16 |
| `dtype: "bfloat16"` | Force BF16 |
| `dtype: "float16"` | Force FP16 (T4, V100) |

---

## 6. Prefix Caching

`use_prefix_caching: true` enables vLLM's automatic KV-cache reuse for shared prompt prefixes.
If the model or vLLM version does not support it, the engine **automatically retries without it**
and logs a warning — no action needed.

---

## 7. Scaling to 10k / 50k Prompts

Only **one value** changes in `generation.yaml`:

```yaml
max_samples: 1000    # benchmark
max_samples: 10000   # medium run
max_samples: 50000   # production run
max_samples: null    # full dataset
```

No code modifications are required.  The checkpoint/resume behaviour is identical at any scale.

---

## 8. GPU Memory Tuning

If you encounter OOM errors:

1. **Automatic recovery**: The pipeline automatically halves the chunk size on OOM
   (`256 → 128 → 64 → 32 → 16`). Check logs for the warning message.

2. **Reduce VRAM allocation**:
   ```yaml
   gpu_memory_utilization: 0.80    # down from 0.90
   ```

3. **Use quantization**:
   ```yaml
   quantization: "awq"    # 4-bit AWQ, ~50% VRAM reduction
   ```

4. **Switch backend** as a last resort:
   ```yaml
   backend: "transformers"
   batch_size: 4
   load_in_4bit: true
   ```

---

## 9. Resuming After Disconnection (Colab)

Checkpoint behaviour is **identical** for both backends.  If Colab disconnects:

1. Restart the runtime.
2. Re-run the pipeline with the **same config**.
3. The `CheckpointManager` reads `checkpoint.json` and skips all completed prefix IDs.

No configuration change is needed.

---

## 10. Chat Mode with vLLM

When `prompt_mode: chat`:
- The vLLM backend loads only the **HuggingFace tokenizer** (no model weights) to apply
  `apply_chat_template()`.
- This is lightweight (seconds, negligible VRAM) and fully compatible.
- The same `PromptFormatter` interface is used for both backends.

---

## 11. Verification Run

Run a 100-prompt smoke test with Qwen2.5 7B on vLLM:

```yaml
# generation.yaml
model: "qwen2_5"
backend: "vllm"
max_samples: 100
gpu_memory_utilization: 0.90
quantization: "auto"
dtype: "auto"
```

```bash
python -m src.generation.generator --config configs/generation.yaml
```

Expected output per checkpoint:
```
  ┌─ Checkpoint ───────────────────────────────────────────
  │  Backend:      vllm
  │  Quantization: None (FP16)
  │  Dtype:        bfloat16
  │  Chunk size:   256
  │  VRAM: 14.23/23.69 GiB (60.1%) | GPU util: 92%
  │  Completed: 100 / 100 records | Elapsed: 38.2s
  └───────────────────────────────────────────────────────
```

---

## 12. Architecture Overview

```
generation.yaml
  └── backend: "vllm" / "transformers"
        │
        ▼
LLMGeneratorPipeline.run()
  │
  ├── [vllm]         load_vllm_model()  →  VLLMInferenceBackend.generate_batch()
  │                                        (adaptive OOM: 256→128→64→32→16)
  │
  └── [transformers] ModelLoaderRegistry.load_model_and_tokenizer()
                     model.generate() + tokenizer.decode()
        │
        ▼ (both paths produce identical output)
  PromptFormatter  →  GeneratedRecord  →  ParquetBatchWriter
  CheckpointManager  →  MetadataTracker
```

---

## 13. Files Changed vs Unchanged

### Changed
| File | Type | What changed |
|------|------|-------------|
| `src/generation/vllm_loader.py` | **NEW** | vLLM engine factory |
| `src/generation/vllm_generator.py` | **NEW** | VLLMInferenceBackend |
| `configs/generation.yaml` | Modified | Added backend + vLLM keys |
| `requirements.txt` | Modified | Added commented vllm entry |
| `src/generation/generator.py` | Modified | Backend branch in run() |
| `src/generation/prompt_formatter.py` | Modified | tokenizer optional |
| `src/generation/__init__.py` | Modified | Exported new symbols |

### Unchanged (preserved by design)
| File | Reason |
|------|--------|
| `src/generation/checkpoint.py` | Identical behaviour for both backends |
| `src/generation/parquet_writer.py` | Output schema untouched |
| `src/generation/metadata.py` | API unchanged; extra_stats passed externally |
| `src/generation/generation_utils.py` | Timer, set_seed reused by both |
| `src/generation/model_loader.py` | Transformers loader for fallback |

---

## 14. Troubleshooting

### `ImportError: vLLM is not installed`
```bash
pip install vllm
```
Or switch to `backend: transformers` in `generation.yaml`.

### `CUDA out of memory`
The pipeline auto-recovers by halving chunk size.  If it fails at `chunk_size=16`,
reduce `gpu_memory_utilization` or switch to a quantized model.

### `Prefix caching not supported`
Logged as a warning; the engine retries without it automatically.  No action needed.

### `prompt_mode='chat' requires a tokenizer`
This warning appears when `prompt_mode: chat` and `backend: vllm`.  The pipeline
automatically loads the HF tokenizer for chat template application.  If you see it
fall back to `instruction`, ensure the model path is reachable for tokenizer download.

### Parquet schema mismatch
The `GeneratedRecord` schema is identical for both backends.  If you see a mismatch,
check that the parquet file was not written by a pre-v2.0.0 run with a different schema.
Delete the existing output file and re-run.
