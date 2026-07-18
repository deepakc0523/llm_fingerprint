"""
src.generation.model_loader
===========================
Implements model and tokenizer loading utilities, featuring a registry-based
architecture that supports Llama 3.2, Qwen 2.5, Gemma 2, Mistral 7B, Phi-3.5,
and extensible loading parameters (precision, device mapping, and quantization).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Tuple, Type
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

_log = logging.getLogger(__name__)


def cleanup_gpu() -> None:
    """Safely triggers garbage collection and empties the PyTorch execution cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _log.info("Cleaned up CUDA GPU memory cache.")
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()
        _log.info("Cleaned up MPS GPU memory cache.")


def get_max_context_length(model: PreTrainedModel, fallback: int = 2048) -> int:
    """
    Exposes the maximum sequence/context length defined in the model configuration.

    Parameters
    ----------
    model:
        The loaded PreTrainedModel instance.
    fallback:
        Default length to return if config attributes are missing.

    Returns
    -------
    int
        The detected maximum sequence length.
    """
    config = getattr(model, "config", None)
    if config is None:
        return fallback

    # Check common keys for context length in HF models
    for key in ("max_position_embeddings", "model_max_length", "max_sequence_length", "n_positions"):
        if hasattr(config, key) and getattr(config, key) is not None:
            val = getattr(config, key)
            if isinstance(val, int) and val > 0:
                _log.info("Detected maximum context length from config key '%s': %d", key, val)
                return val

    _log.warning("Could not detect context length. Using fallback: %d", fallback)
    return fallback


def detect_device() -> str:
    """
    Detects the optimal available device automatically.
    Prefers CUDA, then MPS (Apple Silicon), falling back to CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.mps.is_available():
        return "mps"
    return "cpu"


class BaseModelLoader:
    """Base class for custom model loaders, allowing specialized model loading rules."""

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "auto",
        use_bf16: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Loads the model and its tokenizer with settings optimized for batch generation.

        Parameters
        ----------
        model_path:
            HuggingFace repository ID or absolute local path.
        device:
            Target execution device (e.g., 'cuda', 'cpu', 'mps', 'auto').
        use_bf16:
            Whether to use bfloat16 precision if supported.
        load_in_8bit:
            Use 8-bit quantization (requires bitsandbytes).
        load_in_4bit:
            Use 4-bit quantization (requires bitsandbytes).
        extra_kwargs:
            Optional additional dictionary of arguments to pass to the model loader.

        Returns
        -------
        Tuple[PreTrainedModel, PreTrainedTokenizer]
            The loaded causal language model and its paired tokenizer.
        """
        extra_kwargs = extra_kwargs or {}

        # Validate quantization options
        if load_in_8bit and load_in_4bit:
            raise ValueError(
                "Quantization conflict: Cannot enable load_in_8bit and load_in_4bit simultaneously. "
                "Select only one quantization mode."
            )

        # Resolve target device
        resolved_device = device
        if device == "auto":
            resolved_device = detect_device()
            _log.info("Automatically detected execution device: %s", resolved_device)

        # Configure compute dtype and options
        torch_dtype = torch.float32
        if use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif use_bf16 or (resolved_device == "cuda" and not use_bf16):
            torch_dtype = torch.float16

        loader_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": "auto" if resolved_device == "cuda" else None,
        }

        # Apply quantization configs
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            loader_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            # Remove device_map since bitsandbytes requires auto/specific device mapping
            loader_kwargs["device_map"] = "auto"
        elif load_in_8bit:
            loader_kwargs["load_in_8bit"] = True
            loader_kwargs["device_map"] = "auto"

        # Apply any overrides
        loader_kwargs.update(extra_kwargs)

        _log.info("Loading tokenizer from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        # Ensure left-padding for batch completion generation
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        _log.info("Loading model from %s with options: %s", model_path, loader_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_path, **loader_kwargs)
        
        if resolved_device != "cuda" and loader_kwargs.get("device_map") is None:
            model = model.to(resolved_device)

        model.eval()
        return model, tokenizer


class LlamaLoader(BaseModelLoader):
    """Specialized loader for Llama-based models."""
    pass


class QwenLoader(BaseModelLoader):
    """Specialized loader for Qwen-based models."""
    pass


class GemmaLoader(BaseModelLoader):
    """Specialized loader for Gemma-based models."""
    
    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "auto",
        use_bf16: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        return super().load(
            model_path=model_path,
            device=device,
            use_bf16=use_bf16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            extra_kwargs=extra_kwargs,
        )


class MistralLoader(BaseModelLoader):
    """Specialized loader for Mistral-based models."""
    pass


class PhiLoader(BaseModelLoader):
    """Specialized loader for Phi-based models."""
    pass


class ModelLoaderRegistry:
    """Registry pattern to manage and instantiate model loaders for supported models."""

    _registry: Dict[str, Type[BaseModelLoader]] = {
        "llama3": LlamaLoader,
        "qwen2_5": QwenLoader,
        "gemma2": GemmaLoader,
        "mistral": MistralLoader,
        "phi3": PhiLoader,
    }

    @classmethod
    def register(cls, name: str, loader_cls: Type[BaseModelLoader]) -> None:
        """Register a custom loader class under a name alias."""
        cls._registry[name] = loader_cls
        _log.info("Registered custom loader '%s': %s", name, loader_cls.__name__)

    @classmethod
    def load_model_and_tokenizer(
        cls,
        model_name: str,
        model_path: str,
        device: str = "auto",
        use_bf16: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        extra_kwargs: Dict[str, Any] | None = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Dispatches loading to the registered loader for the specified model name alias.

        Parameters
        ----------
        model_name:
            Registered model alias (e.g., 'llama3', 'qwen2_5', 'gemma2', 'mistral', 'phi3').
        model_path:
            HuggingFace model ID or absolute path.
        device:
            Target execution device or 'auto'.
        use_bf16:
            Whether to use bfloat16.
        load_in_8bit:
            Bitsandbytes 8-bit flag.
        load_in_4bit:
            Bitsandbytes 4-bit flag.
        extra_kwargs:
            Optional overrides.

        Returns
        -------
        Tuple[PreTrainedModel, PreTrainedTokenizer]
        """
        loader_cls = cls._registry.get(model_name.lower())
        if loader_cls is None:
            _log.warning(
                "Model alias '%s' not registered. Falling back to BaseModelLoader.", model_name
            )
            loader_cls = BaseModelLoader

        return loader_cls.load(
            model_path=model_path,
            device=device,
            use_bf16=use_bf16,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            extra_kwargs=extra_kwargs,
        )
