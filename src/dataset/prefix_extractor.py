"""
src.dataset.prefix_extractor
=============================
Tokenisation-based prefix extraction from raw documents.

For each document the extractor:
1. Tokenises the full text once using a HuggingFace fast tokenizer.
2. Applies the configured extraction strategy (e.g. sliding window).
3. Attaches metadata (prefix_id, dataset_name, category, document_id, language,
   prefix_length, tokenizer, extraction_timestamp, prefix_text).
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional

from langdetect import detect_langs
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Make langdetect deterministic.
DetectorFactory.seed = 0


class PrefixExtractor:
    """
    Extract variable-length token-bounded text prefixes from a document.

    Parameters
    ----------
    config:
        Configuration dictionary containing extraction and tokenizer settings.
    logger:
        Optional logger.  Defaults to ``"fingerprint.prefix_extractor"``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config: Dict[str, Any] = config
        tok_cfg = config.get("tokenizer", {})
        self.tokenizer_name: str = tok_cfg.get("model_name", "gpt2")
        self.use_fast: bool = tok_cfg.get("use_fast", True)
        self.add_special_tokens: bool = tok_cfg.get("add_special_tokens", False)
        self.truncation: bool = tok_cfg.get("truncation", False)

        self.prefix_lengths: List[int] = sorted(
            list(config.get("prefix_lengths", [64, 128, 256]))
        )

        ext_cfg = config.get("extraction", {})
        self.strategy: str = ext_cfg.get("strategy", "sliding_window")
        self.overlap_tokens: int = ext_cfg.get("overlap_tokens", 32)
        self.min_doc_tokens: int = ext_cfg.get("minimum_document_tokens", 300)

        self.log: logging.Logger = logger or logging.getLogger(
            "fingerprint.prefix_extractor"
        )
        self._tokenizer: Optional[PreTrainedTokenizerFast] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        document_id: str,
        dataset_name: str,
        split: str,
        category: str,
    ) -> List[Dict[str, Any]]:
        """
        Produce prefix records for a single document.

        Checks if the document has enough tokens. If so, segments it according
        to the configured strategy.

        Parameters
        ----------
        text:
            Raw document text.
        document_id:
            Unique identifier for the source document.
        dataset_name:
            Source dataset key (e.g. ``"wikitext"``).
        split:
            HuggingFace split name (e.g. ``"train"``).
        category:
            Semantic category label.

        Returns
        -------
        List[Dict[str, Any]]
            A list of prefix records.
        """
        tokenizer = self._get_tokenizer()
        token_ids: List[int] = tokenizer.encode(
            text,
            add_special_tokens=self.add_special_tokens,
            truncation=self.truncation,
        )

        if len(token_ids) < self.min_doc_tokens:
            # Document too short. Skip it.
            return []

        records: List[Dict[str, Any]] = []

        for length in self.prefix_lengths:
            if len(token_ids) < length:
                continue

            if self.strategy == "sliding_window":
                stride = max(1, length - self.overlap_tokens)
                start_indices = range(0, len(token_ids) - length + 1, stride)
            else:
                # Fallback to single prefix from the beginning
                start_indices = [0]

            for start_idx in start_indices:
                prefix_ids = token_ids[start_idx : start_idx + length]
                prefix_text: str = tokenizer.decode(
                    prefix_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                lang_code, confidence = self._detect_language_and_confidence(prefix_text)

                records.append(
                    {
                        "prefix_id": str(uuid.uuid4()),
                        "dataset_name": dataset_name,
                        "category": category,
                        "document_id": document_id,
                        "language": lang_code,
                        "language_confidence": confidence,
                        "prefix_length": length,
                        "tokenizer": self.tokenizer_name,
                        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        "prefix_text": prefix_text,
                        "split": split,
                    }
                )

        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_tokenizer(self) -> PreTrainedTokenizerFast:
        """
        Lazy-load and return the HuggingFace tokenizer.

        Returns
        -------
        PreTrainedTokenizerFast
            A loaded tokenizer instance.
        """
        if self._tokenizer is None:
            self.log.info("Loading tokenizer: %s", self.tokenizer_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=self.use_fast,
            )
        return self._tokenizer  # type: ignore[return-value]

    def _detect_language_and_confidence(self, text: str) -> tuple[str, float]:
        """
        Attempt to identify the language of *text* and retrieve the confidence.

        Parameters
        ----------
        text:
            Text whose language should be detected.

        Returns
        -------
        tuple[str, float]
            ISO 639-1 language code and confidence probability.
        """
        try:
            langs = detect_langs(text)
            if langs:
                top_lang = langs[0]
                return top_lang.lang, top_lang.prob
            return "unknown", 0.0
        except LangDetectException:
            return "unknown", 0.0
        except Exception:
            return "unknown", 0.0
