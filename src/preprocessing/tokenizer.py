"""Tokenization & Corpus Statistics module for Preprocessing Layer."""

import logging
import re
from typing import Dict, Any, List
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig

logger = logging.getLogger(__name__)


class CorpusTokenizer:
    """Computes tokenization metrics, sentence statistics, and corpus vocabulary for processed DataFrames."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize tokenizer with pipeline configuration."""
        self.config = config
        self.sentence_pattern = re.compile(r"[.!?]+")

    def compute_statistics(self, df: pd.DataFrame, pipeline_name: str) -> Dict[str, Any]:
        """Compute corpus tokenization metrics.

        Args:
            df: Processed pandas DataFrame.
            pipeline_name: Name of pipeline ('fingerprint' or 'traditional').

        Returns:
            Dictionary containing token, sentence, character, and vocabulary metrics.
        """
        logger.info("Computing token & sentence statistics for Pipeline '%s'...", pipeline_name)

        gen_col = self.config.generated_text_col
        texts = df[gen_col].astype(str).tolist() if gen_col in df.columns else []

        total_tokens = 0
        total_sentences = 0
        total_chars = 0
        unique_tokens: Set[str] = set()

        for t in texts:
            tokens = t.split()
            total_tokens += len(tokens)
            total_chars += len(t)
            unique_tokens.update(tokens)

            s_count = max(1, len(self.sentence_pattern.findall(t))) if len(t.strip()) > 0 else 0
            total_sentences += s_count

        sample_count = len(df)
        avg_doc_len = round(total_chars / sample_count, 2) if sample_count > 0 else 0.0
        avg_token_len = round(total_chars / total_tokens, 2) if total_tokens > 0 else 0.0
        avg_sentence_len = round(total_tokens / total_sentences, 2) if total_sentences > 0 else 0.0
        vocab_size = len(unique_tokens)
        ttr = round(vocab_size / total_tokens, 6) if total_tokens > 0 else 0.0

        stats: Dict[str, Any] = {
            "pipeline_name": pipeline_name,
            "total_samples": sample_count,
            "total_token_count": total_tokens,
            "total_sentence_count": total_sentences,
            "total_character_count": total_chars,
            "vocabulary_size": vocab_size,
            "type_token_ratio": ttr,
            "lexical_diversity": ttr,
            "average_document_length_chars": avg_doc_len,
            "average_token_length_chars": avg_token_len,
            "average_sentence_length_tokens": avg_sentence_len,
        }

        logger.info(
            "Pipeline '%s' metrics: %d samples, %d tokens, %d unique vocab (TTR: %.4f).",
            pipeline_name,
            sample_count,
            total_tokens,
            vocab_size,
            ttr,
        )
        return stats
