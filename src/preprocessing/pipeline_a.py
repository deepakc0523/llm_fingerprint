"""Pipeline A: Fingerprint-Preserving Preprocessor for LLM Fingerprinting project."""

import logging
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig
from src.preprocessing.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class PipelineAFingerprintPreserving:
    """Executes Pipeline A: Strictly preserves capitalization, punctuation, spacing, and formatting."""

    def __init__(self, config: PreprocessingConfig, normalizer: TextNormalizer):
        """Initialize Pipeline A with configuration and shared normalizer."""
        self.config = config
        self.normalizer = normalizer

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataset using Pipeline A rules.

        Args:
            df: Validated input DataFrame.

        Returns:
            Preprocessed DataFrame preserving fingerprint features.
        """
        logger.info("Executing Pipeline A (Fingerprint-Preserving NLP)...")

        gen_col = self.config.generated_text_col
        prefix_col = self.config.human_prefix_col
        unicode_form = self.config.pipeline_a_cfg.get("unicode_normalization", "NFC")

        processed_df = df.copy()

        # Apply fingerprint-preserving normalization to generated_text and human_prefix
        if gen_col in processed_df.columns:
            processed_df[gen_col] = processed_df[gen_col].astype(str).apply(
                lambda s: self.normalizer.normalize_fingerprint_preserving(s, form=unicode_form)
            )

        if prefix_col in processed_df.columns:
            processed_df[prefix_col] = processed_df[prefix_col].astype(str).apply(
                lambda s: self.normalizer.normalize_fingerprint_preserving(s, form=unicode_form)
            )

        # Update character and word lengths post-processing
        processed_df["char_length"] = processed_df[gen_col].str.len()
        processed_df["word_count"] = processed_df[gen_col].apply(lambda s: len(s.split()))

        logger.info("Pipeline A processing completed on %d records.", len(processed_df))
        return processed_df
