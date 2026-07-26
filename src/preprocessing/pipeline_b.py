"""Pipeline B: Traditional Classical NLP Preprocessor."""

import logging
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig
from src.preprocessing.normalizer import TextNormalizer

logger = logging.getLogger(__name__)


class PipelineBTraditionalNLP:
    """Executes Pipeline B: Traditional classical NLP cleaning (lowercasing, punctuation stripping, stopwords, lemmatization)."""

    def __init__(self, config: PreprocessingConfig, normalizer: TextNormalizer):
        """Initialize Pipeline B with configuration and shared normalizer."""
        self.config = config
        self.normalizer = normalizer

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process dataset using Pipeline B rules.

        Args:
            df: Validated input DataFrame.

        Returns:
            Preprocessed DataFrame following traditional NLP rules.
        """
        logger.info("Executing Pipeline B (Traditional NLP)...")

        gen_col = self.config.generated_text_col
        prefix_col = self.config.human_prefix_col
        unicode_form = self.config.pipeline_b_cfg.get("unicode_normalization", "NFC")

        processed_df = df.copy()

        # Apply traditional NLP normalization to generated_text and human_prefix
        if gen_col in processed_df.columns:
            processed_df[gen_col] = processed_df[gen_col].astype(str).apply(
                lambda s: self.normalizer.normalize_traditional(s, form=unicode_form)
            )

        if prefix_col in processed_df.columns:
            processed_df[prefix_col] = processed_df[prefix_col].astype(str).apply(
                lambda s: self.normalizer.normalize_traditional(s, form=unicode_form)
            )

        # Filter empty documents created by traditional cleaning
        if self.config.pipeline_b_cfg.get("remove_empty_docs", True):
            initial_count = len(processed_df)
            processed_df = processed_df[processed_df[gen_col].str.strip().str.len() > 0].reset_index(drop=True)
            logger.info("Filtered %d empty records resulting from traditional NLP cleaning.", initial_count - len(processed_df))

        # Update character and word lengths post-processing
        processed_df["char_length"] = processed_df[gen_col].str.len()
        processed_df["word_count"] = processed_df[gen_col].apply(lambda s: len(s.split()))

        logger.info("Pipeline B processing completed on %d records.", len(processed_df))
        return processed_df
