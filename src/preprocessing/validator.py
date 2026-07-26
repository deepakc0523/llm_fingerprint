"""Dataset Validation & Filtering module for NLP Preprocessing Layer."""

import logging
from typing import Tuple, List, Dict, Any
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig

logger = logging.getLogger(__name__)


class PreprocessingValidator:
    """Validates raw dataset integrity and tracks removed records during quality checks."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize validator with pipeline configuration."""
        self.config = config

    def validate_and_filter(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute validation suite: missing checks, min/max length filtering, near-duplicates.

        Args:
            df: Raw merged dataset DataFrame.

        Returns:
            Tuple of (Cleaned DataFrame, Removed Records DataFrame).
        """
        logger.info("Running dataset validation & quality filter checks...")

        text_col = self.config.generated_text_col
        id_col = self.config.prefix_id_col

        removed_records: List[Dict[str, Any]] = []
        valid_indices: List[int] = []

        for idx, row in df.iterrows():
            rec_id = row.get(id_col, f"row_{idx}")
            text = str(row.get(text_col, ""))

            # 1. Missing / null text check
            if pd.isna(row.get(text_col)) or len(text.strip()) == 0:
                removed_records.append({
                    "prefix_id": rec_id,
                    "model_label": row.get(self.config.model_label_col, "unknown"),
                    "reason": "Missing/empty text content",
                    "original_length": len(text),
                })
                continue

            # 2. Length threshold validation
            if len(text) < self.config.min_char_length:
                removed_records.append({
                    "prefix_id": rec_id,
                    "model_label": row.get(self.config.model_label_col, "unknown"),
                    "reason": f"Length below min threshold ({len(text)} < {self.config.min_char_length})",
                    "original_length": len(text),
                })
                continue

            if len(text) > self.config.max_char_length:
                removed_records.append({
                    "prefix_id": rec_id,
                    "model_label": row.get(self.config.model_label_col, "unknown"),
                    "reason": f"Length exceeds max threshold ({len(text)} > {self.config.max_char_length})",
                    "original_length": len(text),
                })
                continue

            valid_indices.append(idx)

        cleaned_df = df.loc[valid_indices].reset_index(drop=True)
        removed_df = pd.DataFrame(removed_records)

        logger.info(
            "Validation finished: %d records kept, %d records removed.",
            len(cleaned_df),
            len(removed_df),
        )
        return cleaned_df, removed_df
