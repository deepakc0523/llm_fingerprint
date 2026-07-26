"""Dataset Statistics computation module for LLM Fingerprinting Research Project."""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from src.dataset.schema import MergeConfig, ValidationResult

logger = logging.getLogger(__name__)


class DatasetStatisticsGenerator:
    """Computes comprehensive statistics for merged LLM synthetic dataset."""

    def __init__(self, config: MergeConfig):
        """Initialize statistics generator.

        Args:
            config: Clean MergeConfig instance.
        """
        self.config = config

    def generate_statistics(self, df: pd.DataFrame, val_result: ValidationResult) -> Dict[str, Any]:
        """Compute statistics and save dataset_statistics.json.

        Args:
            df: Merged pandas DataFrame.
            val_result: Validation results object.

        Returns:
            Dictionary containing all calculated dataset metrics.
        """
        logger.info("Computing dataset statistics for %d total records...", len(df))

        prefix_col = self.config.prefix_column
        gen_col = self.config.generated_column
        id_col = self.config.id_column
        stratify_col = self.config.stratify_column

        # 1. Samples per model & label distribution
        label_distribution = df[stratify_col].value_counts().to_dict()
        samples_per_model = {str(k): int(v) for k, v in label_distribution.items()}

        # 2. Length calculations
        prefix_lengths = df[prefix_col].astype(str).str.len() if prefix_col in df.columns else pd.Series([0] * len(df))
        gen_lengths = df[gen_col].astype(str).str.len() if gen_col in df.columns else pd.Series([0] * len(df))
        response_lengths = prefix_lengths + gen_lengths

        def calc_stats(series: pd.Series) -> Dict[str, float]:
            if series.empty:
                return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
            return {
                "mean": round(float(series.mean()), 2),
                "median": round(float(series.median()), 2),
                "min": int(series.min()),
                "max": int(series.max()),
                "std": round(float(series.std()), 2),
            }

        # 3. Missing values & duplicate counts
        missing_values = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
        duplicate_ids = int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
        duplicate_texts = int(df[gen_col].duplicated().sum()) if gen_col in df.columns else 0

        # 4. Token statistics (if token length column exists in dataset or computed via whitespace fallback)
        token_stats: Dict[str, Any] = {}
        if "completion_length" in df.columns:
            token_stats["completion_tokens"] = calc_stats(df["completion_length"])
        if "prompt_length" in df.columns:
            token_stats["prompt_tokens"] = calc_stats(df["prompt_length"])

        # Fallback whitespace word tokens if token count columns missing
        word_counts = df[gen_col].astype(str).str.split().str.len() if gen_col in df.columns else pd.Series([0] * len(df))
        token_stats["approx_word_tokens"] = calc_stats(word_counts)

        stats: Dict[str, Any] = {
            "total_samples": int(len(df)),
            "number_of_models": int(len(samples_per_model)),
            "samples_per_model": samples_per_model,
            "label_distribution": samples_per_model,
            "response_length_char_statistics": {
                "combined_response": calc_stats(response_lengths),
                "prefix_length": calc_stats(prefix_lengths),
                "completion_length": calc_stats(gen_lengths),
            },
            "token_statistics": token_stats,
            "duplicates": {
                "duplicate_prefix_ids": duplicate_ids,
                "duplicate_generated_texts": duplicate_texts,
            },
            "missing_values": missing_values,
            "excluded_datasets": [
                {
                    "dataset_name": d.dataset_name,
                    "reason": d.error_message,
                }
                for d in val_result.invalid_datasets
            ],
        }

        # Save dataset_statistics.json
        output_path = self.config.output_dir / "dataset_statistics.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)

        logger.info("Saved dataset statistics to %s", output_path)
        return stats
