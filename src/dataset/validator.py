"""Dataset Validation module for LLM Fingerprinting Research Project."""

import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd

from src.dataset.schema import MergeConfig, DatasetSummary, ValidationResult

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates synthetic datasets dynamically in target input directories."""

    def __init__(self, config: MergeConfig):
        """Initialize validator with pipeline configuration.

        Args:
            config: Clean MergeConfig instance.
        """
        self.config = config

    def validate_all(self) -> ValidationResult:
        """Scan input directory dynamically and validate all model subdirectories.

        Returns:
            ValidationResult containing valid datasets, invalid datasets, warnings, and errors.
        """
        res = ValidationResult()
        input_dir = self.config.input_dir

        if not input_dir.exists():
            msg = f"Input directory '{input_dir}' does not exist."
            logger.error(msg)
            res.errors.append(msg)
            return res

        logger.info("Scanning directory '%s' for model synthetic datasets...", input_dir)
        subdirs = [d for d in input_dir.iterdir() if d.is_dir()]

        if not subdirs:
            msg = f"No dataset subdirectories found inside '{input_dir}'."
            logger.warning(msg)
            res.warnings.append(msg)
            return res

        for folder in sorted(subdirs):
            model_name = folder.name
            logger.info("Validating dataset folder: '%s'...", model_name)
            summary = self._validate_single_folder(folder)

            if summary.is_valid:
                res.valid_datasets.append(summary)
                logger.info("Dataset '%s' passed validation (%d records).", model_name, summary.total_records)
            else:
                res.invalid_datasets.append(summary)
                logger.warning(
                    "Dataset '%s' EXCLUDED from merge: %s",
                    model_name,
                    summary.error_message or "Validation failed.",
                )

        logger.info(
            "Validation finished: %d valid datasets, %d invalid datasets.",
            len(res.valid_datasets),
            len(res.invalid_datasets),
        )
        return res

    def _validate_single_folder(self, folder_path: Path) -> DatasetSummary:
        """Validate a single model folder against required files, schema, and content constraints."""
        summary = DatasetSummary(
            dataset_name=folder_path.name,
            folder_path=folder_path,
        )

        # 1. Check required files
        missing_files = []
        for req_file in self.config.required_files:
            target_path = folder_path / req_file
            if not target_path.exists():
                missing_files.append(req_file)

        if missing_files:
            summary.missing_files = missing_files
            summary.error_message = f"Missing required file(s): {', '.join(missing_files)}"
            return summary

        # 2. Check parquet file readability & corruption
        parquet_path = folder_path / "generated.parquet"
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            summary.error_message = f"Corrupted parquet file: {e}"
            return summary

        # 3. Check for empty dataset
        if df.empty:
            summary.error_message = "Dataset is empty (0 records)."
            return summary

        summary.total_records = len(df)
        summary.columns = list(df.columns)

        # 4. Schema & column validation
        missing_cols = [c for c in self.config.expected_columns.keys() if c not in df.columns]
        if missing_cols:
            summary.error_message = f"Missing expected column(s): {', '.join(missing_cols)}"
            return summary

        # 5. Check null values
        null_counts = df.isnull().sum().to_dict()
        summary.null_counts = {k: int(v) for k, v in null_counts.items() if v > 0}

        # 6. Check duplicate IDs & generated text
        id_col = self.config.id_column
        gen_col = self.config.generated_column

        if id_col in df.columns:
            summary.duplicate_ids = int(df[id_col].duplicated().sum())
        if gen_col in df.columns:
            summary.duplicate_texts = int(df[gen_col].duplicated().sum())

        summary.is_valid = True
        return summary
