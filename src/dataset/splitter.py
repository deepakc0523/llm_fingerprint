"""Dataset Stratified Splitter module for LLM Fingerprinting Research Project."""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset.schema import MergeConfig

logger = logging.getLogger(__name__)


class DatasetSplitter:
    """Performs stratified dataset splitting into train, validation, and test sets."""

    def __init__(self, config: MergeConfig):
        """Initialize splitter with pipeline configuration.

        Args:
            config: Clean MergeConfig instance.
        """
        self.config = config

    def split_dataset(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Perform stratified split (70% train, 15% val, 15% test).

        Verifies every model class appears in every split.

        Args:
            df: Merged pandas DataFrame.

        Returns:
            Dictionary containing 'train', 'validation', and 'test' DataFrames.
        """
        logger.info("Performing stratified dataset split (70%% train, 15%% val, 15%% test)...")

        stratify_col = self.config.stratify_column
        if stratify_col not in df.columns:
            raise ValueError(f"Stratification column '{stratify_col}' not found in dataframe.")

        train_ratio = self.config.train_ratio
        val_ratio = self.config.val_ratio
        test_ratio = self.config.test_ratio
        seed = self.config.random_seed

        if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
            raise ValueError(f"Split ratios must sum to 1.0. Got {train_ratio}+{val_ratio}+{test_ratio}")

        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            stratify=df[stratify_col],
            random_state=seed,
        )

        # Second split: Val vs Test (Equal proportion of temp)
        relative_test_ratio = test_ratio / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_ratio,
            stratify=temp_df[stratify_col],
            random_state=seed,
        )

        splits = {
            "train": train_df.reset_index(drop=True),
            "validation": val_df.reset_index(drop=True),
            "test": test_df.reset_index(drop=True),
        }

        # Verification: Check class coverage in all splits
        all_classes = set(df[stratify_col].unique())
        for split_name, split_df in splits.items():
            split_classes = set(split_df[stratify_col].unique())
            missing_classes = all_classes - split_classes
            if missing_classes:
                raise ValueError(
                    f"Stratified split error: Split '{split_name}' is missing classes: {missing_classes}"
                )
            logger.info(
                "Split '%s': %d records (%.1f%% of total). Covers %d/%d classes.",
                split_name,
                len(split_df),
                (len(split_df) / len(df)) * 100,
                len(split_classes),
                len(all_classes),
            )

        # Save splits to data/merged/
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        splits["train"].to_parquet(output_dir / "train.parquet", index=False)
        splits["validation"].to_parquet(output_dir / "validation.parquet", index=False)
        splits["test"].to_parquet(output_dir / "test.parquet", index=False)

        logger.info("Saved train.parquet, validation.parquet, and test.parquet to %s", output_dir)
        return splits
