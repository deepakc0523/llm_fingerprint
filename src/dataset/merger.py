"""Dataset Merger module and unified CLI orchestrator for LLM Fingerprinting Research Project."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import yaml

from src.dataset.schema import MergeConfig, ValidationResult
from src.dataset.validator import DatasetValidator
from src.dataset.statistics import DatasetStatisticsGenerator
from src.dataset.splitter import DatasetSplitter
from src.dataset.report_generator import MergeReportGenerator

logger = logging.getLogger(__name__)


def setup_logger(output_dir: Path, log_filename: str) -> None:
    """Setup structured logging output to console and merge.log inside merged data folder.

    Args:
        output_dir: Directory where log file will be saved.
        log_filename: Name of the log file (e.g. merge.log).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / log_filename

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers if any
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging initialized. Output log path: %s", log_file_path)


class DatasetMerger:
    """Merges dynamic synthetic datasets into a single unified dataframe."""

    def __init__(self, config: MergeConfig):
        """Initialize merger with pipeline configuration.

        Args:
            config: Clean MergeConfig instance.
        """
        self.config = config

    def merge_datasets(self, validation_result: ValidationResult) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Merge all valid datasets into a single DataFrame.

        Extracts model label dynamically from folder name, appends `model_label`,
        validates schema identity across datasets, and performs deterministic shuffle.

        Args:
            validation_result: Result from DatasetValidator scan.

        Returns:
            Tuple of (Merged pandas DataFrame, Label Mapping Dictionary).
        """
        valid_datasets = validation_result.valid_datasets
        if not valid_datasets:
            raise ValueError("No valid datasets available to merge!")

        logger.info("Starting merge process for %d valid dataset(s)...", len(valid_datasets))
        dataframes: List[pd.DataFrame] = []
        label_mapping: Dict[str, int] = {}

        # 1. Read each valid dataset and dynamically append model_label
        for idx, summary in enumerate(sorted(valid_datasets, key=lambda x: x.dataset_name)):
            folder = summary.folder_path
            model_label = folder.name
            label_mapping[model_label] = idx

            parquet_path = folder / "generated.parquet"
            logger.info("Loading dataset '%s' from %s", model_label, parquet_path)
            df = pd.read_parquet(parquet_path)

            # Extract label dynamically from folder name
            df["model_label"] = model_label

            dataframes.append(df)

        # 2. Validate identical schema across all datasets
        base_columns = list(dataframes[0].columns)
        for df in dataframes[1:]:
            if list(df.columns) != base_columns:
                raise ValueError("Schema mismatch detected across datasets prior to merging!")

        # 3. Concatenate datasets
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info("Concatenated %d records across %d models.", len(merged_df), len(dataframes))

        # 4. Perform deterministic shuffle
        seed = self.config.random_seed
        logger.info("Performing deterministic shuffle with random seed %d...", seed)
        merged_df = merged_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

        # 5. Save merged_dataset.parquet and label_mapping.json
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_parquet_path = output_dir / "merged_dataset.parquet"
        merged_df.to_parquet(merged_parquet_path, index=False)
        logger.info("Saved merged dataset to %s", merged_parquet_path)

        label_map_path = output_dir / "label_mapping.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=4)
        logger.info("Saved label mapping to %s", label_map_path)

        return merged_df, label_mapping


def run_pipeline(config_path: Path = Path("configs/dataset_merge.yaml")) -> None:
    """Master pipeline runner executing Validation -> Merge -> Statistics -> Split -> PDF Report -> Logging.

    Args:
        config_path: Path to dataset merge YAML configuration.
    """
    config = MergeConfig.from_yaml(config_path)
    setup_logger(config.output_dir, config.log_filename)

    logger.info("==================================================")
    logger.info("STARTING LLM FINGERPRINTING DATASET MANAGEMENT PIPELINE")
    logger.info("==================================================")

    # 1. Validation
    logger.info("--- STAGE 1: VALIDATION ---")
    validator = DatasetValidator(config)
    val_result = validator.validate_all()

    if not val_result.valid_datasets:
        logger.error("Pipeline aborted: 0 valid datasets found.")
        return

    # 2. Merge
    logger.info("--- STAGE 2: MERGE ---")
    merger = DatasetMerger(config)
    merged_df, label_mapping = merger.merge_datasets(val_result)

    # 3. Statistics
    logger.info("--- STAGE 3: STATISTICS GENERATION ---")
    stats_gen = DatasetStatisticsGenerator(config)
    stats = stats_gen.generate_statistics(merged_df, val_result)

    # 4. Stratified Splitter
    logger.info("--- STAGE 4: STRATIFIED SPLITTING ---")
    splitter = DatasetSplitter(config)
    splits = splitter.split_dataset(merged_df)

    # 5. PDF Report Generation
    logger.info("--- STAGE 5: PDF REPORT GENERATION ---")
    report_gen = MergeReportGenerator(config)
    report_gen.generate_report(val_result, stats, splits)

    logger.info("==================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Output directory: %s", config.output_dir.resolve())
    logger.info("==================================================")


if __name__ == "__main__":
    run_pipeline()
