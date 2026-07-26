"""Master Preprocessing Loader module and CLI entry point for LLM Fingerprinting project."""

import logging
from pathlib import Path
import time
from typing import Tuple, Dict, Any
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig
from src.preprocessing.validator import PreprocessingValidator
from src.preprocessing.normalizer import TextNormalizer
from src.preprocessing.pipeline_a import PipelineAFingerprintPreserving
from src.preprocessing.pipeline_b import PipelineBTraditionalNLP
from src.preprocessing.tokenizer import CorpusTokenizer
from src.preprocessing.exporter import DatasetExporter
from src.preprocessing.report_generator import PreprocessingReportGenerator

logger = logging.getLogger(__name__)


def setup_logger(output_base_dir: Path, log_filename: str) -> None:
    """Setup structured logging to console and data/processed/preprocessing.log.

    Args:
        output_base_dir: Output base directory.
        log_filename: Log filename (preprocessing.log).
    """
    output_base_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_base_dir / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info("Logging initialized. Output log path: %s", log_file_path)


class PreprocessingLoader:
    """Loads input merged dataset for preprocessing."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize loader."""
        self.config = config

    def load_dataset(self) -> pd.DataFrame:
        """Load merged Parquet dataset.

        Returns:
            Raw merged DataFrame.
        """
        dataset_path = self.config.dataset_path
        logger.info("Loading dataset from '%s'...", dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Target dataset file not found at: {dataset_path}")

        df = pd.read_parquet(dataset_path)
        logger.info("Loaded %d rows across %d columns.", len(df), len(df.columns))
        return df


def run_preprocessing_pipeline(config_path: Path = Path("configs/preprocessing.yaml")) -> None:
    """Master Preprocessing Pipeline Runner executing Pipeline A, Pipeline B, Tokenizer, Exporter, and Comparative Reports.

    Args:
        config_path: Path to preprocessing YAML config.
    """
    config = PreprocessingConfig.from_yaml(config_path)
    setup_logger(config.output_base_dir, config.log_filename)

    logger.info("==================================================")
    logger.info("STARTING LLM FINGERPRINTING PREPROCESSING FRAMEWORK")
    logger.info("==================================================")

    # 1. Load Dataset
    loader = PreprocessingLoader(config)
    raw_df = loader.load_dataset()

    # 2. Validation & Quality Filtering
    validator = PreprocessingValidator(config)
    cleaned_df, removed_df = validator.validate_and_filter(raw_df)

    normalizer = TextNormalizer()
    tokenizer = CorpusTokenizer(config)
    exporter = DatasetExporter(config)
    report_gen = PreprocessingReportGenerator(config)

    stats_a: Dict[str, Any] = {}
    stats_b: Dict[str, Any] = {}
    time_a: float = 0.0
    time_b: float = 0.0

    # 3. Pipeline A Execution
    if config.active_pipeline in ("fingerprint", "all"):
        logger.info("--- EXECUTING PIPELINE A (FINGERPRINT-PRESERVING) ---")
        t0 = time.time()
        pipeline_a = PipelineAFingerprintPreserving(config, normalizer)
        df_a = pipeline_a.process(cleaned_df)
        time_a = time.time() - t0

        stats_a = tokenizer.compute_statistics(df_a, "fingerprint")
        exporter.export(df_a, "fingerprint")
        report_gen.generate_pipeline_reports("fingerprint", stats_a, removed_df, time_a)

    # 4. Pipeline B Execution
    if config.active_pipeline in ("traditional", "all"):
        logger.info("--- EXECUTING PIPELINE B (TRADITIONAL NLP) ---")
        t0 = time.time()
        pipeline_b = PipelineBTraditionalNLP(config, normalizer)
        df_b = pipeline_b.process(cleaned_df)
        time_b = time.time() - t0

        stats_b = tokenizer.compute_statistics(df_b, "traditional")
        exporter.export(df_b, "traditional")
        report_gen.generate_pipeline_reports("traditional", stats_b, removed_df, time_b)

    # 5. Comparative Report Generation
    if config.active_pipeline == "all" and stats_a and stats_b:
        logger.info("--- GENERATING PIPELINE A vs PIPELINE B COMPARISON ---")
        report_gen.generate_comparison_report(stats_a, stats_b, time_a, time_b)

    logger.info("==================================================")
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Processed datasets output directory: %s", config.output_base_dir.resolve())
    logger.info("Preprocessing reports directory: %s", config.reports_base_dir.resolve())
    logger.info("==================================================")


if __name__ == "__main__":
    run_preprocessing_pipeline()
