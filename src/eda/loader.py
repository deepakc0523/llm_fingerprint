"""EDA Dataset Loader module and master pipeline orchestrator for LLM Fingerprinting project."""

import logging
from pathlib import Path
from typing import Tuple
import pandas as pd

from src.eda.schema import EDAConfig
from src.eda.dataset_summary import DatasetSummaryAnalyzer
from src.eda.text_statistics import TextStatisticsAnalyzer
from src.eda.vocabulary_analysis import VocabularyAnalyzer
from src.eda.label_analysis import LabelAnalyzer
from src.eda.visualization import EDAVisualizer
from src.eda.report_generator import EDAReportGenerator

logger = logging.getLogger(__name__)


def setup_logger() -> None:
    """Configure structured logging output to console."""
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


class EDALoader:
    """Loads target merged dataset and performs strict validation checks."""

    def __init__(self, config: EDAConfig):
        """Initialize loader with configuration."""
        self.config = config

    def load_and_validate(self) -> pd.DataFrame:
        """Load target parquet dataset and execute schema & data integrity checks.

        Returns:
            Validated pandas DataFrame.
        """
        dataset_path = self.config.dataset_path
        logger.info("Loading dataset from '%s'...", dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Target dataset file not found at: {dataset_path}")

        try:
            df = pd.read_parquet(dataset_path)
        except Exception as e:
            raise ValueError(f"Failed to read Parquet dataset: {e}")

        if df.empty:
            raise ValueError("Loaded dataset is empty (0 records).")

        # Integrity & schema checks
        gen_col = self.config.generated_text_col
        label_col = self.config.model_label_col

        if gen_col not in df.columns or label_col not in df.columns:
            raise ValueError(f"Dataset missing required column(s): {gen_col}, {label_col}")

        missing_vals = df.isnull().sum().to_dict()
        duplicate_rows = int(df.duplicated().sum())

        logger.info("Dataset loaded successfully: %d rows, %d columns.", len(df), len(df.columns))
        logger.info("Duplicate rows: %d | Missing value summary: %s", duplicate_rows, missing_vals)

        return df


def run_eda_pipeline(config_path: Path = Path("configs/eda.yaml")) -> None:
    """Master EDA Pipeline orchestrator: Load -> Analyze -> CSVs -> Figures -> PDF Report.

    Args:
        config_path: Path to EDA YAML configuration file.
    """
    setup_logger()
    logger.info("==================================================")
    logger.info("STARTING LLM FINGERPRINTING EDA PIPELINE (NON-MODIFYING)")
    logger.info("==================================================")

    config = EDAConfig.from_yaml(config_path)

    # 1. Load Dataset & Validate
    logger.info("--- STAGE 1: DATASET LOADING & VALIDATION ---")
    loader = EDALoader(config)
    df = loader.load_and_validate()

    # 2. Dataset Summary Analysis
    logger.info("--- STAGE 2: DATASET SUMMARY ANALYSIS ---")
    summary_analyzer = DatasetSummaryAnalyzer(config)
    summary_dict, summary_df = summary_analyzer.analyze(df)

    # 3. Text Statistics Analysis
    logger.info("--- STAGE 3: TEXT LENGTH STATISTICS ANALYSIS ---")
    text_analyzer = TextStatisticsAnalyzer(config)
    text_stats_dict, text_stats_df = text_analyzer.analyze(df)

    # 4. Vocabulary Analysis
    logger.info("--- STAGE 4: VOCABULARY ANALYSIS ---")
    vocab_analyzer = VocabularyAnalyzer(config)
    vocab_dict, vocab_df = vocab_analyzer.analyze(df)

    # 5. Model Label Analysis
    logger.info("--- STAGE 5: MODEL LABEL ANALYSIS ---")
    label_analyzer = LabelAnalyzer(config)
    label_stats_dict, label_df, corr_df = label_analyzer.analyze(df)

    # 6. Visualization Figures Generation
    logger.info("--- STAGE 6: VISUALIZATION FIGURES GENERATION ---")
    visualizer = EDAVisualizer(config)
    figure_paths = visualizer.generate_all_figures(df, corr_df)

    # 7. PDF Report Generation
    logger.info("--- STAGE 7: PUBLICATION PDF REPORT GENERATION ---")
    report_gen = EDAReportGenerator(config)
    pdf_path = report_gen.generate_report(
        summary_dict, text_stats_dict, vocab_dict, label_stats_dict, figure_paths
    )

    logger.info("==================================================")
    logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("Reports directory: %s", config.output_dir.resolve())
    logger.info("PDF Report path: %s", pdf_path.resolve())
    logger.info("==================================================")


if __name__ == "__main__":
    run_eda_pipeline()
