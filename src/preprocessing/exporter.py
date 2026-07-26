"""Dataset Exporter module for NLP Preprocessing Layer."""

import logging
from pathlib import Path
import pandas as pd

from src.preprocessing.utils import PreprocessingConfig

logger = logging.getLogger(__name__)


class DatasetExporter:
    """Exports processed datasets to data/processed/[fingerprint|traditional]/ in Parquet and CSV formats."""

    def __init__(self, config: PreprocessingConfig):
        """Initialize dataset exporter."""
        self.config = config

    def export(self, df: pd.DataFrame, pipeline_name: str) -> Path:
        """Save processed dataset in Parquet and CSV formats.

        Args:
            df: Processed pandas DataFrame.
            pipeline_name: Pipeline identifier ('fingerprint' or 'traditional').

        Returns:
            Path to target output directory.
        """
        out_dir = self.config.output_base_dir / pipeline_name
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / "processed_dataset.parquet"
        csv_path = out_dir / "processed_dataset.csv"

        logger.info("Exporting processed dataset for '%s' to %s...", pipeline_name, out_dir)

        # Save Parquet
        df.to_parquet(parquet_path, index=False)
        logger.info("Saved Parquet: %s", parquet_path)

        # Save CSV
        df.to_csv(csv_path, index=False)
        logger.info("Saved CSV: %s", csv_path)

        return out_dir
