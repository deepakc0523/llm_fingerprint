"""Dataset Summary module for LLM Fingerprinting EDA Layer."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd

from src.eda.schema import EDAConfig

logger = logging.getLogger(__name__)


class DatasetSummaryAnalyzer:
    """Computes macro dataset metadata, dimensions, missing values, duplicates, and class balance."""

    def __init__(self, config: EDAConfig):
        """Initialize dataset summary analyzer with configuration."""
        self.config = config

    def analyze(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute macro dataset summary statistics and generate summary DataFrame.

        Args:
            df: Merged dataset pandas DataFrame.

        Returns:
            Tuple of (Summary dictionary, Summary pandas DataFrame).
        """
        logger.info("Computing dataset summary statistics for %d rows...", len(df))

        total_samples = len(df)
        label_col = self.config.model_label_col
        id_col = self.config.prefix_id_col
        text_col = self.config.generated_text_col

        class_counts = df[label_col].value_counts().to_dict() if label_col in df.columns else {}
        total_classes = len(class_counts)
        max_count = max(class_counts.values()) if class_counts else 0
        min_count = min(class_counts.values()) if class_counts else 0
        imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else 0.0

        mem_bytes = df.memory_usage(deep=True).sum()
        mem_mb = round(mem_bytes / (1024 * 1024), 2)

        file_size_bytes = self.config.dataset_path.stat().st_size if self.config.dataset_path.exists() else 0
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

        duplicate_rows = int(df.duplicated().sum())
        duplicate_ids = int(df[id_col].duplicated().sum()) if id_col in df.columns else 0
        duplicate_texts = int(df[text_col].duplicated().sum()) if text_col in df.columns else 0

        null_summary = df.isnull().sum()
        total_missing = int(null_summary.sum())

        summary_dict: Dict[str, Any] = {
            "total_samples": total_samples,
            "total_classes": total_classes,
            "class_imbalance_ratio": imbalance_ratio,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": mem_mb,
            "dataset_file_size_mb": file_size_mb,
            "duplicate_rows": duplicate_rows,
            "duplicate_ids": duplicate_ids,
            "duplicate_generated_texts": duplicate_texts,
            "total_missing_values": total_missing,
            "class_distribution": class_counts,
        }

        # Build tabular CSV DataFrame
        summary_rows = [
            {"Metric": "Total Samples", "Value": str(total_samples)},
            {"Metric": "Total Classes", "Value": str(total_classes)},
            {"Metric": "Class Imbalance Ratio", "Value": str(imbalance_ratio)},
            {"Metric": "Rows", "Value": str(len(df))},
            {"Metric": "Columns", "Value": str(len(df.columns))},
            {"Metric": "Memory Usage (MB)", "Value": str(mem_mb)},
            {"Metric": "Dataset File Size (MB)", "Value": str(file_size_mb)},
            {"Metric": "Duplicate Rows", "Value": str(duplicate_rows)},
            {"Metric": "Duplicate Prefix IDs", "Value": str(duplicate_ids)},
            {"Metric": "Duplicate Generated Texts", "Value": str(duplicate_texts)},
            {"Metric": "Total Missing Values", "Value": str(total_missing)},
        ]
        for cls_name, count in class_counts.items():
            summary_rows.append({"Metric": f"Class Count ({cls_name})", "Value": str(count)})

        summary_df = pd.DataFrame(summary_rows)

        # Save to reports/eda/dataset_summary.csv
        out_path = self.config.output_dir / "dataset_summary.csv"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        logger.info("Saved dataset summary CSV to %s", out_path)

        return summary_dict, summary_df
