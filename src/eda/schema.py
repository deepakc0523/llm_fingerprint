"""Data schemas and dataclasses for the LLM Fingerprinting EDA Layer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


@dataclass
class EDAConfig:
    """Strongly typed configuration object for EDA Pipeline."""
    dataset_path: Path
    output_dir: Path
    figures_dir: Path
    generated_text_col: str
    human_prefix_col: str
    prefix_id_col: str
    model_label_col: str
    target_labels: List[str]
    report_title: str
    report_author: str
    project_name: str
    version: str

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "EDAConfig":
        """Load configuration from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        text_cols = data.get("text_columns", {})
        report_cfg = data.get("report", {})

        return cls(
            dataset_path=Path(data.get("dataset_path", "data/merged/merged_dataset.parquet")),
            output_dir=Path(data.get("output_dir", "reports/eda")),
            figures_dir=Path(data.get("figures_dir", "reports/eda/figures")),
            generated_text_col=text_cols.get("generated_text", "generated_text"),
            human_prefix_col=text_cols.get("human_prefix", "human_prefix"),
            prefix_id_col=text_cols.get("prefix_id", "prefix_id"),
            model_label_col=text_cols.get("model_label", "model_label"),
            target_labels=data.get("target_labels", ["gemma2", "llama3", "mistral", "phi3", "qwen_tiny"]),
            report_title=report_cfg.get("title", "Exploratory Data Analysis Report"),
            report_author=report_cfg.get("author", "Research Team"),
            project_name=report_cfg.get("project_name", "Fingerprint"),
            version=report_cfg.get("version", "1.0.0"),
        )
