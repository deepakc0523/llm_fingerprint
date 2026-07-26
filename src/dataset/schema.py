"""Data schemas and dataclasses for the LLM Fingerprinting Dataset Management Layer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml


@dataclass
class DatasetSummary:
    """Summary metadata for a single model synthetic dataset."""
    dataset_name: str
    folder_path: Path
    total_records: int = 0
    is_valid: bool = False
    columns: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    duplicate_ids: int = 0
    duplicate_texts: int = 0
    error_message: Optional[str] = None


@dataclass
class ValidationResult:
    """Aggregated validation results across all scanned directories."""
    valid_datasets: List[DatasetSummary] = field(default_factory=list)
    invalid_datasets: List[DatasetSummary] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MergeConfig:
    """Strongly typed pipeline configuration object."""
    input_dir: Path
    output_dir: Path
    log_filename: str
    required_files: List[str]
    expected_columns: Dict[str, str]
    prefix_column: str
    generated_column: str
    id_column: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    random_seed: int
    stratify_column: str
    report_title: str
    report_author: str
    project_version: str

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MergeConfig":
        """Load configuration from YAML file cleanly."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        split_cfg = data.get("split", {})
        report_cfg = data.get("report", {})
        text_cols = data.get("text_columns", {})

        return cls(
            input_dir=Path(data.get("input_dir", "data/synthetic")),
            output_dir=Path(data.get("output_dir", "data/merged")),
            log_filename=data.get("log_filename", "merge.log"),
            required_files=data.get("required_files", ["generated.parquet", "metadata.json", "checkpoint.json"]),
            expected_columns=data.get("expected_columns", {}),
            prefix_column=text_cols.get("prefix", "human_prefix"),
            generated_column=text_cols.get("generated", "generated_text"),
            id_column=text_cols.get("id", "prefix_id"),
            train_ratio=float(split_cfg.get("train_ratio", 0.70)),
            val_ratio=float(split_cfg.get("val_ratio", 0.15)),
            test_ratio=float(split_cfg.get("test_ratio", 0.15)),
            random_seed=int(split_cfg.get("random_seed", 42)),
            stratify_column=split_cfg.get("stratify_column", "model_label"),
            report_title=report_cfg.get("title", "LLM Fingerprinting Merge Report"),
            report_author=report_cfg.get("author", "Research Team"),
            project_version=report_cfg.get("project_version", "1.0.0"),
        )
