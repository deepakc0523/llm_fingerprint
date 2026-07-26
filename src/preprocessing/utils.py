"""Preprocessing Utilities and Common NLP Helper Functions."""

from dataclasses import dataclass, field
import html
import logging
from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Any, Set, Optional
import yaml

logger = logging.getLogger(__name__)

# Standard contraction mappings for Pipeline B
CONTRACTIONS_DICT: Dict[str, str] = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
    "'t": " not", "'ve": " have", "'m": " am"
}

# English stopword list for Pipeline B
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


@dataclass
class PreprocessingConfig:
    """Strongly typed pipeline configuration object."""
    active_pipeline: str
    dataset_path: Path
    output_base_dir: Path
    reports_base_dir: Path
    log_filename: str
    generated_text_col: str
    human_prefix_col: str
    prefix_id_col: str
    model_label_col: str
    min_char_length: int
    max_char_length: int
    pipeline_a_cfg: Dict[str, Any]
    pipeline_b_cfg: Dict[str, Any]
    report_title: str
    report_author: str

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PreprocessingConfig":
        """Load configuration from YAML file."""
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        text_cols = data.get("text_columns", {})
        val_cfg = data.get("validation", {})
        report_cfg = data.get("report", {})

        return cls(
            active_pipeline=data.get("active_pipeline", "all"),
            dataset_path=Path(data.get("dataset_path", "data/merged/merged_dataset.parquet")),
            output_base_dir=Path(data.get("output_base_dir", "data/processed")),
            reports_base_dir=Path(data.get("reports_base_dir", "reports/preprocessing")),
            log_filename=data.get("log_filename", "preprocessing.log"),
            generated_text_col=text_cols.get("generated_text", "generated_text"),
            human_prefix_col=text_cols.get("human_prefix", "human_prefix"),
            prefix_id_col=text_cols.get("prefix_id", "prefix_id"),
            model_label_col=text_cols.get("model_label", "model_label"),
            min_char_length=int(val_cfg.get("min_char_length", 5)),
            max_char_length=int(val_cfg.get("max_char_length", 10000)),
            pipeline_a_cfg=data.get("pipeline_a", {}),
            pipeline_b_cfg=data.get("pipeline_b", {}),
            report_title=report_cfg.get("title", "NLP Preprocessing Report"),
            report_author=report_cfg.get("author", "Senior Research Team"),
        )


def unicode_normalize(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalization form."""
    return unicodedata.normalize(form, text)


def decode_html_entities(text: str) -> str:
    """Decode HTML escape entities (e.g. &amp; -> &)."""
    return html.unescape(text)


def clean_extra_whitespace(text: str) -> str:
    """Normalize repeated whitespace while preserving paragraph spacing."""
    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace multiple spaces/tabs within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()
