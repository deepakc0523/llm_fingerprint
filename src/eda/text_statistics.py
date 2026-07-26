"""Text Statistics module for LLM Fingerprinting EDA Layer (Non-Modifying)."""

import logging
import re
from typing import Dict, Any, Tuple
import pandas as pd

from src.eda.schema import EDAConfig

logger = logging.getLogger(__name__)


class TextStatisticsAnalyzer:
    """Computes non-modifying character, word, sentence, and paragraph length metrics."""

    def __init__(self, config: EDAConfig):
        """Initialize text statistics analyzer."""
        self.config = config

    def analyze(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Compute text length statistics across generated_text and human_prefix columns without modifying raw text.

        Args:
            df: Dataset pandas DataFrame.

        Returns:
            Tuple of (Statistics dictionary, Statistics CSV DataFrame).
        """
        logger.info("Computing non-modifying text length statistics...")

        gen_col = self.config.generated_text_col
        prefix_col = self.config.human_prefix_col

        gen_series = df[gen_col].astype(str) if gen_col in df.columns else pd.Series([], dtype=str)
        prefix_series = df[prefix_col].astype(str) if prefix_col in df.columns else pd.Series([], dtype=str)

        # Character length (exact)
        df["gen_char_len"] = gen_series.str.len()
        df["prefix_char_len"] = prefix_series.str.len()
        df["response_char_len"] = df["gen_char_len"] + df["prefix_char_len"]

        # Word count (whitespace split preserving punctuation)
        df["gen_word_count"] = gen_series.apply(lambda s: len(s.split()))
        df["prefix_word_count"] = prefix_series.apply(lambda s: len(s.split()))
        df["response_word_count"] = df["gen_word_count"] + df["prefix_word_count"]

        # Sentence count (regex punctuation match: . ! ?)
        sentence_pattern = re.compile(r"[.!?]+")
        df["gen_sentence_count"] = gen_series.apply(lambda s: max(1, len(sentence_pattern.findall(s))) if len(s.strip()) > 0 else 0)
        df["prefix_sentence_count"] = prefix_series.apply(lambda s: max(1, len(sentence_pattern.findall(s))) if len(s.strip()) > 0 else 0)
        df["response_sentence_count"] = df["gen_sentence_count"] + df["prefix_sentence_count"]

        # Paragraph count (double newline split \n\n)
        df["gen_paragraph_count"] = gen_series.apply(lambda s: len([p for p in s.split("\n\n") if p.strip()]) if len(s.strip()) > 0 else 0)
        df["prefix_paragraph_count"] = prefix_series.apply(lambda s: len([p for p in s.split("\n\n") if p.strip()]) if len(s.strip()) > 0 else 0)
        df["response_paragraph_count"] = df["gen_paragraph_count"] + df["prefix_paragraph_count"]

        metrics_list = [
            ("Generated Text Char Length", "gen_char_len"),
            ("Human Prefix Char Length", "prefix_char_len"),
            ("Total Response Char Length", "response_char_len"),
            ("Generated Text Word Count", "gen_word_count"),
            ("Human Prefix Word Count", "prefix_word_count"),
            ("Total Response Word Count", "response_word_count"),
            ("Generated Text Sentence Count", "gen_sentence_count"),
            ("Human Prefix Sentence Count", "prefix_sentence_count"),
            ("Total Response Sentence Count", "response_sentence_count"),
            ("Generated Text Paragraph Count", "gen_paragraph_count"),
            ("Human Prefix Paragraph Count", "prefix_paragraph_count"),
            ("Total Response Paragraph Count", "response_paragraph_count"),
        ]

        stats_dict: Dict[str, Any] = {}
        rows = []

        for name, col_name in metrics_list:
            s = df[col_name]
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = float(q3 - q1)

            col_stats = {
                "mean": round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "min": int(s.min()),
                "max": int(s.max()),
                "std": round(float(s.std()), 2),
                "q1": round(q1, 2),
                "q3": round(q3, 2),
                "iqr": round(iqr, 2),
            }
            stats_dict[name] = col_stats

            rows.append({
                "Metric": name,
                "Mean": col_stats["mean"],
                "Median": col_stats["median"],
                "Min": col_stats["min"],
                "Max": col_stats["max"],
                "Std Dev": col_stats["std"],
                "Q1 (25%)": col_stats["q1"],
                "Q3 (75%)": col_stats["q3"],
                "IQR": col_stats["iqr"],
            })

        stats_df = pd.DataFrame(rows)

        # Save to reports/eda/text_statistics.csv
        out_path = self.config.output_dir / "text_statistics.csv"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(out_path, index=False)
        logger.info("Saved text statistics CSV to %s", out_path)

        return stats_dict, stats_df
