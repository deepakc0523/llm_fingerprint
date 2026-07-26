"""Model Label Analysis module for LLM Fingerprinting EDA Layer."""

import logging
import re
from typing import Dict, Any, Tuple
import pandas as pd

from src.eda.schema import EDAConfig

logger = logging.getLogger(__name__)


class LabelAnalyzer:
    """Computes comparative text and vocabulary statistics broken down by target model label."""

    def __init__(self, config: EDAConfig):
        """Initialize label analyzer."""
        self.config = config

    def analyze(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """Analyze metrics per LLM model label and compute feature correlations.

        Args:
            df: Merged dataset pandas DataFrame.

        Returns:
            Tuple of (Label statistics dictionary, Label statistics DataFrame, Correlation DataFrame).
        """
        logger.info("Computing per-model label breakdown and feature correlations...")

        label_col = self.config.model_label_col
        text_col = self.config.generated_text_col
        sentence_pattern = re.compile(r"[.!?]+")

        label_stats_dict: Dict[str, Any] = {}
        rows = []

        models = sorted(df[label_col].unique()) if label_col in df.columns else []

        for model in models:
            sub_df = df[df[label_col] == model]
            texts = sub_df[text_col].astype(str).tolist()

            total_samples = len(sub_df)
            char_lengths = [len(t) for t in texts]
            avg_resp_len = round(float(sum(char_lengths) / total_samples), 2) if total_samples > 0 else 0.0

            all_words = []
            sentence_counts = []
            word_lengths = []

            for t in texts:
                words = t.split()
                all_words.extend(words)
                word_lengths.extend([len(w) for w in words])

                s_count = max(1, len(sentence_pattern.findall(t))) if len(t.strip()) > 0 else 0
                sentence_counts.append(s_count)

            total_tokens = len(all_words)
            vocab_size = len(set(all_words))
            ttr = round(vocab_size / total_tokens, 6) if total_tokens > 0 else 0.0
            avg_word_len = round(float(sum(word_lengths) / len(word_lengths)), 2) if word_lengths else 0.0
            avg_sent_len = round(float(sum(char_lengths) / sum(sentence_counts)), 2) if sum(sentence_counts) > 0 else 0.0
            unique_token_ratio = ttr

            model_metrics = {
                "total_samples": total_samples,
                "average_response_length_char": avg_resp_len,
                "vocabulary_size": vocab_size,
                "total_tokens": total_tokens,
                "type_token_ratio": ttr,
                "lexical_diversity": ttr,
                "average_sentence_length_char": avg_sent_len,
                "average_word_length_char": avg_word_len,
                "unique_token_ratio": unique_token_ratio,
            }

            label_stats_dict[str(model)] = model_metrics

            rows.append({
                "Model Label": model,
                "Total Samples": total_samples,
                "Avg Response Length (Chars)": avg_resp_len,
                "Vocabulary Size": vocab_size,
                "Total Tokens": total_tokens,
                "Type-Token Ratio (TTR)": ttr,
                "Avg Sentence Length (Chars)": avg_sent_len,
                "Avg Word Length (Chars)": avg_word_len,
                "Unique Token Ratio": unique_token_ratio,
            })

        label_df = pd.DataFrame(rows)

        # Save to reports/eda/label_statistics.csv
        out_path = self.config.output_dir / "label_statistics.csv"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        label_df.to_csv(out_path, index=False)
        logger.info("Saved label statistics CSV to %s", out_path)

        # Compute feature correlations across quantitative length features
        corr_cols = [
            "gen_char_len", "prefix_char_len", "response_char_len",
            "gen_word_count", "prefix_word_count", "response_word_count",
            "gen_sentence_count", "prefix_sentence_count"
        ]
        available_corr_cols = [c for c in corr_cols if c in df.columns]
        corr_df = df[available_corr_cols].corr().round(4) if available_corr_cols else pd.DataFrame()

        # Save to reports/eda/correlation.csv
        corr_path = self.config.output_dir / "correlation.csv"
        corr_df.to_csv(corr_path)
        logger.info("Saved correlation matrix CSV to %s", corr_path)

        return label_stats_dict, label_df, corr_df
