"""Vocabulary Analysis module for LLM Fingerprinting EDA Layer (Strictly Non-Modifying)."""

from collections import Counter
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd

from src.eda.schema import EDAConfig

logger = logging.getLogger(__name__)


class VocabularyAnalyzer:
    """Computes vocabulary size, Type-Token Ratio (TTR), n-grams, and character frequencies on raw text."""

    def __init__(self, config: EDAConfig):
        """Initialize vocabulary analyzer."""
        self.config = config

    def analyze(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """Analyze corpus vocabulary, diversity metrics, and n-gram distributions.

        Args:
            df: Merged dataset pandas DataFrame.

        Returns:
            Tuple of (Vocabulary metrics dictionary, Vocabulary CSV DataFrame).
        """
        logger.info("Computing vocabulary statistics on raw corpus...")

        text_col = self.config.generated_text_col
        texts = df[text_col].astype(str).tolist() if text_col in df.columns else []

        all_tokens: List[str] = []
        char_counter: Counter = Counter()
        bigram_counter: Counter = Counter()
        trigram_counter: Counter = Counter()

        for text in texts:
            # Update character counts (exact)
            char_counter.update(text)

            # Whitespace split preserving casing & punctuation
            tokens = text.split()
            all_tokens.extend(tokens)

            # N-grams
            if len(tokens) >= 2:
                bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
                bigram_counter.update(bigrams)
            if len(tokens) >= 3:
                trigrams = [f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}" for i in range(len(tokens) - 2)]
                trigram_counter.update(trigrams)

        total_tokens = len(all_tokens)
        token_counter = Counter(all_tokens)
        vocab_size = len(token_counter)

        type_token_ratio = round(vocab_size / total_tokens, 6) if total_tokens > 0 else 0.0
        lexical_diversity = type_token_ratio  # TTR standard measure

        most_frequent = token_counter.most_common(20)
        least_frequent = token_counter.most_common()[:-21:-1] if vocab_size >= 20 else []

        top_bigrams = bigram_counter.most_common(20)
        top_trigrams = trigram_counter.most_common(20)

        vocab_dict: Dict[str, Any] = {
            "total_tokens": total_tokens,
            "vocabulary_size": vocab_size,
            "type_token_ratio": type_token_ratio,
            "lexical_diversity": lexical_diversity,
            "most_frequent_words": dict(most_frequent),
            "least_frequent_words": dict(least_frequent),
            "top_bigrams": dict(top_bigrams),
            "top_trigrams": dict(top_trigrams),
            "character_frequency_top20": dict(char_counter.most_common(20)),
        }

        # Build vocabulary CSV DataFrame
        rows = [
            {"Category": "Overview", "Item": "Total Tokens", "Count": total_tokens, "Frequency_Percentage": 100.0},
            {"Category": "Overview", "Item": "Vocabulary Size (Unique)", "Count": vocab_size, "Frequency_Percentage": 100.0},
            {"Category": "Overview", "Item": "Type-Token Ratio (TTR)", "Count": type_token_ratio, "Frequency_Percentage": type_token_ratio * 100},
        ]

        for word, count in most_frequent:
            pct = round((count / total_tokens) * 100, 4) if total_tokens > 0 else 0.0
            rows.append({"Category": "Most Frequent Words", "Item": repr(word), "Count": count, "Frequency_Percentage": pct})

        for bigram, count in top_bigrams:
            pct = round((count / (total_tokens - 1)) * 100, 4) if total_tokens > 1 else 0.0
            rows.append({"Category": "Top Bigrams", "Item": repr(bigram), "Count": count, "Frequency_Percentage": pct})

        for trigram, count in top_trigrams:
            pct = round((count / (total_tokens - 2)) * 100, 4) if total_tokens > 2 else 0.0
            rows.append({"Category": "Top Trigrams", "Item": repr(trigram), "Count": count, "Frequency_Percentage": pct})

        vocab_df = pd.DataFrame(rows)

        # Save to reports/eda/vocabulary.csv
        out_path = self.config.output_dir / "vocabulary.csv"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        vocab_df.to_csv(out_path, index=False)
        logger.info("Saved vocabulary analysis CSV to %s", out_path)

        return vocab_dict, vocab_df
