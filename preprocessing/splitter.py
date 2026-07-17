"""
Text segmentation and chunking tools.
"""

from typing import Dict, Any, List
import pandas as pd


class TextSplitter:
    """Chunks text into segments of uniform length with optional overlap."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the splitter using configured strategies and boundaries.

        Args:
            config: A dictionary containing chunk size, overlap, and segmentation strategy settings.
        """
        self.config = config
        seg_config = config.get("preprocessing", {}).get("segmentation", {})
        self.strategy = seg_config.get("strategy", "chunk")
        self.chunk_size_words = seg_config.get("chunk_size_words", 250)
        self.overlap_words = seg_config.get("overlap_words", 50)

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Splits a text string into a list of word chunks based on configuration.

        Args:
            text: Cleaned input text.

        Returns:
            A list of overlapping text chunks.
        """
        words = text.split()
        if len(words) <= self.chunk_size_words:
            return [text]

        chunks = []
        step = self.chunk_size_words - self.overlap_words
        # TODO: Implement robust token/word-level sliding window partitioning
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size_words]
            if len(chunk_words) >= self.chunk_size_words // 2:  # Keep chunks of substantial length
                chunks.append(" ".join(chunk_words))
        return chunks

    def process_dataset(self, df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
        """
        Processes a DataFrame, expanding long documents into multiple chunk records.

        Args:
            df: Target DataFrame.
            text_column: Name of the column containing target text.
            label_column: Name of the column containing author labels.

        Returns:
            A new DataFrame containing individual text segments mapping back to their original labels.
        """
        # TODO: Implement segment expansion and preserve auxiliary column metadata (IDs, source)
        rows = []
        for _, row in df.iterrows():
            chunks = self.split_into_chunks(str(row[text_column]))
            for chunk in chunks:
                new_row = row.to_dict()
                new_row[text_column] = chunk
                rows.append(new_row)
        return pd.DataFrame(rows)
