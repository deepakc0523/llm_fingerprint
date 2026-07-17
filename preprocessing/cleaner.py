"""
Text cleaning and normalization tools.
"""

from typing import Dict, Any
import re
import pandas as pd


class TextCleaner:
    """Standardizes text by stripping out noise, markup, and normalizing spacing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the cleaner with dataset/preprocessing rules.

        Args:
            config: A dictionary containing preprocessing configuration keys.
        """
        self.config = config
        self.clean_html = config.get("preprocessing", {}).get("clean_html", True)
        self.remove_extra_whitespace = config.get("preprocessing", {}).get("remove_extra_whitespace", True)
        self.min_char_length = config.get("preprocessing", {}).get("min_char_length", 100)

    def remove_html(self, text: str) -> str:
        """
        Removes HTML tags from the input text string.

        Args:
            text: Raw input string.

        Returns:
            Stripped text.
        """
        # TODO: Implement regex or BeautifulSoup cleaning for markup
        return re.sub(r"<[^>]+>", "", text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Replaces double spaces, tabs, and duplicate newlines with standard spacing.

        Args:
            text: Input string.

        Returns:
            Formatted string.
        """
        # TODO: Optimize whitespace normalization for styling-critical attributes
        return " ".join(text.split())

    def clean_text(self, text: str) -> str:
        """
        Applies a complete sequence of cleaning operations to the raw text.

        Args:
            text: Raw text string.

        Returns:
            Fully cleaned text string.
        """
        cleaned = text
        if self.clean_html:
            cleaned = self.remove_html(cleaned)
        if self.remove_extra_whitespace:
            cleaned = self.normalize_whitespace(cleaned)
        # TODO: Add option to handle unicode character decoding issues
        return cleaned

    def clean_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Cleans all strings in a specific column of a Pandas DataFrame.

        Args:
            df: Target DataFrame.
            text_column: Column name containing raw text.

        Returns:
            DataFrame with a new cleaned text column.
        """
        # TODO: Implement filtering to discard rows where character length is below min_char_length
        df_copy = df.copy()
        df_copy["cleaned_text"] = df_copy[text_column].apply(self.clean_text)
        return df_copy
