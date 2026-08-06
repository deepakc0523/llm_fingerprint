import re
import string
import numpy as np
import pandas as pd
from collections import Counter


def preprocess_text(text: str) -> str:
    """
    Fingerprint-preserving preprocessing.
    Preserves casing, punctuation, and structural cues necessary for LLM fingerprinting.
    Performs type conversion and basic input validation.
    """
    if text is None:
        return ""
    return str(text)


def extract_stylometric_features(text: str) -> list:
    """
    Extracts the 23 stylometric features used at fusion time (Cell 52 of research notebook).
    This function supersedes the simpler 14-feature version from Cell 34.
    The features match exactly what the Hybrid Stacked Ensemble was trained on.

    Features:
        0.  char_count
        1.  word_count
        2.  sentence_count
        3.  unique_words
        4.  avg_word_length
        5.  avg_sentence_length
        6.  type_token_ratio (TTR)
        7.  hapax_ratio
        8.  uppercase_ratio
        9.  digit_ratio
        10. whitespace_ratio
        11. punctuation_ratio
        12. comma
        13. period
        14. semicolon
        15. colon
        16. question
        17. exclamation
        18. quotes
        19. dash
        20. parenthesis
        21. lexical_density
        22. short_word_ratio
    """
    if pd.isna(text) if not isinstance(text, str) else False:
        text = ""

    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)

    characters = len(text)
    word_count = len(words)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    unique_words = len(set(words))

    avg_word_length = (
        np.mean([len(w) for w in words]) if word_count > 0 else 0
    )
    avg_sentence_length = word_count / sentence_count

    # Lexical Richness
    ttr = unique_words / word_count if word_count else 0
    hapax = (
        len([w for w, c in Counter(words).items() if c == 1]) / word_count
        if word_count else 0
    )

    # Character Ratios
    uppercase_ratio = sum(c.isupper() for c in text) / max(characters, 1)
    digit_ratio = sum(c.isdigit() for c in text) / max(characters, 1)
    whitespace_ratio = sum(c.isspace() for c in text) / max(characters, 1)
    punctuation_ratio = sum(c in string.punctuation for c in text) / max(characters, 1)

    # Punctuation Counts
    comma = text.count(",")
    period = text.count(".")
    semicolon = text.count(";")
    colon = text.count(":")
    question = text.count("?")
    exclamation = text.count("!")
    quotes = text.count('"')
    dash = text.count("-")
    parenthesis = text.count("(") + text.count(")")

    # Readability Proxies
    long_words = sum(len(w) >= 7 for w in words)
    short_words = sum(len(w) <= 3 for w in words)
    lexical_density = long_words / word_count if word_count else 0
    short_word_ratio = short_words / word_count if word_count else 0

    return [
        characters,
        word_count,
        sentence_count,
        unique_words,
        avg_word_length,
        avg_sentence_length,
        ttr,
        hapax,
        uppercase_ratio,
        digit_ratio,
        whitespace_ratio,
        punctuation_ratio,
        comma,
        period,
        semicolon,
        colon,
        question,
        exclamation,
        quotes,
        dash,
        parenthesis,
        lexical_density,
        short_word_ratio
    ]
