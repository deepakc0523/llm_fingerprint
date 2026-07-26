"""Text Normalization engine supporting both Fingerprint-Preserving and Traditional NLP operations."""

import logging
import re
import string
from typing import Dict, Any, Set
import nltk
from nltk.stem import WordNetLemmatizer

from src.preprocessing.utils import (
    unicode_normalize,
    decode_html_entities,
    clean_extra_whitespace,
    CONTRACTIONS_DICT,
    DEFAULT_STOPWORDS,
)

logger = logging.getLogger(__name__)

# Ensure NLTK resources are safely available
try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass


class TextNormalizer:
    """Performs configurable text normalization operations for Pipelines A and B."""

    def __init__(self):
        """Initialize normalizer with lemmatizer."""
        self.lemmatizer = WordNetLemmatizer()
        self.contractions_pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r")\b",
            re.IGNORECASE,
        )

    def normalize_fingerprint_preserving(self, text: str, form: str = "NFC") -> str:
        """Pipeline A: Strictly Fingerprint-Preserving Normalization.

        DOES NOT lowercase, remove punctuation, remove stopwords, or stem/lemmatize.
        """
        if not text:
            return ""

        # 1. Unicode Normalization
        text = unicode_normalize(text, form=form)

        # 2. Decode HTML Entities
        text = decode_html_entities(text)

        # 3. Clean invalid control characters while preserving newlines/tabs
        text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or unicodedata_category_valid(ch))

        # 4. Whitespace cleanup
        text = clean_extra_whitespace(text)

        return text

    def normalize_traditional(self, text: str, form: str = "NFC") -> str:
        """Pipeline B: Traditional Classical NLP Normalization.

        Performs lowercasing, punctuation stripping, contraction expansion, stopword removal, and lemmatization.
        """
        if not text:
            return ""

        # 1. Unicode Normalization & HTML decoding
        text = unicode_normalize(text, form=form)
        text = decode_html_entities(text)

        # 2. Lowercase
        text = text.lower()

        # 3. Expand contractions
        def replace_contraction(match):
            return CONTRACTIONS_DICT.get(match.group(0).lower(), match.group(0))

        text = self.contractions_pattern.sub(replace_contraction, text)

        # 4. Remove punctuation & normalize quotes/apostrophes
        text = text.translate(str.maketrans("", "", string.punctuation))

        # 5. Tokenize for stopwords & lemmatization
        words = text.split()
        filtered_words = []

        for word in words:
            if word in DEFAULT_STOPWORDS:
                continue
            try:
                lemmatized = self.lemmatizer.lemmatize(word)
            except Exception:
                lemmatized = word
            filtered_words.append(lemmatized)

        # 6. Rejoin & clean whitespace
        result = " ".join(filtered_words)
        return clean_extra_whitespace(result)


def unicodedata_category_valid(ch: str) -> bool:
    """Check if character is not a non-printable control character."""
    import unicodedata
    cat = unicodedata.category(ch)
    return not cat.startswith("C") or ch in ("\n", "\t")
