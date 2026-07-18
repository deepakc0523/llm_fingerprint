"""
src.dataset.quality_checker
============================
Multi-stage quality filtering for extracted prefix records.

Every prefix must pass all of the quality, language, and duplicate filters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

import xxhash


class QualityChecker:
    """
    Stateful quality filter for prefix records.

    Parameters
    ----------
    config:
        The full dataset_engineering configuration dictionary.
    shared_seen_hashes:
        An optional shared set for tracking global duplicates across datasets.
    logger:
        Optional :class:`logging.Logger`.  Defaults to
        ``"fingerprint.quality_checker"``.
    """

    _REJECTION_LABELS = (
        "empty",
        "too_short",
        "too_long",
        "invalid_unicode",
        "language_mismatch",
        "duplicate",
    )

    def __init__(
        self,
        config: Dict[str, Any],
        shared_seen_hashes: Optional[Set[str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config
        quality_cfg = config.get("quality", {})
        self.min_char_length: int = int(quality_cfg.get("min_char_length", 150))
        self.max_char_length: int = int(quality_cfg.get("max_char_length", 6000))

        lang_cfg = config.get("language", {})
        self.lang_detector: str = lang_cfg.get("detector", "langdetect")
        self.lang_confidence_threshold: float = float(lang_cfg.get("confidence", 0.95))
        self.allowed_languages: Set[str] = set(lang_cfg.get("allowed", ["en"]))

        dup_cfg = config.get("duplicates", {})
        self.dup_method: str = dup_cfg.get("method", "xxhash")
        self.dup_compare_normalized: bool = bool(dup_cfg.get("compare_normalized", True))
        self.global_dup: bool = bool(dup_cfg.get("global", True))

        self.log: logging.Logger = logger or logging.getLogger(
            "fingerprint.quality_checker"
        )

        # Duplicate detection: use shared_seen_hashes if global_dup is True
        if self.global_dup and shared_seen_hashes is not None:
            self._seen_hashes = shared_seen_hashes
        else:
            self._seen_hashes = set()

        # Counters
        self._total_checked: int = 0
        self._total_passed: int = 0
        self._rejections: Dict[str, int] = {label: 0 for label in self._REJECTION_LABELS}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def check(self, prefix_record: Dict[str, Any]) -> bool:
        """
        Run all quality filters on a single prefix record.

        Parameters
        ----------
        prefix_record:
            Dict produced by PrefixExtractor containing text and language info.

        Returns
        -------
        bool
            ``True`` if the record passed every filter, ``False`` otherwise.
        """
        self._total_checked += 1
        text: str = prefix_record.get("prefix_text", "")
        language: str = prefix_record.get("language", "unknown")
        confidence: float = float(prefix_record.get("language_confidence", 1.0))

        # 1. Empty / whitespace-only
        if not text or not text.strip():
            self._reject("empty")
            return False

        stripped = text.strip()
        char_count = len(stripped)

        # 2. Too short
        if char_count < self.min_char_length:
            self._reject("too_short")
            return False

        # 3. Too long
        if char_count > self.max_char_length:
            self._reject("too_long")
            return False

        # 4. Invalid unicode
        if not self._is_valid_unicode(stripped):
            self._reject("invalid_unicode")
            return False

        # 5. Language validation with confidence threshold
        if self.allowed_languages:
            if language not in self.allowed_languages or confidence < self.lang_confidence_threshold:
                self._reject("language_mismatch")
                return False

        # 6. Duplicate detection via xxHash-64
        hash_input = stripped
        if self.dup_compare_normalized:
            hash_input = " ".join(stripped.lower().split())

        fingerprint = xxhash.xxh64(hash_input.encode("utf-8")).hexdigest()
        if fingerprint in self._seen_hashes:
            self._reject("duplicate")
            return False

        self._seen_hashes.add(fingerprint)
        self._total_passed += 1
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of filtering statistics accumulated so far.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing checks, passes, rejections, and pass rate.
        """
        total_rejected = self._total_checked - self._total_passed
        pass_rate = (
            self._total_passed / self._total_checked
            if self._total_checked > 0
            else 0.0
        )
        return {
            "total_checked": self._total_checked,
            "total_passed": self._total_passed,
            "total_rejected": total_rejected,
            "rejection_breakdown": dict(self._rejections),
            "pass_rate": round(pass_rate, 4),
        }

    def reset_duplicates(self) -> None:
        """Clear the duplicate hash set."""
        self._seen_hashes.clear()
        self.log.debug("Duplicate hash set cleared.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reject(self, reason: str) -> None:
        """Increment the counter for *reason* and log at DEBUG level."""
        self._rejections[reason] += 1
        self.log.debug("Prefix rejected: %s", reason)

    @staticmethod
    def _is_valid_unicode(text: str) -> bool:
        """
        Return ``True`` when *text* contains no replacement characters and
        survives a UTF-8 encode → decode round-trip without change.
        """
        if "\ufffd" in text:
            return False
        try:
            encoded = text.encode("utf-8")
            decoded = encoded.decode("utf-8")
            return decoded == text
        except (UnicodeEncodeError, UnicodeDecodeError):
            return False
