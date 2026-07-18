"""
src.dataset.dataset_merger
===========================
Merge per-dataset Parquet files into a single, globally shuffled prefix cache.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.dataset.utils import load_parquet, save_parquet

# Required output columns in the specified order.
_OUTPUT_COLUMNS = [
    "prefix_id",
    "dataset_name",
    "category",
    "document_id",
    "language",
    "prefix_length",
    "tokenizer",
    "extraction_timestamp",
    "prefix_text",
]


class DatasetMerger:
    """
    Combine per-dataset prefix Parquet files into one global prefix cache.

    Parameters
    ----------
    config:
        Configuration dictionary containing storage settings.
    logger:
        Optional :class:`logging.Logger`.  Defaults to
        ``"fingerprint.dataset_merger"``.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
        self._parquet_paths: List[str] = []
        self.config: Dict[str, Any] = config
        storage_cfg = config.get("storage", {})
        self.compression: str = storage_cfg.get("compression", "zstd")
        self.row_group_size: int = int(storage_cfg.get("row_group_size", 10000))
        self.log: logging.Logger = logger or logging.getLogger(
            "fingerprint.dataset_merger"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, parquet_path: str) -> None:
        """
        Register a per-dataset Parquet file for inclusion in the merge.

        Parameters
        ----------
        parquet_path:
            Path to a ``.parquet`` file.
        """
        if parquet_path in self._parquet_paths:
            raise ValueError(
                f"Path already registered: {parquet_path!r}."
            )
        self._parquet_paths.append(parquet_path)
        self.log.debug("Registered parquet for merge: %s", parquet_path)

    def merge(
        self,
        output_path: str,
        target_count: int = 50000,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Load all registered Parquet files, sample, shuffle, and save.

        Parameters
        ----------
        output_path:
            Destination path for the merged prefix cache.
        target_count:
            Total number of prefix rows in the merged cache.
        seed:
            Random seed for shuffling and sampling.

        Returns
        -------
        pd.DataFrame
            The merged, shuffled, and saved DataFrame.
        """
        if not self._parquet_paths:
            raise RuntimeError(
                "No Parquet files registered.  Call DatasetMerger.add() first."
            )

        rng = np.random.default_rng(seed)

        self.log.info("Loading %d dataset parquet files…", len(self._parquet_paths))
        frames: List[pd.DataFrame] = []
        for path in self._parquet_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Registered Parquet file not found at merge time: {path}"
                )
            df = load_parquet(path)
            self.log.info("  %-40s  %d rows", Path(path).name, len(df))
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        total_available = len(combined)
        self.log.info("Total rows available before sampling: %d", total_available)

        if target_count < 0 or target_count >= total_available:
            sampled = combined
        else:
            sampled = self._proportional_sample(frames, target_count, rng)

        # Shuffle globally.
        sampled = sampled.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

        # Assign globally unique prefix IDs if not already present.
        if "prefix_id" not in sampled.columns:
            sampled.insert(0, "prefix_id", [str(uuid.uuid4()) for _ in range(len(sampled))])

        # Enforce canonical column order (keep any extra columns at the end).
        ordered_cols = [c for c in _OUTPUT_COLUMNS if c in sampled.columns]
        extra_cols = [c for c in sampled.columns if c not in _OUTPUT_COLUMNS]
        sampled = sampled[ordered_cols + extra_cols]

        # Persist with zstd compression and configured row group size.
        save_parquet(
            sampled,
            output_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        self.log.info(
            "Saved prefix cache -> %s  (%d rows)", output_path, len(sampled)
        )
        return sampled

    def print_distribution(self, df: pd.DataFrame) -> None:
        """
        Log a breakdown of prefix counts per dataset and per prefix length.
        """
        self.log.info("-" * 60)
        self.log.info("Prefix cache distribution")
        self.log.info("-" * 60)

        by_dataset: Dict[str, int] = (
            df.groupby("dataset_name").size().to_dict()
        )
        for name, count in sorted(by_dataset.items()):
            self.log.info("  %-30s  %8d", name, count)

        self.log.info("-" * 60)
        by_length: Dict[int, int] = (
            df.groupby("prefix_length").size().to_dict()
        )
        for length, count in sorted(by_length.items()):
            self.log.info("  prefix_length=%-6d  %8d", length, count)

        self.log.info("-" * 60)
        self.log.info("  TOTAL  %d", len(df))
        self.log.info("-" * 60)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _proportional_sample(
        self,
        frames: List[pd.DataFrame],
        target_count: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        """
        Down-sample each dataset proportionally.
        """
        sizes = np.array([len(f) for f in frames], dtype=np.int64)
        total = sizes.sum()

        if total == 0:
            return pd.DataFrame()

        # Initial proportional quotas.
        quotas = np.floor(target_count * sizes / total).astype(np.int64)
        quotas = np.minimum(quotas, sizes)  # can't take more than available

        # Redistribute shortfall from capped datasets.
        remaining_budget = target_count - quotas.sum()
        can_give_more = sizes - quotas  # how many extra rows each can provide

        while remaining_budget > 0 and can_give_more.sum() > 0:
            eligible = can_give_more > 0
            if not eligible.any():
                break
            extra = np.zeros_like(quotas)
            share = remaining_budget // eligible.sum()
            if share == 0:
                for i in np.where(eligible)[0]:
                    if remaining_budget <= 0:
                        break
                    extra[i] = 1
                    remaining_budget -= 1
            else:
                extra[eligible] = np.minimum(share, can_give_more[eligible])
            quotas += extra
            quotas = np.minimum(quotas, sizes)
            remaining_budget = target_count - quotas.sum()
            can_give_more = sizes - quotas

        sampled_frames = []
        for df, quota in zip(frames, quotas):
            if quota <= 0:
                continue
            seed_int = int(rng.integers(0, 2**31))
            sampled_frames.append(df.sample(n=int(quota), random_state=seed_int))

        return pd.concat(sampled_frames, ignore_index=True)
