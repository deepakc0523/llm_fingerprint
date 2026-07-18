"""
src.dataset.builder
====================
Top-level pipeline orchestrator for Stage 1 — Dataset Engineering.

:class:`DatasetBuilder` sequences the five source datasets, streams each one
document at a time, extracts prefixes, filters for quality, accumulates records
in memory-bounded batches, writes per-dataset Parquet files, and finally calls
:class:`~src.dataset.dataset_merger.DatasetMerger` to produce the global
``prefix_cache.parquet``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import xxhash

from src.dataset.dataset_merger import DatasetMerger
from src.dataset.prefix_extractor import PrefixExtractor
from src.dataset.quality_checker import QualityChecker
from src.dataset.stream_loader import StreamLoader
from src.dataset.utils import (
    ensure_dir,
    get_logger,
    get_progress_bar,
    load_parquet,
    load_yaml_config,
    save_parquet,
    set_random_seed,
)


class DatasetBuilder:
    """
    Orchestrate the complete Stage 1 dataset engineering pipeline.

    Parameters
    ----------
    config_path:
        Path to ``dataset_engineering.yaml``.
    dry_run_limit:
        When > 0, stop processing each dataset after this many *documents*
        (useful for smoke tests).  Pass ``0`` (default) for a full run.
    """

    def __init__(self, config_path: str, dry_run_limit: int = 0) -> None:
        raw_cfg = load_yaml_config(config_path)
        self.cfg: Dict[str, Any] = raw_cfg.get("dataset_engineering", {})
        self.dry_run_limit: int = dry_run_limit

        # ── Paths ──────────────────────────────────────────────────────
        self.output_dir: str = self.cfg.get("output_dir", "data/prefixes")
        self.checkpoint_dir: str = self.cfg.get(
            "checkpoint_dir", "data/prefixes/.checkpoints"
        )
        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_dir)

        # ── Logger ─────────────────────────────────────────────────────
        self.log = get_logger(
            name="fingerprint.builder",
            level=self.cfg.get("log_level", "INFO"),
            log_file=self.cfg.get("log_file"),
        )

        # ── Reproducibility ────────────────────────────────────────────
        self.random_seed: int = int(self.cfg.get("random_seed", 42))
        self.python_seed: int = int(self.cfg.get("python_seed", 42))
        self.numpy_seed: int = int(self.cfg.get("numpy_seed", 42))
        set_random_seed(self.random_seed, self.python_seed, self.numpy_seed)

        # ── Pipeline parameters ────────────────────────────────────────
        self.target_total: int = int(self.cfg.get("target_total_prefixes", 50000))
        self.batch_size: int = int(self.cfg.get("batch_size", 1000))
        self.checkpoint_interval: int = int(self.cfg.get("checkpoint_interval", 5000))
        self.prefix_lengths: List[int] = list(
            self.cfg.get("prefix_lengths", [64, 128, 256])
        )
        self.tokenizer_name: str = self.cfg.get("tokenizer", {}).get(
            "model_name", "gpt2"
        )
        self.quality_cfg: Dict[str, Any] = self.cfg.get("quality", {})
        self.datasets_cfg: Dict[str, Any] = self.cfg.get("datasets", {})

        # ── Storage settings ──────────────────────────────────────────
        storage_cfg = self.cfg.get("storage", {})
        self.compression: str = storage_cfg.get("compression", "zstd")
        self.row_group_size: int = int(storage_cfg.get("row_group_size", 10000))

        # ── Global duplicate detection ──────────────────────────────────
        self.global_seen_hashes: Set[str] = set()
        self._load_existing_duplicates()

        # ── Pipeline statistics ────────────────────────────────────────
        self.stats_docs_processed: int = 0
        self.stats_docs_skipped: int = 0
        self.stats_prefixes_extracted: int = 0
        self.stats_duplicates_removed: int = 0
        self.stats_language_rejections: int = 0
        self.stats_quality_rejections: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Execute the complete pipeline end-to-end.
        """
        self.log.info("=" * 70)
        self.log.info("Fingerprint - Stage 1: Dataset Engineering")
        self.log.info("Target total prefixes : %d", self.target_total)
        self.log.info("Datasets              : %s", ", ".join(self.datasets_cfg.keys()))
        self.log.info("Tokenizer             : %s", self.tokenizer_name)
        self.log.info("Prefix lengths        : %s", self.prefix_lengths)
        if self.dry_run_limit:
            self.log.info("DRY RUN - %d documents per dataset", self.dry_run_limit)
        self.log.info("=" * 70)

        start_time = time.time()
        produced_parquets: List[str] = []

        all_dataset_names = list(self.datasets_cfg.keys())

        # Determine remaining datasets that are not marked completed
        remaining_active_datasets = [d for d in all_dataset_names if not self._is_completed(d)]

        for dataset_name in all_dataset_names:
            parquet_path = str(
                Path(self.output_dir) / f"{dataset_name}.parquet"
            )

            # Check total collected prefixes so far across all datasets to compute remaining budget
            total_collected_so_far = 0
            for name in all_dataset_names:
                ckpt = self._load_checkpoint(name)
                total_collected_so_far += ckpt.get("prefixes_collected", 0)

            remaining_global_target = max(0, self.target_total - total_collected_so_far)

            if self._is_completed(dataset_name):
                self.log.info(
                    "[%s] Already completed - skipping.", dataset_name
                )
                produced_parquets.append(parquet_path)
                continue

            # Calculate dynamic target prefixes for approximately balanced distribution
            if len(remaining_active_datasets) > 0:
                target_prefixes = int(np.ceil(remaining_global_target / len(remaining_active_datasets)))
            else:
                target_prefixes = 0

            self.log.info("-" * 70)
            self.log.info("[%s] Starting extraction (Dynamic Target: %d) ...", dataset_name, target_prefixes)
            self._process_dataset(dataset_name, self.datasets_cfg[dataset_name], parquet_path, target_prefixes)
            produced_parquets.append(parquet_path)

            # Remove current dataset from active tracking
            if dataset_name in remaining_active_datasets:
                remaining_active_datasets.remove(dataset_name)

        # ── Merge phase ───────────────────────────────────────────────
        self.log.info("-" * 70)
        self.log.info("Merging datasets into prefix_cache.parquet ...")
        cache_path = str(Path(self.output_dir) / "prefix_cache.parquet")
        merger = DatasetMerger(config=self.cfg, logger=self.log)
        for path in produced_parquets:
            if Path(path).is_file():
                merger.add(path)
            else:
                self.log.warning("Parquet not found, skipping from merge: %s", path)

        cache_df = merger.merge(
            output_path=cache_path,
            target_count=self.target_total,
            seed=self.random_seed,
        )
        merger.print_distribution(cache_df)

        elapsed = time.time() - start_time

        # ── Save stats ───────────────────────────────────────────────
        avg_len = float(cache_df["prefix_length"].mean()) if len(cache_df) > 0 else 0.0
        dist = cache_df.groupby("dataset_name").size().to_dict()

        stats_data = {
            "documents_processed": self.stats_docs_processed,
            "documents_skipped": self.stats_docs_skipped,
            "prefixes_extracted": self.stats_prefixes_extracted,
            "duplicates_removed": self.stats_duplicates_removed,
            "language_rejections": self.stats_language_rejections,
            "quality_rejections": self.stats_quality_rejections,
            "average_prefix_length": round(avg_len, 2),
            "dataset_distribution": dist,
            "processing_time": round(elapsed, 2),
        }

        stats_path = os.path.join(self.output_dir, "statistics.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)
        self.log.info("Saved pipeline statistics to %s", stats_path)

        self.log.info("=" * 70)
        self.log.info(
            "Pipeline complete.  Total time: %.1f s  |  Cache rows: %d",
            elapsed,
            len(cache_df),
        )
        self.log.info("=" * 70)

    # ------------------------------------------------------------------
    # Per-dataset orchestration
    # ------------------------------------------------------------------

    def _process_dataset(
        self,
        dataset_name: str,
        dataset_cfg: Dict[str, Any],
        parquet_path: str,
        target_prefixes: int,
    ) -> None:
        """
        Stream, extract, filter, and save prefixes for a single dataset.
        """
        checkpoint = self._load_checkpoint(dataset_name)
        resume_from = checkpoint.get("last_document_index", 0)
        prefixes_collected: int = checkpoint.get("prefixes_collected", 0)

        if prefixes_collected >= target_prefixes:
            self.log.info(
                "[%s] Target already met (%d / %d).  Finalising.",
                dataset_name,
                prefixes_collected,
                target_prefixes,
            )
            self._mark_completed(dataset_name, prefixes_collected)
            return

        loader = StreamLoader(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            logger=self.log,
        )
        extractor = PrefixExtractor(
            config=self.cfg,
            logger=self.log,
        )
        checker = QualityChecker(
            config=self.cfg,
            shared_seen_hashes=self.global_seen_hashes,
            logger=self.log,
        )

        sampling_cfg = self.cfg.get("sampling", {})
        max_prefixes_per_document = int(sampling_cfg.get("max_prefixes_per_document", 10))

        # Buffer that is flushed every `batch_size` accepted prefixes.
        buffer: List[Dict[str, Any]] = []
        doc_index: int = resume_from
        docs_processed: int = 0

        pbar = get_progress_bar(
            loader.load(resume_from_index=resume_from),
            desc=f"[{dataset_name}]",
            unit="doc",
        )

        for doc in pbar:
            doc_index += 1
            docs_processed += 1
            self.stats_docs_processed += 1

            # Dry-run early exit.
            if self.dry_run_limit and docs_processed > self.dry_run_limit:
                self.log.info(
                    "[%s] Dry-run limit (%d docs) reached.",
                    dataset_name,
                    self.dry_run_limit,
                )
                break

            prefix_records = extractor.extract(
                text=doc["text"],
                document_id=doc["document_id"],
                dataset_name=dataset_name,
                split=doc["source_split"],
                category=dataset_cfg.get("category", ""),
            )

            if not prefix_records:
                self.stats_docs_skipped += 1
                continue

            self.stats_prefixes_extracted += len(prefix_records)
            doc_prefixes_accepted = 0

            for record in prefix_records:
                if prefixes_collected >= target_prefixes:
                    break
                if doc_prefixes_accepted >= max_prefixes_per_document:
                    break

                if checker.check(record):
                    # We drop the helper key before saving to file
                    clean_record = record.copy()
                    if "language_confidence" in clean_record:
                        del clean_record["language_confidence"]
                    buffer.append(clean_record)
                    prefixes_collected += 1
                    doc_prefixes_accepted += 1

            # Flush buffer to Parquet when batch is full.
            if len(buffer) >= self.batch_size:
                self._flush_buffer(buffer, parquet_path)
                buffer.clear()

            # Periodic checkpoint save.
            if docs_processed % self.checkpoint_interval == 0:
                self._save_checkpoint(dataset_name, doc_index, prefixes_collected)
                stats = checker.get_stats()
                self.log.info(
                    "[%s] docs=%d  prefixes=%d/%d  pass_rate=%.2f%%",
                    dataset_name,
                    docs_processed,
                    prefixes_collected,
                    target_prefixes,
                    stats["pass_rate"] * 100,
                )

            if prefixes_collected >= target_prefixes:
                self.log.info(
                    "[%s] Reached target of %d prefixes.", dataset_name, target_prefixes
                )
                break

        pbar.close()

        # Flush remaining buffer.
        if buffer:
            self._flush_buffer(buffer, parquet_path)

        # Final stats log & accumulate rejections to builder counters
        stats = checker.get_stats()
        rejections = stats.get("rejection_breakdown", {})
        self.stats_duplicates_removed += rejections.get("duplicate", 0)
        self.stats_language_rejections += rejections.get("language_mismatch", 0)
        self.stats_quality_rejections += (
            rejections.get("empty", 0) +
            rejections.get("too_short", 0) +
            rejections.get("too_long", 0) +
            rejections.get("invalid_unicode", 0)
        )

        self.log.info(
            "[%s] DONE - docs_processed=%d  prefixes_collected=%d  "
            "total_checked=%d  rejected=%d  pass_rate=%.2f%%",
            dataset_name,
            docs_processed,
            prefixes_collected,
            stats["total_checked"],
            stats["total_rejected"],
            stats["pass_rate"] * 100,
        )

        self._save_checkpoint(dataset_name, doc_index, prefixes_collected)
        self._mark_completed(dataset_name, prefixes_collected)

    # ------------------------------------------------------------------
    # Buffer → Parquet flushing
    # ------------------------------------------------------------------

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]], parquet_path: str
    ) -> None:
        """
        Append *buffer* to the per-dataset Parquet file.
        """
        new_df = pd.DataFrame(buffer)

        if Path(parquet_path).is_file():
            existing_df = load_parquet(parquet_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        save_parquet(
            combined_df,
            parquet_path,
            compression=self.compression,
            row_group_size=self.row_group_size,
        )
        self.log.debug(
            "Flushed %d records to %s  (total rows now: %d)",
            len(new_df),
            parquet_path,
            len(combined_df),
        )

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _checkpoint_path(self, dataset_name: str) -> str:
        """Return the path to the JSON checkpoint file for *dataset_name*."""
        return str(
            Path(self.checkpoint_dir) / f"{dataset_name}_checkpoint.json"
        )

    def _load_checkpoint(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load the checkpoint file if it exists.
        """
        path = self._checkpoint_path(dataset_name)
        if not Path(path).is_file():
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _save_checkpoint(
        self, dataset_name: str, last_document_index: int, prefixes_collected: int
    ) -> None:
        """
        Persist a checkpoint.
        """
        checkpoint = {
            "dataset_name": dataset_name,
            "last_document_index": last_document_index,
            "prefixes_collected": prefixes_collected,
            "completed": False,
        }
        path = self._checkpoint_path(dataset_name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)
        self.log.debug("Checkpoint saved: %s", path)

    def _mark_completed(
        self, dataset_name: str, prefixes_collected: int
    ) -> None:
        """
        Update the checkpoint file to mark completed.
        """
        path = self._checkpoint_path(dataset_name)
        if Path(path).is_file():
            with open(path, "r", encoding="utf-8") as fh:
                checkpoint = json.load(fh)
        else:
            checkpoint = {"dataset_name": dataset_name}

        checkpoint["completed"] = True
        checkpoint["prefixes_collected"] = prefixes_collected
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)
        self.log.info("[%s] Marked as completed.", dataset_name)

    def _is_completed(self, dataset_name: str) -> bool:
        """
        Return True if completed.
        """
        checkpoint = self._load_checkpoint(dataset_name)
        return bool(checkpoint.get("completed", False))

    def _load_existing_duplicates(self) -> None:
        """
        Preload duplicate hashes from existing per-dataset Parquet files to support resume.
        """
        dup_cfg = self.cfg.get("duplicates", {})
        compare_normalized = bool(dup_cfg.get("compare_normalized", True))

        for dataset_name in self.datasets_cfg.keys():
            parquet_path = Path(self.output_dir) / f"{dataset_name}.parquet"
            if parquet_path.is_file():
                try:
                    df = load_parquet(str(parquet_path))
                    if "prefix_text" in df.columns:
                        for text in df["prefix_text"]:
                            stripped = str(text).strip()
                            if compare_normalized:
                                hash_input = " ".join(stripped.lower().split())
                            else:
                                hash_input = stripped
                            fingerprint = xxhash.xxh64(hash_input.encode("utf-8")).hexdigest()
                            self.global_seen_hashes.add(fingerprint)
                except Exception as e:
                    self.log.warning("Could not preload duplicates from %s: %s", parquet_path, e)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset.builder",
        description="Fingerprint Stage 1 — Dataset Engineering Pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_engineering.yaml",
        help="Path to dataset_engineering.yaml (default: configs/dataset_engineering.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=0,
        metavar="N",
        help="Process only N documents per dataset (0 = full run)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    builder = DatasetBuilder(
        config_path=args.config,
        dry_run_limit=args.dry_run,
    )
    builder.run()
