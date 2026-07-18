"""
src.generation.checkpoint
=========================
Manages checkpoints for LLM generation runs to enable resumability.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Set

_log = logging.getLogger(__name__)


class CheckpointManager:
    """Manages reading and writing generation state checkpoints to a JSON file."""

    def __init__(self, checkpoint_path: str) -> None:
        """
        Parameters
        ----------
        checkpoint_path:
            Target location of the JSON checkpoint file (e.g., 'data/synthetic/llama3/checkpoint.json').
        """
        self.checkpoint_path = checkpoint_path
        # Keep track of completed prefixes in memory for fast lookup
        self.completed_prefix_ids: Set[str] = set()
        self.last_processed_index: int = -1

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Loads the checkpoint state from the JSON file if it exists.

        Returns
        -------
        Dict[str, Any]
            The loaded checkpoint dictionary, or an empty dictionary if it doesn't exist.
        """
        if not os.path.exists(self.checkpoint_path):
            _log.info("No checkpoint file found at %s. Starting fresh.", self.checkpoint_path)
            return {}

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.last_processed_index = data.get("last_processed_index", -1)
            completed_list = data.get("completed_prefix_ids", [])
            self.completed_prefix_ids = set(completed_list)
            
            _log.info(
                "Successfully loaded checkpoint from %s. Resuming from index %d. Completed: %d prefixes.",
                self.checkpoint_path,
                self.last_processed_index,
                len(self.completed_prefix_ids),
            )
            return data
        except Exception as e:
            _log.error("Failed to load checkpoint file %s: %s. Starting fresh.", self.checkpoint_path, e)
            return {}

    def save_checkpoint(
        self,
        last_processed_index: int,
        new_completed_ids: Set[str],
        processed_batches: int | None = None,
        processed_records: int | None = None,
        remaining_records: int | None = None,
    ) -> None:
        """
        Saves the current generation progress, completed IDs, and extra progress stats to the checkpoint file.

        Parameters
        ----------
        last_processed_index:
            The zero-indexed cursor in the prefix list.
        new_completed_ids:
            Set of newly completed prefix IDs in this batch.
        processed_batches:
            Optional total batches completed so far.
        processed_records:
            Optional total records completed so far.
        remaining_records:
            Optional total records remaining to generate.
        """
        self.last_processed_index = last_processed_index
        self.completed_prefix_ids.update(new_completed_ids)

        checkpoint_data = {
            "last_processed_index": self.last_processed_index,
            "completed_prefix_ids": list(self.completed_prefix_ids),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "progress": {
                "processed_batches": processed_batches,
                "processed_records": processed_records if processed_records is not None else len(self.completed_prefix_ids),
                "remaining_records": remaining_records,
            }
        }

        # Make sure parent directory exists
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=4)
            _log.info(
                "Saved checkpoint to %s (completed records: %d, remaining: %s)",
                self.checkpoint_path,
                len(self.completed_prefix_ids),
                str(remaining_records),
            )
        except Exception as e:
            _log.error("Failed to write checkpoint file %s: %s", self.checkpoint_path, e)

    def is_completed(self, prefix_id: str) -> bool:
        """Helper check to skip completed prefixes."""
        return prefix_id in self.completed_prefix_ids
