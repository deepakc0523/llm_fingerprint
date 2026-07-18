"""
src.generation.parquet_writer
=============================
Handles writing/appending LLM-generated samples to Apache Parquet files
ensuring strict adherence to the target database schema.
Optimized to avoid repetitive file reads and rewrites.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_log = logging.getLogger(__name__)


@dataclass
class GeneratedRecord:
    """Dataclass specifying the exact output schema for the Stage 2 generated records."""
    prefix_id: str
    dataset_name: str
    category: str
    human_prefix: str
    generated_text: str
    model_name: str
    temperature: float
    top_p: float
    max_new_tokens: int
    prompt_length: int          # prompt length in characters or tokens
    completion_length: int      # completion length in characters or tokens
    generation_time: float      # duration in seconds
    timestamp: str              # ISO 8601 formatted timestamp


class ParquetBatchWriter:
    """
    Manages batch serialization of GeneratedRecord objects into a Parquet file.
    Streams incremental batches to avoid reading the whole file repeatedly.
    """

    # Schema definition using PyArrow types for precise database schema enforcement
    SCHEMA = pa.schema([
        ("prefix_id", pa.string()),
        ("dataset_name", pa.string()),
        ("category", pa.string()),
        ("human_prefix", pa.string()),
        ("generated_text", pa.string()),
        ("model_name", pa.string()),
        ("temperature", pa.float64()),
        ("top_p", pa.float64()),
        ("max_new_tokens", pa.int64()),
        ("prompt_length", pa.int64()),
        ("completion_length", pa.int64()),
        ("generation_time", pa.float64()),
        ("timestamp", pa.string()),
    ])

    def __init__(self, output_path: str) -> None:
        """
        Parameters
        ----------
        output_path:
            Target file path to write the parquet output (e.g., 'data/synthetic/llama3/generated.parquet').
        """
        self.output_path = output_path
        self._writer: Optional[pq.ParquetWriter] = None
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def write_batch(self, records: List[GeneratedRecord]) -> None:
        """
        Appends or writes a list of records to the target Parquet file.
        Utilizes a single open ParquetWriter to write directly, only reading existing data once.

        Parameters
        ----------
        records:
            List of GeneratedRecord instances.
        """
        if not records:
            return

        # Convert to dictionary representation matching the schema fields
        dict_records = [asdict(r) for r in records]
        df = pd.DataFrame(dict_records)

        # Convert to PyArrow Table using the strict schema definition
        table = pa.Table.from_pandas(df, schema=self.SCHEMA, preserve_index=False)

        if self._writer is None:
            existing_table = None
            if os.path.exists(self.output_path):
                try:
                    # Read existing data only once to append to the stream
                    _log.debug("Reading existing Parquet file once to initialize append stream: %s", self.output_path)
                    existing_table = pq.read_table(self.output_path)
                except Exception as e:
                    _log.warning(
                        "Could not read existing file at %s: %s. Overwriting instead.",
                        self.output_path,
                        e,
                    )

            try:
                self._writer = pq.ParquetWriter(self.output_path, self.SCHEMA, compression="snappy")
                if existing_table is not None:
                    self._writer.write_table(existing_table)
            except Exception as e:
                _log.error("Failed to initialize ParquetWriter at %s: %s", self.output_path, e)
                # Fallback to load-merge-save if ParquetWriter fails
                if existing_table is not None:
                    table = pa.concat_tables([existing_table, table])
                pq.write_table(table, self.output_path, compression="snappy")
                return

        try:
            self._writer.write_table(table)
            _log.info("Appended batch of %d records directly to %s", len(records), self.output_path)
        except Exception as e:
            _log.error("Failed to stream write table chunk to ParquetWriter: %s", e)
            # Revert writer to None to retry initialization on next batch
            self._writer = None

    def close(self) -> None:
        """Finalizes and closes the active ParquetWriter stream."""
        if self._writer is not None:
            try:
                self._writer.close()
                _log.info("Closed ParquetWriter stream for %s", self.output_path)
            except Exception as e:
                _log.error("Error closing ParquetWriter: %s", e)
            finally:
                self._writer = None
