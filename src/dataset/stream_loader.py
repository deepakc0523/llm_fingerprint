"""
src.dataset.stream_loader
=========================
Streaming document loader for the five Fingerprint source corpora.

Each :class:`StreamLoader` instance handles **one dataset at a time**.  It
delegates to HuggingFace ``datasets`` with ``streaming=True`` wherever the
dataset supports it, and automatically normalises the raw dict returned by each
source into a canonical ``{"text": str, "document_id": str, "source_split": str}``
record.

Checkpoint / resume
-------------------
Before iterating the stream a caller may provide the ``last_document_index``
recovered from a JSON checkpoint file.  The loader will efficiently skip that
many documents via ``IterableDataset.skip()``.

Usage example
-------------
::

    from src.dataset.stream_loader import StreamLoader

    loader = StreamLoader(
        dataset_name="wikitext",
        dataset_cfg={
            "hf_repo": "wikitext",
            "subset": "wikitext-103-v1",
            "split": "train",
            "text_field": "text",
            "streaming": True,
        },
        logger=my_logger,
    )
    for doc in loader.load(resume_from_index=0):
        process(doc)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Optional

import datasets as hf_datasets

# ---------------------------------------------------------------------------
# Canonical document schema
# ---------------------------------------------------------------------------

# Every document yielded by StreamLoader has exactly these three keys.
#   text          – the raw text content of the document
#   document_id   – a string that uniquely identifies the document within its
#                   source (e.g. "wikitext_train_000042")
#   source_split  – the HuggingFace split name ("train", "validation", …)
CANONICAL_FIELDS = ("text", "document_id", "source_split")


class StreamLoader:
    """
    Streaming document source for a single HuggingFace dataset.

    Parameters
    ----------
    dataset_name:
        Short key used for logging and document-ID generation, e.g.
        ``"wikitext"``.
    dataset_cfg:
        The per-dataset sub-dictionary from ``dataset_engineering.yaml``.
        Required keys: ``hf_repo``, ``split``, ``text_field``.
        Optional keys: ``subset`` (HF config name), ``streaming``.
    logger:
        A :class:`logging.Logger` instance.  When *None*, a default logger
        named ``"fingerprint.stream_loader"`` is created.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_cfg: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dataset_name: str = dataset_name
        self.hf_repo: str = dataset_cfg["hf_repo"]
        self.subset: Optional[str] = dataset_cfg.get("subset")
        self.split: str = dataset_cfg.get("split", "train")
        self.text_field: str = dataset_cfg["text_field"]
        self.use_streaming: bool = dataset_cfg.get("streaming", True)
        self.log: logging.Logger = logger or logging.getLogger("fingerprint.stream_loader")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(
        self, resume_from_index: int = 0
    ) -> Generator[Dict[str, str], None, None]:
        """
        Yield canonical document dicts, optionally resuming from a checkpoint.

        Each yielded record has exactly the keys defined in
        :data:`CANONICAL_FIELDS`:

        - ``"text"`` — raw document text (str)
        - ``"document_id"`` — unique identifier string
        - ``"source_split"`` — the HuggingFace split name

        Parameters
        ----------
        resume_from_index:
            Number of documents to skip at the start of the stream.  Pass
            ``0`` (default) to start from the beginning.  When using
            streaming mode the skip is performed via
            :meth:`datasets.IterableDataset.skip`, which is O(n) but does
            not download skipped shards.

        Yields
        ------
        Dict[str, str]
            Canonical document record.

        Notes
        -----
        Documents where the designated text field is missing, ``None``, or
        resolves to an empty string after stripping are silently dropped;
        the index counter still advances so that checkpoints remain valid.
        """
        self.log.info(
            "Loading '%s' from HuggingFace repo '%s' (subset=%s, split=%s, streaming=%s)",
            self.dataset_name,
            self.hf_repo,
            self.subset,
            self.split,
            self.use_streaming,
        )

        dataset = self._load_hf_dataset()

        if resume_from_index > 0:
            self.log.info(
                "Resuming '%s' from document index %d — skipping ahead.",
                self.dataset_name,
                resume_from_index,
            )
            dataset = dataset.skip(resume_from_index)

        global_index = resume_from_index
        for raw_record in dataset:
            text = self._extract_text(raw_record)
            global_index += 1
            if not text:
                continue
            doc_id = f"{self.dataset_name}_{self.split}_{global_index:08d}"
            yield {
                "text": text,
                "document_id": doc_id,
                "source_split": self.split,
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_hf_dataset(self) -> hf_datasets.IterableDataset:
        """
        Call ``datasets.load_dataset`` and return an :class:`IterableDataset`.

        When ``streaming=True`` the HuggingFace library returns an
        ``IterableDataset`` directly.  When ``streaming=False`` the full
        dataset is downloaded first and then converted to an
        ``IterableDataset`` so the rest of the code can use the same
        interface regardless.

        Returns
        -------
        datasets.IterableDataset
            A streaming-compatible dataset object.
        """
        load_kwargs: Dict[str, Any] = {
            "path": self.hf_repo,
            "split": self.split,
            "streaming": self.use_streaming,
        }
        if self.subset:
            load_kwargs["name"] = self.subset

        try:
            ds = hf_datasets.load_dataset(**load_kwargs)
        except Exception as exc:
            self.log.warning(
                "Streaming load of '%s' failed (%s).  Retrying without streaming.",
                self.dataset_name,
                exc,
            )
            load_kwargs["streaming"] = False
            ds = hf_datasets.load_dataset(**load_kwargs)

        # Normalise non-streaming datasets to IterableDataset.
        if isinstance(ds, hf_datasets.Dataset):
            ds = ds.to_iterable_dataset()

        return ds

    def _extract_text(self, record: Dict[str, Any]) -> str:
        """
        Pull the target text field out of a raw HuggingFace record.

        The method handles nested fields for datasets like StackExchange
        where the primary text may be inside a list or dict.  It returns
        an empty string when the field is absent or evaluates to nothing.

        Parameters
        ----------
        record:
            Raw dict returned by iterating a HuggingFace dataset.

        Returns
        -------
        str
            Extracted text, stripped of leading/trailing whitespace.
        """
        value = record.get(self.text_field, "")

        # StackExchange stores answers as a list of dicts; take the first.
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                value = value[0].get("text", value[0].get("body", ""))
            elif value:
                value = str(value[0])
            else:
                return ""

        if isinstance(value, dict):
            # Fall back to any string-valued key.
            for candidate in ("text", "body", "content", "article", "document"):
                if candidate in value and isinstance(value[candidate], str):
                    value = value[candidate]
                    break
            else:
                value = str(value)

        return str(value).strip()
