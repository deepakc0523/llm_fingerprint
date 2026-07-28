"""Sentence Embedding Feature Extractor.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Feature Engineering
Description:
    Generates dense sentence-level embedding vectors using the
    sentence-transformers library.  The default model is
    ``all-MiniLM-L6-v2`` which produces 384-dimensional embeddings.

    Embeddings are computed in configurable batches to support large
    corpora without excessive memory usage.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Encodes text documents into dense sentence embedding vectors.

    Wraps the ``sentence_transformers.SentenceTransformer`` model and
    provides a consistent interface with the other feature extractors
    in this package.

    Attributes:
        cfg: Embeddings sub-configuration dict from feature_engineering.yaml.
        model_name: Identifier of the sentence-transformer model to load.
        batch_size: Number of documents per encoding batch.
        max_seq_length: Maximum token sequence length.
        normalize: Whether to L2-normalise the output embeddings.
        device: Torch device string ("cpu" or "cuda").
        _model: Loaded SentenceTransformer instance (None until first use).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        """Initialise with embedding configuration dict.

        Args:
            cfg: The ``embeddings`` section from feature_engineering.yaml.
        """
        self.cfg = cfg
        self.model_name: str = cfg.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.batch_size: int = int(cfg.get("batch_size", 64))
        self.max_seq_length: int = int(cfg.get("max_seq_length", 256))
        self.normalize: bool = bool(cfg.get("normalize_embeddings", True))
        self.device: str = cfg.get("device", "cpu")
        self._model: Optional[Any] = None   # SentenceTransformer loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Encode all documents and return the embedding matrix.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            Dense float32 numpy array of shape (n_samples, embedding_dim).
        """
        logger.info(
            "Encoding %d documents with '%s' (batch_size=%d, device=%s) ...",
            len(texts),
            self.model_name,
            self.batch_size,
            self.device,
        )
        model = self._load_model()
        embeddings = model.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        embeddings = embeddings.astype(np.float32)
        logger.info("Embedding matrix shape: %s", embeddings.shape)
        return embeddings

    def transform(self, texts: pd.Series) -> np.ndarray:
        """Encode documents using the already-loaded model.

        Args:
            texts: 1-D pandas Series of text documents.

        Returns:
            Dense float32 numpy array of shape (n_samples, embedding_dim).
        """
        return self.fit_transform(texts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_matrix(self, matrix: np.ndarray, out_path: Path) -> None:
        """Persist embedding matrix as a compressed .npz file.

        Args:
            matrix: Dense numpy float32 array.
            out_path: Output path (e.g. data/features/embedding/emb_fingerprint.npz).
        """
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(out_path), embeddings=matrix)
        logger.info("Embeddings saved → %s  %s", out_path, matrix.shape)

    @staticmethod
    def load_matrix(npz_path: Path) -> np.ndarray:
        """Load embedding matrix from a compressed .npz file.

        Args:
            npz_path: Path to .npz file produced by ``save_matrix``.

        Returns:
            Dense float32 numpy array.
        """
        data = np.load(str(npz_path))
        embeddings = data["embeddings"]
        logger.info("Embeddings loaded from %s  %s", npz_path, embeddings.shape)
        return embeddings

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self, matrix: np.ndarray) -> Dict[str, Any]:
        """Return summary statistics for the embedding matrix.

        Args:
            matrix: Dense numpy array (n_samples × embedding_dim).

        Returns:
            Dict with shape, norm stats, and model metadata.
        """
        norms = np.linalg.norm(matrix, axis=1)
        return {
            "shape": matrix.shape,
            "embedding_dim": matrix.shape[1],
            "n_samples": matrix.shape[0],
            "model_name": self.model_name,
            "normalized": self.normalize,
            "norm_mean": float(np.mean(norms)),
            "norm_std": float(np.std(norms)),
            "norm_min": float(np.min(norms)),
            "norm_max": float(np.max(norms)),
            "global_mean": float(np.mean(matrix)),
            "global_std": float(np.std(matrix)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Lazily load the SentenceTransformer model.

        Returns:
            Loaded SentenceTransformer instance.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            logger.info("Loading SentenceTransformer: %s ...", self.model_name)
            self._model = SentenceTransformer(
                self.model_name, device=self.device
            )
            self._model.max_seq_length = self.max_seq_length
            logger.info(
                "Model loaded. Embedding dimension: %d",
                self._model.get_sentence_embedding_dimension(),
            )
        return self._model
