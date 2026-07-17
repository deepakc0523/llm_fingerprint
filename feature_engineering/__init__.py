"""
Feature engineering package for extracting stylometric vectors and dense neural embeddings.
"""

from .stylometry import StylometricExtractor
from .embeddings import TransformerEmbedder

__all__ = ["StylometricExtractor", "TransformerEmbedder"]
