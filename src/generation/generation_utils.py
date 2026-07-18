"""
src.generation.generation_utils
==============================
General helpers and metric-tracking classes for the LLM generation pipeline.
"""

from __future__ import annotations

import os
import random
import time
from typing import Generator, Iterable, List, TypeVar

import numpy as np

T = TypeVar("T")


def set_seed(seed: int) -> None:
    """
    Ensure reproducibility by seeding random, numpy, and torch.

    Parameters
    ----------
    seed:
        The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def batch_iterator(iterable: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """
    Chunks an iterable into batches of a given size.

    Parameters
    ----------
    iterable:
        The sequence or iterable of items.
    batch_size:
        The maximum number of items per batch.

    Yields
    ------
    List[T]
        A batch of items.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class Timer:
    """Simple context manager to measure execution time."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self.start_time
