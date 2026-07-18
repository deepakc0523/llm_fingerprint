"""
src.dataset.utils
=================
Shared utility functions used across the entire Stage 1 pipeline.

Responsibilities
----------------
- Structured logging (console + rotating file handler).
- tqdm progress-bar factory.
- Deterministic random seeding (random, numpy, torch when available).
- YAML configuration loading with lightweight schema validation.
- Apache Parquet save / load helpers via PyArrow.
- Directory creation helper.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

T = TypeVar("T")

# Module-level logger (used only inside utils itself).
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 20 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Return a named logger with a console handler and an optional rotating file
    handler.  If a logger with *name* has already been configured, the existing
    instance is returned unchanged (safe to call multiple times).

    Parameters
    ----------
    name:
        Dot-separated logger name, e.g. ``"fingerprint.dataset.builder"``.
    level:
        String log level understood by :func:`logging.getLevelName`, e.g.
        ``"INFO"``, ``"DEBUG"``, ``"WARNING"``.
    log_file:
        Optional path to a log file.  The parent directory is created
        automatically.  When *None*, only the console handler is attached.
    max_bytes:
        Maximum size of each log file before rotation (default 20 MiB).
    backup_count:
        Number of rotated backup files to keep (default 5).

    Returns
    -------
    logging.Logger
        A fully configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (stdout). On Windows the default encoding is cp1252 which
    # cannot represent Unicode separators; reconfigure to UTF-8 if possible.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass  # read-only stream (e.g. pytest capture) — skip silently.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # Optional rotating file handler.
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoids duplicate output).
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------

def get_progress_bar(
    iterable: Iterable[T],
    desc: str = "",
    total: Optional[int] = None,
    unit: str = "doc",
    leave: bool = True,
) -> tqdm:
    """
    Wrap an iterable with a tqdm progress bar.

    Parameters
    ----------
    iterable:
        Any Python iterable to track.
    desc:
        Short description shown to the left of the bar.
    total:
        Optional total element count used to display completion percentage.
        For streaming datasets where the length is unknown, pass ``None``.
    unit:
        The label for each progress unit (default ``"doc"``).
    leave:
        Whether to leave the progress bar visible after completion.

    Returns
    -------
    tqdm
        A tqdm instance wrapping *iterable*.
    """
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave, dynamic_ncols=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_random_seed(
    random_seed: int,
    python_seed: Optional[int] = None,
    numpy_seed: Optional[int] = None,
) -> None:
    """
    Seed all relevant random number generators to ensure deterministic behaviour.

    Generators seeded
    -----------------
    - Python built-in :mod:`random` (uses python_seed)
    - NumPy (:func:`numpy.random.seed`) (uses numpy_seed)
    - PyTorch (:func:`torch.manual_seed`) — only when torch is importable (uses random_seed).

    Parameters
    ----------
    random_seed:
        Global/PyTorch seed value.
    python_seed:
        Seed for standard random library.
    numpy_seed:
        Seed for NumPy.
    """
    p_seed = python_seed if python_seed is not None else random_seed
    n_seed = numpy_seed if numpy_seed is not None else random_seed
    random.seed(p_seed)
    np.random.seed(n_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    try:
        import torch  # type: ignore

        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
    except ImportError:
        pass  # torch not installed — silently skip.

    _log.debug("Random seeds set: random_seed=%d, python_seed=%d, numpy_seed=%d", random_seed, p_seed, n_seed)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a nested Python dictionary.

    The file must contain a top-level mapping.  An empty file returns ``{}``.

    Parameters
    ----------
    path:
        Absolute or relative path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    ValueError
        If the YAML file does not parse to a mapping (dict).
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping at the top level of {path!r}, got {type(data).__name__}."
        )

    return data


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> Path:
    """
    Create *path* and all intermediate parent directories if they do not exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The resolved :class:`pathlib.Path` object for *path*.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(
    df: pd.DataFrame,
    path: str,
    compression: str = "snappy",
    row_group_size: Optional[int] = None,
) -> None:
    """
    Persist a :class:`pandas.DataFrame` to a Parquet file using PyArrow.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    df:
        DataFrame to serialise.
    path:
        Destination file path (should end in ``.parquet``).
    compression:
        PyArrow compression codec (e.g., "zstd", "snappy").
    row_group_size:
        Number of rows per row group.
    """
    ensure_dir(os.path.dirname(path))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression=compression, row_group_size=row_group_size)
    _log.debug("Saved %d rows -> %s (compression=%s, row_group_size=%s)", len(df), path, compression, row_group_size)


def load_parquet(path: str) -> pd.DataFrame:
    """
    Read an Apache Parquet file into a :class:`pandas.DataFrame`.

    Parameters
    ----------
    path:
        Path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Contents of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *path*.
    """
    if not Path(path).is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pq.read_table(path).to_pandas()
