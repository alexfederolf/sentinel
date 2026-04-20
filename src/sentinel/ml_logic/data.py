"""
Data-loading utilities for the SENTINEL anomaly-detection project.

Provides functions to load the raw competition parquets and the
ground-truth label array, plus helpers to parse contiguous anomaly
segments and to enumerate column groups.

All paths default to ``<repo_root>/data/raw/`` resolved relative to
this file so the package works after ``pip install -e .`` from any
working directory.
"""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_train(path: Path = RAW_DIR / "train.parquet") -> pd.DataFrame:
    """
    Load the training set (14.7 M rows × 89 columns).

    Parameters
    ----------
    path : Path, optional
        Override the default raw-data location.

    Returns
    -------
    pd.DataFrame
        All 76 channel columns, 11 telecommand columns, ``id``, and
        ``is_anomaly`` label.
    """
    return pd.read_parquet(path)


def load_test(path: Path = RAW_DIR / "test.parquet") -> pd.DataFrame:
    """
    Load the test set (521 K rows × 88 columns, no ``is_anomaly`` column).

    Parameters
    ----------
    path : Path, optional
        Override the default raw-data location.

    Returns
    -------
    pd.DataFrame
        Same structure as train minus the label column.
    """
    return pd.read_parquet(path)


def load_target_channels(path: Path = RAW_DIR / "target_channels.csv") -> list[str]:
    """
    Return the 58 scored channel names from the competition CSV.

    Parameters
    ----------
    path : Path, optional
        Override the default CSV location.

    Returns
    -------
    list[str]
        Channel names in the order they appear in ``target_channels.csv``.
    """
    return pd.read_csv(path)["target_channels"].tolist()


def find_anomaly_segments(labels: np.ndarray | pd.Series) -> list[dict]:
    """
    Identify contiguous anomaly segments in a binary label array.

    A segment is a maximal run of consecutive 1s.  The function is
    equivalent to run-length encoding restricted to the anomaly class.

    Parameters
    ----------
    labels : array-like of 0/1
        Ground-truth anomaly labels (1 = anomalous, 0 = nominal).

    Returns
    -------
    list[dict]
        Each dict has keys ``start``, ``end``, ``length`` (all ints,
        using the original index space of ``labels``).

    Examples
    --------
    >>> find_anomaly_segments(np.array([0, 1, 1, 0, 1, 0]))
    [{'start': 1, 'end': 2, 'length': 2}, {'start': 4, 'end': 4, 'length': 1}]
    """
    if isinstance(labels, pd.Series):
        labels = labels.values

    segments = []
    in_anomaly = False
    start = None

    for i, v in enumerate(labels):
        if v == 1 and not in_anomaly:
            start = i
            in_anomaly = True
        elif v == 0 and in_anomaly:
            segments.append({"start": start, "end": i - 1, "length": i - start})
            in_anomaly = False

    if in_anomaly:
        segments.append({"start": start, "end": len(labels) - 1, "length": len(labels) - start})

    return segments


def get_channel_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all ``channel_*`` column names in ascending numeric order.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    cols = [c for c in df.columns if c.startswith("channel_")]
    return sorted(cols, key=lambda c: int(c.split("_")[1]))


def get_telecommand_cols(df: pd.DataFrame) -> list[str]:
    """
    Return all ``telecommand_*`` column names.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list[str]
    """
    return [c for c in df.columns if c.startswith("telecommand_")]
