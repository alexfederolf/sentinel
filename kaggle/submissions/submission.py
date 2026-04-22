"""
Build and validate Kaggle submission parquets.

The ESA Kaggle submission schema is:
    - two columns: ``id`` (int64), ``is_anomaly`` (int8 / uint8, 0 or 1)
    - row-aligned to ``sample_submission.parquet`` (same length, same id order)

``make_submission`` enforces this schema and optionally validates id alignment
against the competition's ``sample_submission.parquet``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from ...src.sentinel.ml_logic.data import RAW_DIR, SUBMISSIONS_DIR


def make_submission(
    predictions: np.ndarray,
    test_ids: np.ndarray,
    out_path: Union[str, Path],
    sample_submission_path: Optional[Union[str, Path]] = None,
    validate_schema: bool = True,
) -> pd.DataFrame:
    """
    Assemble, validate, and persist a Kaggle submission parquet.

    Parameters
    ----------
    predictions            : 0/1 int array of length ``len(test_ids)``
    test_ids               : int array of Kaggle test IDs
    out_path               : destination path (directory will be created)
    sample_submission_path : optional path to ``sample_submission.parquet``. If
                             not given, falls back to ``data/raw/sample_submission.parquet``
                             when ``validate_schema=True``. Pass ``None`` and
                             ``validate_schema=False`` for internal-only
                             submissions (e.g. test_intern predictions).
    validate_schema        : when True (default), checks dtype, shape and id
                             order against the sample submission.

    Returns
    -------
    pd.DataFrame — the written submission.
    """
    predictions = np.asarray(predictions).astype(np.uint8)
    test_ids    = np.asarray(test_ids)

    if len(predictions) != len(test_ids):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} vs test_ids={len(test_ids)}"
        )

    unique = np.unique(predictions)
    if not set(unique.tolist()).issubset({0, 1}):
        raise ValueError(f"predictions must be binary 0/1, got values {unique}")

    df = pd.DataFrame({"id": test_ids, "is_anomaly": predictions})

    if validate_schema:
        if sample_submission_path is None:
            sample_submission_path = RAW_DIR / "sample_submission.parquet"
        sample_submission_path = Path(sample_submission_path)
        if not sample_submission_path.exists():
            raise FileNotFoundError(
                f"sample submission not found at {sample_submission_path}; "
                f"pass an explicit path or set validate_schema=False"
            )
        sample = pd.read_parquet(sample_submission_path)
        if len(df) != len(sample):
            raise ValueError(
                f"row count mismatch vs sample: {len(df)} vs {len(sample)}"
            )
        if not (df["id"].values == sample["id"].values).all():
            raise ValueError("id order mismatch vs sample_submission.parquet")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def submission_summary(df: pd.DataFrame) -> dict:
    """
    Quick stats for a submission DataFrame — positive rate, segment count,
    row count. Useful for notebook print-out and sanity checks.
    """
    y = np.asarray(df["is_anomaly"].values, dtype=np.uint8)
    n = len(y)
    pos_rate = float(y.mean()) if n > 0 else 0.0
    # Contiguous positive segments
    padded = np.concatenate([[0], y, [0]])
    segs = int((np.diff(padded) == 1).sum())
    return {
        "rows"         : n,
        "positive_rows": int(y.sum()),
        "positive_rate": round(pos_rate, 6),
        "n_segments"   : segs,
    }


# Convenience default path so callers can just pass a filename stem.
def default_submission_path(stem: str) -> Path:
    """Return ``submissions/<stem>.parquet``."""
    return SUBMISSIONS_DIR / f"{stem}.parquet"
