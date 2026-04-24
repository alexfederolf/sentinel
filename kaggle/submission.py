"""
Build and validate Kaggle submission parquets.

The ESA Kaggle submission schema is:
    - two columns: ``id`` (int64), ``is_anomaly`` (int8 / uint8, 0 or 1)
    - row-aligned to ``sample_submission.parquet`` (same length, same id order)

Two layers:
    * build_submission       — high-level: model + val + test → tuned submission
    * make_submission        — low-level writer with schema validation
    * submission_summary     — quick stats for a submission DataFrame
    * default_submission_path — path helper
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from sentinel.ml_logic.data import RAW_DIR, SUBMISSIONS_DIR
from sentinel.ml_logic.metrics import esa_metric
from sentinel.ml_logic.scorer import score_windows
from sentinel.ml_logic.thresholds import tune_threshold
from sentinel.params import WINDOW_SIZE


# ── high-level: model + data in, tuned submission out ────────────────────────
def build_submission(
    model,
    scaler,
    features: list[str],
    X_val: Union[np.ndarray, pd.DataFrame],
    y_val: np.ndarray,
    X_test: Union[np.ndarray, pd.DataFrame],
    test_ids: Optional[np.ndarray] = None,
    out_path: Optional[Union[str, Path]] = None,
    win: int = WINDOW_SIZE,
    topk: Optional[int] = None,
    n_sweep: int = 60,
    sample_submission_path: Optional[Union[str, Path]] = None,
    validate_schema: bool = True,
) -> dict:
    """
    Tune threshold on (X_val, y_val) via ESA metric, score X_test, write parquet.

    Parameters
    ----------
    model            : fitted reconstruction model (PCA or Keras)
    scaler           : fitted sklearn scaler
    features         : channel names to select (order matters for ndarray input)
    X_val, y_val     : validation rows + 0/1 labels — used for threshold tuning
    X_test           : test rows to score (DataFrame or ndarray)
    test_ids         : ids for the submission. If None and X_test is a DataFrame
                       with an ``id`` column, that column is used; otherwise
                       ``np.arange(len(X_test))``.
    out_path         : where to write the parquet. If None, no file is written.
    win              : window size for scoring
    topk             : forward to ``score_windows`` (LSTM/CNN top-k channels)
    n_sweep          : number of thresholds in the log-spaced sweep
    sample_submission_path, validate_schema : forwarded to ``make_submission``.

    Returns
    -------
    dict:
        threshold        : float — tuned threshold
        esa_score        : float — best ESA (corrected_event_f05) score on val
        sweep_thresholds : (n_sweep,) array
        sweep_scores     : (n_sweep,) array
        val_scores       : (n_val,) row-level scores on val
        test_scores      : (n_test,) row-level scores on test
        submission       : pd.DataFrame — the written submission (id, is_anomaly)
        summary          : dict from ``submission_summary``
    """
    features = list(features)

    # 1. Score validation rows ------------------------------------------------
    X_val_scaled  = _prep(scaler, features, X_val)
    val_scores    = score_windows(model, X_val_scaled, win=win, topk=topk)

    # 2. Tune threshold on ESA metric ----------------------------------------
    tuned = tune_threshold(
        scores    = val_scores,
        y_true    = np.asarray(y_val, dtype=np.int8),
        metric_fn = esa_metric,
        n_sweep   = n_sweep,
        score_key = "f_score",
    )
    threshold = tuned["threshold"]

    # 3. Score test rows, apply threshold ------------------------------------
    X_test_scaled = _prep(scaler, features, X_test)
    test_scores   = score_windows(model, X_test_scaled, win=win, topk=topk)
    labels        = (test_scores > threshold).astype(np.uint8)

    # 4. Resolve ids ---------------------------------------------------------
    if test_ids is None:
        if isinstance(X_test, pd.DataFrame) and "id" in X_test.columns:
            test_ids = X_test["id"].values
        else:
            test_ids = np.arange(len(labels), dtype=np.int64)
    test_ids = np.asarray(test_ids)

    # 5. Assemble + (optionally) write --------------------------------------
    if out_path is not None:
        submission = make_submission(
            predictions            = labels,
            test_ids               = test_ids,
            out_path               = out_path,
            sample_submission_path = sample_submission_path,
            validate_schema        = validate_schema,
        )
    else:
        submission = pd.DataFrame({"id": test_ids, "is_anomaly": labels})

    return {
        "threshold"       : threshold,
        "esa_score"       : tuned["score"],
        "sweep_thresholds": tuned["sweep_thresholds"],
        "sweep_scores"    : tuned["sweep_scores"],
        "val_scores"      : val_scores,
        "test_scores"     : test_scores,
        "submission"      : submission,
        "summary"         : submission_summary(submission),
    }


def _prep(scaler, features: list[str], X) -> np.ndarray:
    """Pick feature columns (if DataFrame), cast to float32, scale."""
    if isinstance(X, pd.DataFrame):
        arr = X[features].values
    else:
        arr = np.asarray(X)
    arr = arr.astype(np.float32, copy=False)
    return scaler.transform(arr).astype(np.float32)


# ── low-level: write + validate ──────────────────────────────────────────────
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
