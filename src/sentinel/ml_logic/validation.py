"""
Validation infrastructure for the bootcamp pipeline.

Three concerns live here:

    1. ``time_block_cv_splits`` — event-aware temporal K-fold that mirrors the
       Kaggle public/private split structure. Each fold is a contiguous time
       block so future-leakage is impossible and evaluation captures
       temporal-drift behaviour.

    2. ``bootstrap_f05_ci`` — bootstrap confidence interval for event-level
       F0.5 (or any supplied metric). Uses event-block resampling by default
       because row-level resampling breaks event structure.

    3. ``replay_submission`` — load a submission parquet and score it
       against a row-level label array. The "local private leaderboard"
       workflow: replay past submissions against ``test_intern`` to check
       that local ranking correlates with Kaggle public. Length-aware — the
       old Kaggle parquets (521,280 rows) cannot be replayed against
       test_intern (different ID space), so the function raises rather than
       silently misalign.

This module is lazy about the metrics import so that ``metrics`` can be
further refactored without creating a circular dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from .data import find_anomaly_segments


# ══════════════════════════════════════════════════════════════════════════════
# Event-aware time-block CV
# ══════════════════════════════════════════════════════════════════════════════

def time_block_cv_splits(
    y_labels: np.ndarray,
    n_folds: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return ``n_folds`` (train_idx, val_idx) tuples, each val fold a contiguous
    temporal block. Fold boundaries are snapped forward to nominal rows so no
    anomaly event is sliced across fold boundaries.

    Parameters
    ----------
    y_labels : 0/1 int array
    n_folds  : number of folds

    Returns
    -------
    list of (train_idx_array, val_idx_array) — both are int arrays of row indices.
    """
    y = np.asarray(y_labels, dtype=np.int8)
    N = len(y)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    block = N // n_folds
    splits = []
    for i in range(n_folds):
        val_start = i * block
        val_end   = (i + 1) * block if i < n_folds - 1 else N
        while val_start < val_end and y[val_start] == 1:
            val_start += 1
        # Snap val_end forward into nominal if it lands inside an event
        while val_end < N and y[val_end - 1] == 1:
            val_end += 1
        val_end = min(val_end, N)
        val_idx   = np.arange(val_start, val_end)
        train_idx = np.concatenate(
            [np.arange(0, val_start), np.arange(val_end, N)]
        )
        splits.append((train_idx, val_idx))
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ══════════════════════════════════════════════════════════════════════════════

def bootstrap_f05_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Optional[Callable] = None,
    score_key: str = "f_score",
    n_boot: int = 200,
    event_block: bool = True,
    seed: int = 42,
) -> dict:
    """
    Bootstrap a 95 % confidence interval for a metric.

    With ``event_block=True`` (default) the bootstrap resamples whole events
    and keeps (truth, prediction) aligned inside each event. Concretely:

    * Each iteration draws ``len(segs)`` event indices with replacement.
    * Events in the sample are marked in a fresh ``y_boot`` (duplicates
      collapse to "present"), and predictions over their original row spans
      are copied from ``y_pred``.
    * Events **not** in the sample are zeroed out in BOTH ``y_boot`` and the
      bootstrapped prediction. This is essential: if you only resampled
      ``y_true`` and kept the full ``y_pred``, predictions inside dropped
      events would show up as spurious FPs and bias the CI downward. Zeroing
      predictions over dropped event regions gives the CI the intended
      interpretation — "metric variance under event-presence sampling,
      holding the prediction model fixed."

    Row-level bootstrap (``event_block=False``) resamples rows with
    replacement. It destroys event structure and is included only as a
    baseline — do not use it for event-wise metrics.

    Parameters
    ----------
    y_true      : 0/1 label array
    y_pred      : 0/1 prediction array, same length
    metric_fn   : callable(y_true, y_pred) → dict. Default: ``event_f05``.
    score_key   : key extracted from the metric dict (default ``f_score``)
    n_boot      : number of bootstrap iterations
    event_block : if True, resample events; if False, resample rows
    seed        : RNG seed for reproducibility

    Returns
    -------
    dict: mean, std, ci_lo_95, ci_hi_95, all_scores (length n_boot)
    """
    if metric_fn is None:
        from .metrics import event_f05
        metric_fn = event_f05

    rng    = np.random.default_rng(seed)
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    N      = len(y_true)

    scores = np.empty(n_boot, dtype=np.float32)

    if event_block:
        segs = find_anomaly_segments(y_true)
        if len(segs) == 0:
            # No events to bootstrap → return the single deterministic score.
            s = float(metric_fn(y_true, y_pred)[score_key])
            return {
                "mean": s, "std": 0.0, "ci_lo_95": s, "ci_hi_95": s,
                "all_scores": np.full(n_boot, s, dtype=np.float32),
            }
        event_idx = np.arange(len(segs))
        for b in range(n_boot):
            sampled = rng.choice(event_idx, size=len(event_idx), replace=True)
            kept    = set(int(i) for i in sampled)
            y_boot  = np.zeros(N, dtype=np.int8)
            p_boot  = y_pred.copy()
            # Drop predictions inside events that weren't sampled — otherwise
            # a fixed y_pred would accumulate bogus FPs in those regions.
            for i, seg in enumerate(segs):
                if i not in kept:
                    p_boot[seg["start"]:seg["end"] + 1] = 0
            # Mark kept events in the resampled truth.
            for i in kept:
                y_boot[segs[i]["start"]:segs[i]["end"] + 1] = 1
            scores[b] = float(metric_fn(y_boot, p_boot)[score_key])
    else:
        for b in range(n_boot):
            idx = rng.integers(0, N, size=N)
            scores[b] = float(metric_fn(y_true[idx], y_pred[idx])[score_key])

    return {
        "mean"      : float(scores.mean()),
        "std"       : float(scores.std()),
        "ci_lo_95"  : float(np.percentile(scores, 2.5)),
        "ci_hi_95"  : float(np.percentile(scores, 97.5)),
        "all_scores": scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Replay past submissions against a local holdout
# ══════════════════════════════════════════════════════════════════════════════

def replay_submission(
    path: Union[str, Path],
    y_true_row: np.ndarray,
    metric_fn: Optional[Callable] = None,
    id_col: str = "id",
    pred_col: str = "is_anomaly",
    assert_length: bool = True,
) -> dict:
    """
    Score a stored submission parquet against a row-level label array.

    Intended for the local-private-leaderboard workflow: produce a submission
    for ``test_intern`` from a notebook (11/12/13), save it to
    ``submissions/``, then replay it here against ``y_test_intern`` to pick up
    a trustworthy local score — one that doesn't depend on Kaggle.

    The function deliberately refuses to silently misalign: if the parquet
    length doesn't match ``y_true_row``, it raises. The old Kaggle parquets
    (521,280 rows keyed to the competition test set) are **not** replayable
    against ``test_intern`` — different ID spaces, different row counts.

    Parameters
    ----------
    path         : path to the submission parquet
    y_true_row   : 0/1 int array of ground-truth row labels
    metric_fn    : callable(y_true, y_pred) → dict. Default: ``event_f05``.
    id_col       : name of the id column in the parquet (default ``id``)
    pred_col     : name of the prediction column (default ``is_anomaly``)
    assert_length: raise if length mismatch (True) or truncate (False)

    Returns
    -------
    The metric's return dict, plus a ``source`` key with the parquet path.
    """
    if metric_fn is None:
        from .metrics import event_f05
        metric_fn = event_f05

    path = Path(path)
    df = pd.read_parquet(path)
    if pred_col not in df.columns:
        raise KeyError(f"{pred_col!r} not in parquet columns {list(df.columns)}")

    preds      = df[pred_col].values.astype(np.int8)
    y_true_row = np.asarray(y_true_row, dtype=np.int8)

    if len(preds) != len(y_true_row):
        if assert_length:
            raise ValueError(
                f"length mismatch: submission {path.name} has {len(preds):,} "
                f"rows, y_true has {len(y_true_row):,}. "
                f"Kaggle submissions (Kaggle test.parquet IDs) are not replayable "
                f"against test_intern — different ID spaces."
            )
        n = min(len(preds), len(y_true_row))
        preds = preds[:n]
        y_true_row = y_true_row[:n]

    result = dict(metric_fn(y_true_row, preds))
    result["source"] = str(path)
    return result
