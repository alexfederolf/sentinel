"""
Threshold tuning for reconstruction-based anomaly detection.

Sweeps a log-spaced range of candidate thresholds, evaluates a user-supplied
event-wise metric on each, and returns the argmax. The metric is a
first-class argument — pass ``event_f05`` for the primary bootcamp metric,
``esa_metric`` to reproduce Kaggle-leaderboard ranking, or any other metric
that returns a dict with an ``f_score`` key (or a bare scalar via
``score_key=None``).

Defaults are chosen to match pca_full.py: 60 log-spaced thresholds between
the 50th percentile of *nominal* scores and the 99th percentile of
*anomaly* scores. Log spacing keeps the grid dense in the transition region
where MSE distributions differ the most.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def _percentile_of_class(scores: np.ndarray, y_true: np.ndarray,
                         class_id: int, percentile: float) -> float:
    """Percentile of ``scores`` restricted to rows where y_true == class_id."""
    pool = scores[y_true == class_id]
    if len(pool) == 0:
        pool = scores
    return float(np.percentile(pool, percentile))


def tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    metric_fn: Optional[Callable] = None,
    n_sweep: int = 60,
    lo_percentile: tuple[int, float] = (0, 50.0),
    hi_percentile: tuple[int, float] = (1, 99.0),
    score_key: Optional[str] = "f_score",
) -> dict:
    """
    Sweep ``n_sweep`` log-spaced thresholds and pick the argmax of ``metric_fn``.

    Parameters
    ----------
    scores       : float array, per-row anomaly scores
    y_true       : 0/1 int array of the same length
    metric_fn    : callable(y_true, y_pred) → dict or float. Default:
                   ``sentinel.ml_logic.metrics.event_f05``.
    n_sweep      : number of candidate thresholds
    lo_percentile: (class_id, percentile) — lower bound. Default: 50th
                   percentile of nominal scores.
    hi_percentile: (class_id, percentile) — upper bound. Default: 99th
                   percentile of anomaly scores (guards against outlier-driven
                   range explosion, cf. pca_full.tune_threshold docstring).
    score_key    : key extracted from the metric's return dict. Pass None if
                   ``metric_fn`` returns a bare scalar.

    Returns
    -------
    dict:
        threshold        : float — best threshold
        score            : float — best metric value
        sweep_thresholds : (n_sweep,) array
        sweep_scores     : (n_sweep,) array
    """
    # Lazy import so the metrics module can be refactored without circular deps.
    if metric_fn is None:
        from .metrics import event_f05
        metric_fn = event_f05

    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int8)

    lo = _percentile_of_class(scores, y_true, *lo_percentile)
    hi = _percentile_of_class(scores, y_true, *hi_percentile)

    # Guard against degenerate ranges (e.g. when anomaly class is empty or
    # scores collapse to a constant). np.geomspace requires lo > 0 and hi > lo.
    lo = max(lo, float(scores.min()), 1e-9)
    if hi <= lo:
        hi = lo * 10.0

    thresholds = np.geomspace(lo, hi, n_sweep)
    sweep = np.empty(n_sweep, dtype=np.float32)
    for i, t in enumerate(thresholds):
        pred = (scores > t).astype(np.int8)
        out  = metric_fn(y_true, pred)
        if score_key is None:
            sweep[i] = float(out)
        else:
            sweep[i] = float(out[score_key])

    best = int(np.argmax(sweep))
    return {
        "threshold"       : float(thresholds[best]),
        "score"           : float(sweep[best]),
        "sweep_thresholds": thresholds,
        "sweep_scores"    : sweep,
    }
