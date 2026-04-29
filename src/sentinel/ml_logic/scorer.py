"""
Generic window-mean MSE scoring for reconstruction anomaly-detection models.

The models all follow the same scoring recipe:

    1. Reshape row-level data into non-overlapping windows.
    2. Reconstruct each window with the model.
    3. MSE → reduce to one number per window.
    4. Broadcast each window score to its rows (the tail inherits the
       last full window's score).

Two model backends are supported via duck typing:

    * sklearn PCA   — ``model.transform`` / ``model.inverse_transform`` on
                      flattened windows (shape ``(n_win, win * n_feat)``)
    * Keras models  — ``model.predict`` on 3D tensors (shape
                      ``(n_win, win, n_feat)``)

Public API
----------
score_windows                    row-level anomaly scores (threshold on these)
window_scores_only               window-level scalar scores
broadcast_window_scores_to_rows  broadcast helper
score_report                     single-pass, everything-at-once
detrend_scores                   remove rolling baseline drift from a score array
"""
from __future__ import annotations

import numpy as np

from ..params import WINDOW_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# INTENAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _is_pca(model) -> bool:
    """sklearn-style PCA exposes transform + inverse_transform."""
    return hasattr(model, "inverse_transform") and hasattr(model, "transform")


def _is_keras(model) -> bool:
    """Keras models expose predict."""
    return hasattr(model, "predict")


def _reconstruct_windows(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Shape rows into windows and run them through the model.

    Returns
    -------
    X_win : (n_win, win, n_feat) float32 — the input windows
    X_hat : (n_win, win, n_feat) float32 — the reconstructions

    If fewer than one full window fits, returns two empty arrays of shape*
    ``(0, win, n_feat)`` and does NOT dispatch to the model.

    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    N, n_feat = X_rows.shape
    n_complete = N // win

    if n_complete == 0:
        empty = np.zeros((0, win, n_feat), dtype=np.float32)
        return empty, empty

    X_win = X_rows[:n_complete * win].reshape(n_complete, win, n_feat)

    # PCA -> inverse_transform() on flattened windows---------------------------
    if _is_pca(model):
        # flatten
        X_flat = X_win.reshape(n_complete, win * n_feat)
        # reconstruct and reshape back
        X_hat  = model.inverse_transform(model.transform(X_flat)).reshape(
            n_complete, win, n_feat
        )

    # Keras -> predict() in batches on 3D windows-------------------------------
    elif _is_keras(model):
        X_hat = model.predict(X_win, batch_size=batch, verbose=0)
        if X_hat.shape != X_win.shape:
            raise ValueError(
                f"Reconstruction shape {X_hat.shape} != input shape {X_win.shape}"
            )
    else:
        raise TypeError(
            f"Unsupported model type {type(model).__name__!r} — needs either "
            f"(transform + inverse_transform) or predict()"
        )

    # X_win & X_hat reconstructions
    return X_win, np.asarray(X_hat, dtype=np.float32)


def _window_scores_from_sq_err(
    sq_err: np.ndarray,
    topk: int | None,
) -> np.ndarray:
    """
    Reduce per-window squared errors to one MSE per window.

    Parameters
    ----------
    sq_err : (n_win, win, n_feat) float
    topk   : None → mean over (win, n_feat);
             k    → mean of the k largest per-channel MSEs

    Returns
    -------
    (n_win,) float32
    """
    if sq_err.size == 0:
        return np.zeros(0, dtype=np.float32)
    n_feat = sq_err.shape[2]

    # PCA default: mean over all channels and time steps
    if topk is None:
        return sq_err.mean(axis=(1, 2)).astype(np.float32)
    if not 1 <= topk <= n_feat:
        raise ValueError(f"topk must be in [1, {n_feat}], got {topk}")

    # topk in LSTM/CNN: only a few channels show high MSE, so take the mean of the top-k per-channel MSEs
    per_channel = sq_err.mean(axis=1)                        # (n_win, n_feat)
    vals = np.partition(per_channel, -topk, axis=1)[:, -topk:]
    return vals.mean(axis=1).astype(np.float32)


def broadcast_window_scores_to_rows(
    win_scores: np.ndarray,
    n_rows: int,
    win: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Repeat each window score ``win`` times to produce a row-level array.

    Tail rows (fewer than ``win`` at the end of the input) inherit the last
    full window's score rather than being zero-padded. This keeps the
    timeline continuous, but a short anomalous tail can be masked by a
    nominal last window — worth being aware of for downstream thresholding.
    """
    if len(win_scores) == 0:
        return np.zeros(n_rows, dtype=np.float32)
    # repeat
    row_scores = np.repeat(win_scores.astype(np.float32), win)
    if len(row_scores) < n_rows:
        pad = np.full(n_rows - len(row_scores),
                      win_scores[-1], dtype=np.float32)
        row_scores = np.concatenate([row_scores, pad])

    return row_scores[:n_rows]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN public scoring functions
# ──────────────────────────────────────────────────────────────────────────────

def score_windows(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
    topk: int | None = None,
) -> np.ndarray:
    """
    Row-level anomaly scores via window reconstruction MSE.

    Parameters
    ----------
    model   : fitted reconstruction model (PCA or Keras)
    X_rows  : float32 (n_rows, n_features)
    win     : window size
    batch   : Keras batch size
    topk    : if set, each window's score is the mean of itstopk largest
              per-channel MSE values instead of the mean over all channels.
              Useful when anomalies affect only a few channels (LSTM/CNN).
              Note: under topk the returned array is no longer a
              overall MSE — it is the mean of the top-k per-channel MSEs,
              broadcast to rows.

    Returns
    -------
    float32 (n_rows,) — per-row anomaly score (window statistic broadcast)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)

    # reconstraction
    X_win, X_hat = _reconstruct_windows(model, X_rows, win=win, batch=batch)
    if X_win.shape[0] == 0:
        return np.zeros(X_rows.shape[0], dtype=np.float32)

    # error
    sq_err = (X_win - X_hat) ** 2

    # calculates MSE pro window for all or topk chanels (LSTM/CNN)
    win_scores = _window_scores_from_sq_err(sq_err, topk=topk)

    # broadcast window scores to rows
    return broadcast_window_scores_to_rows(win_scores, X_rows.shape[0], win)

# ──────────────────────────────────────────────────────────────────────────────
# WIP: more detailed report for the showcase
# ──────────────────────────────────────────────────────────────────────────────

def window_scores_only(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
    topk: int | None = None,
) -> np.ndarray:
    """
    Like ``score_windows`` but returns the *window-level* score array
    (length ``n_rows // win``). Shares the ``topk`` option with
    ``score_windows`` so both views agree on what a window score means.
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    X_win, X_hat = _reconstruct_windows(model, X_rows, win=win, batch=batch)
    if X_win.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    sq_err = (X_win - X_hat) ** 2
    return _window_scores_from_sq_err(sq_err, topk=topk)


def score_report(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
    topk: int | None = None,
    n_top_channels: int | None = None,
) -> dict:
    """
    One-pass report: runs the reconstruction once and derives every statistic
    the showcase / frontend / multi-plot. Replaces the
    pattern of ``score_windows`` for labels plus a hand-rolled PCA
    loop for per-channel MSE.

    Parameters
    ----------
    model           : PCA or Keras reconstruction model
    X_rows          : float32 (n_rows, n_features)
    win             : window size
    batch           : Keras batch size
    topk            : score-reduction switch (LSTM/CNN). If set, ``window_scores``
                      / ``row_scores`` use the top-k per-channel MSE rule
                      (see ``score_windows``). PCA: leave as None.
    n_top_channels  : diagnostic only — if set, the output ``window_top_channels``
                      holds the indices of the N channels with the largest
                      per-channel MSE per window, ranked descending.
                      Decoupled from ``topk`` so PCA can return suspect
                      channels without changing the score reduction.

    Returns
    -------
    dict with keys
        row_scores          : (n_rows,)               float32 — threshold on this
        window_scores       : (n_win,)                float32
        per_channel_mse     : (n_feat,)               float32 — mean across all windows
        window_channel_mse  : (n_win, n_feat)         float32 — per-window per-channel
        window_top_channels : (n_win, n_top_channels) int64 or None
                              ranked descending by per-window per-channel MSE
                              (column 0 = largest MSE)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X_rows.shape

    X_win, X_hat = _reconstruct_windows(model, X_rows, win=win, batch=batch)

    if X_win.shape[0] == 0:
        return {
            "row_scores"         : np.zeros(n_rows, dtype=np.float32),
            "window_scores"      : np.zeros(0, dtype=np.float32),
            "per_channel_mse"    : np.zeros(n_feat, dtype=np.float32),
            "window_channel_mse" : np.zeros((0, n_feat), dtype=np.float32),
            "window_top_channels": None,
        }

    sq_err = (X_win - X_hat) ** 2                                 # (n_win, win, n_feat)
    window_channel_mse = sq_err.mean(axis=1).astype(np.float32)   # (n_win, n_feat)
    per_channel_mse    = window_channel_mse.mean(axis=0).astype(np.float32)
    window_scores      = _window_scores_from_sq_err(sq_err, topk=topk)
    row_scores         = broadcast_window_scores_to_rows(window_scores, n_rows, win)

    # diagnostic top-N channels per window, ranked by reconstruction MSE
    # (decoupled from score reduction)
    top_channel_idx = None
    if n_top_channels is not None:
        if not 1 <= n_top_channels <= n_feat:
            raise ValueError(
                f"n_top_channels must be in [1, {n_feat}], got {n_top_channels}"
            )
        # 1) argpartition pulls the N largest per row in O(n_feat)
        # 2) argsort the N-slice (descending) so column 0 is the largest MSE
        part = np.argpartition(
            window_channel_mse, -n_top_channels, axis=1
        )[:, -n_top_channels:]
        # gather MSE values, then sort indices by descending MSE
        mse_part = np.take_along_axis(window_channel_mse, part, axis=1)
        order    = np.argsort(-mse_part, axis=1)
        top_channel_idx = np.take_along_axis(part, order, axis=1)

    return {
        "row_scores"            : row_scores,
        "window_scores"         : window_scores,
        "per_channel_mse"       : per_channel_mse,
        "window_channel_mse"    : window_channel_mse,
        "window_top_channels"   : top_channel_idx,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Drift detrending for score arrays
# ──────────────────────────────────────────────────────────────────────────────
#
# Reconstruction MSE on the ESA-ADB task inherits a slow baseline drift from
# a ~14-channel subsystem that shifts regime between the val and test_intern
# splits (see NB 18 Section 6). The symptom in NB 14: the row-score trace
# step-changes mid-test and a val-tuned threshold flags everything past the
# step. ``detrend_scores`` removes that baseline before thresholding.
#
# Two modes:
#   mode="median"  — score - rolling_median(score)          (mean-drift only)
#   mode="zscore"  — (score - rolling_median) / rolling_MAD (mean + variance)
#
# Use "median" first; switch to "zscore" only if the widening variance
# diagnosed on channels 14/21/29 etc. still hurts after mean subtraction.
# ──────────────────────────────────────────────────────────────────────────────

def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """
    Centred rolling median with reflection at the edges.

    Delegates to ``scipy.ndimage.median_filter``, which runs in bounded
    memory regardless of window size — important for ~10⁷-row score
    arrays where a striding-view + ``np.median`` would OOM.
    """
    if window <= 1:
        return x.astype(np.float32, copy=True)
    from scipy.ndimage import median_filter
    return median_filter(x, size=window, mode="reflect").astype(np.float32, copy=False)


def _rolling_mad(x: np.ndarray, median: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling median-absolute-deviation matching the same window as ``_rolling_median``.

    MAD is robust to anomaly spikes that would blow up a rolling std; that's
    why we use it here instead of ``np.std`` for variance detrending.
    """
    if window <= 1:
        return np.ones_like(x, dtype=np.float32)
    abs_dev = np.abs(x - median)
    return _rolling_median(abs_dev, window)


def detrend_scores(
    scores: np.ndarray,
    window: int = 100_000,
    mode: str = "median",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Remove a rolling baseline from a row-level score array.

    Anomaly-detection thresholds become unstable when the score has a slow
    drift: a val-tuned threshold that separates spikes from baseline on val
    ends up below the drifted baseline on test, and the model flags the
    entire drifted region. Subtracting a rolling baseline aligns the nominal
    level across regimes so the threshold only reacts to excursions above
    the local level.

    Parameters
    ----------
    scores : (n_rows,) float array
        Row-level anomaly scores (typically from ``score_windows``).
    window : int, default 100_000
        Rolling-window size. Should be larger than an expected anomaly event
        (so events aren't absorbed into the baseline) and smaller than the
        timescale of the drift itself. 50k–200k is a reasonable range for
        ESA-ADB's ~15M-row timelines.
    mode : {"median", "zscore"}, default "median"
        * ``"median"`` — return ``scores - rolling_median(scores)``. Kills
          mean-drift. Use first.
        * ``"zscore"`` — return ``(scores - rolling_median) / (rolling_MAD + eps)``.
          Also normalises variance drift. Use when channels also widen
          (test_intern std_ratio ≫ 1, documented for channels 14/21/29 in NB 18).
    eps : float, default 1e-6
        Floor added to the MAD denominator in "zscore" mode to guard against
        flat nominal regions where MAD is zero.

    Returns
    -------
    detrended : (n_rows,) float32 array
        Baseline-removed scores. The threshold should be re-tuned on the
        detrended val scores — the old val threshold is not meaningful here.

    Notes
    -----
    Backed by ``scipy.ndimage.median_filter`` (C implementation, bounded
    memory). Safe to call on ~10⁷-row arrays with ``window=100_000`` without
    blowing memory. Prefer per-split calls over a concatenated array — that
    keeps the baseline estimate from leaking across regimes and avoids
    cross-split smoothing near the boundary.
    """
    if mode not in ("median", "zscore"):
        raise ValueError(f"mode must be 'median' or 'zscore', got {mode!r}")

    x = np.asarray(scores, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"scores must be 1D, got shape {x.shape}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    window = min(window, len(x))  # clamp so we don't pad bigger than the array

    base = _rolling_median(x, window)
    if mode == "median":
        return (x - base).astype(np.float32, copy=False)

    scale = _rolling_mad(x, base, window) + eps
    return ((x - base) / scale).astype(np.float32, copy=False)


def detrend_window_scores(
    win_scores: np.ndarray,
    n_rows: int,
    window: int = 1000,
    mode: str = "median",
    eps: float = 1e-6,
    win: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Detrend at window granularity, then broadcast back to rows.

    Equivalent to ``detrend_scores(broadcast_window_scores_to_rows(...))``
    but ~``win``x faster because the median filter runs on ``n_rows/win``
    points instead of ``n_rows`` points. Numerically identical as long as
    the score array is piecewise-constant (which it is, by construction
    of ``broadcast_window_scores_to_rows``).

    ``window`` here is measured in WINDOWS, not rows -- i.e. ``window=1000``
    ~= 100k rows for ``win=100``. Pick it the same way you'd pick the
    row-level window in ``detrend_scores``, divided by ``win``.
    """
    detrended_win = detrend_scores(win_scores, window=window, mode=mode, eps=eps)
    return broadcast_window_scores_to_rows(detrended_win, n_rows, win=win)


def score_windows_detrended(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
    topk: int | None = None,
    detrend_window: int = 1000,
    detrend_mode: str = "median",
) -> np.ndarray:
    """
    ``score_windows`` + window-level detrending in one call.

    Convenience for the standard pipeline: reconstruct, score, subtract
    rolling baseline, broadcast to rows. The returned array is
    **threshold-ready** but the threshold must be re-tuned on the
    detrended val output -- the pre-detrend threshold is meaningless here.
    """
    win_scores = window_scores_only(model, X_rows, win=win, batch=batch, topk=topk)
    return detrend_window_scores(
        win_scores, n_rows=X_rows.shape[0],
        window=detrend_window, mode=detrend_mode, win=win,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Prediction post-processing — block-length filter
# ──────────────────────────────────────────────────────────────────────────────
#
# Recall-tuned thresholds (e.g. NB 31's nom_p95) generate many short isolated
# predicted blocks where the score grazes the threshold inside nominal regions.
# Real ESA-ADB events are typically hundreds-to-thousands of rows long, so two
# simple post-processing rules clean up the timeline without losing recall:
#
#   1. Merge predicted blocks separated by short gaps (≤ ``max_gap`` rows).
#   2. Drop predicted blocks shorter than ``min_len`` rows.
#
# Used by the FE pipeline (NB 22c) to turn a 100k-FP-row scatter into a
# small number of clean event blocks.

def clean_predictions(
    y_pred: np.ndarray,
    min_len: int = 500,
    max_gap: int = 0,
) -> np.ndarray:
    """
    Post-process a binary prediction array: merge near-adjacent blocks,
    then drop blocks shorter than ``min_len``.

    Parameters
    ----------
    y_pred : (n_rows,) array of 0/1
        Row-level predicted labels (e.g. from ``score > threshold``).
    min_len : int, default 500
        Minimum predicted-block length in rows. Anything shorter is zeroed.
        Must be >= 1.
    max_gap : int, default 0
        If > 0, runs of zeros up to ``max_gap`` rows long that sit *between*
        two predicted-positive runs are filled with 1s before the min-length
        filter is applied. Useful when a single event is split into two
        predicted blocks by a brief score dip below threshold.

    Returns
    -------
    (n_rows,) int8 array — cleaned labels.

    Notes
    -----
    Operates only on the prediction array, not on the underlying scores or
    ground truth. Idempotent for ``max_gap=0`` only — gap-filling is a
    one-shot operation; running it twice with the same gap is the same as
    running it once.
    """
    y = np.asarray(y_pred, dtype=np.int8).copy()
    if y.ndim != 1:
        raise ValueError(f"y_pred must be 1D, got shape {y.shape}")
    if min_len < 1:
        raise ValueError(f"min_len must be >= 1, got {min_len}")
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0, got {max_gap}")

    # Step 1 — fill short gaps between predicted-positive runs.
    if max_gap > 0:
        padded = np.concatenate(([1], y, [1]))      # treat the array boundaries as positive
        d      = np.diff(padded.astype(np.int8))
        gap_starts = np.where(d == -1)[0]            # 1 → 0 transitions in padded
        gap_ends   = np.where(d ==  1)[0] - 1        # 0 → 1 transitions in padded
        for s, e in zip(gap_starts, gap_ends):
            if 1 <= s and e <= len(y) - 2 and (e - s + 1) <= max_gap:
                y[s:e + 1] = 1

    # Step 2 — drop predicted-positive runs shorter than min_len.
    padded = np.concatenate(([0], y, [0]))
    d      = np.diff(padded.astype(np.int8))
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0] - 1
    for s, e in zip(starts, ends):
        if (e - s + 1) < min_len:
            y[s:e + 1] = 0

    return y
