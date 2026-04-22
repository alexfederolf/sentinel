"""
Generic window-mean MSE scoring for reconstruction anomaly-detection models.

The models all follow the same scoring recipe:

    1. Reshape row-level data into non-overlapping windows.
    2. Reconstruct each window with the model.
    3. Squared error → reduce to one number per window.
    4. Broadcast each window score to its ``win`` rows (the tail inherits the
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
broadcast_window_scores_to_rows  broadcast helper (tail inherits last score)
score_report                     single-pass, everything-at-once; use this for
                                 the showcase / frontend / multi-plot notebooks
"""
from __future__ import annotations

import numpy as np

from ..params import WINDOW_SIZE


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
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

    If fewer than one full window fits, returns two empty arrays of shape
    ``(0, win, n_feat)`` and does NOT dispatch to the model (matches the
    previous short-input behaviour of ``score_windows``).

    The PCA check comes first so a future Keras model that happens to expose
    ``transform``/``inverse_transform`` does not get mis-routed.
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    N, n_feat = X_rows.shape
    n_complete = N // win

    if n_complete == 0:
        empty = np.zeros((0, win, n_feat), dtype=np.float32)
        return empty, empty

    X_win = X_rows[:n_complete * win].reshape(n_complete, win, n_feat)

    if _is_pca(model):
        X_flat = X_win.reshape(n_complete, win * n_feat)
        X_hat  = model.inverse_transform(model.transform(X_flat)).reshape(
            n_complete, win, n_feat
        )
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

    return X_win, np.asarray(X_hat, dtype=np.float32)


def _window_scores_from_sq_err(
    sq_err: np.ndarray,
    topk: int | None,
) -> np.ndarray:
    """
    Reduce per-window squared errors to one number per window.

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
    if topk is None:
        return sq_err.mean(axis=(1, 2)).astype(np.float32)
    if not 1 <= topk <= n_feat:
        raise ValueError(f"topk must be in [1, {n_feat}], got {topk}")
    per_channel = sq_err.mean(axis=1)                        # (n_win, n_feat)
    vals = np.partition(per_channel, -topk, axis=1)[:, -topk:]
    return vals.mean(axis=1).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Broadcast helper
# ──────────────────────────────────────────────────────────────────────────────

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
    row_scores = np.repeat(win_scores.astype(np.float32), win)
    if len(row_scores) < n_rows:
        pad = np.full(n_rows - len(row_scores),
                      win_scores[-1], dtype=np.float32)
        row_scores = np.concatenate([row_scores, pad])
    return row_scores[:n_rows]


# ──────────────────────────────────────────────────────────────────────────────
# Public scoring functions
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
    model   : fitted reconstruction model (PCA or Keras — see module docstring)
    X_rows  : float32 (n_rows, n_features)
    win     : window size
    batch   : Keras batch size
    topk    : if set, each window's score is the mean of its ``topk`` largest
              per-channel MSE values instead of the mean over all channels.
              Useful when anomalies affect only a few channels (LSTM/CNN
              failure mode in NB 12/13). Note: under ``topk`` the returned
              array is no longer a mean MSE — it is the mean of the top-k
              per-channel MSEs, broadcast to rows.

    Returns
    -------
    float32 (n_rows,) — per-row anomaly score (window statistic broadcast)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    X_win, X_hat = _reconstruct_windows(model, X_rows, win=win, batch=batch)
    if X_win.shape[0] == 0:
        return np.zeros(X_rows.shape[0], dtype=np.float32)
    sq_err = (X_win - X_hat) ** 2
    win_scores = _window_scores_from_sq_err(sq_err, topk=topk)
    return broadcast_window_scores_to_rows(win_scores, X_rows.shape[0], win)


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
) -> dict:
    """
    One-pass report: runs the reconstruction once and derives every statistic
    the showcase / frontend / multi-plot notebooks consume. Replaces the
    pattern of calling ``score_windows`` for labels plus a hand-rolled PCA
    loop for per-channel MSE.

    Parameters
    ----------
    model   : PCA or Keras reconstruction model
    X_rows  : float32 (n_rows, n_features)
    win     : window size
    batch   : Keras batch size
    topk    : if set, ``window_scores`` / ``row_scores`` use the top-k
              per-channel MSE rule (see ``score_windows``). ``topk_channels``
              then holds the indices of the k largest channels per window.

    Returns
    -------
    dict with keys
        row_scores         : (n_rows,)        float32 — threshold on this
        window_scores      : (n_win,)         float32
        per_channel_mse    : (n_feat,)        float32 — mean across all windows
        window_channel_mse : (n_win, n_feat)  float32 — per-window per-channel
        topk_channels      : (n_win, topk)    int64 or None (if topk is None)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X_rows.shape

    X_win, X_hat = _reconstruct_windows(model, X_rows, win=win, batch=batch)

    if X_win.shape[0] == 0:
        return {
            "row_scores"        : np.zeros(n_rows, dtype=np.float32),
            "window_scores"     : np.zeros(0, dtype=np.float32),
            "per_channel_mse"   : np.zeros(n_feat, dtype=np.float32),
            "window_channel_mse": np.zeros((0, n_feat), dtype=np.float32),
            "topk_channels"     : None,
        }

    sq_err = (X_win - X_hat) ** 2                                 # (n_win, win, n_feat)
    window_channel_mse = sq_err.mean(axis=1).astype(np.float32)   # (n_win, n_feat)
    per_channel_mse    = window_channel_mse.mean(axis=0).astype(np.float32)
    window_scores      = _window_scores_from_sq_err(sq_err, topk=topk)
    row_scores         = broadcast_window_scores_to_rows(window_scores, n_rows, win)

    topk_channels = None
    if topk is not None:
        # indices of the k largest channels per window (unordered inside the slice)
        topk_channels = np.argpartition(
            window_channel_mse, -topk, axis=1
        )[:, -topk:]

    return {
        "row_scores"        : row_scores,
        "window_scores"     : window_scores,
        "per_channel_mse"   : per_channel_mse,
        "window_channel_mse": window_channel_mse,
        "topk_channels"     : topk_channels,
    }
