"""
Generic window-mean MSE scoring for reconstruction anomaly-detection models.

The bootcamp notebooks (11_pca, 12_lstm_ae, 13_cnn_ae) all follow the same
scoring recipe:

    1. Reshape row-level scaled data into non-overlapping windows
    2. Feed through the model to get a reconstruction
    3. Take mean squared error over (time, channel) axes → one score per window
    4. Broadcast each window score to its WIN rows (tail rows inherit the last
       full window's score)

This module lifts that recipe out of the notebooks so the three baselines
share byte-identical scoring code. It supports both Keras-style models
(``model.predict(X_3d)``) and sklearn PCA (``model.inverse_transform(
model.transform(X_flat))``).
"""
from __future__ import annotations

import numpy as np

from ..params import WINDOW_SIZE


def broadcast_window_scores_to_rows(
    win_scores: np.ndarray,
    n_rows: int,
    win: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Repeat each window score ``win`` times. Tail rows (< win) inherit the
    last window's score.
    """
    if len(win_scores) == 0:
        return np.zeros(n_rows, dtype=np.float32)
    row_scores = np.repeat(win_scores.astype(np.float32), win)
    if len(row_scores) < n_rows:
        pad = np.full(n_rows - len(row_scores),
                      win_scores[-1], dtype=np.float32)
        row_scores = np.concatenate([row_scores, pad])
    return row_scores[:n_rows]


def score_windows(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
) -> np.ndarray:
    """
    Row-level anomaly scores via window-mean MSE reconstruction.

    Parameters
    ----------
    model   : fitted reconstruction model. Two interfaces are auto-detected:
                  * Keras  → ``model.predict(X_3d, batch_size=batch)`` returns a
                    3D tensor of the same shape
                  * sklearn PCA → ``model.inverse_transform(model.transform(
                    X_flat))`` on a (n_windows, win * n_feat) flattened array
    X_rows  : float32 (n_rows, n_features)
    win     : window size
    batch   : batch size for the forward pass

    Returns
    -------
    float32 (n_rows,) — per-row anomaly score (window-mean MSE broadcast)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    N, n_feat = X_rows.shape
    n_complete = N // win
    if n_complete == 0:
        return np.zeros(N, dtype=np.float32)

    X_win = X_rows[:n_complete * win].reshape(n_complete, win, n_feat)

    # Dispatch: sklearn PCA has both inverse_transform and transform;
    # Keras models have predict. Check PCA-shape first so a future Keras model
    # that happens to expose transform/inverse_transform doesn't get mis-routed.
    has_pca_api   = hasattr(model, "inverse_transform") and hasattr(model, "transform")
    has_keras_api = hasattr(model, "predict")

    if has_pca_api:
        X_flat = X_win.reshape(n_complete, win * n_feat)
        X_hat  = model.inverse_transform(model.transform(X_flat))
        win_scores = ((X_flat - X_hat) ** 2).mean(axis=1).astype(np.float32)
    elif has_keras_api:
        # Keras autoencoder: same shape in and out
        X_hat = model.predict(X_win, batch_size=batch, verbose=0)
        if X_hat.shape != X_win.shape:
            raise ValueError(
                f"Reconstruction shape {X_hat.shape} != input shape {X_win.shape}"
            )
        win_scores = ((X_win - X_hat) ** 2).mean(axis=(1, 2)).astype(np.float32)
    else:
        raise TypeError(
            f"Unsupported model type {type(model).__name__!r} — needs either "
            f"(transform + inverse_transform) or predict()"
        )

    return broadcast_window_scores_to_rows(win_scores, N, win)


def window_scores_only(
    model,
    X_rows: np.ndarray,
    win: int = WINDOW_SIZE,
    batch: int = 256,
) -> np.ndarray:
    """
    Like ``score_windows`` but returns the *window-level* score array
    (length = n_rows // win). Useful when callers want to inspect or
    manipulate window scores before broadcasting.
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    N, n_feat = X_rows.shape
    n_complete = N // win
    if n_complete == 0:
        return np.zeros(0, dtype=np.float32)
    X_win = X_rows[:n_complete * win].reshape(n_complete, win, n_feat)

    if hasattr(model, "inverse_transform") and hasattr(model, "transform"):
        X_flat = X_win.reshape(n_complete, win * n_feat)
        X_hat  = model.inverse_transform(model.transform(X_flat))
        return ((X_flat - X_hat) ** 2).mean(axis=1).astype(np.float32)
    if hasattr(model, "predict"):
        X_hat = model.predict(X_win, batch_size=batch, verbose=0)
        return ((X_win - X_hat) ** 2).mean(axis=(1, 2)).astype(np.float32)
    raise TypeError(
        f"Unsupported model type {type(model).__name__!r}"
    )
