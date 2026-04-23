"""
Two public entry points:
    * predict         — Kaggle-style (id, is_anomaly) DataFrame
    * predict_report  — full dict for the NB 15 showcase plots

Every argument is optional. Anything left as ``None`` is loaded from the defaults:
    model    : models/lstm_ae.keras
    scaler   : models/scaler.pkl
    features : data/raw/target_channels.csv
    X_raw    : data/processed/test_intern_raw.npy

``WINDOW_SIZE`` and ``LSTM_THRESHOLD`` default to the values in
``sentinel.params``.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..params import LSTM_THRESHOLD, WINDOW_SIZE
from .data import MODELS_DIR, PROCESSED_DIR, load_target_channels
from .scorer import score_report, score_windows


# ── internal helpers ──────────────────────────────────────────────────────
def _load(model=None, scaler=None, features=None, X_raw=None):
    """Fill in any missing artefact from the bootcamp defaults."""
    if model is None:
        # lazy import so PCA-only callers don't pay the tensorflow cost
        from tensorflow.keras.models import load_model
        model = load_model(MODELS_DIR / "lstm_ae.keras", compile=False)
    if scaler is None:
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    if features is None:
        features = load_target_channels()
    if X_raw is None:
        X_raw = np.load(PROCESSED_DIR / "test_intern_raw.npy")
    return model, scaler, list(features), X_raw


def _scale(scaler, features, X_raw):
    # DataFrame: pick columns in the right order. ndarray: assume feature order.
    if isinstance(X_raw, pd.DataFrame):
        X = X_raw[features].values
    else:
        X = np.asarray(X_raw)
    X = X.astype(np.float32, copy=False)
    return scaler.transform(X).astype(np.float32)


# ── public API ────────────────────────────────────────────────────────────
def predict(
    model=None,
    scaler=None,
    features=None,
    X_raw=None,
    threshold: float = LSTM_THRESHOLD,
    win: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """
    Output: DataFrame with columns ``id`` and ``is_anomaly``.

    Pass nothing to run the default LSTM-AE pipeline on test_intern; pass any
    subset of artefacts to override individual defaults.
    """
    model, scaler, features, X_raw = _load(model, scaler, features, X_raw)
    X_scaled = _scale(scaler, features, X_raw)

    row_scores = score_windows(model, X_scaled, win=win)
    labels = (row_scores > threshold).astype(np.int8)

    if isinstance(X_raw, pd.DataFrame) and "id" in X_raw.columns:
        ids = X_raw["id"].values
    else:
        ids = np.arange(len(labels), dtype=np.int64)

    return pd.DataFrame({"id": ids, "is_anomaly": labels})


def predict_report(
    model=None,
    scaler=None,
    features=None,
    X_raw=None,
    threshold: float = LSTM_THRESHOLD,
    win: int = WINDOW_SIZE,
    topk: int | None = None,
) -> dict:
    """Full dict for the NB 15 showcase plots. Same defaults as ``predict``."""
    model, scaler, features, X_raw = _load(model, scaler, features, X_raw)
    X_scaled = _scale(scaler, features, X_raw)

    rep = score_report(model, X_scaled, win=win, topk=topk)
    labels = (rep["row_scores"] > threshold).astype(np.int8)

    return {
        "labels"            : labels,
        "row_scores"        : rep["row_scores"],
        "window_scores"     : rep["window_scores"],
        "per_channel_mse"   : rep["per_channel_mse"],
        "window_channel_mse": rep["window_channel_mse"],
        "topk_channels"     : rep["topk_channels"],
        "threshold"         : float(threshold),
        "features"          : list(features),
    }
