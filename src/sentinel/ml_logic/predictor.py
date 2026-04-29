"""
Inference façade over ``sentinel.ml_logic.scorer``.

Two entry points, split by *use-case* rather than by score type:

    predict()         — operational path. Returns only the Kaggle-style
                        submission frame (id, is_anomaly). Cheap, no extras.
    predict_report()  — analytical path. Returns the full diagnostic dict
                        (scores, per-channel MSE, top channels) used by the
                        notebooks and the API ``/report`` endpoint.

Both share the same load → scale → score pipeline; they only differ in what
they package on the way out. Keeping them separate avoids paying the report-
assembly cost on the hot prediction path.

Every artefact argument is optional. Anything left as ``None`` is loaded from
the bootcamp defaults on disk:
    model    : models/pca.pkl
    scaler   : models/scaler.pkl
    features : data/raw/target_channels.csv
    X_raw    : data/processed/test_api.npy

``threshold`` and ``win`` default to ``sentinel.params``.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from ..params import PCA_THRESHOLD, WINDOW_SIZE
from .data import MODELS_DIR, PROCESSED_DIR, load_target_channels
from .scorer import score_report, score_windows


# ── internal helpers ─────────────────────────────────────────────────────────
def _load(model=None, scaler=None, features=None, X_raw=None):
    """Backfill any missing artefact from the bootcamp defaults on disk.

    Designed so callers can override one piece at a time (e.g. inject a
    freshly-trained model in a notebook or test) without having to wire up
    the other three.
    """
    if model is None:
        with open(MODELS_DIR / "pca.pkl", "rb") as f:
            model = pickle.load(f)
    if scaler is None:
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    if features is None:
        features = load_target_channels()
    if X_raw is None:
        X_raw = np.load(PROCESSED_DIR / "test_api.npy")
    return model, scaler, list(features), X_raw


def _scale(scaler, features, X_raw):
    """Align columns to ``features`` order, then apply the scaler.

    DataFrame input  → reindexed by name; extra columns are dropped, missing
                       columns raise. This is what makes the API robust to
                       payloads that include an ``id`` column or that ship
                       channels in a different order.
    Ndarray input    → taken as-is. The caller MUST have it in ``features``
                       order already — there is no name to align on.
    """
    if isinstance(X_raw, pd.DataFrame):
        X = X_raw[features].values
    else:
        X = np.asarray(X_raw)
    X = X.astype(np.float32, copy=False)
    return scaler.transform(X).astype(np.float32)


# ── public API ───────────────────────────────────────────────────────────────
def predict(
    model=None,
    scaler=None,
    features=None,
    X_raw=None,
    threshold: float = PCA_THRESHOLD,
    win: int = WINDOW_SIZE,
) -> pd.DataFrame:
    """Score rows and return a Kaggle-style submission frame.

    Output: DataFrame with columns ``id`` (int64) and ``is_anomaly`` (int8,
    0/1). One row per input row.

    Pass nothing to run the default PCA pipeline on the test_api slice; pass
    any subset of artefacts to override individual defaults.
    """
    model, scaler, features, X_raw = _load(model, scaler, features, X_raw)
    X_scaled = _scale(scaler, features, X_raw)

    row_scores = score_windows(model, X_scaled, win=win)
    labels = (row_scores > threshold).astype(np.int8)

    # Preserve caller-supplied ids when present so submissions stay aligned
    # with the original frame. Ndarray callers get a fresh 0..N range — they
    # are responsible for tracking offsets if that matters.
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
    threshold: float = PCA_THRESHOLD,
    win: int = WINDOW_SIZE,
    topk: int | None = None,
    n_top_channels: int | None = None,
) -> dict:
    """Score rows and return the full diagnostic dict for plots / the API report.

    Two knobs that look similar but do completely different things:

    ``topk``
        Score-reduction switch for LSTM/CNN models — limits each window's
        score to the top-k channel MSEs *before* averaging. CHANGES the
        scoring math. PCA ignores it; leave ``None``.

    ``n_top_channels``
        Diagnostic only — controls how many suspect channels per window
        are returned in the output ``window_top_channels`` array (ranked by
        MSE, descending; column 0 is the largest). Does NOT affect
        ``row_scores`` / ``window_scores``.
    """
    model, scaler, features, X_raw = _load(model, scaler, features, X_raw)
    X_scaled = _scale(scaler, features, X_raw)

    rep = score_report(
        model, X_scaled, win=win, topk=topk, n_top_channels=n_top_channels,
    )
    labels = (rep["row_scores"] > threshold).astype(np.int8)

    return {
        "labels"                : labels,
        # Per-row reconstruction MSE (PCA default = mean over all used channels)
        "row_scores"            : rep["row_scores"],

        # Window scores
        "window_scores"         : rep["window_scores"],
        "window_channel_mse"    : rep["window_channel_mse"],
        "window_top_channels"   : rep["window_top_channels"],

        "per_channel_mse"       : rep["per_channel_mse"],
        "threshold"             : float(threshold),
        "features"              : list(features),
    }
