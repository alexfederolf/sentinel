from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from ..params import WINDOW_SIZE
from .scorer import score_report, score_windows

# helper to load all the artefacts needed for prediction (model, scaler, feature list)
def load_artefacts(model_path, scaler_path, config_path, loader="pickle"):

    # pickle for sklearn (PCA); keras for LSTM-AE / CNN-AE later on.
    if loader == "pickle":
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # keras for LSTM/CNN AE
    elif loader == "keras":
        # lazy import so PCA doesn't pay the tensorflow cost
        from tensorflow.keras.models import load_model
        model = load_model(model_path, compile=False)
    else:
        raise ValueError(f"Unknown loader {loader!r} — use 'pickle' or 'keras'")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(config_path) as f:
        features = json.load(f)["target_channels"]

    return model, scaler, features


def _scale(scaler, features, X_raw):

    # DataFrame → pick the right columns in the right order
    if isinstance(X_raw, pd.DataFrame):
        X = X_raw[features].values

    # ndarray is assumed to already be in feature order
    else:
        X = np.asarray(X_raw)
    X = X.astype(np.float32, copy=False)

    return scaler.transform(X).astype(np.float32)


# simple variant: Kaggle-style submission (id, is_anomaly)
def predict(
    model,
    scaler,
    features: list[str],
    X_raw,
    threshold: float,
    win: int = WINDOW_SIZE,
) -> pd.DataFrame:
    X_scaled = _scale(scaler, features, X_raw)

    # score_windows = row-level MSE, broadcast from window scores
    row_scores = score_windows(model, X_scaled, win=win)
    labels = (row_scores > threshold).astype(np.int8)

    # id column: prefer X_raw['id'] if present, else a 0..n range
    if isinstance(X_raw, pd.DataFrame) and "id" in X_raw.columns:
        ids = X_raw["id"].values
    else:
        ids = np.arange(len(labels), dtype=np.int64)

    return pd.DataFrame({"id": ids, "is_anomaly": labels})


# detailed variant: full dict for the showcase plots (per-channel, MSE, etc.)
def predict_report(
    model,
    scaler,
    features: list[str],
    X_raw,
    threshold: float,
    win: int = WINDOW_SIZE,
    topk: int | None = None,
) -> dict:
    X_scaled = _scale(scaler, features, X_raw)

    # score_report does reconstruction + MSE + broadcast in one pass
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
