from __future__ import annotations

import numpy as np
import pandas as pd

from ..params import WINDOW_SIZE
from .scoring import score_report


def predict(
    model,
    scaler,
    features: list[str],
    X_raw,
    threshold: float,
    win: int = WINDOW_SIZE,
    topk: int | None = None,
) -> dict:
    # DataFrame → pick the right columns in the right order;
    # ndarray is assumed to already be in feature order.
    if isinstance(X_raw, pd.DataFrame):
        X = X_raw[features].values
    else:
        X = np.asarray(X_raw)
    X = X.astype(np.float32, copy=False)

    X_scaled = scaler.transform(X).astype(np.float32)

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
