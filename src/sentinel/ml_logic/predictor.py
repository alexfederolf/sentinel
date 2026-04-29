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

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from ..params import PCA_THRESHOLD, WINDOW_SIZE
from .data import MODELS_DIR, PROCESSED_DIR, RAW_DIR, load_target_channels
from .scorer import (
    clean_predictions,
    detrend_window_scores,
    score_report,
    score_windows,
    score_windows_detrended,
)


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


# ── FE 46-channel detrended PCA pipeline ─────────────────────────────────────
# NB 11e recipe: Tier A+B prune (46ch) + per-channel rolling-median input
# detrend + tail-fit PCA + score-level rolling-median detrend + threshold tuned
# on val 10.7M..12.7M with corrected_event_f05. Optional clean_predictions
# post-filter (min_len=100, max_gap=500) is the FE-blessed default.
#
# Artifacts (all under <repo>/models or <repo>/data/raw):
#   models/pca_fe_46ch.pkl                  fitted PCA (46ch, k_keep≈316)
#   models/fe_46ch_<ts>.json                sidecar (threshold + config + metrics)
#   data/raw/target_channels_fe.csv         46-channel order
#
# Inference path: scaler.pkl (kaggle 58ch RobustScaler) → slice 46ch →
# per-channel rolling-median input detrend (window 100k) → score_windows_detrended
# (window 1000 windows = 100k rows, mode median) → threshold → optional postfilter.

FE46_PCA_PATH      = MODELS_DIR / "pca_fe_46ch.pkl"
FE46_FEATURES_PATH = RAW_DIR / "target_channels_fe.csv"
FE46_INPUT_DETREND_WIN = 100_000
FE46_SCORE_DETREND_WIN = 1_000
FE46_POST_MIN_LEN  = 100
FE46_POST_MAX_GAP  = 500


def _detrend_per_channel(arr2d: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr2d, dtype=np.float32)
    for j in range(arr2d.shape[1]):
        col = arr2d[:, j].astype(np.float32, copy=False)
        out[:, j] = col - median_filter(col, size=window, mode="reflect")
    return out


def _load_fe46_sidecar() -> dict:
    """Pick the latest fe_46ch_<ts>.json sidecar (for the threshold)."""
    candidates = sorted(MODELS_DIR.glob("fe_46ch_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No fe_46ch_*.json sidecar found in {MODELS_DIR}. "
            "Run scripts/train_fe46.py first."
        )
    with open(candidates[-1]) as f:
        return json.load(f)


def _load_fe46(model, scaler, features_full, features_fe, threshold, X_raw):
    """Backfill any missing FE artefact from disk. Mirror of ``_load`` for FE."""
    if model is None:
        with open(FE46_PCA_PATH, "rb") as f:
            model = pickle.load(f)
    if scaler is None:
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    if features_full is None:
        features_full = load_target_channels()
    if features_fe is None:
        features_fe = list(pd.read_csv(FE46_FEATURES_PATH)["target_channels"])
    if threshold is None:
        threshold = float(_load_fe46_sidecar()["threshold"])
    if X_raw is None:
        X_raw = np.load(PROCESSED_DIR / "test_api.npy")
    return model, scaler, list(features_full), list(features_fe), float(threshold), X_raw


def _prep_fe46_input(X_raw, scaler, features_full, features_fe):
    """Shared FE prefix: align → scale (58ch) → slice (46ch) → input detrend.

    Returns ``(X_dt, ids)``.

    ``X_dt`` is the per-channel rolling-median-detrended 46-channel float32
    array that both ``predict_fe46`` and ``predict_fe46_report`` feed into the
    scorer. ``ids`` is the caller-supplied id column (DataFrame input) or
    ``None``; the public functions fall back to ``arange(n_rows)`` when None.
    """
    if isinstance(X_raw, pd.DataFrame):
        ids   = X_raw["id"].values if "id" in X_raw.columns else None
        X_raw = X_raw[features_full].values
    else:
        ids   = None
        X_raw = np.asarray(X_raw)
    X_scaled = scaler.transform(X_raw.astype(np.float32, copy=False)).astype(np.float32)

    fe_idx = np.array([features_full.index(c) for c in features_fe])
    X_fe   = np.ascontiguousarray(X_scaled[:, fe_idx])

    in_win = min(FE46_INPUT_DETREND_WIN, X_fe.shape[0])
    X_dt   = _detrend_per_channel(X_fe, in_win)
    return X_dt, ids


def load_fe46_artefacts() -> dict:
    """Load the full FE inference bundle from disk in one shot.

    Designed for callers (e.g. the FastAPI lifespan) that need to cache every
    artefact at startup. The first five keys map 1:1 onto the
    ``predict_fe46`` / ``predict_fe46_report`` signatures so the bundle can be
    spread directly:

        artefacts = load_fe46_artefacts()
        sub = predict_fe46(X_raw=X, **artefacts)

        model            : fitted PCA (46-ch detrended)
        scaler           : RobustScaler (58-ch — predict_fe46 re-applies and slices)
        features_full    : 58-ch order (matches the scaler)
        features_fe      : 46-ch order (FE channels, Tier A+B dropped)
        threshold        : val-tuned threshold from the sidecar
        post_filter_cfg  : {"min_len": ..., "max_gap": ...} from the sidecar
                           (renamed to avoid colliding with the ``post_filter``
                           bool param on the predict functions)
        sidecar          : the full sidecar dict (kept for /report metadata)
    """
    with open(FE46_PCA_PATH, "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    sidecar = _load_fe46_sidecar()
    return {
        "model"          : model,
        "scaler"         : scaler,
        "features_full"  : load_target_channels(),
        "features_fe"    : list(pd.read_csv(FE46_FEATURES_PATH)["target_channels"]),
        "threshold"      : float(sidecar["threshold"]),
        "post_filter_cfg": sidecar["post_filter"],
        "sidecar"        : sidecar,
    }


def predict_fe46(
    X_raw=None,
    model=None,
    scaler=None,
    features_full=None,
    features_fe=None,
    threshold: float | None = None,
    win: int = WINDOW_SIZE,
    post_filter: bool = True,
    **_unused,
) -> pd.DataFrame:
    """FE-blessed operational inference: 46ch detrended PCA → labels.

    Pipeline mirrors NB 11e + scripts/train_fe46.py exactly:
        1. Align raw 58ch frame to ``features_full`` order, RobustScaler.
        2. Slice to the 46 FE channels (Tier A+B dropped).
        3. Per-channel rolling-median input detrend (window 100k rows).
        4. ``score_windows_detrended`` (window 1000 windows × 100 = 100k rows,
           mode "median") → row-level scores.
        5. ``scores > threshold`` → binary labels.
        6. Optional ``clean_predictions(min_len=100, max_gap=500)`` post-filter.

    Threshold defaults to the value in the latest ``models/fe_46ch_*.json``
    sidecar (tuned on val with ``corrected_event_f05``).

    For per-channel MSE / top-channel diagnostics / row scores (e.g. the API
    ``/report`` endpoint or the score distribution plot), use
    ``predict_fe46_report()`` instead.

    Returns
    -------
    DataFrame with columns ``id`` (int64) and ``is_anomaly`` (int8).

    Notes
    -----
    ``**_unused`` accepts and ignores extra keys such as ``post_filter`` config
    or ``sidecar`` so callers can spread the full ``load_fe46_artefacts()``
    bundle directly:

        artefacts = load_fe46_artefacts()
        predict_fe46(X_raw=X, **artefacts)
    """
    model, scaler, features_full, features_fe, threshold, X_raw = _load_fe46(
        model, scaler, features_full, features_fe, threshold, X_raw,
    )

    X_dt, ids = _prep_fe46_input(X_raw, scaler, features_full, features_fe)

    scores = score_windows_detrended(
        model, X_dt,
        win=win,
        detrend_window=FE46_SCORE_DETREND_WIN,
        detrend_mode="median",
    )

    labels = (scores > threshold).astype(np.int8)
    if post_filter:
        labels = clean_predictions(
            labels, min_len=FE46_POST_MIN_LEN, max_gap=FE46_POST_MAX_GAP,
        )

    if ids is None:
        ids = np.arange(len(labels), dtype=np.int64)
    return pd.DataFrame({"id": ids, "is_anomaly": labels})


def predict_fe46_report(
    X_raw=None,
    model=None,
    scaler=None,
    features_full=None,
    features_fe=None,
    threshold: float | None = None,
    win: int = WINDOW_SIZE,
    post_filter: bool = True,
    n_top_channels: int | None = None,
    **_unused,
) -> dict:
    """FE-blessed analytical inference: full diagnostic dict for plots / API.

    Same input pipeline as ``predict_fe46`` (scale → slice 46ch → input
    detrend), but the scoring step goes through ``score_report`` instead of
    ``score_windows_detrended`` so we get ``per_channel_mse``,
    ``window_channel_mse`` and ``window_top_channels`` alongside the row
    scores. The score-level rolling-median detrend is then applied to
    ``window_scores`` and re-broadcast to rows — bit-identical to the
    operational path on the threshold-ready ``row_scores``.

    Mirrors ``predict_report`` for the bootcamp pipeline.

    Returns
    -------
    dict with keys
        labels              : (n_rows,) int8     (post-filtered if requested)
        row_scores          : (n_rows,) float32  detrended, threshold-ready
        window_scores       : (n_win,)  float32  pre-detrend, raw window MSE
        window_channel_mse  : (n_win, 46) float32
        window_top_channels : (n_win, n_top_channels) int64 or None
        per_channel_mse     : (46,)    float32
        threshold           : float
        features            : list[str]  the 46 FE channels in canonical order
    """
    model, scaler, features_full, features_fe, threshold, X_raw = _load_fe46(
        model, scaler, features_full, features_fe, threshold, X_raw,
    )

    X_dt, _ = _prep_fe46_input(X_raw, scaler, features_full, features_fe)

    rep = score_report(model, X_dt, win=win, n_top_channels=n_top_channels)

    # Score-level detrend on window_scores, then re-broadcast to rows. Matches
    # what score_windows_detrended (used by predict_fe46) does end-to-end.
    row_scores = detrend_window_scores(
        rep["window_scores"], n_rows=X_dt.shape[0],
        window=FE46_SCORE_DETREND_WIN, mode="median", win=win,
    )

    labels = (row_scores > threshold).astype(np.int8)
    if post_filter:
        labels = clean_predictions(
            labels, min_len=FE46_POST_MIN_LEN, max_gap=FE46_POST_MAX_GAP,
        )

    return {
        "labels"             : labels,
        "row_scores"         : row_scores,
        "window_scores"      : rep["window_scores"],
        "window_channel_mse" : rep["window_channel_mse"],
        "window_top_channels": rep["window_top_channels"],
        "per_channel_mse"    : rep["per_channel_mse"],
        "threshold"          : float(threshold),
        "features"           : list(features_fe),
    }
