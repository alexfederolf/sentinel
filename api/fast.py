"""
SENTINEL Anomaly Detection API — FE 46ch detrended PCA pipeline.

Endpoints
---------
GET  /               → health check
GET  /timeline       → cached labels for the test_api_fe slice
GET  /predict_by_id  → filter cached timeline by ID range
GET  /report         → cached anomaly report (row scores, per-channel MSE,
                       top channels per window, threshold)
GET  /channels       → raw signal values + anomaly label for a single channel
GET  /features       → list of available channel names (58ch)
POST /predict        → score user-supplied raw 58-channel rows using the
                       cached FE bundle

Heavy work (model + scaler + sidecar load, prediction over the demo slice)
runs once at startup via the lifespan and is cached on ``app.state``.
The previous Kaggle-pipeline lifespan + endpoints are kept at the bottom of
this file, commented out, for reference.
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.ml_logic.predictor import (
    load_fe46_artefacts,
    predict_fe46,
    predict_fe46_report,
)
from sentinel.params import PROCESSED_DIR


# ── Lifespan: load FE bundle + run cached prediction once at startup ─────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading FE 46ch detrended PCA bundle …")
    fe = load_fe46_artefacts()
    app.state.fe          = fe                      # bundle, spread into predict_fe46(**fe)
    app.state.model       = fe["model"]
    app.state.scaler      = fe["scaler"]
    app.state.features    = fe["features_full"]     # 58ch — POST /predict validates against this
    app.state.features_fe = fe["features_fe"]       # 46ch — FE inference order
    app.state.threshold   = fe["threshold"]

    X_api = np.load(PROCESSED_DIR / "test_api_fe.npy")
    y_api = np.load(PROCESSED_DIR / "y_test_api_fe.npy")
    app.state.y_true = y_api.astype(int).tolist()

    print("⏳ Running cached FE prediction over test_api_fe slice …")
    sub = predict_fe46(X_raw=X_api, **fe)
    app.state.timeline = sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
    print(f"✅ Timeline cached: {len(app.state.timeline):,} rows")

    print("⏳ Computing FE report cache …")
    rep = predict_fe46_report(X_raw=X_api, n_top_channels=6, **fe)
    app.state.report = {
        # Per-row reconstruction MSE (PCA default = mean over the 46 FE channels,
        # then score-level rolling-median detrend, broadcast to rows).
        "row_scores"     : rep["row_scores"].tolist(),

        # Per-channel overall reconstruction errors (MSE), sortable for the
        # per-channel diagnostic panel in the showcase.
        "per_channel_mse": [
            {"channel": ch, "mse": float(mse)}
            for ch, mse in zip(rep["features"], rep["per_channel_mse"])
        ],

        # Indices of the n_top_channels with the largest MSE PER WINDOW,
        # ranked descending. Diagnostic only — does not affect scoring.
        "window_top_channels": rep["window_top_channels"].tolist(),

        # Threshold from the sidecar (val-tuned on corrected_event_f05).
        "threshold"      : rep["threshold"],

        # The 46 FE channels used by the model.
        "features"       : rep["features"],

        # Row-based #anomalies & anomaly rate (post-filter applied).
        "n_anomalies"    : int((rep["labels"] == 1).sum()),
        "anomaly_rate"   : round(float(rep["labels"].mean()), 4),
    }
    print("✅ Report cached")
    app.state.X_api = X_api  # cache raw values for /channels endpoint
    yield


app = FastAPI(
    title="SENTINEL Anomaly Detection",
    description="ESA satellite telemetry anomaly detector (FE 46ch detrended PCA pipeline)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """rows: list of rows, each row is a list of N raw channel values (N = 58)."""
    rows: list[list[float]]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — confirms the API is alive."""
    return {"status": "ok", "message": "SENTINEL Anomaly Detection API is running"}


@app.get("/timeline")
def timeline() -> list[dict]:
    """Cached predictions over the test_api_fe slice. Returns [{id, is_anomaly}]."""
    return app.state.timeline


@app.get("/predict_by_id")
def predict_by_id(start: int, end: int) -> list[dict]:
    """Filter cached timeline by ID range [start, end] inclusive."""
    return [r for r in app.state.timeline if start <= r["id"] <= end]


@app.get("/report")
def report() -> dict:
    """Cached report: row_scores, per_channel_mse, window_top_channels, threshold, features, anomaly_rate."""
    return app.state.report


@app.get("/channels")
def channels(channel: str, start: int = 0, end: int = 149999) -> list[dict]:
    """
    Raw signal values for a single channel over an ID range.
    Returns [{"id": int, "value": float, "is_anomaly": 0|1}].
    """
    if channel not in app.state.features:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown channel '{channel}'. Available: {app.state.features}",
        )

    col_idx = app.state.features.index(channel)
    timeline = {r["id"]: r["is_anomaly"] for r in app.state.timeline}

    result = []
    for i in range(start, min(end + 1, len(app.state.X_api))):
        result.append({
            "id"        : i,
            "value"     : float(app.state.X_api[i, col_idx]),
            "is_anomaly": timeline.get(i, 0),
        })
    return result


@app.get("/features") # --> get channel info (names)
def features() -> list[str]:
    """Returns the list of available channel names."""
    return app.state.features


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> list[dict]:
    """Score user-supplied raw 58-channel rows. Returns [{id, is_anomaly}]."""
    if len(request.rows) == 0:
        raise HTTPException(status_code=400, detail="No rows provided")

    n_feat_expected = len(app.state.features)
    n_feat_got      = len(request.rows[0])
    if n_feat_got != n_feat_expected:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {n_feat_expected} features per row, got {n_feat_got}",
        )

    X_raw = np.array(request.rows, dtype=np.float32)
    sub = predict_fe46(X_raw=X_raw, **app.state.fe)
    return sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# Old pipeline (predict / predict_report on the first PCA model).
# Kept commented out for reference; the FE pipeline above is what the demo
# runs. To revert: re-enable these imports + the Kaggle lifespan + the
# Kaggle POST /predict (and disable the FE block above).
# ─────────────────────────────────────────────────────────────────────────────
#
# from sentinel.ml_logic.data import load_target_channels
# from sentinel.ml_logic.predictor import predict, predict_report
# from sentinel.ml_logic.registry import load_model, load_scaler
# from sentinel.params import PCA_THRESHOLD
#
# @asynccontextmanager
# async def lifespan_kaggle(app: FastAPI):
#     print("⏳ Loading model + scaler …")
#     app.state.model     = load_model("pca")
#     app.state.scaler    = load_scaler()
#     app.state.features  = load_target_channels()
#     app.state.threshold = PCA_THRESHOLD
#
#     X_api = np.load(PROCESSED_DIR / "test_api_fe.npy")
#     y_api = np.load(PROCESSED_DIR / "y_test_api_fe.npy")
#     app.state.y_true = y_api.astype(int).tolist()
#
#     print("⏳ Running cached prediction over test_api slice …")
#     sub = predict(
#         model     = app.state.model,
#         scaler    = app.state.scaler,
#         features  = app.state.features,
#         X_raw     = X_api,
#         threshold = app.state.threshold,
#     )
#     app.state.timeline = sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
#     print(f"✅ Timeline cached: {len(app.state.timeline):,} rows")
#
#     print("⏳ Computing report cache …")
#     rep = predict_report(
#         model     = app.state.model,
#         scaler    = app.state.scaler,
#         features  = app.state.features,
#         X_raw     = X_api,
#         threshold = app.state.threshold,
#         n_top_channels = 6,
#     )
#     app.state.report = {
#         "row_scores"     : rep["row_scores"].tolist(),
#         "per_channel_mse": [
#             {"channel": ch, "mse": float(mse)}
#             for ch, mse in zip(rep["features"], rep["per_channel_mse"])
#         ],
#         "window_top_channels": rep["window_top_channels"].tolist(),
#         "threshold"      : rep["threshold"],
#         "features"       : rep["features"],
#         "n_anomalies"    : int((rep["labels"] == 1).sum()),
#         "anomaly_rate"   : round(float(rep["labels"].mean()), 4),
#     }
#     print("✅ Report cached")
#     yield
#
# # POST /predict (Kaggle pipeline) — replaces the FE version above when reverting.
# # @app.post("/predict")
# # def predict_endpoint_kaggle(request: PredictRequest) -> list[dict]:
# #     if len(request.rows) == 0:
# #         raise HTTPException(status_code=400, detail="No rows provided")
# #     n_feat_expected = len(app.state.features)
# #     n_feat_got      = len(request.rows[0])
# #     if n_feat_got != n_feat_expected:
# #         raise HTTPException(
# #             status_code=400,
# #             detail=f"Expected {n_feat_expected} features per row, got {n_feat_got}",
# #         )
# #     X_raw = np.array(request.rows, dtype=np.float32)
# #     sub = predict(
# #         model     = app.state.model,
# #         scaler    = app.state.scaler,
# #         features  = app.state.features,
# #         X_raw     = X_raw,
# #         threshold = app.state.threshold,
# #     )
# #     return sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
