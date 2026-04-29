"""
SENTINEL Anomaly Detection API

Endpoints
---------
GET  /               → health check
GET  /timeline       → cached labels for the test_api slice
GET  /predict_by_id  → filter cached timeline by ID range
GET  /report         → cached anomaly report (scores, per-channel MSE, top channels per window, threshold)
POST /predict        → score user-supplied rows using the cached model + scaler

All heavy computation runs once at startup and is cached in app.state.
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.ml_logic.data import load_target_channels
from sentinel.ml_logic.predictor import predict, predict_report
from sentinel.ml_logic.registry import load_model, load_scaler
from sentinel.params import PCA_THRESHOLD, PROCESSED_DIR


# ── Lifespan: load everything once at startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Loading model + scaler …")
    app.state.model     = load_model("pca")
    app.state.scaler    = load_scaler()
    app.state.features  = load_target_channels()
    app.state.threshold = PCA_THRESHOLD

    X_api = np.load(PROCESSED_DIR / "test_api_2.npy")

    print("⏳ Running cached prediction over test_api slice …")
    sub = predict(
        model     = app.state.model,
        scaler    = app.state.scaler,
        features  = app.state.features,
        X_raw     = X_api,
        threshold = app.state.threshold,
    )
    app.state.timeline = sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
    print(f"✅ Timeline cached: {len(app.state.timeline):,} rows")

    print("⏳ Computing report cache …")


    rep = predict_report(
        model     = app.state.model,
        scaler    = app.state.scaler,
        features  = app.state.features,
        X_raw     = X_api,
        threshold = app.state.threshold,
        # topk = 6      for LSTM/CNN only (changes scoring!)
        n_top_channels = 6,   # diagnostic top contributing channels per WINDOW (does not change scoring)
    )
    app.state.report = {
        # Per-row reconstruction MSE (PCA default = mean over all used channels)
        "row_scores"     : rep["row_scores"].tolist(),

        # per-channel overall reconstruction errors (MSE)
        # sortable by most contributing channel for anomaly detection
        "per_channel_mse": [
            {"channel": ch, "mse": float(mse)}
            for ch, mse in zip(rep["features"], rep["per_channel_mse"])
        ],

        # TODO: to decide if needed. Other window metrics not used: window_scores, window_channel_mse
        # OLD naming: topk_channels
        # Indices of the n_top_channels with the largest MSE PER WINDOW, ranked descending
        "window_top_channels": rep["window_top_channels"].tolist(),

        # threshold set by threshold tuning on val set with relevant metric
        "threshold"      : rep["threshold"],
        # TODO: metric used for tuning (current event F0.5, which makes no sense for current FE)

        # list of all channels used in the model
        "features"       : rep["features"],

        # row-based #anomalies & anomaly rate
        "n_anomalies"    : int((rep["labels"] == 1).sum()),
        "anomaly_rate"   : round(float(rep["labels"].mean()), 4),
    }
    print("✅ Report cached")
    app.state.X_api = X_api  # cache raw values for /channels endpoint
    yield


app = FastAPI(
    title="SENTINEL Anomaly Detection",
    description="ESA satellite telemetry anomaly detector",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """rows: list of rows, each row is a list of N channel values (N = 58)."""
    rows: list[list[float]]


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — confirms the API is alive."""
    return {"status": "ok", "message": "SENTINEL Anomaly Detection API is running"}


@app.get("/timeline")
def timeline() -> list[dict]:
    """Cached predictions over the test_api slice. Returns [{id, is_anomaly}]."""
    return app.state.timeline


@app.get("/predict_by_id")
def predict_by_id(start: int, end: int) -> list[dict]:
    """Filter cached timeline by ID range [start, end] inclusive."""
    return [r for r in app.state.timeline if start <= r["id"] <= end]


@app.get("/report")
def report() -> dict:
    """Cached report: row_scores, per_channel_mse (named), window_top_channels, threshold, features, anomaly_rate."""
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
    """Score user-supplied rows. Returns [{id, is_anomaly}]."""
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
    sub = predict(
        model     = app.state.model,
        scaler    = app.state.scaler,
        features  = app.state.features,
        X_raw     = X_raw,
        threshold = app.state.threshold,
    )
    return sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
