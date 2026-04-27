"""
SENTINEL Anomaly Detection API

Endpoints
---------
GET  /          → health check
GET  /timeline  → cached labels + ground truth for the test_api slice
POST /predict   → score user-supplied rows using the cached model + scaler

The model, scaler, target channels, and the internal test_api slice are loaded
once at startup. /timeline returns a precomputed prediction so the frontend
renders instantly; /predict reuses the same cached model for ad-hoc inputs.

Usage
-----
    make run_api
    curl http://localhost:8000/
    curl http://localhost:8000/timeline
"""

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.ml_logic.data import load_target_channels
from sentinel.ml_logic.predictor import predict
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

    print("⏳ Running cached prediction over test_api slice …")
    X_api = np.load(PROCESSED_DIR / "test_api.npy")
    y_api = np.load(PROCESSED_DIR / "y_test_api.npy")
    sub = predict(
        model    = app.state.model,
        scaler   = app.state.scaler,
        features = app.state.features,
        X_raw    = X_api,
        threshold= app.state.threshold,
    )
    app.state.timeline = sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")
    print(f"✅ Ready: {len(app.state.timeline):,} rows cached")
    yield
    # Nothing to tear down.


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
    return {
        "status": "ok",
        "message": "SENTINEL Anomaly Detection API is running",
    }


@app.get("/timeline")
def timeline() -> list[dict]:
    """
    Precomputed prediction over the internal test_api slice.
    Returns [{"id": int, "is_anomaly": 0|1}, ...].
    """
    return app.state.timeline


@app.post("/predict")
def predict_endpoint(request: PredictRequest) -> list[dict]:
    """
    Returns [{"id": int, "is_anomaly": 0|1}].
    """
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
        model    = app.state.model,
        scaler   = app.state.scaler,
        features = app.state.features,
        X_raw    = X_raw,
        threshold= app.state.threshold,
    )
    return sub.astype({"id": int, "is_anomaly": int}).to_dict(orient="records")


@app.get("/predict_by_id")
def predict_endpoint_by_id(start: int, end: int) -> list[dict]:
    """
    Filter the cached timeline by ID range [start, end] inclusive.
    Returns [{"id": int, "is_anomaly": 0|1}].
    """
    return [r for r in app.state.timeline if start <= r["id"] <= end]


@app.get("/report")
def report_endpoint(topk: int = 6) -> dict:
    """
    Full anomaly report over the cached test_api slice.
    Returns row_scores, per_channel_mse, topk_channels, threshold, features.
    """
    from sentinel.ml_logic.predictor import predict_report

    X_api = np.load(PROCESSED_DIR / "test_api.npy")
    rep = predict_report(
        model     = app.state.model,
        scaler    = app.state.scaler,
        features  = app.state.features,
        X_raw     = X_api,
        threshold = app.state.threshold,
        topk      = topk,
    )
    return {
        "row_scores"      : rep["row_scores"].tolist(),
        "per_channel_mse" : rep["per_channel_mse"].tolist(),
        "topk_channels"   : rep["topk_channels"].tolist() if rep["topk_channels"] is not None else None,
        "threshold"       : rep["threshold"],
        "features"        : rep["features"],
        "n_anomalies"     : int((rep["labels"] == 1).sum()),
        "anomaly_rate"    : round(float(rep["labels"].mean()), 4),
    }



## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------
### OLD STRCUTURE OF API, NOW DELETED
## ---------------------------------------------------------------------------------
## ---------------------------------------------------------------------------------

# """
# SENTINEL Anomaly Detection API

# Endpoints
# ---------
# GET  /          → health check, confirms API is alive
# POST /predict   → full prediction pipeline:
#                   raw sensor data → preprocess → model → predictions

# Usage
# -----
#     make run_api
#     curl http://localhost:8000/
# """

# import numpy as np
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel

# from sentinel.ml_logic.registry import load_model, load_scaler
# from sentinel.ml_logic.scorer import score_windows
# from sentinel.params import WINDOW_SIZE

# # ── App ───────────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="SENTINEL Anomaly Detection",
#     description="ESA satellite telemetry anomaly detector",
#     version="1.0.0",
# )

# # ── Load model and scaler once at startup ────────────────────────────────────
# # Loading at startup means the first request isn't slow
# # Both are kept in memory for the lifetime of the API process
# @app.on_event("startup")
# def load_resources():
#     app.state.model  = load_model("pca")
#     app.state.scaler = load_scaler()
#     print("✅ Model and scaler ready")


# # ── Request / Response schemas ────────────────────────────────────────────────
# class PredictRequest(BaseModel):
#     """
#     Raw sensor readings from the user.
#     rows: list of rows, each row is a list of 58 channel values
#     threshold: anomaly score cutoff (optional, default 0.5)

#     Example:
#     {
#         "rows": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
#         "threshold": 0.5
#     }
#     """
#     rows: list[list[float]]
#     threshold: float = 0.5


# class PredictResponse(BaseModel):
#     """
#     Prediction results returned to the user.
#     predictions: 0 = nominal, 1 = anomaly, one per input row
#     n_anomalies: total number of flagged rows
#     anomaly_rate: fraction of rows flagged
#     """
#     predictions: list[int]
#     n_anomalies: int
#     anomaly_rate: float


# # ── Endpoints ─────────────────────────────────────────────────────────────────
# @app.get("/")
# def root():
#     """Health check — confirms the API is alive."""
#     return {
#         "status": "ok",
#         "message": "SENTINEL Anomaly Detection API is running",
#     }


# @app.post("/predict", response_model=PredictResponse) # --> leave this
# def predict(request: PredictRequest): # --> chamge this def name (tno to clash to eachother)
#     """
#     Full prediction pipeline:
#     1. Receive raw sensor rows from user
#     2. Preprocess with the fitted RobustScaler
#     3. Load PCA model (already in memory)
#     4. Score windows → per-row anomaly scores
#     5. Apply threshold → binary predictions
#     6. Return results as a dict
#     """
#     # ── 1. Validate input ─────────────────────────────────────────────────────
#     if len(request.rows) == 0:
#         raise HTTPException(status_code=400, detail="No rows provided")

#     n_feat = len(request.rows[0])
#     if n_feat != 58:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Expected 58 features per row, got {n_feat}"
#         )

#     # ── 2. Preprocess — apply the same RobustScaler used in training ──────────
#     X_raw    = np.array(request.rows, dtype=np.float32)
#     X_scaled = app.state.scaler.transform(X_raw).astype(np.float32)

#     # ── 3 & 4. Score windows using the loaded PCA model ──────────────────────
#     scores = score_windows(
#         model=app.state.model,
#         X_rows=X_scaled,
#         win=WINDOW_SIZE,
#     )

#     # ── 5. Apply threshold → binary predictions ───────────────────────────────
#     predictions = (scores > request.threshold).astype(int).tolist()

#     # ── 6. Return results ─────────────────────────────────────────────────────
#     n_anomalies = int(sum(predictions))
#     anomaly_rate = round(n_anomalies / len(predictions), 4)

#     return PredictResponse(
#         predictions=predictions,
#         n_anomalies=n_anomalies,
#         anomaly_rate=anomaly_rate,
#     )


# ## create end point predict from predictor
# # change name of predictor // import def predictor
