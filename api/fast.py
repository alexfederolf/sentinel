"""
SENTINEL Anomaly Detection API

Endpoints
---------
GET  /          → health check, confirms API is alive
POST /predict   → full prediction pipeline delegated to predictor.py:
                  raw sensor data → preprocess → model → predictions

Usage
-----
    make run_api
    curl http://localhost:8000/
"""

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sentinel.ml_logic.predictor import predict as run_predict

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SENTINEL Anomaly Detection",
    description="ESA satellite telemetry anomaly detector",
    version="1.0.0",
)

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Raw sensor readings from the user.
    rows: list of rows, each row is a list of 58 channel values

    Example:
    {
        "rows": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
    }
    """
    rows: list[list[float]]


class PredictResponse(BaseModel):
    """
    predictions: 0 = nominal, 1 = anomaly, one per input row
    n_anomalies: total number of flagged rows
    anomaly_rate: fraction of rows flagged
    """
    predictions: list[int]
    n_anomalies: int
    anomaly_rate: float


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — confirms the API is alive."""
    return {
        "status": "ok",
        "message": "SENTINEL Anomaly Detection API is running",
    }


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """
    Full prediction pipeline — delegates to predictor.predict()
    1. Receive raw sensor rows from user
    2. predictor.py handles: preprocess → load model → score → threshold
    3. Return results as a dict
    """
    if len(request.rows) == 0:
        raise HTTPException(status_code=400, detail="No rows provided")

    n_feat = len(request.rows[0])
    if n_feat != 58:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 58 features per row, got {n_feat}"
        )

    X_raw = np.array(request.rows, dtype=np.float32)

    # delegates everything to predictor.py
    result = run_predict(X_raw=X_raw)

    predictions = result["is_anomaly"].tolist()
    n_anomalies = int(sum(predictions))
    anomaly_rate = round(n_anomalies / len(predictions), 4)

    return PredictResponse(
        predictions=predictions,
        n_anomalies=n_anomalies,
        anomaly_rate=anomaly_rate,
    )





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
