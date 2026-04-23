"""
Project-wide constants for SENTINEL

Import with:
    from sentinel.params import WINDOW_SIZE, RANDOM_STATE, ANOMALY_COLOR
"""
from pathlib import Path

# ── Resource Paths ───────────────────────────────────────────────────────────
DATA_DIR        = Path(__file__).resolve().parents[3] / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = Path(__file__).resolve().parents[3] / "models"
#SUBMISSIONS_DIR = Path(__file__).resolve().parents[3] / "submissions"


# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE  = 42

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.80          # temporal split fraction

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 100           # rows per window
TRAIN_STRIDE  = 100           # stride for pre-computed nominal windows (non-overlapping)

# ── Model fitting ─────────────────────────────────────────────────────────────

# FIT_SIZE is the fit-sample parameter
# Any model accepts fit_size=FIT_SIZE or None (fit on all available data).
FIT_SIZE            =  50_000  # fit sample for investigation
#FIT_SIZE            =  None # use all available data(X_train_nom) for fitting

SCORE_BATCH         = 500_000  # rows scored per batch during inference
CV_FOLDS            = 5        # folds for temporal cross-validation

# ── Default three-way labeled split (train.parquet only) ────────────────────
#70 % train / 15 % val / 15 % test_intern, all labeled
BOOTCAMP_TRAIN_RATIO = 0.70
BOOTCAMP_VAL_RATIO   = 0.15    # test_intern = 1 − TRAIN_RATIO − VAL_RATIO = 0.15


# ── Plot colours (consistent across all notebooks) ────────────────────────────
ANOMALY_COLOR = '#e74c3c'
NOMINAL_COLOR = '#2980b9'

# ── Trained model thresholds ──────────────────────────────────────────────────
# Tuned on val set with event-wise F0.5
PCA_THRESHOLD = 0.060219   # from pca-full: PCA k=39, all 92k nominal windows
PCA_THRESHOLD = 0.060404   # from pca: PCA k=39, 35k nominal windows (70 % of 50k)
