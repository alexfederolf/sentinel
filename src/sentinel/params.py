"""
Project-wide constants for SENTINEL

Import with:
    from sentinel.params import WINDOW_SIZE, RANDOM_STATE, ANOMALY_COLOR
"""
from pathlib import Path

# ── Resource Paths ───────────────────────────────────────────────────────────
# params.py lives at src/sentinel/params.py, so the project root is parents[2]
# (params.py → sentinel/ → src/ → <project root>). Note: ml_logic/data.py is one
# level deeper and correctly uses parents[3] for the same project root.
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

MODELS_DIR      = Path(__file__).resolve().parents[2] / "models"
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

# ── API demo slice (inside test_intern) ─────────────────────────────────────
# Low-drift 150k-row window containing one ~1.5k-row anomaly event.
# Used by the web-app API so inference is fast and the demo stays meaningful.
API_SLICE_START = 350_000
API_SLICE_END   = 500_000


# ── Plot colours (consistent across all notebooks) ────────────────────────────
ANOMALY_COLOR = '#e74c3c'
NOMINAL_COLOR = '#2980b9'

# ── Trained model thresholds ──────────────────────────────────────────────────

# Tuned on val set with event-wise F0.5
PCA_THRESHOLD = 0.060404 # tuned on PCA in NB11 with event-wise F0.5 (not ESA)
# PCA_THRESHOLD = 0.053293   # manuelly changes for FE

LSTM_THRESHOLD = 1.323612  # from lstm: LSTM k=16, 35k nominal windows (70 % of 50k)

# Detrended-score thresholds (tuned on detrended val, see NB 20).
# Use with sentinel.ml_logic.scorer.score_windows_detrended. The original
# PCA_THRESHOLD / LSTM_THRESHOLD do NOT apply to detrended scores -- the
# subtraction shifts the score distribution so the tuned value is different.
PCA_DETRENDED_THRESHOLD  = None   # fill after first NB 20 run
LSTM_DETRENDED_THRESHOLD = None
