"""
Project-wide constants for SENTINEL.

Import with:
    from sentinel.params import WINDOW_SIZE, RANDOM_STATE, ANOMALY_COLOR
"""

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE  = 42

# ── Data split ────────────────────────────────────────────────────────────────
TRAIN_RATIO   = 0.80          # temporal split fraction

# ── Windowing ─────────────────────────────────────────────────────────────────
WINDOW_SIZE   = 100           # rows per window
TRAIN_STRIDE  = 100           # stride for pre-computed nominal windows (non-overlapping)

# ── Model fitting ─────────────────────────────────────────────────────────────
IFOREST_FIT_SAMPLES = 500_000  # training rows used to fit IsolationForest (NB03)
PCA_FIT_SAMPLES     =  50_000  # training windows used to fit PCA (NB04)
SCORE_BATCH         = 500_000  # rows scored per batch during inference
CV_FOLDS            = 5        # folds for temporal cross-validation

# ── Plot colours (consistent across all notebooks) ────────────────────────────
ANOMALY_COLOR = '#e74c3c'
NOMINAL_COLOR = '#2980b9'
