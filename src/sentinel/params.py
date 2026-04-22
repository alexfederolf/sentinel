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
# FIT_SIZE is the unified fit-sample parameter for the bootcamp notebooks
# (11_pca, 12_lstm_ae, 13_cnn_ae). Any model accepts fit_size=FIT_SIZE or
# None (fit on all available data). The legacy constants below preserve
# byte-identical behaviour for the old Kaggle-baseline notebooks (03, 04).
FIT_SIZE            =  50_000  # unified fit sample for PCA / IForest / LSTM-AE / CNN-AE
IFOREST_FIT_SAMPLES = 500_000  # legacy: NB 03 training rows for IsolationForest
PCA_FIT_SAMPLES     =  50_000  # legacy: NB 04 training windows for PCA
SCORE_BATCH         = 500_000  # rows scored per batch during inference
CV_FOLDS            = 5        # folds for temporal cross-validation

# ── Bootcamp three-way labeled split (train.parquet only) ────────────────────
# A local "private leaderboard": 70 % train / 15 % val / 15 % test_intern,
# all labeled. Used by run_preprocessing_bootcamp() in preprocessor.py and
# the 10-series notebooks. The Kaggle test.parquet is NOT touched by this
# split — Kaggle submissions still go through the 80/20 Kaggle pipeline.
BOOTCAMP_TRAIN_RATIO = 0.70
BOOTCAMP_VAL_RATIO   = 0.15    # test_intern = 1 − TRAIN_RATIO − VAL_RATIO = 0.15

# ── LSTM Autoencoder hyperparameters ─────────────────────────────────────────
LATENT_DIM    = 32             # LSTM bottleneck dimension
HIDDEN_DIM    = 64             # LSTM encoder/decoder hidden dimension
DROPOUT       = 0.2            # LSTM dropout (standard, not recurrent_dropout)

# ── Plot colours (consistent across all notebooks) ────────────────────────────
ANOMALY_COLOR = '#e74c3c'
NOMINAL_COLOR = '#2980b9'
