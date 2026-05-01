"""Core building blocks for SENTINEL.

Submodules
----------
data         loaders + path constants (DATA_DIR, RAW_DIR, ...)
metrics      event-wise + ESA-corrected F-scores
preprocessor 3-way and Kaggle pipelines
scorer       window-mean MSE + detrending
predictor    inference façade (PCA / LSTM-AE / FE 46ch)
thresholds   sweep + tune
validation   bootstrap confidence intervals
cv           cross-validation harness (also runnable as ``python -m sentinel.ml_logic.cv``)
fusion       multi-model fusion diagnostics
viz          plotting helpers
ek_data      personal data utilities (parallel to ``data``, used by ek_* notebooks)

Import what you need explicitly, e.g. ``from sentinel.ml_logic.scorer
import score_report``. The package __init__ is intentionally empty so
heavy submodules (``viz`` → matplotlib, ``predictor`` → tensorflow) only
load when actually used.
"""
