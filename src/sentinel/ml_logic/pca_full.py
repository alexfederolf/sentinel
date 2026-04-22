"""
PCA Full — end-to-end submission generator.

Designed to be called from the CLI after
preprocessing (`python -m sentinel.main preprocess`) has been run.

Pipeline
--------
1. Load processed arrays from `data/processed/`.
2. Fit PCA on **all** nominal training windows (no subsampling).
3. Pick `k` components for 95% cumulative explained variance, refit.
4. Score the val set (row-level) via window-mean MSE.
5. Sweep thresholds, pick the one that maximises event-wise F0.5 on val.
6. Score the test set, apply the threshold, write the submission parquet.

Artifacts written
-----------------
- `models/pca_full.pkl`              — fitted PCA model
- `submissions/pca_full.parquet`     — submission (id, is_anomaly)

Usage
-----
    python -m sentinel.main pca_full     # standard entry point
    python -m sentinel.ml_logic.pca_full # direct alternative

Preconditions
-------------
- `python -m sentinel.main preprocess` has been run, so
  `data/processed/` contains X_train_nom.npy, train_full_scaled.npy,
  y_train_row.npy, test_scaled.npy, test_ids.npy and
  preprocessing_config.json.
"""

import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..params import RANDOM_STATE
from .data import MODELS_DIR, PROCESSED_DIR, SUBMISSIONS_DIR
from .metrics import corrected_event_f05, f05_score

# ── Hyperparameters ───────────────────────────────────────────────────────────
VAR_TARGET    = 0.95   # cumulative explained variance for choosing k
BATCH_WINDOWS = 5000   # windows scored per PCA batch (bounds memory)
N_SWEEP       = 60     # number of thresholds in the F0.5 sweep


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def score_windows(model: PCA, X_rows: np.ndarray, win: int,
               batch_windows: int = BATCH_WINDOWS) -> np.ndarray:
    """
    Row-level anomaly scores from window-level PCA reconstruction MSE.

    Slides non-overlapping `win`-row windows over X_rows, flattens each to
    `(win * n_feat,)`, computes MSE between the window and its PCA
    reconstruction, then broadcasts each window score back to all `win`
    rows in that window. Trailing rows (< win) inherit the last full
    window's score.

    Parameters
    ----------
    model         : fitted sklearn PCA
    X_rows        : float32 array (n_rows, n_features)
    win           : window size in rows
    batch_windows : windows per PCA batch (memory bound)

    Returns
    -------
    float32 array (n_rows,) — anomaly score per row
    """
    N, n_feat  = X_rows.shape
    n_complete = N // win
    flat_dim   = win * n_feat

    # 1) MSE per window, in batches.
    win_scores = np.empty(n_complete, dtype=np.float32)
    for b0 in range(0, n_complete, batch_windows):
        b1 = min(b0 + batch_windows, n_complete)
        batch = np.stack([
            X_rows[i*win:(i+1)*win].reshape(flat_dim) for i in range(b0, b1)
        ])
        rec = model.inverse_transform(model.transform(batch))
        win_scores[b0:b1] = ((batch - rec) ** 2).mean(axis=1)

    # 2) Broadcast window → row.
    row_scores = np.empty(N, dtype=np.float32)
    for i in range(n_complete):
        row_scores[i*win:(i+1)*win] = win_scores[i]
    if n_complete * win < N and n_complete > 0:
        row_scores[n_complete*win:] = win_scores[-1]
    return row_scores


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline stages
# ══════════════════════════════════════════════════════════════════════════════

def load_arrays():
    """Load preprocessed arrays + config. Returns all tensors needed downstream."""
    with open(PROCESSED_DIR / "preprocessing_config.json") as f:
        cfg = json.load(f)

    split_idx = cfg["split_idx"]
    win       = cfg["window_size"]
    n_feat    = cfg["n_features"]

    print("Loading arrays …")
    t0 = time.time()
    X_train_nom = np.load(PROCESSED_DIR / "X_train_nom.npy")
    X_all       = np.load(PROCESSED_DIR / "train_full_scaled.npy")
    y_all       = np.load(PROCESSED_DIR / "y_train_row.npy")
    X_test      = np.load(PROCESSED_DIR / "test_scaled.npy")
    test_ids    = np.load(PROCESSED_DIR / "test_ids.npy")

    # Val = last 20% of the temporal split.
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]
    del X_all, y_all
    gc.collect()

    print(f"  loaded in {time.time() - t0:.1f}s")
    print(f"  X_train_nom : {X_train_nom.shape}  ← fit on ALL of this")
    print(f"  X_val       : {X_val.shape}  ({int(y_val.sum()):,} anomalous rows)")
    print(f"  X_test      : {X_test.shape}")

    return X_train_nom, X_val, y_val, X_test, test_ids, win, n_feat


def fit_pca(X_train_nom: np.ndarray, win: int, n_feat: int) -> PCA:
    """
    Two-pass fit: probe full PCA to pick k for VAR_TARGET variance, then refit.
    Uses every nominal training window, no subsampling.
    """
    win_flat = win * n_feat
    X_fit = X_train_nom.reshape(len(X_train_nom), win_flat)
    print(f"\nX_fit: {X_fit.shape}  ({X_fit.nbytes / 1e9:.2f} GB)")

    # Pass 1: full PCA to read off component count.
    print(f"\nFitting probe PCA on {X_fit.shape[0]:,} nominal windows …")
    t0 = time.time()
    pca_probe = PCA(n_components=None, random_state=RANDOM_STATE)
    pca_probe.fit(X_fit)
    cum_var = np.cumsum(pca_probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum_var, VAR_TARGET)) + 1
    print(f"  probe fit in {time.time() - t0:.1f}s  →  k = {k} for {VAR_TARGET*100:.0f}% variance")

    # Pass 2: refit at chosen k.
    print(f"\nRefitting PCA with n_components={k} …")
    t0 = time.time()
    pca = PCA(n_components=k, random_state=RANDOM_STATE)
    pca.fit(X_fit)
    print(f"  refit in {time.time() - t0:.1f}s  "
          f"(explained var: {pca.explained_variance_ratio_.sum()*100:.2f}%)")

    # Persist.
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "pca_full.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pca, f)
    print(f"  saved → {model_path}")

    del X_fit
    gc.collect()
    return pca


def tune_threshold(pca: PCA, X_val: np.ndarray, y_val: np.ndarray, win: int) -> float:
    """
    Score the val set, sweep N_SWEEP thresholds across the transition region
    between the nominal and anomaly MSE distributions, pick the threshold
    with the highest event-wise F0.5.

    Notes
    -----
    - Thresholds are **log-spaced** (geomspace) because MSE distributions are
      strongly right-skewed; linear spacing wastes grid points on the tail.
    - Upper bound is the 99th percentile of anomaly MSE, not the max — a
      single extreme outlier can otherwise stretch a linear grid so wide
      that the optimum falls on a threshold which exceeds all test scores,
      flagging nothing on Kaggle.
    """
    print(f"\nScoring val ({len(X_val):,} rows) …")
    t0 = time.time()
    scores_val = score_windows(pca, X_val, win=win)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  nominal mean MSE : {scores_val[y_val == 0].mean():.5f}")
    print(f"  anomaly mean MSE : {scores_val[y_val == 1].mean():.5f}")

    # Search range: transition region between nominal and anomaly distributions.
    # Clipping hi at p99 anomaly guards against outlier-driven range explosion.
    lo = float(np.percentile(scores_val[y_val == 0], 50))
    hi = float(np.percentile(scores_val[y_val == 1], 99))
    thresholds = np.geomspace(lo, hi, N_SWEEP)

    print(f"\nSweeping {N_SWEEP} log-spaced thresholds in [{lo:.5f}, {hi:.5f}] …")
    t0 = time.time()
    f05s = np.array([
        f05_score(y_val, (scores_val > t).astype(np.int8)) for t in thresholds
    ])
    best_i    = int(np.argmax(f05s))
    threshold = float(thresholds[best_i])
    print(f"  swept in {time.time() - t0:.1f}s")
    print(f"\nBest threshold : {threshold:.5f}  →  val F0.5 = {f05s[best_i]:.4f}")

    # Full breakdown.
    y_pred_val = (scores_val > threshold).astype(np.int8)
    report = corrected_event_f05(y_val, y_pred_val)
    print("\nVal metrics @ best threshold:")
    for k, v in report.items():
        print(f"  {k:<16} {v}")

    return threshold


def write_submission(pca: PCA, X_test: np.ndarray, test_ids: np.ndarray,
                     threshold: float, win: int) -> None:
    """Score test rows, apply threshold, write parquet."""
    print(f"\nScoring test ({len(X_test):,} rows) …")
    t0 = time.time()
    scores_test = score_windows(pca, X_test, win=win)
    print(f"  done in {time.time() - t0:.1f}s")

    y_test = (scores_test > threshold).astype(np.int8)
    n_flag = int(y_test.sum())
    print(f"  flagged: {n_flag:,} / {len(y_test):,} rows  "
          f"({n_flag/len(y_test)*100:.2f}%)")

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    out_path = SUBMISSIONS_DIR / "pca_full.parquet"
    pd.DataFrame({"id": test_ids, "is_anomaly": y_test}).to_parquet(
        out_path, index=False
    )
    print(f"\nSubmission → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_pca_full() -> None:
    """Full PCA submission pipeline — load → fit → tune → score test → write."""
    X_train_nom, X_val, y_val, X_test, test_ids, win, n_feat = load_arrays()
    pca       = fit_pca(X_train_nom, win, n_feat)
    del X_train_nom
    gc.collect()
    threshold = tune_threshold(pca, X_val, y_val, win)
    write_submission(pca, X_test, test_ids, threshold, win)


if __name__ == "__main__":
    run_pca_full()
