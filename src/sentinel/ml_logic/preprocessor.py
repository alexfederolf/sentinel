"""
Preprocessing module for SENTINEL.

Contains the full preprocessing pipeline:
- temporal 80/20 train/val split
- RobustScaler fit on nominal training rows (no leakage)
- sliding-window creation (100-row non-overlapping windows)
- saves processed arrays + scaler + config to disk

Exposed functions:
    create_windows(data, window_size, stride, labels=None)
        Low-level utility: slices a 2D time series into 3D windows.

    run_preprocessing()
        Full pipeline — loads raw data, scales, windows, saves.
        Called by `python -m sentinel.main preprocess`.
"""

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler

from ..params import RANDOM_STATE, TRAIN_RATIO, TRAIN_STRIDE, WINDOW_SIZE
from .data import PROCESSED_DIR, load_target_channels, load_test, load_train

# Models directory is at <repo_root>/models — three levels up from this file
MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def create_windows(
    data: np.ndarray,
    window_size: int,
    stride: int,
    labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Slice a 2D time series into overlapping or non-overlapping 3D windows.

    Parameters
    ----------
    data        : float32 array (n_rows, n_features)
    window_size : number of timesteps per window (e.g. 100)
    stride      : step between consecutive windows — use stride=window_size
                  for non-overlapping windows (faster, less memory)
    labels      : optional int8 array (n_rows,) — if provided, a window is
                  labelled 1 if ANY sample inside it is anomalous (conservative)

    Returns
    -------
    X : float32 array (n_windows, window_size, n_features)
    y : int8 array    (n_windows,)  or None
    """
    n_rows, n_feat = data.shape
    n_windows = (n_rows - window_size) // stride + 1

    X = np.empty((n_windows, window_size, n_feat), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.int8) if labels is not None else None

    for i in range(n_windows):
        start = i * stride
        X[i] = data[start : start + window_size]
        if labels is not None:
            # Window is anomalous if at least one row inside it is anomalous.
            # This is conservative — better to over-flag than miss an event.
            y[i] = int(labels[start : start + window_size].max())

    return X, y


def run_preprocessing() -> None:
    """
    Full preprocessing pipeline.

    Reads data/raw/, writes data/processed/ and models/robust_scaler.pkl.
    Idempotent — safe to re-run.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    print("Loading data …")
    t0 = time.time()
    train = load_train()
    test = load_test()
    target_channels = load_target_channels()  # 58 scored channels from target_channels.csv
    print(f"  loaded in {time.time() - t0:.1f}s")

    N_TRAIN = len(train)
    N_TEST = len(test)
    N_FEAT = len(target_channels)
    N_ANOM = int(train["is_anomaly"].sum())
    N_NOMINAL = N_TRAIN - N_ANOM

    print(f"  train : {N_TRAIN:,} rows  |  test : {N_TEST:,} rows  |  features : {N_FEAT}")
    print(f"  anomaly : {N_ANOM:,} ({N_ANOM / N_TRAIN * 100:.2f}%)  |  nominal : {N_NOMINAL:,}")

    # ── 2. Temporal 80/20 split ───────────────────────────────────────────────
    # Rows are ordered by time — no shuffling to avoid data leakage.
    # First 80% for training, last 20% for validation.
    SPLIT_IDX = int(N_TRAIN * TRAIN_RATIO)
    train_full_df = train                    # kept intact for threshold tuning later
    train_df = train.iloc[:SPLIT_IDX]        # first 80%
    val_df = train.iloc[SPLIT_IDX:]          # last 20%

    # Semi-supervised setup: only NOMINAL rows are used to fit the scaler
    # and the reconstruction model. Anomalous rows are excluded from training
    # so the model learns what "normal" looks like.
    nom_train_df = train_df[train_df["is_anomaly"] == 0]
    nom_val_df = val_df[val_df["is_anomaly"] == 0]

    print(f"\nSplit at row {SPLIT_IDX:,} ({TRAIN_RATIO * 100:.0f}% / {(1 - TRAIN_RATIO) * 100:.0f}%)")
    print(f"  nominal train : {len(nom_train_df):,}  |  nominal val : {len(nom_val_df):,}")

    # ── 3. Fit RobustScaler on nominal training rows only ─────────────────────
    # RobustScaler uses median and IQR instead of mean and std — it is robust
    # to extreme anomaly values and to the non-stationarity of 14-year telemetry.
    # Important: fit ONLY on nominal training rows to avoid data leakage.
    print("\nFitting RobustScaler on nominal training rows …")
    scaler = RobustScaler()
    scaler.fit(nom_train_df[target_channels].values.astype(np.float32))

    scaler_path = MODELS_DIR / "robust_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler saved → {scaler_path}")

    # ── 4. Scale all splits ───────────────────────────────────────────────────
    # All splits are transformed with the same scaler fitted in step 3.
    # float32 halves memory vs float64 with no meaningful precision loss.
    print("\nScaling …")
    nom_train_scaled = scaler.transform(
        nom_train_df[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    nom_val_scaled = scaler.transform(
        nom_val_df[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    train_full_scaled = scaler.transform(
        train_full_df[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    test_scaled = scaler.transform(
        test[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    # Extract labels and IDs before freeing the dataframes
    y_train_full = train_full_df["is_anomaly"].values.astype(np.int8)
    test_ids = test["id"].values.astype(np.int64)

    # Free raw dataframes — no longer needed, reclaim memory
    del train, train_full_df, train_df, val_df, nom_train_df, nom_val_df, test
    gc.collect()

    # ── 5. Create sliding windows ─────────────────────────────────────────────
    # Windows of 100 rows with stride=100 (non-overlapping) give one MSE score
    # per window. This smooths the anomaly signal and avoids the FP-segment
    # explosion that per-row scoring causes (see NB03 for why this matters).
    print(f"\nCreating windows (size={WINDOW_SIZE}, stride={TRAIN_STRIDE}) …")

    print("  X_train_nom …")
    X_train_nom, _ = create_windows(nom_train_scaled, WINDOW_SIZE, TRAIN_STRIDE)

    print("  X_val_nom …")
    X_val_nom, _ = create_windows(nom_val_scaled, WINDOW_SIZE, TRAIN_STRIDE)

    print("  X_train_full …")
    X_train_full, y_train_full_win = create_windows(
        train_full_scaled, WINDOW_SIZE, TRAIN_STRIDE, labels=y_train_full
    )

    # Free intermediate scaled arrays — windows are now in memory
    del nom_train_scaled, nom_val_scaled
    gc.collect()

    # ── 6. Save arrays to disk ────────────────────────────────────────────────
    print("\nSaving arrays …")
    save_manifest = {
        # 3D tensors — feed directly to model.fit()
        "X_train_nom.npy": X_train_nom,
        "X_val_nom.npy": X_val_nom,
        "X_train_full.npy": X_train_full,
        "y_train_full.npy": y_train_full_win,
        # 2D row-level arrays — used for inference and threshold tuning
        "train_full_scaled.npy": train_full_scaled,
        "y_train_row.npy": y_train_full,
        "test_scaled.npy": test_scaled,
        "test_ids.npy": test_ids,
    }

    total_bytes = 0
    for fname, arr in save_manifest.items():
        path = PROCESSED_DIR / fname
        print(f"  {fname:<30} {str(arr.shape):<25} {arr.nbytes / 1e6:>8.1f} MB")
        np.save(path, arr)
        total_bytes += arr.nbytes

    print(f"\n  Total saved : {total_bytes / 1e9:.2f} GB → {PROCESSED_DIR}")

    # ── 7. Save preprocessing config ──────────────────────────────────────────
    # The config records all hyperparameters and shapes so downstream notebooks
    # can load it and know what was produced without re-running this pipeline.
    config = {
        "window_size": WINDOW_SIZE,
        "train_stride": TRAIN_STRIDE,
        "train_ratio": TRAIN_RATIO,
        "split_idx": SPLIT_IDX,
        "n_features": N_FEAT,
        "target_channels": target_channels,
        "n_train_rows": N_TRAIN,
        "n_test_rows": N_TEST,
        "n_anomaly_rows": int(N_ANOM),
        "n_nominal_rows": int(N_NOMINAL),
        "shapes": {k.replace(".npy", ""): list(v.shape) for k, v in save_manifest.items()},
    }
    config_path = PROCESSED_DIR / "preprocessing_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config saved → {config_path}")

    # ── 8. Sanity checks ──────────────────────────────────────────────────────
    print("\nSanity checks …")
    checks = [
        ("X_train_nom",       X_train_nom,       3, np.float32),
        ("X_val_nom",         X_val_nom,          3, np.float32),
        ("X_train_full",      X_train_full,       3, np.float32),
        ("y_train_full",      y_train_full_win,   1, np.int8),
        ("train_full_scaled", train_full_scaled,  2, np.float32),
        ("y_train_row",       y_train_full,       1, np.int8),
        ("test_scaled",       test_scaled,        2, np.float32),
        ("test_ids",          test_ids,           1, np.int64),
    ]

    all_ok = True
    for name, arr, ndim, dtype in checks:
        ok = arr.ndim == ndim and arr.dtype == dtype
        # Only float arrays can have NaN/Inf — integer arrays are always clean
        nan_inf = (
            bool(np.isnan(arr).any() or np.isinf(arr).any())
            if arr.dtype.kind == "f"
            else False
        )
        status = "✓" if ok and not nan_inf else "✗"
        print(f"  {status} {name:<22} shape={str(arr.shape):<25} dtype={arr.dtype}")
        if not ok or nan_inf:
            all_ok = False

    print(f"\n{'All checks passed ✓' if all_ok else 'Some checks FAILED ✗'}")
