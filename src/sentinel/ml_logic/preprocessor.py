"""
Preprocessing module

Two pipelines:

- ``run_preprocessing()`` — three-way labelled chronological split
  (70 / 15 / 15 train / val / test_intern) used by all current notebooks.
  Writes directly to ``data/processed/``.

- ``run_preprocessing_kaggle()`` — original Kaggle two-way split
  (80 / 20 train / val + Kaggle test set) used to build a competition
  submission. Writes to ``data/processed/kaggle/``.

Exposed functions:
    create_windows(data, window_size, stride, labels=None)
        Low-level utility: slices a 2D time series into 3D windows.

    run_preprocessing()
        Default pipeline (formerly ``run_preprocessing_bootcamp``).
        Called by ``python -m sentinel.main preprocess``.

    run_preprocessing_kaggle()
        Kaggle submission pipeline
"""

import gc
import json
import pickle
import time

import numpy as np
from sklearn.preprocessing import RobustScaler

from ..params import (
    API_SLICE_END,
    API_SLICE_START,
    BOOTCAMP_TRAIN_RATIO,
    BOOTCAMP_VAL_RATIO,
    RANDOM_STATE,
    TRAIN_RATIO,
    TRAIN_STRIDE,
    WINDOW_SIZE,
)
from .data import MODELS_DIR, PROCESSED_DIR, load_target_channels, load_test, load_train


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


# ══════════════════════════════════════════════════════════════════════════════
# Default pipeline — three-way labelled split
# ══════════════════════════════════════════════════════════════════════════════

def _snap_to_nominal(labels: np.ndarray, target: int) -> int:
    """
    Advance `target` until labels[target] == 0, so the split lies inside a
    nominal stretch and no anomaly segment is cut across splits.
    Returns the snapped index (or len(labels) if no nominal row is found
    after `target`, which should never happen in practice).
    """
    t = int(target)
    N = len(labels)
    while t < N and labels[t] == 1:
        t += 1
    return t


def run_preprocessing(
    train_ratio: float = BOOTCAMP_TRAIN_RATIO,
    val_ratio: float   = BOOTCAMP_VAL_RATIO,
) -> None:
    """
    Three-way chronological labeled split of train.parquet:
    70 % train / 15 % val / 15 % test_intern (defaults), everything is labeled

    Cuts are snapped forward to the nearest nominal row so no true event is
    sliced across splits. RobustScaler is fit on **nominal training rows
    only** (70 % slice, `is_anomaly == 0`) to avoid leakage.

    Artifacts are written directly to `data/processed/`.

    Parameters
    ----------
    train_ratio, val_ratio : float

    Writes
    ------
    data/processed/
        X_train_nom.npy          (3D, nominal training windows — fit target)
        train_scaled.npy         (2D, all scaled training rows — for windowing)
        y_train.npy              (1D, int8 labels)
        val_scaled.npy           (2D)
        y_val.npy                (1D)
        test_intern_scaled.npy   (2D)
        y_test_intern.npy        (1D)
        preprocessing_config.json (indices, shapes, ratios)
    """
    assert 0.0 < train_ratio < 1.0, f"train_ratio out of range: {train_ratio}"
    assert 0.0 < val_ratio   < 1.0, f"val_ratio out of range: {val_ratio}"
    assert train_ratio + val_ratio < 1.0, (
        f"train+val must leave room for test_intern, got {train_ratio + val_ratio}"
    )

    OUT_DIR = PROCESSED_DIR
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load raw data (train only — no Kaggle test) ────────────────────────
    print("Loading train.parquet …")
    t0 = time.time()
    train = load_train()
    target_channels = load_target_channels()
    print(f"  loaded in {time.time() - t0:.1f}s")

    N       = len(train)
    N_FEAT  = len(target_channels)
    labels  = train["is_anomaly"].values.astype(np.int8)

    # ── 2. Snap split indices to nominal rows ─────────────────────────────────
    raw_split_train = int(N * train_ratio)
    raw_split_val   = int(N * (train_ratio + val_ratio))
    split_train = _snap_to_nominal(labels, raw_split_train)
    split_val   = _snap_to_nominal(labels, raw_split_val)

    print(f"\nSplit targets: train={raw_split_train:,}  val={raw_split_val:,}")
    print(f"Snapped:       train={split_train:,}  val={split_val:,}  "
          f"(drift {split_train-raw_split_train:+d} / {split_val-raw_split_val:+d})")

    train_slice        = train.iloc[:split_train]
    val_slice          = train.iloc[split_train:split_val]
    test_intern_slice  = train.iloc[split_val:]

    y_train       = labels[:split_train]
    y_val         = labels[split_train:split_val]
    y_test_intern = labels[split_val:]

    nom_train_slice = train_slice[train_slice["is_anomaly"] == 0]

    print(f"\nSlices:")
    print(f"  train        : {len(train_slice):>12,} rows  "
          f"({int(y_train.sum()):>7,} anom)")
    print(f"  val          : {len(val_slice):>12,} rows  "
          f"({int(y_val.sum()):>7,} anom)")
    print(f"  test_intern  : {len(test_intern_slice):>12,} rows  "
          f"({int(y_test_intern.sum()):>7,} anom)")
    print(f"  nominal train (fit pool): {len(nom_train_slice):,} rows")

    # ── 3. Fit RobustScaler on nominal training rows only ─────────────────────
    print("\nFitting RobustScaler on nominal training rows …")
    scaler = RobustScaler()
    scaler.fit(nom_train_slice[target_channels].values.astype(np.float32))

    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  scaler saved → {scaler_path}")

    # ── 4. Scale the three splits ─────────────────────────────────────────────
    print("\nScaling splits …")
    train_scaled = scaler.transform(
        train_slice[target_channels].values.astype(np.float32)
    ).astype(np.float32)
    val_scaled = scaler.transform(
        val_slice[target_channels].values.astype(np.float32)
    ).astype(np.float32)
    test_intern_scaled = scaler.transform(
        test_intern_slice[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    nom_train_scaled = scaler.transform(
        nom_train_slice[target_channels].values.astype(np.float32)
    ).astype(np.float32)

    test_intern_raw = test_intern_slice[target_channels].values.astype(np.float32)

    # ── API demo slice: small low-drift window carved out of test_intern ──────
    api_start = min(API_SLICE_START, len(test_intern_raw))
    api_end   = min(API_SLICE_END,   len(test_intern_raw))
    test_api  = test_intern_raw[api_start:api_end]
    y_test_api = y_test_intern[api_start:api_end]
    print(f"\nAPI slice: rows [{api_start:,} : {api_end:,}]  "
          f"({len(test_api):,} rows, {int(y_test_api.sum()):,} anom)")

    del train, train_slice, val_slice, test_intern_slice, nom_train_slice
    gc.collect()

    # ── 5. Build nominal-training windows for model fitting ──────────────────
    print(f"\nCreating X_train_nom windows (size={WINDOW_SIZE}, stride={TRAIN_STRIDE}) …")
    X_train_nom, _ = create_windows(nom_train_scaled, WINDOW_SIZE, TRAIN_STRIDE)
    del nom_train_scaled
    gc.collect()

    # ── 6. Save arrays ────────────────────────────────────────────────────────
    print("\nSaving arrays …")
    save_manifest = {
        "X_train_nom.npy"        : X_train_nom,
        "train_scaled.npy"       : train_scaled,
        "y_train.npy"            : y_train,
        "val_scaled.npy"         : val_scaled,
        "y_val.npy"              : y_val,
        "test_intern_scaled.npy" : test_intern_scaled,
        "test_intern_raw.npy"    : test_intern_raw,
        "y_test_intern.npy"      : y_test_intern,
        "test_api.npy"           : test_api,
        "y_test_api.npy"         : y_test_api,
    }
    total = 0
    for fname, arr in save_manifest.items():
        path = OUT_DIR / fname
        print(f"  {fname:<26} {str(arr.shape):<22} {arr.nbytes / 1e6:>8.1f} MB")
        np.save(path, arr)
        total += arr.nbytes
    print(f"\n  Total saved : {total / 1e9:.2f} GB → {OUT_DIR}")

    # ── 7. Save preprocessing config ──────────────────────────────────────────
    config = {
        "window_size"          : WINDOW_SIZE,
        "train_stride"         : TRAIN_STRIDE,
        "train_ratio"          : train_ratio,
        "val_ratio"            : val_ratio,
        "split_train_idx"      : int(split_train),
        "split_val_idx"        : int(split_val),
        "n_features"           : N_FEAT,
        "target_channels"      : target_channels,
        "n_train_rows"         : int(len(y_train)),
        "n_val_rows"           : int(len(y_val)),
        "n_test_intern_rows"   : int(len(y_test_intern)),
        "n_train_anom"         : int(y_train.sum()),
        "n_val_anom"           : int(y_val.sum()),
        "n_test_intern_anom"   : int(y_test_intern.sum()),
        "api_slice_start"      : int(api_start),
        "api_slice_end"        : int(api_end),
        "n_test_api_rows"      : int(len(y_test_api)),
        "n_test_api_anom"      : int(y_test_api.sum()),
        "shapes"               : {k.replace(".npy", ""): list(v.shape)
                                  for k, v in save_manifest.items()},
    }
    config_path = OUT_DIR / "preprocessing_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  config saved → {config_path}")

    # ── 8. Sanity checks ──────────────────────────────────────────────────────
    print("\nSanity checks …")
    checks = [
        ("X_train_nom",        X_train_nom,        3, np.float32),
        ("train_scaled",       train_scaled,       2, np.float32),
        ("y_train",            y_train,            1, np.int8),
        ("val_scaled",         val_scaled,         2, np.float32),
        ("y_val",              y_val,              1, np.int8),
        ("test_intern_scaled", test_intern_scaled, 2, np.float32),
        ("y_test_intern",      y_test_intern,      1, np.int8),
        ("test_api",           test_api,           2, np.float32),
        ("y_test_api",         y_test_api,         1, np.int8),
    ]
    all_ok = True
    for name, arr, ndim, dtype in checks:
        ok = arr.ndim == ndim and arr.dtype == dtype
        nan_inf = (
            bool(np.isnan(arr).any() or np.isinf(arr).any())
            if arr.dtype.kind == "f"
            else False
        )
        status = "✓" if ok and not nan_inf else "✗"
        print(f"  {status} {name:<22} shape={str(arr.shape):<25} dtype={arr.dtype}")
        if not ok or nan_inf:
            all_ok = False

    # Verify no event is sliced across splits
    if y_train[-1] == 1 and len(y_val) > 0 and y_val[0] == 1:
        print("  ✗ WARNING: anomaly straddles train/val boundary")
        all_ok = False
    if y_val[-1] == 1 and len(y_test_intern) > 0 and y_test_intern[0] == 1:
        print("  ✗ WARNING: anomaly straddles val/test_intern boundary")
        all_ok = False

    print(f"\n{'All checks passed ✓' if all_ok else 'Some checks FAILED ✗'}")


# ══════════════════════════════════════════════════════════════════════════════
# Kaggle submission pipeline — two-way split (train, val) + Kaggle test set
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing_kaggle() -> None:
    """
    Kaggle submission preprocessing pipeline.

    Two-way temporal 80/20 train/val split, plus the unlabelled Kaggle test
    set scaled with the same scaler. RobustScaler is fit on nominal training
    rows only (no leakage).

    Reads data/raw/ and writes data/processed/kaggle/.
    Idempotent — safe to re-run.
    """
    OUT_DIR = PROCESSED_DIR / "kaggle"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
        path = OUT_DIR / fname
        print(f"  {fname:<30} {str(arr.shape):<25} {arr.nbytes / 1e6:>8.1f} MB")
        np.save(path, arr)
        total_bytes += arr.nbytes

    print(f"\n  Total saved : {total_bytes / 1e9:.2f} GB → {OUT_DIR}")

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
    config_path = OUT_DIR / "preprocessing_config.json"
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
