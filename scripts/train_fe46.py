"""
Train the FE 46-channel detrended PCA model and save the FE artefact bundle.

Recipe = NB 11e (renamed from 11d):
  - Channel set: 58 target channels minus Tier A (drift-flooders 64-66, 70, 71, 73-76)
                 minus Tier B (val-99-percentile noise 16, 24, 32) → 46 channels.
  - Per-channel rolling-median input detrend (window 100k rows).
  - Tail-fit PCA on the last 50k nominal windows whose end row ≤ TRAIN_END (10.7M).
  - PCA truncated to cumulative EV ≥ 0.95.
  - Score val + test with score_windows_detrended (window 1000 windows = 100k rows).
  - Threshold via tune_threshold(corrected_event_f05) on val 10.7M..12.7M.

Outputs (all written to models/ and data/raw/):
  - models/pca_fe_46ch_<ts>.pkl                     timestamped artefact
  - models/pca_fe_46ch.pkl                          stable alias
  - data/raw/target_channels_fe.csv                 46 channel names in canonical order
  - models/fe_46ch_<ts>.json                        sidecar config + metrics

Run:  python scripts/train_fe46.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinel.params import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE, WINDOW_SIZE
from sentinel.ml_logic.scorer import score_windows_detrended, clean_predictions
from sentinel.ml_logic.thresholds import tune_threshold
from sentinel.ml_logic.metrics import corrected_event_f05, event_f05, row_precision_recall
from sentinel.ml_logic.data import find_anomaly_segments


# ── Config ───────────────────────────────────────────────────────────────────
KAGGLE_DIR        = PROCESSED_DIR / "kaggle"
RAW_DIR           = PROJECT_ROOT / "data" / "raw"

TRAIN_END         = 10_700_000
VAL_END           = 12_700_000
FIT_SIZE          = 50_000
DETREND_WIN       = 100_000        # input detrend, rows per channel
SCORE_DETREND_WIN = 1_000          # score detrend, in WINDOWS (1000 * 100 = 100k rows)
SCORE_DETREND_MD  = "median"
EV_THRESHOLD      = 0.95
POST_MIN_LEN      = 100
POST_MAX_GAP      = 500

TIER_A = {"channel_64", "channel_65", "channel_66",
          "channel_70", "channel_71",
          "channel_73", "channel_74", "channel_75", "channel_76"}
TIER_B = {"channel_16", "channel_24", "channel_32"}


def detrend_per_channel(arr2d: np.ndarray, window: int) -> np.ndarray:
    out = np.empty_like(arr2d, dtype=np.float32)
    for j in range(arr2d.shape[1]):
        col = arr2d[:, j].astype(np.float32, copy=False)
        out[:, j] = col - median_filter(col, size=window, mode="reflect")
    return out


def main() -> None:
    print("Loading kaggle preprocessing artefacts ...")
    t0 = time.time()
    train_full_scaled = np.load(KAGGLE_DIR / "train_full_scaled.npy", mmap_mode="r")
    y_train_full      = np.load(KAGGLE_DIR / "y_train_full.npy")
    y_train_row       = np.load(KAGGLE_DIR / "y_train_row.npy")
    with open(KAGGLE_DIR / "preprocessing_config.json") as f:
        kaggle_cfg = json.load(f)
    target_channels = kaggle_cfg["target_channels"]
    print(f"  loaded in {time.time()-t0:.1f}s — train_full_scaled {train_full_scaled.shape}")

    # ─ Channel selection ─
    drop = TIER_A | TIER_B
    ch_names_fe = [c for c in target_channels if c not in drop]
    ch_indices  = np.array([target_channels.index(c) for c in ch_names_fe])
    n_ch = len(ch_names_fe)
    assert n_ch == 46, f"expected 46 channels, got {n_ch}"
    print(f"\nChannel selection: {n_ch}/58  (drop Tier A+B = {sorted(drop)})")

    # ─ Slice + detrend ─
    print("\nSlicing channels and detrending input (per channel) ...")
    t0 = time.time()
    train_sliced = np.ascontiguousarray(train_full_scaled[:, ch_indices]).astype(np.float32)
    print(f"  slice  in {time.time()-t0:.1f}s   shape {train_sliced.shape}")

    t0 = time.time()
    train_dt = detrend_per_channel(train_sliced, DETREND_WIN)
    print(f"  detrend in {time.time()-t0:.1f}s   window={DETREND_WIN:,}")
    del train_sliced

    # ─ Tail-fit PCA ─
    print("\nTail-fitting PCA on last 50k nominal windows (end row ≤ TRAIN_END) ...")
    n_windows  = len(y_train_full)
    win_end    = (np.arange(n_windows) + 1) * WINDOW_SIZE
    eligible   = (y_train_full == 0) & (win_end <= TRAIN_END)
    elig_idx   = np.flatnonzero(eligible)
    fit_widx   = elig_idx[-FIT_SIZE:]
    row_off    = np.arange(WINDOW_SIZE)
    row_idx    = (fit_widx[:, None] * WINDOW_SIZE + row_off[None, :]).ravel()
    X_fit      = train_dt[row_idx].reshape(len(fit_widx), -1).astype(np.float32)
    print(f"  X_fit shape {X_fit.shape}  ({len(fit_widx):,} windows × {WINDOW_SIZE * n_ch} flat dims)")

    t0 = time.time()
    K_OVER = min(800, X_fit.shape[1] - 1, X_fit.shape[0] - 1)
    pca = PCA(n_components=K_OVER, svd_solver="randomized",
              random_state=RANDOM_STATE).fit(X_fit)
    cum_ev = np.cumsum(pca.explained_variance_ratio_)
    k_keep = int(np.searchsorted(cum_ev, EV_THRESHOLD) + 1)
    pca.components_              = pca.components_[:k_keep]
    pca.explained_variance_      = pca.explained_variance_[:k_keep]
    pca.explained_variance_ratio_= pca.explained_variance_ratio_[:k_keep]
    pca.n_components_            = k_keep
    print(f"  PCA fit in {time.time()-t0:.1f}s   "
          f"k_keep={k_keep}  EV_sum={pca.explained_variance_ratio_.sum():.4f}")
    del X_fit

    # ─ Score val + test ─
    print("\nScoring val (10.7M..12.7M) and test (12.7M..14.7M) ...")
    X_val  = np.ascontiguousarray(train_dt[TRAIN_END:VAL_END]).astype(np.float32)
    X_test = np.ascontiguousarray(train_dt[VAL_END:]).astype(np.float32)
    y_val  = y_train_row[TRAIN_END:VAL_END]
    y_test = y_train_row[VAL_END:]

    sv = score_windows_detrended(pca, X_val,
                                 detrend_window=SCORE_DETREND_WIN,
                                 detrend_mode=SCORE_DETREND_MD)
    st = score_windows_detrended(pca, X_test,
                                 detrend_window=SCORE_DETREND_WIN,
                                 detrend_mode=SCORE_DETREND_MD)
    print(f"  val  scores  range [{sv.min():.4f}, {sv.max():.4f}]")
    print(f"  test scores  range [{st.min():.4f}, {st.max():.4f}]")

    # ─ Threshold tuning on val ─
    print("\nTuning threshold via corrected_event_f05 on val ...")
    tuned = tune_threshold(sv, y_val, metric_fn=corrected_event_f05, n_sweep=80)
    threshold = float(tuned["threshold"])
    print(f"  threshold     = {threshold:.6f}")
    print(f"  val ESA F0.5  = {tuned['score']:.4f}")

    # ─ Test metrics (baseline + post-filter) ─
    def metrics(y_pred):
        ef = event_f05(y_test, y_pred)
        es = corrected_event_f05(y_test, y_pred)
        rw = row_precision_recall(y_test, y_pred)
        return {
            "esa_f05"       : float(es["f_score"]),
            "esa_pr_ew"     : float(es["precision"]),
            "esa_tnr"       : float(es["tnr"]),
            "event_f05"     : float(ef["f_score"]),
            "event_recall"  : float(ef["recall"]),
            "event_precision": float(ef["precision"]),
            "row_f1"        : float(rw["f1"]),
            "row_precision" : float(rw["precision"]),
            "row_recall"    : float(rw["recall"]),
            "events_hit"    : int(ef["tp_events"]),
            "events_total"  : int(ef["tp_events"] + ef["fn_events"]),
            "flag_rate"     : float(y_pred.mean()),
            "n_pred_blocks" : len(find_anomaly_segments(y_pred)),
        }

    y_pred_baseline = (st > threshold).astype(np.int8)
    y_pred_clean    = clean_predictions(y_pred_baseline,
                                        min_len=POST_MIN_LEN, max_gap=POST_MAX_GAP)
    m_baseline = metrics(y_pred_baseline)
    m_clean    = metrics(y_pred_clean)

    print("\nTest-slice metrics:")
    for label, m in [("baseline ", m_baseline), ("postfilter", m_clean)]:
        print(f"  [{label}]  ESA={m['esa_f05']:.3f}  recall={m['event_recall']:.3f}  "
              f"({m['events_hit']}/{m['events_total']}) flag={m['flag_rate']*100:.2f}%  "
              f"blocks={m['n_pred_blocks']}")

    # ─ Save artefacts ─
    ts = time.strftime("%Y%m%d_%H%M%S")

    pca_path_ts    = MODELS_DIR / f"pca_fe_46ch_{ts}.pkl"
    pca_path_alias = MODELS_DIR / "pca_fe_46ch.pkl"
    with open(pca_path_ts, "wb") as f:
        pickle.dump(pca, f)
    with open(pca_path_alias, "wb") as f:
        pickle.dump(pca, f)
    print(f"\nsaved → {pca_path_ts}")
    print(f"saved → {pca_path_alias}  (stable alias)")

    features_csv = RAW_DIR / "target_channels_fe.csv"
    pd.DataFrame({"target_channels": ch_names_fe}).to_csv(features_csv, index=False)
    print(f"saved → {features_csv}  ({n_ch} channels)")

    sidecar = {
        "model_pkl"          : str(pca_path_alias.relative_to(PROJECT_ROOT)),
        "model_pkl_ts"       : str(pca_path_ts.relative_to(PROJECT_ROOT)),
        "features_csv"       : str(features_csv.relative_to(PROJECT_ROOT)),
        "n_channels"         : n_ch,
        "channels"           : ch_names_fe,
        "channels_dropped"   : sorted(drop),
        "fit"                : {
            "method"      : "tail_50k_NB11e",
            "fit_size"    : FIT_SIZE,
            "train_end"   : TRAIN_END,
            "ev_threshold": EV_THRESHOLD,
            "n_components": int(pca.n_components_),
            "ev_sum"      : float(pca.explained_variance_ratio_.sum()),
            "random_state": RANDOM_STATE,
        },
        "input_detrend"      : {"window": DETREND_WIN, "mode": "median"},
        "score_detrend"      : {"window_in_windows": SCORE_DETREND_WIN,
                                "window_in_rows"  : SCORE_DETREND_WIN * WINDOW_SIZE,
                                "mode": SCORE_DETREND_MD},
        "threshold"          : threshold,
        "threshold_method"   : "tune_threshold(corrected_event_f05) on val 10.7M..12.7M",
        "val_esa_f05"        : float(tuned["score"]),
        "post_filter"        : {"min_len": POST_MIN_LEN, "max_gap": POST_MAX_GAP},
        "test_slice"         : {"start": VAL_END, "end": train_full_scaled.shape[0]},
        "test_metrics"       : {"baseline": m_baseline, "post_filter": m_clean},
        "trained_at"         : ts,
    }
    sidecar_path = MODELS_DIR / f"fe_46ch_{ts}.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, default=float)
    print(f"saved → {sidecar_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
