"""
End-to-end sanity check for the FE 46-channel detrended PCA pipeline.

Runs `predict_fe46()` over the kaggle internal test slice (rows 12.7M..14.7M)
and compares the resulting metrics against the values stored in
`models/fe_46ch_<ts>.json`. The training script and the predictor share no
code, so matching numbers prove the inference path is wired up correctly.

Run:  python scripts/verify_fe46.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentinel.params import PROCESSED_DIR, MODELS_DIR
from sentinel.ml_logic.predictor import predict_fe46
from sentinel.ml_logic.metrics import (
    corrected_event_f05,
    event_f05,
    row_precision_recall,
)
from sentinel.ml_logic.data import find_anomaly_segments


VAL_END = 12_700_000


def metrics_for(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    ef = event_f05(y_true, y_pred)
    es = corrected_event_f05(y_true, y_pred)
    rw = row_precision_recall(y_true, y_pred)
    return {
        "esa_f05"      : float(es["f_score"]),
        "event_f05"    : float(ef["f_score"]),
        "event_recall" : float(ef["recall"]),
        "row_f1"       : float(rw["f1"]),
        "events_hit"   : int(ef["tp_events"]),
        "events_total" : int(ef["tp_events"] + ef["fn_events"]),
        "flag_rate"    : float(y_pred.mean()),
        "n_pred_blocks": len(find_anomaly_segments(y_pred)),
    }


def main() -> None:
    sidecar_path = sorted(MODELS_DIR.glob("fe_46ch_*.json"))[-1]
    with open(sidecar_path) as f:
        sidecar = json.load(f)
    expected_baseline = sidecar["test_metrics"]["baseline"]
    expected_postfilt = sidecar["test_metrics"]["post_filter"]
    print(f"Loaded sidecar: {sidecar_path.name}")
    print(f"  threshold     = {sidecar['threshold']:.6f}")
    print(f"  trained_at    = {sidecar['trained_at']}")

    print("\nLoading kaggle internal test slice (rows 12.7M..end) ...")
    train_full_scaled = np.load(PROCESSED_DIR / "kaggle" / "train_full_scaled.npy",
                                mmap_mode="r")
    y_train_row = np.load(PROCESSED_DIR / "kaggle" / "y_train_row.npy")
    n_total = train_full_scaled.shape[0]
    print(f"  train_full_scaled {train_full_scaled.shape}, "
          f"test slice = rows {VAL_END:,}..{n_total:,}")

    # predict_fe46 expects raw 58ch and re-applies the scaler. The kaggle
    # array is already scaled, so feed the unscaled raw frame from disk for a
    # true end-to-end check.
    # Feed the FULL raw frame through predict_fe46, then slice predictions to
    # the test rows. This matches the training script's detrend boundaries
    # (full-timeline rolling median); slicing first would change the median
    # context near the slice edge.
    print("\nLoading raw train.parquet (for end-to-end inference) ...")
    import pandas as pd
    t0 = time.time()
    df_full = pd.read_parquet(PROJECT_ROOT / "data" / "raw" / "train.parquet")
    print(f"  loaded in {time.time()-t0:.1f}s   df_full shape={df_full.shape}")
    y_test = y_train_row[VAL_END:]

    print("\nRunning predict_fe46(post_filter=False) on full train ...")
    t0 = time.time()
    sub_baseline_full = predict_fe46(X_raw=df_full, post_filter=False)
    print(f"  done in {time.time()-t0:.1f}s")
    m_baseline = metrics_for(y_test,
                             sub_baseline_full["is_anomaly"].values[VAL_END:])

    print("\nRunning predict_fe46(post_filter=True) on full train ...")
    t0 = time.time()
    sub_post_full = predict_fe46(X_raw=df_full, post_filter=True)
    print(f"  done in {time.time()-t0:.1f}s")
    m_post = metrics_for(y_test,
                         sub_post_full["is_anomaly"].values[VAL_END:])

    # ─ Compare ─
    def fmt_row(label, m):
        return (f"  [{label:10s}] ESA={m['esa_f05']:.3f}  "
                f"recall={m['event_recall']:.3f}  "
                f"({m['events_hit']}/{m['events_total']})  "
                f"flag={m['flag_rate']*100:.2f}%  "
                f"blocks={m['n_pred_blocks']}")

    print("\n──────── observed ────────")
    print(fmt_row("baseline",  m_baseline))
    print(fmt_row("postfilter", m_post))
    print("──────── expected (from sidecar) ────────")
    print(fmt_row("baseline",  expected_baseline))
    print(fmt_row("postfilter", expected_postfilt))

    # The training script's sidecar metrics were computed by score-detrending
    # the 2M-row test slice in isolation, while predict_fe46 score-detrends
    # the full 14.7M-row timeline (real inference path). Near the val/test
    # boundary the rolling-median baseline differs by ~50k rows of context,
    # which shifts a handful of borderline scores across the threshold and
    # nudges flag rate / ESA. Recall and event-hits should match exactly;
    # flag rate should be within ~0.5 pp; ESA within ~0.1.
    ok = True
    for label, observed, expected in [
        ("baseline",   m_baseline, expected_baseline),
        ("postfilter", m_post,      expected_postfilt),
    ]:
        if observed["events_hit"] != expected["events_hit"]:
            print(f"  HARD MISMATCH [{label}] events_hit: "
                  f"{observed['events_hit']} vs {expected['events_hit']}")
            ok = False
        if abs(observed["flag_rate"] - expected["flag_rate"]) > 5e-3:
            print(f"  HARD MISMATCH [{label}] flag_rate drift > 0.5pp: "
                  f"observed={observed['flag_rate']:.4f} vs expected={expected['flag_rate']:.4f}")
            ok = False
        if abs(observed["esa_f05"] - expected["esa_f05"]) > 0.1:
            print(f"  HARD MISMATCH [{label}] ESA delta > 0.1: "
                  f"observed={observed['esa_f05']:.4f} vs expected={expected['esa_f05']:.4f}")
            ok = False

    print("\nNote: small ESA / flag-rate deltas vs sidecar are expected — "
          "training metrics use in-slice score detrend, predict_fe46 uses "
          "full-timeline detrend (real inference path).")
    print("Result:", "PASS — predict_fe46 reproduces FE pipeline within tolerance"
          if ok else "FAIL — drift beyond expected boundary effect")


if __name__ == "__main__":
    main()
