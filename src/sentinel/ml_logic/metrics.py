"""
Corrected event-wise F-score — ESA Anomaly Detection Benchmark.

Source: https://github.com/kplabs-pl/ESA-ADB
        timeeval/metrics/ESA_ADB_metrics.py

Formula (Kotowski et al., 2024 / Sehili et al., 2023):
────────────────────────────────────────────────────────
  Re_e   = TP_events / (TP_events + FN_events)
              TP_events  = true anomaly segments with ≥ 1 predicted positive
              FN_events  = true anomaly segments with 0 predicted positives

  Pr_ew  = TP_events / (TP_events + FP_pred_events)
              FP_pred_events = predicted contiguous segments that do NOT
                               overlap any true anomaly segment

  TNR    = 1 − fp_samples / N_nominal
              fp_samples = predicted-positive samples in truly nominal regions

  Pr_c   = Pr_ew × TNR          (corrected precision)

  F_β    = (1+β²) · Pr_c · Re_e / (β² · Pr_c + Re_e)
           default β = 0.5  →  precision weighted 2×

Sanity checks:
  all-zeros : Re_e = 0            → F = 0
  all-ones  : TNR  = 0 → Pr_c = 0 → F = 0
  perfect   : Pr_c = 1, Re_e = 1  → F = 1
  1 sample/event, 0 FP: Pr_c = 1, Re_e = 1 → F = 1
"""

import numpy as np
from .data import find_anomaly_segments


def _find_predicted_segments(y_pred: np.ndarray) -> list[dict]:
    """Return contiguous predicted-anomaly segments — vectorised via np.diff."""
    padded = np.concatenate(([0], y_pred.astype(np.int8), [0]))
    d      = np.diff(padded)
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0] - 1
    return [{"start": int(s), "end": int(e)} for s, e in zip(starts, ends)]


def corrected_event_f05(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict:
    """
    Compute the corrected event-wise F-beta score matching the ESA-ADB framework.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_pred : array-like of 0/1
    beta   : default 0.5 (precision-weighted)

    Returns
    -------
    dict: f_score, precision, recall, tp_events, fn_events,
          fp_pred_events, fp_samples, tnr
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    true_segs = find_anomaly_segments(y_true)   # ground-truth events
    pred_segs = _find_predicted_segments(y_pred)

    n_nominal  = int((y_true == 0).sum())
    n_events   = len(true_segs)

    if n_events == 0:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0,
                "tp_events": 0, "fn_events": 0,
                "fp_pred_events": len(pred_segs), "fp_samples": 0, "tnr": 1.0}

    # ── Step 1: event-wise TP / FN (over ground-truth segments) ──────────────
    tp_events = 0
    fn_events = 0
    matched_pred = [False] * len(pred_segs)   # track which pred segs overlap GT

    for ts in true_segs:
        detected = False
        for p, ps in enumerate(pred_segs):
            # overlap: not (ps.end < ts.start or ps.start > ts.end)
            if ps["end"] >= ts["start"] and ps["start"] <= ts["end"]:
                matched_pred[p] = True
                detected = True
        if detected:
            tp_events += 1
        else:
            fn_events += 1

    # ── Step 2: FP predicted events (pred segments with NO GT overlap) ────────
    fp_pred_events = sum(1 for m in matched_pred if not m)

    # ── Step 3: TNR correction ────────────────────────────────────────────────
    fp_samples = int(((y_pred == 1) & (y_true == 0)).sum())
    tnr = (1.0 - fp_samples / n_nominal) if n_nominal > 0 else 1.0

    # ── Step 4: corrected precision ───────────────────────────────────────────
    denom_pr = tp_events + fp_pred_events
    pr_ew    = (tp_events / denom_pr) if denom_pr > 0 else 0.0
    precision = pr_ew * tnr                        # Pr_c = Pr_ew × TNR

    # ── Step 5: recall & F-beta ───────────────────────────────────────────────
    recall  = tp_events / n_events

    if precision + recall == 0:
        f_score = 0.0
    else:
        f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    return {
        "f_score"       : round(f_score,    6),
        "precision"     : round(precision,  6),
        "recall"        : round(recall,     6),
        "tp_events"     : tp_events,
        "fn_events"     : fn_events,
        "fp_pred_events": fp_pred_events,
        "fp_samples"    : fp_samples,
        "tnr"           : round(tnr, 6),
    }


def f05_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Convenience wrapper — returns just the F0.5 scalar."""
    return corrected_event_f05(y_true, y_pred, beta=0.5)["f_score"]
