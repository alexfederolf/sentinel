"""

ESA Corrected event-wise F0.5-score from the ESA Anomaly Detection Benchmark (ESA-ADB)
https://github.com/kplabs-pl/ESA-ADB/timeeval/metrics/ESA_ADB_metrics.py

Formula
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
    """Return contiguous predicted-anomaly segments - vectorised via np.diff."""
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
    Compute the corrected event-wise F-beta score.

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
    """Convenience wrapper - returns just the F0.5 scalar."""
    return corrected_event_f05(y_true, y_pred, beta=0.5)["f_score"]


# ══════════════════════════════════════════════════════════════════════════════
# Bootcamp metrics (additive - do not touch corrected_event_f05 above)
# ══════════════════════════════════════════════════════════════════════════════
#
# The ESA corrected metric (above) is kept for Kaggle-leaderboard comparison
# and exposed here as `esa_metric`. Everything else in this block is a
# *standard* event / point / row metric - no TNR correction, no secret sauce,
# easy to defend in a bootcamp presentation.
#
# All event-level metrics share the same TP/FN/FP-event tally, computed once
# by `_event_counts`. Individual metric functions return dicts so they can be
# destructured by callers; `compute_all_metrics` returns a flat dict for
# reporting and frontend display.
# ══════════════════════════════════════════════════════════════════════════════


# Alias - `esa_metric` is the name used in notebooks 11–13 for clarity.
esa_metric = corrected_event_f05


def _event_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Event-level TP / FN / FP / total - shared by every event-wise metric."""
    true_segs = find_anomaly_segments(y_true)
    pred_segs = _find_predicted_segments(y_pred)
    n_events  = len(true_segs)

    tp = 0
    matched = [False] * len(pred_segs)
    for ts in true_segs:
        hit = False
        for p, ps in enumerate(pred_segs):
            if ps["end"] >= ts["start"] and ps["start"] <= ts["end"]:
                matched[p] = True
                hit = True
        if hit:
            tp += 1
    fn      = n_events - tp
    fp_pred = sum(1 for m in matched if not m)
    return tp, fn, fp_pred, n_events


def event_fbeta(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> dict:
    """
    Standard event-wise F-beta. **No TNR correction** - this is the metric
    the bootcamp notebooks tune against.

    Pr_ew = TP_events / (TP_events + FP_pred_events)
    Re_e  = TP_events / N_true_events
    F_β   = (1 + β²) · Pr_ew · Re_e / (β² · Pr_ew + Re_e)
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    tp, fn, fp_pred, n_events = _event_counts(y_true, y_pred)

    if n_events == 0:
        return {"f_score": 0.0, "precision": 0.0, "recall": 0.0,
                "tp_events": 0, "fn_events": 0, "fp_pred_events": fp_pred}

    denom_p = tp + fp_pred
    precision = (tp / denom_p) if denom_p > 0 else 0.0
    recall    = tp / n_events

    if precision + recall == 0:
        f = 0.0
    else:
        f = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    return {
        "f_score"       : round(f, 6),
        "precision"     : round(precision, 6),
        "recall"        : round(recall, 6),
        "tp_events"     : tp,
        "fn_events"     : fn,
        "fp_pred_events": fp_pred,
    }


def event_f05(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-wise F0.5 (precision-weighted). Primary tuning metric."""
    return event_fbeta(y_true, y_pred, beta=0.5)


def event_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-wise F1 (balanced)."""
    return event_fbeta(y_true, y_pred, beta=1.0)


def event_f2(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-wise F2 (recall-weighted)."""
    return event_fbeta(y_true, y_pred, beta=2.0)


def event_detection_rate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    TP_events / N_true_events - the jury-friendly "caught X of Y events" number.
    Same as event-wise recall, but returned with the raw counts so the frontend
    can print `32 / 38 (84%)` without recomputing.
    """
    tp, _, _, n_events = _event_counts(
        np.asarray(y_true, dtype=np.int8),
        np.asarray(y_pred, dtype=np.int8),
    )
    rate = (tp / n_events) if n_events > 0 else 0.0
    return {"rate": round(rate, 6), "tp_events": tp, "n_events": n_events}


def point_adjust_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Standard point-adjust F1 (Xu et al. 2018): within each true event, if any
    predicted row is positive, mark the entire event as predicted. Then
    compute point-level precision / recall / F1 on the adjusted prediction.

    This is forgiving - it rewards partial event detection - and is one of
    the default reference metrics in the time-series anomaly-detection
    literature. We report it alongside the stricter event-wise metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    adj    = y_pred.copy()

    for ts in find_anomaly_segments(y_true):
        s, e = ts["start"], ts["end"]
        if y_pred[s:e + 1].any():
            adj[s:e + 1] = 1

    tp = int(((adj == 1) & (y_true == 1)).sum())
    fp = int(((adj == 1) & (y_true == 0)).sum())
    fn = int(((adj == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "f1"       : round(f1, 6),
        "precision": round(precision, 6),
        "recall"   : round(recall, 6),
        "tp": tp, "fp": fp, "fn": fn,
    }


def row_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Per-row precision, recall, F1. Used for the frontend headline numbers
    (no event structure required).
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 6),
        "recall"   : round(recall, 6),
        "f1"       : round(f1, 6),
        "tp": tp, "fp": fp, "fn": fn,
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Flat dict of every reported metric. Notebooks 11–13 call this once per
    submission and display the result as a summary table.

    Keys (flat, so the dict can feed straight into pandas):
      - event_f05 / event_f1 / event_f2
      - event_precision / event_recall
      - esa_f05 / esa_precision / esa_recall / esa_tnr (ESA Kaggle metric)
      - event_detection_rate, tp_events, n_events, fp_pred_events
      - pa_f1 / pa_precision / pa_recall
      - row_precision / row_recall / row_f1
    """
    ef05 = event_f05(y_true, y_pred)
    ef1  = event_f1(y_true, y_pred)
    ef2  = event_f2(y_true, y_pred)
    esa  = esa_metric(y_true, y_pred)
    edr  = event_detection_rate(y_true, y_pred)
    pa   = point_adjust_f1(y_true, y_pred)
    rpr  = row_precision_recall(y_true, y_pred)

    return {
        # event-wise (uncorrected) - primary tuning metric
        "event_f05"          : ef05["f_score"],
        "event_f1"           : ef1["f_score"],
        "event_f2"           : ef2["f_score"],
        "event_precision"    : ef05["precision"],
        "event_recall"       : ef05["recall"],
        # ESA Kaggle metric (corrected)
        "esa_f05"            : esa["f_score"],
        "esa_precision"      : esa["precision"],
        "esa_recall"         : esa["recall"],
        "esa_tnr"            : esa["tnr"],
        # jury-friendly event counts
        "event_detection_rate": edr["rate"],
        "tp_events"          : edr["tp_events"],
        "n_events"           : edr["n_events"],
        "fp_pred_events"     : ef05["fp_pred_events"],
        # point-adjust F1 (ref metric from TS-AD literature)
        "pa_f1"              : pa["f1"],
        "pa_precision"       : pa["precision"],
        "pa_recall"          : pa["recall"],
        # row-level (for frontend headlines)
        "row_precision"      : rpr["precision"],
        "row_recall"         : rpr["recall"],
        "row_f1"             : rpr["f1"],
    }
