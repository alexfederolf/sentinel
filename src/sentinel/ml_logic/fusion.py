"""
Segment-aware diagnostics for OR-fusing two binary prediction streams.

Why this module exists
----------------------
A naive row-count fusion check (``Δ flag_rate ≤ 0.2 pp``) is misleading when
the base submission already has ``Pr_ew ≈ 1.0``. ESA-corrected event-wise F0.5
counts **`FP_pred_events`** — contiguous predicted segments outside any true
event — not rows. Adding a 100-row block outside a true event raises
``FP_pred_events`` by 1 and, if ``Pr_ew`` was already near 1.0, drops F0.5
substantially.

Concrete project example (NB 12c freq6 fusion, 2026-04-29):
    Hybrid alone        : public 0.867 / private 0.915
    Hybrid OR 12c-freq6 : public 0.830 / private 0.845    (−0.037 / −0.070)
    Row-count check said +0.058 pp flag rate → "safe".
    Segment-count check would have said +3-4 FP segments → "reject".

Usage
-----
    from sentinel.ml_logic.fusion import fusion_diagnostics
    d = fusion_diagnostics(y_base, y_new, y_true)
    print(d['verdict'], d['reason'])
    # 'reject' / 'borderline' / 'submit' / 'submit_strong'
"""
from __future__ import annotations

import numpy as np

from .data import find_anomaly_segments


# ─────────────────────────────────────────────────────────────────────────────
# Per-prediction diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def event_diagnostics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute event-wise stats for a binary prediction array on a labeled split.

    Parameters
    ----------
    y_pred : array-like of 0/1
    y_true : array-like of 0/1 (same length as y_pred)

    Returns
    -------
    dict with keys
        n_true_events     : int  — # of contiguous 1-runs in y_true
        n_pred_segments   : int  — # of contiguous 1-runs in y_pred
        TP_events         : int  — # true events with ≥1 overlapping pred-positive row
        FN_events         : int  — n_true_events − TP_events
        FP_pred_events    : int  — # pred segments that don't overlap any true event
        flag_rate         : float — float fraction of predicted positives
        true_event_idx_hit: list[int] — indices into find_anomaly_segments(y_true)
                                        for each true event that was hit
    """
    y_pred = np.asarray(y_pred, dtype=np.int8)
    y_true = np.asarray(y_true, dtype=np.int8)
    if y_pred.shape != y_true.shape:
        raise ValueError(f'shape mismatch: {y_pred.shape} vs {y_true.shape}')

    true_segs = find_anomaly_segments(y_true)
    pred_segs = find_anomaly_segments(y_pred)

    # TP_events: each true segment with at least one overlapping pred-positive row
    tp_idx = []
    for i, ev in enumerate(true_segs):
        if y_pred[ev['start']:ev['end'] + 1].any():
            tp_idx.append(i)
    tp_events = len(tp_idx)

    # FP_pred_events: pred segments that don't overlap any true event
    # Compute by sweeping pred segments against y_true.
    fp_segs = 0
    for ps in pred_segs:
        if not y_true[ps['start']:ps['end'] + 1].any():
            fp_segs += 1

    return {
        'n_true_events'     : len(true_segs),
        'n_pred_segments'   : len(pred_segs),
        'TP_events'         : tp_events,
        'FN_events'         : len(true_segs) - tp_events,
        'FP_pred_events'    : fp_segs,
        'flag_rate'         : float(y_pred.mean()),
        'true_event_idx_hit': tp_idx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise OR-fusion diagnostics + verdict
# ─────────────────────────────────────────────────────────────────────────────

def fusion_diagnostics(
    y_base: np.ndarray,
    y_new: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Compare base submission vs (base OR new) on a labeled split.

    Returns a dict with three sub-dicts (`base`, `new`, `fused`) of event stats,
    a `delta` dict of changes vs base, and a `verdict` ∈
    {'reject', 'borderline', 'submit', 'submit_strong'} with a human-readable
    `reason`.

    Verdict rules
    -------------
        delta_TP > delta_FP   and delta_TP ≥ 2          → 'submit_strong'
        delta_TP > delta_FP   and delta_TP ≥ 1          → 'submit'
        delta_TP == delta_FP  and delta_TP > 0          → 'borderline'
        delta_TP < delta_FP                             → 'reject'
        delta_TP == 0 and delta_FP == 0                 → 'reject'   (no-op fuse)
        delta_TP == 0 and delta_FP > 0                  → 'reject'   (only adds FPs)

    Note on row-count: this function deliberately ignores the row-count flag
    delta — it's a misleading proxy. Use `fused.flag_rate − base.flag_rate` if
    you also need the row-level number for a Kaggle preview.
    """
    y_base = np.asarray(y_base, dtype=np.int8)
    y_new  = np.asarray(y_new,  dtype=np.int8)
    y_true = np.asarray(y_true, dtype=np.int8)

    y_fused = (y_base | y_new).astype(np.int8)

    base  = event_diagnostics(y_base,  y_true)
    new   = event_diagnostics(y_new,   y_true)
    fused = event_diagnostics(y_fused, y_true)

    delta_tp = fused['TP_events']      - base['TP_events']
    delta_fp = fused['FP_pred_events'] - base['FP_pred_events']
    delta_flag = fused['flag_rate']    - base['flag_rate']

    if delta_tp == 0 and delta_fp == 0:
        verdict = 'reject'
        reason  = 'no-op: fusion adds no new TP events and no new FP segments.'
    elif delta_tp == 0 and delta_fp > 0:
        verdict = 'reject'
        reason  = (f'only adds FP segments (+{delta_fp}), no new TP events. '
                   f'Pr_ew will drop, F0.5 will drop.')
    elif delta_tp < delta_fp:
        verdict = 'reject'
        reason  = (f'adds {delta_fp} FP segments but only {delta_tp} new TP events. '
                   f'Net negative for F0.5.')
    elif delta_tp == delta_fp:
        verdict = 'borderline'
        reason  = (f'gains and losses balance ({delta_tp} TP vs {delta_fp} FP). '
                   f'TNR and recall trade-off may go either way; submit only if '
                   f'you also want the diagnostic signal.')
    elif delta_tp >= 2 * max(1, delta_fp):
        verdict = 'submit_strong'
        reason  = (f'gains clearly exceed losses: +{delta_tp} TP events vs '
                   f'+{delta_fp} FP segments.')
    else:  # delta_tp > delta_fp
        verdict = 'submit'
        reason  = (f'gains exceed losses: +{delta_tp} TP events vs +{delta_fp} FP segments.')

    return {
        'base'  : base,
        'new'   : new,
        'fused' : fused,
        'delta' : {
            'TP_events'      : delta_tp,
            'FP_pred_events' : delta_fp,
            'flag_rate'      : delta_flag,
        },
        'verdict': verdict,
        'reason' : reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print for diagnostic logs
# ─────────────────────────────────────────────────────────────────────────────

def format_fusion_report(d: dict, label_base: str = 'base', label_new: str = 'new') -> str:
    """One-glance text report from a `fusion_diagnostics` result."""
    b, n, f, dt = d['base'], d['new'], d['fused'], d['delta']
    nev = b['n_true_events']
    lines = [
        f'─── Fusion diagnostics  ({nev} true events on this split) ───',
        f'                          {label_base:>10}   {label_new:>10}    fused',
        f'  TP_events            {b["TP_events"]:>10}  {n["TP_events"]:>10}  {f["TP_events"]:>7}',
        f'  FP_pred_events       {b["FP_pred_events"]:>10}  {n["FP_pred_events"]:>10}  {f["FP_pred_events"]:>7}',
        f'  n_pred_segments      {b["n_pred_segments"]:>10}  {n["n_pred_segments"]:>10}  {f["n_pred_segments"]:>7}',
        f'  flag_rate            {b["flag_rate"]:>10.4%}  {n["flag_rate"]:>10.4%}  {f["flag_rate"]:>7.4%}',
        '',
        f'  Δ TP_events          {dt["TP_events"]:+d}',
        f'  Δ FP_pred_events     {dt["FP_pred_events"]:+d}',
        f'  Δ flag_rate          {dt["flag_rate"]*100:+.3f} pp',
        '',
        f'  VERDICT : {d["verdict"].upper()}',
        f'  reason  : {d["reason"]}',
    ]
    return '\n'.join(lines)
