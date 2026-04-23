"""Tests for sentinel.ml_logic.metrics — both the ESA corrected metric and
the bootcamp event / point / row metrics."""
import numpy as np
import pytest
from sentinel.ml_logic.metrics import (
    compute_all_metrics,
    corrected_event_f05,
    esa_metric,
    event_detection_rate,
    event_f05,
    event_f1,
    event_f2,
    event_fbeta,
    point_adjust_f1,
    row_precision_recall,
)


def make_gt(event_starts_lengths):
    N = 1000
    y = np.zeros(N, dtype=np.int8)
    for start, length in event_starts_lengths:
        y[start:start + length] = 1
    return y


def test_all_zeros_pred_gives_zero_f():
    y_true = make_gt([(100, 50)])
    y_pred = np.zeros(1000, dtype=np.int8)
    m = corrected_event_f05(y_true, y_pred)
    assert m["f_score"] == 0.0


def test_all_ones_pred_tnr_zero():
    y_true = make_gt([(100, 50)])
    y_pred = np.ones(1000, dtype=np.int8)
    m = corrected_event_f05(y_true, y_pred)
    assert m["tnr"] == 0.0
    assert m["f_score"] == 0.0


def test_perfect_prediction():
    y_true = make_gt([(100, 50), (400, 30)])
    y_pred = y_true.copy()
    m = corrected_event_f05(y_true, y_pred)
    assert m["f_score"] == pytest.approx(1.0, abs=1e-6)


def test_one_sample_per_event_no_fp():
    y_true = make_gt([(100, 50), (300, 20)])
    y_pred = np.zeros(1000, dtype=np.int8)
    y_pred[110] = 1   # one sample inside first event
    y_pred[305] = 1   # one sample inside second event
    m = corrected_event_f05(y_true, y_pred)
    assert m["tp_events"] == 2
    assert m["fp_pred_events"] == 0
    assert m["f_score"] > 0.0


def test_single_sample_event_detected():
    N = 200
    y_true = np.zeros(N, dtype=np.int8)
    y_true[100] = 1   # single-sample event
    y_pred = np.zeros(N, dtype=np.int8)
    y_pred[100] = 1
    m = corrected_event_f05(y_true, y_pred)
    assert m["tp_events"] == 1
    assert m["recall"] == pytest.approx(1.0)


def test_fp_segment_penalises_precision():
    y_true = make_gt([(100, 50)])
    y_pred = y_true.copy()
    y_pred[500:510] = 1   # spurious segment in nominal region
    m_no_fp  = corrected_event_f05(y_true, y_true)
    m_with_fp = corrected_event_f05(y_true, y_pred)
    assert m_with_fp["fp_pred_events"] >= 1
    assert m_with_fp["f_score"] < m_no_fp["f_score"]


# ══════════════════════════════════════════════════════════════════════════════
# Bootcamp metrics — additive tests
# ══════════════════════════════════════════════════════════════════════════════

def test_esa_metric_is_corrected_alias():
    assert esa_metric is corrected_event_f05


def test_event_f05_has_no_tnr_key():
    """event_f05 must be the *uncorrected* event metric — no TNR term."""
    y_true = make_gt([(100, 50)])
    y_pred = y_true.copy()
    out = event_f05(y_true, y_pred)
    assert "tnr" not in out
    assert "f_score" in out


def test_event_f05_perfect():
    y_true = make_gt([(100, 50), (400, 30)])
    out = event_f05(y_true, y_true)
    assert out["f_score"] == pytest.approx(1.0, abs=1e-6)
    assert out["tp_events"] == 2
    assert out["fn_events"] == 0
    assert out["fp_pred_events"] == 0


def test_event_f05_no_tnr_pull_down():
    """
    The key difference between event_f05 (bootcamp) and corrected_event_f05
    (ESA Kaggle) — under lots of nominal FP rows, corrected F0.5 drops via
    TNR, but standard event_f05 stays the same because precision is measured
    in *event* units, not row units.
    """
    y_true = make_gt([(100, 50)])
    y_pred = y_true.copy()
    y_pred[600:610] = 1  # one extra FP event, no change in rows inside true event
    std = event_f05(y_true, y_pred)
    esa = corrected_event_f05(y_true, y_pred)
    # Standard F0.5 is higher (not pulled down by TNR)
    assert std["f_score"] > esa["f_score"]


def test_event_f1_vs_f2_weighting():
    """F1 balances precision+recall; F2 weights recall higher."""
    y_true = make_gt([(100, 50), (400, 30)])
    y_pred = np.zeros_like(y_true)
    y_pred[110] = 1  # only first event detected → recall 0.5, precision 1.0
    f05 = event_f05(y_true, y_pred)["f_score"]
    f1  = event_f1(y_true, y_pred)["f_score"]
    f2  = event_f2(y_true, y_pred)["f_score"]
    # With precision=1.0, recall=0.5: F0.5 > F1 > F2
    assert f05 > f1 > f2


def test_event_fbeta_matches_individual_helpers():
    y_true = make_gt([(100, 50), (400, 30)])
    y_pred = y_true.copy()
    y_pred[105] = 0
    assert event_fbeta(y_true, y_pred, beta=0.5) == event_f05(y_true, y_pred)
    assert event_fbeta(y_true, y_pred, beta=1.0) == event_f1(y_true, y_pred)
    assert event_fbeta(y_true, y_pred, beta=2.0) == event_f2(y_true, y_pred)


def test_event_detection_rate_counts():
    y_true = make_gt([(100, 50), (400, 30), (700, 20)])
    y_pred = np.zeros_like(y_true)
    y_pred[110] = 1  # first event
    y_pred[710] = 1  # third event; second is missed
    edr = event_detection_rate(y_true, y_pred)
    assert edr["n_events"] == 3
    assert edr["tp_events"] == 2
    assert edr["rate"] == pytest.approx(2 / 3, abs=1e-6)


def test_point_adjust_f1_forgives_partial_detection():
    """
    PA-F1: one positive row inside a true event is enough to mark the
    entire event as predicted — so recall = 1.0 even if only 1/50 rows hit.
    """
    y_true = make_gt([(100, 50)])
    y_pred = np.zeros_like(y_true)
    y_pred[110] = 1
    pa = point_adjust_f1(y_true, y_pred)
    assert pa["recall"] == pytest.approx(1.0)
    # Standard row recall would be 1/50 = 0.02 — PA recall is much higher.
    row = row_precision_recall(y_true, y_pred)
    assert row["recall"] < 0.05


def test_row_precision_recall_perfect():
    y_true = make_gt([(100, 50)])
    out = row_precision_recall(y_true, y_true)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"]    == pytest.approx(1.0)
    assert out["f1"]        == pytest.approx(1.0)


def test_compute_all_metrics_keys_and_shape():
    y_true = make_gt([(100, 50), (400, 30)])
    y_pred = y_true.copy()
    m = compute_all_metrics(y_true, y_pred)
    expected = {
        "event_f05", "event_f1", "event_f2",
        "event_precision", "event_recall",
        "esa_f05", "esa_precision", "esa_recall", "esa_tnr",
        "event_detection_rate", "tp_events", "n_events", "fp_pred_events",
        "pa_f1", "pa_precision", "pa_recall",
        "row_precision", "row_recall", "row_f1",
    }
    assert set(m.keys()) == expected
    # Perfect prediction → every F-score is 1.0
    assert m["event_f05"] == pytest.approx(1.0)
    assert m["pa_f1"]     == pytest.approx(1.0)
    assert m["row_f1"]    == pytest.approx(1.0)
    assert m["event_detection_rate"] == pytest.approx(1.0)


def test_compute_all_metrics_empty_prediction():
    y_true = make_gt([(100, 50)])
    y_pred = np.zeros_like(y_true)
    m = compute_all_metrics(y_true, y_pred)
    assert m["event_f05"] == 0.0
    assert m["event_detection_rate"] == 0.0
    assert m["row_recall"] == 0.0
