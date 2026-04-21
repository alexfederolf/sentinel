"""Tests for sentinel.ml_logic.metrics.corrected_event_f05."""
import numpy as np
import pytest
from sentinel.ml_logic.metrics import corrected_event_f05


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
