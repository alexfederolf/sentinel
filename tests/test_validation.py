"""Tests for sentinel.ml_logic.validation."""
import numpy as np
import pandas as pd
import pytest

from sentinel.ml_logic.validation import (
    bootstrap_f05_ci,
    replay_submission,
    time_block_cv_splits,
)


def _labels_with_events(N=1000, events=((100, 50), (400, 30), (700, 20))):
    y = np.zeros(N, dtype=np.int8)
    for start, length in events:
        y[start:start + length] = 1
    return y


# ══════════════════════════════════════════════════════════════════════════════
# time_block_cv_splits
# ══════════════════════════════════════════════════════════════════════════════

def test_time_block_cv_basic_shape():
    y = _labels_with_events()
    splits = time_block_cv_splits(y, n_folds=5)
    assert len(splits) == 5
    for train_idx, val_idx in splits:
        # Disjoint train and val
        assert len(np.intersect1d(train_idx, val_idx)) == 0
        # Together they cover all indices
        total = np.union1d(train_idx, val_idx)
        assert len(total) == len(y)


def test_time_block_cv_val_blocks_are_contiguous():
    y = _labels_with_events()
    splits = time_block_cv_splits(y, n_folds=5)
    for _, val_idx in splits:
        if len(val_idx) > 1:
            assert (np.diff(val_idx) == 1).all()


def test_time_block_cv_no_event_sliced_at_boundary():
    """No fold boundary lands strictly inside an anomaly event."""
    y = _labels_with_events()
    splits = time_block_cv_splits(y, n_folds=5)
    for _, val_idx in splits:
        if len(val_idx) == 0:
            continue
        start = val_idx[0]
        end   = val_idx[-1] + 1
        # The row just before val_start must not be mid-event with y[start]==1
        if 0 < start < len(y):
            if y[start] == 1 and y[start - 1] == 1:
                raise AssertionError(
                    f"fold starts inside an event at row {start}"
                )
        if 0 < end < len(y):
            if y[end - 1] == 1 and y[end] == 1:
                raise AssertionError(
                    f"fold ends inside an event at row {end}"
                )


def test_time_block_cv_rejects_too_few_folds():
    y = _labels_with_events(N=100, events=((10, 5),))
    with pytest.raises(ValueError):
        time_block_cv_splits(y, n_folds=1)


# ══════════════════════════════════════════════════════════════════════════════
# bootstrap_f05_ci
# ══════════════════════════════════════════════════════════════════════════════

def test_bootstrap_event_block_returns_expected_keys():
    y_true = _labels_with_events()
    y_pred = y_true.copy()
    out = bootstrap_f05_ci(y_true, y_pred, n_boot=30, seed=0)
    assert set(out.keys()) == {"mean", "std", "ci_lo_95", "ci_hi_95", "all_scores"}
    assert out["all_scores"].shape == (30,)
    assert out["ci_lo_95"] <= out["mean"] <= out["ci_hi_95"]


def test_bootstrap_perfect_pred_has_zero_variance():
    """With y_pred == y_true and proper event-block bootstrap that drops
    predictions in dropped-event regions, every bootstrap sample scores 1.0:
    kept events are perfectly predicted, dropped events and their predictions
    vanish together. No spurious FPs → no variance."""
    y_true = _labels_with_events()
    y_pred = y_true.copy()
    out = bootstrap_f05_ci(y_true, y_pred, n_boot=25, seed=1)
    assert out["mean"] == pytest.approx(1.0, abs=1e-6)
    assert out["std"]  == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(out["all_scores"], 1.0, atol=1e-6)


def test_bootstrap_produces_real_variance_under_mixed_preds():
    """When the predictor hits some events and misses others, the bootstrap
    must produce non-zero variance: resamples that happen to draw only-hit
    events score higher than resamples that draw missed events. This proves
    the fix still exposes meaningful sampling variation."""
    y_true = _labels_with_events()  # 3 events
    # Predict only the first event → 1 TP, 2 FN, no FP
    y_pred = np.zeros_like(y_true)
    y_pred[105:115] = 1
    out = bootstrap_f05_ci(y_true, y_pred, n_boot=80, seed=5)
    assert out["std"] > 0.01
    assert out["ci_lo_95"] < out["ci_hi_95"]


def test_bootstrap_no_events_degenerate_branch():
    """If y_true has no events, function short-circuits to the deterministic
    single-score branch."""
    y_true = np.zeros(500, dtype=np.int8)
    y_pred = np.zeros(500, dtype=np.int8)
    out = bootstrap_f05_ci(y_true, y_pred, n_boot=15, seed=2)
    # All scores identical → std 0, ci lo == hi
    assert out["std"] == pytest.approx(0.0)
    assert out["ci_lo_95"] == pytest.approx(out["ci_hi_95"])
    assert out["all_scores"].shape == (15,)


def test_bootstrap_row_level_branch_runs():
    y_true = _labels_with_events()
    y_pred = y_true.copy()
    out = bootstrap_f05_ci(
        y_true, y_pred, n_boot=20, seed=3, event_block=False
    )
    assert out["all_scores"].shape == (20,)


def test_bootstrap_custom_metric_and_key():
    """score_key routing for non-default metrics. With y_pred == y_true and
    the aligned event-block bootstrap, row_precision_recall's f1 stays at
    1.0 for every resample."""
    from sentinel.ml_logic.metrics import row_precision_recall
    y_true = _labels_with_events()
    y_pred = y_true.copy()
    out = bootstrap_f05_ci(
        y_true, y_pred,
        metric_fn=row_precision_recall, score_key="f1",
        n_boot=10, seed=4,
    )
    assert out["mean"] == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(out["all_scores"], 1.0, atol=1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# replay_submission
# ══════════════════════════════════════════════════════════════════════════════

def test_replay_submission_roundtrip(tmp_path):
    y_true = _labels_with_events()
    preds = y_true.copy()
    path = tmp_path / "sub.parquet"
    pd.DataFrame({"id": np.arange(len(preds)),
                  "is_anomaly": preds.astype(np.uint8)}) \
        .to_parquet(path, index=False)

    out = replay_submission(path, y_true)
    assert out["f_score"] == pytest.approx(1.0, abs=1e-6)
    assert out["source"] == str(path)


def test_replay_submission_length_mismatch_raises(tmp_path):
    y_true = _labels_with_events()
    path = tmp_path / "short.parquet"
    pd.DataFrame({"id": np.arange(500),
                  "is_anomaly": np.zeros(500, dtype=np.uint8)}) \
        .to_parquet(path, index=False)

    with pytest.raises(ValueError, match="length mismatch"):
        replay_submission(path, y_true, assert_length=True)


def test_replay_submission_truncate_option(tmp_path):
    """With assert_length=False, submission is truncated to min length."""
    y_true = _labels_with_events(N=1000)
    # Short parquet (first 500 rows) with the first event (100–149) intact
    path = tmp_path / "short.parquet"
    preds = y_true[:500].copy()
    pd.DataFrame({"id": np.arange(500),
                  "is_anomaly": preds.astype(np.uint8)}) \
        .to_parquet(path, index=False)

    out = replay_submission(path, y_true, assert_length=False)
    # No crash; got back a metric dict
    assert "f_score" in out


def test_replay_submission_missing_pred_column(tmp_path):
    y_true = _labels_with_events()
    path = tmp_path / "nocol.parquet"
    pd.DataFrame({"id": np.arange(len(y_true)),
                  "wrong_col": np.zeros(len(y_true), dtype=np.uint8)}) \
        .to_parquet(path, index=False)

    with pytest.raises(KeyError):
        replay_submission(path, y_true)
