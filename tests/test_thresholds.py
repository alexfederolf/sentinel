"""Tests for sentinel.ml_logic.thresholds.tune_threshold."""
import numpy as np
import pytest

from sentinel.ml_logic.thresholds import tune_threshold


def _synthetic_scores(n_nom=900, n_anom=100, seed=0):
    """Nominal scores concentrated near 0.1, anomaly scores near 1.0."""
    rng = np.random.default_rng(seed)
    nom  = rng.normal(loc=0.1, scale=0.02, size=n_nom).clip(min=1e-4)
    anom = rng.normal(loc=1.0, scale=0.05, size=n_anom).clip(min=1e-4)
    scores = np.concatenate([nom, anom]).astype(np.float32)
    y = np.concatenate([np.zeros(n_nom, np.int8), np.ones(n_anom, np.int8)])
    # Keep the anomaly block contiguous so event-wise metrics see 1 event.
    return scores, y


def test_tune_threshold_finds_separating_value():
    scores, y = _synthetic_scores()
    out = tune_threshold(scores, y, n_sweep=40)
    # With perfectly separable distributions we should score close to 1.0
    assert out["score"] > 0.9
    # Chosen threshold lies between the two distributions
    assert 0.15 < out["threshold"] < 0.9


def test_tune_threshold_returns_expected_keys_and_shapes():
    scores, y = _synthetic_scores()
    out = tune_threshold(scores, y, n_sweep=25)
    assert set(out.keys()) == {
        "threshold", "score", "sweep_thresholds", "sweep_scores",
    }
    assert out["sweep_thresholds"].shape == (25,)
    assert out["sweep_scores"].shape == (25,)
    # geomspace is monotonically increasing
    assert (np.diff(out["sweep_thresholds"]) > 0).all()


def test_tune_threshold_custom_scalar_metric():
    """score_key=None supports metric_fn that returns a bare float."""
    scores, y = _synthetic_scores()

    def scalar_metric(y_true, y_pred):
        # Row-level accuracy — a bare float, not a dict
        return float((y_true == y_pred).mean())

    out = tune_threshold(
        scores, y, metric_fn=scalar_metric, score_key=None, n_sweep=20
    )
    assert out["score"] > 0.95  # nearly perfectly separable


def test_tune_threshold_degenerate_constant_scores():
    """Constant scores collapse the range; function must not blow up."""
    scores = np.full(200, 0.5, dtype=np.float32)
    y = np.zeros(200, dtype=np.int8)
    y[50:60] = 1
    out = tune_threshold(scores, y, n_sweep=10)
    # Should return without error and produce a real sweep.
    assert np.isfinite(out["threshold"])
    assert out["sweep_scores"].shape == (10,)


def test_tune_threshold_argmax_agrees_with_sweep():
    scores, y = _synthetic_scores()
    out = tune_threshold(scores, y, n_sweep=30)
    best_idx = int(np.argmax(out["sweep_scores"]))
    assert out["threshold"] == pytest.approx(out["sweep_thresholds"][best_idx])
    assert out["score"]     == pytest.approx(out["sweep_scores"][best_idx])
