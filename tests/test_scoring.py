"""Tests for sentinel.ml_logic.scoring."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from sentinel.ml_logic.scoring import (
    broadcast_window_scores_to_rows,
    score_windows,
    window_scores_only,
)


def test_broadcast_window_scores_to_rows_exact():
    win_scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = broadcast_window_scores_to_rows(win_scores, n_rows=30, win=10)
    assert out.shape == (30,)
    # Each window's score fills its 10 rows
    assert (out[:10] == 1.0).all()
    assert (out[10:20] == 2.0).all()
    assert (out[20:30] == 3.0).all()


def test_broadcast_tail_inherits_last_window():
    win_scores = np.array([1.0, 2.0], dtype=np.float32)
    out = broadcast_window_scores_to_rows(win_scores, n_rows=25, win=10)
    assert out.shape == (25,)
    assert (out[20:25] == 2.0).all()   # tail inherits last full window


def test_broadcast_truncates_when_rows_match_multiple():
    # n_rows exactly divisible — no tail padding
    win_scores = np.arange(5, dtype=np.float32)
    out = broadcast_window_scores_to_rows(win_scores, n_rows=50, win=10)
    assert len(out) == 50
    assert out[0] == 0 and out[49] == 4


def test_broadcast_empty_window_scores():
    out = broadcast_window_scores_to_rows(
        np.array([], dtype=np.float32), n_rows=100, win=10
    )
    assert out.shape == (100,)
    assert (out == 0.0).all()


def test_score_windows_with_pca_matches_hand_computation():
    """PCA reconstruction path: score_windows result must equal the direct
    (flatten → reconstruct → MSE) computation."""
    rng = np.random.default_rng(0)
    win, n_feat, n_win = 20, 6, 8
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(
        X_rows[: n_win * win].reshape(n_win, win * n_feat)
    )

    scores = score_windows(pca, X_rows, win=win)
    assert scores.shape == (n_win * win,)

    # Hand-compute expected window scores
    X_flat = X_rows.reshape(n_win, win * n_feat)
    X_hat  = pca.inverse_transform(pca.transform(X_flat))
    expected_win = ((X_flat - X_hat) ** 2).mean(axis=1)
    # Broadcast
    for i in range(n_win):
        assert np.allclose(scores[i * win:(i + 1) * win], expected_win[i], atol=1e-5)


def test_score_windows_with_keras_like_model(monkeypatch):
    """A bare object exposing .predict(X_3d) must dispatch through the
    Keras branch. We use a fake model that zero-reconstructs so that
    MSE = mean(X**2)."""
    class ZeroReconstruction:
        def predict(self, X, batch_size=None, verbose=0):
            return np.zeros_like(X)

    rng = np.random.default_rng(1)
    win, n_feat, n_win = 10, 3, 4
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    scores = score_windows(ZeroReconstruction(), X_rows, win=win, batch=2)
    assert scores.shape == (n_win * win,)
    # Each window's score should equal mean(X**2) over (time, channel) axes
    X_win = X_rows.reshape(n_win, win, n_feat)
    expected = (X_win ** 2).mean(axis=(1, 2))
    for i in range(n_win):
        assert np.allclose(scores[i * win:(i + 1) * win], expected[i], atol=1e-5)


def test_score_windows_rejects_unsupported_model():
    class Useless:
        pass
    X = np.zeros((100, 5), dtype=np.float32)
    with pytest.raises(TypeError):
        score_windows(Useless(), X, win=10)


def test_score_windows_short_input_returns_zeros():
    """Fewer than one full window → zeros, no crash."""
    class Dummy:
        def predict(self, X, **kwargs): return np.zeros_like(X)

    X = np.zeros((5, 3), dtype=np.float32)  # win=10 → 0 complete windows
    out = score_windows(Dummy(), X, win=10)
    assert out.shape == (5,)
    assert (out == 0.0).all()


def test_window_scores_only_consistent_with_score_windows():
    """window_scores_only must return the same per-window array that
    score_windows broadcasts."""
    rng = np.random.default_rng(2)
    win, n_feat, n_win = 10, 4, 5
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=2, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )
    win_only = window_scores_only(pca, X_rows, win=win)
    row      = score_windows(pca, X_rows, win=win)
    for i in range(n_win):
        assert np.allclose(row[i * win:(i + 1) * win], win_only[i], atol=1e-5)
