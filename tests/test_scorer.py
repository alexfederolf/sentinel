"""Tests for sentinel.ml_logic.scorer."""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from sentinel.ml_logic.scorer import (
    broadcast_window_scores_to_rows,
    score_report,
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


# ──────────────────────────────────────────────────────────────────────────────
# topk parity between window_scores_only and score_windows
# ──────────────────────────────────────────────────────────────────────────────

def test_window_scores_only_topk_matches_score_windows_topk():
    """Both views must agree under topk so the histogram and the timeline
    use the same definition of 'window score'."""
    rng = np.random.default_rng(3)
    win, n_feat, n_win = 10, 6, 8
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )
    win_only = window_scores_only(pca, X_rows, win=win, topk=2)
    row      = score_windows(pca, X_rows, win=win, topk=2)
    for i in range(n_win):
        assert np.allclose(row[i * win:(i + 1) * win], win_only[i], atol=1e-5)


def test_window_scores_only_rejects_bad_topk():
    rng = np.random.default_rng(4)
    win, n_feat, n_win = 10, 4, 3
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=2, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )
    with pytest.raises(ValueError):
        window_scores_only(pca, X_rows, win=win, topk=0)
    with pytest.raises(ValueError):
        window_scores_only(pca, X_rows, win=win, topk=n_feat + 1)


# ──────────────────────────────────────────────────────────────────────────────
# score_report
# ──────────────────────────────────────────────────────────────────────────────

def test_score_report_keys_and_shapes_pca():
    rng = np.random.default_rng(10)
    win, n_feat, n_win = 10, 5, 7
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )

    out = score_report(pca, X_rows, win=win)

    assert set(out) == {
        "row_scores", "window_scores", "per_channel_mse",
        "window_channel_mse", "topk_channels",
    }
    assert out["row_scores"].shape         == (n_win * win,)
    assert out["window_scores"].shape      == (n_win,)
    assert out["per_channel_mse"].shape    == (n_feat,)
    assert out["window_channel_mse"].shape == (n_win, n_feat)
    assert out["topk_channels"] is None

    # Dtypes
    assert out["row_scores"].dtype         == np.float32
    assert out["window_scores"].dtype      == np.float32
    assert out["per_channel_mse"].dtype    == np.float32
    assert out["window_channel_mse"].dtype == np.float32


def test_score_report_matches_score_windows_and_window_scores_only():
    """score_report must not drift from the existing scalar paths."""
    rng = np.random.default_rng(11)
    win, n_feat, n_win = 20, 6, 5
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )

    out = score_report(pca, X_rows, win=win)

    assert np.allclose(out["row_scores"],    score_windows(pca, X_rows, win=win))
    assert np.allclose(out["window_scores"], window_scores_only(pca, X_rows, win=win))


def test_score_report_per_channel_mse_is_mean_over_windows():
    rng = np.random.default_rng(12)
    win, n_feat, n_win = 15, 4, 6
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=2, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )

    out = score_report(pca, X_rows, win=win)
    # per_channel_mse must equal mean of window_channel_mse across windows
    expected = out["window_channel_mse"].mean(axis=0)
    assert np.allclose(out["per_channel_mse"], expected, atol=1e-6)


def test_score_report_topk_channels_pick_largest():
    rng = np.random.default_rng(13)
    win, n_feat, n_win = 10, 5, 4
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    pca = PCA(n_components=2, random_state=0).fit(
        X_rows.reshape(n_win, win * n_feat)
    )

    out = score_report(pca, X_rows, win=win, topk=2)

    assert out["topk_channels"].shape == (n_win, 2)
    # Each picked index must be among the two largest per-channel MSEs.
    for i in range(n_win):
        ranked = np.argsort(out["window_channel_mse"][i])[::-1]
        top2   = set(ranked[:2])
        assert set(out["topk_channels"][i].tolist()) == top2


def test_score_report_short_input_returns_zeros():
    """Fewer than one full window → empty window arrays, zero row scores,
    no model dispatch."""
    class Useless:
        pass

    X = np.zeros((5, 3), dtype=np.float32)   # win=10 → 0 complete windows
    out = score_report(Useless(), X, win=10)

    assert out["row_scores"].shape         == (5,)
    assert (out["row_scores"] == 0.0).all()
    assert out["window_scores"].shape      == (0,)
    assert out["per_channel_mse"].shape    == (3,)
    assert out["window_channel_mse"].shape == (0, 3)
    assert out["topk_channels"] is None


def test_score_report_with_keras_like_model():
    """Zero-reconstruction model: per_channel_mse == mean of X**2 per channel."""
    class ZeroReconstruction:
        def predict(self, X, batch_size=None, verbose=0):
            return np.zeros_like(X)

    rng = np.random.default_rng(14)
    win, n_feat, n_win = 10, 3, 4
    X_rows = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)

    out = score_report(ZeroReconstruction(), X_rows, win=win, batch=2)

    expected_per_channel = (X_rows ** 2).mean(axis=0)
    assert np.allclose(out["per_channel_mse"], expected_per_channel, atol=1e-5)
