"""Smoke tests for sentinel.ml_logic.predictor."""
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from sentinel.ml_logic.predictor import _load, predict, predict_report


@pytest.fixture
def fitted_stack():
    rng = np.random.default_rng(0)
    win, n_feat, n_win = 10, 6, 8
    features = [f"ch_{i}" for i in range(n_feat)]

    X = rng.normal(size=(n_win * win, n_feat)).astype(np.float32)
    scaler = RobustScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)
    pca = PCA(n_components=3, random_state=0).fit(
        X_scaled.reshape(n_win, win * n_feat)
    )

    df = pd.DataFrame(X, columns=features)
    # distractor column so the DataFrame path has to pick by name
    df["noise"] = rng.normal(size=len(df)).astype(np.float32)

    return {
        "scaler": scaler, "pca": pca, "features": features,
        "df": df, "X": X, "win": win, "n_win": n_win,
    }


# ── simple predict: Kaggle submission DataFrame (id, is_anomaly) ──────────

def test_predict_returns_submission_frame(fitted_stack):
    s = fitted_stack
    sub = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=1e9, win=s["win"],          # never flags
    )
    assert isinstance(sub, pd.DataFrame)
    assert list(sub.columns) == ["id", "is_anomaly"]
    assert len(sub) == len(s["df"])
    assert sub["is_anomaly"].dtype == np.int8
    assert (sub["is_anomaly"] == 0).all()
    assert (sub["id"].values == np.arange(len(sub))).all()


def test_predict_threshold_actually_flags(fitted_stack):
    s = fitted_stack
    sub = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=-1.0, win=s["win"],         # always flags
    )
    assert (sub["is_anomaly"] == 1).all()


def test_predict_uses_id_column_when_present(fitted_stack):
    """If X_raw has an 'id' column, it carries through to the submission."""
    s = fitted_stack
    df_with_id = s["df"].copy()
    df_with_id["id"] = np.arange(1000, 1000 + len(df_with_id), dtype=np.int64)
    sub = predict(
        s["pca"], s["scaler"], s["features"], df_with_id,
        threshold=1e9, win=s["win"],
    )
    assert (sub["id"].values == df_with_id["id"].values).all()


def test_predict_accepts_ndarray(fitted_stack):
    s = fitted_stack
    sub = predict(
        s["pca"], s["scaler"], s["features"], s["X"],
        threshold=1e9, win=s["win"],
    )
    assert list(sub.columns) == ["id", "is_anomaly"]
    assert len(sub) == len(s["X"])
    assert (sub["id"].values == np.arange(len(sub))).all()


# ── detailed predict_report: full dict ────────────────────────────────────

def test_predict_report_returns_expected_keys_and_shapes(fitted_stack):
    s = fitted_stack
    out = predict_report(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=1e9, win=s["win"],
    )
    assert set(out) == {
        "labels", "row_scores", "window_scores",
        "per_channel_mse", "window_channel_mse", "window_top_channels",
        "threshold", "features",
    }
    n_rows = len(s["df"])
    n_feat = len(s["features"])
    assert out["labels"].shape             == (n_rows,)
    assert out["labels"].dtype             == np.int8
    assert out["row_scores"].shape         == (n_rows,)
    assert out["window_scores"].shape      == (s["n_win"],)
    assert out["per_channel_mse"].shape    == (n_feat,)
    assert out["window_channel_mse"].shape == (s["n_win"], n_feat)
    assert out["window_top_channels"] is None
    assert out["threshold"] == pytest.approx(1e9)
    assert out["features"]  == s["features"]


def test_predict_and_predict_report_agree_on_labels(fitted_stack):
    """Both variants must flag the same rows under the same threshold."""
    s = fitted_stack
    sub = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=0.5, win=s["win"],
    )
    rep = predict_report(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=0.5, win=s["win"],
    )
    assert (sub["is_anomaly"].values == rep["labels"]).all()


# ── _load: defaults only fill in what's missing ──────────────────────────

def test_load_passes_through_explicit_values(fitted_stack):
    """When every argument is supplied, _load returns them untouched — no disk hit."""
    s = fitted_stack
    model, scaler, features, X_raw = _load(
        model=s["pca"], scaler=s["scaler"],
        features=s["features"], X_raw=s["X"],
    )
    assert model    is s["pca"]
    assert scaler   is s["scaler"]
    assert features == s["features"]
    assert X_raw    is s["X"]
