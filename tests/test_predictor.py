"""Smoke tests for sentinel.ml_logic.predictor."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from sentinel.ml_logic.predictor import load_artefacts, predict


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


def test_predict_returns_expected_keys_and_shapes(fitted_stack):
    s = fitted_stack
    out = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=1e9, win=s["win"],          # never flags
    )

    assert set(out) == {
        "labels", "row_scores", "window_scores",
        "per_channel_mse", "window_channel_mse", "topk_channels",
        "threshold", "features",
    }
    n_rows = len(s["df"])
    n_feat = len(s["features"])
    assert out["labels"].shape            == (n_rows,)
    assert out["labels"].dtype            == np.int8
    assert out["row_scores"].shape        == (n_rows,)
    assert out["window_scores"].shape     == (s["n_win"],)
    assert out["per_channel_mse"].shape   == (n_feat,)
    assert out["window_channel_mse"].shape == (s["n_win"], n_feat)
    assert out["topk_channels"] is None
    assert out["threshold"] == pytest.approx(1e9)
    assert out["features"]  == s["features"]
    assert (out["labels"] == 0).all()


def test_predict_threshold_actually_flags(fitted_stack):
    s = fitted_stack
    out = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=-1.0, win=s["win"],         # always flags
    )
    assert (out["labels"] == 1).all()


def test_predict_accepts_ndarray(fitted_stack):
    s = fitted_stack
    out_df = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=1e9, win=s["win"],
    )
    out_np = predict(
        s["pca"], s["scaler"], s["features"], s["X"],
        threshold=1e9, win=s["win"],
    )
    assert np.allclose(out_df["row_scores"], out_np["row_scores"], atol=1e-5)


def test_predict_picks_columns_by_name(fitted_stack):
    """Shuffling DataFrame columns must not change the result."""
    s = fitted_stack
    shuffled = s["df"][["noise"] + s["features"][::-1]]
    out_a = predict(
        s["pca"], s["scaler"], s["features"], s["df"],
        threshold=1e9, win=s["win"],
    )
    out_b = predict(
        s["pca"], s["scaler"], s["features"], shuffled,
        threshold=1e9, win=s["win"],
    )
    assert np.allclose(out_a["row_scores"], out_b["row_scores"], atol=1e-5)


def test_load_artefacts_roundtrip(fitted_stack, tmp_path):
    s = fitted_stack
    model_path  = tmp_path / "pca.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    config_path = tmp_path / "preprocessing_config.json"
    with open(model_path, "wb")  as f: pickle.dump(s["pca"],    f)
    with open(scaler_path, "wb") as f: pickle.dump(s["scaler"], f)
    with open(config_path, "w")  as f:
        json.dump({"target_channels": s["features"]}, f)

    model, scaler, features = load_artefacts(model_path, scaler_path, config_path)
    assert features == s["features"]

    out = predict(model, scaler, features, s["df"],
                  threshold=1e9, win=s["win"])
    assert out["row_scores"].shape == (len(s["df"]),)


def test_load_artefacts_rejects_unknown_loader(tmp_path):
    with pytest.raises(ValueError):
        load_artefacts(tmp_path / "x", tmp_path / "y", tmp_path / "z",
                       loader="onnx")
