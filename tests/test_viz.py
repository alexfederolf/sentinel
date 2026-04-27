"""
Smoke tests for the prediction-phase plots in ``sentinel.ml_logic.viz``.

We don't assert on visual content — matplotlib snapshot tests are
fragile. The goal here is to make sure the four prediction-phase plots
accept the documented inputs, return a ``plt.Figure``, and produce at
least one axes. That catches the usual bugs (shape mismatches, wrong
column names, broken imports) without coupling to pixel output.
"""
import matplotlib
matplotlib.use("Agg")   # headless backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sentinel.ml_logic.viz import (
    plot_confusion_and_channel_errors,
    plot_event_zoom_with_score,
    plot_score_distribution,
    plot_score_timeline,
)


N_ROWS = 500
N_CH   = 10


@pytest.fixture
def synthetic():
    """Small reproducible dataset: scores + labels + raw channel DataFrame."""
    rng = np.random.default_rng(0)
    y_true = np.zeros(N_ROWS, dtype=np.int8)
    y_true[200:260] = 1           # one 60-row anomaly segment
    y_true[400:410] = 1           # one 10-row anomaly segment

    # Scores: nominal ~ N(0.05, 0.02), anomaly ~ N(0.3, 0.05)
    scores = rng.normal(0.05, 0.02, size=N_ROWS).astype(np.float32)
    scores[y_true == 1] = rng.normal(0.3, 0.05, size=int(y_true.sum())).astype(np.float32)
    scores = np.clip(scores, 0.0, None)

    threshold = 0.15
    y_pred = (scores > threshold).astype(np.int8)

    channels = [f"channel_{i}" for i in range(N_CH)]
    df = pd.DataFrame(
        rng.normal(size=(N_ROWS, N_CH)).astype(np.float32),
        columns=channels,
    )
    df["is_anomaly"] = y_true

    per_channel_mse = rng.random(N_CH).astype(np.float32)

    return {
        "scores": scores,
        "y_true": y_true,
        "y_pred": y_pred,
        "threshold": threshold,
        "df": df,
        "channels": channels,
        "per_channel_mse": per_channel_mse,
    }


def _assert_figure(fig):
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_score_distribution_returns_figure(synthetic):
    fig = plot_score_distribution(
        synthetic["scores"], synthetic["y_true"], synthetic["threshold"],
    )
    _assert_figure(fig)


def test_plot_score_distribution_log_scale_branch():
    """Force the log-x branch with a > 2-decade range."""
    scores = np.array([1e-4, 1e-3, 1e-2, 1.0, 10.0, 100.0], dtype=np.float32)
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int8)
    fig = plot_score_distribution(scores, y_true, threshold=0.5)
    _assert_figure(fig)


def test_plot_score_timeline_returns_figure(synthetic):
    fig = plot_score_timeline(
        synthetic["scores"], synthetic["y_true"], synthetic["threshold"],
    )
    _assert_figure(fig)


def test_plot_score_timeline_with_sampling(synthetic):
    fig = plot_score_timeline(
        synthetic["scores"], synthetic["y_true"], synthetic["threshold"],
        sample_frac=0.5,
    )
    _assert_figure(fig)


def test_plot_event_zoom_with_score_returns_figure(synthetic):
    fig = plot_event_zoom_with_score(
        df_raw=synthetic["df"],
        scores=synthetic["scores"],
        seg_start=200,
        seg_end=259,
        channels=synthetic["channels"][:3],
        threshold=synthetic["threshold"],
        context=50,
    )
    # n_channels + 1 score panel
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_event_zoom_handles_missing_label_col(synthetic):
    df = synthetic["df"].drop(columns=["is_anomaly"])
    fig = plot_event_zoom_with_score(
        df_raw=df,
        scores=synthetic["scores"],
        seg_start=200,
        seg_end=259,
        channels=synthetic["channels"][:2],
        threshold=synthetic["threshold"],
        context=20,
    )
    _assert_figure(fig)


def test_plot_confusion_and_channel_errors_returns_figure(synthetic):
    fig = plot_confusion_and_channel_errors(
        synthetic["y_true"],
        synthetic["y_pred"],
        synthetic["per_channel_mse"],
        synthetic["channels"],
        top_k=5,
    )
    assert len(fig.axes) == 2   # confusion + bar chart
    plt.close(fig)


def test_plot_confusion_top_k_clamps_when_too_large(synthetic):
    """top_k greater than channel count should not crash."""
    fig = plot_confusion_and_channel_errors(
        synthetic["y_true"],
        synthetic["y_pred"],
        synthetic["per_channel_mse"],
        synthetic["channels"],
        top_k=N_CH + 10,
    )
    _assert_figure(fig)
