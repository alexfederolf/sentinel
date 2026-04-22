"""
Reusable plotting utilities for the SENTINEL anomaly-detection project.

Use this module from any notebook or script that needs to visualise
sensor channels, anomaly segments, value distributions, or correlation
matrices.  All functions return a ``matplotlib.figure.Figure`` so the
caller can save, display, or embed as needed.

Functions
---------
plot_channels                     — multi-panel time-series with anomaly shading
plot_segment_zoom                 — zoom into a single event with context
plot_distributions                — per-channel KDE histograms (nominal vs anomaly)
plot_correlation                  — Pearson correlation heatmap across channels
plot_score_distribution           — score histograms split by true label + threshold
plot_score_timeline               — score vs row index with threshold + GT shading
plot_event_zoom_with_score        — channel zoom plus PCA score panel
plot_confusion_and_channel_errors — confusion matrix + top-k channel MSE bar chart
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from ..params import ANOMALY_COLOR, NOMINAL_COLOR


def _shade_anomalies(ax: plt.Axes, index: np.ndarray, labels: np.ndarray) -> None:
    """Shade anomalous regions on a matplotlib Axes."""
    in_anom = False
    start = None
    for i, v in enumerate(labels):
        if v == 1 and not in_anom:
            start = index[i]
            in_anom = True
        elif v == 0 and in_anom:
            ax.axvspan(start, index[i - 1], color=ANOMALY_COLOR, alpha=0.25, linewidth=0)
            in_anom = False
    if in_anom:
        ax.axvspan(start, index[-1], color=ANOMALY_COLOR, alpha=0.25, linewidth=0)


def plot_channels(
    df: pd.DataFrame,
    channels: list[str],
    label_col: str = "is_anomaly",
    figsize: tuple = (18, 3),
    title: str = "Sensor channels over time",
    sample_frac: float = 0.05,
) -> plt.Figure:
    """
    Multi-panel time series plot for a list of channels.

    Anomalous periods are shaded red.  Down-sampled for display speed.

    Parameters
    ----------
    df          : DataFrame with channel columns and optionally a label column.
    channels    : list of column names to plot (one panel per channel).
    label_col   : name of the binary anomaly column.
    figsize     : (width, height_per_panel) — height scales with channel count.
    title       : figure title.
    sample_frac : fraction of rows to plot (random, seed=42).

    Returns
    -------
    fig : matplotlib Figure
    """
    df_s = df.sample(frac=sample_frac, random_state=42).sort_index()
    idx = df_s.index.values
    has_labels = label_col in df_s.columns

    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    sns.set_style("whitegrid")
    for ax, ch in zip(axes, channels):
        ax.plot(idx, df_s[ch].values, lw=0.6, color=NOMINAL_COLOR, alpha=0.8)
        if has_labels:
            _shade_anomalies(ax, idx, df_s[label_col].values)
        ax.set_ylabel(ch, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("Row index", fontsize=9)

    if has_labels:
        anom_patch = mpatches.Patch(color=ANOMALY_COLOR, alpha=0.4, label="Anomaly")
        axes[0].legend(handles=[anom_patch], fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


def plot_segment_zoom(
    df: pd.DataFrame,
    channels: list[str],
    seg_start: int,
    seg_end: int,
    context: int = 500,
    label_col: str = "is_anomaly",
    figsize: tuple = (16, 2.5),
) -> plt.Figure:
    """
    Zoom into one anomaly segment with surrounding context rows.

    Parameters
    ----------
    df        : full DataFrame (channel + label columns).
    channels  : channels to plot.
    seg_start : first row of the anomaly segment.
    seg_end   : last row of the anomaly segment.
    context   : number of rows to show before and after the segment.
    label_col : anomaly label column name.
    figsize   : (width, height_per_panel).

    Returns
    -------
    fig : matplotlib Figure
    """
    lo = max(0, seg_start - context)
    hi = min(len(df) - 1, seg_end + context)
    sub = df.iloc[lo:hi]
    idx = sub.index.values

    n = len(channels)
    fig, axes = plt.subplots(n, 1, figsize=(figsize[0], figsize[1] * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ch in zip(axes, channels):
        ax.plot(idx, sub[ch].values, lw=0.8, color=NOMINAL_COLOR)
        if label_col in sub.columns:
            _shade_anomalies(ax, idx, sub[label_col].values)
        ax.set_ylabel(ch, fontsize=9)

    axes[0].set_title(
        f"Anomaly segment rows {seg_start}–{seg_end} (±{context} context)", fontsize=11
    )
    axes[-1].set_xlabel("Row index", fontsize=9)
    fig.tight_layout()
    return fig


def plot_distributions(
    df: pd.DataFrame,
    channels: list[str],
    label_col: str = "is_anomaly",
    figsize_per_col: tuple = (4, 3),
    ncols: int = 4,
    sample_n: int = 50_000,
) -> plt.Figure:
    """
    KDE-overlaid histograms for each channel, split by anomaly vs nominal.

    Parameters
    ----------
    df             : DataFrame with channel and label columns.
    channels       : list of channel names to plot.
    label_col      : anomaly label column.
    figsize_per_col: (width, height) per subplot.
    ncols          : number of subplot columns.
    sample_n       : max rows to sample from each class.

    Returns
    -------
    fig : matplotlib Figure
    """
    nrows = int(np.ceil(len(channels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_col[0] * ncols, figsize_per_col[1] * nrows),
    )
    axes = np.array(axes).flatten()

    has_labels = label_col in df.columns
    if has_labels:
        nom = df[df[label_col] == 0].sample(
            min(sample_n, (df[label_col] == 0).sum()), random_state=42
        )
        anom = df[df[label_col] == 1].sample(
            min(sample_n, (df[label_col] == 1).sum()), random_state=42
        )

    for i, ch in enumerate(channels):
        ax = axes[i]
        if has_labels:
            sns.histplot(nom[ch], ax=ax, color=NOMINAL_COLOR, alpha=0.5,
                         stat="density", bins=50, label="Nominal", kde=True)
            sns.histplot(anom[ch], ax=ax, color=ANOMALY_COLOR, alpha=0.5,
                         stat="density", bins=50, label="Anomaly", kde=True)
        else:
            sub = df.sample(min(sample_n, len(df)), random_state=42)
            sns.histplot(sub[ch], ax=ax, bins=50, stat="density", kde=True)
        ax.set_title(ch, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(channels), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Channel value distributions: Nominal vs Anomaly", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation(
    df: pd.DataFrame,
    channels: list[str],
    sample_n: int = 100_000,
    figsize: tuple = (14, 12),
) -> plt.Figure:
    """
    Heatmap of Pearson correlations between channels.

    Parameters
    ----------
    df       : DataFrame with channel columns.
    channels : list of channel names to include.
    sample_n : max rows to sample for correlation calculation.
    figsize  : figure size.

    Returns
    -------
    fig : matplotlib Figure
    """
    sub = df[channels].sample(min(sample_n, len(df)), random_state=42)
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        annot=len(channels) <= 20,
        fmt=".2f",
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Channel Correlation Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Prediction-phase plots
# ----------------------------------------------------------------------------
# The four functions below visualise the output of a trained reconstruction
# model (PCA, LSTM-AE, CNN-AE). They take the row-level score array produced
# by `score_windows` and, optionally, a binary ground-truth label array.
# ══════════════════════════════════════════════════════════════════════════════


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Overlaid KDE histograms of anomaly scores, split by ground-truth label.

    A vertical line marks the decision threshold. The x-axis switches to log
    scale automatically when the score range spans more than two decades,
    which is typical for reconstruction MSE.

    Parameters
    ----------
    scores    : float array (n_rows,) — row-level anomaly scores
    y_true    : int array of 0/1 (n_rows,) — ground-truth labels
    threshold : decision threshold
    figsize   : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)

    nom  = scores[y_true == 0]
    anom = scores[y_true == 1]

    # Log-x when positive scores span > 2 decades
    pos = scores[scores > 0]
    use_log = pos.size > 0 and (pos.max() / max(pos.min(), 1e-12)) > 1e2

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")

    if use_log:
        # log-spaced bins so the histogram renders sensibly on a log x-axis
        lo = max(pos.min(), 1e-12)
        hi = pos.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 60)
    else:
        bins = 60

    if nom.size > 0:
        sns.histplot(nom, ax=ax, color=NOMINAL_COLOR, alpha=0.5,
                     stat="density", bins=bins, kde=True,
                     label=f"Nominal (n={nom.size:,})")
    if anom.size > 0:
        sns.histplot(anom, ax=ax, color=ANOMALY_COLOR, alpha=0.5,
                     stat="density", bins=bins, kde=True,
                     label=f"Anomaly (n={anom.size:,})")

    if use_log:
        ax.set_xscale("log")

    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:g}")
    ax.set_xlabel("Anomaly score" + (" (log scale)" if use_log else ""))
    ax.set_ylabel("Density")
    ax.set_title("Score distribution: Nominal vs Anomaly", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def plot_score_timeline(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    figsize: tuple = (18, 3),
    sample_frac: float | None = None,
) -> plt.Figure:
    """
    Row-index timeline of anomaly scores with threshold and ground-truth shading.

    Parameters
    ----------
    scores      : float array (n_rows,)
    y_true      : int array of 0/1 (n_rows,) — used to shade true anomaly runs
    threshold   : decision threshold (horizontal line)
    figsize     : figure size
    sample_frac : if set, random-sample this fraction of rows (seed=42) to keep
                  the plot responsive for multi-million-row arrays. The labels
                  are re-aligned so shading stays on the sampled index.

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    idx    = np.arange(len(scores))

    if sample_frac is not None and 0 < sample_frac < 1:
        rng = np.random.default_rng(42)
        keep = rng.random(len(scores)) < sample_frac
        # sort sampled indices so shading segments remain contiguous
        sel = np.sort(np.where(keep)[0])
        idx, scores, y_true = sel, scores[sel], y_true[sel]

    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    ax.plot(idx, scores, lw=0.5, color=NOMINAL_COLOR, alpha=0.9)
    _shade_anomalies(ax, idx, y_true)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.0,
               label=f"Threshold = {threshold:g}")

    ax.set_xlabel("Row index", fontsize=9)
    ax.set_ylabel("Anomaly score", fontsize=9)
    ax.set_title("Score timeline (red = true anomaly segments)",
                 fontsize=11, fontweight="bold")
    anom_patch = mpatches.Patch(color=ANOMALY_COLOR, alpha=0.4, label="True anomaly")
    ax.legend(handles=[anom_patch, ax.get_lines()[-1]], fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_event_zoom_with_score(
    df_raw: pd.DataFrame,
    scores: np.ndarray,
    seg_start: int,
    seg_end: int,
    channels: list[str],
    threshold: float,
    context: int = 500,
    label_col: str = "is_anomaly",
    figsize: tuple = (16, 2.5),
) -> plt.Figure:
    """
    Zoom into one anomaly segment: one panel per channel plus a bottom panel
    with the row-level anomaly score and its threshold.

    Anomalous periods are shaded across every panel.

    Parameters
    ----------
    df_raw    : DataFrame with channel and (optional) label columns. Its index
                must line up with ``scores`` row-for-row.
    scores    : float array (n_rows,)
    seg_start : first row of the true anomaly segment
    seg_end   : last row of the true anomaly segment
    channels  : channel column names to plot (one panel each)
    threshold : decision threshold (horizontal line on the score panel)
    context   : rows to include before/after the segment
    label_col : anomaly label column in df_raw
    figsize   : (width, height_per_panel)

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(df_raw)
    lo = max(0, seg_start - context)
    hi = min(n - 1, seg_end + context)

    sub   = df_raw.iloc[lo:hi + 1]
    s_sub = scores[lo:hi + 1]
    idx   = sub.index.values
    labels = sub[label_col].values if label_col in sub.columns else np.zeros(len(sub), dtype=np.int8)

    n_ch = len(channels)
    fig, axes = plt.subplots(
        n_ch + 1, 1,
        figsize=(figsize[0], figsize[1] * (n_ch + 1)),
        sharex=True,
    )

    for ax, ch in zip(axes[:-1], channels):
        ax.plot(idx, sub[ch].values, lw=0.8, color=NOMINAL_COLOR)
        _shade_anomalies(ax, idx, labels)
        ax.set_ylabel(ch, fontsize=9)

    ax_s = axes[-1]
    ax_s.plot(idx, s_sub, lw=0.9, color="#555555")
    _shade_anomalies(ax_s, idx, labels)
    ax_s.axhline(threshold, color="black", linestyle="--", linewidth=1.0,
                 label=f"Threshold = {threshold:g}")
    ax_s.set_ylabel("PCA score", fontsize=9)
    ax_s.set_xlabel("Row index", fontsize=9)
    ax_s.legend(fontsize=8, loc="upper right")

    axes[0].set_title(
        f"Event zoom rows {seg_start}–{seg_end} (±{context} context)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_confusion_and_channel_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_channel_mse: np.ndarray,
    channel_names: list[str],
    top_k: int = 10,
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """
    Two-panel diagnostic: row-level confusion matrix plus a bar chart of the
    top-k channels ranked by mean reconstruction MSE.

    Parameters
    ----------
    y_true          : int array of 0/1
    y_pred          : int array of 0/1
    per_channel_mse : float array (n_channels,) — mean MSE per channel
    channel_names   : list[str] aligned with ``per_channel_mse``
    top_k           : number of worst channels to show
    figsize         : figure size

    Returns
    -------
    fig : matplotlib Figure
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    per_channel_mse = np.asarray(per_channel_mse, dtype=np.float64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        cm, ax=ax_cm, annot=True, fmt="d",
        cmap="Blues", cbar=False, square=True,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_cm.set_title("Row-level confusion matrix", fontsize=11, fontweight="bold")

    k = min(top_k, len(channel_names))
    order = np.argsort(per_channel_mse)[::-1][:k]
    names = [channel_names[i] for i in order]
    vals  = per_channel_mse[order]

    sns.barplot(x=vals, y=names, ax=ax_bar, color=ANOMALY_COLOR)
    ax_bar.set_xlabel("Mean reconstruction MSE")
    ax_bar.set_ylabel("")
    ax_bar.set_title(f"Top {k} channels by reconstruction error",
                     fontsize=11, fontweight="bold")

    fig.tight_layout()
    return fig
