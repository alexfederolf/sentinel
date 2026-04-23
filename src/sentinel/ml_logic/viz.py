"""
Reusable plotting utilities for the SENTINEL anomaly-detection project.

Use this module from any notebook or script that needs to visualise
sensor channels, anomaly segments, value distributions, or correlation
matrices.  All functions return a ``matplotlib.figure.Figure`` so the
caller can save, display, or embed as needed.

Functions
---------
plot_channels        — multi-panel time-series with anomaly shading
plot_segment_zoom    — zoom into a single event with context
plot_distributions   — per-channel KDE histograms (nominal vs anomaly)
plot_correlation     — Pearson correlation heatmap across channels
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
