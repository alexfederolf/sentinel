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
plot_score_panels                 — 3-panel: scores, true anomalies, predicted anomalies
plot_timeline                     — 2-panel: MSE (linear/log x/y) + true/predicted ribbon
plot_event_analysis               — per-event detection bars + length-bucket summary
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
from .data import find_anomaly_segments


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
    ax.set_title("Score timeline",
                 fontsize=11, fontweight="bold")
    anom_patch = mpatches.Patch(color=ANOMALY_COLOR, alpha=0.4, label="True anomaly")
    ax.legend(handles=[anom_patch, ax.get_lines()[-1]], fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_score_panels(
    scores: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    index: np.ndarray | None = None,
    figsize: tuple = (18, 6),
    sample_frac: float | None = None,
) -> plt.Figure:
    """
    Three stacked panels sharing the same x-axis:

        1. Row-level reconstruction error (score) with the threshold line
        2. Ground-truth anomaly ribbon
        3. Predicted anomaly ribbon

    Splitting scores from labels makes it visually obvious where the model
    over- or under-fires compared to ground truth — something the overlaid
    ``plot_score_timeline`` can hide when shaded regions pile up.

    Parameters
    ----------
    scores      : (n_rows,) row-level anomaly score
    y_true      : (n_rows,) 0/1 ground-truth labels
    y_pred      : (n_rows,) 0/1 predicted labels (same threshold that is drawn)
    threshold   : decision threshold (horizontal line on the score panel)
    index       : x-axis values, e.g. ``df.index.values``. Defaults to ``np.arange(n_rows)``.
    figsize     : figure size (width, total height)
    sample_frac : optional random sub-sample (seed=42) to keep huge arrays responsive

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    if index is None:
        index = np.arange(len(scores))
    else:
        index = np.asarray(index)

    if sample_frac is not None and 0 < sample_frac < 1:
        rng = np.random.default_rng(42)
        keep = rng.random(len(scores)) < sample_frac
        sel = np.sort(np.where(keep)[0])
        index, scores, y_true, y_pred = index[sel], scores[sel], y_true[sel], y_pred[sel]

    sns.set_style("whitegrid")
    fig, (ax_s, ax_t, ax_p) = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    # Panel 1: reconstruction error + threshold
    ax_s.plot(index, scores, lw=0.5, color=NOMINAL_COLOR, alpha=0.9)
    ax_s.axhline(threshold, color="black", linestyle="--", linewidth=1.0,
                 label=f"Threshold = {threshold:g}")
    ax_s.set_ylabel("Reconstruction\nerror", fontsize=9)
    ax_s.set_title("Score timeline — errors, ground truth, predictions",
                   fontsize=11, fontweight="bold")
    ax_s.legend(fontsize=8, loc="upper right")

    # Panel 2: ground-truth ribbon
    ax_t.fill_between(index, 0, y_true, step="pre",
                      color=ANOMALY_COLOR, alpha=0.7, linewidth=0)
    ax_t.set_ylim(-0.1, 1.1)
    ax_t.set_yticks([0, 1])
    ax_t.set_ylabel("True\nanomaly", fontsize=9)

    # Panel 3: predicted ribbon
    ax_p.fill_between(index, 0, y_pred, step="pre",
                      color=ANOMALY_COLOR, alpha=0.7, linewidth=0)
    ax_p.set_ylim(-0.1, 1.1)
    ax_p.set_yticks([0, 1])
    ax_p.set_ylabel("Predicted\nanomaly", fontsize=9)
    ax_p.set_xlabel("Row index", fontsize=9)

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
    total = cm.sum()
    # cell labels: raw count on top, percentage of all rows below
    annot = np.array([[f"{v:,}\n{v / total:.1%}" for v in row] for row in cm])

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=figsize)

    sns.heatmap(
        cm, ax=ax_cm, annot=annot, fmt="",
        cmap="Blues", cbar=False, square=True,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_cm.set_title(f"Row-level confusion matrix (n = {total:,})",
                    fontsize=11, fontweight="bold")

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


def plot_timeline(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    title: str = "Timeline",
    ds: int = 500,
    log_y: bool = False,
    log_x: bool = False,
    index: np.ndarray | None = None,
    figsize: tuple = (17, 6),
) -> plt.Figure:
    """
    Top panel: per-row MSE score with the decision threshold drawn as a dashed
    line and true-anomaly bands shaded lightly across the full height.
    Bottom panel: overlaid ribbons of *predicted* vs *true* anomaly labels,
    so over-flagged nominal regions become immediately visible.

    For the showcase, call with ``log_y=True`` — a single large-magnitude
    event (e.g. the ~650 spike in Val) otherwise flattens every smaller
    event to near-zero on linear scale; log spreads them out so you can
    see which true-anomaly bands actually produce a visible score bump
    and which don't. ``log_x`` is available for symmetry (e.g. when the
    early part of the timeline is much denser in events than the late part).

    The score curve is down-sampled via **block-max aggregation** (each plotted
    point is the max over ``ds`` consecutive rows) so short score peaks remain
    visible regardless of ``ds``.  The bottom ribbon is drawn as spans from the
    full-resolution segment lists of ``y_true`` and ``y_pred`` — so even a
    single-row prediction is still visible at any ``ds``.
    This has **no impact on any metric** — metrics should be computed
    separately on the full-resolution ``scores`` / ``y_true``.

    Parameters
    ----------
    scores    : (n_rows,) float — row-level anomaly scores
    y_true    : (n_rows,) 0/1   — ground-truth labels
    threshold : float           — decision threshold
    title     : figure title
    ds        : int             — downsample stride for plotting (no metric impact)
    log_y     : bool            — log scale on the MSE (y) axis
    log_x     : bool            — log scale on the row-index (x) axis
    index     : (n_rows,) optional — x-axis values, same length as ``scores``.
                Defaults to ``np.arange(n_rows)`` (0-based positions). Pass
                e.g. ``df_test.index.values`` to show the *absolute* row
                indices from the original dataset on the x-axis instead of
                the 0-based positions within the split.
    figsize   : (width, height) — forwarded to plt.subplots

    Returns
    -------
    fig : matplotlib Figure
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int8)
    n      = len(scores)

    # Resolve x-axis values: caller-supplied or 0..n-1.
    if index is None:
        x_full = np.arange(n)
    else:
        x_full = np.asarray(index)
        if len(x_full) != n:
            raise ValueError(
                f"index length {len(x_full)} must match scores length {n}"
            )

    segments      = find_anomaly_segments(y_true)
    y_pred        = (scores > threshold).astype(np.int8)
    pred_segments = find_anomaly_segments(y_pred)

    # Downsample for plot speed using *block-max* aggregation so short score
    # peaks and sparse predictions survive — stride sampling would miss them.
    # Each output point = max of its `ds`-row block. x position = block start.
    n_full   = (n // ds) * ds
    if n_full > 0:
        scores_blk = scores[:n_full].reshape(-1, ds).max(axis=1)
        x_blk      = x_full[:n_full:ds]
    else:
        scores_blk = scores
        x_blk      = x_full
    # Append the tail (last partial block) so the right edge isn't cut off.
    if n_full < n:
        scores_blk = np.concatenate([scores_blk, [scores[n_full:].max()]])
        x_blk      = np.concatenate([x_blk,      [x_full[n_full]]])

    scores_ds = scores_blk
    x_ds      = x_blk

    fig, axes = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1]},
    )
    ax, ax2 = axes

    # ── Axis scaling ────────────────────────────────────────────────────────
    # Log scales need strictly-positive values; floor non-positives to a small
    # epsilon below the smallest positive so nothing is clipped off the axis.
    if log_y:
        pos = scores_ds[scores_ds > 0]
        eps_y = float(pos.min()) * 0.5 if pos.size else 1e-9
        scores_plot = np.where(scores_ds > 0, scores_ds, eps_y)
        ax.set_yscale("log")
    else:
        scores_plot = scores_ds

    if log_x:
        pos_x = x_ds[x_ds > 0]
        eps_x = float(pos_x.min()) * 0.5 if pos_x.size else 1.0
        x_plot = np.where(x_ds > 0, x_ds, eps_x)
        # sharex=True propagates limits; set scale on both axes to be safe.
        ax.set_xscale("log")
        ax2.set_xscale("log")
    else:
        x_plot = x_ds

    # Minimum visible span for event rectangles: at full-dataset zoom, a short
    # event (1–5 rows) is sub-pixel wide on a ~2 M-row axis and won't render.
    # Enforce a minimum visual width of ~0.3% of the plotted range.
    x_range  = float(x_full[-1] - x_full[0]) if n > 1 else 1.0
    min_span = max(1.0, x_range * 0.003)

    def _expand(seg):
        """Convert segment dict → (x_s, x_e) in plot coords, min-span-padded."""
        s_pos = min(seg["start"], n - 1)
        e_pos = min(seg["end"],   n - 1)
        x_s   = float(x_full[s_pos])
        x_e   = float(x_full[e_pos])
        if x_e - x_s < min_span:
            mid = 0.5 * (x_s + x_e)
            x_s = mid - 0.5 * min_span
            x_e = mid + 0.5 * min_span
        if log_x:
            if x_s <= 0: x_s = eps_x
            if x_e <= 0: x_e = eps_x
        return x_s, x_e

    def _merge(pairs):
        """Union overlapping/touching (x_s, x_e) intervals → list of disjoint pairs.
        Avoids alpha-stacking when adjacent events are expanded to min_span."""
        if not pairs:
            return []
        pairs = sorted(pairs)
        merged = [list(pairs[0])]
        for a, b in pairs[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return merged

    def _spans(axis, segs, y_lo, y_hi, color, alpha):
        for x_s, x_e in _merge([_expand(s) for s in segs]):
            axis.fill_between([x_s, x_e], y_lo, y_hi,
                              color=color, alpha=alpha, linewidth=0)

    # ── Top panel: True-anomaly bands behind the score curve ───────────────
    # Bands cover the full y-range via axvspan; merged after expansion so
    # close-together events don't alpha-stack into darker regions.
    for x_s, x_e in _merge([_expand(s) for s in segments]):
        ax.axvspan(x_s, x_e, color=ANOMALY_COLOR, alpha=0.45, lw=0)

    ax.plot(x_plot, scores_plot, lw=0.6, color="#5b9bd5", alpha=0.95)
    ax.axhline(threshold, color=ANOMALY_COLOR, lw=1.5, ls="--")

    ax.set_ylabel("MSE score" + (" (log)" if log_y else ""))
    ax.set_title(title, fontweight="bold", fontsize=12)

    handles = [
        plt.Line2D([0], [0], color="#5b9bd5", lw=1, label="Recon MSE"),
        plt.Line2D([0], [0], color=ANOMALY_COLOR, lw=1.5, ls="--",
                   label=f"Threshold {threshold:.4f}"),
        mpatches.Patch(color=ANOMALY_COLOR, alpha=0.45,
                       label=f"True anomaly ({len(segments)} events)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    # ── Bottom panel: predicted events, split into TP vs FP by overlap ─────
    # A predicted event is TP iff it overlaps any true segment (event-wise).
    tp_segs, fp_segs = [], []
    for pseg in pred_segments:
        ps, pe = pseg["start"], pseg["end"]
        hit = any(not (pe < tseg["start"] or ps > tseg["end"]) for tseg in segments)
        (tp_segs if hit else fp_segs).append(pseg)

    TP_COLOR = "#8fbc8f"  # sage / darkseagreen — soft, muted
    FP_COLOR = "#b19cd9"  # soft lavender — muted, matches the sage/salmon palette
    _spans(ax2, tp_segs, 0.0, 1.0, TP_COLOR, 0.85)
    _spans(ax2, fp_segs, 0.0, 1.0, FP_COLOR, 0.85)

    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.5])
    ax2.set_yticklabels(["Pred"])
    ax2.set_ylabel("Anomaly")
    ax2.set_xlabel("Row index" + (" (log)" if log_x else ""))
    legend_handles = [
        mpatches.Patch(color=TP_COLOR, alpha=0.85,
                       label=f"TP ({len(tp_segs)})"),
        mpatches.Patch(color=FP_COLOR, alpha=0.85,
                       label=f"FP ({len(fp_segs)})"),
    ]
    ax2.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


def plot_event_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Event analysis",
    figsize: tuple = (16, 4),
) -> plt.Figure | None:
    """
    Per-event detection diagnostic — three side-by-side panels:

    1. Bar chart: counts of *Detected* vs *Missed* events.
    2. Scatter:  event length (log-x) vs hit-rate, colored by detected.
    3. Histogram of per-event hit-rate.

    Style mirrors the z-score notebook (nb21) so the output is consistent
    across baselines.

    Parameters
    ----------
    y_true  : (n_rows,) 0/1 — ground-truth labels
    y_pred  : (n_rows,) 0/1 — predictions at the chosen threshold
    title   : figure suptitle
    figsize : forwarded to plt.subplots

    Returns
    -------
    fig : matplotlib Figure (or None if no events).  When events exist a
          summary line is also printed listing missed events (start/end/length).
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)
    segments = find_anomaly_segments(y_true)

    rows = []
    for seg in segments:
        s, e = seg["start"], seg["end"]
        L    = seg["length"]
        hits = int(y_pred[s:e + 1].sum())
        rows.append({
            "start"   : s,
            "end"     : e,
            "length"  : L,
            "detected": hits > 0,
            "hit_rate": round(hits / L, 3) if L > 0 else 0.0,
        })
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print(f"{title}: no true events.")
        return None

    n_det  = int(df["detected"].sum())
    n_miss = len(df) - n_det
    n_ev   = len(df)

    DET_COLOR = "#27ae60"
    MIS_COLOR = ANOMALY_COLOR

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel 1: detected vs missed counts ───────────────────────────────
    ax = axes[0]
    counts = pd.Series({"Detected": n_det, "Missed": n_miss})
    bars = ax.bar(counts.index, counts.values, color=[DET_COLOR, MIS_COLOR],
                  edgecolor="white", width=0.4)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Event Detection ({n_ev} events)", fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_ylim(0, n_ev + 4)

    # ── Panel 2: coverage vs event length ────────────────────────────────
    ax2 = axes[1]
    colors_pt = [DET_COLOR if d else MIS_COLOR for d in df["detected"]]
    ax2.scatter(df["length"], df["hit_rate"], c=colors_pt, s=60, alpha=0.75,
                edgecolors="white", lw=0.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Event length (rows, log scale)")
    ax2.set_ylabel("Fraction of event flagged")
    ax2.set_title("Coverage by Event Length", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(handles=[
        mpatches.Patch(color=DET_COLOR, label="Detected"),
        mpatches.Patch(color=MIS_COLOR, label="Missed"),
    ], fontsize=9)

    # ── Panel 3: hit-rate distribution ───────────────────────────────────
    ax3 = axes[2]
    sns.histplot(df["hit_rate"], bins=20, ax=ax3, color="#8e44ad", edgecolor="white")
    ax3.set_xlabel("Fraction of event rows flagged")
    ax3.set_ylabel("Events")
    ax3.set_title("Hit-rate Distribution", fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if n_miss:
        print("Missed events:")
        print(df[~df["detected"]][["start", "end", "length"]].to_string(index=False))

    return fig
