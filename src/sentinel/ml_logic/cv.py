"""
Rolled-origin CV harness for the SENTINEL hybrid (NB 11d).

Why this exists
---------------
NB 11d tunes ``thr_freq``, ``k_env``, and ``thr_env`` on a single fixed val
window (rows 10.7M-12.7M). That single split is unrepresentative — it hides
overfitting that only shows up on Kaggle private. The classic example:
``ENV_MA_WINDOW=20_000`` scores **val 0.92 / private 0.903**, while the
champion ``ma_w=5_000`` scores **val 0.94 / private 0.915**. Local val picked
the wrong one.

This module replaces single-split tuning with expanding-origin K-fold CV:
multiple val windows over the labelled-train timeline. For each fold,
``thr_freq`` and ``(k_env, thr_env)`` are tuned independently. The spread
across folds (e.g. ``thr_freq`` CV ≈ 24 % at ma_w=5k) tells you whether the
threshold is a stable optimum or a single-spoonful artefact.

Decision rule (used 2026-04-29 for the ma_w sweep):
    val→test_internal gap > 0.10  →  config is overfit (skip).
    Smaller gap is more important than higher mean val score.

Usage
-----
    >>> from sentinel.ml_logic.cv import run_cv, run_sweep
    >>> r = run_cv(env_ma_window=5_000)               # one config
    >>> sweep = run_sweep([3_000, 5_000, 7_000, 10_000, 20_000])

CLI:
    python -m sentinel.ml_logic.cv --ma-w 5000
    python -m sentinel.ml_logic.cv --sweep

Architecture (mirrors NB 11d cell 4 verbatim):
    train_envr = rolling_min(channels 14/21/29, env_w=200) − centred_MA(ma_w)
    train_envr /= nominal-train std    (per-fold)
    ref_per_ch = p99(|train_envr| in nominal-train rows) (per-fold)
    PCA on tail-50k nominal windows of channels 41-46 [0, fold_train_end)
    Score: row-level PCA recon error  +  per-row top-k env z-score
    Threshold: independent OR fusion, tuned via tune_threshold(score, y, esa, n_sweep=80)

The held-out test [12.7M, end] is **never touched during CV**.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .metrics import corrected_event_f05
from .scorer import score_windows
from .thresholds import tune_threshold
from .data import PROCESSED_DIR


# ─── pipeline constants (= NB 11d champion) ──────────────────────────────────
WINDOW_SIZE   = 100
FIT_SIZE      = 50_000
ENV_WINDOW    = 200
ENV_REF_PCT   = 99
RANDOM_STATE  = 42
DEFAULT_ENV_MA_WINDOW = 5_000

DEFAULT_FREQ_NAMES = [f'channel_{i}' for i in range(41, 47)]
DEFAULT_ENV_NAMES  = ['channel_14', 'channel_21', 'channel_29']

# CV layout
DEFAULT_HELDOUT_START = 12_700_000   # rows [12.7M, end] = test_internal


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (lifted from NB 11d, useful enough to re-export)
# ═══════════════════════════════════════════════════════════════════════════════

def envr_cols(arr2d: np.ndarray, env_w: int, ma_w: int) -> np.ndarray:
    """
    Per-column envelope-residual: ``rolling_min(env_w) − centered_MA(ma_w)``.

    Used to drift-correct the slowly-shifting drifter channels (14/21/29 by
    default) before z-scoring. Centered MA means each row uses ±ma_w/2 rows
    of context; the centring is what NB 11d / champion submission use.
    """
    out = np.empty_like(arr2d, dtype=np.float32)
    for j in range(arr2d.shape[1]):
        s = pd.Series(arr2d[:, j])
        env = s.rolling(window=env_w, min_periods=1).min()
        ma  = env.rolling(window=ma_w, min_periods=1, center=True).mean()
        out[:, j] = (env - ma).values.astype(np.float32)
    return out


def top_p_mean(arr2d: np.ndarray, p: float) -> np.ndarray:
    """
    Mean of the top-p fraction of columns by absolute value, per row.

    p=1/n_env equals "max across channels" (k=1); p=1.0 equals plain mean
    (k=n_env). Hybrid env-stream tunes p ∈ {1/3, 2/3, 3/3} → k ∈ {1, 2, 3}.
    """
    n_top = max(1, int(arr2d.shape[1] * p))
    top = np.partition(arr2d, -n_top, axis=1)[:, -n_top:]
    return top.mean(axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVData:
    """Container for the four arrays needed by the harness."""
    y_train_full      : np.ndarray   # window-level labels (~147k entries)
    y_train_row       : np.ndarray   # row-level labels  (14.7M entries)
    train_scaled_full : np.ndarray   # RobustScaler-normalised train (mmap)
    name_to_idx       : dict         # channel name → array column index

    @property
    def n_rows(self) -> int:
        return self.train_scaled_full.shape[0]

    @property
    def n_channels(self) -> int:
        return self.train_scaled_full.shape[1]


def load_cv_data(kaggle_dir: Optional[Path] = None) -> CVData:
    """
    Load the processed Kaggle arrays needed for CV (mmap where possible).

    Returns a ``CVData`` container. The ``name_to_idx`` dict maps channel
    names (e.g. ``'channel_41'``) to column indices in ``train_scaled_full``.
    """
    kaggle_dir = kaggle_dir or (PROCESSED_DIR / 'kaggle')
    y_train_full      = np.load(kaggle_dir / 'y_train_full.npy')
    y_train_row       = np.load(kaggle_dir / 'y_train_row.npy')
    train_scaled_full = np.load(kaggle_dir / 'train_full_scaled.npy', mmap_mode='r')
    cfg = json.loads((kaggle_dir / 'preprocessing_config.json').read_text())
    chan_names = (cfg.get('target_channels')
                  or cfg.get('feature_names')
                  or cfg.get('channel_names'))
    if chan_names is None:
        chan_names = [f'channel_{i}' for i in range(train_scaled_full.shape[1])]
    chan_names = chan_names[:train_scaled_full.shape[1]]
    return CVData(
        y_train_full      = y_train_full,
        y_train_row       = y_train_row,
        train_scaled_full = train_scaled_full,
        name_to_idx       = {n: i for i, n in enumerate(chan_names)},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Fold definition
# ═══════════════════════════════════════════════════════════════════════════════

def make_expanding_origin_folds(
    n_folds: int = 5,
    cv_end: int = DEFAULT_HELDOUT_START,
    val_width: int = 2_000_000,
    first_train_end: int = 2_500_000,
) -> list[dict]:
    """
    Expanding-origin CV: each fold's train=[0, val_start), val=[val_start, val_end).

    val_starts are evenly spaced inside [first_train_end, cv_end - val_width].
    Fold 4 (the last one with cv_end=12.7M and val_width=2M) covers
    [10.7M, 12.7M] — bit-identical to NB 11d's val window.
    """
    if n_folds < 2:
        raise ValueError(f'n_folds must be >= 2, got {n_folds}')
    last_val_start = cv_end - val_width
    val_starts = np.linspace(first_train_end, last_val_start, n_folds).astype(int)
    return [
        {
            'fold_id'  : i,
            'train_end': int(vs),
            'val_start': int(vs),
            'val_end'  : int(vs) + val_width,
        }
        for i, vs in enumerate(val_starts)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Per-fold scoring + threshold tuning
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold_id      : int
    train_end    : int
    val_start    : int
    val_end      : int
    n_pca        : int
    thr_freq     : float
    k_env        : int
    thr_env      : float
    val_f05_freq : float
    val_f05_env  : float
    val_f05_or   : float


def run_fold(
    fold              : dict,
    freq_idx          : list[int],
    env_idx           : list[int],
    train_scaled_full : np.ndarray,
    train_envr_raw    : np.ndarray,    # un-normalised env residual (full length)
    y_train_full      : np.ndarray,
    y_train_row       : np.ndarray,
) -> FoldResult:
    """One fold: PCA fit, env normalisation, val scoring, threshold tuning."""
    train_end = fold['train_end']
    val_start = fold['val_start']
    val_end   = fold['val_end']
    n_freq, n_env = len(freq_idx), len(env_idx)

    # ── Freq: PCA fit on tail-FIT_SIZE nominal windows of [0, train_end) ─────
    win_end_row = (np.arange(len(y_train_full)) + 1) * WINDOW_SIZE
    nominal_mask = (y_train_full == 0)
    eligible = nominal_mask & (win_end_row <= train_end)
    fit_idx = np.flatnonzero(eligible)[-FIT_SIZE:]
    if len(fit_idx) < 1000:
        raise RuntimeError(
            f'fold {fold["fold_id"]}: only {len(fit_idx)} eligible nominal '
            f'windows in [0, {train_end:,}) — fold too short')
    _row_idx = (fit_idx[:, None] * WINDOW_SIZE
                + np.arange(WINDOW_SIZE)[None, :]).ravel()
    train_freq = np.ascontiguousarray(train_scaled_full[:, freq_idx],
                                      dtype=np.float32)
    X_fit_freq = train_freq[_row_idx].reshape(len(fit_idx), -1).astype(np.float32)
    pca_freq = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_fit_freq)
    del X_fit_freq

    # ── Env: per-fold std + p99 ref using only [0, train_end) nominals ───────
    if n_env:
        train_envr = train_envr_raw.copy()
        nom_row_train = (y_train_row[:train_end] == 0)
        envr_std = np.maximum(
            train_envr[:train_end][nom_row_train].std(axis=0, keepdims=True), 1e-6)
        train_envr = (train_envr / envr_std).astype(np.float32)
        ref_per_ch = np.maximum(
            np.percentile(np.abs(train_envr[:train_end]), ENV_REF_PCT, axis=0),
            1e-8).astype(np.float32)
    else:
        train_envr = np.empty((train_scaled_full.shape[0], 0), dtype=np.float32)
        ref_per_ch = np.empty(0, dtype=np.float32)

    # ── Score val window ─────────────────────────────────────────────────────
    X_val_freq = train_freq[val_start:val_end]
    val_scores_freq = (score_windows(pca_freq, X_val_freq) if pca_freq is not None
                       else np.zeros(val_end - val_start, dtype=np.float32))
    y_val = y_train_row[val_start:val_end]
    z_val_env = (np.abs(train_envr[val_start:val_end] / ref_per_ch).astype(np.float32)
                 if n_env else np.empty((val_end - val_start, 0), dtype=np.float32))

    # ── Tune thresholds ──────────────────────────────────────────────────────
    tuned_freq = tune_threshold(val_scores_freq, y_val,
                                corrected_event_f05, n_sweep=80)
    thr_freq = float(tuned_freq['threshold'])
    val_f05_freq = float(tuned_freq['score'])

    best_env_f05, best_k, best_thr_env = -1.0, 1, np.inf
    if n_env:
        for k in range(1, n_env + 1):
            sc = top_p_mean(z_val_env, p=k / n_env)
            res = tune_threshold(sc, y_val, corrected_event_f05, n_sweep=80)
            if res['score'] > best_env_f05:
                best_env_f05 = float(res['score'])
                best_k = k
                best_thr_env = float(res['threshold'])

    # ── OR-fused val ESA ─────────────────────────────────────────────────────
    val_scores_env = (top_p_mean(z_val_env, p=best_k / n_env) if n_env
                      else np.zeros_like(val_scores_freq))
    y_or_val = ((val_scores_freq > thr_freq) |
                (val_scores_env > best_thr_env)).astype(np.int8)
    val_f05_or = float(corrected_event_f05(y_val, y_or_val)['f_score'])

    return FoldResult(
        fold_id=fold['fold_id'], train_end=train_end,
        val_start=val_start, val_end=val_end,
        n_pca=int(pca_freq.n_components_),
        thr_freq=thr_freq, k_env=best_k, thr_env=best_thr_env,
        val_f05_freq=val_f05_freq, val_f05_env=best_env_f05,
        val_f05_or=val_f05_or,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Held-out test_internal evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HeldOutResult:
    fit_train_end  : int
    n_pca          : int
    thr_freq_use   : float
    k_env_use      : int
    thr_env_use    : float
    esa_f05        : float
    precision      : float
    recall         : float
    tp_events      : int
    fn_events      : int
    fp_pred_events : int
    tnr            : float
    flag_rate      : float


def eval_held_out(
    freq_idx          : list[int],
    env_idx           : list[int],
    train_scaled_full : np.ndarray,
    train_envr_raw    : np.ndarray,
    y_train_full      : np.ndarray,
    y_train_row       : np.ndarray,
    fit_train_end     : int,
    thr_freq_use      : float,
    k_env_use         : int,
    thr_env_use       : float,
    heldout_start     : int = DEFAULT_HELDOUT_START,
) -> HeldOutResult:
    """Re-fit at fit_train_end and score [heldout_start, end) at given thresholds."""
    n_freq, n_env = len(freq_idx), len(env_idx)

    # PCA fit on [0, fit_train_end)
    win_end_row = (np.arange(len(y_train_full)) + 1) * WINDOW_SIZE
    eligible = (y_train_full == 0) & (win_end_row <= fit_train_end)
    fit_idx = np.flatnonzero(eligible)[-FIT_SIZE:]
    _row_idx = (fit_idx[:, None] * WINDOW_SIZE
                + np.arange(WINDOW_SIZE)[None, :]).ravel()
    train_freq = np.ascontiguousarray(train_scaled_full[:, freq_idx], dtype=np.float32)
    X_fit_freq = train_freq[_row_idx].reshape(len(fit_idx), -1).astype(np.float32)
    pca_freq = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_fit_freq)

    # Env normalisation from [0, fit_train_end)
    if n_env:
        train_envr = train_envr_raw.copy()
        nom_row_train = (y_train_row[:fit_train_end] == 0)
        envr_std = np.maximum(
            train_envr[:fit_train_end][nom_row_train].std(axis=0, keepdims=True), 1e-6)
        train_envr = (train_envr / envr_std).astype(np.float32)
        ref_per_ch = np.maximum(
            np.percentile(np.abs(train_envr[:fit_train_end]), ENV_REF_PCT, axis=0),
            1e-8).astype(np.float32)
    else:
        train_envr = np.empty((train_scaled_full.shape[0], 0), dtype=np.float32)
        ref_per_ch = np.empty(0, dtype=np.float32)

    # Score test_internal
    test_start, test_end = heldout_start, len(y_train_row)
    test_freq_arr = train_freq[test_start:test_end]
    test_scores_freq = score_windows(pca_freq, test_freq_arr)
    if n_env:
        z_test_env = np.abs(train_envr[test_start:test_end] / ref_per_ch).astype(np.float32)
        test_scores_env = top_p_mean(z_test_env, p=k_env_use / n_env)
    else:
        test_scores_env = np.zeros_like(test_scores_freq)

    y_test = y_train_row[test_start:test_end]
    y_pred = ((test_scores_freq > thr_freq_use) |
              (test_scores_env > thr_env_use)).astype(np.int8)

    m = corrected_event_f05(y_test, y_pred)
    return HeldOutResult(
        fit_train_end=fit_train_end, n_pca=int(pca_freq.n_components_),
        thr_freq_use=thr_freq_use, k_env_use=k_env_use, thr_env_use=thr_env_use,
        esa_f05=m['f_score'], precision=m['precision'], recall=m['recall'],
        tp_events=m['tp_events'], fn_events=m['fn_events'],
        fp_pred_events=m['fp_pred_events'], tnr=m['tnr'],
        flag_rate=float(y_pred.mean()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level orchestration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVRunResult:
    env_ma_window    : int
    n_folds          : int
    fold_results     : list[FoldResult]
    thr_freq_avg     : float
    thr_freq_std     : float
    thr_env_avg      : float
    thr_env_std      : float
    k_env_mode       : int
    k_env_per_fold   : list[int]
    val_f05_or_mean  : float
    val_f05_or_std   : float
    eval_A_11d_fit   : HeldOutResult     # fit on [0, 10.7M) — comparable to NB 11d
    eval_B_full_fit  : HeldOutResult     # fit on [0, 12.7M) — uses all CV data

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def run_cv(
    env_ma_window : int = DEFAULT_ENV_MA_WINDOW,
    n_folds       : int = 5,
    freq_names    : list[str] = None,
    env_names     : list[str] = None,
    data          : Optional[CVData] = None,
    verbose       : bool = True,
) -> CVRunResult:
    """
    Run K-fold CV for one ``ENV_MA_WINDOW`` value, then evaluate averaged
    thresholds on the held-out test_internal at two fit-window choices:
        A: fit on [0, 10.7M)   — comparable to NB 11d
        B: fit on [0, 12.7M)   — uses all CV data, for Kaggle prep

    Returns a ``CVRunResult`` with per-fold details + summary stats.
    """
    freq_names = freq_names or DEFAULT_FREQ_NAMES
    env_names  = env_names  or DEFAULT_ENV_NAMES
    if data is None:
        data = load_cv_data()

    if verbose:
        print('═' * 78)
        print(f' CV harness — ENV_MA_WINDOW = {env_ma_window:,}')
        print('═' * 78)

    freq_idx = [data.name_to_idx[n] for n in freq_names]
    env_idx  = [data.name_to_idx[n] for n in env_names]

    # Compute env-residual ONCE (same for all folds; only normalisation per-fold)
    if verbose:
        print(f'Building env-residual stream (env_w={ENV_WINDOW}, ma_w={env_ma_window:,}) …')
    t0 = time.time()
    train_envr_raw = envr_cols(
        np.ascontiguousarray(data.train_scaled_full[:, env_idx], dtype=np.float32),
        ENV_WINDOW, env_ma_window)
    if verbose:
        print(f'  done in {time.time()-t0:.1f}s   shape={train_envr_raw.shape}')

    folds = make_expanding_origin_folds(n_folds=n_folds)

    if verbose:
        print(f'\n{"fold":>4}  {"train_end":>12}  {"val":>22}  '
              f'{"thr_freq":>9}  {"k":>1}  {"thr_env":>8}  {"val_F0.5_OR":>11}')
        print('─' * 78)

    fold_results = []
    for f in folds:
        t0 = time.time()
        r = run_fold(f, freq_idx, env_idx, data.train_scaled_full,
                     train_envr_raw, data.y_train_full, data.y_train_row)
        dt = time.time() - t0
        if verbose:
            print(f'{r.fold_id:>4}  {r.train_end:>12,}  '
                  f'[{r.val_start:>10,}, {r.val_end:>10,})  '
                  f'{r.thr_freq:>9.5f}  {r.k_env:>1}  '
                  f'{r.thr_env:>8.3f}  {r.val_f05_or:>11.4f}  ({dt:.1f}s)')
        fold_results.append(r)

    # ── Aggregate ────────────────────────────────────────────────────────────
    thr_freqs = np.array([r.thr_freq    for r in fold_results])
    thr_envs  = np.array([r.thr_env     for r in fold_results])
    ks        = np.array([r.k_env       for r in fold_results])
    val_f05s  = np.array([r.val_f05_or  for r in fold_results])

    k_mode = int(np.bincount(ks, minlength=ks.max() + 1).argmax())
    same_k_mask = (ks == k_mode)
    thr_env_avg = float(thr_envs[same_k_mask].mean())
    thr_env_std = float(thr_envs[same_k_mask].std()) if same_k_mask.sum() > 1 else 0.0
    thr_freq_avg = float(thr_freqs.mean())
    thr_freq_std = float(thr_freqs.std())

    if verbose:
        print('─' * 78)
        cv_freq = thr_freq_std / abs(thr_freq_avg) if thr_freq_avg else 0.0
        print(f'mean ± std  thr_freq = {thr_freq_avg:.5f} ± {thr_freq_std:.5f}   '
              f'(CV = {cv_freq:.1%})')
        if same_k_mask.sum() > 1:
            cv_env = thr_env_std / abs(thr_env_avg) if thr_env_avg else 0.0
            print(f'mean ± std  thr_env  = {thr_env_avg:.4f} ± {thr_env_std:.4f}   '
                  f'(CV = {cv_env:.1%}, k={k_mode}, {same_k_mask.sum()}/{len(folds)} folds)')
        else:
            print(f'k_env split across folds: {ks.tolist()}  (mode={k_mode}); '
                  f'thr_env using mode-folds only: {thr_env_avg:.4f}')
        print(f'val F0.5 (OR)        = {val_f05s.mean():.4f} ± {val_f05s.std():.4f}')

    # ── Held-out test_internal eval ──────────────────────────────────────────
    if verbose:
        print('\n' + '─' * 78)
        print(f' Held-out test_internal eval: rows '
              f'[{DEFAULT_HELDOUT_START:,}, {len(data.y_train_row):,})')
        print('─' * 78)

    eval_a = eval_held_out(freq_idx, env_idx, data.train_scaled_full,
                           train_envr_raw, data.y_train_full, data.y_train_row,
                           fit_train_end=10_700_000,
                           thr_freq_use=thr_freq_avg, k_env_use=k_mode,
                           thr_env_use=thr_env_avg)
    eval_b = eval_held_out(freq_idx, env_idx, data.train_scaled_full,
                           train_envr_raw, data.y_train_full, data.y_train_row,
                           fit_train_end=DEFAULT_HELDOUT_START,
                           thr_freq_use=thr_freq_avg, k_env_use=k_mode,
                           thr_env_use=thr_env_avg)

    if verbose:
        print(f'  [A] fit on [0, 10.7M) (= NB 11d):  ESA F0.5 = {eval_a.esa_f05:.4f}   '
              f'TP={eval_a.tp_events}/25  FP_seg={eval_a.fp_pred_events}  '
              f'flag={eval_a.flag_rate:.4%}  TNR={eval_a.tnr:.6f}')
        print(f'  [B] fit on [0, 12.7M) (all CV):    ESA F0.5 = {eval_b.esa_f05:.4f}   '
              f'TP={eval_b.tp_events}/25  FP_seg={eval_b.fp_pred_events}  '
              f'flag={eval_b.flag_rate:.4%}  TNR={eval_b.tnr:.6f}')

    return CVRunResult(
        env_ma_window=env_ma_window, n_folds=n_folds,
        fold_results=fold_results,
        thr_freq_avg=thr_freq_avg, thr_freq_std=thr_freq_std,
        thr_env_avg=thr_env_avg,   thr_env_std=thr_env_std,
        k_env_mode=k_mode, k_env_per_fold=ks.tolist(),
        val_f05_or_mean=float(val_f05s.mean()),
        val_f05_or_std=float(val_f05s.std()),
        eval_A_11d_fit=eval_a, eval_B_full_fit=eval_b,
    )


def run_sweep(
    env_ma_windows : Iterable[int],
    n_folds        : int = 5,
    data           : Optional[CVData] = None,
    verbose        : bool = True,
) -> list[CVRunResult]:
    """Run ``run_cv`` for each ``ENV_MA_WINDOW`` in ``env_ma_windows``, print summary."""
    if data is None:
        data = load_cv_data()
    results = [run_cv(env_ma_window=w, n_folds=n_folds, data=data, verbose=verbose)
               for w in env_ma_windows]

    if verbose and len(results) > 1:
        print('\n' + '═' * 96)
        print(' SWEEP SUMMARY')
        print('═' * 96)
        print(f'{"ma_w":>7}  {"thr_f μ±σ":>16}  {"k":>1}  {"thr_e μ±σ":>14}  '
              f'{"val_OR μ±σ":>14}  {"test_A":>7}  {"test_B":>7}')
        print('─' * 96)
        for r in results:
            print(f'{r.env_ma_window:>7,}  '
                  f'{r.thr_freq_avg:.4f}±{r.thr_freq_std:.4f}  '
                  f'{r.k_env_mode:>1}  '
                  f'{r.thr_env_avg:>5.2f}±{r.thr_env_std:.2f}   '
                  f'{r.val_f05_or_mean:.4f}±{r.val_f05_or_std:.4f}  '
                  f'{r.eval_A_11d_fit.esa_f05:>7.4f}  '
                  f'{r.eval_B_full_fit.esa_f05:>7.4f}')
        print()
        best_a  = max(results, key=lambda r: r.eval_A_11d_fit.esa_f05)
        best_b  = max(results, key=lambda r: r.eval_B_full_fit.esa_f05)
        stable  = min(results, key=lambda r: r.thr_freq_std)
        print(f'Best test_A (11d-comparable):     ma_w = {best_a.env_ma_window:,}  '
              f'ESA = {best_a.eval_A_11d_fit.esa_f05:.4f}')
        print(f'Best test_B (all-CV-fit):         ma_w = {best_b.env_ma_window:,}  '
              f'ESA = {best_b.eval_B_full_fit.esa_f05:.4f}')
        print(f'Most stable (lowest thr_freq σ):  ma_w = {stable.env_ma_window:,}  '
              f'σ = {stable.thr_freq_std:.5f}')
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI: ``python -m sentinel.ml_logic.cv``
# ═══════════════════════════════════════════════════════════════════════════════

def _main():
    p = argparse.ArgumentParser(prog='sentinel.ml_logic.cv',
                                description=__doc__.split('\n\n')[0])
    p.add_argument('--ma-w', type=int, default=DEFAULT_ENV_MA_WINDOW,
                   help='ENV_MA_WINDOW value to test (default: 5000).')
    p.add_argument('--folds', type=int, default=5,
                   help='Number of expanding-origin folds (default: 5).')
    p.add_argument('--sweep', action='store_true',
                   help='Sweep ma_w ∈ {3000, 5000, 7000, 10000, 20000}.')
    p.add_argument('--out', type=Path, default=Path('/tmp/cv_results.json'),
                   help='Where to write the JSON results.')
    args = p.parse_args()

    data = load_cv_data()
    if args.sweep:
        results = run_sweep([3_000, 5_000, 7_000, 10_000, 20_000],
                            n_folds=args.folds, data=data)
    else:
        results = [run_cv(env_ma_window=args.ma_w, n_folds=args.folds, data=data)]

    # Write JSON (round-trip via to_dict so dataclasses serialise cleanly)
    args.out.write_text(json.dumps([r.to_dict() for r in results],
                                    indent=2, default=str))
    print(f'\nWritten: {args.out}')


if __name__ == '__main__':
    _main()
