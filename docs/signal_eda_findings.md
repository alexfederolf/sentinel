# Signal EDA Findings — Channels 41–46

Summary of exploratory analysis on the ESA-ADB dataset relevant to anomaly detection model design,
particularly for sequence models (LSTM AE).

---

## Dataset Overview

- **Train**: 14.7 M rows, 76 channel columns + 11 telecommand columns + `is_anomaly` label
- **Test**: 521 K rows, no labels (Kaggle leaderboard only)
- **Anomaly rate**: ~10.5% of rows, but anomalies occur in **contiguous events** — the real count is ~26–28 events in the labeled portion
- **Evaluation metric**: event-wise F0.5 (ESA metric) — an anomaly event is *detected* if at least one predicted row overlaps it; precision-weighted (β=0.5), plus a TNR correction term. Catching one row per event is enough; false positives are costly.
- **Semi-supervised**: models are trained on nominal data only (`is_anomaly == 0`)

---

## Signal Structure

Each channel signal decomposes into three additive components:

```
y(t) = sin1(t)  +  sin2(t)  +  add(t)  +  noise(t)
```

| Component | Period | Description |
|-----------|--------|-------------|
| `sin1` | very long (~2.5 M rows) | Slow orbital/environmental trend |
| `sin2` | ~2 000 rows | Medium-frequency oscillation, clearly visible after detrending |
| `add` | ~100 rows (pseudo-period) | Step-like component: zero for most of the time (Type 1), or a constant value sustained for 2–5 samples (Type 2) — normal behaviour in both cases |
| `noise` | — | Residual after removing all three components above |

All six channels (41–46) share the same qualitative structure and the same dominant periods.

### Period Detection (channels 41–46 only)

FFT-based detection on nominal data (`ek_freq_eda.ipynb`):

- **`sin2` period (~2 000 samples)**: detected reliably on the *detrended* signal (raw minus `sin1` smoothed with a large window). Without detrending, the red-noise slope (power ∝ 1/f) masks the peak.
- **`sin1` period**: not reliably detected via FFT — the component may not be strictly periodic at this scale. A fixed smoothing window (`BIG_WINDOW = 50 000`) is used to extract it.
- Detected periods are saved per-channel in `data/freq_map.json`.

---

## Key Finding: Anomalies Live in the Residual

The z-score baseline (`ek_baseline_zscore.ipynb`) uses a causal rolling z-score with
`window = 300`. This window is large enough to absorb both `sin1` and `sin2`
(periods >> 300) while preserving high-frequency content, leaving the model to score
deviations in `add + noise`.

This directly implies:

> **Anomalies manifest primarily in the high-frequency residual (`noise`),
> not in the slow trend (`sin1`) or medium oscillation (`sin2`, `sin3`).**

The slow components carry large signal amplitude but are regular and predictable.
The residual, while low-amplitude, has a much better signal-to-noise ratio for anomaly detection.

---

## Implications for LSTM AE Input Features

Feeding raw channel values directly to an LSTM forces the model to simultaneously learn:
1. The large-amplitude sinusoidal trends (easy, but wastes capacity)
2. The anomaly signal buried in a small residual (hard, low SNR)

**Recommended input**: high-frequency residual rather than raw signal.

### Option A — Full cascaded decomposition (channels 41–46 only)

Requires per-channel period detection via `freq_map.json`. Three-level `uniform_filter1d` cascade:

```python
sin1 = smooth(raw, bw)   # bw = 50 000
r1   = raw  - sin1
sin2 = smooth(r1,  mw)   # mw = sqrt(T_small * T_medium), from freq_map
r2   = r1   - sin2
sin3 = smooth(r2,  sw)   # sw = T_small // 3, from freq_map
res  = r2   - sin3
```

Feed `res` (6 channels = **6 input dimensions**) to the LSTM.
Reconstruction error on `res` is the anomaly score.

### Option B — Single-window filter (all 76 channels, no freq_map needed)

```python
residual = raw - smooth(raw, W=300)   # causal uniform_filter1d
```

**Preprocessing for LSTM-AE / CNN-AE: single MA subtraction vs full cascade**

Context: Full cascade decomposition (bw=50000 → mw≈720 → sw≈19) is known for channels 41–46, but not yet for other channel groups. Question: is subtracting a single MA(300) sufficient as interim preprocessing?

What MA(300) removes vs keeps:
- Slow trend (period >> 300): removed ✓
- T1 ≈ 98 (short sinusoid): removed ✓ — window 300 covers ~3 full periods, averaging it out to ~2% amplitude
- T2 ≈ 2160 (medium sinusoid): kept — window 300 covers only ~14% of one T2 period, so T2 passes through at ~97% amplitude
How T2 looks inside a 100-row window:
Within a window of 100 rows, T2 (period 2160) covers ~4.6% of a cycle — it appears as a near-linear slope, not a sinusoid. The slope magnitude and direction change slowly as the phase of T2 advances.

Can LSTM-AE / CNN-AE learn this slope?
Yes. During training the model sees all phases of T2 (many full cycles fit in the training set), so it learns the full range of slopes as normal behavior. An anomaly that breaks the expected slope pattern produces high reconstruction error.

Practical conclusion: For LSTM-AE / CNN-AE cascade is not critical — the model can learn T2 as a slope implicitly. MA(300) is likely sufficient.

**This approach generalises to all 76 channels** without requiring per-channel period detection.
Input: residual for all channels = **76 input dimensions** (or whichever channel subset is used).

### Recommended window size for LSTM

| Input type | Suggested window |
|------------|-----------------|
| Option A `res` | 500–1 000 rows (5–10 × T_small) |
| Option B `raw - smooth(raw, 300)` | 1 000–2 000 rows (covers ~1 full `sin2` cycle) |

---

## Z-Score Baseline Context

`ek_baseline_zscore.ipynb` applies a rolling z-score (`window = 300`) per channel,
then aggregates via top-p% mean across channels, and tunes a threshold on the val split.
It is a direct implementation of Option B above applied to z-scores rather than raw residuals.
Its results serve as the reference baseline; numbers to be updated after the latest run.

---

*Notebook references: `notebooks/ek_freq_eda.ipynb`, `notebooks/ek_baseline_zscore.ipynb`*
*Data artefact: `data/freq_map.json` — detected periods per channel (channels 41–46)*
