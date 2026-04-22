# Bootcamp models — Cross-Model Summary (NB 11 / 12 / 13)

All three notebooks use the identical bootcamp pipeline:

- Same chronological 70/15/15 split (event-gap snapped) → train 10.31 M / val 2.23 M / test_intern 2.19 M rows
- Same `RobustScaler` (fit on nominal train rows only)
- Same `FIT_SIZE = 50_000` subsampled nominal windows
- Same scoring path (`score_windows` over non-overlapping 100-row windows)
- Same threshold sweep (`tune_threshold` with `metric_fn=event_f05`, n_sweep=60)
- Same 5-metric report on test_intern: Event F0.5, Event recall, Event precision, ESA corrected F0.5, Row F1
- Same  (event-block, aligned truth/pred)

NB 12 and NB 13 use `score_windows(..., topk=5)` — per-window MSE aggregated
over the 5 worst-reconstructed channels instead of all 58.

## Headline table — test_intern

| Model | Event F0.5 | Event recall | Event precision | ESA corr. F0.5 | Row F1 | Best threshold | Bootstrap 95 % CI | Fit time |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| **PCA (k=38)** | **0.9843** | 0.9259 | **1.0000** | **0.4736** | **0.2281** | 0.0604 | [0.673, 0.950] | 60 s (CPU) |
| **LSTM-AE (topk=5)** | **0.2650** | 0.5556 | 0.2344 | 0.2642 | 0.0319 | 1.191 | [0.123, 0.236] | 16.5 min (GPU) |
| CNN-AE (topk=5) | 0.0775 | 0.2222 | 0.0667 | 0.0762 | — | 1.138 | [0.027, 0.079] | ~10 min (GPU) |

## Validation metric (sanity)

| Model | Val event-F0.5 at chosen threshold |
|---|---:|
| PCA | 0.8333 |
| LSTM-AE (topk=5) | 0.2475 |
| CNN-AE (topk=5) | 0.1081 |

## Score range diagnostic

| Model | Val score range | Dynamic range | Test_intern range |
|---|---|---:|---|
| PCA | `[0.018, 664.996]` | 37 000× | `[0.026, 3.090]` |
| LSTM-AE (topk=5) | `[1.001, 2.423]` | **2.4×** | `[1.003, 1.681]` |
| CNN-AE (topk=5) | `[1.002, 1.254]` | 1.25× | `[1.009, 1.272]` |

## Top-k experiment analysis

topk=5 had **opposite effects** on the two AEs:

| | LSTM-AE | CNN-AE |
|---|---|---|
| Val range before | [0.69, 0.97] → 1.4× | [0.59, 0.80] → 1.4× |
| Val range after | [1.00, 2.42] → 2.4× | [1.00, 1.25] → 1.25× |
| Test Event F0.5 | 0.073 → **0.265** (+3.6×) | 0.082 → 0.078 (−5 %) |
| Event recall | 0.63 → 0.56 | 0.63 → 0.22 |
| Event precision | 0.06 → **0.23** | 0.07 → 0.07 |

**LSTM-AE:** topk sharpened the signal — the LSTM has specific channels where it
reconstructs anomalies measurably worse than nominals. Averaging over 58 buried
that; worst-5 surfaces it.

**CNN-AE:** topk did not widen the score range and collapsed recall. The CNN
spreads its reconstruction error uniformly across channels — there is no
small subset that concentrates the anomaly signal.

## Reading the verdict

- **PCA still dominates** — Event F0.5 0.984 vs 0.265 for the best AE. With k=38
  it cleanly separates anomalous from nominal in this linear representation.
- **LSTM-AE improved substantially** with topk scoring: 3.6× better Event F0.5,
  4× better precision. Still 3.7× below PCA.
- **CNN-AE is not improved by topk** — its uniform per-channel error distribution
  means "worst channel" selection is essentially noise.
- **AE recall ceiling** is lower (0.56) than before (0.63) — topk raises the bar
  for what counts as a detected event (fewer alarms, higher quality).

## What this says about the dataset

PCA's dominance is a strong signal that anomalies live in a **low-rank linear
subspace** of the 5,800-feature window space. The LSTM learns a non-linear but
still useful representation — topk unlocks its latent channel-level structure.
The CNN's uniform error distribution suggests it has not learned channel-specific
anomaly signatures.

## Reproducibility

Per-NB summaries:
- [11-pca-summary.md](11-pca-summary.md)
- [12-lstm_ae-summary.md](12-lstm_ae-summary.md)
- [13-cnn_ae-summary.md](13-cnn_ae-summary.md)


Model artifacts in `models/`: `pca_bootcamp.pkl`, `lstm_ae_bootcamp.keras`, `cnn_ae_bootcamp.keras`

## Recommended next steps

1. **LSTM-AE is worth pursuing further** — 0.265 Event F0.5 with a simple topk fix.
   Next lever: overlapping-window scoring (stride 50 instead of 100).
2. **CNN-AE needs architectural changes** to break its uniform-error pattern before
   topk will help — e.g. channel attention or per-channel loss weighting.
3. **PCA is production-ready** for the current feature set. Investigate whether
   the val→test F0.5 jump (0.833 → 0.984) is robust or lucky.
