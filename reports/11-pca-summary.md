# NB 11 — PCA Baseline (Summary)

Notebook: [notebooks/11-pca.ipynb](../notebooks/11-pca.ipynb)
Build script: [scripts/build_nb11.py](../scripts/build_nb11.py)
Model artifact: `models/pca_bootcamp.pkl`

## Setup

| Item | Value |
|---|---|
| Window size | 100 (stride 100) |
| Features | 58 target channels |
| Fit set | `X_train_nom.npy` — nominal-only training windows |
| Fit sample | `FIT_SIZE = 50_000` of 92,271 windows (subsampled) |
| Scaler | `RobustScaler` fit on nominal train rows only |
| PCA target | `n_components=0.95` (cumulative explained variance) |
| Random state | from `sentinel.params.RANDOM_STATE` |

## Data scoped

| Split | Rows | Anomaly rows | Anomaly ratio |
|---|---:|---:|---:|
| Train (fit pool) | 10,309,824 | 1,082,680 | 10.5 % |
| Validation | 2,232,277 | 246,463 | 11.0 % |
| Test_intern | 2,186,220 | 214,961 | 9.8 % |

## Fit

- Fit time: **59.6 s** on 50,000 subsampled nominal windows (5,800 features each)
- `n_components` kept: **38** (cumulative EV = **0.9501**)

## Scoring

`score_windows` reshapes each row array into non-overlapping 100-row windows, runs PCA `inverse_transform(transform(X))`, returns the per-row mean-squared reconstruction error (broadcast back to row level).

| Split | Rows | Time | Score range |
|---|---:|---:|---|
| Validation | 2,232,277 | 0.8 s | `[0.0175, 664.9958]` |
| Test_intern | 2,186,220 | 0.6 s | `[0.0258, 3.0895]` |

## Threshold tuning

Log-spaced sweep of 60 thresholds over val scores, picked by `event_f05`:

- **Best threshold:** `0.060404`
- **Val event-F0.5 at best threshold:** `0.8333`

## Test_intern results (5 metrics, the only ones reported)

| Metric | Value |
|---|---:|
| Event F0.5 | **0.9843** |
| Event recall | 0.9259 |
| Event precision | 1.0000 |
| ESA corrected F0.5 | 0.4736 |
| Row F1 | 0.2281 |

## Bootstrap CI on test_intern (event-block resampling)

| | Value |
|---|---:|
| Resamples | 1,000 |
| Wall time | 573.7 s |
| Mean event-F0.5 | 0.8053 |
| Std | 0.0770 |
| 95 % CI | **[0.6730, 0.9501]** |

CI uses the corrected event-block bootstrap: when an event is dropped from the resample, its prediction region is dropped in lockstep, so the CI reflects metric variance under event-presence sampling, not spurious FPs from misaligned truth/pred.

## Reading the metrics

- **Event-level (F0.5 / recall / precision)** — counts each contiguous anomaly event as one TP/FN regardless of length. PCA hits 25 of 27 events (recall 0.926) with zero spurious event predictions (precision 1.000), giving the headline event-F0.5 of **0.984**.
- **ESA corrected F0.5 (0.474)** — TNR-pulldown correction penalises models that over-trigger on nominal regions. The drop from 0.984 → 0.474 means PCA's per-row alarm density on nominal windows is non-trivial even when event-level coverage is perfect.
- **Row F1 (0.228)** — strict per-row precision/recall on anomalous rows. Low because PCA flags whole windows: even when an event is detected correctly, only a fraction of the flagged rows are inside the true event extent.

## Honest caveats

- Submissions are **no longer written** from this notebook (1x notebooks are evaluation-only).
- `test_extern` (unlabeled competition set) is not loaded here.
- Fit on 50k subsampled windows trades a typical <0.01 F0.5 delta for ~15× speedup vs. fitting on all 92k windows (`FIT_SIZE=None`).
- Bootstrap CI lower bound (0.673) is meaningfully below the point estimate (0.984), driven by the small event count: dropping any of the high-leverage events in a resample collapses the score for that draw.
