# NB 13 — 1D CNN Autoencoder Baseline (Summary)

Notebook: [notebooks/13-cnn_ae.ipynb](../notebooks/13-cnn_ae.ipynb)
Build script: [scripts/build_nb13.py](../scripts/build_nb13.py)
Model artifact: `models/cnn_ae_bootcamp.keras`

## Setup

| Item | Value |
|---|---|
| Model | 1D CNN-AE — encoder Conv1D 32 → 16 → 8 (with MaxPool stride 2), decoder Conv1D 16 → 32 → 58 (with UpSampling1D) |
| Params | 32,034 (125 KB) — all trainable |
| Window size | 100 (stride 100), 58 features |
| Fit set | `X_train_nom.npy` — 92,271 nominal windows |
| Fit sample | `FIT_SIZE = 50_000` subsampled |
| Per-window normalisation | z-score, with `ZNormAdapter` so `score_windows` sees the z-space residual |
| Optimiser / LR | Adam, lr=1e-3 |
| Epochs run | ~31 (early-stop region; max 50 configured) |
| Hardware | TF 2.16.2 with one GPU |

## Training

- Per-epoch time: ~19–21 s
- Final val loss: ~0.683 (lower than LSTM-AE's 0.784)
- Smooth monotonic convergence

## Scoring

Scoring uses `score_windows(..., topk=5)` — per-window MSE is the mean of the
**5 worst-reconstructed channels** (out of 58) instead of the mean over all
channels. Motivation: real anomalies often affect only a handful of
channels, and averaging over all 58 dilutes the signal.

| Split | Rows | Time | Score range |
|---|---:|---:|---|
| Validation | 2,232,277 | 3.3 s | `[1.0017, 1.2538]` |
| Test_intern | 2,186,220 | 1.6 s | `[1.0089, 1.2718]` |

Range shifted upward (top-5 channels naturally have higher MSE than the
mean over 58) but the relative spread did not widen — the CNN still
reconstructs nominal and anomalous windows in similar quality bands.

## Threshold tuning

- Best threshold: `1.138344`
- Val event-F0.5 at best threshold: **0.1081**

## Test_intern results (5 metrics)

| Metric | Value |
|---|---:|
| Event F0.5 | **0.0775** |
| Event recall | 0.2222 |
| Event precision | 0.0667 |
| ESA corrected F0.5 | 0.0762 |
| Row F1 | — |

## Bootstrap CI on test_intern

| | Value |
|---|---:|
| Resamples | 200 |
| Mean event-F0.5 | 0.0507 |
| 95 % CI | **[0.0271, 0.0792]** |

## Top-k experiment — verdict

Compared to the original mean-over-all-channels scoring:

| Metric | Mean (all 58) | Top-5 |
|---|---:|---:|
| Val event-F0.5 | 0.0737 | 0.1081 |
| Test event-F0.5 | 0.0821 | 0.0775 |
| Event recall | 0.6296 | 0.2222 |
| Event precision | 0.0675 | 0.0667 |

Val improved (sharper score), but **test recall collapsed from 0.63 → 0.22**
without precision compensating. Top-k overfits the sweep to val and
discards the easy events the CNN was previously catching. Conclusion:
top-k channel MSE is **not** the right lever for this CNN-AE — its
per-channel reconstruction errors are too uniform for "worst-channel"
selection to add information.

## Failure analysis

- The CNN-AE's narrow score range hurts the threshold sweep regardless of
  scoring variant — any cut that catches events also flags large nominal
  regions.
- Both AEs (LSTM and CNN) detect the same easy events and miss the same
  hard ones. The bottleneck is the model, not the score aggregator.

## Possible improvements (not implemented)

- Larger receptive field (more conv layers, dilated conv) to capture multi-scale anomaly signatures
- Learnable channel attention before the bottleneck
- Score smoothing / overlapping windows at inference time
- Combine CNN-AE residual with PCA residual as an ensemble — they may catch different anomaly types
