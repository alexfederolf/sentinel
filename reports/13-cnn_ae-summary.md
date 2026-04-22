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

| Split | Rows | Time | Score range |
|---|---:|---:|---|
| Validation | 2,232,277 | 3.3 s | `[0.5863, 0.8041]` |
| Test_intern | 2,186,220 | 1.6 s | `[0.5879, 0.8207]` |

Even tighter score range than LSTM-AE — the CNN reconstructs everything in a similar quality band.

## Threshold tuning

- Best threshold: `0.730655`
- Val event-F0.5 at best threshold: **0.0737**

## Test_intern results (5 metrics)

| Metric | Value |
|---|---:|
| Event F0.5 | **0.0821** |
| Event recall | 0.6296 |
| Event precision | 0.0675 |
| ESA corrected F0.5 | 0.0810 |
| Row F1 | 0.0407 |

## Bootstrap CI on test_intern

| | Value |
|---|---:|
| Resamples | 1,000 |
| Wall time | 614.8 s |
| Mean event-F0.5 | 0.0533 |
| Std | 0.0077 |
| 95 % CI | **[0.0390, 0.0680]** |

## Failure analysis

- Marginally better than the LSTM-AE on every metric (Event F0.5 0.082 vs 0.073) but in the same regime: high recall, near-zero precision.
- The CNN-AE's narrower score range hurts the threshold sweep — any cut that catches the events also flags large nominal regions.
- Identical event recall (0.6296) to the LSTM-AE → both AEs detect the same easy events and miss the same hard ones.

## Possible improvements (not implemented)

- Larger receptive field (more conv layers, dilated conv) to capture multi-scale anomaly signatures
- Learnable channel attention before the bottleneck
- Score smoothing / overlapping windows at inference time
- Combine CNN-AE residual with PCA residual as an ensemble — they may catch different anomaly types
