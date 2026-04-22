# NB 12 — LSTM Autoencoder Baseline (Summary)

Notebook: [notebooks/12-lstm_ae.ipynb](../notebooks/12-lstm_ae.ipynb)
Build script: [scripts/build_nb12.py](../scripts/build_nb12.py)
Model artifact: `models/lstm_ae_bootcamp.keras`

## Setup

| Item | Value |
|---|---|
| Model | LSTM-AE — encoder LSTM 64 → LSTM 32, decoder RepeatVector → LSTM 64 → TimeDistributed Dense 58 |
| Params | 72,506 (283 KB) — all trainable |
| Window size | 100 (stride 100), 58 features |
| Fit set | `X_train_nom.npy` — 92,271 nominal windows |
| Fit sample | `FIT_SIZE = 50_000` subsampled |
| Per-window normalisation | z-score, with `ZNormAdapter` so `score_windows` sees the z-space residual |
| Optimiser / LR | Adam, lr=1e-3 |
| Epochs | 25 (no early-stop trigger; loss still slowly improving) |
| Hardware | TF 2.16.2 with one GPU |

## Training

- Wall time: **16.5 min** for 25 epochs
- Final train loss: 0.7876
- Final val loss: 0.7842
- Val loss curve: monotonic but very flat — model is converging slowly toward a "reconstruct everything" minimum

## Scoring

| Split | Rows | Time | Score range |
|---|---:|---:|---|
| Validation | 2,232,277 | 5.5 s | `[0.6878, 0.9665]` |
| Test_intern | 2,186,220 | 3.9 s | `[0.7053, 0.9875]` |

Score range is **very narrow** — the model reconstructs nominal and anomalous windows almost equally well, so MSE residuals barely separate classes.

## Threshold tuning

- Best threshold: `0.829141`
- Val event-F0.5 at best threshold: **0.0565**

## Test_intern results (5 metrics)

| Metric | Value |
|---|---:|
| Event F0.5 | **0.0733** |
| Event recall | 0.6296 |
| Event precision | 0.0601 |
| ESA corrected F0.5 | 0.0721 |
| Row F1 | 0.0343 |

## Bootstrap CI on test_intern

| | Value |
|---|---:|
| Resamples | 1,000 |
| Wall time | 611.7 s |
| Mean event-F0.5 | 0.0475 |
| Std | 0.0069 |
| 95 % CI | **[0.0347, 0.0606]** |

## Failure analysis

- **High recall (0.63), terrible precision (0.06)** — the model fires constantly on nominal windows. ~17 of 27 events get flagged but at the cost of an enormous false-alarm rate.
- The score distribution dynamic range (max/min ≈ 1.4) is so compressed that any threshold either lets through a flood of nominal alarms or drops most events.
- The PCA baseline (NB 11) achieves Event F0.5 0.984 on the same data — the LSTM-AE underperforms by ~13×.

## Possible improvements (not implemented)

- More capacity (deeper / wider stacks) and/or longer training
- Per-channel normalisation rather than per-window z-score
- Overlapping stride at scoring time so each row is reconstructed in multiple window contexts and the score per row is averaged
- Switch reconstruction loss to per-channel weighted MSE
