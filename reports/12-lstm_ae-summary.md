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

Scoring uses `score_windows(..., topk=5)` — per-window MSE is the mean of the
**5 worst-reconstructed channels** (out of 58) instead of the mean over all
channels. Anomalies tend to affect only a handful of channels; averaging over all
58 dilutes the signal.

| Split | Rows | Time | Score range |
|---|---:|---:|---|
| Validation | 2,232,277 | 5.2 s | `[1.0013, 2.4233]` |
| Test_intern | 2,186,220 | 3.6 s | `[1.0029, 1.6810]` |

Dynamic range: **2.4×** on val (vs ~1.4× with mean-over-all-58) — topk successfully sharpens the anomaly signal in LSTM space.

## Threshold tuning

- Best threshold: `1.191029`
- Val event-F0.5 at best threshold: **0.2475**

## Test_intern results (5 metrics)

| Metric | Value |
|---|---:|
| Event F0.5 | **0.2650** |
| Event recall | 0.5556 |
| Event precision | 0.2344 |
| ESA corrected F0.5 | 0.2642 |
| Row F1 | 0.0319 |

## Bootstrap CI on test_intern

| | Value |
|---|---:|
| Resamples | 200 |
| Wall time | 611.3 s |
| Mean event-F0.5 | 0.1755 |
| Std | 0.0298 |
| 95 % CI | **[0.1232, 0.2355]** |

## Top-k experiment — verdict

Compared to the original mean-over-all-channels baseline:

| Metric | Mean (all 58) | Top-5 |
|---|---:|---:|
| Val event-F0.5 | 0.0565 | **0.2475** |
| Test event-F0.5 | 0.0733 | **0.2650** |
| Event recall | 0.6296 | 0.5556 |
| Event precision | 0.0601 | **0.2344** |
| Bootstrap 95 % CI | [0.035, 0.061] | **[0.123, 0.236]** |

**3.6× improvement in Event F0.5.** Recall drops slightly (−0.07) but precision jumps 4× — the LSTM has several channels that it reconstructs very differently for anomalies vs nominals; selecting the worst-5 channels surfaces that signal that was buried by averaging over well-reconstructed channels.

## Remaining gap vs PCA

The PCA baseline (NB 11) still achieves Event F0.5 **0.984** — the LSTM-AE with topk=5 is at 0.265, a ~3.7× gap. The LSTM can detect ~15 of 27 events (recall 0.56); PCA detects ~25 (recall 0.93).

## Possible further improvements

- Overlapping stride at scoring time (stride < window_size) so each row appears in multiple window reconstructions — reduces non-overlapping boundary effects
- Longer training (early-stop never triggered at epoch 25; loss still descending)
- More capacity (deeper / wider LSTM stacks)
- Per-channel weighted reconstruction loss to push the model to learn tight representations on the most informative channels
