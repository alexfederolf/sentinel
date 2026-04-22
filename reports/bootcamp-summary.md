# Bootcamp models — Cross-Model Summary (NB 11 / 12 / 13)

All three notebooks use the identical bootcamp pipeline:

- Same chronological 70/15/15 split (event-gap snapped) → train 10.31 M / val 2.23 M / test_intern 2.19 M rows
- Same `RobustScaler` (fit on nominal train rows only)
- Same `FIT_SIZE = 50_000` subsampled nominal windows
- Same scoring path (`score_windows` over non-overlapping 100-row windows → per-row mean MSE)
- Same threshold sweep (`tune_threshold` with `metric_fn=event_f05`, n_sweep=60)
- Same 5-metric report on test_intern: Event F0.5, Event recall, Event precision, ESA corrected F0.5, Row F1
- Same bootstrap CI (event-block, aligned truth/pred)
- No submission written (this is evaluation-only)

## Headline table — test_intern

| Model | Event F0.5 | Event recall | Event precision | ESA corr. F0.5 | Row F1 | Best threshold | Bootstrap 95 % CI | Fit time |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| **PCA (k=38)** | **0.9843** | 0.9259 | **1.0000** | **0.4736** | **0.2281** | 0.0604 | [0.673, 0.950] | 60 s (CPU) |
| LSTM-AE | 0.0733 | 0.6296 | 0.0601 | 0.0721 | 0.0343 | 0.829 | [0.035, 0.061] | 16.5 min (GPU) |
| CNN-AE | 0.0821 | 0.6296 | 0.0675 | 0.0810 | 0.0407 | 0.731 | [0.039, 0.068] | ~10 min (GPU) |

## Validation metric (sanity)

| Model | Val event-F0.5 at chosen threshold |
|---|---:|
| PCA | 0.8333 |
| LSTM-AE | 0.0565 |
| CNN-AE | 0.0737 |

## Score range diagnostic

A wide dynamic range on val scores is a precondition for a clean threshold sweep:

| Model | Val score range | Test_intern range |
|---|---|---|
| PCA | `[0.018, 664.996]` (37 000× spread) | `[0.026, 3.090]` |
| LSTM-AE | `[0.688, 0.967]` (~1.4× spread) | `[0.705, 0.987]` |
| CNN-AE | `[0.586, 0.804]` (~1.4× spread) | `[0.588, 0.821]` |

The two AEs' MSE distributions are almost flat — their threshold sweep cannot find a separation that catches events without flooding nominal alarms.

## Reading the verdict

- **PCA dominates by ~12×** on event-F0.5. With k=38 it perfectly distinguishes anomalous from nominal windows in this representation.
- **AEs hit the same recall ceiling (0.6296)** — they detect the same easy events and miss the same hard ones; the model architecture does not change which events are reachable.
- **AE precision is the failure mode**, not recall. Both AEs flag huge swaths of nominal windows as anomalies.
- **PCA's ESA corrected drop** (0.984 → 0.474) is real: PCA still over-flags nominal rows at row-level density, just less aggressively than the AEs.
- **Row F1 is low for everyone** (≤ 0.23) because all three models flag at window granularity (100 rows at a time), not per-row.

## What this says about the dataset

The fact that a linear PCA reconstruction beats two non-linear neural baselines is a strong signal that anomalies in this dataset live in a **low-rank linear subspace** of the 5,800-feature window space. The AEs are wasting capacity reconstructing both classes well, instead of learning a tight nominal manifold whose residual blows up on anomalies.

## Reproducibility

Each report has its own per-NB summary:

- [11-pca-summary.md](11-pca-summary.md)
- [12-lstm_ae-summary.md](12-lstm_ae-summary.md)
- [13-cnn_ae-summary.md](13-cnn_ae-summary.md)

Build scripts:

- [scripts/build_nb11.py](../scripts/build_nb11.py)
- [scripts/build_nb12.py](../scripts/build_nb12.py)
- [scripts/build_nb13.py](../scripts/build_nb13.py)

Model artifacts in `models/`:

- `pca_bootcamp.pkl`
- `lstm_ae_bootcamp.keras`
- `cnn_ae_bootcamp.keras`

## Recommended next steps

1. **Don't ship either AE in current form** — they cost 10–17× more compute and produce 12× worse F0.5.
2. **Try a wider PCA threshold sweep** with bootstrap-stable threshold selection (the val 0.833 → test 0.984 jump suggests the sweep got lucky).
3. **For AE improvements:** overlapping-window scoring, longer training, deeper stacks, per-channel weighted MSE, or scoring on PCA-residual features instead of raw inputs.
4. **Ensemble:** PCA + best-AE residual concatenated → logistic head, trained on val. Cheap, may help even though AE is weak alone.
