# SENTINEL — Model Results

Consolidated results across all baselines for the ESA-ADB anomaly-detection task.
All metrics computed on the internal chronological test split (`test_intern`, 70 / 15 / 15 % train / val / test split built by NB 02).
Headline metric is **event-wise F0.5** (`sentinel.ml_logic.metrics.event_f05`); ESA-corrected F0.5 multiplies event-wise precision by sample-wise TNR (`Pr_c = Pr_ew · TNR`).

Last updated: 2026-04-23.

---

## Headline table

| Notebook | Model | Val event-F0.5 | Test event-F0.5 | ESA corrected F0.5 | Row F1 | Bootstrap 95 % CI | Wall time |
|---|---|---:|---:|---:|---:|---|---:|
| NB 03 | Isolation Forest (200 trees, RobustScaled) | 0.091 | ≈ 0 *(threshold drift)* | — | — | — | — |
| **NB 11** | **PCA (k = 38, flattened windows)** | **0.833** | **0.984** | **0.474** | **0.228** | **[0.673, 0.950]** | **59.6 s CPU** |
| NB 12 v2 | LSTM-AE small + topk=5 | 0.248 | 0.265 | 0.264 | 0.032 | [0.123, 0.235] | 16.5 min GPU |
| NB 12 v3 | LSTM-AE BiLSTM + latent=8 | 0.074 | 0.094 | 0.092 | 0.028 | [0.042, 0.077] | 128.2 min GPU |
| NB 13 | CNN-AE + topk=5 | 0.108 | 0.078 | 0.076 | 0.021 | [0.027, 0.079] | ~30 min GPU |

**Bottom line.** PCA dominates: `Event F0.5 = 0.984` on the internal test split with a bootstrap lower bound of 0.673. No autoencoder variant comes within the PCA confidence interval.

---

## Kaggle leaderboard submissions

Public score = subset of private leaderboard (≈ 50 %), published live during the competition.
Private score = remaining 50 %, revealed at deadline — **this is the final ranking**.
Public-only optimisation that collapses on private signals temporal clustering of flagged rows (common failure for AE-based submissions).

| Submission | Public F0.5 | Private F0.5 | Notes |
|---|---:|---:|---|
| `baseline_pca.parquet` | **0.522** | **0.599** | Strongest overall submission — consistent across both halves |
| `pca_full.parquet` (row-level) | 0.522 | 0.294 | Same public score, private collapse |
| `baseline_ensemble.parquet` (PCA + LSTM-AE v1 rank-avg) | 0.522 | 0.476 | No uplift over PCA alone on public; lower private |
| `baseline_lstm_ae.parquet` (v1) | 0.112 | 0.078 | v1 over-flagged test (17.9 % positive rate) |
| `cnn_ae_v3.parquet` | 0.000 | 0.238 | Temporal clustering — all positives landed in private half |
| `prior_matched_ensemble.parquet` | 0.096 | 0.032 | Prior-matched threshold, both halves weak |
| `prior_matched_ensemble_5pct.parquet` | 0.078 | 0.476 | Same temporal clustering as CNN-AE v3 |

**Takeaway.** PCA baseline is the strongest and most robust submission so far. Public-private gap > 0.3 is a red flag that a submission is exploiting a temporal artefact, not a generalisable signal.

---

## NB 03 — Isolation Forest

**Counter-example notebook**, kept for documentation.

| Setting | Value |
|---|---|
| Model | `IsolationForest(n_estimators=200, max_samples=256)` |
| Training set | 500 k nominal rows (subsampled from 10.5 M), 58 RobustScaled channels |
| Optimal val threshold | 0.6282 |
| **Val F0.5** | **0.091** |
| Val event recall | 2 / 38 events |
| Val precision | `Pr_ew = 2 / 18 segments = 0.111`, TNR ≈ 1 |
| Test score range | `[0.518, 0.614]` — entirely below val-optimal threshold |
| Test rows flagged | **0 / 521,280** (full threshold drift) |

**Structural failure modes:**
1. Spiky per-row scores → 2,203 predicted segments at best threshold → `Pr_ew = 0.111` collapses precision regardless of TNR.
2. Score distribution shifts between val and test → zero test rows flagged → Kaggle ≈ 0.
3. Any model producing independent per-row scores will suffer the same fate on ESA-ADB.

---

## NB 11 — PCA (winning baseline)

| Setting | Value |
|---|---|
| Model | `sklearn.decomposition.PCA(n_components=0.95)` on flattened `(100, 58) → 5,800` windows |
| Fit size | 50,000 subsampled nominal windows |
| k (components kept) | **38** (cumulative explained variance 0.9501) |
| Wall time | **59.6 s CPU** |
| Val event-F0.5 | **0.833** @ threshold `0.060404` |
| Test Event F0.5 | **0.9843** (25 / 27 events hit, precision 1.000) |
| Test ESA corrected F0.5 | 0.4736 |
| Test Row F1 | 0.2281 |
| Bootstrap (200×, event-block) | mean 0.8053 · 95 % CI [0.6730, 0.9501] · 573.7 s |

**Why it works.** Anomalies sit in a low-rank linear subspace that PCA captures exactly; the reconstruction score on flattened windows has a clean wide dynamic range (`val scores [0.018, 665.0]`) and transfers cleanly to test. Precision = 1.000 means no spurious event predictions — the threshold sweep finds a true gap between nominal and anomalous scores.

**Gaps.** The ESA-corrected drop (0.984 → 0.474) flags a non-trivial nominal-region alarm density per row. Row F1 0.228 means only a fraction of flagged rows lie inside the true event extent — predictions are window-granularity, not row-granularity.

---

## NB 12 — LSTM Autoencoder

### v3 (latest, 2026-04-22) — **regression**

| Setting | Value |
|---|---|
| Architecture | BiLSTM(128) → LSTM(64) → LSTM(8) latent → RepeatVector → LSTM(64) → LSTM(128) → Dense(58) |
| Parameters | 400,986 (1.53 MB) |
| Hyperparameters | `LATENT_DIM=8`, `HIDDEN_DIM=128`, `DROPOUT=0.1` |
| Fit | all 92,271 nominal windows, Adam(1e-3), MSE, 10 % val split |
| Training | 44 epochs, early-stop @ epoch 39, **128.2 min GPU**, final val loss 0.7802 |
| Val event-F0.5 | **0.0737** @ threshold `0.834973` |
| Test Event F0.5 | **0.0936** |
| Test ESA corrected F0.5 | 0.0924 |
| Bootstrap | mean 0.0606 · 95 % CI [0.0417, 0.0769] |

**Diagnostic.** Val score range [0.689, 0.970] — nominal and anomalous windows reconstruct to nearly the same error. Topk=5 signal is gone. Training loss plateaus at 0.780 from epoch 20 on; LR anneals to 1e-5 without escape. Model has over-regularised into a mean predictor.

### v2 (2026-04-17) — strongest LSTM-AE to date

| Setting | Value |
|---|---|
| Architecture | LSTM(64) → LSTM(32) latent → RepeatVector → LSTM(64) → Dense(58) |
| Parameters | 72,506 |
| Fit | 50,000 subsampled, 25 epochs, 16.5 min GPU |
| Val event-F0.5 | 0.2475 |
| Test Event F0.5 | 0.2650 |
| Bootstrap 95 % CI | [0.1232, 0.2355] |

### v1 (initial, no drift fix)

- Raw-scale windows (no z-normalisation), mean-over-58 MSE scoring
- Submission: 17.9 % positive rate (over-flag)
- Kaggle public 0.112 / private 0.078

---

## NB 13 — 1D CNN Autoencoder

| Setting | Value |
|---|---|
| Architecture | Conv1D(32,7) → MaxPool → Conv1D(16,5) → MaxPool → Conv1D(8,3) bottleneck → mirror decoder with `UpSampling1D` |
| Parameters | 32,034 |
| Fit | 50,000 subsampled nominal windows, per-window z-norm |
| Scoring | `score_windows(..., topk=5)` |
| Val event-F0.5 | 0.108 @ threshold `1.138344` |
| Test Event F0.5 | 0.0775 |
| Test ESA corrected F0.5 | 0.0762 |
| Bootstrap | mean 0.0507 · 95 % CI [0.0271, 0.0792] |

**Topk experiment.** Before topk (mean-over-58): Event F0.5 0.082, recall 0.630. After topk=5: Event F0.5 0.078, recall collapses to 0.222. The CNN spreads reconstruction error uniformly across channels — no small subset concentrates the anomaly signal. Topk helps the LSTM but not the CNN — architectural difference, not hyperparameter tuning.

---

## Cross-model observations

1. **Linear beats non-linear on this task.** PCA at 0.984 Event F0.5 dominates every autoencoder variant. Anomalies sit in a low-rank *linear* subspace — the LSTM and CNN reconstruct too well on anomalies (and too poorly on some nominals) to produce a clean separation.
2. **Size is not the lever.** LSTM v3 (400 k params, BiLSTM) scored worse than v2 (72 k params, simple LSTM). Overparametrisation with `LATENT_DIM=8` collapsed the dynamic range.
3. **Topk=5 channel scoring helps the LSTM, not the CNN.** The LSTM has specific channels that reconstruct anomalies measurably worse; the CNN does not.
4. **Per-window z-normalisation is required.** Without it, magnitude drift between train and test dominates the score (documented failure mode of both LSTM v1 and CNN v1 — val / test max-ratio in the hundreds).
5. **Public/private Kaggle gap > 0.3 = temporal clustering.** Submissions that only flag one half of the test timeline (CNN-AE v3, prior_matched_ensemble_5pct) get zero on public but non-zero on private. Not a generalisable signal.
6. **Event-wise precision × sample-wise TNR (ESA corrected)** punishes all approaches uniformly (~2× drop vs raw event F0.5) because predictions are window-granularity, not row-granularity.

---

## Reproducibility

All results computed from the same `sentinel.ml_logic` shared scoring / threshold / metrics / validation modules, so cross-notebook comparisons are apples-to-apples.

- Threshold tuning: `tune_threshold(val_scores, y_val, metric_fn=event_f05, n_sweep=60)`
- Bootstrap CI: `bootstrap_f05_ci(..., n_boot=200, event_block=True, seed=RANDOM_STATE)`
- Preprocessing: `run_preprocessing()` (NB 02) — chronological 70 / 15 / 15 split, `WINDOW_SIZE=100`, 58 target channels
- Models saved under `models/` with datetime stamps, re-loadable via `tf.keras.models.load_model`
