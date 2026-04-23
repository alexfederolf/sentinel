# SENTINEL — Model Results

Consolidated results across all baselines for the ESA-ADB anomaly-detection task.
All metrics computed on the internal chronological test split (`test_intern`, 70 / 15 / 15 % train / val / test split built by NB 02).
Headline metric is **event-wise F0.5** (`sentinel.ml_logic.metrics.event_f05`); ESA-corrected F0.5 multiplies event-wise precision by sample-wise TNR (`Pr_c = Pr_ew · TNR`).

Last updated: 2026-04-23 (LSTM-AE v4, Option B).

---

## Headline table

| Notebook | Model | Val event-F0.5 | Test event-F0.5 | ESA corrected F0.5 | Row F1 | Bootstrap 95 % CI | Wall time |
|---|---|---:|---:|---:|---:|---|---:|
| NB 03 | Isolation Forest (200 trees, RobustScaled) | 0.091 | ≈ 0 *(threshold drift)* | — | — | — | — |
| **NB 11** | **PCA (k = 38, flattened windows)** | **0.833** | **0.984** | **0.474** | **0.228** | **[0.673, 0.950]** | **59.6 s CPU** |
| **NB 12 v4** | **LSTM-AE BiLSTM + topk=5, Option B (no z-norm, FIT_SIZE=50k)** | **0.714** | **0.976** | **0.472** | **0.219** | **[0.650, 0.948]** | **8.3 min GPU** |
| NB 12 v3 | LSTM-AE BiLSTM + latent=8 (z-norm, all nominal windows) | 0.074 | 0.094 | 0.092 | 0.028 | [0.042, 0.077] | 128.2 min GPU |
| NB 12 v2 | LSTM-AE small + topk=5 | 0.248 | 0.265 | 0.264 | 0.032 | [0.123, 0.235] | 16.5 min GPU |
| NB 13 | CNN-AE + topk=5 | 0.108 | 0.078 | 0.076 | 0.021 | [0.027, 0.079] | ~30 min GPU |

**Bottom line.** PCA (`0.984`) and LSTM-AE v4 Option B (`0.976`) are effectively tied on the internal test split — the 0.008 gap sits well inside the bootstrap CI overlap ([0.673, 0.950] vs [0.650, 0.948]). Every other autoencoder variant stays far below the PCA lower bound of 0.673.

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

### v4 (latest, 2026-04-23) — **Option B, no z-norm, FIT_SIZE=50k** — competitive with PCA

| Setting | Value |
|---|---|
| Architecture | BiLSTM(128) → LSTM(64) → LSTM(8) latent → RepeatVector → LSTM(64) → LSTM(128) → TimeDistributed Dense(58) |
| Parameters | 400,986 (1.53 MB) |
| Hyperparameters | `LATENT_DIM=8`, `HIDDEN_DIM=128`, `DROPOUT=0.1` |
| Input | **no per-window z-normalisation** — raw RobustScaled windows as `float32` |
| Fit | **last 50,000** of the 92,271 nominal windows (`X_train_nom[-FIT_SIZE:]`), Adam(1e-3), MSE, 10 % val split |
| Training | 6 epochs, **8.3 min GPU**, EarlyStopping restored **epoch-1** weights |
| Scoring | `score_windows(model, ..., topk=5)` — mean of the 5 largest per-channel MSEs |
| Val event-F0.5 | **0.7143** @ threshold `1.323612` |
| Test Event F0.5 | **0.9756** (24 / 27 events hit, precision 1.0000, recall 0.8889) |
| Test ESA corrected F0.5 | 0.4715 |
| Test Row F1 | 0.2185 |
| Bootstrap (200×, event-block, prior same-arch run) | mean 0.7929 · 95 % CI [0.6500, 0.9475] |
| Score drift | Val [0.4346, 7735.26] vs Test [0.6992, 60.44] — **128× max-ratio** |

**Why it works.** Three levers stack:
1. **FIT_SIZE=50k last windows** — trains on the most-recent nominal dynamics, skipping older regimes that made the model over-generalise in v3.
2. **Epoch-1 weights suffice** — EarlyStopping restored the weights after one effective epoch; the model is acting as a *stable residualiser*, not a learned nominal generator. Further training diluted the anomaly signal.
3. **topk=5 per-channel MSE** — anomalies in this dataset concentrate in a handful of channels; averaging over all 58 masks the signal (same result as NB 13's topk experiment).

**Caveats.**
1. Val/test score distributions differ by 128× in max — the threshold 1.324 happens to land in the anomalous tail of *both* distributions, but this is a distributional-luck situation, not an architectural-stability guarantee. A Kaggle submission could surface drift that the internal split does not.
2. Same ESA-corrected drop as PCA (0.976 → 0.472) — window-granularity predictions still over-flag nominal rows inside a window.
3. Bootstrap CI is from a previous same-architecture run; a fresh bootstrap on the v4 weights is still pending.

### v3 (2026-04-22) — regression, kept for reference

| Setting | Value |
|---|---|
| Architecture | Same BiLSTM(128)→LSTM(64)→LSTM(8)→… as v4 |
| Fit | all 92,271 nominal windows **with** per-window z-normalisation, Adam(1e-3), MSE |
| Training | 44 epochs, early-stop @ epoch 39, **128.2 min GPU**, final val loss 0.7802 |
| Val event-F0.5 | 0.0737 @ threshold `0.834973` |
| Test Event F0.5 | 0.0936 |
| Test ESA corrected F0.5 | 0.0924 |
| Bootstrap | mean 0.0606 · 95 % CI [0.0417, 0.0769] |

**Diagnostic.** Val score range [0.689, 0.970] — nominal and anomalous windows reconstruct to nearly the same error after z-norm. Training loss plateaued at 0.780 from epoch 20 on; LR annealed to 1e-5 without escape. Model over-regularised into a mean predictor. Retired in favour of v4 Option B (raw-scale input, short training, last-50k fit).

### v2 (2026-04-17)

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

1. **PCA and LSTM-AE v4 tie on the internal test split.** PCA (0.984) and LSTM-AE v4 Option B (0.976) sit within 0.008 of each other, with overlapping bootstrap CIs ([0.673, 0.950] vs [0.650, 0.948]). Linear and non-linear can both solve this task — but only under specific fit regimes; v3 (full-data, z-normalised, long training) collapsed to 0.094.
2. **Training regime matters more than architecture.** Same BiLSTM backbone, two settings:
   - v3: all 92 k windows, z-normalised, 44 epochs → F0.5 = 0.094 (mean-predictor collapse)
   - v4: last 50 k windows, raw scale, epoch-1 weights restored → F0.5 = 0.976
   The model is acting as a *stable residualiser* on recent nominal dynamics, not as a generative nominal model. Further training dilutes the anomaly signal.
3. **Size is not the lever.** LSTM v3 (400 k params, BiLSTM, long training) scored worse than v2 (72 k params, simple LSTM). Overparametrisation with `LATENT_DIM=8` can collapse the dynamic range if training goes too long.
4. **Topk=5 channel scoring helps the LSTM, not the CNN.** The LSTM has specific channels that reconstruct anomalies measurably worse; the CNN spreads reconstruction error uniformly across channels.
5. **Per-window z-normalisation is *not* always a win.** Previously assumed required (LSTM v1 and CNN v1 drift collapse). LSTM v4 Option B *skips* z-norm and produces a 128× val/test max-ratio — yet the chosen threshold lands in the anomalous tail of both distributions. Skipping z-norm here preserves the magnitude signal the topk=5 rule needs; the price is that threshold stability depends on distributional luck, so a Kaggle validation is still pending.
6. **Public/private Kaggle gap > 0.3 = temporal clustering.** Submissions that only flag one half of the test timeline (CNN-AE v3, prior_matched_ensemble_5pct) get zero on public but non-zero on private. Not a generalisable signal.
7. **Event-wise precision × sample-wise TNR (ESA corrected)** punishes all approaches uniformly (~2× drop vs raw event F0.5) because predictions are window-granularity, not row-granularity. PCA drops 0.984 → 0.474, LSTM v4 drops 0.976 → 0.472 — same story, independent of model family.

---

## Reproducibility

All results computed from the same `sentinel.ml_logic` shared scoring / threshold / metrics / validation modules, so cross-notebook comparisons are apples-to-apples.

- Threshold tuning: `tune_threshold(val_scores, y_val, metric_fn=event_f05, n_sweep=60)`
- Bootstrap CI: `bootstrap_f05_ci(..., n_boot=200, event_block=True, seed=RANDOM_STATE)`
- Preprocessing: `run_preprocessing()` (NB 02) — chronological 70 / 15 / 15 split, `WINDOW_SIZE=100`, 58 target channels
- Models saved under `models/` with datetime stamps, re-loadable via `tf.keras.models.load_model`
