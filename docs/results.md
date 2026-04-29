# SENTINEL — Project Results & Knowledge Base

ESA Anomaly Detection Benchmark (ESA-ADB) / Kaggle competition.
This document is the **single source of truth** for what was tried on Kaggle, what worked,
what failed, and why. If you only read one file to catch up on the project, read this one.

**Last updated:** 2026-04-29 · **Current Kaggle champion:** `11d_pca_hybrid_envzscore.ipynb` — public **0.867** / **private 0.915** ⭐

---

## 0. The Kaggle challenge — what makes this hard

The task: given **58 channels of spacecraft telemetry** (≈14.7 M labelled training rows + 521 k unlabelled test rows), output a 0/1 anomaly prediction for every test row. Unsupervised — the test labels are hidden.

Four things make it genuinely difficult. Each one bit us at least once during the project.

### Challenge 1 — Distribution drift between train and Kaggle test (the biggest problem)

The data isn't stationary. Between train-start and the Kaggle test set, **10 sensor channels** (`29, 14, 21, 30, 38, 22, 31, 39, 15, 23`) shift their mean by +4.5–6.7 standard deviations *in lock-step* — i.e. one physical subsystem changing regime. Mean KS(train, test) is 0.43 across all channels, with a max of 0.97 (channel_15). For comparison, KS > 0.2 is already considered substantial in most ML problems.

**What this means in practice:** any model fit on early training data sees a different physical regime than the one Kaggle scores it on. A reconstruction model trained to recognise "normal" on the early period will see the late period (Kaggle test) as anomalous everywhere, even when it isn't. This is the **single biggest source of train/test metric divergence** in the project.

NB 18 and NB 19 contain the drift diagnosis. Every working submission in §6 is, in some way, a different strategy for handling these 10 channels.

### Challenge 2 — The metric weighs precision 4× more than recall

The Kaggle scoring metric is **F0.5** (β = 0.5), not F1. That means false positives hurt 4× more than missed events do. A submission that catches 24/25 events but flags 5 % of nominal rows scores worse than one that catches only 14/25 events and flags 1 %.

The intuitive instinct — "catch as many anomalies as possible" — is the **wrong** instinct here. Most attempts to lift event recall by making the model more sensitive end up flooding nominal rows with false positives, and the F0.5 score collapses. See §1 for the exact formula and §4 for the "drift-flood" failure mode this produces.

### Challenge 3 — Public and private leaderboards score differently

Kaggle splits the 521 k test set roughly 50/50 into a **public** leaderboard (visible during the competition) and a **private** leaderboard (revealed at deadline; the one that determines final ranking). On this dataset the two halves come from different parts of the timeline, and submissions score very differently on each.

A real example from this project: a hybrid variant I tuned to maximise local val/test scored **public 0.897 / private 0.903**. A slightly less-aggressive variant from Ekaterina scored **public 0.867 / private 0.915**. Public says mine wins (+0.030). Private says hers wins (+0.012). **Private is what counts** — the "worse" public is the actual competition winner. See §5.3 for the full story and §5.4 for the cautionary post-mortem.

### Challenge 4 — Local validation doesn't predict private leaderboard

Compounding Challenge 3: even our best local val/test split (rows 10.7 M / 12.7 M / 14.7 M of train) doesn't reliably predict private score. Two specific traps:

- **Sharp local optima don't transfer.** `ENV_MA_WINDOW=20_000` gave local Test 0.8545 — but neighbours `ma_w=15k` and `ma_w=25k` dropped to 0.71-0.80. That sharp peak was the smoking gun: sharp = overfit. Smoothly-flat regions of parameter space generalise to private; pointy peaks don't.
- **Adding "obviously useful" channels often regresses.** NB 11e added 17/25/34 to the 6-channel champion (NB 11c) and got higher local val (0.872 vs 0.853). Kaggle: 0.832 / 0.853 — both lower than 11c's 0.897 / 0.887.

**Rule of thumb:** prefer flat plateaus to sharp peaks. Submit ablations to Kaggle rather than trusting a single local split. When in doubt, do the simpler thing.

---

**The thread through all four:** drift dominates. Every working submission in §6 attacks drift — by selecting away from the drifting channels (NB 11c), residualising against an envelope baseline (NB 11d hybrid), or fitting on the tail of train where the regime matches test. Every failed submission, including the high-capacity LSTM and CNN autoencoders, has a drift-related root cause hidden behind a different symptom.

---

## 1. The metric — corrected event-wise F0.5

Kaggle scores submissions with the **ESA-ADB corrected event-wise F0.5** (`corrected_event_f05` in `sentinel.ml_logic.metrics`).

### What "event-wise" means

A predicted row is judged in the context of the **ground-truth event** it overlaps, not in isolation:

- **TP_events** = the number of *true* anomaly events that have at least one overlapping predicted-positive row. (Catching the event "at all" counts as a hit — overlap fraction does not matter.)
- **FN_events** = N_events − TP_events.
- **FP_pred_events** = the number of *predicted* contiguous positive segments that don't overlap any true event. (Each spurious flagged block counts once, regardless of length.)

So 100 isolated false-positive flags become 100 FP_pred_events; one giant 100 k-row false-positive block becomes 1 FP_pred_event.

### The metric formula (ESA correction)

```
Pr_ew     = TP_events / (TP_events + FP_pred_events)   # event-wise precision
recall    = TP_events / N_events                        # event-wise recall
TNR       = TN_rows / (TN_rows + FP_rows)               # row-level true-negative rate

Pr_c      = Pr_ew · TNR                                 # ESA-corrected precision
F0.5      = (1 + 0.5²) · Pr_c · recall / (0.5² · Pr_c + recall)
          = 1.25 · Pr_c · recall / (0.25 · Pr_c + recall)
```

The β=0.5 means **precision (Pr_c) counts 4× as much as recall**.

### What the TNR correction fixes

Without the TNR correction, a "predict everything" submission gets `Pr_ew = TP_events / TP_events = 1.0`, `recall = 1.0`, F0.5 = 1.0 — the metric is gameable by flagging the whole timeline. The TNR correction multiplies `Pr_ew` by the row-level true-negative rate. Predict 50 % of rows as positive ⇒ TNR ≈ 0.5 ⇒ `Pr_c` halves even if `Pr_ew = 1`. This kills the "drift flood" failure mode (§4) where models accidentally output one giant predicted block that overlaps every late event.

### Practical consequence — what to optimise

For any new model, monitor these four numbers in this order:
1. **`FP_pred_events`** — should ideally be 0 at the chosen threshold. If it's 0, `Pr_ew = 1` and the rest of the metric is dominated by TNR and recall. The threshold sweep in `tune_threshold` is essentially looking for the lowest threshold where `FP_pred_events = 0` still holds.
2. **Flag rate** — fraction of test rows predicted positive. Should be 0.5 %–10 %. Below 0.5 % under-flags; above 10 % is the flood.
3. **TNR** — should be ≥ 0.99 for a submittable model.
4. **Event recall** — between 0.5 and 0.8 is realistic given the metric's recall weight; pushing recall higher costs precision.

**Always tune on `corrected_event_f05`, never on plain `event_f05`.** The latter has no TNR penalty and converges on the flood.

---

## 2. Data, splits, infrastructure

### Data shape

| | Rows | Events | Anomaly rate | Labels |
|---|---:|---:|---:|:-:|
| `train.parquet` | 14,728,321 | 190 | 10.48 % | yes |
| `test.parquet` (Kaggle) | 521,280 | — | — | **no** |
| Internal val (Kaggle split, rows 10.7 M – 12.7 M) | 2,000,000 | 26 | — | yes |
| Internal test (Kaggle split, rows 12.7 M – 14.7 M) | 2,028,321 | 25 | — | yes |

Event lengths span 1 to 116 061 samples (median 602). Wide range ⇒ fixed 100-row windows are a compromise: short events get diluted, long events get over-represented.

### The 58 target channels

Of 76 sensor channels, 58 are competition-relevant (`data/raw/target_channels.csv`). Notable groups identified by drift analysis (NB 18/19):

- **Frequency cluster `41-46`** (6 channels) — the spectral magnitude signal for the dominant subsystem. Stable across train→test. **The single most important channel set for Kaggle scoring** (§6).
- **Drifters `29, 14, 21, 30, 38, 22, 31, 39, 15, 23`** (10 channels) — lock-step level-shift between train and Kaggle test. Useful as anomaly-signal sources via envelope/z-score scoring (§6.6) but lethal as direct PCA inputs.
- **Stable reference `42, 64, 65, 70-75`** (8 channels) — drift-free, no anomaly signal either.
- **Low-variance cluster `57-60`** — high relative drift but tiny absolute MSE; misleading on local-test, near-invisible on Kaggle.

### Two split conventions

1. **Bootcamp split** (`data/processed/`): 70/15/15 chronological with anomaly-aware boundary snapping. Used by NB 11/12/13/14 for apples-to-apples comparison. **Reported `test_intern` numbers on this split are systematically optimistic for Kaggle** because the bootcamp test sits before the Kaggle drift regime.
2. **Kaggle split** (`data/processed/kaggle/`): all of train kept for fitting; `test.parquet` scored separately. Internal val/test carved from rows 10.7 M / 12.7 M / 14.7 M, snapped so a ±2 000-row neighbourhood is fully nominal. Used by every Kaggle-facing notebook (`04`, `11c`, `11d_*`, `11e_*`, `12b`, `13`, `20`).

### Shared modules (`src/sentinel/ml_logic/`)

- `scorer.score_windows(model, X_rows, win=100, topk=None)` — reshape rows into non-overlapping windows, compute reconstruction MSE, broadcast back to rows. The smoothing this provides is what keeps `FP_pred_events = 0` (see §3).
- `metrics.corrected_event_f05` — the Kaggle metric. Always tune on this.
- `thresholds.tune_threshold(scores, y_true, metric_fn, n_sweep=80)` — log-spaced sweep, argmax the metric.
- `viz.plot_timeline`, `viz.plot_event_analysis` — standard timeline + per-event detection visualisations.

### Constants (`src/sentinel/params.py`)

`WINDOW_SIZE=100`, `FIT_SIZE=50_000`, `RANDOM_STATE=42`, `PCA_THRESHOLD=0.060404`. The blessed model pickles (`models/pca.pkl`, `models/pca_kaggle.pkl`, `models/lstm_ae.keras`, `models/cnn_ae.keras`, `models/scaler.pkl`) are protected — experiments save with `_nb{XX}_{timestamp}` suffix.

---

## 3. The two scoring decisions that matter

### 3.1 Window-MSE > per-row MSE

The single most important design choice. Reconstruction error is reduced to **one score per 100-row window**, then broadcast to its 100 rows. This produces a smooth curve where the threshold sweep can find a gap.

Per-row MSE produces a spiky curve, no clear gap, hundreds of FP_pred_events. NB 04 ablation: window-mean F0.5 0.770 → public 0.522. Per-row F0.5 0.698 → public **0.277**.

NB 03 (Isolation Forest, per-row by design) hits the same wall — 2/38 events caught, threshold drift, Kaggle ≈ 0. Any model with native per-row scoring has to be reduced to windows before it competes.

### 3.2 Drift mitigation > model capacity

Three mitigation layers, increasing in effectiveness on this dataset:

| Layer | Technique | Used by | Effect |
|---|---|---|---|
| **Channel selection** | Drop drifting channels, fit on stable subset | NB 11c (channels 41-46) | NB 04 0.522/0.599 → NB 11c **0.897/0.887** (+0.30 public, biggest single jump) |
| **Score-level detrending** | Subtract rolling-median from row scores | side experiments (PCA + median detrend) | Lifts ESA F0.5 from drift-flood baseline to ≈ 0.874 on local proxy; ties NB 04 on Kaggle private (0.599) |
| **Per-channel processing** | Envelope-residual + z-score on drifters | env stream of 11d hybrid (channels 14/21/29) | Adds events that PCA misses; lifts hybrid private from 0.887 → **0.915** |

PCA, LSTM-AE, CNN-AE all give comparable scores when given the *same* drift-mitigation recipe. Architecture is a smaller lever than recipe.

---

## 4. The drift-flood failure mode

The pathology that contaminated every "0.98 Event F0.5 on test_intern" headline before NB 04.

**Pattern:** model fit on early train; reconstruction MSE drifts upward across test; val-tuned threshold lands above pre-drift baseline but below post-drift; predicted-positive rows are one giant block from the drift onset to the end; this block happens to contain all late events.

**Consequence:** `Pr_ew = 1.0`, `event_f05 ≈ 0.98`, but row-level TNR ≈ 0.5 → `corrected_event_f05 ≈ 0.47` and Kaggle ≈ 0.

**What catches it:**
- `corrected_event_f05` (the Kaggle metric, always tune on this).
- Flag rate > 20 % at the chosen threshold.
- Timeline plot showing one contiguous predicted block starting mid-test.

**What fixes it:** any of the three drift-mitigation layers in §3.2.

---

## 5. Per-notebook Kaggle results

### 5.1 `04-pca_kaggel_1.ipynb` — full-58 PCA baseline

| Setting | Value |
|---|---|
| Channels | **all 58** |
| Model | `PCA(n_components=0.95)` on flattened (100 × 58) → 5,800-dim windows |
| k components | 39 (cumulative EV 0.95) |
| Fit | 80/20 split (`SPLIT_IDX=11_782_656`), tune metric `corrected_event_f05` |
| Best threshold | 0.110 |
| Local val F0.5 | 0.770 (events: 21/38, FP_pred=0, TNR=0.855) |
| **Kaggle public / private** | **0.522 / 0.599** |
| Submission | `kaggle/submissions/nb04_pca_50k_599.parquet` |

The first submission with healthy public/private gap (+0.077). All later notebooks beat it but it's the reference.

### 5.2 `11c-pca_6ch.ipynb` — 6-channel PCA (huge jump)

| Setting | Value |
|---|---|
| Channels | **6** (`channel_41..46`) |
| Model | `PCA(n_components=0.95)` on flattened (100 × 6) → 600-dim windows |
| Fit | tail-50k nominal windows of full-train (Kaggle split), `TRAIN_END=10.7M` |
| Best threshold | 0.028049 (val ESA F0.5 0.8534) |
| Local val ESA F0.5 | 0.8534 |
| Local test ESA F0.5 | 0.8217 (events 12/25 = 48 %) |
| Flag rate | val 0.26 % / test 0.15 % |
| **Kaggle public / private** | **0.897 / 0.887** |
| Submission | `kaggle/submissions/pca_nb11c_6ch_BEST.parquet` |

Why dropping 52 channels gave +0.375 public: the dropped channels include the 10 lock-step drifters and the misleading 57-60 cluster. Fitting on 6 stable spectral channels eliminates the drift-flood at the source. **This is the foundation for the hybrid champion (§5.3).**

### 5.3 `11d_pca_hybrid_envzscore.ipynb` ⭐ **current Kaggle champion**

OR-fusion of two streams with independent thresholds. Bit-identical to Ekaterina's `submissions/pca_pca_6ch_zsenv_3ch.parquet` (commit `93a611e`).

| Stream | Channels | Count | Recipe |
|---|---|---|---|
| **Freq (PCA)** | `channel_41..46` | **6** | PCA reconstruction error on raw scaled values, fit on tail-50k nominal windows, EV=0.95. Same as NB 11c. |
| **Env (z-score)** | `channel_14, 21, 29` | **3** | `rolling_min(env_w=200) − centered_MA(ma_w=5_000)`, normalize by nominal-train std and per-channel p99, aggregate via `top_p_mean(k=2)` over the 3 channels. |
| **Fusion** | — | — | OR. Independent thresholds tuned via `tune_threshold(corrected_event_f05)` on local val. |

| Setting | Value |
|---|---|
| Total channels used | **9** of 58 |
| `ENV_WINDOW` (envelope) | 200 |
| `ENV_MA_WINDOW` | **5,000** |
| `ENV_REF_PCT` | 99 |
| Freq threshold | 0.028049 |
| Env threshold | 1.236990 (k_env=2/3, val grid pick) |
| Local val ESA F0.5 | **0.9391** (events 20/26 = 76.9 %) |
| Local test ESA F0.5 | 0.8216 (events 14/25 = 56 %) |
| Flag rate | val 1.18 % / test 1.29 % |
| **Kaggle public / private** | **0.867 / 0.915** ⭐ |
| Submission | `kaggle/submissions/pca_hybrid_envzscore_BEST.parquet` (bit-identical to Ekaterina's `submissions/pca_pca_6ch_zsenv_3ch.parquet`) |

**Why it wins on private:** the env stream catches anomalies that the PCA stream misses, *without* flooding — because the envelope-residual baseline is itself drift-corrected (the centered MA absorbs lock-step level shifts in channels 14/21/29). The OR fusion only fires when an event has either spectral disturbance (PCA stream) or envelope deviation (env stream).

**How the two streams came together.** The freq stream is NB 11c's 6-channel PCA. The env stream originated in two precursor notebooks (`ek_baseline_zscore.ipynb` — per-channel rolling z-score with top-p% aggregation and F0.5 threshold tuning; `ek_freq_eda.ipynb` — FFT-based period detection that identified channels 14/21/29 as periodic carriers). The hybrid is a straight OR-fusion of those two streams with independent val-tuned thresholds. `ek_no_model_thr.ipynb` was used along the way to sanity-check which channels actually separate anomalies from nominal.

### 5.4 The `ma_w=20_000` overfit trap (cautionary subsection)

A variant of §5.3 with `ENV_MA_WINDOW=20_000` (instead of 5_000) and an added joint-threshold cell pushed local Val 0.9214 / local Test **0.8545** (above the 5k variant's 0.9391/0.8216). Looked like an improvement. Submitted as `pca_zsenv_pca41,42,43,44,45,46_zsenv14,21,29_k1_*.parquet`: Kaggle public **0.897** / private **0.903**.

The **+0.030 public was real**, but the **−0.012 private was the actual cost** — and Kaggle's final ranking is by private. The 20k variant overfit the local val/test split. Sharp local optimum (neighbours `ma_w=25k` drop to 0.71-0.80, `env_w=250` drop to 0.79) was the warning sign in retrospect.

**Memory marker:** prefer the version that reproduces Ekaterina's submission exactly. Param diff to reproduce 0.915: `ENV_MA_WINDOW=5_000`, joint-threshold disabled, `k_env` tuned by val grid (which lands on k=2/3 with val ESA 0.7829 vs k=1 0.7816).

---

## 6. Kaggle leaderboard — every submitted model

| Submission | Model / notebook | Channels | Public | **Private** | Δ | Notes |
|---|---|---:|---:|---:|---:|---|
| `pca_hybrid_envzscore_BEST.parquet` ⭐ | NB 11d hybrid (PCA 41-46 + env z-score 14/21/29, ma_w=5k) | **9** | 0.867 | **0.915** | +0.048 | **Current champion.** Bit-identical to Ekaterina's `submissions/pca_pca_6ch_zsenv_3ch.parquet`. |
| `pca_zsenv_..._k1_20260429_043247.parquet` | Same architecture, `ma_w=20_000` + joint thr | 9 | 0.897 | 0.903 | +0.006 | Won public, lost private — sharp local-test optimum overfit. |
| `pca_nb11c_6ch_BEST.parquet` | NB 11c (PCA on freq 41-46) | **6** | 0.897 | 0.887 | −0.010 | The big jump from full-58. Was champion before hybrid. |
| `pca_nb11_ch_sel*.parquet` | NB 11e (9-channel selection: 41-46 + 17/25/34) | 9 | 0.832 | 0.853 | +0.021 | Val-overfit; 9 channels including 17/25/34 added drift on Kaggle. |
| `nb20_pca_median.parquet` | NB 20 (PCA + score-level median detrend) | **58** | 0.454 | 0.599 | +0.145 | Ties NB 04 on private; loses 0.068 public. Detrend helps drift-heavy private half. |
| `nb04_pca_50k_599.parquet` | NB 04 (full-58 PCA baseline) | 58 | 0.522 | 0.599 | +0.077 | Reference baseline; healthy gap. |
| `pca_full.parquet` | Row-level PCA ablation | 58 | 0.522 | 0.294 | −0.228 | Same public, private collapse. Row-level scoring fails under drift. |
| `baseline_ensemble.parquet` | PCA + LSTM-AE rank-average | 58 | 0.522 | 0.476 | −0.046 | No uplift; lower private. |
| `baseline_lstm_ae.parquet` | NB 12 LSTM-AE v1 | 58 | 0.112 | 0.078 | −0.034 | Over-flag (17.9 % positive). |
| `cnn_ae_v3.parquet` | NB 13 CNN-AE v3 | 58 | 0.000 | 0.238 | +0.238 | Temporal clustering — all positives on private side. |

**Headline observations:**

- The **architecture move from 58→6 channels (NB 04 → NB 11c)** gave +0.375 public / +0.288 private. Single biggest jump in the project.
- The **OR-fusion with envelope z-score (NB 11c → NB 11d hybrid)** gave −0.030 public but **+0.028 private**. The env stream lifts private because that's where the drift-affected events live.
- **Public/private gap of |0.10| or more is the temporal-clustering red flag** (CNN-AE v3, prior_matched_5pct). The healthy submissions all have gap ≤ 0.10 with consistent direction.

---

## 7. Side experiments (not Kaggle-facing)

These notebooks explored alternative metrics or research directions and are **not** submitted to Kaggle. They are kept for context but did not feed into the leaderboard.

- **NB 11 (PCA, bootcamp split)** — primary diagnostic baseline for the drift-flood pathology. Tunes on `event_f05` (legacy metric, predates the metric-switch). Reports test_intern Event F0.5 0.9843 = drift-flood artefact. Useful as the canonical cautionary tale, not as a Kaggle submission.
- **NB 14 (`14-model_eval.ipynb`) and `14c-score-all.ipynb`** — viewers / multi-metric panels for any trained model. Used for diagnostic tables, not submission. Reports `event_f05`, `event_f1`, `event_f2`, `corrected_event_f05` side-by-side; `event_f0.5 ≫ corrected_event_f05` is the flood signature.
- **NB 16 (`16_anomalies.ipynb`)** — manual anomaly investigation per event. Used to characterise what the missed events look like (4 ultra-short clustered ~row 937-940k, 4 huge-z in continuously-drifting channels, 3 weak everywhere).
- **NB 18 / 19 (`level_shift_*`, `19-level_shift_2.ipynb`)** — the drift diagnosis notebooks. Output is the channel groupings used everywhere else (drifters list, stable reference list).
- **NB 22 / 22b / 23 (`api`, `test_slices`)** — FastAPI demo for the API showcase, not Kaggle.
- **`Alex - model 1/2/3/3b`** — early Isolation Forest / LSTM / LSTM-AE prototypes (2-3 cells each); superseded by NB 03/12/12b.

---

## 8. Open directions worth trying

Ranked by expected impact on Kaggle private:

1. **Submission ensembling** — union-OR of `pca_hybrid_envzscore_BEST.parquet` (private 0.915) with a complementary high-precision submission. If they catch different events, private may exceed both. Cheap to try.
2. **Sweep `ENV_MA_WINDOW ∈ {3k, 5k, 7k, 10k}` and submit each.** Don't trust local val — it picked 20k and that overfit. Submit them blind and read Kaggle.
3. **Cross-validate threshold selection on rolled splits** (multiple val windows averaged) instead of the fixed `TRAIN_END=10.7M / VAL_END=12.7M`. Reduces single-split overfitting that produced the ma_w=20k regression.
4. **Soft-OR fusion** — combined_score = `freq_norm + α · env_norm` for α ∈ {0.5, 1, 2}. Smoother than hard OR; may give better threshold transfer.
5. **Forecast-residual stream as a third OR channel** — AR(p) per channel on the env channels, score = |residual|. Different failure mode than envelope-MA, may catch the 4 ultra-short events the current hybrid misses without flooding.
6. **Shorter PCA windows (`WINDOW_SIZE=25` or `50`)** specifically for ultra-short events. Tested as 3rd-stream addition (failed locally with flag rate 4.5 %), but as a *replacement* for the freq stream's window=100 it's untested on Kaggle.

Anything that **catches more events** has to be checked for flag-rate inflation on Kaggle test specifically. Local-test recall ≠ Kaggle-private recall.

---

## 9. Reproducing the champion

```python
# In notebooks/11d_pca_hybrid_envzscore.ipynb:
ENV_WINDOW    = 200
ENV_MA_WINDOW = 5_000     # NOT 20_000
ENV_REF_PCT   = 99
FREQ_NAMES    = [f'channel_{i}' for i in range(41, 47)]   # 6 channels
ENV_NAMES     = ['channel_14', 'channel_21', 'channel_29']  # 3 channels
# joint-threshold cell: SKIPPED (independent thresholds only)
# k_env: tune by val grid (lands on k=2/3)
```

Run the notebook end-to-end on the Kaggle split (`DATA_SOURCE='kaggle'`). The submission cell writes `kaggle/submissions/pca_zsenv_{tag}_{ts}.parquet`. The output is bit-identical to `submissions/pca_pca_6ch_zsenv_3ch.parquet` (already on the leaderboard at private 0.915) — verified row-by-row, 521,280/521,280 match.

**Pinned constants** (`src/sentinel/params.py`): `WINDOW_SIZE=100`, `FIT_SIZE=50_000`, `RANDOM_STATE=42`. Blessed pickles (do not overwrite): `models/pca.pkl`, `models/pca_kaggle.pkl`, `models/lstm_ae.keras`, `models/cnn_ae.keras`, `models/scaler.pkl`. Experiments save with `_nb{XX}_{YYYYmmdd_HHMMSS}` suffix.
