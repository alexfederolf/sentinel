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
- `fusion.fusion_diagnostics(y_base, y_new, y_true)` — segment-aware OR-fusion safety check. Counts `FP_pred_events`, not rows; returns verdict ∈ {`reject`, `borderline`, `submit`, `submit_strong`}. Use **before** any OR-fusion submission. Added 2026-04-29 after the freq6 fusion lost 0.070 private (§4.1).
- `cv.run_cv(env_ma_window=...)` / `cv.run_sweep([...])` — rolled-origin K-fold CV harness. Tunes `(thr_freq, k_env, thr_env)` on each of 5 expanding-origin folds, reports threshold variance, evaluates averaged thresholds on held-out test_internal at two fit-window choices (Mode A: 11d-equivalent fit; Mode B: full-CV fit). The `val→test_B gap` correlates monotonically with the Kaggle private gap (validated 2026-04-29 across ma_w ∈ {5k, 7k, 20k}). Use **before** any submission with `gap ≤ 0.10` as the gate. CLI: `python -m sentinel.ml_logic.cv --sweep`.
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

### 4.1 The fusion failure mode (sister pathology)

Adding a low-flag-rate stream as a 3rd OR channel sounds free — "+300 rows is only +0.058 pp flag rate, can't hurt much." It can. F0.5 counts **`FP_pred_events`** (contiguous predicted segments outside any true event), not rows.

**The math.** Window-MSE scoring produces 100-row contiguous blocks. If the 3rd stream's flags land outside true events, every isolated 100-row block is a fresh `FP_pred_event`. A hybrid sitting near `Pr_ew = 1.0` (no FP segments) loses substantially per added FP segment:

```
Pr_ew_new = TP_events / (TP_events + new_FP_segments)
```

With ~16 true events caught and 0 → 4 new FP segments, `Pr_ew` drops 1.00 → 0.80 → `Pr_c` drops 20 % → F0.5 drops ≈ 0.06–0.08. Exactly what the freq6 fusion submission cost (private 0.915 → 0.845).

**The fusion-check rule (now mandatory in `scripts/fusion_check.py`):** before any OR-fusion submission, verify on `test_internal`:
1. **`Δ TP_events ≥ Δ FP_pred_events`** (the new stream catches at least as many missed events as it adds false segments).
2. The flag-rate delta ≤ 0.2 pp on Kaggle (necessary but **not sufficient** — see above).

If rule 1 fails, do not submit, regardless of how clean the row-count looks.

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
| `pca_hybrid_envzscore_ma_w_7000_*.parquet` | Same architecture, `ma_w=7_000` (single-split, k=3) | 9 | 0.867 | 0.913 | +0.046 | **CV-harness ablation, near-tie.** Public bit-identical to champion (0.867); private −0.002 (0.913 vs 0.915). The harness flagged 7k as "fit-window-sensitive" (Mode A 0.8415 vs Mode B 0.7622 on test_internal) and Kaggle confirmed the marginal loss. Validates harness's val→test gap as a usable proxy for Kaggle private gap. |
| `fused_hybrid_OR_lstmforecast12c_freq6_*.parquet` | NB 11d hybrid OR NB 12c LSTM-Forecaster on freq 41-46 | 9 | 0.830 | 0.845 | +0.015 | **Documented loss.** OR-fusion only added 300 rows (+0.058 pp flag rate) but those rows formed 3-4 isolated 100-row blocks → +3-4 `FP_pred_events`. `Pr_ew` dropped from ~1.0 to ~0.85, F0.5 collapsed. **Lesson: count `FP_pred_events`, not rows, in fusion checks (§4 + §1).** |
| _(not submitted)_ NB 12c LSTM-Forecaster on raw `channel_14, 21, 29` | env3 raw-channels variant | 3 | — | — | — | **Drift-flood, not submitted.** Forecaster trained on early data; channels shift +5σ on Kaggle test → forecast residuals saturate everywhere → val-tuned threshold below entire Kaggle score range (0.063 vs Kaggle range [4.35, 10.87]) → 100 % flag rate. Same architecture-independent failure as NB 12 / 12b / 13 (§4). |
| _(not submitted)_ NB 12c LSTM-Forecaster on **env-residual** of `channel_14, 21, 29` | env3 drift-corrected variant | 3 | — | — | — | **No value-add, not submitted.** Drift-corrected input avoids the flood (Kaggle range [0.0001, 98] with p99=1.12, normal scale). Segment-aware gating (`fusion.fusion_diagnostics`) on test_internal at threshold ti-p99 = 3.36 says SUBMIT (+1 TP / +0 FP events). **But** at that threshold the forecaster's Kaggle positives are a strict subset of the hybrid's (`12c ∩ Kaggle ⊆ hybrid ∩ Kaggle`), so OR-fused output is bit-identical to `pca_hybrid_envzscore_BEST.parquet` (7,937 / 7,937 rows match). Submission would just re-score 0.867/0.915. |
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
2. ~~**Sweep `ENV_MA_WINDOW` and submit each.**~~ **Closed 2026-04-29.** CV harness (item 3) eliminated 10k and 20k from contention. The single remaining candidate `ma_w=7k` was submitted and scored **public 0.867 / private 0.913** — bit-identical public, −0.002 private vs champion. Direction fully exhausted: 5k is the optimum across all tested values. The harness's `val→test_B` gap (5k: 0.084 < 7k: 0.129 < 20k: 0.197) correlates monotonically with the Kaggle private loss (5k: champion < 7k: −0.002 < 20k: −0.012). Use this gap as the future submission gate.
3. **Cross-validate threshold selection on rolled splits** (multiple val windows averaged) instead of the fixed `TRAIN_END=10.7M / VAL_END=12.7M`. Reduces single-split overfitting that produced the ma_w=20k regression. **Update 2026-04-29:** built `/tmp/cv_harness.py` (5 expanding-origin folds spanning rows 2.5M–12.7M, val width 2M, held-out test [12.7M, 14.7M]) and ran it across `ENV_MA_WINDOW ∈ {3k, 5k, 7k, 10k, 20k}`. Findings:
   - **Threshold variance is large**: `thr_freq` CV ≈ 24 % across folds (mean 0.0356 ± 0.0086, identical for all ma_w because the freq stream is independent of `ma_w`); `thr_env` CV ≈ 33–62 % depending on ma_w. The single-split NB 11d threshold isn't a stable optimum — it's tuned to fold 4 (val=10.7M-12.7M), and other folds prefer noticeably different thresholds.
   - **The harness reproduces NB 11d exactly on fold 4**: thr_freq=0.0281, k_env=2, thr_env=1.237, val F0.5=0.9391 — bit-identical to the champion notebook's reported numbers.
   - **ma_w=20k is provably overfit by the harness**: highest mean val F0.5 (0.9216 ± 0.0379) but Mode B held-out test = 0.7253 → gap 0.196. The val→test divergence the harness sees mirrors the val=0.9214 / private=0.903 production failure.
   - **ma_w=5k is the most balanced**: val F0.5=0.9055 ± 0.063, Mode B held-out test = 0.8211 (gap 0.084). Mode B essentially reproduces the champion's local Test=0.8216 with averaged thresholds. Confirms 5k > 20k for generalisation.
   - **ma_w=7k is a candidate ablation**: highest Mode A held-out test (0.8415, fit on [0, 10.7M) like NB 11d, with CV-averaged thresholds). But Mode B drops to 0.7622 — high sensitivity to fit-window means it's not robust either. Worth submitting once to Kaggle as an ablation, not as champion replacement.
   - **Conclusion**: the CV harness is now the reliable diagnostic for "is this val score overfit?". Use it before any future submission. The harness is at `/tmp/cv_harness.py`; promote to `src/sentinel/ml_logic/cv.py` if it survives one more use cycle. The current champion (ma_w=5k) remains the recommended submission.
4. **Soft-OR fusion** — combined_score = `freq_norm + α · env_norm` for α ∈ {0.5, 1, 2}. Smoother than hard OR; may give better threshold transfer.
5. **Forecast-residual stream as a third OR channel** — AR(p) per channel on the env channels, score = |residual|. Different failure mode than envelope-MA, may catch the 4 ultra-short events the current hybrid misses without flooding. Non-linear variant: an LSTM forecaster (predict next window from previous) instead of AR(p) — same residual-as-score idea. **Update 2026-04-29:** the LSTM-forecaster variant has now been tested as NB 12c in two configurations (freq 41-46 → submitted, lost 0.070 private; env3 raw → drift-flood; env3 env-residual → bit-identical to hybrid on Kaggle, no value-add). The forecaster idea is **exhausted** for now on this dataset; AR(p) is still untested but inherits the same input-drift sensitivity. Lower priority than items 2/3.
6. **Shorter PCA windows (`WINDOW_SIZE=25` or `50`)** specifically for ultra-short events. Tested as 3rd-stream addition (failed locally with flag rate 4.5 %), but as a *replacement* for the freq stream's window=100 it's untested on Kaggle.
7. **Re-test LSTM-AE / CNN-AE on the drift-mitigated input.** The deep autoencoders (NB 12, NB 12b, NB 13) only ever saw the **full 58 channels** — and they all collapsed via drift-flood (Kaggle 0.078–0.238). They were never trained on the **6 stable freq channels** (the recipe that lifted PCA from 0.522 → 0.887) or on **envelope-residuals of the drifters**. Two specific configurations worth submitting:
   - LSTM-AE on `channel_41..46` only, `WINDOW_SIZE=100`, fit on tail-50 k nominal windows of Kaggle-train. Direct apples-to-apples replacement for the freq stream of the hybrid.
   - LSTM-AE on the envelope-residual sequence of `channel_14, 21, 29` (after `rolling_min(env_w=200) − centered_MA(ma_w=5_000)`). Drift-corrected input, sequence model on top. Could become a third OR stream in the hybrid.
   - CNN-AE has the same status: untested on the restricted channel set. Conditional priority — only worth doing after the LSTM-AE-on-6-channels result is in.

   Caveat: §3.2's headline finding still applies — *given the same drift-mitigation recipe, architecture is a smaller lever than recipe*. Don't expect a leap over 0.915 from architecture alone; expect at best a complementary stream that catches different events than PCA does.

Anything that **catches more events** has to be checked for flag-rate inflation on Kaggle test specifically. Local-test recall ≠ Kaggle-private recall.

**Why "just retrain LSTM-AE" is *not* in this list:** retraining the same NB 12 / NB 12b / NB 13 setup on full-58 channels is a known-failed direction — drift-flood is architecture-independent (§4). The open directions above all change the *input* (restricted channels, env-residuals, forecast targets), not just the model class.

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

---

## 10. Session log

### 2026-04-29 — Fusion safety, forecaster directions, CV harness

**Goal entering session:** push private leaderboard above the 0.915 champion via (a) a complementary OR-fusion stream and (b) more robust threshold tuning.

**Submissions made (1 ablation, 1 documented loss, 2 internal-only experiments):**

| Submission | Public | Private | Δ vs champion | Verdict |
|---|---:|---:|---:|---|
| `fused_hybrid_OR_lstmforecast12c_freq6_*.parquet` | 0.830 | 0.845 | −0.070 | **Documented loss.** OR-fused with LSTM-Forecaster on freq 41-46. Naive row-count fusion check (+0.058 pp flag rate) said "safe" but added 3-4 isolated 100-row blocks → +3-4 `FP_pred_events` → `Pr_ew` collapsed. |
| `pca_hybrid_envzscore_ma_w_7000_*.parquet` | 0.867 | 0.913 | −0.002 | **Near-miss.** Public bit-identical to champion; private just below. Validated the CV harness's `val→test_B gap` as a private-leaderboard-loss proxy. |
| _(not submitted)_ NB 12c env3 raw `channel_14/21/29` | — | — | — | Drift-flooded (val-thr below entire Kaggle range → 100% flag). Same architecture-independent failure as full-58 LSTM-AE / CNN-AE. |
| _(not submitted)_ NB 12c env3 env-residual stream | — | — | — | Drift avoided ✓; segment-aware verdict said SUBMIT at thr ti-p99=3.36 (+1 TP / +0 FP); but Kaggle output was bit-identical to champion (forecaster's positives are subset of hybrid's) → no information gain. |

**Infrastructure added:**

- **`src/sentinel/ml_logic/fusion.py`** — `fusion_diagnostics(y_base, y_new, y_true)` returns segment-counted verdict in {`reject`, `borderline`, `submit`, `submit_strong`}. Counts `FP_pred_events`, **not rows** — see §4.1 for why row-counting is misleading. Made the env-residual no-op visible before submission, prevented a redundant Kaggle slot.
- **`src/sentinel/ml_logic/cv.py`** — `run_cv(env_ma_window=...)` / `run_sweep([...])` 5-fold expanding-origin CV. Reproduces NB 11d val=0.9391 / k=2 / thr_freq=0.0281 bit-identically on fold 4. Sweep across ma_w ∈ {3k, 5k, 7k, 10k, 20k} eliminated 10k and 20k pre-submission; the surviving candidate (7k) was submitted and lost 0.002 — within harness's predicted "uncertain" band. CLI: `python -m sentinel.ml_logic.cv --sweep`.
- **`notebooks/12c-lstm_forecaster.ipynb`** — 27-cell LSTM-Forecaster pipeline with MODE toggle (freq6 / env3). Submission cell commented out by default.

**Empirical result, sorted by private:**

| Approach | Public | **Private** | val→test_B gap (CV) |
|---|---:|---:|---:|
| **Champion (ma_w=5k)** ⭐ | 0.867 | **0.915** | 0.084 |
| ma_w=7k ablation | 0.867 | 0.913 | 0.129 |
| ma_w=20k (March overfit) | 0.897 | 0.903 | 0.197 |
| freq6 fusion | 0.830 | 0.845 | n/a (fusion, not single-stream) |

**The CV harness's val→test_B gap correlates monotonically with the private loss across all submitted configs.** Use it as the submission gate going forward.

**Directions exhausted:**

- ENV_MA_WINDOW: all five values tested. 5k optimal; no further submissions in this direction.
- LSTM-Forecaster as 3rd OR stream: three input variants (freq6, env3-raw, env3-envresid) all failed. Forecast-residual approach doesn't add complementary signal on this dataset; AR(p) untested but inherits the same input-drift sensitivity. Lower priority than items 1, 4, 6, 7 in §8.

**Champion remains:** `pca_hybrid_envzscore_BEST.parquet`, public 0.867 / private 0.915, ma_w=5_000.

**Open items going into next session (ranked by expected impact, see §8):**

1. Submission ensembling — union-OR of champion with a complementary high-precision submission. Cheap to try, **gated on `cv.run_cv(...)` + `fusion.fusion_diagnostics(...)`** before submitting.
2. Cross-validate threshold selection on rolled splits — **DONE** via `cv.py`. Use it.
4. Soft-OR fusion `α · freq + β · env` — untested.
6. Shorter PCA windows (25/50) as freq-stream replacement — untested on Kaggle.
7. LSTM-AE / CNN-AE on drift-mitigated input — untested.

---

## 11. Final Evaluation (FE) pipeline

Distinct from Kaggle. The FE demo scores a 300 k-row internal slice through the API and a notebook walkthrough. The model and recipe come from **NB 11e** with `PRUNE_AGGRESSIVE=True` (46-channel variant), not from the Kaggle champion (NB 11d hybrid).

### 11.1 Why a different model than Kaggle

NB 11e produces two artefacts (see its title cell):
- **49 channels** (`PRUNE_AGGRESSIVE=False`) — Kaggle internal-test ESA F0.5 0.853, beats 46ch on the public leaderboard.
- **46 channels** (`PRUNE_AGGRESSIVE=True`) — Kaggle internal-test ESA F0.5 0.810 but **0.818 on the FE internal-test slice (post-filter)** vs 0.716 / 0.753 for ESA-baseline / 49ch / 58ch under the same recipe. **46ch wins on the FE proxy and is what the demo shows.** This is recorded in memory `project_nb11d_variants` and is the reason FE diverges from Kaggle.

### 11.2 Blessed FE artefacts

| File | Contents |
|---|---|
| `models/pca_fe_46ch.pkl` | Trained PCA (46 channels × 100 rows = 4 600-dim windows, 316 components, EV 0.950) |
| `models/scaler_fe.pkl` | StandardScaler fit on the same tail-50 k nominal slice |
| `models/fe_46ch_20260429_153751.json` | Sidecar — threshold, post-filter, fit config, val/test metrics, channel list |
| `data/raw/target_channels_fe.csv` | The 46 FE channels (76 total − Tier A drifters 64-66/70-76 − Tier B 16/24/32) |
| `data/processed/test_api_fe.npy` + `y_test_api_fe.npy` | 300 000-row demo slice (rows 14 175 000 – 14 475 000 of train.parquet, 6 events, 47.6 % anomaly density) |

Sidecar key values (frozen at train time, do not retune in the demo):

| Field | Value |
|---|---|
| `threshold` | **0.001 970** (tuned via `tune_threshold(corrected_event_f05)` on val 10.7 M – 12.7 M) |
| `val_esa_f05` | 0.8534 |
| `input_detrend` | rolling **median**, window 100 000 rows (per-channel input detrend before scaling/PCA) |
| `score_detrend` | rolling **median**, window 1 000 windows (= 100 000 rows) on the row-broadcast scores |
| `post_filter` | `clean_predictions(min_len=100, max_gap=500)` — gap-fill ≤ 500-row gaps, then drop blocks shorter than 100 rows |
| `fit` | `tail_50k_NB11e`, `train_end=10 700 000`, `random_state=42` |

Internal-test (12.7 M – 14.7 M) ESA F0.5: **0.717 baseline → 0.818 with post-filter** (events 14/25, TNR 0.991, flag rate 4.2 %, predicted blocks 36 → 19). The post-filter is the single biggest lift on this slice — `min_len=100` removes spurious singleton windows; `max_gap=500` merges fragmented event-coverage into one segment so the same physical event isn't double-counted as 2-3 false segments.

### 11.3 Inference path

`src/sentinel/ml_logic/predictor.py::predict_fe46(X_raw, model, scaler, features, threshold, …)` — applies the NB 11e recipe (per-channel input detrend, scale, window-MSE, score-level detrend, threshold, post-filter) and returns `{id, is_anomaly}` rows. `predict()` and `predict_report()` (the API entrypoints in [api/fast.py](api/fast.py)) wrap the same recipe with the cached sidecar threshold.

### 11.4 Notebooks for the demo

| Notebook | Role |
|---|---|
| [11e-pca_detrended.ipynb](notebooks/11e-pca_detrended.ipynb) | **Source of `pca_fe_46ch.pkl`.** Run with `PRUNE_AGGRESSIVE=True` to reproduce. |
| [14b-model_eval_fe.ipynb](notebooks/14b-model_eval_fe.ipynb) | Mirror of NB 14, but evaluates the FE model on internal val/test (10.7 M – 14.7 M). Reproduces the sidecar metrics. |
| [23b-test_slices_fe.ipynb](notebooks/23b-test_slices_fe.ipynb) | Slice picker — chose rows 14.175 M – 14.475 M (rank-1 candidate, 6 events, ~47.6 % density) as the FE demo window. Saves `test_api_fe.npy` / `y_test_api_fe.npy`. |
| [32-api.ipynb](notebooks/32-api.ipynb) | API showcase — loads the FE artefacts, runs inference over `test_api_fe.npy`, balanced-transition preview around the on/off boundaries, end-to-end ESA F0.5 0.665 / recall 5/6 on the demo slice. |

### 11.5 API integration

[api/fast.py](api/fast.py) caches one prediction over `test_api_fe.npy` at startup using `predict_report(..., n_top_channels=6)`. Cached endpoints:

- `GET /timeline` — `[{id, is_anomaly}]` for all 300 k rows.
- `GET /predict_by_id?start=..&end=..` — filter the timeline.
- `GET /report` — row-MSE, per-channel MSE, top contributing channels per window, threshold, anomaly rate.
- `POST /predict` — score user-supplied rows (must be 46 channels in the order of `target_channels_fe.csv`).

### 11.6 Verification

`scripts/verify_fe46.py` reloads the pickle + sidecar and re-runs internal val/test. Sidecar metrics (val 0.8534, test 0.717 baseline / 0.818 post-filter) should reproduce bit-identically. Run before any FE re-demo.
