# SENTINEL — Project Results & Knowledge Base

ESA Anomaly Detection Benchmark (ESA-ADB) / Kaggle competition.
This document is the **single source of truth** for what was tried, what worked,
what failed, and why. If you only read one file to catch up on the project,
read this one.

**Last updated:** 2026-04-24 · **Current champion:** NB 04 PCA — Kaggle public `0.522` / private `0.599`.

---

## 0. Executive summary — read this first

### What we set out to do
Build an unsupervised anomaly detector for 58 spacecraft telemetry channels
(~14.7 M rows training, 521 k rows test). The Kaggle metric is **ESA-corrected
event-wise F0.5** (`Pr_ew × TNR`, recall weighted ½). Ground truth is a
per-row 0/1 anomaly label; the task rewards **detecting the event at all with
few predicted segments** much more than pinpointing its exact row extent.

### What was tried (chronological, condensed)

| Phase | Notebooks | Idea | Outcome |
|---|---|---|---|
| 1. EDA | `01`, `02`, `Alex - EDA *` | Characterise the data, label noise, class balance, KS drift between train/test. | 10.5 % anomaly rows, 190 events, **mean KS(train, test) = 0.43** — large distribution shift baked into the task. |
| 2. Preprocessing | `02`, `preprocessor.py` | 58-channel RobustScaler (fit on nominal-only), 100-row non-overlapping windows, 70/15/15 chronological split that does not straddle events. | `X_train_nom`, `val_scaled`, `test_intern_scaled` arrays — the common substrate for every downstream notebook. |
| 3. Row-level baseline | `03-iforest` | Isolation Forest on scaled rows. | **Failure** — threshold drifts, 2/38 events hit, Kaggle ≈ 0. *Useful counter-example*: per-row scoring is incompatible with event-wise metrics on a drifty signal. |
| 4. Reconstruction baseline | `04-baseline_pca` | PCA on flattened (100 × 58) windows → window-mean MSE → threshold tune on ESA-corrected F0.5 (Kaggle split 80/20). | **Current champion.** Val `0.770`, Kaggle public `0.522` / private `0.599`. |
| 5. Short-window ablation | `04b-pca_win50` | Same as NB 04, WIN=50 to lift short events. | Run for comparison; does not beat NB 04's transferability. |
| 6. Deep-learning attempts | `12-lstm_ae`, `13-cnn_ae` | BiLSTM-AE and 1D CNN-AE, same window-MSE scoring scheme. | LSTM-AE ties PCA on `test_intern` headline — but that tie is a drift artefact (see §8). CNN-AE does not converge to a useful score. |
| 7. Evaluation notebook | `11-pca`, `14-model_eval` | Clean 70/15/15 eval for any trained model, confusion matrix, multi-metric panel, timeline plots. | Reveals the **drift-flood** failure mode. Both PCA and LSTM score 0.98 Event F0.5 on `test_intern` by flagging the whole post-drift half as one giant block. |
| 8. Drift diagnosis | `18-level_shift_1`, `19-level_shift_2` | KDE per split, rolling-mean timelines, per-channel drift heatmap. | **10 channels** (`29, 14, 21, 30, 38, 22, 31, 39, 15, 23`) move in lock-step +4.5–6.7 σ between train-start and Kaggle test. Not a time-of-mission effect — a subsystem regime shift. |
| 9. Kaggle-specialised PCA | `11b-pca` | Fit PCA on the **last 50 k nominal windows** of full train (the drift regime), score detrended, tune on ESA F0.5. | k=17 components. Works on the post-drift regime but the public-side earlier Kaggle rows fall outside that regime. Not better than NB 04 on Kaggle. |
| 10. Kaggle-specialised LSTM | `12b-lstm_ae` | NB 12 recipe + **per-window z-norm** (drift-invariant scoring) + threshold candidates A–J. | Z-norm collapses the signal: val nominal/anomaly means 1.055 vs 1.060. Best ESA F0.5 `0.33` on val, `0.23` on test_intern. Retired as a Kaggle contender. |
| 11. Score-level detrending | `20-detrending` | Rolling-**median** and **z-score** detrending of row-scores, evaluated on a labeled proxy for Kaggle test (last 521 k rows of train). | **PCA + median-detrend wins** on the proxy: ESA F0.5 `0.874` vs `0.000` baseline. Ties NB 04 on Kaggle private (`0.599`) but loses `0.068` on public. |

### The five insights that matter most

1. **Reconstruction score, not per-row score.** A model that produces *one MSE per 100-row window* (broadcast to rows) gives a smooth curve. A model that produces *per-row scores* (IForest, row-level MLPs) gives spikes — the threshold sweep finds no gap, and FP-predicted-segments explode. This is the single most important design choice in the whole project. See §5.
2. **`FP_pred_events` is the variable that decides whether a model works.** It is the direct denominator of event-wise precision. `FP_pred_events = 0` means `Pr_ew = 1.0` and the threshold sweep is free to raise recall. Any technique that produces compact predicted segments (window-MSE, detrending, score smoothing) inherits this property.
3. **The headline `0.98 Event F0.5` on `test_intern` is a lie.** It rewards "flag one wide block that happens to contain all the late events" identically to "precise event-by-event detection". The honest read is **ESA-corrected F0.5** (which includes TNR) or **Row F1** — both drop to ~0.22–0.47 for the same models. Event F0.5 without the corrected column is a pitfall, not a metric.
4. **Train/test distribution shift dominates model choice.** PCA, LSTM-AE, CNN-AE all hit the same drift ceiling on `test_intern`. This is a **scoring-side problem**, not a model-capacity problem. Detrending (subtracting a rolling-median baseline from the score) improves *all* models. The drift lives in a 10-channel subsystem (NB 18/19) — worth knowing but not necessary to treat per-channel; a score-level detrender captures the effect.
5. **Public-private Kaggle gap > 0.1 is a temporal-clustering red flag.** Submissions whose flagged rows concentrate in one half of the test timeline score well on one split and zero on the other. This is the row-level version of the Event-F0.5 pathology in insight 3. NB 04 has consistent public `0.522` / private `0.599` (+0.077) — that's a *healthy* gap. CNN-AE v3 had public `0.000` / private `0.238`, `baseline_ensemble` public `0.522` / private `0.476` — those are the pathological signature.

### How to use these findings

- **For the presentation / write-up**: lead with NB 04 (PCA baseline) as the champion. Frame the deep-learning models (NB 12/13) as ablations that confirm architecture is *not* the lever — the lever is the scoring recipe (window MSE + detrend + corrected-F0.5 tuning). The drift-analysis notebooks (NB 18/19/20) are the *explanation* of why DL doesn't beat PCA, not a separate thread.
- **For future modelling**: any new model needs to slot into `score_windows(...)` (i.e. produce a reconstruction). Evaluate on the Kaggle-split (`data/processed/kaggle/`) using NB 11b's recipe (last-50k tail fit, ESA F0.5 tuning, median detrend). Tune threshold on **`corrected_event_f05`**, not `event_f05` — see §4.
- **For threshold tuning**: default recipe is `tune_threshold(val_scores, y_val, metric_fn=corrected_event_f05, n_sweep=60)` with log-spaced candidates. If val has a different drift regime than test, fall back to the "drift-robust candidate sweep" (NB 12b Section 8b) — generate candidates from val-percentile, test-percentile, and Kaggle-percentile of the score, pick the one with the best proxy-test ESA F0.5 and a reasonable Kaggle flag rate (3–10 %).
- **What NOT to do**: don't submit anything that flags > 10 % of test rows, or whose predicted flags are contiguous for > 50 k rows — both are the drift-flood failure mode. Don't tune on `event_f05` alone — use `corrected_event_f05` so over-flagging is penalised by TNR.

### What is currently "our best model"

**`NB 04 — baseline_pca`** is the champion Kaggle submission. The artefact on disk is `kaggle/submissions/nb04_pca_50k_599.parquet`. **Do not modify NB 04** — any experiment variant goes into a **new** notebook (`04x-*.ipynb`) with its own save path. The blessed model pickles referenced by `src/sentinel/params.py` are `models/pca.pkl`, `models/pca_kaggle.pkl`, `models/lstm_ae.keras`, `models/cnn_ae.keras`, `models/scaler.pkl` — those five files are protected; all experimental saves go to `{model}_nb{XX}_{ts}.(pkl|keras)` as per the naming convention.

---

## 1. The problem in one page

**Task.** Given 58 channels of spacecraft telemetry (76 sensor channels subset to 58 *target* channels per the competition, plus 11 telecommand columns), flag rows that are part of an anomaly event.

**Data shape.**

| | Rows | Events | Anomaly rate | Labels |
|---|---:|---:|---:|:-:|
| train.parquet | 14,728,321 | 190 | 10.48 % | yes |
| test.parquet | 521,280 | — | — | **no** (Kaggle) |
| internal val (NB 02) | 2,232,277 | 38 | 11.0 % | yes |
| internal test (NB 02) | 2,186,220 | 27 | 9.8 % | yes |
| Kaggle-split internal val (NB 11b/12b) | 2,000,000 | 28 | 10.8 % | yes |
| Kaggle-split internal test (NB 11b/12b) | 2,028,321 | ~30 | 10.6 % | yes |

**Event lengths.** Median 602 samples, longest 116,061 samples, shortest 1 sample. Wide range ⇒ fixed-window scoring can dilute short events (100-row window vs 30-row event → 30 % weight), motivating NB 04b.

**Distribution shift (drift).** Mean KS(train, test) = 0.43, max 0.97 (channel_15). NB 18/19 isolate the drift to a 10-channel subsystem (channels 14, 15, 21, 22, 23, 29, 30, 31, 38, 39) that shifts by ~+4.5–6.7 σ between train-start and Kaggle test. 9 channels (`42, 64, 65, 70–75`) are stationary — useful as a drift-free reference. This drift is the **single biggest source of train/test metric divergence** in the whole project.

**Metric.** Kaggle scores with **ESA-corrected event-wise F0.5** (`Pr_c = Pr_ew · TNR`, `F = 1.25 · Pr_c · Re / (0.25 · Pr_c + Re)`). See §4 for why `Pr_ew` ≠ row precision and why `TNR` matters.

---

## 2. Infrastructure — the shared `sentinel.ml_logic` modules

Every notebook imports from a shared package so that cross-notebook comparisons are apples-to-apples.

### 2.1 `scorer.py` — one reconstruction, five views

Core API: `score_windows(model, X_rows, win=100, topk=None)` — reshape rows into non-overlapping windows, call `model.inverse_transform(model.transform(X))` (PCA) or `model.predict(X)` (Keras), reduce squared error to one scalar per window, broadcast to rows. Trailing rows inherit the last window's score.

**Reduction choice (`topk` argument):**
- `topk=None` → mean over all 100 × 58 squared errors. **Use for PCA.** Linear reconstruction spreads error across all channels; the mean is a reasonable aggregate.
- `topk=k` → mean of the `k` largest *per-channel* MSEs (per-channel mean over time first, then pick the top-k across channels). **Use `topk=5` for LSTM-AE and CNN-AE.** Non-linear AEs localise error in a few channels; the mean-over-58 dilutes the signal. Emperical: without topk=5 the LSTM's Event F0.5 drops from 0.976 to near-baseline.

**Score-level detrending** (`detrend_scores` / `score_windows_detrended`):
- `mode="median"`: `score - rolling_median(score, window=100_000)`. Kills slow mean drift. The **first and usually only thing to try**.
- `mode="zscore"`: `(score - rolling_median) / (rolling_MAD + eps)`. Also normalises variance drift. Use only if median alone doesn't fix the variance widening.
- Backed by `scipy.ndimage.median_filter` (C implementation, bounded memory on ~15 M-row arrays).
- **Per split, not on a concatenation** — detrending across a split boundary leaks the baseline estimate.

**`score_report`** returns `{row_scores, window_scores, per_channel_mse, window_channel_mse, topk_channels}` in one pass. Used by NB 14 and the front-end "Confusion matrix + top channels" panel.

### 2.2 `metrics.py` — the metric zoo

Nine metric functions, all returning dicts so callers can destructure:

| Function | What | When |
|---|---|---|
| `event_f05` | Event-wise F0.5, **no TNR correction**. `Pr_ew = TP_events / (TP_events + FP_pred_events)`. | Primary tuning metric in NB 11/12 (the "bootcamp" metric). Easy to optimise *too* well — any over-flagging that merges predictions into one event passes `Pr_ew = 1`. |
| `event_f1` / `event_f2` | Same but β=1 / β=2. | Secondary columns in NB 14 panel. |
| `corrected_event_f05` (alias `esa_metric`) | ESA / Kaggle metric: `Pr_c = Pr_ew · TNR`, `F_0.5(Pr_c, Re_e)`. | **The actual Kaggle objective.** Used in NB 04 / NB 11b / NB 12b / NB 13 for threshold tuning. |
| `event_detection_rate` | `TP_events / N_events`. | Jury-friendly "caught X of Y events" number. |
| `point_adjust_f1` | Standard Xu-2018 point-adjust F1: if any predicted row lands inside a true event, the *whole* true event is marked predicted. | Reference metric from the TS-AD literature. Permissive — do not use for tuning. |
| `row_precision_recall` | Per-row TP/FP/FN → P, R, F1. | Row F1 is the honest row-granularity view of a model. |
| `compute_all_metrics` | Flat dict of everything above. | NB 14 multi-metric panel. |

**Why both `event_f05` and `corrected_event_f05` exist:** NB 11/12 were originally tuned on `event_f05` (the bootcamp metric). Once the drift-flood failure mode surfaced (see §8), all Kaggle-facing notebooks (04, 11b, 12b, 13) switched to `corrected_event_f05` because it penalises the flood through TNR. Keeping both keeps us honest: when `event_f05 ≫ corrected_event_f05`, the delta *is* the over-flagging.

### 2.3 `thresholds.py` — `tune_threshold`

Sweep `n_sweep` (default 60) log-spaced thresholds between the 50th percentile of nominal scores and the 99th percentile of anomaly scores, evaluate `metric_fn` on each, return the argmax. Log-spacing keeps the grid dense in the transition region where distributions differ most.

```python
tune = tune_threshold(val_scores, y_val, metric_fn=corrected_event_f05, n_sweep=60)
t_best = tune['threshold']    # use on test
```

Generic over the metric — pass `event_f05`, `corrected_event_f05`, `event_f2`, or any custom metric that returns `{'f_score': ...}`. Pass `score_key=None` for a bare-scalar metric.

### 2.4 `validation.py` — event-block bootstrap

`bootstrap_f05_ci(y_true, y_pred, metric_fn, n_boot=200, event_block=True, seed=42)` — resamples whole anomaly events with replacement (not individual rows), drops predictions inside dropped events, re-scores. Gives a CI that reflects event-presence variance rather than spurious row-level noise.

`n_boot=200` is the project default; runtime scales linearly (typical: 570–610 s for `event_f05`).

### 2.5 `viz.py` — shared plotting

- `plot_timeline(scores, y_true, threshold, title, log_y=True)` — two-panel val/test timeline with anomaly bands. **The standard visualisation** used in NB 04 / 04b / 11 / 11b / 12 / 12b / 13 / 14.
- `plot_confusion_and_channel_errors(...)` — the NB 15 / NB 14 confusion matrix + top-channels panel.

All notebooks use the same `ANOMALY_COLOR = '#e74c3c'`, `NOMINAL_COLOR = '#2980b9'`, seaborn theme.

### 2.6 `params.py` — project-wide constants

| Constant | Value | Meaning |
|---|---|---|
| `WINDOW_SIZE` | 100 | rows per non-overlapping window |
| `FIT_SIZE` | 50,000 | #windows used for model fitting (None = all) |
| `RANDOM_STATE` | 42 | seed everywhere |
| `TRAIN_RATIO` | 0.80 | temporal split for NB 04 |
| `BOOTCAMP_TRAIN_RATIO` / `_VAL_RATIO` | 0.70 / 0.15 | three-way split for NB 11/12/13/14 |
| `PCA_THRESHOLD` | 0.060404 | NB 11 val-tuned |
| `LSTM_THRESHOLD` | 1.323612 | NB 12 val-tuned |
| `PCA_DETRENDED_THRESHOLD` / `LSTM_DETRENDED_THRESHOLD` | None | to be filled after NB 20 run (not reused from baseline because detrending shifts the distribution) |

---

## 3. Scoring strategy — the decision that matters most

### Window-MSE vs per-row MSE

The single most important design choice in this project. The output of every reconstruction model is per-element squared error of shape `(n_windows, 100, 58)`. We reduce to **one score per window** (mean over 100 × 58 or top-5-channel mean), then broadcast each window's score to its 100 rows.

**Why not per-row?** Per-row MSE gives a spiky score curve. The threshold sweep then finds either "high enough to miss events" or "low enough to produce hundreds of spurious segments". `FP_pred_events` blows up, `Pr_ew` collapses.

**Ablation from NB 04 Section 7:**

| Variant | Val F0.5 | Kaggle public |
|---|---:|---:|
| Window-mean MSE (A) | 0.770 | 0.522 |
| Per-row MSE (B) | 0.698 | **0.277** |

Per-row tuned threshold transfers catastrophically. Window-MSE has enough smoothing to survive distribution drift. This is the foundation of every working model in the project.

### Top-k channel reduction

Anomalies in this dataset concentrate in a few channels. For AEs (which can localise error), taking the **mean of the top-5 per-channel MSEs** amplifies the signal.

| Model | `topk=None` Event F0.5 | `topk=5` Event F0.5 |
|---|---:|---:|
| PCA | **0.9843** | 0.97 (minor drop) |
| LSTM-AE v4 | 0.094 | **0.9756** |
| CNN-AE | 0.082 (recall 0.630) | 0.078 (recall 0.222) |

**Architectural, not hyperparameter.** PCA spreads error across every component's basis vector, so it already *is* a top-k view in the spectral sense. LSTM-AE concentrates error in 2–5 channels per anomalous window — top-k is a perfect match. CNN-AE spreads error uniformly across channels (convolutional smoothing) and *loses* recall under top-k.

### Score-level detrending

Slow baseline drift pushes the test MSE above a val-tuned threshold long before any true anomaly occurs (see NB 14 timeline plots). Two mitigations:

- **Input-level z-norm** (per-window, per-channel, before the model sees the data). Used by NB 12b and NB 13 via `ZNormAdapter`. Clean algebra: `(X − adapter_output)² = (X − (X − Xn + Xhat_n))² = (Xn − Xhat_n)²` — MSE is computed in z-space without touching `score_windows`. Trade-off: kills sustained single-level anomalies (rare in this dataset).
- **Score-level median detrend** (subtract a rolling median from the row-score). Used by NB 11b and NB 20. **This is the lever that turned the NB 20 experiment around**: PCA baseline proxy ESA F0.5 was `0.000` (the baseline drifts through the threshold), PCA-median proxy ESA F0.5 was **`0.874`**.

Empirically median detrend beats z-norm for this task (cf. NB 20 table in §9).

---

## 4. Metric selection — why `corrected_event_f05`, not `event_f05`

### The pathology

`event_f05` counts a prediction as a true positive *at event-level*: one row overlap ⇒ the whole event is "detected". Merged predictions = high precision + high recall = high F0.5 = false signal of a working model.

**Concrete example on `test_intern`.** 27 true events. Predict `y_pred[row >= 900_000] = 1`, everything else 0. This is one big predicted segment that overlaps every post-row-900k event. Result: `TP_events = all-late-events`, `FP_pred_events = 0`, `Pr_ew = 1.000`, `event_f05 ≈ 0.98`. You've just "solved" the task by hard-coding a row index.

### The correction

ESA-ADB's `corrected_event_f05` multiplies `Pr_ew` by **TNR** (true-negative rate over *rows*). Flagging 1.2 M nominal rows to cover one event crashes TNR from 1.0 to ~0.5 and `Pr_c` from 1.0 to 0.5 even if `Pr_ew = 1.000`. This is the Kaggle metric. **Tune on this.**

| Model | `event_f05` (`test_intern`) | `corrected_event_f05` | Row F1 |
|---|---:|---:|---:|
| NB 11 PCA | **0.9843** | 0.4736 | 0.2281 |
| NB 12 LSTM-AE v4 | **0.9756** | 0.4715 | 0.2185 |

The 2× drop is the drift flood. Row F1 ≈ 0.22 is the *honest* answer to "how precisely does this model flag anomaly rows?"

### Practical guidance

- **Final tuning**: always `corrected_event_f05`.
- **Diagnostic**: report `event_f05` *and* `corrected_event_f05` side-by-side. The delta is the flood magnitude.
- **For jury / stakeholders**: lead with `event_detection_rate` ("caught 25 / 27 events") — it's what people actually care about — plus `Row F1` for row-granularity honesty.
- **In the ensemble / submission stage**: watch `fp_pred_events`. `fp_pred_events = 0` with a plausible flag rate (3–10 %) is a submittable model. `fp_pred_events = 0` with flag rate ≥ 20 % is the flood.

---

## 5. Threshold tuning strategies — beyond the default sweep

### Default: log-sweep on val

`tune_threshold(val_scores, y_val, metric_fn=corrected_event_f05)` — 60 log-spaced candidates, pick argmax. Works when val and test come from the same distribution.

### When val-tuning fails

On a drifty test split (Kaggle), val-tuned thresholds can land *below* the test baseline, flagging everything. Diagnosis: flag rate > 20 %, `Pr_ew = 1.000`, `TNR ≪ 1`.

NB 12b Section 8b explored a **drift-robust candidate sweep** with 10 alternatives:

| Candidate | Rule | Used in NB 12b | Outcome |
|---|---|---|---|
| A | `tune_threshold(val)` on ESA F0.5 | baseline | test ESA `0.234` |
| B–D | 99, 99.5, 99.9 percentile of val *nominal* scores | principled upper bound | best: `C_val_nom_p99_9` → test ESA **`0.310`** |
| E–G | 90, 95, 98 percentile of `test_internal` score distribution | lets test's own distribution pick the threshold | over-flags (10 % pos rate at p90) |
| H–J | 90, 95, 98 percentile of Kaggle score distribution | purely unsupervised prior-matching | H over-flags, J reasonable |

**Rule of thumb.** Pick the candidate that maximises proxy-test ESA F0.5 *and* has Kaggle flag rate between 0.5 % and 10 %. Reject anything outside that band as a drift-induced flood.

### Threshold-transfer check

Always print **both** val and test score ranges before tuning. If `max(test) < threshold_candidate < max(val)` and `max(test) > min(val)`, the threshold will flag zero test rows (NB 03 / IForest pathology). If `min(test) > threshold_candidate`, it will flag all test rows.

---

## 6. The splits — which notebook uses which

Two preprocessing schemes coexist, each with its own `data/processed/` subdirectory:

1. **BOOTCAMP split** (`data/processed/`): 70 / 15 / 15 chronological, anomaly-aware boundary snapping. Used by NB 11 / NB 12 / NB 13 (original recipe) / NB 14 / NB 18 / NB 19.
2. **Kaggle split** (`data/processed/kaggle/`): everything of `train.parquet` kept for training; `test.parquet` is scored separately. NB 11b / NB 12b / NB 20 carve an internal val/test *from inside the train arrays* at row indices `TRAIN_END=10_700_000`, `VAL_END=12_700_000`, snapping each boundary so a ±2 000-row neighbourhood is fully nominal.

**Why two?** The BOOTCAMP split is the apples-to-apples model comparison. The Kaggle split puts training and scoring on the drift regime Kaggle test actually lives in. Reported `test_intern` F0.5 on the BOOTCAMP split is **systematically optimistic** for Kaggle — NB 18 Section 3 KDE plots show why: train + BOOTCAMP test_intern curves sit together, Kaggle test is shifted further.

---

## 7. Per-notebook results

**Conventions:**
- All F0.5 values are on the internal test of the respective split (`test_intern` for BOOTCAMP, `test_internal` for Kaggle split) unless marked "Kaggle".
- Wall times are the reported figure from the executed notebook; GPU times were measured on an M-series Mac GPU via Metal.
- All file paths are relative to the project root.

### 7.1 `NB 03 — Isolation Forest` (failed baseline, kept for contrast)

| Setting | Value |
|---|---|
| Model | `IsolationForest(n_estimators=200, max_samples=256)` |
| Training set | 500 k nominal rows (subsampled from 10.5 M), 58 RobustScaled channels |
| Val F0.5 | **0.091** @ threshold 0.6282 |
| Val events detected | 2 / 38 |
| Test rows flagged | 0 / 521,280 (full threshold drift) |

**Why it fails:** per-row scoring → 2,203 predicted segments at best threshold → `Pr_ew = 2/18 = 0.111`. Score distribution shifts `[0.518, 0.614]` between val and test so 0 test rows exceed the val-optimal threshold. *Any model producing independent per-row scores hits the same wall.*

### 7.2 `NB 04 — baseline_pca` ⭐ **champion Kaggle submission**

| Setting | Value |
|---|---|
| Split | Kaggle 80 / 20 (`SPLIT_IDX=11_782_656`) |
| Model | `PCA(n_components=0.95)` on flattened (100 × 58) → 5,800 windows |
| k (components) | **39** (cumulative EV 0.95) |
| Threshold tuning metric | `corrected_event_f05` |
| Best threshold | `0.110` |
| Val F0.5 | **0.770** |
| Val events detected | 21 / 38 |
| Val `FP_pred_events` | **0** |
| Val `Pr_ew` | 1.000 |
| Val TNR | 0.855 |
| Test rows flagged | 33,100 / 521,280 (**6.35 %**) — plausible rate |
| **Kaggle public F0.5** | **0.522** |
| **Kaggle private F0.5** | **0.599** |
| Submission | `kaggle/submissions/nb04_pca_50k_599.parquet` |

**Why it's the champion.** The combination window-MSE + PCA's mean-over-all-errors + `corrected_event_f05` tuning produces a smooth score, compact predicted segments, zero FP predicted events at the best threshold, and a plausible 6.35 % test flag rate. The narrow test-score range means the val-tuned threshold transfers to Kaggle — the only model in the project where this holds.

**Do not modify NB 04.** Experiments go in `04x-*` copies. The PCA pickle is `models/pca.pkl` (blessed).

### 7.3 `NB 04b — pca_win50` (short-window fork)

| Setting | Value |
|---|---|
| Change vs NB 04 | `WIN = 50` (doubles weight of short events; 30-row anomaly → 60 % weight in a 50-row window) |
| Aggregation | flat MSE over 2 900 residuals |
| `n_components` | `k_95` ≈ 20–25 |

**Status:** comparison notebook. Submits only if val F0.5 > 0.770, `fp_pred_events=0` at best threshold, and test pos_rate in 3–10 %. Best artefact currently keeps pace with NB 04 but does not beat it on Kaggle.

### 7.4 `NB 11 — PCA` (BOOTCAMP split — diagnostic only)

| Setting | Value |
|---|---|
| Split | BOOTCAMP 70 / 15 / 15 |
| Model | `PCA(n_components=0.95)` on 50,000 subsampled nominal windows |
| k | **38** (cumulative EV 0.9501) |
| Wall time | 59.6 s CPU |
| Threshold tuning metric | `event_f05` *(legacy — predates the metric-switch)* |
| Best threshold | 0.060404 |
| Val Event F0.5 | **0.8333** |
| Test Event F0.5 | **0.9843** — but see §8 |
| Test `corrected_event_f05` | 0.4736 |
| Test Row F1 | 0.2281 |
| Bootstrap (200×, event-block) | mean 0.8053 · 95 % CI **[0.6730, 0.9501]** · 573.7 s |

**Headline trap.** The 0.9843 on `test_intern` is the drift-flood artefact. Corrected F0.5 0.474 and Row F1 0.228 are the real numbers. This notebook is the primary **diagnostic baseline** for the drift-flood pathology — see §8.

### 7.5 `NB 11b — PCA Kaggle-specialised`

| Setting | Value |
|---|---|
| Split | Kaggle split (train=[:10.7 M], internal val=[10.7–12.7 M], internal test=[12.7–14.7 M]) |
| Model | `PCA(n_components=0.95)` on **last 50 000 nominal windows of full train** (row-coverage [9.07 M, 14.73 M] — inside the drift regime) |
| k | **17** (cumulative EV 0.9503) |
| Wall time | 76.2 s CPU |
| Score variants | baseline + rolling-median detrend (window=1 000 windows ≈ 100 000 rows) |
| Threshold tuning metric | `corrected_event_f05` |
| Val score range (baseline) | `[0.0180, 1180.5267]` |
| Internal-test score range (baseline) | `[0.0176, 1.8731]` — 650× narrower, drift regime |

**Why fit on the tail, not random subsample?** Random subsample averages pre-shift and post-shift regimes → PCA learns a "blurred" basis that reconstructs neither well. Fit-on-tail learns the regime Kaggle test actually lives in.

**Why `FIT_SIZE=50_000`?** An earlier `FIT_SIZE=20_000` version produced only 2 predicted segments / 600 rows on Kaggle test → public 0.000. The tail slice was *too* narrow. 50k is the empirical sweet spot.

**Status:** detrended variant is competitive on internal test but does not beat NB 04 on the Kaggle leaderboard — the Kaggle public-side rows sit in an earlier regime that the tail-fit model does not cover.

### 7.6 `NB 12 — LSTM Autoencoder`

**v4 (latest, Option B)** — competitive with PCA on `test_intern`:

| Setting | Value |
|---|---|
| Architecture | BiLSTM(128) → LSTM(64) → LSTM(8) latent → RepeatVector → LSTM(64) → LSTM(128) → TimeDistributed Dense(58) |
| Parameters | 400,986 (1.53 MB), `LATENT_DIM=8`, `HIDDEN_DIM=128`, `DROPOUT=0.1` |
| Input | **no per-window z-norm** — raw RobustScaled windows as `float32` |
| Fit | **last 50,000 of 92,271 nominal windows** (`X_train_nom[-FIT_SIZE:]`), Adam(1e-3), MSE, 10 % val |
| Training | 6 epochs, 8.3 min GPU, EarlyStopping restored **epoch-1 weights** |
| Scoring | `score_windows(model, topk=5)` |
| Best threshold | 1.323612 (val `event_f05` tuned) |
| Val Event F0.5 | 0.7143 |
| Test Event F0.5 | **0.9756** (24/27 events, recall 0.8889, precision 1.0000) |
| Test `corrected_event_f05` | 0.4715 |
| Test Row F1 | 0.2185 |
| Bootstrap (same-arch prior run) | mean 0.7929 · 95 % CI [0.6500, 0.9475] |
| Score drift | val `[0.4346, 7735.26]` vs test `[0.6992, 60.44]` — **128× max-ratio** |

**Why it works:** three levers stack. (1) `FIT_SIZE=50 000` *last* windows → model sees the same regime as val/test; (2) EarlyStopping kept epoch-1 weights — the model is acting as a stable residualiser, not a learned nominal generator; (3) `topk=5` concentrates the signal on the channels that actually differ.

**Caveat:** same drift-flood pathology as NB 11 (test_intern Event F0.5 is inflated). The 128× val/test max-ratio means the threshold 1.324 is working by *luck* — it lands above the nominal mode and below the anomaly peaks in both distributions, but a small upward shift would produce zero test flags.

**v3 regression** (retired): all 92 k windows + per-window z-norm + 44 epochs → val loss plateau at 0.78, model became a mean predictor, Event F0.5 `0.094`.

**v2** (retired): smaller LSTM, 25 epochs, Event F0.5 `0.265`.

**v1 Kaggle submission**: `baseline_lstm_ae.parquet` — 17.9 % positive rate, public `0.112` / private `0.078`. Over-flag.

### 7.7 `NB 12b — LSTM-AE Kaggle-specialised` (z-norm retry)

| Setting | Value |
|---|---|
| Split | Kaggle split |
| Architecture | identical to NB 12 v4 (≈ 401 k params) |
| Input | **per-window z-norm** via `ZNormAdapter` — drift-invariant MSE in z-space |
| Fit | last 50 000 nominal windows **before `TRAIN_END`** |
| Training | 20 epochs, 28 min GPU |
| Val score range | `[1.0029, 2.2206]` (nominal mean 1.055, anomaly mean 1.060 — **tiny separation**) |
| Test score range | `[1.0033, 1.6870]` (nominal mean 1.055, anomaly mean 1.058) |
| Candidate A (val-ESA tuned, t=1.1925) | test event F0.5 **0.2347**, ESA 0.2339, TNR 0.9964, flag rate 0.48 % |
| Candidate C (val nominal p99.9, t=1.3149) | test event F0.5 **0.3106**, ESA 0.3101, flag rate 0.22 % — **best** |
| Candidates E–J (percentile-based) | over-flag at 5–10 % flag rate → low ESA |

**Diagnosis:** z-norm kills the LSTM signal on this dataset. Anomalies in ESA-ADB are mostly *shape* + *magnitude* deviations; z-norm normalises the magnitude away. What's left is a compressed dynamic range (val nominal/anomaly means differ in the third decimal place). The candidate sweep confirms: even the best drift-robust threshold gives test ESA F0.5 `0.31`, well below PCA.

**Retired as a Kaggle contender.** The z-norm recipe was designed for drift-robustness at the input layer; it works *too* well and removes the anomaly signal along with the drift.

### 7.8 `NB 13 — 1D CNN Autoencoder`

| Setting | Value |
|---|---|
| Architecture | Conv1D(32,7) → MaxPool → Conv1D(16,5) → MaxPool → Conv1D(8,3) bottleneck → mirror decoder with UpSampling1D |
| Parameters | 32,034 |
| Split | Kaggle 80 / 20 (`SPLIT_IDX=11_782_656`) |
| Fit | 50,000 random nominal windows, per-window z-norm |
| Threshold tuning metric | `corrected_event_f05` (400-point sweep) |
| Best threshold | 1.1560 |
| Val `corrected_event_f05` | 0.1811 |
| Val precision / recall | 0.1998 / 0.1316 |
| Val events detected | 5 / 38 |
| Val `FP_pred_events` | 20 |
| Test flag rate | 200 / 521,280 (0.04 %) — under-flags |

**Interesting ablation on event diversity.** CNN-AE detects 5 / 38 events. Of those 5, **4 were missed by NB 04**. The overlap is complementary, which makes CNN-AE a theoretical ensemble candidate — but its own precision is too low to lift an ensemble. Pursuing this further requires better CNN-AE recall (deeper architecture? residual connections?), not ensembling with its current form.

**Why topk=5 hurts the CNN:** convolutional smoothing spreads reconstruction error uniformly across channels. `topk=5` picks 5 random-ish channels — lose information that mean-over-58 retained.

### 7.9 `NB 14 — Model Evaluation` (diagnostic)

Not a model notebook — a **viewer** for any trained model's score on the BOOTCAMP 70/15/15 split. Features:
- Model selector (PCA / LSTM-AE / CNN-AE) with glob fallback over timestamped pickles
- Multi-metric DataFrame (Event F0.5/F1/F2, ESA F0.5/P/R/TNR, Event detection rate, FP pred events, Row P/R)
- Confusion matrix with % + top per-channel MSEs in anomaly windows (from NB 15 pattern)
- `plot_timeline` val+test
- Event-block bootstrap CI

**Key diagnostic output** (LSTM-AE v4 via NB 14):

| | Val | Test intern |
|---|---:|---:|
| Events detected | 13 / 26 | **25 / 27** |
| Missed events (count) | 13 | 2 |
| Bootstrap Event F0.5 | 0.6590 ± 0.1107 · CI [0.4542, 0.8475] | **0.6083 ± 0.0558** · CI [0.5118, 0.7088] |

Note the **val < test** inversion (0.659 vs 0.608 bootstrap mean): the val-tuned threshold fits val "correctly" but happens to land above the pre-drift baseline on test, flagging the whole post-drift block. This is the drift-flood pattern reified in a bootstrap.

### 7.10 `NB 18 / NB 19 — Drift diagnosis`

**NB 18** produces four views of drift: per-split summary table, rolling-mean continuous timeline, per-split KDEs per channel, and a per-channel z-score heatmap across all 58 channels.

**Findings:**
- **Top-10 drifting channels** (sorted by |kaggle_test_z|): `29, 14, 21, 30, 38, 22, 31, 39, 15, 23`. Each moves +1.7 σ on `test_intern`, +4.5–6.7 σ on Kaggle test — lock-step, i.e. one physical subsystem shifting regime.
- **Low-variance cluster** (`57, 58, 59, 60`): largest `test_intern` |z| (up to 3.5) but small absolute error (std ≈ 0.02). Invisible to top-k MSE; inflates test_intern metrics in a way Kaggle does not reproduce.
- **Quiet channels** (`42, 64, 65, 70–75`): stable across every split. Drift-free reference signal.

**NB 19** confirms the drift is subsystem-specific, not a global time-of-mission effect. Any detrender needs to cover ≥ 14 channels (not 4). Because Group A moves in lock-step, a **single shared detrend factor per row** (e.g. their rolling mean) works as well as per-channel detrending — the basis of NB 20.

### 7.11 `NB 20 — Score detrending`

Labeled proxy for Kaggle test: `proxy_test = train_full[-521_280:]` (last 521 k rows, same length as Kaggle test). `val = train_full[-2·521_280 : -521_280]` (the block before). Six variants compared (2 models × 3 scoring strategies). Models loaded from disk (`pca.pkl`, `lstm_ae.keras`), no retraining.

**Results on proxy-test (val-tuned threshold):**

| Model | Variant | Threshold | Event F0.5 | **ESA F0.5** | Row Prec | Row Recall |
|---|---|---:|---:|---:|---:|---:|
| LSTM-AE | baseline | 5.585 | 1.000 | **0.000** | 0.266 | 1.000 |
| LSTM-AE | median | 0.561 | 0.875 | **0.861** | 0.830 | 0.304 |
| LSTM-AE | zscore | 4.191 | 0.796 | **0.795** | 0.950 | 0.058 |
| PCA | baseline | 0.140 | 1.000 | **0.000** | 0.266 | 1.000 |
| **PCA** | **median** | **0.0305** | **0.875** | **0.874** | **0.992** | 0.297 |
| PCA | zscore | 12.76 | 0.875 | 0.874 | 0.927 | 0.029 |

**Winner: `PCA + median detrend`** — ESA F0.5 **0.874** vs baseline **0.000**. The baseline entries have `event_f05 = 1.000` and `esa_f05 = 0.000` in the same row — that is the canonical flood signature (predict everything → Event F0.5 = 1 by merging, TNR = 0, ESA = 0).

**Kaggle submission of this winner:** `nb20_pca_median.parquet` — public **0.454**, private **0.599**. Ties NB 04 on private. Loses 0.068 on public, interpreted as the public half sitting in a regime where median detrend over-subtracts (the drift hasn't yet kicked in on the public half).

---

## 8. The drift-flood pathology — why 0.98 on `test_intern` is a lie

### The pattern

On the BOOTCAMP `test_intern` timeline, *both* PCA and LSTM-AE v4 show:
1. Row-score trace is flat and below the val-tuned threshold for the first ~900 k rows of test.
2. At row ~900 k–1 M, the baseline **steps up** above the threshold and stays there for the rest of the timeline.
3. The `y_pred` curve is therefore one giant contiguous block from row ≈ 900 k to the end.
4. All true anomaly events after row 900 k happen to sit inside that block.
5. Event-wise: `TP_events = late-events`, `FP_pred_events = 0`, `Pr_ew = 1.000`, `Event F0.5 ≈ 0.98`.

### Why it's a lie

A hard-coded `y_pred[row >= 900_000] = 1` gets the same Event F0.5. The model is not detecting anomalies — it is detecting *the drift boundary*, which happens to partition the test set such that all late events land on the "positive" side. Event F0.5 has no mechanism to penalise this.

### What catches it

- **`corrected_event_f05`**: drops from 0.98 → 0.47 because TNR ≈ 0.5 (half the nominal rows are flagged).
- **Row F1**: drops to 0.22 because row-level precision is abysmal.
- **Timeline visualisation** (`plot_timeline`): the step is obvious.
- **`fp_samples`** in the metric dict: ~1 M for the PCA case (half the nominal rows).

### What causes it

NB 18 heatmap: 10 channels (Group A) shift +1.7 σ between val and test_intern rows. The reconstruction MSE on those channels rises post-shift; the aggregated window MSE crosses the val threshold; the rest is mechanical.

### Mitigations that work

| Mitigation | Layer | Status |
|---|---|---|
| Score-level rolling-median detrend | score | **Works** (NB 20: PCA ESA 0.00 → 0.87 on proxy). First thing to try. |
| Score-level rolling z-score detrend | score | Works slightly less well than median (lower row recall). |
| Per-window z-norm (input level) | input | Works for drift but kills the magnitude signal → NB 12b signal collapse. |
| Fit-on-tail (last 50k windows) | model | Works **if** test is in the tail regime; breaks if test is earlier. NB 11b. |
| `corrected_event_f05` tuning | threshold | Necessary but not sufficient — still picks the flood threshold if no other lever is applied. |

---

## 9. Kaggle leaderboard — everything submitted so far

Public = public leaderboard (~50 % of test, live). Private = the remaining 50 %, revealed at deadline.

| Submission | File | Public F0.5 | Private F0.5 | Delta | Notes |
|---|---|---:|---:|---:|---|
| NB 04 PCA baseline ⭐ | `baseline_pca.parquet` / `nb04_pca_50k_599.parquet` | **0.522** | **0.599** | +0.077 | **Current champion.** Healthy gap, consistent across halves. |
| NB 20 PCA + median-detrend | `nb20_pca_median.parquet` | 0.454 | **0.599** | +0.145 | Ties champion on private, loses 0.068 on public. Gap consistent with median-detrend helping the drifted (private) half but penalising the earlier (public) half. |
| Row-level PCA (ablation) | `pca_full.parquet` | 0.522 | 0.294 | −0.228 | Same public as champion, **private collapse**. Row-level scoring fails under private drift. |
| PCA + LSTM-AE v1 rank-avg ensemble | `baseline_ensemble.parquet` | 0.522 | 0.476 | −0.046 | No uplift over PCA alone on public; lower private. |
| LSTM-AE v1 | `baseline_lstm_ae.parquet` | 0.112 | 0.078 | −0.034 | Over-flag (17.9 % positive rate). |
| CNN-AE v3 | `cnn_ae_v3.parquet` | 0.000 | 0.238 | +0.238 | Temporal clustering — all positives on private half. |
| Prior-matched threshold | `prior_matched_ensemble.parquet` | 0.096 | 0.032 | −0.064 | Weak on both halves. |
| Prior-matched 5% threshold | `prior_matched_ensemble_5pct.parquet` | 0.078 | 0.476 | +0.398 | Same temporal clustering as CNN-AE v3. |

**The rule:** public-private gap > |0.1| is a red flag. Gap of +0.08 (NB 04) is healthy. Gap of ±0.24+ means the flagged rows are non-uniformly distributed along the timeline.

---

## 10. Cross-model observations

1. **Scoring recipe > architecture.** PCA (linear, k=38, ~50 s) and LSTM-AE (BiLSTM, 400 k params, 8 min GPU) tie on the honest metrics. Model capacity is not the bottleneck.
2. **`FP_pred_events = 0` is the gateway.** Every working model in this project hits this at its best threshold. Every failing model doesn't. It is the single number to monitor during threshold sweep.
3. **Training regime matters more than architecture.** Same BiLSTM backbone, two recipes: v3 (all 92 k windows, z-normalised, 44 epochs) → F0.5 0.094, v4 (last 50 k, raw scale, 6 epochs) → F0.5 0.976. The "better" recipe is the one that lets the MSE baseline drift through the val threshold in a way the test events land on the right side of.
4. **Per-window z-norm is ambivalent.** Kills drift; can also kill the signal. Check anomaly-mean vs nominal-mean on val scores — if they are in the third decimal place (NB 12b: 1.055 vs 1.060), z-norm has eaten the signal.
5. **Public/private Kaggle gap > 0.1 = temporal-clustering red flag.** Row-level Kaggle scoring does not forgive the flood.
6. **Detrending is the most under-utilised lever.** NB 20 lifts PCA ESA F0.5 from `0.000` to `0.874` on the proxy. One `scipy.ndimage.median_filter` call. Should be the default on any new reconstruction model.
7. **`corrected_event_f05` for tuning, always.** `event_f05` is fine as a secondary column but should not drive threshold selection. The drift-flood converges on the same threshold whether you tune on `event_f05` or not — only `corrected_event_f05` sees the TNR penalty.

---

## 11. Known failure modes & their signatures

| Failure mode | Signature | Fix |
|---|---|---|
| **Drift flood** | Event F0.5 ≫ corrected F0.5; flag rate > 20 %; one giant contiguous prediction block | Score-level median detrend; switch tuning metric to `corrected_event_f05` |
| **Row-level spike explosion** | `fp_pred_events` > 100; ROC-AUC < 0.6; smooth PR curve with low F0.5 everywhere | Switch from per-row to window-MSE scoring |
| **Threshold transfer gap** | Test score range entirely below or entirely above the val threshold | Fit-on-tail (11b-style) or drift-robust candidate sweep (NB 12b 8b) |
| **Z-norm signal collapse** | Anomaly-mean and nominal-mean score differ only in 3rd decimal; test flag rate < 0.1 % or > 20 % | Drop z-norm; use score-level detrend instead |
| **Over-regularised LSTM** | Val loss plateaus > 0.7, anomaly/nominal score ranges overlap | Cut training (EarlyStopping with restore_best_weights=True); reduce training-set size so the model has less to generalise away |

---

## 12. Open questions / suggested next steps

1. **Replace NB 04's tuning with `corrected_event_f05` in a copy** (`04-baseline_pca_esa.ipynb`) — then see whether Kaggle private lifts above 0.599. NB 04 currently tunes on `corrected_event_f05` per its docstring, but this should be spot-checked end-to-end.
2. **Apply rolling-median detrend to the NB 04 score** and submit the detrended version. NB 20 says this helps on proxy; NB 04's exact recipe + detrend has not been submitted.
3. **Channel-subset models.** NB 18 flagged 10 drifting channels; `target_channels.csv` lets us run PCA on the 48 stable channels only. Leaderboard impact unknown. Cheap experiment: one NB 04-copy.
4. **Ensemble diversity.** CNN-AE's 4-of-5-events missed-by-NB04 pattern is theoretically useful but the CNN's own recall is too low. A deeper CNN-AE (residual, longer training) targeting those short events would be a principled ensemble candidate.
5. **Fill `PCA_DETRENDED_THRESHOLD` and `LSTM_DETRENDED_THRESHOLD` in `params.py`** from the NB 20 results once the detrended variant is promoted to a blessed model pickle.
6. **Re-run NB 12 bootstrap on v4 weights.** The CI in the table is inherited from a same-architecture prior run, not the current weights.

---

## 13. Reproducibility

- `tune_threshold(val_scores, y_val, metric_fn=corrected_event_f05, n_sweep=60)` — default tuning.
- `bootstrap_f05_ci(..., n_boot=200, event_block=True, seed=RANDOM_STATE=42)` — default CI.
- `run_preprocessing()` (`python -m sentinel.main preprocess`) rebuilds `data/processed/` (BOOTCAMP split) from `data/raw/train.parquet`.
- `run_preprocessing_kaggle()` (`python -m sentinel.main preprocess_kaggle`) rebuilds `data/processed/kaggle/` (Kaggle split).
- Blessed model pickles (do not overwrite): `models/pca.pkl`, `models/pca_kaggle.pkl`, `models/lstm_ae.keras`, `models/cnn_ae.keras`, `models/scaler.pkl`.
- Experimental saves: `models/{model}_nb{XX}_{YYYYmmdd_HHMMSS}.{pkl|keras}`.
- Experimental submissions: `kaggle/submissions/{model}_nb{XX}_{YYYYmmdd_HHMMSS}.parquet`.
- Experimental score dumps: `data/processed/scores/scores_{split}_{model}_nb{XX}.npy`.
- Save / submission blocks at the end of every notebook are **commented out** by default — uncomment after reviewing the run's results. The blessed files are never produced by Run-All.

**Pinned constants** (`src/sentinel/params.py`): `WINDOW_SIZE=100`, `FIT_SIZE=50_000`, `TRAIN_RATIO=0.80`, `BOOTCAMP_TRAIN_RATIO=0.70`, `RANDOM_STATE=42`, `PCA_THRESHOLD=0.060404`, `LSTM_THRESHOLD=1.323612`, `CV_FOLDS=5`.
