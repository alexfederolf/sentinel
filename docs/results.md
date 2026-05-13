# SENTINEL — Project Results & Knowledge Base

ESA Anomaly Detection Benchmark (ESA-ADB) / Kaggle competition.

**Last updated:** 2026-05-01 · **Kaggle champion:** `11d-pca_hybrid_BEST.ipynb` — public **0.867** / private **0.915** ⭐

---

## 0. The Kaggle challenge — what makes this hard

The task: given **58 channels of spacecraft telemetry** (≈14.7 M labelled training rows + 521 k unlabelled test rows), output a 0/1 anomaly prediction for every test row. Unsupervised — the test labels are hidden.

### Challenge 1 — Baseline shift between train and Kaggle test (primary obstacle)

The data is not stationary. Between the training period and the Kaggle test set, **10 sensor channels** (`29, 14, 21, 30, 38, 22, 31, 39, 15, 23`) undergo a sudden jump to a new baseline — a lock-step level shift of +4.5–6.7 standard deviations across those channels simultaneously, consistent with one physical subsystem changing operating regime. Mean KS(train, test) is 0.43 across all channels, with a max of 0.97 (channel_15).

**What this means in practice:** any model trained on the early portion of the data learns what "normal" looks like in the pre-shift regime. When it encounters the post-shift regime (where Kaggle scores it), the shifted channels appear anomalous everywhere — even when they are not. This is the **primary source of train/test metric divergence** in the project.

Note: alongside this sudden shift, channels also exhibit a slow sinusoidal baseline change over time (orbital/environmental trend). This is a separate phenomenon — a gradual drift handled by per-channel rolling-median detrending in the input preprocessing, not by channel selection.

NB 18 and NB 19 contain the shift analysis. Every working submission in §6 is, in some way, a different strategy for handling these 10 channels.

### Challenge 2 — The metric weighs precision 4× more than recall

The Kaggle scoring metric is **F0.5** (β = 0.5), not F1. That means false positives hurt 4× more than missed events do. A submission that catches 24/25 events but flags 5 % of nominal rows scores worse than one that catches only 14/25 events and flags 1 %.

The intuitive approach — "catch as many anomalies as possible" — is the **wrong** approach here. Most attempts to lift event recall by making the model more sensitive end up flooding nominal rows with false positives, and the F0.5 score collapses. See §1 for the exact formula and §4 for the shift-flood failure mode this produces.

### Challenge 3 — Public and private leaderboards score differently

Kaggle splits the 521 k test set roughly 50/50 into a **public** leaderboard (visible during the competition) and a **private** leaderboard (revealed at deadline; the one that determines final ranking). On this dataset the two halves come from different parts of the timeline, and submissions score very differently on each.

A real example from this project: a hybrid variant tuned to maximise local val/test scored **public 0.897 / private 0.903**. A slightly less-aggressive variant from Ekaterina scored **public 0.867 / private 0.915**. Public says the first wins (+0.030). Private says hers wins (+0.012). **Private is what counts** — the "worse" public is the actual competition winner. See §5.3 for the full story and §5.4 for the post-mortem.

### Challenge 4 — Local validation doesn't predict private leaderboard

Compounding Challenge 3: even the best local val/test split (rows 10.7 M / 12.7 M / 14.7 M of train) doesn't reliably predict private score. Two specific traps:

- **Sharp local optima don't transfer.** `ENV_MA_WINDOW=20_000` gave local Test 0.8545 — but neighbours `ma_w=15k` and `ma_w=25k` dropped to 0.71–0.80. That sharp peak was the sign of overfitting. Smoothly-flat regions of parameter space generalise to private; pointy peaks do not.
- **Adding "obviously useful" channels often regresses.** NB 11e added channels 17/25/34 to the 6-channel champion (NB 11c) and got higher local val (0.872 vs 0.853). Kaggle: 0.832 / 0.853 — both lower than 11c's 0.897 / 0.887.

**Rule of thumb:** prefer flat plateaus to sharp peaks. Submit ablations to Kaggle rather than trusting a single local split. When in doubt, do the simpler thing.

---

**The thread through all four:** the channel-level baseline shift dominates. Every working submission in §6 attacks it — by selecting away from the shifted channels (NB 11c), residualising against a long-window envelope baseline (NB 11d hybrid), or fitting on the tail of training data where the operating regime matches the test period. Every failed submission, including the high-capacity LSTM and CNN autoencoders, has a shift-related root cause behind a different symptom.

---

## 1. The metric — corrected event-wise F0.5

Kaggle scores submissions with the **ESA-ADB corrected event-wise F0.5** (`corrected_event_f05` in `sentinel.ml_logic.metrics`).

### What "event-wise" means

A predicted row is judged in the context of the **ground-truth event** it overlaps, not in isolation:

- **TP_events** = the number of *true* anomaly events that have at least one overlapping predicted-positive row. (Catching the event at all counts as a hit — overlap fraction does not matter.)
- **FN_events** = N_events − TP_events.
- **FP_pred_events** = the number of *predicted* contiguous positive segments that do not overlap any true event. (Each spurious flagged block counts once, regardless of length.)

So 100 isolated false-positive flags become 100 FP_pred_events; one giant 100 k-row false-positive block becomes 1 FP_pred_event.

### The metric formula (ESA correction)

```
Pr_ew     = TP_events / (TP_events + FP_pred_events)   # event-wise precision
Rec_e     = TP_events / N_events                        # event-wise recall
TNR       = TN_rows / (TN_rows + FP_rows)               # row-level true-negative rate

Pr_corr   = Pr_ew · TNR                                 # ESA-corrected precision
F0.5      = 1.25 · Pr_corr · Rec_e / (0.25 · Pr_corr + Rec_e)
```

The β=0.5 means **precision (Pr_corr) counts 4× as much as recall**.

### What the TNR correction fixes

Without the TNR correction, a "predict everything" submission gets `Pr_ew = 1.0`, `recall = 1.0`, F0.5 = 1.0 — the metric is gameable by flagging the whole timeline. The TNR correction multiplies `Pr_ew` by the row-level true-negative rate. Predicting 50 % of rows as positive ⇒ TNR ≈ 0.5 ⇒ `Pr_corr` halves even if `Pr_ew = 1`. This eliminates the shift-flood failure mode (§4).

### Practical guidance — what to monitor

For any new model, track these four numbers in order:
1. **`FP_pred_events`** — should be 0 at the chosen threshold. If it is 0, `Pr_ew = 1` and the metric is dominated by TNR and recall.
2. **Flag rate** — fraction of test rows predicted positive. Should be 0.5 %–10 %. Below 0.5 % under-flags; above 10 % is the flood.
3. **TNR** — should be ≥ 0.99 for a submittable model.
4. **Event recall** — between 0.5 and 0.8 is realistic; pushing recall higher costs precision.

**Always tune on `corrected_event_f05`, never on plain `event_f05`.** The latter has no TNR penalty and converges on the flood.

---

## 2. Data structure

### Data shape

| | Rows | Events | Anomaly rate | Labels |
|---|---:|---:|---:|:-:|
| `train.parquet` | 14,728,321 | 190 | 10.48 % | yes |
| `test.parquet` (Kaggle) | 521,280 | — | — | **no** |
| Internal val (rows 10.7 M – 12.7 M) | 2,000,000 | 26 | — | yes |
| Internal test (rows 12.7 M – 14.7 M) | 2,028,321 | 25 | — | yes |

Event lengths span 1 to 116 061 samples (median 602). Wide range means fixed 100-row windows are a compromise: short events get diluted, long events get over-represented.


### The 58 target channels

Of 76 sensor channels, 58 are competition-relevant (`data/raw/target_channels.csv`). Notable groups identified by shift analysis (NB 18/19):

- **Spectral cluster `41–46`** (6 channels) — frequency-domain magnitude for the dominant subsystem. Stable across train→test. **The single most important channel set for Kaggle scoring** (§6).
- **Level-shifted channels `29, 14, 21, 30, 38, 22, 31, 39, 15, 23`** (10 channels) — lock-step baseline jump between train and Kaggle test. Useful as anomaly-signal sources via envelope/z-score scoring (§5.3) but harmful as direct PCA inputs.
- **Stable reference `42, 64, 65, 70–75`** (8 channels) — no baseline shift, no anomaly signal.
- **Low-variance cluster `57–60`** — high relative change but tiny absolute MSE; misleading on local test, near-invisible on Kaggle.

### Signal structure

Each channel signal decomposes into additive components:

```
y(t) = sin1(t)  +  sin2(t)  +  add(t)  +  noise(t)
```

| Component | Period | Description |
|---|---|---|
| `sin1` | ~2.5 M rows | Slow orbital/environmental baseline — the gradual drift component |
| `sin2` | ~2 000 rows | Medium-frequency oscillation; visible after removing `sin1` |
| `add` | ~100 rows | Step-like component: zero most of the time (Type 1) or a constant sustained for 2–5 samples (Type 2) — normal behaviour in both cases |
| `noise` | — | Residual after removing all three components |

All six spectral channels (41–46) share the same qualitative structure and dominant periods. FFT-based period detection on detrended nominal data (`ek_freq_eda.ipynb`) reliably identifies `sin2` (~2 000 samples); `sin1` is extracted with a fixed smoothing window (BIG_WINDOW = 50 000) rather than FFT because its period is too long for reliable spectral detection. Detected periods are saved per-channel in `data/freq_map.json`.

**Key finding: anomalies live in the residual (`noise`).** The slow components (`sin1`, `sin2`) carry large amplitude but are regular and predictable. Anomalies manifest as deviations in the high-frequency residual — the component the model actually needs to score. This is why per-channel input detrending (removing `sin1`) improves reconstruction quality: it removes the large-amplitude slow trend so the model can focus on the anomaly-carrying residual.

### Two split conventions

1. **Alternative split** (`data/processed/`): 70/15/15 chronological with anomaly-aware boundary snapping. Used by NB 11/12/13/14 for initial comparison. **Reported `test_intern` numbers on this split are systematically optimistic for Kaggle** because the test portion sits before the Kaggle shift regime.
2. **Kaggle split** (`data/processed/kaggle/`): all of train kept for fitting; `test.parquet` scored separately. Internal val/test carved from rows 10.7 M / 12.7 M / 14.7 M, snapped so a ±2 000-row neighbourhood is fully nominal. Used by every Kaggle-facing notebook (`04`, `11c`, `11d-*`, `11e-*`, `12b`, `13`, `20`).

### Shared modules (`src/sentinel/ml_logic/`)

- `scorer.score_windows(model, X_rows, win=100)` — reshape rows into non-overlapping windows, compute reconstruction MSE, broadcast back to rows.
- `metrics.corrected_event_f05` — the Kaggle metric. Always tune on this.
- `thresholds.tune_threshold(scores, y_true, metric_fn, n_sweep=80)` — log-spaced sweep, argmax the metric.
- `fusion.fusion_diagnostics(y_base, y_new, y_true)` — segment-aware OR-fusion safety check. Counts `FP_pred_events`, not rows; returns verdict ∈ {`reject`, `borderline`, `submit`, `submit_strong`}.
- `cv.run_cv(env_ma_window=...)` / `cv.run_sweep([...])` — rolled-origin K-fold CV harness. CLI: `python -m sentinel.ml_logic.cv --sweep`.
- `viz.plot_timeline`, `viz.plot_event_analysis` — standard timeline and per-event detection visualisations.

### Constants (`src/sentinel/params.py`)

`WINDOW_SIZE=100`, `FIT_SIZE=50_000`, `RANDOM_STATE=42`, `PCA_THRESHOLD=0.060404`. Path constants (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `SUBMISSIONS_DIR=PROJECT_ROOT/kaggle/submissions`) live in `params.py` and are re-exported by `ml_logic.data` for backward compatibility. Active model files in `models/`: `pca.pkl` (Kaggle baseline + 6ch hybrid freq stream), `lstm_ae.keras`, `scaler.pkl` (58-ch RobustScaler, used by both Kaggle preprocessing and FE inference), plus the FE bundle (`pca_fe_46ch.pkl`, `fe_46ch_*.json` sidecar). Experiments save with `_nb{XX}_{timestamp}` suffix so re-runs don't overwrite each other.

---

## 3. The two scoring decisions that matter

### 3.1 Window-MSE beats per-row MSE

The single most important design choice. Reconstruction error is reduced to **one score per 100-row window**, then broadcast to its 100 rows. This produces a smooth score curve where the threshold sweep can find a clear gap between nominal and anomalous windows.

Per-row MSE produces a spiky curve with no clear gap and hundreds of FP_pred_events. NB 04 ablation: window-mean F0.5 0.770 → public 0.522. Per-row F0.5 0.698 → public **0.277**.

NB 03 (Isolation Forest, per-row by design) hits the same wall — 2/38 events caught, unstable threshold, Kaggle ≈ 0. Any model with native per-row scoring must be reduced to windows before it competes.

### 3.2 Shift mitigation outweighs model capacity

Three mitigation layers, in increasing effectiveness on this dataset:

| Layer | Technique | Used by | Effect |
|---|---|---|---|
| **Channel selection** | Drop level-shifted channels, fit on stable subset | NB 11c (channels 41–46) | NB 04 0.522/0.599 → NB 11c **0.897/0.887** (+0.30 public, biggest single jump) |
| **Score-level detrending** | Subtract rolling-median from row scores (removes slow baseline in reconstruction MSE) | NB 20 (PCA + median detrend) | Lifts ESA F0.5 to ≈ 0.874 on local proxy; ties NB 04 on Kaggle private (0.599) |
| **Per-channel envelope processing** | Envelope-residual + z-score on level-shifted channels | env stream of NB 11d hybrid (channels 14/21/29) | Adds events that PCA misses; lifts hybrid private from 0.887 → **0.915** |

PCA, LSTM-AE, and CNN-AE all give comparable scores when given the *same* shift-mitigation recipe. Architecture is a smaller lever than the input recipe.

---

## 4. The shift-flood failure mode

The pathology underlying every "0.98 Event F0.5 on test_intern" headline before NB 04.

**Pattern:** model trained on the pre-shift period; reconstruction MSE rises sharply at the shift onset; the val-tuned threshold lands above the pre-shift baseline but below the post-shift level; predicted-positive rows form one giant block from the shift onset to the end of the test period; this block happens to contain all late events.

**Consequence:** `Pr_ew = 1.0`, `event_f05 ≈ 0.98`, but row-level TNR ≈ 0.5 → `corrected_event_f05 ≈ 0.47` and Kaggle ≈ 0.

**What catches it:**
- `corrected_event_f05` (the Kaggle metric; always tune on this).
- Flag rate > 20 % at the chosen threshold.
- Timeline plot showing one contiguous predicted block starting mid-test.

**What fixes it:** any of the three shift-mitigation layers in §3.2.

### 4.1 The fusion failure mode (related pathology)

Adding a low-flag-rate stream as a third OR channel looks free — "+300 rows is only +0.058 pp flag rate, can't hurt much." It can. F0.5 counts **`FP_pred_events`** (contiguous predicted segments outside any true event), not individual rows.

**The math.** Window-MSE scoring produces 100-row contiguous blocks. If the third stream's flags land outside true events, every isolated 100-row block is a fresh `FP_pred_event`. A hybrid sitting near `Pr_ew = 1.0` (no FP segments) loses substantially per added FP segment:

```
Pr_ew_new = TP_events / (TP_events + new_FP_segments)
```

With ~16 true events caught and 0 → 4 new FP segments, `Pr_ew` drops 1.00 → 0.80 → `Pr_corr` drops 20 % → F0.5 drops ≈ 0.06–0.08. This is exactly what the freq6 fusion submission cost (private 0.915 → 0.845).

**Mandatory fusion check** (use `fusion.fusion_diagnostics(...)` from `sentinel.ml_logic.fusion`): before any OR-fusion submission, verify on `test_internal`:
1. **`Δ TP_events ≥ Δ FP_pred_events`** — the new stream catches at least as many missed events as it adds false segments.
2. Flag-rate delta ≤ 0.2 pp on Kaggle (necessary but **not sufficient**).

If rule 1 fails, do not submit, regardless of how small the row-count delta looks.

---

## 5. Per-notebook results

### 5.1 `04-pca_1.ipynb` — full-58-channel PCA baseline

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

First submission with a healthy public/private gap (+0.077). All later notebooks beat it, but it serves as the reference baseline.

### 5.2 `11c-pca_6ch.ipynb` — 6-channel PCA (largest single improvement)

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

Dropping 52 channels gave +0.375 public. The dropped channels include the 10 lock-step level-shifted channels and the misleading 57–60 cluster. Fitting on 6 stable spectral channels eliminates the shift-flood at the source. **This is the foundation for the hybrid champion (§5.3).**

### 5.3 `11d-pca_hybrid_BEST.ipynb` ⭐ — Kaggle champion

OR-fusion of two streams with independent thresholds. Bit-identical to Ekaterina's original `submissions/pca_pca_6ch_zsenv_3ch.parquet` (preserved at commit `93a611e`; file removed from working tree as redundant — see §9).

| Stream | Channels | Count | Recipe |
|---|---|---|---|
| **Freq (PCA)** | `channel_41..46` | **6** | PCA reconstruction error on raw scaled values, fit on tail-50k nominal windows, EV=0.95. Same as NB 11c. |
| **Env (z-score)** | `channel_14, 21, 29` | **3** | `rolling_min(env_w=200) − centered_MA(ma_w=5_000)`, normalise by nominal-train std and per-channel p99, aggregate via `top_p_mean(k=2)` over 3 channels. |
| **Fusion** | — | — | OR. Independent thresholds tuned via `tune_threshold(corrected_event_f05)` on local val. |

| Setting | Value |
|---|---|
| Total channels used | **9** of 58 |
| `ENV_WINDOW` (envelope) | 200 |
| `ENV_MA_WINDOW` | **5,000** |
| `ENV_REF_PCT` | 99 |
| Freq threshold | 0.028049 |
| Env threshold | 1.236990 (k_env=2/3, val grid) |
| Local val ESA F0.5 | **0.9391** (events 20/26 = 76.9 %) |
| Local test ESA F0.5 | 0.8216 (events 14/25 = 56 %) |
| Flag rate | val 1.18 % / test 1.29 % |
| **Kaggle public / private** | **0.867 / 0.915** ⭐ |
| Submission | `kaggle/submissions/pca_hybrid_envzscore_BEST.parquet` |

**Why it wins on private:** the env stream catches anomalies that the PCA stream misses, without flooding — because the envelope-residual baseline is itself shift-corrected (the centered MA absorbs the lock-step baseline jump in channels 14/21/29). The OR fusion fires only when an event has either spectral disturbance (PCA stream) or envelope deviation (env stream).

**How the two streams came together.** The freq stream is NB 11c's 6-channel PCA. The env stream originated in two precursor notebooks (`ek_baseline_zscore.ipynb` — per-channel rolling z-score with top-p% aggregation and F0.5 threshold tuning; `ek_freq_eda.ipynb` — FFT-based period detection that identified channels 14/21/29 as periodic carriers). The hybrid is a straight OR-fusion of those two streams with independent val-tuned thresholds.

### 5.4 The `ma_w=20_000` overfitting trap

A variant of §5.3 with `ENV_MA_WINDOW=20_000` (instead of 5_000) and an added joint-threshold cell pushed local Val 0.9214 / local Test **0.8545** (above the 5k variant's 0.9391/0.8216). Submitted as `pca_zsenv_pca41,42,43,44,45,46_zsenv14,21,29_k1_*.parquet`: Kaggle public **0.897** / private **0.903**.

The +0.030 public was real, but the −0.012 private was the actual cost. The 20k variant overfit the local val/test split. The sharp local optimum (neighbours `ma_w=25k` drop to 0.71–0.80) was the warning sign in retrospect.


---

## 6. Kaggle leaderboard — key submissions

| Model | Channels | Public | **Private** | Δ | Notes |
|---|---:|---:|---:|---:|---|
| **NB 11d hybrid** ⭐ — PCA 41–46 + env z-score 14/21/29, ma_w=5k | **9** | 0.867 | **0.915** | +0.048 | **Champion.** Bit-identical to Ekaterina's submission. |
| NB 11d, ma_w=20 000 | 9 | 0.897 | 0.903 | +0.006 | Won public, lost private — sharp local-test optimum overfit (§5.4). |
| NB 12d — LSTM-AE on env-residual 14/21/29 OR champion | **3** | 0.867 | 0.915 | +0.048 | Ties champion. +1 TP on internal test; event not present in Kaggle private half. |
| NB 12c fusion failure — NB 11d OR LSTM-Forecaster on 41–46 | 9 | 0.830 | 0.845 | +0.015 | 0 → 4 FP_pred_events, Pr_ew: 1.00 → 0.85. Lesson: count segments, not rows (§4.1). |
| Env stream alone — channels 14/21/29, k=2 | **3** | 0.788 | **0.914** | +0.126 | Largest public/private gap. Confirms env carries private-half signal; PCA-freq carries public. Justifies OR-fusion. |
| NB 11c — PCA on channels 41–46 | **6** | 0.897 | 0.887 | −0.010 | Biggest single jump (+0.288 private vs baseline). Foundation for hybrid. |
| NB 20 — full-58 PCA + score-level median detrend | 58 | 0.454 | 0.599 | +0.145 | Detrend recovers private signal but suppresses public. |
| NB 04 — full-58 PCA baseline | 58 | 0.522 | 0.599 | +0.077 | Reference baseline. |

**Headline observations:**

- **58 → 6 channels (NB 04 → NB 11c):** +0.375 public / +0.288 private. Single biggest jump in the project.
- **OR-fusion with env stream (NB 11c → NB 11d):** −0.030 public but **+0.028 private**. The env stream alone scores 0.914 private — the two streams are complementary: PCA-freq carries the public half, env carries the private half.
- **Architecture is a smaller lever than input recipe** (§3.2): LSTM-AE on the same env-residual (NB 12d) ties the PCA champion at 0.915.

---

## 7. Side experiments

These notebooks explored alternative metrics or research directions and are not submitted to Kaggle. Kept for reference.

- **NB 11 (PCA, alternative split)** — primary diagnostic baseline for the shift-flood pathology. Tunes on `event_f05` (legacy metric, predates the metric switch). Reports test_intern Event F0.5 0.9843 = shift-flood artefact. Canonical cautionary example.
- **NB 14 / 14c** — multi-metric diagnostic panels for any trained model. Reports `event_f05`, `event_f1`, `event_f2`, `corrected_event_f05` side-by-side. `event_f0.5 ≫ corrected_event_f05` is the flood signature.
- **NB 16** — manual anomaly investigation per event. Characterises missed events: 4 ultra-short clustered (~rows 937–940k), 4 large-deviation events in continuously-evolving channels, 3 weak everywhere.
- **NB 18 / 19** — the shift analysis notebooks. Output is the channel groupings used everywhere (level-shifted list, stable reference list).
- **NB 32** — FastAPI demo for the API showcase.
- **`Alex — model 1/2/3/3b`** — early Isolation Forest / LSTM / LSTM-AE prototypes; superseded by NB 03/12/12b.

---

## 8. Open directions worth trying

Ranked by expected impact on Kaggle private:

1. **Submission ensembling** — union-OR of `pca_hybrid_envzscore_BEST.parquet` (private 0.915) with a complementary high-precision submission. If they catch different events, private may exceed both. Cheap to try.
2. **Soft-OR fusion** — combined_score = `freq_norm + α · env_norm` for α ∈ {0.5, 1, 2}. Smoother than hard OR; may give better threshold transfer.
3. **Shorter PCA windows (`WINDOW_SIZE=25`)** specifically for ultra-short events. Tested as a third-stream addition (failed locally with flag rate 4.5 %). `WINDOW_SIZE=50` was submitted as `archive/04b-pca_win50.ipynb` (full-58 channels, replacement for freq stream's window=100): public 0.522 / private 0.476 — same public as NB 04, **−0.123 private**. Halving the window doubles short-event weight but on the full-58-channel input the shift-flood still dominates. Re-running win=50 on the **6-channel freq subset** (NB 11c recipe) is the open variant.
4. **LSTM-AE on shift-mitigated input (NB 12d — completed).** Trained on the envelope-residual of channels 14/21/29 (same recipe as the env stream of NB 11d). OR-fused with champion: internal test +1 TP event, Kaggle ties champion exactly at 0.915. The gained event is not present in the Kaggle private half. Confirms §3.2: given the same shift-mitigation recipe, architecture is a smaller lever than the input recipe.

**Why retraining LSTM-AE on full-58 channels is not in this list:** the shift-flood is architecture-independent (§4). The open directions above all change the *input*, not just the model class.

---

## 9. Reproducing the champion

```python
# In notebooks/11d-pca_hybrid_BEST.ipynb:
ENV_WINDOW    = 200
ENV_MA_WINDOW = 5_000     # NOT 20_000
ENV_REF_PCT   = 99
FREQ_NAMES    = [f'channel_{i}' for i in range(41, 47)]   # 6 channels
ENV_NAMES     = ['channel_14', 'channel_21', 'channel_29']  # 3 channels
# joint-threshold cell: SKIPPED (independent thresholds only)
# k_env: tune by val grid (lands on k=2/3)
```

Run the notebook end-to-end on the Kaggle split (`DATA_SOURCE='kaggle'`). The submission cell writes `kaggle/submissions/pca_hybrid_envzscore_{ts}.parquet` (commented out by default — uncomment to enable). The output is bit-identical to `kaggle/submissions/pca_hybrid_envzscore_BEST.parquet` (the canonical champion submission at private 0.915) — verified row-by-row, 521,280/521,280 match. The original Ekaterina file (`submissions/pca_pca_6ch_zsenv_3ch.parquet` at commit `93a611e`) was also bit-identical and has been removed from the repo as redundant.

**Pinned constants** (`src/sentinel/params.py`): `WINDOW_SIZE=100`, `FIT_SIZE=50_000`, `RANDOM_STATE=42`, plus all path constants (`DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `SUBMISSIONS_DIR`). Active model files (do not overwrite): `models/pca.pkl`, `models/lstm_ae.keras`, `models/scaler.pkl`. Experiments save with `_nb{XX}_{YYYYmmdd_HHMMSS}` suffix.

---

## 10. FE pipeline

Distinct from Kaggle. The FE demo scores a 300 k-row internal slice through the API and a notebook walkthrough. The model and recipe come from **NB 11e** with `PRUNE_AGGRESSIVE=True` (46-channel variant), not from the Kaggle champion (NB 11d hybrid).

### 10.1 Why a different model than Kaggle

NB 11e produces two artefacts:
- **49 channels** (`PRUNE_AGGRESSIVE=False`) — Kaggle internal-test ESA F0.5 0.853, beats 46ch on the public leaderboard.
- **46 channels** (`PRUNE_AGGRESSIVE=True`) — Kaggle internal-test ESA F0.5 0.810 but **0.818 on the FE internal-test slice (post-filter)** vs 0.716 / 0.753 for baseline / 49ch / 58ch under the same recipe. **46ch wins on the FE proxy and is what the demo shows.**

### 10.2 FE model artefacts

| File | Contents |
|---|---|
| `models/pca_fe_46ch.pkl` | Trained PCA (46 channels × 100 rows = 4 600-dim windows, 316 components, EV 0.950) |
| `models/scaler.pkl` | RobustScaler fit on 58 channels (predict_fe46 applies this and then slices to 46) |
| `models/fe_46ch_20260429_153751.json` | Sidecar — threshold, post-filter config, fit config, val/test metrics, channel list |
| `data/raw/target_channels_fe.csv` | The 46 FE channels (58 competition channels minus 12 dropped: near-constant or level-shifted channels 16, 24, 32, 64–66, 70–76) |
| `data/processed/test_api_fe.npy` + `y_test_api_fe.npy` | 300 000-row demo slice (rows 14 175 000 – 14 475 000 of train.parquet, 6 events, ~47.6 % anomaly density) |

Sidecar key values (frozen at train time, do not retune in the demo):

| Field | Value |
|---|---|
| `threshold` | **0.001 970** (tuned via `tune_threshold(corrected_event_f05)` on val 10.7 M – 12.7 M) |
| `val_esa_f05` | 0.8534 |
| `input_detrend` | rolling **median**, window 100 000 rows (per-channel input detrend — removes slow baseline drift before PCA) |
| `score_detrend` | rolling **median**, window 1 000 windows (= 100 000 rows) on the row-broadcast scores |
| `post_filter` | `clean_predictions(min_len=100, max_gap=500)` — gap-fill ≤ 500-row gaps, then drop blocks shorter than 100 rows |
| `fit` | `tail_50k_NB11e`, `train_end=10 700 000`, `random_state=42` |

Internal-test (12.7 M – 14.7 M) ESA F0.5: **0.717 baseline → 0.818 with post-filter** (events 14/25, TNR 0.991, flag rate 4.2 %, predicted blocks 36 → 19). The post-filter is the single biggest lift on this slice — `min_len=100` removes spurious singleton windows; `max_gap=500` merges fragmented event coverage.

### 10.3 Inference path

`src/sentinel/ml_logic/predictor.py::predict_fe46(X_raw, model, scaler, features_full, features_fe, threshold, …)` — applies the NB 11e recipe and returns `{id, is_anomaly}` rows. `predict_fe46_report()` is the analytical variant returning the full diagnostic dict (row scores, per-channel MSE, top-N channels per window).

### 10.4 Notebooks for the demo

| Notebook | Role |
|---|---|
| `notebooks/11e-pca_detrended_FE.ipynb` | Source of `pca_fe_46ch.pkl`. Run with `PRUNE_AGGRESSIVE=True` to reproduce. |
| `notebooks/30-model_eval_fe.ipynb` | Evaluates the FE model on internal val/test. Reproduces the sidecar metrics. |
| `notebooks/31-slices_fe.ipynb` | Slice picker — chose rows 14.175 M – 14.475 M as the FE demo window. Saves `test_api_fe.npy` / `y_test_api_fe.npy`. |
| `notebooks/32-api.ipynb` | API showcase — loads the FE artefacts, runs inference, end-to-end diagnostics. |

### 10.5 API endpoints

`api/fast.py` caches one prediction over `test_api_fe.npy` at startup. Available endpoints:

| Endpoint | Returns |
|---|---|
| `GET /timeline` | `[{id, is_anomaly}]` for all 300 k rows |
| `GET /predict_by_id?start=..&end=..` | Filtered timeline |
| `GET /report` | Row-MSE, per-channel MSE, top contributing channels per window, threshold, anomaly rate |
| `GET /channels?channel=..&start=..&end=..` | Raw signal values + anomaly label for a single channel |
| `GET /features` | List of 58 channel names |
| `POST /predict` | Score user-supplied rows (58 raw channels in `target_channels.csv` order) |

### 10.6 Verification

`scripts/verify_fe46.py` reloads the model + sidecar and re-runs internal val/test. Sidecar metrics (val 0.8534, test 0.717 baseline / 0.818 post-filter) should reproduce exactly. Run before any FE re-demo.
