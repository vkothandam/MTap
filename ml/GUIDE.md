# ML guide: the daily model, its terms, decisions and known weak spots

A reference for reviews and fine-tuning of the daily model (`ml/tft_vwap`). It explains the
terms as they apply here, how a run is built and graded, what each run in the 5-year series
found, the decisions taken without the user's say, the known shortcomings, and the earlier
explanations that turned out wrong.

- Every number here was checked against the code and the run files on 2026-10-28 (22:00 PT,
  R4 running). Where a number changes, the source named next to it wins.
- Deeper sources: [FEATURES.md](FEATURES.md) (every column, folds, parameters),
  [MODEL-STRATEGY.md](MODEL-STRATEGY.md) (the Q&A behind each choice),
  [docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-ml-daily` (measured results),
  [README.md](README.md) (commands).
- Sections carry tags (`guide-…`) so other docs can point at them.

| § | tag | what |
| --- | --- | --- |
| 1 | `guide-target` | what the model forecasts |
| 2 | `guide-terms` | the terms, in the order they're needed |
| 3 | `guide-data` | the panel and the input groups |
| 4 | `guide-model` | the TFT and the other model types |
| 5 | `guide-training` | folds, epochs, early stopping, hardware |
| 6 | `guide-report` | every file and column a run writes, and how to read it |
| 7 | `guide-compare` | comparing two runs |
| 8 | `guide-runs` | the 5-year series R0–R8 and its results so far |
| 9 | `guide-decisions` | decisions taken without asking, and their status |
| 10 | `guide-shortcomings` | known weak spots |
| 11 | `guide-corrections` | earlier explanations that were wrong |

## 1. What the model forecasts

Tags: `guide-target`

- **One number per stock per day:** tomorrow's VWAP move vs today's,
  `vwap_pct_prev_day(t+1) = VWAP(t+1) / VWAP(t) − 1`.
- **Given as 19 quantiles,** P5, P10 … P95 (`q05` … `q95`). P10 = −1.2% means the model gives a
  10% chance that the move is −1.2% or lower. P5 and P95 bound the band the move should land in
  90% of the time; P10–P90 bounds an 80% band.
- **Each sample** is up to 60 sessions of history (at least 20, the *encoder*) and one step
  ahead, tomorrow (the *decoder*).
- **Quantiles are sorted after prediction,** so they never cross. The run files store them as
  prices: `prev_vwap × (1 + q)`. Metrics convert back to moves.

## 2. Terms

Tags: `guide-terms`

### Scoring a forecast

| term | meaning here | where |
| --- | --- | --- |
| **Pinball (quantile) loss** | Per quantile level τ and error e = actual − forecast: τ·e if e ≥ 0, else (τ − 1)·e. Averaged over the 19 levels and all rows. A miss counts in proportion to its size. Lower is better. The model trains on it. It is in move units (0.0079 ≈ 0.79% per quantile), not bp. | `evaluate.metrics`: `quantile_loss` |
| **Naive forecast** | Every quantile at a 0 move: "tomorrow's VWAP = today's", with zero width. Not a spread of quantiles around 0. | `naive_quantile_loss` |
| **Skill** | 1 − loss / naive loss. R0: 1 − 0.007884 / 0.010738 = **26.6%**. Not based on coverage. 100% would mean every quantile hits the actual exactly, which is impossible for a band; there's no fixed "ideal", only higher is better. | computed from the two above |
| **Coverage** | Share of actuals inside a band. P5–P95 should hold 90%, P10–P90 80%. Below that, the bands are too narrow (overconfident); above, too wide. | `interval_coverage` (P5–P95); compare also gives P10–P90 |
| **Directional accuracy** | Share of rows where the sign of P50 matches the sign of the actual move, on rows where both are non-zero. | `directional_accuracy` |
| **MAPE skill** | 1 − (mean \|P50 − actual\| / actual) / (mean \|prev VWAP − actual\| / actual). How much closer the median lands than "no move". | `mape_skill` |

**Why the naive score flatters.** A zero-width forecast at 0 is easy to beat, so 26% "skill"
says little on its own. The real tests are run vs run (§7), against production, and against
the plain HGB baseline R7 (§4).

### Calibration (do stated probabilities come true?)

| term | meaning here |
| --- | --- |
| **Per-level calibration** | For each τ, the share of actuals at or below P_τ. Should be τ. R1c: P5 → 5.6%, P50 → 50.5%, P95 → 94.0%. |
| **Event probability** | The chance of a move ≥ +m (or ≤ −m), read off the 19 quantiles by interpolation. Moves m: ±0.5% … ±3%. |
| **Grid** | Per move and probability bucket: n, mean forecast probability, how often the move happened (`hit_rate`), `deviation` = hit_rate − mean_prob, and its binomial z. Negative deviation: it happened less often than forecast. \|z\| > 2 is unlikely to be noise. |
| **Signals** | Per move and confidence c (60–90%): the rows where the event probability ≥ c, how many, and how often the event then happened. `base_rate` is how often it happens on all rows. |
| **Ladder** | The move the model gives at each confidence c: up_c = the quantile at 1 − c (P(move ≥ up_c) = c), down_c = the quantile at c. `hit_rate` should be c; `deviation` = hit_rate − c. |
| **PIT and bias** | PIT u = the forecast's own probability of a move at or below the actual. Calibrated forecasts give u spread evenly, mean 0.5. bias = 0.5 − mean u. **bias > 0: forecasts too high ("optimistic")**; < 0: too low ("pessimistic"); "calibrated" when \|bias z\| < 2. |

### Training and testing

| term | meaning here |
| --- | --- |
| **Walk-forward fold** | Train on everything before a date, then test on the next 63 sessions, which the model never saw. 15 folds cover 2023-01-20 → 2026-10-26. |
| **Early-stop block** | The 21 sessions just before a fold's test block. Training stops when the loss there stops improving. They are not training data and not test data. |
| **Epoch** | Here: 500 random batches of 512 samples = 256k samples, not a full pass over the data (§5). |
| **Patience** | Epochs without improvement before stopping: 3. The best epoch's weights are kept (`best.ckpt`). |
| **Out of fold** | A forecast for a day the model producing it never trained on. The 15 test folds together give one per (stock, day). Overlays, the screener and the intraday handoff use only these. |
| **Overfitting** | Learning the training days rather than patterns that carry over. Shows as early stopping after few epochs and a worse test score. |
| **Date fingerprint** | An input with one value per day shared by every stock (market return, sector returns, yields, CPI). 60 sessions of it identify the dates, so the model can memorise training days. Why R1 lost (§8). |
| **Ablation** | Train with and without one input group, compare on the same test days. |

### Model parts

| term | meaning here |
| --- | --- |
| **Static / known-future / observed** | The TFT's input roles. Static: fixed per stock (symbol, industry, exchange, issue type). Known-future: known for tomorrow too (calendar; the pre-market groups). Observed: known only up to today (prices, volumes, earnings flags, the schema-4 groups). |
| **Embedding** | A few learned numbers per category that stand for it: 16 per symbol, 4 per industry, 2 per exchange, 1 for issue type. |
| **Variable selection weight** | The share of attention the TFT gives each input, per sample; the shares sum to 100% within the static, the encoder and the decoder inputs separately. Reported per group and fold (`importance_by_group`). A high weight means used, not useful. |
| **Encoder normaliser** | Each sample's target history is centred and scaled on its own encoder, so stocks with different volatility look alike to the network. |
| **Series break** | A gap of over 20 sessions or a 10× move splits a stock into segments; earlier ones become `SYMBOL~1`, `~2`. No encoder spans a break. Compare maps them back to the ticker. |
| **Masking** | Hiding the pre-market inputs on some training samples so one model also learns to forecast without them (§3). |
| **Overlay** | A small model on top of a base run's out-of-fold quantiles that adjusts them (R6p, R6f, R6c). |
| **HGB** | sklearn `HistGradientBoostingRegressor` with quantile loss, one per level: gradient-boosted trees, no neural net. |

## 3. Data and inputs

Tags: `guide-data`

- **The panel** is the only input: one row per stock and session, written by
  `sourcing-py etrade export-tft`. The 5-year series uses `out/tft/exp5y-r1/panel.parquet`
  (schema 4, 4,279,044 rows, 4,356 common stocks, 2021-10-21 → 2026-10-26). R0 used
  `out/tft/exp5y/panel.parquet`, same rows; its macro and after-hours columns were 0, which R0
  doesn't read.
- **It is on this Mac's internal disk and read once per run,** at start. The NAS holds the source
  data and the run backups, not the training input.
- **Common stocks only** (`issue_type` = CS): ADRs dropped 2026-10-27 at the user's request
  (`export-tft --issue-types CS,DR` brings them back).
- **Base inputs (R0):** the target's history, intraday spread, close/open position vs VWAP,
  volume velocity, transactions per volume, 10-K/10-Q flags, calendar, the static categoricals,
  and the earnings inputs (`--earnings exact`: sessions to and since results, window flags).
- **Schema-4 groups,** each behind a flag ([FEATURES.md](FEATURES.md) §18):

  | group | flag | columns | per stock or per day |
  | --- | --- | --- | --- |
  | market/sector | `--market` | market return, own sector, 11 sectors | 12 of 13 per day |
  | macro | `--macro` | yield changes and slopes, CPI YoY/MoM | per day |
  | session shape | `--session-shape` | high/low position, gaps vs open and previous VWAP | per stock |
  | after-hours | `--after-hours` | after-hours VWAP and volume, close vs VWAP, `ah_known` | per stock |
  | pre-market prices | `--premarket`, group `px` | last pre-market price, open, `pm_px_known` | per stock, day d |
  | pre-market full | group `full` | pre-market VWAP, volume, range, `pm_full_known` | per stock, day d |

- **Pre-market masking** (`--premarket optional`, R4). One uniform draw u per training sample:
  prices hidden when u < 0.25, the full group hidden when u < 0.5. So **25% of samples see no
  pre-market (night), 25% prices only (09:30), 50% everything (09:45)**. Full is never on without
  prices. Only the decoder step (day d) carries pre-market values. At scoring, each state is
  graded separately: the run directory is the 09:45 state, `<run>~px` the 09:30 one and
  `<run>~night` the night one.
- **Earnings modes.** `exact` (the series default): the real next results date where the panel
  has it, else an estimate (`days_to_results_is_est` = 1). `pit`: always the estimate, a
  year-ago results date + 364 days. Training rows nearly always have the exact date; the live
  forecast often has the estimate (§10).

## 4. The models

Tags: `guide-model`

- **TFT** (Temporal Fusion Transformer, pytorch-forecasting). Variable selection per input role,
  an LSTM over the encoder, attention over the 60 sessions, then 19 quantile outputs. Settings
  ([config.py](tft_vwap/config.py)): hidden size 32, 2 attention heads, continuous hidden 16,
  dropout 0.1, Adam at 1e-3, gradient clip 0.1, batch 512.
- **Overlays on a base run** (minutes to fit, trained on the base's out-of-fold forecasts of
  earlier folds only):
  - **R6p / R6f** (`tft-vwap-overlay`): one HGB per quantile, inputs = the base quantiles plus the
    pre-market prices (R6p) or everything (R6f).
  - **R6c** (`tft-vwap-event-overlay`): keeps the base median and scales the band,
    `P50 + s·(q − P50)`, with s = exp(a + Σ b·f). Each f is how far one of yesterday's day-level
    moves went beyond 2σ of its past 63 sessions, capped at 4; s stays within 0.5–3.
- **R7, plain HGB** (`tft-vwap-gbm`): one HGB per quantile on the last day's inputs plus lags
  0, 1, 2, 5 and 20 of the target, same folds and run files. It checks that the TFT earns its
  cost, and it is the cheap screen for any new input group (Q21 rule 3).
- **Screener** (`ml/screener`, R8): HGB on the range from the open, for the 09:30 watch list.
- **Phase 2:** AutoGluon, Chronos-2, DeepAR and others ([docs/TODO.md](../docs/TODO.md)).

## 5. Training

Tags: `guide-training`

- **Folds** (`--n-folds 15 --fold-len 63 --early-stop-len 21 --min-train-sessions 252
  --test-end 1713`): 15 test blocks of 63 sessions, 2023-01-20 → 2026-10-26, each after a
  21-session early-stop block. Fold 0 trains on about 1.2 years (0.95M samples), fold 14 on
  about 4.7 (3.9M). A test fold holds about 211k–216k forecasts.
- **Odd folds** 1, 3, 5, 7, 9, 11, 13 (7 folds) halve the cost of the bake-off runs.
- **One sample per (stock, day)** (`dedupe_train`).
- **An epoch is 500 batches × 512 = 256k samples,** 27% of fold 0's samples and 7% of fold 14's.
  Up to 15 epochs, patience 3. R0 stopped at 4–13 epochs, R1c at 4–10.
- **Hardware:** the Mac's GPU (MPS) with 8 data-loader workers (`--num-workers 8`). With 0 workers
  the GPU sat at about 10% busy. A fold takes about 10–30 min; a 15-fold run about 4 h, 7 odd folds
  about 2 h. Closing other apps helps little; the runs don't adapt to free resources.
- **Pause and resume:** `touch runs/<id>/STOP` stops at the next epoch end;
  `tft-vwap --resume runs/<id>`. `tft-vwap-status runs/<id>` shows the time left.
- **A trained run:** per fold `best.ckpt` (weights plus the data scalers and encoders),
  `metrics.json`, `test_predictions.parquet`, `importance.json`, `logs/`; per run
  `config.json`, `summary.json`, `state.json`, the calibration CSVs and `report/`. Runs are not in
  git; they are copied to the NAS `PProjects/MTap/ml/runs/` (`scripts/storage/backup-runs.sh`).
- **Seed:** 42 for every run (§10).

## 6. Reading a run's report

Tags: `guide-report`

| file | what it holds |
| --- | --- |
| `summary.json` → `pooled_test` | the §2 metrics over all test rows: `rows`, `quantile_loss`, `naive_quantile_loss`, `interval_coverage`, `directional_accuracy`, `mape`, `mape_skill`, `median_ape`, `up_rate` (share of rows that moved up) |
| `summary.json` → `test_table` | the same per fold, with `test_days` (day_idx range) and `epochs` |
| `summary.json` → `importance_by_group` | variable selection weight per group, per fold and the mean |
| `calibration_quantiles.csv` | per-level calibration (`quantile`, `observed`) |
| `calibration_grid.csv` | the grid: `side`, `move`, `bucket`, `n`, `mean_prob`, `hit_rate`, `deviation`, `z` |
| `calibration_signals.csv` | signals: `move`, `confidence`, `signals`, `signal_share`, `symbols`, `mean_prob`, `hit_rate`, `base_rate` |
| `report/ladder_overall.csv`, `ladder_by_symbol.csv` | the ladder: `side`, `confidence`, `n`, `mean_move`, `hit_rate`, `deviation`, `z` |
| `report/bias_by_day.csv`, `bias_by_symbol.csv` | `n`, `mean_pit`, `bias`, `bias_z`, `up_deviation`, `down_deviation` (n-weighted grid deviation per side), `label` |
| `report/grid_overall.csv`, `grid_by_symbol/industry/day.csv` | the grid, split |
| `compare*.json` | a comparison with another run (§7) |

**How to read them, with R1c's numbers:**
- Skill 27.0% and P5–P95 88.4%: the band holds 1.6 points less than it should, split about
  evenly (5.6% below P5, 6.0% above P95). Every 5-year run is a little narrow.
- Ladder, up side at 60%: hit 59.2%, deviation −0.8 points, z −30. With 3.2M rows even small
  deviations have large z; judge the size, not only the z.
- A day with label "pessimistic" and bias −0.26 (2023-01-20) is a day the market rose far more
  than forecast. Day-level bias is the market's move, not a stock problem.

## 7. Comparing runs

Tags: `guide-compare`

`tft-vwap-compare A B [--folds …] [--state/--state-b night|px|full] [--only-pm px|full]`
([compare.py](tft_vwap/compare.py)):
- **Rows matched on (symbol, date)** both runs forecast; series-break segments map to the ticker.
- **Paired difference per day:** mean pinball loss of A minus B on that day's shared rows.
  **Positive means B is better.**
- **95% CI by bootstrap over days,** not rows: stocks move together on a day, so rows from one day
  are not independent evidence. A CI clear of 0 means the difference is unlikely to be day noise.
- **Splits:** earnings days (Earnings_Eve, Earnings_Day, Post_Earnings_Reaction) vs other days;
  per fold, to show a gain in one regime isn't hiding a loss in another; coverage of both runs.
- **The base rule** (MODEL-STRATEGY Q21 rule 1): R1c replaces R0 if the overall CI is clear of
  0; or, on a tie, if R1c is clearly better on earnings days and clearly worse in no fold (the
  user's addition, 10-28). `logs/r4-queue.sh` applies it.
- **Many runs at once, in the dashboard:** the Compare runs tab scores any set of runs on their
  common rows, with skill, direction, rank IC, coverage and the calibration grade, but no CI
  (MODEL-STRATEGY Q30). The per-model tables and the legend are in docs/STATUS.md § Model
  performance.

## 8. The 5-year series

Tags: `guide-runs`

Status 2026-10-29 18:45 PT. This section explains the series; the **current** choices and
results are kept in the Scoreboard at the top of [MODEL-STRATEGY.md](MODEL-STRATEGY.md). Results
in detail: [docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-ml-daily`.

| run | what | folds | status and result |
| --- | --- | --- | --- |
| R0 `20261028-030847Z` | baseline, no new groups | 0–14 | skill 26.6%, P5–P95 88.9%; 6.2% better than production on shared days |
| R1 `20261028-172535Z` | R0 + market, macro, session shape, after-hours | 0–14 | worse than R0 (+2.5%), worse in 7 folds; date fingerprints |
| R1a `20261028-214331Z` | R1 − macro | 0–6 (stopped) | worse than R0 |
| R1c `20261028-233305Z` | R0 + session shape + after-hours | 0–14 | **better than R0 by 0.6%,** CI clear of 0; the TFT base (R7 now beats it at night) |
| R6c `…-event` | event overlay on R0 | 1–14 | no gain |
| R4 `20261029-040858Z` | R1c + pre-market, masked (A) | odd | night ties R1c; loses to the overlays at 09:30 and 09:45 |
| R6p, R6f | overlays on R1c (C) | odd | **R6p leads at 09:30**; R6f ties R7f at 09:45 |
| R4p, R4f | R1c + pre-market always on (prices; everything) (B) | odd | second at 09:30 and 09:45 |
| R7 `20261029-153450Z-gbm` | HGB on R1c's inputs, no TFT | 0–14 | **leads at night:** 1.3% better than R1c, CI clear |
| R7f `20261029-170229Z-gbm` | R7 + full pre-market | odd | **leads at 09:45:** ties R6f (CI spans 0), fewer models wins |
| R7f-cal, R6f-cal, R6p-cal … | walk-forward recalibration (`tft-vwap-recal --by pm,width --window 4`) of each run | odd 3–13 | **R7f-cal best calibrated:** mean \|q err\| 0.78 → 0.10 pts, same pinball; R6f-cal, R6p-cal 0.16; the TFTs only to ~0.9 |
| R5 `20261029-211059Z-gbm` (+`~px`) | **one HGB for 09:30 and 09:45** (`--premarket optional`), + the open for every stock, past volatility, 600 tail iterations | 0–14 | **best so far:** skill 38.3%; after recal 3.5% better than R7f-cal, 4.4% better than R6p-cal (CIs clear, every fold) |
| R5-cal20s, R5~px-cal20s | R5 with rolling recal: each day on the 20 sessions before it (`tft-vwap-recal --sessions 20 --by pm,width`) | 0–14 | **the candidate single model:** mean \|q err\| 0.18 pts (excellent), good or excellent in 13 of 14 folds; the 4-fold R5-cal only 0.57 |
| R8 `screener/20261030-011637Z-…` | screener on R5~px-cal | odd | **kept:** top 20 range 12.5% vs ATR 11.0%, CI clear |
| R2 | + fundamentals | | phase 2 (needs dated EPS) |

**R1c vs R0** (3,169,270 forecasts, 945 days; `compare_vs_r0.json`):

| | R0 | R1c |
| --- | --- | --- |
| pinball loss | 0.007884 | 0.007839 (−0.58%) |
| skill vs naive | 26.6% | 27.0% |
| P5–P95 / P10–P90 coverage | 88.9% / 78.9% | 88.4% / 78.4% |
| direction of P50 | 59.1% | 59.5% |

- Paired difference R0 − R1c: +0.000046 (+0.000021 … +0.000071): R1c better, CI clear of 0.
  It passed the base rule on the first test, not the tie rule.
- Earnings days: +0.000339 (+0.000287 … +0.000387), 17× the other days' gain per forecast
  (+0.000020, CI spans 0). The after-hours reaction to results is where it helps.
- Per fold: clearly better in 1, 2, 7, 9, 10, 12 and 14; clearly worse in none. Fold 0 (2023-Q1)
  leans R0's way (−0.000188, CI just reaches 0).
- Encoder weight: after-hours 6–44%, session shape 5–24% per fold. They are per-stock, so no
  date fingerprint, but the after-hours weight still swings between folds.
- Bands are 0.5 points narrower than R0's.

## 9. Decisions taken without asking

Tags: `guide-decisions`

Choices made while building or running, not put to the user first. Status as of 10-28.

| # | decision | why | status |
| --- | --- | --- | --- |
| 1 | **`issue_type` kept as an input** in R0–R1c and R4. It is a panel column listed in `STATIC_CATEGORICALS`, constant "CS" since the CS-only panel. | Q22/Q23 planned to drop it "in the next run that changes the base (R4 or R5)"; the R4 queue was written before that was built, so R4 kept it. | **User: remove it.** Built 10-28 22:30 (`drop_constant_statics`, MODEL-STRATEGY Q27), after R4 had started: R4 keeps it, R4p, R4f and later TFT runs drop it. The HGB runs (R7, R7f) don't use the statics. |
| 2 | **15 folds and 7 odd folds,** not the plan's 16 and 8. | The plan estimated 16. On the real panel (1,258 sessions) only 15 blocks fit after 252 training and 21 early-stop sessions; the odd ones of 0–14 are 1–13. | Kept. Fixed in the docs 10-28. |
| 3 | **R6c constants:** 2σ event cut, cap 4, 63-session window, band scale 0.5–3, 400 rows per day. | Picked as reasonable, not tuned. | R6c gave no gain; not worth tuning. |
| 4 | **CPI carry of 2 months** when a month is never published (2025-10, shutdown). | Without it every December 2025 session had CPI YoY = 0. | Kept. Matters only for macro, which is out of the TFT. |
| 5 | **`--earnings exact`** for the series. | It was the default and tested better in the design (FEATURES §6). | Kept. Its train/live mismatch is §10 item 6. |
| 6 | **`limit_train_batches` 500 and seed 42** carried over from the 2-year runs. | Defaults. | Open; see §10 items 4–5. |
| 7 | **The R4 queue picks the base itself** by Q21 rule 1 and runs R6p/R6f on that base (R1c), not on R4's night state. | So the bake-off could run unattended. | The user set the rule; the script applied it (R1c). |
| 8 | **The intraday handoff table is still R0's** (`handoff.parquet`). | Q19 said use R0 until a run beats it; R1c now does. | To rebuild from R1c (`tft-vwap-handoff runs/20261028-233305Z-return-4438sym`) when the intraday work resumes. |

Added 10-29 (MODEL-STRATEGY Q35):

| # | decision | why | status |
| --- | --- | --- | --- |
| 9 | **Recalibration settings:** groups pm × width tercile, window 4 folds. | Best on the pre-market rows (R7f 0.10 pts); window 2 is better on rows without pre-market (0.45 vs 0.86), but those are not the 09:45 case. | Re-checked on R5: superseded by decision 13. |
| 10 | **`--tail-iter 600`** for P15 and below, P85 and above; 200 for the rest. | The probe: tails improve to ~1,000 iterations, the median is best at 200–400. 600 is the middle. | In R5. |
| 11 | **`--open` and `--vol` both in R5,** with no separate run each. | The probe on folds 5 and 13 showed each helps (the open at every level, volatility in the tails) and they add up. | In R5. |
| 12 | **No averaging of models** (R7f + R4f had 0.7% lower loss). | The user wants one model, calibration first; the average needs a TFT and was worse calibrated after recal (0.46 vs 0.10). | Not used. |
| 13 | **Rolling recal on the last 20 sessions** (`--sessions 20`, pm × width groups, at least 2,000 rows), not 4 folds. | R5's error drifts between quarters (bias +2.5 to −1.9), so the 4-fold shift was often the wrong way (0.57 pts, 3 poor folds). 20 sessions: 0.18 pts, 13 of 14 folds good or better, for 0.4% pinball. 60 is as good pooled but uneven by fold; 5–10 track months better but cost 0.7–1.4% pinball. Live it is the last 20 graded days. | Chosen for the candidate (MODEL-STRATEGY Q36). |

User decisions on record (MODEL-STRATEGY Current position): **one model for 09:30 and after, the
night forecast dropped, calibration ranked first (10-29);** news backfill and Benzinga are not
important now; production switches only after all models are evaluated; drop ADRs; never stop a
queued training run without asking.

## 10. Known shortcomings

Tags: `guide-shortcomings`

1. **The naive reference is a strawman.** Zero width at 0. Skill numbers look high; only run vs
   run comparisons, production and R7 say whether a model is good.
2. **Bands are a little narrow everywhere,** about 1 point short at P5–P95 and P10–P90. Much worse
   on some days: 74–75% P10–P90 on CPI, jobs and day-after-FOMC days, 64.7% on days the market
   itself moves a lot. A band scale fixes coverage on release days but barely moves the loss
   (LEARNINGS "Release days in R0").
3. **Day-level inputs can't go into the TFT directly** (date fingerprints, R1). Market context
   would need per-stock terms (beta, excess move, own sector), not built.
4. **One seed per run.** The CI covers day-to-day noise, not training randomness. A second R0 with
   another seed would show how much two identical runs differ; R1c's 0.6% gain has not been
   checked against that.
5. **Possibly undertrained.** An epoch is 7–27% of a fold's samples, and folds stop after 4–13
   epochs, so the later folds see each sample about once. Learning curves or a run with more
   batches per epoch would tell.
6. **Earnings `exact` differs live.** Training rows almost always have the real results date; the
   live forecast often has only the estimate (on the right session about half the time).
7. **The bootstrap treats days as independent.** Volatile days cluster, so the true CI is likely
   somewhat wider. A block bootstrap (weeks) would be safer for close calls.
8. **Overlays are thin in fold 1.** R6p/R6f for fold 1 train on fold 0 only (flagged `thin`).
9. **The release-day check used dates typed by hand** (FOMC and jobs 2023–2026, CPI 2023–2025).
   No calendar source exists yet.
10. **Live pre-market prices at 09:30 come from E*TRADE,** training from Massive. The gap is
    measured on one morning only (within ~20 bp on 10-27); spike S1 is still to run.
11. **The intraday model has no daily input for its first 14 months** (no out-of-fold daily
    forecast before 2023-01-20).
12. **`issue_type` was a constant input** up to R4 (§9 item 1): harmless to comparisons, noise in
    the importance report. Left out automatically from R4p on.

## 11. Corrections to earlier explanations

Tags: `guide-corrections`

Answers given in chat, or text in the docs, that turned out wrong. The right version is in the
section named.

| when | what was said | correct | § |
| --- | --- | --- | --- |
| 10-28 chat | Skill 97.2%, from (coverage − 50%) / 40% | 26.6% = 1 − loss / naive loss; coverage is a separate measure | 2 |
| 10-28 chat | Report headers: "pinball in bp"; "optimistic" = forecasts too low; ladder deviation = confidence − hit rate | pinball is in move units; optimistic = forecasts too **high** (bias = 0.5 − mean PIT > 0); deviation = hit rate − confidence | 2, 6 |
| 10-28 chat | R1c includes macro | R1c = R0 + session shape + after-hours; no market, no macro | 8 |
| 10-28 chat | "R1c is not that great" | R1c beats R0 with the CI clear of 0 (rule 1 outright), 17× more per forecast on earnings days | 8 |
| 10-28 chat | Masking: 50% / 25% / 25% / 25% | 25% night, 25% prices only, 50% everything | 3 |
| 10-28 chat | "Training doesn't use the GPU" | It did (MPS), but sat ~10% busy with 0 data-loader workers; 8 workers since | 5 |
| 10-28 chat | A trained model = config + forecast files | Per fold `best.ckpt`, `metrics.json`, `test_predictions.parquet`, `importance.json`, logs | 5 |
| 10-28 chat | Training reads the panel from the NAS every epoch | The panel is on the Mac's internal disk and read once | 3 |
| 10-28 chat | An early R0 vs production table | It compared different calendar quarters; the right one matches on (symbol, date) (Q17) | 7 |
| 10-27 plan, Q11, FEATURES §12, README | 16 folds, 8 odd folds (1 … 15) | 15 folds, 7 odd (1 … 13) | 5, 9 |
| 10-27 plan | Folds early-stop at about 10–20 epochs | 4–13 | 5 |
| Q16 | All always-available inputs together can only help | R1 showed they can hurt, and bundling hid which (Q21) | 8 |
| Q21, Q22 | `issue_type` is a setting in config.py, not a panel column | It is a panel column; config.py lists it as an input, so dropping it still needs no new export | 9 |
| first GUIDE.md draft (10-28, replaced by this file) | naive = equally spaced quantiles; P5 "exceeded 95%" framing mixed up; R1 skill 24.1%; invented `coverage_p5` fields; 7 odd folds because "fold 0 is excluded"; best.pth; epochs 8–12; issue_type "hardcoded to CS in config"; R1c "via the tie rule"; handoff "trained on folds 0–14, tests 15"; CPI fix 10-17; fold-1 overlay "~23k rows" | the sections above (R1 skill 24.7%; overlay fold 1 trains on fold 0's ~216k rows) | all |
