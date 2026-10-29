# ML model strategy: running notes and Q&A

A hand-maintained reference for the modelling choices behind `ml/`.
- **Current position** is the latest state. It is updated whenever an answer changes it.
- **Q&A log** keeps every question and its answer, newest last. Numbers and facts are as of each entry's date.
- **Glossary** explains the terms the answers use.

## Scoreboard: model choices and results now (2026-10-29, 14:30 PT)

The one place for "which model, and how good". Updated whenever a run finishes or a choice
changes; details and CIs are in [docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-ml-daily`, terms in
[GUIDE.md](GUIDE.md).

**In use**

| role | run | inputs | result |
| --- | --- | --- | --- |
| production (live night forecast) | `20261005-025909Z-return-3712sym --fold 3` | 2-year, stock inputs, no earnings | reference; switches only after all models are evaluated (user, 10-28) |
| research base (TFT) | **R1c** `20261028-233305Z-return-4438sym` | R0 + session shape + after-hours | skill 27.0%, P5–P95 88.4%; 0.58% better than R0 (CI clear), most on earnings days; R7 (HGB) now beats it at night |
| intraday handoff | R0 `20261028-030847Z-return-4438sym/handoff.parquet` | | to rebuild from R1c |

**Goal (user, 10-29, Q35):** one model for every forecast at and after 09:30, with full
pre-market; **the night forecast is dropped**. Ranked first by how well the quantiles hit their
probabilities (dashboard mean/max |q err|, grade, ECE, coverage, bias), then by pinball loss.

**Calibration first** (6 odd folds 3–13, the 716,862 rows with full pre-market every run has;
walk-forward recalibration `-cal` = `tft-vwap-recal --by pm,width --window 4`):

| model | forecast | pinball | mean / max \|q err\| (pts) | ECE | P5–P95 / P10–P90 | bias | grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **R7f-cal** | 09:45 | 0.007901 | **0.10 / 0.22** | **0.97** | 90.0 / 80.0 | +0.05 | excellent |
| R7f | 09:45 | 0.007902 | 0.78 / 1.08 | 1.17 | 89.4 / 79.3 | −0.72 | good |
| R6f-cal | 09:45 | 0.007908 | 0.16 / 0.37 | 1.31 | 90.0 / 80.1 | +0.15 | excellent |
| R6f | 09:45 | 0.007907 | 0.94 / 1.55 | 1.61 | 90.7 / 80.9 | −0.86 | good |
| R4f-cal | 09:45 | 0.007950 | 0.87 / 1.55 | 1.45 | 90.8 / 81.5 | −0.57 | good |
| R7f+R4f-cal (average) | 09:45 | **0.007846** | 0.46 / 0.83 | 1.08 | 90.3 / 80.7 | −0.29 | excellent |
| R6p-cal | 09:30 | 0.007967 | 0.16 / 0.40 | 1.30 | 90.0 / 80.0 | +0.16 | excellent |
| R6p | 09:30 | 0.007965 | 0.91 / 1.51 | 1.58 | 90.6 / 80.9 | −0.84 | good |
| R4p-cal | 09:30 | 0.007971 | 0.89 / 1.94 | 1.59 | 91.0 / 82.1 | +0.70 | good |

- **Recalibration removes most of the calibration error at no pinball cost** (R7f: 0.78 → 0.10
  pts). R7f-cal is the best-calibrated model so far. Averaging R7f with R4f lowers the loss 0.7%
  but stays worse calibrated even after recal, and it needs a TFT: not chosen.
- **R5 is running** (queued 14:10 PT, `runs/queue-r5.sh`, log `runs/queue-r5.log`): one HGB for
  09:30 and 09:45 (`--premarket optional`), + the open for every stock, past volatility, 600
  iterations for the tails; folds 0–14; then `-cal` for both states, the calibration table
  against R7f-cal/R6f-cal/R6p-cal, and R8 (screener) on R5~px-cal. Then the NAS backup.

**Per forecast** (the pre-market bake-off, 7 odd folds; Q12; pinball first, before Q35)

| forecast | candidates | chosen |
| --- | --- | --- |
| night (no pre-market) | R1c; R4's night state; R7 (HGB) | **R7 leads** (plain HGB, folds 0–14): 1.3% better than R1c (CI clear), on earnings and other days; R4~night ties R1c |
| 09:30 (prices) | R4 `~px`, R4p, R6p | **R6p leads** (overlay on R1c): 1.05% better than R4~px, 0.35% better than R4p, both CIs clear of 0 |
| 09:45 (full pre-market) | R4, R4f, R6f, R7f | **R6f and R7f tie** (R7f 0.06% better, CI spans 0): both beat R4f (0.55%, 0.76%) and R4 (1.55%, 1.6%), CIs clear. On a tie fewer models wins: R7f is one model, R6f is R1c + an overlay |

**Finished runs** (pooled over test folds 0–14 unless stated)

| run | what | vs R0 | verdict |
| --- | --- | --- | --- |
| R0 | 5-year baseline | — (skill 26.6%; −6.2% vs production) | superseded by R1c |
| R1 | + market, macro, session shape, after-hours | +2.5% worse | rejected: date fingerprints |
| R1a | R1 − macro (folds 0–6) | worse | rejected |
| R1c | + session shape, after-hours | −0.58% better | **base** |
| R6c | event band overlay on R0 (folds 1–14) | no gain | kept as a tool |
| R4 | A: one masked TFT (night / 09:30 / 09:45), 7 odd folds | night ties R1c; 09:30 and 09:45 lose to the overlays | not chosen for any forecast so far |
| R4p | B: TFT with prices always on, 7 odd folds | 0.70% better than R4~px, 0.35% worse than R6p | second at 09:30 |
| R4f | B: TFT with prices and full pre-market always on, 7 odd folds | 1.0% better than R4, 0.55% worse than R6f (CIs clear) | second at 09:45 |
| R6p, R6f | C: HGB overlays on R1c (prices; everything), 7 odd folds | see per forecast | **leading** at 09:30; R6f ties R7f at 09:45 |
| R7 | plain HGB per quantile, R1c's inputs, no TFT | 1.3% better than R1c (CI clear); skill 28.0%, calibration good (0.80 pts) | **leads at night**; gains largest in early folds |
| R7f | R7 + full pre-market always on, 7 odd folds | ties R6f; 0.76% better than R4f, 1.6% better than R4 | **tied lead at 09:45** |

**Running and queued:** R5 → R5-cal, R5~px-cal → calibration tables → R8 → NAS backup
(`runs/queue-r5.sh`, 14:10 PT). R5 replaces the planned R7p: one model covers 09:30 too.

**Run labels:** every run directory has a `MODEL.md` card (label R0, R1c, …, forecast, one-line
summary, purpose, result). Every trainer writes it (`--label R7f --about "…" --purpose "…"`;
tft-vwap, -gbm, -overlay, -event-overlay, tft-intraday, screener-train): at the start, and with
the pooled result at the end. `ml/scripts_model_cards.py` holds the hand-written verdicts of the
runs so far; re-run it when a verdict changes. The dashboard shows the label and the card.
R2 is phase 2.

**Accuracy per model** (skill, direction, calibration grade, legend):
[docs/STATUS.md § Model performance](../docs/STATUS.md#model-performance). To see it live, open the
dashboard's **Compare runs** tab (Q30).

**Known deviations:** R4 (started before the change) still has the constant `issue_type` input.
Every TFT run started from 10-28 22:30 drops it automatically (Q27): R4p, R4f, R5 on. So R4 vs
R4p/R4f also differs in that input, expected to be inside the noise (Q23).

## Current position (2026-10-29, 14:30 PT)

**Direction (user, 2026-10-29, Q35):** one daily model for every forecast at and after 09:30,
using full pre-market; **no night forecast**; quantile calibration (the dashboard's q err, grade,
coverage) ranks first, pinball loss second. Candidate: R5 (one HGB, 09:30 + 09:45) with
walk-forward recalibration.

**Design**
- **Two models, trained separately.**
  - The **daily** model forecasts tomorrow's VWAP move as 19 quantiles (P5..P95).
  - The **intraday** model forecasts moves over the next minutes and takes the daily forecast as an input. This is already wired in [daily_join.py](tft_intraday/daily_join.py).
- **Retraining, fine-tuning and a calibration layer are allowed.** The earlier "frozen, report-only" rule applied only to the first analysis pass.

**Before comparing any models**
- **Add honest baselines to the daily report.** The daily "naive" score ([evaluate.py:63](tft_vwap/evaluate.py#L63)) is a forecast with zero width at a 0% move, so the 21–25% "skill" is measured against a strawman.
- Intraday already has a proper baseline (climatology, in [baselines.py](tft_intraday/baselines.py)).

**Contenders.** Each one uses the same folds, report and dashboard.
- **TFT**: the current model.
- **LightGBM (quantile)**: fast to train, so it's the place to try new features.
- **Chronos-2**: pretrained. Lower priority for the daily model, more promising for intraday.
- **Dropped:** Kronos and PatchTST, because they can't take news or earnings inputs.
- **AutoGluon-TimeSeries** (free) can train all of the above from one table. It's optional.

**Data for the 5-year retrain**
- 5 years of Massive bars. Pick the stock universe per date, including stocks that later delisted.
  - First load the NAS files (5 years, extended hours) into this Mac's DuckDB. It still ends 2026-10-06.
  - Then give the 2,557 stopped tickers with no issue type one. Until then the export drops them, and the panel over-represents survivors ([FEATURES.md](FEATURES.md) section 15).
- **News history:** backfill from Massive `/v2/reference/news`, which goes back to 2016 on paid plans (2 years on Basic) with per-ticker sentiment. Score every year with one method.
- **Event types:** all EDGAR 8-K items (not only 2.02), plus Benzinga earnings vs estimates, rating changes and guidance, if the plan includes partner data.
- **Market context:** SPY, QQQ and sector-ETF returns, plus a volatility index.

**Intraday design direction (2026-10-29, Q31–Q34; proposals for the intraday deep dive, not built)**
- 1-minute grid from the 5-year `bars_1m`, empty minutes filled; 60-minute encoder plus day-so-far fields; training decisions every 5 minutes per stock.
- Volume relative to the time of day (`vol_rel`), price moves in the stock's volatility (`ret_z`), effort vs result (`impact`), regime summaries (quiet / normal / burst, minutes since the last burst), and multi-scale windows (Q33).
- Per-second detail as within-minute summary columns from the 1-second bars (2 years), tested as an ablation.
- The handoff rebuilt from the base daily run; the 09:30/09:45 forecasts exist only on the odd folds until the overlays run on all 15 (Q31).
- Baseline: climatology scaled by each stock's volatility. Screen feature groups on HGB first, then TFT. A streaming dataset for 5 years.

**Jumps and news**
- Keep real jumps in training. Remove only data errors such as unadjusted splits and vendor glitches.
- When ready, model jumps explicitly: a jump probability, or a two-part forecast (normal band + jump chance + jump size). See Q7 and Q8.
- While the news history is short, "no news collected" must not look like "no news happened". Use a coverage flag, or a small news adjuster on top of the 5-year base model (Q6).

**Open decisions**
1. **Forecast cutoff:** after the close (16:30), after midnight (the current routine), or pre-open (~09:15). This decides which news and prices the daily model can use (Q3).
2. ~~**News backfill:** Massive history (and which sentiment score), or only the news collected from now on.~~ User, 10-28: not important now.
3. ~~**Benzinga partner data:** whether the Massive plan includes it.~~ User, 10-28: not important now.
4. **Production run:** keep `20261005-025909Z-return-3712sym --fold 3`, or switch. The 10-11 retrain `20261011-173109Z-return-3785sym` tests 2.7% better; the 5-year **R0** `20261028-030847Z-return-4438sym` tests **6.2% better** on 756k shared forecasts, CI clear of 0 in every shared fold (Q17). R1c `20261028-233305Z` beats R0 by a further 0.58% (Q21). **User, 10-28: production switches only after all models (R4–R8) are evaluated.**
5. ~~**Resuming a stopped training run (`--resume`):** not built.~~ Done 2026-10-27 (Q13): runs pause and resume, and `tft-vwap-status` shows the time left.
6. ~~**Pre-market approach per forecast** (night, 09:30, 09:45).~~ Bake-off done 10-29 (Scoreboard); then the user chose one model for 09:30 and after, no night (Q35): R5.

**5-year retrain (decided 2026-10-27)**
- **Testing:** 15 walk-forward folds of 63 sessions (2023-01-20 → 2026-10-26), 21-session early stop, at least 252 training sessions (Q11). Bake-off runs use the 7 odd folds 1, 3, …, 13.
- **R0 done (2026-10-28):** skill 26.6% vs naive, P5–P95 88.9%, steady 24–28% in every fold; beats production by 6.2% (Q17).
- **R1 lost to R0** (+2.5%, worse in 7 of 15 folds); so did R1a (R1 − macro, stopped after 7 folds). Day-level inputs (market, sectors, macro) let the model memorise dates. **R1c (R0 + after-hours + session shape) beats R0:** −0.58%, CI clear of 0, 17× the gain per forecast on earnings days, worse in no fold. **R1c is the base** for R4 onward. Further inputs: one group at a time on the base, screened on the HGB first, market only per stock (Q21).
- **R2 (fundamentals, Q18) is phase 2:** it needs an EPS fetch with dates. Market and macro move to an overlay on the base forecast, a band-width fraction active only on event days (R6c, Q22). **R6c with yesterday's market, sector and yield moves gave no gain on R0** (Q24); next is the release calendar (CPI, jobs, FOMC days), a new NAS source, not built yet.
- **Inputs:** ~~all always-available groups in one run, R1 (Q16)~~; superseded by Q21: one group per run. Pre-market in two groups with their own flags: prices (09:30) and full (09:45, Massive) (Q14).
- **Three daily forecasts from the same inputs:** night (no pre-market), 09:30 (prices), 09:45 (full pre-market).
- **A 09:30 screener** ranks the liquid stocks by expected range from the open (Q15).
- **Baselines:** a plain HGB per quantile (R7). AutoGluon and other model types are phase 2 ([docs/TODO.md](../docs/TODO.md)).
- The runs, R0–R8, and their results: [docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-ml-daily`.

## Glossary

| Term | Meaning here |
|---|---|
| Quantile (P10, P90 …) | "P90 = +2%" means the model gives a 90% chance that the move is at most +2% |
| Calibration | Whether the stated probabilities come true as often as stated (P90 holds 90% of the time) |
| Pinball (quantile) loss | The score for quantile forecasts; lower is better. A big miss counts in proportion to its size, not squared |
| Baseline | A simple rule with no learning, e.g. "tomorrow's spread looks like this stock's last 60 days". Every model must beat it |
| Skill | The % improvement in pinball loss over a baseline |
| Walk-forward folds | Train on the past, then test on the next block of sessions the model never saw, and move forward |
| Early stopping | Stop training once a held-out block of sessions stops improving |
| Overfitting ("overtraining") | Memorising the training days instead of learning patterns. The model looks good on days it trained on and worse on new days |
| Ablation | Train with and without one group of inputs, then compare on the same test days |
| Embedding | The few learned numbers per stock (or industry) that the model uses as that stock's ID |
| Point in time | Every input holds only what was known at forecast time |
| Train/serve mismatch | An input means something different in training than it does live |
| Step / grid | The spacing of a series (15 s, 1 minute). The TFT assumes every step is the same length |
| Encoder (look-back) | How many past steps the model reads for one forecast |
| Decision sampling | Which steps become training samples; fewer samples, same grid |
| Relative volume (`vol_rel`) | Volume divided by what is normal for that stock at that time of day |
| Regime feature | A summary of recent activity, e.g. "quiet for 20 min since a burst" |
| Volume bars | Steps of equal volume instead of equal time |
| Multi-scale windows | The same measure over several look-backs (1, 5, 15, 60 min) at every step |
| Model type vs library | TFT, LightGBM, DeepAR and Chronos-2 are model types. pytorch-forecasting and AutoGluon are libraries that train them |

## Q&A log

### Q1 (2026-10-14): What improvements does an ML engineer typically make? Is adjusting weights a good way? Where does the manual effort go?

**Weights are not adjusted by hand.** Gradient descent sets them.
- Engineers tune **hyperparameters** instead: in [config.py](tft_vwap/config.py), `hidden_size`, `dropout`, `learning_rate`, `encoder_length`, epochs and batches.
- They also use **sample and loss weights**. The 1/price loss weight was one; weighting recent sessions more is another.
- **Fine-tuning** means continuing to train an already-trained model on new data.

The levers, from biggest typical payoff to smallest:
1. Data quality.
2. Point-in-time correctness (no leakage).
3. Features. Pit earnings and series breaks gave −2.7% pinball loss.
4. Target and loss design.
5. Ensembling.
6. Hyperparameters, usually worth 1–3%.
7. Swapping the architecture.
8. Calibration.

One hyperparameter to check: `limit_train_batches=500 × 512` means each epoch sees about 256k of the ~1.9M rows. The model may be undertrained, and the learning curves would show it.

Where the effort goes, roughly:
- data cleaning, 40–50%
- leakage and train/serve consistency, ~15%
- evaluation design, 15–20%
- interpreting results, ~10%
- hyperparameter tuning, 5–10% (the most automatable part)

### Q2 (2026-10-14): Is there a better fit than TFT (Chronos and others)? Explore every alternative.

**What in the target can be predicted:** next-day VWAP move = location + width × shape.
- **Location:** mostly mechanical. `close_position_pct` = (close − VWAP)/VWAP, and tomorrow's VWAP starts near today's close. This likely explains most of the 59–60% direction accuracy.
- **Width:** very predictable. Volatility clusters, each stock has its own level, and earnings days are wider. Most of the gain lives here.
- **Shape:** fat tails and earnings jumps. There are only about 8 earnings days per stock in 2 years, so this has to be learned pooled across stocks.

| Family | Examples | Verdict |
|---|---|---|
| Statistical baselines | Trailing empirical quantiles; EWMA/GARCH volatility × pooled residual quantiles | Required as the yardstick |
| Gradient-boosted trees | LightGBM / CatBoost quantile, LightGBMLSS | The main model for feature work |
| Deep supervised | TFT, DeepAR, TiDE, TSMixer, N-HiTS, PatchTST | TFT is fine; the siblings are unlikely to differ much |
| General pretrained | Chronos-2, TimesFM 2.5/3.0, Moirai 2.0, TiRex, Toto | Try Chronos-2 once. Research finds gains on returns small (arXiv 2606.27100, 2511.18578) and better on volatility (2505.11163) |
| Finance pretrained | Kronos, KiT | Weaker fit: no extra inputs, sampling-based spread |
| Excluded | TimeGPT (hosted), LLM forecasters, ranking models | Data leaves the machine / wrong output |

The run-file contract makes trying models cheap. Each candidate writes `fold_*/test_predictions.parquet` + `config.json` on the same folds, and the report and dashboard grade it unchanged.

### Q3 (2026-10-14): Does the advice change with 5 years of paid data, news, and no frozen model?

**The ranking shifts:**
- LightGBM stays the place to try new features, in its quantile variant. News days have shapes a single Student-t can't fit.
- TFT moves up as the final candidate: 2.5× the data and several market regimes.
- Chronos-2 drops in priority for the daily model.
- Kronos drops out, since it can't take news.

**The main point is timing.** Next-day jumps mostly come from information that arrives *after* the close: after-close earnings, evening news, after-hours and pre-market trading. News from before the close is largely already in the closing price, so with features cut at the close, news mostly widens the band and says little about direction.

The daily routine already runs after midnight ET. At that point the after-close 8-Ks, the after-hours price (Massive daily ticker summary) and the evening news are all known, but the panel ignores them. A later cutoff is probably worth more than any model swap. Whatever the cutoff, every input must be filed by its timestamp relative to it.

**Checks before training on 5 years:**
1. **Sentiment history.** Local news only covers 2026-07-27 → 08-26 (Google, 5,131 stories) and 10-11 → 10-14 (Yahoo, 384).
2. **Event type** matters more than positive/negative sentiment.
3. **Market context** is needed to tell market regimes apart.
4. **Survivorship.** Include stocks that were later delisted.

The calibration layer and retraining schedule are now allowed. Show raw and calibrated grades side by side, so the model's own bias stays visible.

### Q4 (2026-10-14): Is trelis.com/ADVANCED-time-series a good or better alternative?

**No.** It's a paid, gated GitHub repo with scripts for PatchTST and Chronos (forecasting, evaluation, full or LoRA fine-tuning) and no finance content.
- PatchTST forecasts each series alone, so it can't use earnings or news.
- It has no TFT, LightGBM, baselines, walk-forward testing or calibration grading.

**Use the free AutoGluon-TimeSeries (v1.6.1, Apache-2.0) instead.** It has:
- baselines
- `DirectTabular` (LightGBM on quantile loss)
- TFT, DeepAR, TiDE and PatchTST
- `Chronos2`, zero-shot or LoRA fine-tuned, with known-ahead and past inputs
- a weighted ensemble

If you use it, run it inside our fold loop and have it write our run files, so our report stays the judge.

### Q5 (2026-10-14): Two models, daily then intraday, with the daily quantiles as intraday input. Are AutoGluon, LightGBM and DeepAR alternatives to TFT, or baselines?

**The two-model design is sound, and it's already built.** [daily_join.py](tft_intraday/daily_join.py) gives the intraday model:
- the daily forecast's centre (`d_med`)
- its width (`d_width`, P10–P90)
- its lean (`d_skew`, `d_p_up`)
- the daily model's recent record on that stock (`hist_mae`, `hist_cov80`)
- the sector's average forecast (`sec_d_med`)

Two rules to keep:
- **The intraday model only sees daily forecasts the daily model made without having trained on that day.** The code enforces this by reading only the fold whose test window contains the day. Once retraining starts, keep each live daily forecast (`forecast_<date>.parquet`) and use those. Never re-predict history with a model that trained on it, or the intraday model learns to trust a daily forecast that looks better than it really is.
- **Retrain or re-check the intraday model whenever the daily model changes.** A new daily model can produce narrower or wider bands, so the same input column changes meaning. Keep the two runs paired.

What the names are:

| Name | What it is | Role |
|---|---|---|
| Baselines (climatology, trailing quantiles) | Simple rules, no learning | The yardstick. Not an alternative |
| **TFT** | A neural network that reads each stock's recent days and outputs quantiles. What you have now | Contender |
| **LightGBM** | Hundreds of small decision trees ("if earnings tomorrow and 20-day volatility > 3% …"), each one correcting the previous ones' errors. It reads a table, not a sequence, so you hand it summary columns such as the last 5 days' return. Trains in minutes | Contender |
| **DeepAR** | An older, simpler neural network than TFT. It reads the history day by day and outputs the centre and width of a distribution | Contender (low priority) |
| **Chronos-2** | A neural network Amazon already trained on many time series. You give it history and it forecasts without training | Contender |
| **AutoGluon** | Not a model. A toolbox that trains many model types (including all of the above) from one table and compares them | Optional tool |
| pytorch-forecasting | The library your TFT comes from | Current tool |

So a "comparison" means the baselines vs TFT vs LightGBM (and maybe Chronos-2), on the same test days with the same report. Do it once for the daily model and once for the intraday model.

### Q6 (2026-10-14): How do I avoid overtraining? News history may be only a few months, and I worry jumps without visible news will cause overfitting.

**Overfitting** shows up as good scores on the days the model trained on and worse scores on new days.
- The main guard is already in place: walk-forward test folds the model never trains on, plus a separate early-stopping block.
- Watch three things: the gap between the early-stop score and the test score, test skill against the baselines, and whether per-stock test results are worse than early-stop results.

**Unexplained jumps mostly widen the band; they aren't memorised.**
- Quantile loss counts a 40% jump in proportion to its size, not squared. So it barely moves the median and mainly shapes the outer quantiles.
- That is the right outcome: it teaches "jumps this size happen this often for this kind of stock".
- The real overfitting risk is the model pinning jumps on the wrong input. Mostly that's the per-stock **embedding**: 16 numbers per stock that can memorise "this stock jumped last March".
- Controls: a smaller symbol embedding (e.g. 16 → 8), dropout, early stopping, fewer epochs.

**The bigger risk with short news history is "missing" vs "zero".**
- If the news columns are 0 for the 4¾ years you weren't collecting, the model learns that "no news" includes days that actually had news. Live, "no news" means none happened. The same input then means different things in training and live, which is a train/serve mismatch.
- Yahoo coverage is also uneven per stock: topic pages plus a combined most-active stream, capped at 200 items. So a small-cap with no Yahoo story may still have had news, even live.

Fixes, best first:
1. **Backfill the news history.** Massive `/v2/reference/news` covers 2016 onward on paid plans (2 years on Basic), queried per ticker, with sentiment. That removes the problem. Score all of it with one method, either Massive's sentiment everywhere or your FinBERT on titles and descriptions everywhere.
2. **If you don't backfill, add a coverage flag.** `news_covered` = 1 on days and for stocks you were collecting, so 0 means "unknown", not "none". Alternatively, train the news part only on covered days.
3. **Two stages.**
   - Train the **base model** on 5 years without news.
   - Train a small **news adjuster** on the covered months only. It takes the base quantiles plus the news inputs and widens or shifts the band.
   - A small model (e.g. LightGBM with depth 3, or a quantile regression on a handful of inputs) has few knobs, so it's hard to overfit.
   - Train it only on base-model forecasts for days the base model did not train on, as in Q5, or it learns to correct a forecast that looks better than it really is.
   - Retrain it weekly as news accumulates. Once there's about a year of news, fold the news inputs into the base model.

**Always run the ablation.** Score the model with and without the news inputs on the same test days. The news inputs must improve news days without hurting other days, or they don't belong.

### Q7 (2026-10-14): Can the model infer that a jump is external (news-driven)? Do intraday predictions need more than quantiles?

**A model can only use what it sees at forecast time.** It can't conclude "this was news" about news it never saw. What it can learn is:
- "jumps happen at rate X for this kind of stock and day", which is the width of the tails
- "when there's fresh news, the rate is higher", which is the news inputs at work

That is the realistic form of "inferring external factors".

**Labelling jumps after the fact is a separate job: a report, not the model.**
- Join each big miss (actual outside P5–P95) with the news and 8-Ks in the hours before it.
- Label it as news / earnings / market-wide / unexplained (often a data glitch).
- This shows how much of the tail error news could fix, and it flags bad data to clean. It would make a good dashboard panel.

**Extra outputs beyond the quantiles. Yes, two are worth adding:**
1. **Jump probability**, P(|move| > k × the stock's normal range), from a small separate classifier or an extra output head.
2. **A two-part forecast**: a normal-day band, plus the chance of a jump, plus the jump size band. For example: "normal ±1.2%; 6% chance of a jump; if it happens, typically ±7%".
   - The 19 quantiles can be computed from these parts, so the dashboard contract stays the same.
   - It reads more clearly and keeps jumps from distorting the normal-day band (see Q8).

**For intraday, add news-timing inputs:** minutes since the stock's last story, the number of stories in the last N minutes, and the daily model's jump probability for the day. Outputs: per-horizon quantiles (already there), P(jump in the next N minutes), and the expected range.

**Feed latency limits all of this.** Prices react to news within minutes. Massive's news endpoint is updated hourly, and the Yahoo scraper only runs when started (scheduling isn't set up yet), so both are too slow for live intraday inputs. Intraday news needs a near-real-time feed (e.g. Benzinga live news through Massive) or very frequent scraping.

### Q8 (2026-10-14): Should the model treat jumps beyond its forecast as external and drop them, to avoid overfitting? And train on the 2–3 months that have news?

**The instinct is right:** unexplained jumps shouldn't distort the normal-day forecast. But both methods would backfire.

**Dropping jumps from training** makes the bands too narrow.
- Live jumps still happen, so P5/P95 and "P(move ≥ 3%)" become overconfident. The per-stock grading would flag exactly that.
- The goal is move probabilities, so jump frequency has to stay in the data.

Instead:
- **Drop only data errors.** PSKY's unadjusted split, the XRPN glitch and AKTS are fake jumps. Handle them with the series-break logic plus a glitch check.
- **Keep real jumps.** Quantile loss already limits how hard they pull.
- **For a clean separation, use the two-part forecast from Q7.** Jump days count toward the jump part instead of widening the normal-day band. This is the principled version of "dropping it out": the jump doesn't distort the normal band, but it still counts in the overall probabilities.

**Training the main model on only 2–3 months would overfit much more than 5 years would:**
- it covers one market regime
- each stock has at most one earnings day in it
- each stock has only a few jumps
- with a 60-session look-back window, little history sits behind each sample

Use those months only for the small news adjuster from Q6. Grow it as news accumulates, and always test it on the newest weeks, which it hasn't trained on.

**Resulting design:**

```
5-year base model (prices, earnings, market context; news if backfilled)
        │ quantiles
        ▼
news adjuster (small, trained on the news-covered months, retrained weekly)  ──►  daily forecast
        │                                                       + jump probability (later: two-part forecast)
        ▼
intraday model (minute-level inputs + daily forecast + news timing)  ──►  per-horizon quantiles + P(jump next N min)
```

### Q9 (2026-10-25): Is there a clean writeup of the fields used in daily training? I want to improve the inputs and add or change some.

**Yes: [FEATURES.md](FEATURES.md).**
- Sections 1–11 list every field by its TFT role: what is predicted, the inputs, and what is kept out.
- Sections 12–16 (added 2026-10-25) cover the training and testing exercise:
  - how runs are trained and tested
  - the results so far
  - every training parameter
  - candidate inputs
  - a step-by-step recipe for adding an input and testing it

The 10-11 comparison is also in [docs/LEARNINGS.md](../docs/LEARNINGS.md) (`tag-ml-daily`).

Where to start, in order:
1. **Earnings days are the biggest gap.** The 90% band holds on only 49% of them. The cheapest fixes use data already collected:
   - a "results already out" flag for releases after the close, from EDGAR acceptance times
   - the after-hours move, once the extended-hours files are loaded
2. **Market context.** SPY, QQQ, IWM, the sector ETFs and VXX are already in `daily_bars`.
3. **Two data gaps to close before the 5-year retrain:**
   - load the NAS files into the DuckDB
   - give an issue type to the 2,557 stopped tickers that the export drops (survivorship)
4. **Make small gains trustworthy.** Add an honest baseline. Change one thing per run, and train each configuration with a second seed.

### Q10 (2026-10-25): I want to plan daily training refinement in a new chat. What context does it need, so it can be concise without losing details?

**The docs now hold everything the earlier chats found.** The facts that lived only in chat notes were moved in on 2026-10-25:
- [FEATURES.md](FEATURES.md) section 12: pitfalls (the `export-tft --help` trap, the sklearn warning that makes runs take 13 h, and others).
- FEATURES.md section 17: the production run, the daily routine, which runs to keep (the intraday model reads the 10-11 run's fold 1), known data issues and standing decisions.
- FEATURES.md section 5: `is_10k` / `is_10q` have been stale since 2026-08-24. EDGAR has the same forms and is current.
- [docs/LEARNINGS.md](../docs/LEARNINGS.md) (`tag-ml-daily`): the fold 0 calibration (2026-10-04) and the first live grades (10-05, 10-06).

**Give the new chat, in this order:**
1. **FEATURES.md**: the core. Every input by role, how runs are trained and tested, results, parameters, candidate inputs, the recipe for adding one, the current state.
2. **This file**: the current position, open decisions, glossary and Q1–Q10.
3. **docs/LEARNINGS.md, the `tag-ml-daily` topic only**: the measured results with examples.
4. [ml/README.md](README.md): commands, outputs, predict and grade.
5. Only if needed:
   - [docs/STATUS.md](../docs/STATUS.md): hosts, schedules and where the data is
   - [docs/TODO.md](../docs/TODO.md): the NAS load and the extended-hours item
   - [dashboard/README.md](../dashboard/README.md): the run-file contract any new model must write

Not needed: FEATURES-intraday.md, and the original plans (docs/stock-prediction-plan.md, docs/next-session-parquet-tft.md), whose content is in FEATURES.md now.

### Q11 (2026-10-27): With 5 years of data, how should training and testing be split?

**Walk-forward quarters across all 5 years, not only the last year.**
- 16 test blocks of 63 sessions each, about 2022-11 → 2026-10. They span the 2022 bear market,
  the 2023 rally, the April 2025 sell-off and 2026, so a gain must hold in several regimes.
  **(As built, 10-28: 15 blocks, 2023-01-20 → 2026-10-26; only 15 fit on the real panel. Fold 0
  trains on about 1.2 years, fold 14 on about 4.7; the odd folds are 1, 3, …, 13.)**
- Each block has a 21-session early-stop block in front of it. Each fold trains on everything
  before that: fold 0 on about 1 year, fold 15 on about 4.6 years
  (`--fold-len 63 --early-stop-len 21 --min-train-sessions 252`).
- Comparisons of input groups (ablations) use the odd folds (as built: 7), to halve the cost.
- **Inside training, batches mix all training quarters at random**, as before. Each (stock, day)
  is now one sample (`dedupe_train`): before, the last training day of each stock appeared in
  about 41 samples, 16% of the set.
- Scores are paired per (symbol, date) with a bootstrap interval over test days, and shown per
  fold, so a win in one regime can't hide a loss in another.

### Q12 (2026-10-27): One TFT with masked pre-market, separate models, or an overlay? Does masking cost accuracy?

**Measured, not assumed: three approaches compete, judged per forecast.**
- **Why masking might cost:** one model spreads its capacity over three cases (no pre-market,
  prices, everything), and each case sees fewer samples with its inputs on.
- **Why it might not:** the cases share almost everything (the stock's history, market, macro);
  hiding inputs works like dropout; and a pure pre-market model has never seen a failed morning,
  so it needs the night model as a fallback anyway.
- **The contenders:**
  - **A. R4**, one TFT: full pre-market off for 50% of training samples, prices off for 25%, full
    on only when prices are on. One checkpoint gives night, 09:30 and 09:45.
  - **B.** R1 for the night, **R4p** (prices always on) for 09:30, **R4f** (everything on) for 09:45.
  - **C.** R1 plus **overlays** R6p/R6f: gradient-boosted trees per quantile that adjust R1's
    quantiles with the pre-market inputs. They train only on R1's out-of-fold test forecasts
    (the TFT's training-day outputs are overconfident), for fold k only on folds before k.
- **Judged** on the odd folds (as built: 7, base R1c), rows with pre-market data: night R4 vs R1; 09:30 R4 vs R4p vs
  R6p; 09:45 R4 vs R4f vs R6f. Lowest pinball wins when its interval is clear of the others;
  otherwise fewer models (A, then C, then B). A mixed answer is fine.
- **Also:** a plain HGB per quantile (R7) checks that the TFT is worth its cost. Cloning weights
  and fine-tuning with an added input was rejected: the variable-selection layers change shape.
- **Review of [docs/training-options-daily.md](../docs/training-options-daily.md):**
  - "A single TFT collapses when the feed drops": only if never trained with blank inputs; the
    masks and known flags handle it.
  - Raw OHLCV prices as inputs: wrong across stocks priced 1000× apart; ratios to VWAP instead.
  - 30 fundamental line items: E*TRADE has about 5 quarters and no filing dates, so they leak.
  - 3 quantiles instead of 19: no gain, and it breaks the dashboard.
  - XGBoost for fast ranking: right, for the 09:30 screener (Q15).

### Q13 (2026-10-27): Can a training run stop and resume? Will I see how long is left?

**Yes, both.**
- **Pause:** `touch runs/<id>/STOP` stops at the next epoch end, or `--max-hours H` stops on a
  time budget. At most the current epoch is lost. **Resume:** `tft-vwap --resume runs/<id>`.
  - Each fold keeps a `best` and a `resume` checkpoint (no `best-v1` shadowing), `state.json`
    records the run state, `test_end` is pinned, and a resumed fold reseeds with
    `seed + start_epoch`. A resume refuses if `panel.parquet` changed since the run started.
- **Time left:** a `[progress]` line each epoch, `runs/<id>/progress.json` every 30 s, and
  `tft-vwap-status [runs/<id> | runs]`. It shows fold and run time left as a range, because a
  fold ends either by early stopping or at `max_epochs`:
  - earliest: the fold early-stops as soon as it can, and every later fold after patience + 1 epochs;
  - estimate: the mean epochs and time of the folds already finished;
  - latest: every fold runs to `max_epochs`.
  Example: `fold 6 (7/16) epoch 6/15, early-stop wait 2/3, 14 min/epoch | fold left 14 min–2.1 h
  (est 47 min) | run left 8.6 h–33.6 h (est 20.4 h), ETA 03:40`.

### Q14 (2026-10-27): Where does today's pre-market come from, live and in training?

**Training uses Massive only. Live, prices come first (09:30), and the full pre-market at 09:45.**
- **Two input groups**, each with its own known flag:
  - **prices** (`pm_px_known`): the last pre-market trade before 09:30 and day d's open, each vs
    the previous VWAP;
  - **full** (`pm_full_known`): pre-market VWAP, volume vs the 10-day mean, and range, from
    Massive minute bars to 09:30.
  - Both are 0 when data is missing, the previous session is missing, or a split executes that day.
- **Why no live volume at 09:30:** no live source matches training for every stock. E*TRADE
  ChartIQ's minute buckets carry 0.48–0.66 of Massive's volume, varying by stock and day; the
  watchlist's `volume_cum` covers only ~1k stocks. Prices agree within ~20 bp everywhere
  ([docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-chartiq`).
- **Why 09:45:** Massive's delayed minute bars reach 09:30 by 09:45 for every stock, at 120
  calls/s, so the full pre-market is the same quantity as in training.
- **Live prices** (`[premarket] live_source = "etrade"`): the watchlist bid/ask mid where it
  covers the stock, else ChartIQ price-only calls, 20 symbols per call. When live Massive
  arrives, only this adapter changes. No ChartIQ history, no volume scale factors.
- **The cutoff is 09:30** for every model, in training and live; predict checks it.

### Q15 (2026-10-27): What does the 09:30 screener rank, and how is it tested?

**The biggest expected range from the open, top 20 of the liquid stocks.**
- **Target:** `max(high − open, open − low) / open` on day d (the low-to-high range is stored too).
- **Model:** sklearn HistGradientBoostingRegressor on log(range), plus a 0.8-quantile model for a
  "big range" score; trains in minutes, scores 200 stocks in under 50 ms.
- **Inputs (all known by 09:31):** the open vs previous close and VWAP; pre-market prices (and at
  09:45 the full group); the TFT's band width and median for d (**out-of-fold forecasts only**);
  yesterday's range, ATR(14)%, 20-day dollar volume; market and sector moves; earnings timing;
  day of week. The TFT's direction is shown as a lean, not used for ranking.
- **Universe:** average dollar volume ≥ $5M and price ≥ $3, picked each evening; rule filters
  (`[screener]`) are dashboard sliders.
- **Test (R8):** its top-20 realised range vs a random 20, the top 20 by ATR (naive), by TFT
  band width alone, and by pre-market volume alone. Kept only if it beats naive ATR with an
  interval clear of 0.

### Q16 (2026-10-27): Why test market, macro, after-hours and session shape in separate runs? Aren't all always-known inputs better together?

**They are now one run, R1.** The per-group runs were dropped (saves ~3 TFT runs, 6–10 h).
- These inputs are known every evening for every stock, so they need no masking, and the TFT's
  variable selection already learns to down-weight weak ones.
- **Separate runs would only answer "which group helps"**, which variable-selection weights answer
  more cheaply, per group and per fold.
- **The risk is not "more inputs = worse" in general but specific failure modes:** an input that
  lets the model memorise dates (macro has one value per day for every stock), or one that shifts
  meaning across regimes. So R1 is checked against R0 per fold, and **a drop-one run is made only
  on evidence**: R1 fails to beat R0, loses in some folds, or a group has near-zero weight.
- Pre-market is different: it is not always available, which is why it needs the masks and the
  bake-off (Q12).

### Q17 (2026-10-28): How did R0, the 5-year baseline, do per fold and against production?

**Better than production everywhere they overlap, and steady across 3.8 years of test days.**
- **Pooled** (3.17M forecasts, 4,218 stocks, 2023-01 → 2026-10): pinball 0.00788 vs naive
  0.01074, a 26.6% skill; P5–P95 holds 88.9% (aim 90%); the median gets the direction right
  59.1% of the time.
- **Per fold:** 24.4%–28.4% skill in all 15 folds, including the April 2025 sell-off (26.8%).
  The weakest are the 2023 folds, which train on only 1.2–1.7 years.
- **Against production** on the 756,726 forecasts both made (2025-09 → 2026-08): 6.2% lower
  pinball, 95% CI clear of 0 overall and in each of the five shared folds. On earnings days the
  gain per forecast is 2.6× the other days' (production has no earnings inputs). R0 is also
  better calibrated: P5–P95 88.3% vs 86.6%.
- **What a trained run is:** one `best.ckpt` per fold (weights, plus the scalers and encoders the
  data needs), `config.json` with the settings and panel, and the test predictions and metrics.
  `ml/runs/` is not in git (3 GB); runs are copied to the NAS (`PProjects/MTap/ml/runs/`).
- **Is the naive comparison fair?** The loss trains against the actual next-day VWAP move, not
  the naive forecast; "naive" (no move, zero width) is only the reference the skill is quoted
  against. That reference is a strawman, so the honest measures are R0 vs production and,
  later, R0 vs R1 and the HGB baseline R7.
- Details and tables: [docs/LEARNINGS.md](../docs/LEARNINGS.md) `tag-ml-daily`, "R0, the 5-year
  baseline".

### Q18 (2026-10-28): Are the fundamentals fractions? Absolute numbers don't compare across stocks. EPS matters most; how should it go in?

**Today no fundamentals go in at all, absolute or ratio.** The panel's stock inputs are already
scale-free (returns and ratios to VWAP; volume as log or vs its own average). E*TRADE's
fundamentals were left out because it has only ~5 recent quarters and no filing dates, so they
would leak (Q9, FEATURES.md).

**Proposal: a separate group, `--fundamentals` (R2, after R1), all price- or per-share-scaled
and point in time:**

| input | definition | why scale-free |
| --- | --- | --- |
| `eps_surprise_px` | (actual EPS − consensus EPS) ÷ price before the release | surprise per $ of price, comparable across a $5 and a $500 stock |
| `sue` | (actual − consensus) ÷ std of the last 8 surprises | standardised surprise, the classic post-earnings-drift signal |
| `earnings_yield` | trailing-12-month EPS ÷ price | the inverse P/E; handles negative EPS, unlike P/E |
| `eps_growth_px` | (TTM EPS − TTM EPS a year ago) ÷ price | growth without dividing by a near-zero EPS |
| `fundamentals_known` | 1 once the first release is known | so "no data" doesn't look like "zero surprise" |

- **Point in time:** each value starts on the session after its 8-K acceptance time (EDGAR,
  already used for the earnings windows), and carries forward to the next release.
- **Sources to check first:** Benzinga earnings (actual and consensus) through Massive, if the
  plan includes it; otherwise EDGAR XBRL `companyfacts` for reported EPS (no consensus, so
  `eps_surprise_px` and `sue` would use the change vs the same quarter last year).
- **Clipping:** each ratio clipped at ±1 (±100% of price) and z-scored on training rows, like
  the other groups.
- **Test:** R2 = R1 + `--fundamentals` on the odd folds, judged on earnings-window days and
  other days separately; kept only if the CI is clear of 0.

### Q19 (2026-10-28): Which fields from the daily forecast should go into intraday training?

**Already built** (branch `intraday-model`, [daily_join.py](tft_intraday/daily_join.py)). Eight
static inputs per (stock, day), each a log move from yesterday's VWAP: `has_daily`, `d_med`
(P50), `d_width` (P10–P90), `d_skew`, `d_p_up`, `hist_mae` and `hist_cov80` (the daily model's
record on that stock over the last 20 sessions), and `sec_d_med`. The 19 quantiles themselves
are not inputs; they are only the morning baseline for `r_close`. The 10-12 run read one fold of
the old 2-year daily run.

**Proposed set, best first:**

| field | definition | role | why |
| --- | --- | --- | --- |
| `d_pos` | (ln(price_t ÷ prev VWAP) − P50) ÷ P10–P90 width | observed, every step | where the price is now inside the day's band; the static forecast becomes a moving signal |
| `d_to_p90`, `d_to_p10` | ln(P90 ÷ price_t), ln(price_t ÷ P10) | observed | room left to the band's edges |
| `d_vwap_pos` | the same as `d_pos` for the running day VWAP (Massive WS `a`) | observed | the daily target is the day VWAP, so this compares like with like |
| `d_rest_med` | the rest-of-day VWAP the daily P50 implies: (P50 − f · VWAP so far) ÷ (1 − f), f = volume so far ÷ `vol_10d_mean`, clipped to [0, 0.9] | observed | the most direct link to `r_rod` and `r_close`; f is an estimate, so it is noisy early in the day and the clip matters |
| `d_med`, `d_width`, `d_skew`, `d_p_up` | as today | static | the centre, width and lean of the band |
| `d_width50`, `d_tail` | P75 − P25; (P95 − P5) ÷ (P75 − P25) | static | the core width and how heavy the tails are |
| `d_width_rel` | `d_width` ÷ the stock's mean `d_width` over the last 20 sessions | static | "an unusual day is expected" |
| `hist_mae`, `hist_cov80`, `hist_bias` | as today, plus the mean signed error | static | how far to trust the daily forecast for this stock |
| `sec_d_med`, `mkt_d_med`, `d_width_rank` | sector and market mean of `d_med`; the width's percentile across stocks that day | static | the day's context, and screener-like ranking |
| `d_state` | 0 night, 1 09:30 prices, 2 09:45 full, once the pre-market variants exist | known | which forecast is in force at time t |

**Not as inputs:**
- all 19 quantiles: they are near-collinear, and the summaries above carry the information;
- dollar levels;
- `actual`;
- any daily forecast for a day the daily model trained on.

**Rules:**
- **Out-of-fold only.** R0's 15 test folds together give exactly one forecast per (stock, day)
  from 2023-01-20 to 2026-10-26. **Built 2026-10-28:** `tft-vwap-handoff runs/<daily run>`
  writes them as `runs/<daily run>/handoff.parquet`, with every static field above (`--live`
  adds the production run's `forecast_<date>.parquet` after the last fold). The intraday side
  reads that table instead of one `daily_fold`, and computes the per-step fields (`d_pos`,
  `d_to_p90`/`p10`, `d_vwap_pos`, `d_rest_med`) itself. Contract: [docs/intraday-model.md](../docs/intraday-model.md) §1.
- **Point in time.** `d_state` and the 09:30 and 09:45 quantiles apply only from the step at
  which they exist live. Before that, the night forecast is in force.
- **Paired runs.** The intraday run records its daily run id (`daily_run`). Retrain or re-check
  the intraday model when the daily model changes (Q5).
- **Which daily run.** Use R0 until R1, R1a or R1c beats it, then the winner. (10-28: R1c beat it; the handoff table is still R0's until it is rebuilt from R1c.)
- **Test.** Run the intraday model with and without the daily group on the same days, per
  horizon. Expect the gain at 60 min, rest of day and close rather than at 1–5 min.

### Q20 (2026-10-28): Should the intraday model train on the daily model's training days and test on its test days, so the test data runs in parallel?

**Match the test days, yes. Match the training days, no.** Every daily forecast the intraday
model sees, on its training days as well as its test days, must come from a daily model that did
not train on that day.

- **Why not the daily training days.** On those days the daily forecast is better than it will
  ever be live: the median sits near the answer and the bands are narrow. An intraday model
  trained on them learns to trust the daily forecast too much, and leans on it wrongly on test
  days and live. The screener and the overlays follow the same rule.
- **Where the right inputs come from.** R0's 15 walk-forward test folds together give exactly one
  out-of-fold forecast per (stock, day), 2023-01-20 … 2026-10-26, each from a model that never
  saw that day. That one table feeds every intraday day.
- **Why align the test days.** If intraday test block k is daily test block k, both models are
  out of sample on the same days. The combined system (morning daily forecast plus intraday
  updates) can then be graded as it would run live, quarter by quarter.
- **Folds: the same 5 years as daily.** Massive 1-minute bars go back to 2021-10-29 (NAS
  `bars_1m/`, every common stock), so intraday can train from 2021-10-29 and test on all 15 daily
  test blocks (2023-01-20 … 2026-10-26). Each intraday fold trains on the intraday days before
  its own early-stop block. Finer sources are shorter: 1-second REST bars from 2024-10-21
  (daily folds 7–14), WebSocket and the E*TRADE feed only from 2026-10.
- **The first 14 months have no daily input.** No honest daily forecast exists before the first
  daily test block (2023-01-20): it would need a daily model that never saw that day, trained on
  less than a year. Those intraday days train with `has_daily` = 0, or only feed the earliest
  folds; the intraday revision decides.
- **What still shifts.** Later daily folds, and the live production model, are trained on more
  data, so their bands can be narrower. Relative fields (`d_pos`, `d_width_rel`, `hist_*`) keep
  the same meaning across them; re-check the intraday model when the daily model changes (Q5).
- **Scope.** The daily side of this (the handoff table) is done. The intraday targets, quantile
  strategy and training are revised in a separate session; Q19's per-step fields and these folds
  are inputs to it, not decisions.
- Details: [docs/intraday-model.md](../docs/intraday-model.md).

### Q21 (2026-10-28): R1 and R1a lost to R0. What should the remaining models (R2, R4–R8) take as inputs?

**What the runs showed** (LEARNINGS `tag-ml-daily`, sections R1 and R1a):

| run | inputs added to R0 | vs R0 (pinball, 95% CI) | P10–P90 coverage |
| --- | --- | --- | --- |
| R1 | market/sector, macro, after-hours, session shape | +2.5% worse (CI clear); worse in 7 of 15 folds, better in none | 75.7% (R0 78.9%) |
| R1a, folds 0–5 | R1 − macro | worse (+0.000232, CI clear); no clear difference from R1 | 74.9% (R0 79.6%) |
| R1c `20261028-233305Z` | R1 − macro − market (after-hours, session shape) | **better:** −0.58%, +0.000046 (+0.000021 … +0.000071); earnings days +0.000339 (+0.000287 … +0.000387); worse in no fold (result added 10-28 21:08) | 78.4% |

- **The inputs are correct;** the market index tracks SPY at 0.945 and the CPI gap is filled.
  The loss is in how the model uses them.
- **Both halves of the forecast got worse:**
  - The median is further from the answer, 130 vs 126 bp median error.
  - The bands are narrower, 4.9% vs 5.1% P10–P90, so they miss more often.
- **The loss is spread evenly.** R1 is worse on 61% of days, about the same on calm and big
  market days. So the market inputs don't even pay off on big market days.
- **The weight goes to day-level inputs.** 22 of R1's 31 new columns have one value per day,
  the same for every stock: the market return, the 11 sector returns and the 10 macro series.
  - Together they took a third to two-thirds of the encoder weight.
  - With macro removed (R1a), market alone took 34–49% in folds 0–4.
  - The TFT sees 60 sessions of them, which is a fingerprint of the dates. That lets it
    memorise training days rather than learn something that carries to new days.
- **Signs of that memorising:**
  - Every fold stops after 4–6 epochs, where R0 kept improving to 7–13 in the later folds.
  - Removing macro brought fold 0 (2023, after the 2022 rate shock in its training years) back
    level with R0, but no other fold.
- **After-hours** is per stock, so it isn't a date fingerprint, but its weight swings from 5% to
  46% between folds. `close_pct_rth_vwap` (where the close sits vs the VWAP) alone took 45% in
  R1 fold 5. R1c tests it.
- **Q16 was wrong to bundle.** It assumed inputs known every evening can only help. R1 shows they
  can hurt, and bundling four groups hid which one.

**Rules for the remaining runs:**
1. **The base is whichever of R0 and R1c wins.**
   - If R1c beats R0 with a CI clear of 0, R1c's inputs are the base.
   - If they tie overall but R1c is clearly better on earnings days and clearly worse in no fold,
     R1c is the base (user's choice, 2026-10-28: keeps the after-hours earnings reaction).
   - Otherwise on a tie, R0 is the base (fewer inputs).
   - If R1c loses, R0 is the base and after-hours and session shape are tested one at a time.
2. **One input group per run, against the base.** No more bundles.
3. **Screen a group on the plain HGB (R7) first.** It takes minutes; a TFT run takes about 4 h.
   Only a group that helps the HGB gets a TFT run.
4. **No shared one-per-day columns as TFT inputs.** Market context goes in per stock:
   - the stock's rolling 60-session beta to the market;
   - its excess move (own move − beta × market move);
   - its own sector's move (`own_sector_cap_ret`, ~12 values a day), not all 11 sectors.
   This is a new export step and is not built.
5. **Macro is out of the TFT.** Market and macro go into an overlay on the base forecast
   instead, active only on days with news (R6c, Q22).

**Inputs per remaining model:**

| model | inputs | change from the 10-27 plan |
| --- | --- | --- |
| R4, R4p, R4f (pre-market TFTs) | base + pre-market prices / full | base was R1. The pre-market inputs are per stock, so they stay. |
| R6p, R6f (overlays) | base run's out-of-fold forecasts + pre-market | base was R1; R0 already has out-of-fold forecasts for all 15 folds |
| R7 (HGB) | base inputs; a second variant with the per-stock market terms (rule 4) | was R1 inputs; now also the screen for every new group |
| R2 (fundamentals, Q18) | base + fundamentals, alone | **phase 2:** needs a fetch that isn't built (Q22) |
| R6c (new) | base out-of-fold forecasts + market/macro event inputs, a band-width fraction | Q22; built before R1m. Done on R0 with yesterday's moves: no gain (Q24) |
| R1m (new, optional) | base + per-stock market terms | needs the export step in rule 4; only if R6c shows the information helps |
| R8 (screener) | base out-of-fold forecasts + its own features; market only as own-sector and beta × market | a one-per-day value can't reorder stocks within a day; it only helps through interactions |
| R5 (final) | the winners above | |
| intraday | R0's handoff table until a daily run beats R0 | unchanged |

**Smaller points found along the way:**
- **`issue_type` is constant.** The panel is common stocks only, yet the input still gets 27–31%
  of the static weight. R0, R1, R1a and R1c all have it, so their comparisons are fair. It is a
  panel column that `config.py` lists as an input (correction 10-28: earlier written as "a
  setting, not a panel column"); drop it with the next base change (Q22).
- **Every 5-year run is a little narrow.** R0's P10–P90 holds 78.9% against 80%. A per-fold
  width scale fitted on the early-stop block is a cheap fix, to try in the overlays or R5.

### Q22 (2026-10-28): Does `issue_type` affect R1c? Is R2 phase 2? Can market and macro be an overlay that acts only on the days they change?

**1. `issue_type` and R1c.**
- **The comparison is fair.** R0, R1, R1a and R1c all use `issue_type`, and it is "CS" on every
  row. R1c vs R0 still measures only after-hours and session shape.
- **Whether it costs accuracy is not known.**
  - A constant input can't tell one stock from another. Its 27–31% of the static weight is a
    fixed vector added to every stock's context.
  - That may leave a little less room for `symbol` and `industry_code`, or it may just act as a
    bias. Measuring it would take a TFT run of its own, which is not worth it alone.
- **Dropping it needs no new export.** It is a panel column, constant "CS"; the model uses it
  because `STATIC_CATEGORICALS` in `config.py` lists it. (Corrected 10-28: this said "a setting,
  not a panel column".)
- **Plan:** drop it, via a flag, in the next TFT run that changes the base anyway (R4 or R5),
  and say so in that run's notes. Not now: R1c keeps running unchanged.

**2. R2 (fundamentals): yes, phase 2.** Q18 needs EPS with dates, and nothing fetched today has
both.
- E*TRADE has about 5 quarters and no filing dates, so it would leak.
- Benzinga earnings (actual and consensus) through Massive depend on the plan (open decision 3).
  No code fetches them.
- EDGAR XBRL `companyfacts` would give reported EPS, but has no consensus and isn't fetched.
- **R2 waits for one of these fetches.** It is not in the current series.

**3. Market and macro as an overlay, only when they change: yes, with one refinement.**
- **Why it fits better than TFT inputs:**
  - The overlay sees one row per (stock, day), with that day's values only, not a 60-session
    window. So there is no date fingerprint to memorise.
  - It trains on the base model's out-of-fold forecasts, like R6p/R6f.
  - It retrains in minutes. When the macro data or the event list changes, only the overlay is
    refitted, never the TFT.
- **The form: a width fraction, active only on event days.**
  - Output: a scale on the base band (P50 ± width × s), so s = 1.10 means a 10% wider band.
    Optionally also a small shift of the median.
  - Inputs are 0 on ordinary days, so s = 1 and the base forecast passes through unchanged.
  - Inputs:
    - the size of yesterday's market move beyond its normal range;
    - the size of the yield change;
    - flags for scheduled event days.
- **The refinement: use release dates, not value changes.**
  - The panel uses CPI from the first session of month M+2. That is 2–5 weeks after the real
    release, so the day the panel's value changes is not the day the market reacts.
  - The CPI, jobs-report and FOMC dates are published in advance, so a flag for "today is a
    release day" is known the night before. It can even be a known-future input.
  - **This needs a release calendar,** which isn't fetched: the BLS schedule and the FOMC
    calendar. Small, but new.
- **What R0 already does, and where there is room** (R0 test days, 2023-01 → 2026-10, P10–P90
  coverage, target 80%):

  | day type | days | coverage | other days | R0's band | other days |
  | --- | --- | --- | --- | --- | --- |
  | after a big market day (top 10% of yesterday's market move) | 95 | 77.8% | 79.0% | 5.64% | 5.07% |
  | after a big 10-year yield move (top 10%) | 82 | 78.6% | 78.9% | 5.26% | 5.11% |
  | first session using a new CPI value (M+2) | 44 | 75.2% | 79.1% | 5.27% | 5.12% |
  | **day of** a big market move (top 10%, not known the night before) | 95 | **64.7%** | 80.5% | 5.51% | 5.08% |

  - After a big market day R0 already widens its band by 11%, through the stocks' own moves.
    Coverage drops only 1 point, so there is little left to gain from yesterday's market move.
  - The misses are on the days the market itself moves a lot. Those aren't known the night
    before. Scheduled release days are the known stand-in, and the calendar is how to test it.
  - The new-CPI-value row is 44 days at the lagged date, so it is weak evidence either way.
- **How to judge it:** the same paired CI as the other runs, on all 15 folds (R0 has out-of-fold
  forecasts for each).
  - Event inputs have one value per day, so the real sample is the number of event days (tens),
    not rows. So: few inputs, a large minimum leaf size, and per-fold checks.
  - Kept only if the CI is clear of 0 and no fold gets clearly worse.
- **Order:**
  1. R6c with the inputs we have: yesterday's market and sector moves, yield changes.
  2. Add the release calendar.
  3. Only if R6c shows the information helps, try the per-stock market terms in the TFT (R1m).

### Q23 (2026-10-28): Should `issue_type` be removed from future training? Would it free up weight for more important inputs?

**Remove it, but expect a difference too small to measure.**
- **Why the weight looks large but means little.** The TFT's variable-selection weights are a
  softmax: per stock, they share 100% among the static inputs. A constant input becomes a fixed
  learned vector, the same for every stock. That works like a bias on the static context: it
  helps no stock more than another. Its 17–31% share is part of that bias, not attention taken
  from `symbol` or `industry_code`.
- **What removing it frees.** Almost nothing:
  - a 1-wide embedding and its small gated network, a few hundred of the model's parameters;
  - the bias it supplied is absorbed by the network's other bias terms.
  - The other static weights will rise in the report, but that is the softmax redistributing,
    not the model learning more about the stocks.
- **What it does gain:**
  - a cleaner importance report, where every static weight means something;
  - one less input that could interact oddly with the variable selection.
- **Why not a dedicated run:** the expected change is inside the noise of a paired CI, and it
  would cost a 4-hour run.
- **How:** drop any static categorical with a single value in the training rows, automatically,
  and record it in the run's config. So if ADRs return (`export-tft --issue-types CS,DR`),
  `issue_type` comes back as a real input. Apply it from the next run that changes the base
  anyway (R4 or R5), noted in that run's results.
- **Built 2026-10-28 22:30** (`drop_constant_statics`, Q27), after R4 had started: R4 keeps it;
  R4p, R4f and later runs drop it.

### Q24 (2026-10-28): R6c with the inputs we have: does it help? And where will the release-calendar fetch be?

**1. R6c with yesterday's market, sector and yield moves: no gain.** Details and tables are in
[docs/LEARNINGS.md](../docs/LEARNINGS.md), "R6c, a market/macro event overlay on R0".
- **What was built.** `tft-vwap-event-overlay` keeps the base median and scales the band on event
  days. An event is a day-level move beyond 2σ of its past 63 sessions. The scale is fitted per
  fold on the base's out-of-fold forecasts of earlier folds. On ordinary days the base passes
  through unchanged.
- **Result on R0, folds 1–14.** R0 is very slightly better (−0.000002, CI just clear of 0). The
  fitted coefficients flip sign between folds, because there are only 40–50 event days per
  series to learn from.
- **Why.** R0 already widens its band after big days through the stocks' own moves. The misses
  are on the days the market itself moves (Q22), and yesterday's data can't see those.
- **Kept as a tool.** It is not used in any forecast. It is the place where the release-calendar
  flags get tested (below), and it can be rerun on any base in seconds.
- **So step 3 of the Q22 order, R1m (per-stock market terms in the TFT), is not justified by
  this.** It stays optional, behind the calendar test.

**2. The release-calendar fetch: where it will be.** It is not built yet. The plan follows the
pattern of the existing economy sources:
- **Code:** a new source `python/sourcing_py/sources/release_calendar/`, next to
  `massive_treasury_yields` and `massive_inflation`. It reads:
  - the BLS release schedule (CPI, the jobs report, PPI), published a year ahead;
  - the Federal Reserve's FOMC meeting calendar (the statement day);
  - optionally the BEA schedule (GDP, PCE).
- **Where it runs:** on the NAS, in `daily` (`daily-capture.sh`), like the yields and inflation.
  It only needs to check monthly; it is cheap.
- **Output:** vintage files in `MTap-Data/out/release_calendar/`. Each file keeps the schedule as
  published that day, so a model never sees a date that wasn't announced yet.
- **Into the model:** `ingest-reference` loads it into DuckDB on this Mac. `derive-features`
  turns it into known-ahead flags per session: `is_cpi_day`, `is_jobs_day`, `is_fomc_day`.
  `export-tft` adds them to the panel. Since they're known the night before (and months
  ahead), they can be known-future inputs.
- **Test first in R6c:** add the flags as event features (the same overlay, now active on
  release days). Same judging: paired CI on all 15 folds, no fold clearly worse. The test
  folds span about 3.8 years: about 30 FOMC days (8 a year) and about 45 each of CPI and jobs
  days (12 a year). That is thin, but these are the days R0 actually misses.
- **Massive** has no release calendar endpoint in its economy API (only the values), so the
  source reads the agencies' own published schedules.

### Q25 (2026-10-28): Does the release calendar improve R6c? How long would building and testing it take?

**A pre-check, run before building anything.** It used release dates typed in by hand: FOMC and
jobs 2023–2026, CPI 2023–2025. The tables are in [docs/LEARNINGS.md](../docs/LEARNINGS.md),
"Release days in R0".
- **R0 does under-cover on release days.** P10–P90 coverage is 74–75% on CPI, jobs and
  day-after-FOMC days, against 79.5% on other days. Pinball loss is 9–16% higher on those days.
  They hold 20 of the 95 biggest market days.
- **One band scale per release type, fitted walk-forward, is stable.** From fold 4 on it is
  about +16% on jobs days, +10–15% on CPI days, +9–20% on the day after FOMC and about 0 on FOMC
  day itself (the statement comes at 14:00 ET). That is unlike the yesterday's-move features in
  Q24, which flipped sign.
- **It fixes coverage, but barely moves the loss.** On release days, folds 5–14:
  - jobs: coverage 74.6% → 80.2%, loss −0.3%;
  - CPI: coverage 75.7% → 79.6%, loss −0.15%.
  - Across all days that is +0.0000001 per day, CI (−0.0000029 … +0.0000034): no measurable
    change.
  - The misses on release days are big moves in either direction. A symmetric wider band buys
    coverage, but costs almost as much on the rows that didn't need it.
- **So:** the calendar would make the bands honest on about 10% of days, which helps the
  screener and the intraday handoff more than it helps the daily pinball score. It is not a
  model improvement on the daily score.

**What it would take:**

| option | work | what it gives |
| --- | --- | --- |
| **A. A checked-in dates file** (`ml/tft_vwap/release_days.csv`), read by `tft-vwap-event-overlay --calendar`. The agencies publish a year ahead, so it gets a yearly update by hand. | about 1–2 h with tests | the overlay above, usable live; no NAS change |
| **B. The full source** (Q24 §2): a BLS + Fed calendar fetcher, the NAS `daily` job, ingest, `derive-features` flags, then `export-tft`. | about 1 day of building, plus the NAS push and a check of the first live fetch. The BLS and Fed pages are HTML, with no API, so the parsers need tests on saved pages. | the flags in the panel, so a later TFT or HGB run (R5, R7) can use them as known-future inputs |

**Recommendation:** A, if calibrated bands on release days matter to the screener or intraday.
B only if a later run wants the flags as model inputs. Neither changes the R0 vs R1c decision.

### Q26 (2026-10-28): Could the jobs and CPI numbers be scored (FinBERT or similar) for a positive or negative impact? Would the calendar overlay apply to every future model?

**1. Scoring the release itself: the 09:30 pre-market price already does it.**
- **Timing.** CPI and the jobs report come out at 08:30 ET, an hour before the open. The night
  forecast is made before the release, so at that point there's nothing to score.
- **By 09:30 the market has reacted.** On R0's test days with a pre-market price (55% of rows):

  | days | corr(pre-market move, actual) | corr(R0 median, actual) | day-average corr(pre-market, actual) |
  | --- | --- | --- | --- |
  | CPI or jobs (75) | 0.61 | 0.27 | **0.83** |
  | other (870) | 0.56 | 0.26 | 0.66 |

  On release days the market-wide pre-market move tracks the day's move more closely than on
  other days. The reaction to the surprise is in the price by 09:30, and R4, R6p and R6f (queued)
  use that price.
- **Why a text score would add little:**
  - FinBERT reads tone, not surprise. What moves markets is the number against expectations,
    and that needs consensus estimates (Benzinga, plan-dependent, Q22).
  - The direction isn't fixed. A strong jobs report can push stocks down through rate
    expectations, so "positive" text doesn't mean "up".
  - There are only about 75 release days in the test folds, too few to learn a mapping from a
    score to a move.
- **Where it could help:** a surprise feature (actual − consensus, scaled) on the 09:30 and 09:45
  forecasts, if Benzinga's economic consensus becomes available. That is phase 2, next to R2.

**2. Does the overlay apply to every future model? Only where it is refitted and passes.**
- It sits on top of whichever base model is chosen (R0 now, R4 or R5 later). It learns its
  scale from that base's own out-of-fold forecasts, in seconds, so it is refitted per base, not
  carried over.
- **Kept only if it passes for that base:** a paired CI with no loss, and coverage better on
  release days.
- **It becomes unnecessary if** the base already handles release days. For the 09:30/09:45
  forecasts the pre-market price may do this (to be checked on R4's results). For the night
  forecast, the calendar flags would have to be a known-future input of the base.
- **It isn't in the forecast pipeline today.** Using it live adds one step after predict:
  `forecast_D.parquet` → overlay → the published forecast.

### Q27 (2026-10-28): R4 is running with `issue_type`. Can the next runs, or even R4's remaining folds and epochs, train without it?

**The next runs, yes (done). R4's remaining folds, no.**
- **R4 itself.** It is one process with its config fixed at start, and every fold of a run must
  use the same inputs, or its folds and states can't be pooled into one result. Its model is built
  from that config at each fold, so editing the code changes nothing for it. Removing the input
  mid-run would need a restart of R4 (about 2.5 h lost), for a change Q23 expects to be inside the
  noise. So R4 finishes as it is.
- **The next runs.** Built 22:30: `TrainConfig.drop_constant_statics` (default on, CLI
  `--no-drop-constant-statics` to keep them). Each fold leaves out any static categorical with one
  value in its training rows; `symbol` always stays. On the CS-only panel only `issue_type` goes
  (`exchange` has 15 values, `industry_code` 60). The overnight queue starts each run as a new
  process, so R4p, R4f and every later TFT run pick it up with no change to the queue.
- **Old runs are unchanged.** A run saved before the setting existed reloads with it off
  (`ADDED_INPUTS`), so resuming, predicting or reporting R0–R4 rebuilds their models as trained.
- **What it means for the bake-off.** R4 vs R4p/R4f now also differs in `issue_type`. Q23's
  reasoning (a constant acts as a fixed bias) says the effect is too small to measure; if R4p or
  R4f wins by a hair at 09:30/09:45, R5 decides it anyway, as all R5 models drop the input.
- The HGB runs (R7, R7f) and the overlays never used the statics.

### Q28 (2026-10-29): What is the ideal skill %, and how are skill % values classified?

**There is no standard scale. 100% is the ceiling and can't be reached; the useful number is the gain over the previous best.**
- **What 100% means:** zero pinball loss, i.e. every quantile exactly on the actual move every day. A
  market move has an unpredictable part, so the real ceiling is well below 100% and unknown.
- **What 0% means:** no better than the naive forecast, every quantile at a 0 move. That naive is
  weak (zero width), so much of the 27% comes from knowing how volatile each stock is, not from
  calling the move. A climatology baseline (each stock's past spread of moves) is not computed yet
  (GUIDE §10); skill against it would be lower.
- **Proposed bands for this target** (next-session VWAP move vs a zero-move naive; set 10-29, not in
  the code, to revisit once climatology is measured):

  | skill | class | here |
  | --- | --- | --- |
  | < 0% | broken: worse than "no move" | |
  | 0–15% | weak: little beyond a volatility guess | |
  | 15–25% | fair | 2-year production run, R1a |
  | 25–30% | good for a night forecast | R0 26.6%, R1c 27.2% |
  | 30–35% | very good for a night forecast | |
  | > 35% night | suspect: check for leakage first | |
  | 35–45% | expected with same-day pre-market | R4p 39.6%, R6p 39.8%, R6f 40.3% |

- **Compare like with like.** Skill moves with the rows scored (the pre-market rows are more active
  stocks: R0 26.8% there vs 26.6% overall) and with the forecast time. Decisions use the paired
  difference and its CI on common rows (Q21), not the skill level.
- **Skill is not calibration.** A model can gain skill and still have bands that are off; the grade in
  LEARNINGS ("2026-10-29 All models so far vs R0") rates calibration separately.

### Q29 (2026-10-29): Which metric tells whether a model predicts the movement and direction well?

**No single one: pinball/skill for the whole forecast, then direction measures checked against one-input rules.**

| question | metric | where |
| --- | --- | --- |
| Is the whole forecast (move + uncertainty) better? | pinball loss, paired diff with CI; skill % | `tft-vwap-compare`, summary.json |
| Does it call up or down? | direction hit rate (P50 sign) | summary `directional_accuracy` |
| ... on the moves that matter? | direction on \|move\| > 2% | `scripts_direction.py` |
| When it is sure, is it right? | hit rate of P(up) ≥ 65% / ≤ 35% calls; calibration signals | `calibration_signals.csv` |
| Does it order stocks by move (screener)? | rank IC per day | `scripts_direction.py` |
| How far off is the point forecast? | median \|P50 error\| | `scripts_direction.py` |
| Are the bands honest? | P5–P95 / P10–P90 coverage, quantile error | compare, LEARNINGS grade |

- **Always against a simple rule.** A direction hit rate means little alone: 70% at 09:30 looks
  strong, but "today moves the way the open moved" also gets 70.7% (LEARNINGS "Direction and move
  accuracy", 10-29). At night "the way yesterday closed vs its VWAP" gets 61.3%, above every night
  model. A model is good at direction only if it beats these on the same rows.
- **So far** the models' gains are in size and uncertainty (pinball, coverage), not direction.
  Pinball still decides the model choice, since the forecast is used as a range.
- **To add:** these direction and rule baselines in `tft-vwap-compare` and summary.json, so every
  run reports them (not built).

### Q30 (2026-10-29): Can the forecast dashboard compare the models' accuracy?

**Yes: the Compare runs tab.** It was built for this, and from 10-29 it shows the Q29 metrics too.

- **How.**
  1. `cd dashboard && uv run streamlit run forecast_dash/app.py`, then pick the cross-day section.
  2. Open **Compare runs** and pick the runs. The first one in the box is the reference for
     `vs_first`; the sidebar's run is put there by default.
  3. Keep **Only rows present in every chosen run**
     ticked. Rows match on (stock, date), so runs from different panels line up.
- **Which runs.** Every run directory with `fold_*/test_predictions.parquet` is listed, including
  the morning states: R4's `~px` (09:30) and `~night`, and R6p/R6f as `…-overlay-px` and
  `…-overlay-full`. R4 itself is its 09:45 state.
- **What it shows**, one row per run:
  - accuracy: `pinball`, `vs_first`, `pinball_skill`;
  - direction and movement: `direction`, `direction_big` (|move| > 2%), `rank_ic`,
    `median_abs_err`;
  - calibration: `coverage` (P5–P95), `coverage80` (P10–P90), `q_err_mean` / `q_err_max` and the
    `grade`, plus `bias`, `ece`;
  - the reliability chart and the calls per run.
  Every column has a hover help, and the table's ⓘ explains the bands.
- **What it does not do.**
  - No CI on the differences. `tft-vwap-compare` still decides a choice.
  - No one-input rule baselines (Q29).
  - No pre-market-rows-only filter. With R0 or R1c in the set, the common rows are every row the
    odd-fold runs have, not only those with pre-market data. So skill reads ~35% for the morning
    runs there (1.48M rows), vs ~40% on the 812k pre-market rows in STATUS.
  - It is slow on the 5-year runs: about 1.5 min for 6 runs.

### Q31 (2026-10-29): I'll soon train intraday on the renewed daily quantiles. What has to be in place?

**The handoff table from the chosen daily run, built the same way as R0's** (`tft-vwap-handoff`, Q19/Q20). Three things to check first:
- **Rebuild it from the base.** R0's `handoff.parquet` exists; R1c's (the base) does not yet. Run `uv run tft-vwap-handoff runs/20261028-233305Z-return-4438sym` once the base is final. If the bake-off changes the base, rebuild again.
- **The morning forecasts only cover the odd folds.** The leading 09:30 and 09:45 forecasts (R6p, R6f) were run on the 7 odd folds only (about 440 sessions). As intraday inputs, `d_state` 1 or 2 therefore exist only on those days. Either run the chosen overlays on all 15 folds before building the handoff, or train intraday with the night forecast only and add the morning ones later as an ablation.
- **Point in time inside the day.** The night forecast is in force from the open. The 09:30 and 09:45 forecasts apply only from the step at which they exist live (`d_state` changes then), never earlier in the encoder.
- The rules still hold: out-of-fold forecasts only, the intraday run records its `daily_run`, and a new daily model means re-checking the intraday model (Q5).

### Q32 (2026-10-29): What interval does intraday training use? Must the steps be equally spaced (one entry every 15 s)? Can they vary?

**Today: 15-second steps.** Each stock-day is one series. A step is taken every 15 s from the open, with one random offset per date shared by all stocks (`phase_s`). Every value is computed from the E*TRADE feed's own polls at that second (cumulative sums, not bars). The encoder is the last 80 steps (20 minutes), and the day-so-far fields carry the rest of the session (FEATURES-intraday.md).

**The TFT expects evenly spaced steps.**
- pytorch-forecasting numbers the steps with an integer `time_idx` and treats each step as one position. The model does not know how many seconds lie between steps; it assumes they are all the same.
- `allow_missing_timesteps=True` only fills the missing steps in. It does not make the spacing variable.
- **Massive's bars skip intervals with no trades.** A thin stock's minute bars have gaps. Insert the empty minutes explicitly (volume 0, price carried forward) before training: a quiet minute is information (the "trickle" in Q33), not a missing row.

**Variable spacing is possible only by changing what a step means.**
- **Volume bars:** one step per N shares (e.g. 1/200 of the stock's normal daily volume) instead of per N seconds. Quiet stretches become a few long steps and rushes many short ones, and each step's duration becomes an input. This is the closest thing to "variable intervals" that the TFT can take.
- The cost: the targets are clock-time horizons (1/5/15/60 minutes); the market and sector aggregates need every stock at the same clock second; live decisions are made at clock times. All three need extra work.
- **Recommendation:** keep a regular clock grid and put the activity into the inputs (Q33). Volume bars are a later experiment.

### Q33 (2026-10-29): How do I capture volume rate against time and price, e.g. 10 minutes of trickling volume with little price movement, then a 1-minute rush with a big spike, then 20 minutes of below-normal volume with the price drifting down? Does it matter for other models too?

**Three things make the pattern visible to any model: normalise volume for the time of day, measure price moves in the stock's own volatility, and describe the recent regimes explicitly.**

**1. Volume relative to what is normal for that minute.**
- Volume is naturally high at the open and close and low at midday (the U-shape). A 09:31 minute with 3× the stock's average minute volume is ordinary; the same at 12:30 is a rush.
- Today's `vol_step` divides by the 10-day average only, so it doesn't separate the two. Use `vol_rel` = step volume ÷ (the stock's 10-day average × the typical share of the day's volume that falls in this minute). `vol_rel` ≈ 1 is a normal minute, 0.3 a trickle, 5 a rush.
- The typical-share curve already exists (`out/tft_intraday/profiles/`, used by `expected_vol_remaining`).

**2. Price moves in units of the stock's normal step move.**
- `ret_z` = the step's log return ÷ the stock's typical step move (e.g. its 20-day intraday volatility, scaled to the step). A +0.4% minute is a spike for a utility and noise for a small-cap biotech.
- Also the step's range, (high − low) ÷ price, and the signed volume (already `sv_*`).

**3. "Effort vs result": how much price moved per unit of volume.**
- `impact` = |`ret_z`| ÷ `vol_rel`. A big rush with a small move means buyers and sellers absorbed each other; a small volume with a big move means a thin market. Your example's rush is "high effort, high result"; the fade is "low effort, steady result".

**4. Regime summaries: the shape, written down.**
- Classify each minute as quiet / normal / burst by `vol_rel` (e.g. < 0.5, 0.5–3, > 3).
- At every step give: the current regime, minutes spent in it, the previous regime and its length, minutes since the last burst, that burst's size (`vol_rel`) and its price move (`ret_z`), and the price drift since the burst.

Your example, as the model would see it at the end of the fade:

| inputs at minute 31 | value | reads as |
| --- | --- | --- |
| current regime, minutes in it | quiet, 20 | the fade |
| previous regime | burst, 1 minute, `vol_rel` 8, `ret_z` +6 | the rush and its spike |
| regime before that | quiet, 10 min | the trickle |
| minutes since burst | 20 | |
| drift since burst | −2.5 (in `ret_z` units) | giving back the spike |
| `impact` over the last 20 min | high | price moves on little volume |

**5. Windows long enough to hold the pattern.** Rolling sums and means of `vol_rel`, `ret_z` and `impact` over 1, 5, 15, 30 and 60 minutes. The example spans about 31 minutes, longer than today's 20-minute encoder, so the encoder has to reach at least 60 minutes (Q34).

**Does it matter for other models? Yes, more so.**
- A sequence model (TFT, Chronos-2) can in principle learn the shape from the raw step series, if items 1–2 are normalised and the window covers the pattern.
- A tree model (LightGBM/HGB, AutoGluon's tabular models) sees one row at a time and has no sequence. It only sees the pattern through items 3–5.
- So build items 1–5 as panel columns. Both kinds of model can use them, and the regime fields make the results explainable ("the model widens the band 20 minutes after a burst").

**Check it worked:** report skill by regime (after a burst, during a trickle, normal) and run the model with and without the regime group. The gain should show in the minutes after bursts.

### Q34 (2026-10-29): How do I balance the interval: per-second data is too much, 15 minutes misses the pattern? And what design strategy should the intraday deep dive follow?

**Separate three things that are often mixed up:**

| choice | what it decides | proposal |
| --- | --- | --- |
| **step size** (the grid) | how finely the series is recorded | **1 minute** |
| **look-back** (encoder) | how far back the model reads step by step | **60 steps (1 hour)**, plus the day-so-far fields |
| **decision sampling** | how many training samples are taken | **every 5 minutes per stock** for training (random offset), every minute for testing |

The data volume comes mostly from the third, not the first. Sampling decisions sparsely keeps the fine grid but cuts the training samples by 5×; neighbouring 1-minute samples are nearly duplicates anyway.

**Why 1 minute:**
- Your patterns are minute-scale (10 minutes, 1 minute, 20 minutes).
- Massive minute bars cover all 5 years (2021-10-29 on, NAS `bars_1m/`) for every common stock, and test on all 15 daily folds. Finer sources cover 2 years at most.
- The per-second detail need not be lost: **summarise each minute's seconds into extra columns** from the 1-second bars (2024-10-21 on), for example the busiest second's share of the minute's volume, the number of seconds with trades, the 1-second volatility, and where in the minute the volume fell. That is "the best of both worlds": a 1-minute grid that still knows whether a minute's volume came in one burst or evenly.
- Multi-scale windows (Q33 item 5) give the model 5, 15, 30 and 60-minute views at every step without coarser grids.

**Live must match training.** Training reads REST minute bars; live would build minutes from the WebSocket's 1-second bars. The 1-second bars hold about 73% of daily volume, and the minute bars' share isn't checked. Compare minute bars with summed 1-second bars on the same days before training. If they differ, build the training minutes from the same source as live, or rescale.

**Size and memory, rough estimates:**
- 1,500 liquid stocks × 390 minutes × ~1,255 sessions ≈ 730M stock-minutes. That's too much for pytorch-forecasting's in-memory dataset: today's 14.4M rows took 30 GB.
- Options: fewer stocks (a point-in-time liquid universe, e.g. the top 1,000 by dollar volume each day); 2 years first; or a dataset that streams pre-built samples from parquet shards per day. The streaming dataset is the real fix for 5 years and the main engineering item.

**Proposed order for the intraday deep dive:**
1. **Panel builder from `bars_1m`.** Regular session, empty minutes filled, a point-in-time liquid universe, Q33's features, the handoff joined (Q31), the same 6 targets as moves from the last price.
2. **Baselines.** Climatology per time of day, **scaled by each stock's recent volatility** (today's climatology is the same for every stock, which is a weak bar), plus momentum and random walk as today.
3. **HGB/LightGBM on sampled decisions.** Minutes per run, so screen the feature groups (relative volume, regimes, multi-scale windows, the daily handoff) one at a time, as R7 did for daily.
4. **TFT** on the winning groups, with a 60-minute encoder, on the same folds (the 15 daily test blocks).
5. **The 1-second summaries** as an ablation on the last 2 years: they should help 1–5 minutes most.
6. **Later experiments:** volume bars (Q32), Chronos-2 zero-shot on the 1-minute series, a jump/burst probability output (Q7), live predict.

Each step reports skill vs the volatility-scaled climatology per horizon, by time of day and by regime, in LEARNINGS under `tag-ml-intraday`.

### Q35 (2026-10-29): My goal is accurate quantiles, with full pre-market. I don't care for lesser models. Prioritise accuracy of the probabilities/percentiles in the dashboard. I prefer one model: the night forecast won't help, since predictions and stock filtering are needed only at and after 09:30 for intraday. Plan and run whatever helps.

**What changes.**
- The ranking metric is now **calibration**: does P_tau have tau of the actuals below it? On the
  dashboard that is mean/max |q err| (and its grade), ECE, P5–P95/P10–P90 coverage and bias.
  Pinball loss comes second, as a tie-breaker and a sanity check (a model can be calibrated and
  still wide).
- **The night forecast is dropped.** No R7/R1c night model is kept for production; the TFT
  routes stay as references only.
- **One model for 09:30 and 09:45.** Needed because 09:30 has only the prices (last pre-market
  trade and the open), 09:45 has the full pre-market (VWAP, volume, range).

**What was found today** (common rows, the dashboard's own `calib.summary`):
1. **Every model so far is a little pessimistic** (bias −0.7 to −1.2 pts: actuals land above the
   median more often than half the time) and the bands a little narrow. That is a level error,
   the same in most folds, so it can be learned from earlier folds.
2. **Walk-forward recalibration fixes it** (`tft-vwap-recal`, new): for test fold k, each
   quantile moves by the tau-quantile of (actual − P_tau) over the 4 folds before k, never k
   itself. Shifts are learned per group: with or without full pre-market × forecast width
   terciles. R7f's mean |q err| drops 0.78 → 0.10 pts (max 1.08 → 0.22), ECE 1.17 → 0.97,
   coverage 90.0/80.0, bias +0.05, with the pinball loss unchanged. R6f/R6p go to 0.16 pts, the
   TFTs R4f/R4p only to 0.87–0.89 (their errors change between folds).
   - The pre-market group matters: rows without pre-market are biased differently (R7f without it:
     1.05 → 0.45 pts with pm×width groups and window 2). Window 4 is best on the pre-market rows,
     2 on the others; 4 is kept (it is steadier).
3. **Inputs, probed on folds 5 and 13 (P5, P50, P95):**
   - **The open for every stock** (`--open`): the panel sets `open_pct_prev_vwap` to 0 when there
     were no pre-market trades, although the open is known at 09:30 for every stock. Adding it
     gives the stocks with no pre-market **7–13% lower loss** (P50 9.5–9.9%), and 0.2–0.7% for the
     others.
   - **Past volatility** (`--vol`: std of the move over 5/20/60 sessions, mean |move| and mean
     spread over 20): **1.5–2.4% better tails** (P5, P95), no change at P50.
   - Together: P5 4.8–5.5% better, P95 3.9–7.8%, P50 2.5–4.8% over all rows.
   - **More iterations help only the tails:** P50 is best at 200–400 iterations and overfits by
     1,000; P5/P95 keep improving to 1,000 (0.6–1.1%). So `--tail-iter 600` for P15 and below,
     P85 and above. 63 leaves or learning rate 0.1: no better.
4. **Averaging two models** (R7f with R4f) lowers the loss 0.7% but needs a TFT and stays worse
   calibrated; not used.

**The plan (running, `ml/runs/queue-r5.sh`):**
- **R5** = one HGB per quantile, `--premarket optional`: every training row with full
  pre-market appears twice, as it is and with the full group blank (as at 09:30). Scored in both
  states: `runs/<R5>` (09:45) and `runs/<R5>~px` (09:30). + `--open --vol --tail-iter 600`.
  Folds 0–14 (it is the final model; the odd folds compare with the bake-off).
- **R5-cal and R5~px-cal**: recalibration by pm × width on 4 earlier folds. Live, the shifts
  come from the latest 4 test blocks (later: from graded live forecasts).
- **Judged** on the calibration table against R7f-cal (09:45) and R6p-cal (09:30), same rows,
  plus `tft-vwap-compare` for the pinball CI.
- **R8** (the screener) on R5~px-cal's out-of-fold 09:30 forecasts.
- Production switches only after this is evaluated (user rule, 10-28).

**Why one HGB can serve both times.** The trees see the full group's flag; training on both
views of every pre-market row teaches them what to do when it is blank, as the TFT's masking
did (R4), but without the TFT's cost. If R5~px loses clearly to a dedicated 09:30 model, the
fallback is two HGBs from the same features (minutes to train).
