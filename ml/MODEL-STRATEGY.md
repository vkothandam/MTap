# ML model strategy: running notes and Q&A

A hand-maintained reference for the modelling choices behind `ml/`.
- **Current position** is the latest state. It is updated whenever an answer changes it.
- **Q&A log** keeps every question and its answer, newest last. Numbers and facts are as of each entry's date.
- **Glossary** explains the terms the answers use.

## Current position (2026-10-25)

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

**Jumps and news**
- Keep real jumps in training. Remove only data errors such as unadjusted splits and vendor glitches.
- When ready, model jumps explicitly: a jump probability, or a two-part forecast (normal band + jump chance + jump size). See Q7 and Q8.
- While the news history is short, "no news collected" must not look like "no news happened". Use a coverage flag, or a small news adjuster on top of the 5-year base model (Q6).

**Open decisions**
1. **Forecast cutoff:** after the close (16:30), after midnight (the current routine), or pre-open (~09:15). This decides which news and prices the daily model can use (Q3).
2. **News backfill:** Massive history (and which sentiment score), or only the news collected from now on.
3. **Benzinga partner data:** whether the Massive plan includes it.
4. **Production run:** keep `20261005-025909Z-return-3712sym --fold 3`, or switch to the 10-11 retrain `20261011-173109Z-return-3785sym --fold 3`, which tests 2.7% better ([FEATURES.md](FEATURES.md) sections 13 and 17).
5. **Resuming a stopped training run (`--resume`):** not built. A stopped run starts over.

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
