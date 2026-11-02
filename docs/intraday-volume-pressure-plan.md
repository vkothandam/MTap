# Intraday volume pressure: study plan

**Status: draft, not started (saved 2026-10-29).** It is a plan, not a result. The study runs in a later session, and nothing in `ml/` is built for it yet. Related pages:
- [intraday-model.md](intraday-model.md) §5 links here.
- The intraday design questions are in [MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q31–Q34.
- **2026-10-30:** the [intraday redesign](intraday-redesign.md) folds this in. The pressure mass
  is its pressure reading (on future volume, S), and the first-passage race is its reach reading.
  Both come out of the same analog engine. The study measures here remain valid checks.

**Predictions in this plan**, added as the user lists them:

| # | prediction | section |
| --- | --- | --- |
| 0 | where the price can go, in each direction, with no fixed horizon: which side is hit first and how fast | §1 signal, §2 target |
| 1 | the rate of the volume-pressure mass against time, which is not linear in time | §2b |

When a new prediction is added: give it a row above and its own § with the target, the method, the measures and the tests, and add a line to [STATUS.md](STATUS.md) Changes.

## Context

The intraday model forecasts fixed horizons: the VWAP of the next 1/5/15/60 minutes, the rest of the day, and the close. The user wants something different: a sense of the general directional momentum that doesn't depend on a fixed horizon.

The user's picture is a 3-D volume over price × volume × time:
- Volume traded above the current price counts positive, volume traded below counts negative.
- The change in that 3-D volume should show the price potential in each direction.
- Time is a full dimension. The user wants to know how quickly a bulge (an immediate price bump) is forming, and whether it is building through volume or through price.

The user's answers to the clarifying questions:
- The target must be **continuous in time**, not fixed 5/15/30-minute horizons.
- **Study first:** measure whether the signal predicts before it becomes model inputs.

This round is a standalone study on the 5-year Massive minute bars. It produces a written result and a go/no-go for adding the signal (and a horizon-free target) to the intraday panel. The panel builder and training are not changed.

## The design

### 1. The signal: a decaying volume-price "mass", across a continuum of time scales

For stock s at minute t, with current price p_t, minute bars i ≤ t (VWAP `vw_i`, volume `v_i`), and a time scale τ:

```
V_τ(t) = Σ_i v_i · e^{-(t - t_i)/τ}  ÷  normal volume over τ        (how much volume, recent-weighted)
D_τ(t) = (Σ_i v_i · vw_i · e^{-(t - t_i)/τ} / Σ_i v_i · e^{-(t - t_i)/τ}  −  p_t) ÷ (p_t · σ_τ)
                                                                     (where that volume sits vs now, in vol units)
M_τ(t) = V_τ · D_τ                                                   (the signed 3-D volume: + above, − below)
```

- **Time** enters twice:
  - through the decay: old volume fades with time constant τ;
  - through rates: the change of M per minute.
- **A continuum of τ.** τ ∈ {1, 2, 4, 8, 16, 32, 64, 128} minutes, log-spaced, gives a term structure: short τ is "right now", long τ is "the last two hours". The study reports results as curves over τ, not a single pick.
- **The rate, split into its two dimensions** (the product rule on M = V·D):
  - `dM_τ = ΔV·D` (**the volume dimension**: a bulge growing because volume floods in) + `V·ΔD` (**the price dimension**: a bulge growing because the price moves away from where volume sits).
  - Their second differences, the acceleration, show how fast a bulge is forming.
  - The two parts must sum exactly to ΔM. That is a unit test.
- **Normalisation, so values compare across stocks and times of day:**
  - Volume is divided by the stock's normal volume for that minute of the day (median of the prior 20 sessions).
  - Price is divided by σ_τ, the stock's typical τ-minute move from the prior 20 sessions.
  - Both are point in time.
- **Sign is not assumed.** The user's convention is "above = positive potential", which reads as a magnet. Classic technical analysis says volume above is resistance. The study measures which one the data shows, per τ and per time of day.

### 2. The target: horizon-free, continuous in time

**Primary: the first-passage race.** From minute t, take barriers ±k·σ_daily, for k ∈ {0.25, 0.5, 1}. Record:
- which barrier is hit first: up, down, or neither by the close;
- **how many minutes it takes:** T_up and T_down, each the first time that side is hit, censored at the close.

That gives a continuous-time outcome. The model-side reading is two rates: how fast the price tends to reach each side. That is "price potential in each direction" plus "how quickly" in one object, with no fixed horizon.

**Secondary: the potential curve.**
- `up(h)` = max high over (t, t+h] ÷ p_t − 1, and `down(h)` likewise, for every h from 1 minute to the close, in σ units.
- They are summarised by their area under a decay weight, which makes them horizon-free: `U = Σ_h up(h)·w(h)`, `Dn = Σ_h down(h)·w(h)`, `lean = U − Dn`.

Targets use the bars' high and low, regular session only.

### 2b. Second prediction: the mass's own rate over time (non-linear)

The user, 2026-10-29: predict the "rate of volume-pressure mass against time. This is also not a linear one, since the rate changes are not linear to time."

**What is predicted:** the path the mass will take from now, ΔM_τ(t, h) = M_τ(t+h) − M_τ(t) for every h, not one number at one horizon. Its rate is dM/dh. A bulge builds fast, peaks, then fades, so the rate rises and decays with time rather than staying constant.

**How to make a non-linear curve predictable, horizon-free.** Describe each future path with a few shape numbers, then predict those numbers.

1. **A parametric shape**, fitted to each realised path for h = 1 … 120 min, or to the close:

   ```
   ΔM(h) = A · (1 − e^{−h/κ}) − F · (1 − e^{−h/λ}) + B · h
   ```

   | number | meaning |
   | --- | --- |
   | A | the size of the bulge that builds |
   | κ | how fast it builds, its time constant |
   | F, λ | how much of it fades back, and how fast |
   | B | the steady drift left over |

   Derived from these: the peak rate (A/κ at h ≈ 0), the time of the peak, and the peak size.

2. **A data-driven shape (functional PCA):**
   - Sample every realised path on a log-spaced h grid (1, 2, 4 … 128 min).
   - Take the 3–4 main components. These are the shapes the data actually shows, for example "steady build", "spike and fade" and "reversal", with no assumed formula.
   - Predict each component's score.
   - This is the check that the parametric shape isn't forcing the wrong form.

3. **Volume clock check:**
   - Recompute the same path against cumulative volume instead of minutes.
   - If the rate is close to linear in volume time, the non-linearity comes from the uneven arrival of volume. Then predicting the volume arrival rate plus a simple slope is the simpler model.

4. **Burst dynamics (Hawkes-style intensity), as an explanatory fit:**
   - Model the volume arrival rate as a base rate plus self-excitation after each burst, decaying with a fitted kernel.
   - The kernel's shape is the non-linear decay of the rate. The branching ratio says how much one burst triggers more.
   - Fit per liquidity tercile and time-of-day bucket.

**Inputs for predicting it:** the same §1 features at t, especially the current dM split (volume part vs price part), the acceleration, time of day and regime.

**Study measures, added to §3:**
- How predictable the shape numbers are out of sample: rank IC and R² per number.
- Whether a forecast path shape improves the first-passage prediction (the §2 target) over the current features alone. That ties "how will the pressure evolve" to "where will the price go".

**Code:**
- `ml/pressure/targets.py` gains `mass_path` (ΔM on the log h grid) and the shape fit, vectorised with `scipy.optimize.least_squares` in batches and bounded parameters.
- `ml/pressure/study.py` gains the fPCA (`sklearn.decomposition.PCA` on the log-grid paths) and the volume-clock variant.
- **Tests:**
  - A synthetic path built from known A, κ, F, λ, B is recovered within tolerance.
  - fPCA on synthetic mixtures recovers the components.
  - The mass path at t uses only bars after t as the target, and never as inputs.

### 3. What the study measures

All results are out of sample in time: features are point in time, and comparisons use later months than any fitted piece.
1. **Rank IC per day** of M_τ, dM_τ (volume part and price part separately) and their acceleration, against `lean` and against P(up first). Shown as curves over τ, with mean and t-stat over days.
2. **Conditional tables by M_τ decile:**
   - P(up first)
   - median T_up and T_down
   - censoring rate

   This directly answers "does more mass above mean the price goes up, and how fast?"
3. **Volume-led vs price-led bulges.** Split moments by which part dominates dM. Compare the outcome and the speed, which answers the user's "in what dimension" question.
4. **Does it add anything over what we already have?** This is the key honesty check. D_τ at long τ is close to the existing `vwap_dev` input, and short-τ M is close to signed volume (`sv_*`).
   - Fit a small HGB classifier on (up first vs down first), with and without the pressure group.
   - The baseline group is ret_τ, vwap_dev, rvol and signed volume, recomputed in the study.
   - Train on earlier months and test on later ones. Report log loss and AUC, with a bootstrap CI over days.
   - Also fit a Cox / discrete-time hazard on T_up, as a check that the timing part carries signal.
5. **Breakdowns:** time-of-day bucket (open / midday / close), liquidity tercile, and the regime from Q33 (after a burst vs quiet).

**Go/no-go:** the pressure group goes into the intraday panel only if item 4 shows a gain whose CI is clear of 0 in most test months. The first-passage target becomes a candidate new head for the intraday model only if item 3 shows usable timing signal.

### 4. Data and scope

- **Source:** NAS `bars_1m/month=YYYY-MM/bars-<quarter>.parquet`. Columns are `ticker, t (ms UTC), o h l c v vw n`, about 27M rows per quarter for all tickers, 04:00–19:59 ET. The NAS is read only; no copies are written to it.
- **Regular session 09:30–16:00 ET.** Half days come from the exchange calendar (`pandas_market_calendars`, if it is already a dependency; otherwise the 13:00 closes are listed explicitly). Empty minutes are filled with v = 0 and the price carried forward.
- **Universe:** a point-in-time top 500 by the prior session's dollar volume (Σ v·vw), computed from the bars themselves. That keeps ml/ independent of sourcing_py (the ml-decoupling rule).
- **Development quarter:** 2026-07..09. **Stability check:** one quarter each from 2022, 2023, 2024 and 2025, the same code. About 500 × 63 × 390 ≈ 12M stock-minutes per quarter, which fits in memory.

### 5. Where the code goes

A new small package, `ml/pressure/`, standalone, with no import from `tft_intraday`, `tft_vwap` or sourcing:
- `ml/pressure/bars.py`: load a quarter, keep the regular session, build the point-in-time universe, fill the grid, compute the normal-volume curve and σ_τ from the prior 20 sessions.
- `ml/pressure/features.py`: V_τ, D_τ, M_τ through EWMA recursions per (stock, day) (`scipy.signal.lfilter`, vectorised), the dM split into volume and price parts, acceleration, and the baseline group.
- `ml/pressure/targets.py`: the first-passage race (a vectorised loop over minute offsets, censored at the close) and the potential curve with its decay-weighted areas.
- `ml/pressure/study.py`: the CLI `uv run pressure-study --quarter 2026-09 [--universe 500] [--out runs/pressure/<id>]`. It writes the IC curves, the decile tables, the HGB with/without comparison and the breakdowns as parquet + `summary.json`.
- An entry point in `ml/pyproject.toml`.
- `ml/tests/test_pressure.py`, on synthetic series:
  - The EWMA matches a brute-force sum.
  - ΔV·D + V·ΔD equals ΔM exactly.
  - First passage on a hand-made path gives the known side and time; a censored path gives "neither".
  - No feature at t changes when bars after t are altered (no look-ahead).
  - The normal volume and σ use only the prior sessions.

Reuse patterns, not imports:
- the decision-at-t / no-look-ahead conventions of `python/sourcing_py/intraday/export.py` (cumulative sums, the tick rule for signed volume, the typical volume curve by session fraction);
- the per-day bootstrap and the honest-baseline framing of `tft_intraday/report.py` and `baselines.py`;
- the study-script style of `ml/scripts_direction.py`.

### 6. Docs, per the keep-docs-current rule

- `docs/LEARNINGS.md`: a new topic `tag-ml-intraday` with "2026-10 volume pressure study": the method, the IC curves over τ, the decile tables with examples, the volume-led vs price-led comparison, and the go/no-go.
- `ml/MODEL-STRATEGY.md`:
  - Q35, "3-D volume pressure and a horizon-free target", in plain language;
  - glossary rows for first passage, hazard rate, term structure and rank IC;
  - the Current position intraday block updated with the verdict.
- `docs/intraday-model.md` §5: the result and, on "go", the inputs and target to add.
- `docs/STATUS.md`: a Changes line.

## Verification

1. `cd ml && uv run pytest tests/test_pressure.py`: all the synthetic checks above.
2. `uv run pressure-study --quarter 2026-09 --universe 50` as a smoke run. Check:
   - the row count is 50 × sessions × 390 (minus half-day minutes);
   - there are no NaN features after the warm-up;
   - P(up first) ≈ 0.5 overall, and the censoring rate is sensible per k.
3. A spot check on one stock-day:
   - Plot price, volume and M_τ for 3 values of τ. Confirm that a known spike minute shows up as a volume-led then price-led dM.
   - Confirm that T_up / T_down match the bars by eye.
4. The full development quarter, then the 4 stability quarters. The verdict needs the same sign and a CI clear of 0 in most quarters.
5. The docs are updated as in §6, and the numbers are checked against `summary.json`, not memory.
