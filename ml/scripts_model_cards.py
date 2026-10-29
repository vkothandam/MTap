"""Write MODEL.md, the model card the dashboard shows, into each research run.

Front matter: label (R0, R1c, ...), forecast (night | 09:30 | 09:45), summary (one line); then a
short markdown body: purpose, inputs, folds, result, status. Re-run after a result or a verdict
changes (docs/STATUS.md § Model performance has the numbers). Runs not listed are left alone.
New runs write their own card (model_card, `--label`); list a run here only to give it a
hand-written verdict, which replaces the generated card.

    uv run python scripts_model_cards.py            # all known runs
"""
from pathlib import Path

RUNS = Path(__file__).parent / "runs"

BAKE = "Bake-off rows: 7 odd folds, rows with full pre-market data (812k forecasts, 441 days)."

CARDS = {
    "20261005-025909Z-return-3712sym": dict(
        label="PROD", forecast="night",
        summary="Production: the live night forecast (2-year panel, stock inputs, fold 3)",
        body="""**Purpose.** The model in use for the live night forecast until all research runs are
evaluated (user, 10-28).

**Inputs.** 2-year panel, the stock's own price and volume history; no earnings, market or
pre-market inputs. 4 folds; live forecasts use fold 3's checkpoint.

**Result.** Skill 22.9% on its test days. R0 is 6.2% better on the shared days.""" ),
    "20261011-173109Z-return-3785sym": dict(
        label="PIT", forecast="night",
        summary="2-year retrain with point-in-time earnings and series breaks (10-11)",
        body="""**Purpose.** Checked earnings inputs (point-in-time) and series breaks on the 2-year
panel.

**Result.** 2.7% better than production on shared rows. Superseded by the 5-year series
(R0 on).""" ),
    "20261028-030847Z-return-4438sym": dict(
        label="R0", forecast="night",
        summary="5-year baseline: stock inputs + exact earnings dates, no new groups",
        body="""**Purpose.** The reference for the 5-year series: every later run is measured against it.

**Inputs.** 5-year panel (4,438 series), the stock's history, exact EDGAR earnings windows.
Folds 0–14.

**Result.** Skill 26.6%, P5–P95 88.7%, calibration excellent (mean |q err| 0.40 pts).
6.2% better than production. **Superseded by R1c.**""" ),
    "20261028-172535Z-return-4438sym": dict(
        label="R1", forecast="night",
        summary="R0 + market/sector, macro, session shape, after-hours (all groups)",
        body="""**Purpose.** Test all new always-available inputs at once.

**Result.** 1.7–2.5% worse than R0, worse in 7 folds: macro and market inputs are one value per
day for every stock, so the model learns date fingerprints. **Rejected.**""" ),
    "20261028-214331Z-return-4438sym": dict(
        label="R1a", forecast="night",
        summary="R1 − macro (stopped after fold 6)",
        body="""**Purpose.** Drop-one check: is macro the problem in R1?

**Result.** Still worse than R0 (+3.0% on folds 1–6), calibration poor. **Rejected**; stopped
to start R1c.""" ),
    "20261028-233305Z-return-4438sym": dict(
        label="R1c", forecast="night",
        summary="BASE: R0 + session shape + after-hours (no market, no macro)",
        body="""**Purpose.** R1 without the date-fingerprint groups. **The research base** and the night
forecast choice.

**Inputs.** R0's + the day's shape (high/low position, gaps vs VWAP) and after-hours (VWAP,
volume, close vs the session VWAP). Folds 0–14.

**Result.** Skill 27.2%, 0.6–0.8% better than R0 (CI clear), most on earnings days;
calibration good (0.57 pts). Direction 59.6%, below the one-input rule (61.3%).""" ),
    "20261028-030847Z-return-4438sym-event": dict(
        label="R6c", forecast="night",
        summary="Event-band overlay on R0: widen the band after big market/sector/yield moves",
        body="""**Purpose.** Widen R0's band on days after large market, sector or yield moves.

**Result.** No gain (+0.02% vs R0); coefficients flip sign between folds. Kept as a tool.""" ),
    "20261028-030847Z-return-4438sym-event-cal": dict(
        label="R6c-cal", forecast="night",
        summary="R6c + one band scale for all days (--calibrate)",
        body="""**Purpose.** R6c variant: also rescale every day's band.

**Result.** Slightly worse than R0 (−0.000004 per day, CI clear). Not used.""" ),
    "20261028-030847Z-return-4438sym-event-cal-shift": dict(
        label="R6c-shift", forecast="night",
        summary="R6c + band scale + median shift (--calibrate --shift)",
        body="""**Purpose.** R6c variant: also shift the median.

**Result.** Worse than R0 (−0.000011 per day, CI clear): overfits early folds. Not used.""" ),
    "20261029-040858Z-return-4438sym": dict(
        label="R4", forecast="09:45",
        summary="A: one masked TFT for all three forecasts; this is its 09:45 (full pre-market) state",
        body=f"""**Purpose.** Approach A of the pre-market bake-off: one checkpoint that gives the night,
09:30 and 09:45 forecasts, by hiding the pre-market inputs at random in training.
Siblings: `~px` (09:30) and `~night`.

**Inputs.** R1c's + pre-market prices and full pre-market, with masks. 7 odd folds.
{BAKE}

**Result.** Skill 39.3% at 09:45; loses to R4f (1.0%) and R6f (1.55%). Not chosen.""" ),
    "20261029-040858Z-return-4438sym~px": dict(
        label="R4~px", forecast="09:30",
        summary="A: the masked TFT R4 with only the pre-market prices on (09:30 state)",
        body=f"""**Purpose.** R4's 09:30 forecast: pre-market last price and the open known, full
pre-market hidden.

{BAKE}

**Result.** Skill 39.2%; R4p 0.7% and R6p 1.05% better (CIs clear). Not chosen.""" ),
    "20261029-040858Z-return-4438sym~night": dict(
        label="R4~night", forecast="night",
        summary="A: the masked TFT R4 with all pre-market hidden (night state)",
        body=f"""**Purpose.** R4's night forecast: does masking cost the no-pre-market forecast?

{BAKE}

**Result.** Ties R1c (+0.10%, CI spans 0), worse on earnings days; calibration fair. R1c
stays the night choice.""" ),
    "20261029-090243Z-return-4438sym": dict(
        label="R4p", forecast="09:30",
        summary="B: a dedicated TFT with the pre-market prices always on",
        body=f"""**Purpose.** Approach B at 09:30: a TFT trained only for the 09:30 case.

**Inputs.** R1c's + pre-market last price and the open, always on. 7 odd folds. No
`issue_type` (constant on the common-stocks panel).
{BAKE}

**Result.** Skill 39.6%; 0.7% better than R4~px, 0.35% worse than R6p (CIs clear).
Second at 09:30.""" ),
    "20261029-122046Z-return-4438sym": dict(
        label="R4f", forecast="09:45",
        summary="B: a dedicated TFT with prices and full pre-market always on",
        body=f"""**Purpose.** Approach B at 09:45: a TFT trained only for the 09:45 case.

**Inputs.** R1c's + pre-market prices, VWAP, volume and range, always on. 7 odd folds.
{BAKE}

**Result.** Skill 39.9%; 1.0% better than R4, 0.55% worse than R6f (CIs clear). Level with or
better than R6f in the last folds (11–13). Second at 09:45.""" ),
    "20261028-233305Z-return-4438sym-overlay-px": dict(
        label="R6p", forecast="09:30",
        summary="C: HGB overlay on R1c's quantiles using the pre-market prices — LEADS at 09:30",
        body=f"""**Purpose.** Approach C at 09:30: keep R1c and adjust its quantiles with a gradient-boosted
model of the pre-market prices. Trained on R1c's out-of-fold forecasts from earlier folds only.

{BAKE}

**Result.** Skill 39.8%, calibration good (0.99 pts), P5–P95 90.8%. 0.35% better than R4p,
1.05% better than R4~px. **Leads at 09:30.**""" ),
    "20261028-233305Z-return-4438sym-overlay-full": dict(
        label="R6f", forecast="09:45",
        summary="C: HGB overlay on R1c's quantiles using the full pre-market — LEADS at 09:45",
        body=f"""**Purpose.** Approach C at 09:45: R1c adjusted with all pre-market inputs (prices, VWAP,
volume, range).

{BAKE}

**Result.** Skill 40.3%, calibration good (1.00 pts), P5–P95 90.8%. 0.55% better than R4f,
1.55% better than R4. **Leads at 09:45.**""" ),
    "20261029-153450Z-gbm-4438sym": dict(
        label="R7", forecast="night",
        summary="Plain HGB per quantile on R1c's inputs, no TFT (baseline)",
        body="""**Purpose.** Checks the TFT is worth its cost: sklearn gradient boosting per quantile on
the last-day inputs plus lags of the move. Folds 0–14.

**Inputs.** R1c's (stock history, exact earnings, session shape, after-hours) as last-day
values + lags 1, 2, 5, 20 of the move; 200 iterations. No pre-market.

**Result.** Skill 28.0%, **1.3% better than R1c** on 3.17M shared rows (CI clear), on
earnings and other days; gains largest in the early folds (less training data), later folds
mostly within noise. P5–P95 88.8%, calibration good (0.80 pts vs R1c's 0.67). Direction
60.1%. Trains in minutes, no GPU.""" ),
    "20261029-170229Z-gbm-4438sym": dict(
        label="R7f", forecast="09:45",
        summary="Plain HGB with full pre-market always on, no TFT — ties R6f at 09:45",
        body=f"""**Purpose.** R7 with the full pre-market group (prices, VWAP, volume, range) as features:
can one gradient-boosted model replace the TFT plus overlay at 09:45?

{BAKE}

**Result.** Skill 36.2% on its own rows. On the bake-off rows: **ties R6f** (0.06% better, CI
spans 0, no fold clear), 0.76% better than R4f (CI clear; R4f better in fold 13) and 1.6%
better than R4. P5–P95 89.3% vs R6f's 90.8%. One model instead of R1c + an overlay.""" ),
}
CARDS["intraday/20261012-003925Z-intraday-1055sym"] = dict(
    label="I1", forecast="intraday",
    summary="Intraday TFT: next 1–60 min, rest of day and whole-day VWAP from 15 s points (10-12)",
    body="""**Purpose.** First intraday model: one TFT for six targets (VWAP move over the next 1, 5,
15, 60 minutes, the rest of the day, the whole day) from each 15 s decision point, with
market/sector inputs and the morning daily forecast joined in.

**Inputs.** 1,055 stocks, 80 encoder steps, hidden 32. Test days 2025-12-29/30 and 2026-01-12.

**Result.** 2–3% better than climatology at 1–15 min (CI clear), level at 60 min, worse for the
rest of the day (−17%); whole day +5% (CI spans 0). No direction edge (dir_last ≈ 50%). See
docs/intraday-model.md.""")


def write(name: str, card: dict) -> bool:
    run = RUNS / name
    if not run.is_dir():
        return False
    (run / "MODEL.md").write_text(
        f"---\nlabel: {card['label']}\nforecast: {card['forecast']}\nsummary: {card['summary']}\n"
        f"status: done\n---\n\n{card['body'].strip()}\n")
    return True


if __name__ == "__main__":
    for name, card in CARDS.items():
        print(("wrote " if write(name, card) else "missing ") + f"{card['label']:10} {name}")
