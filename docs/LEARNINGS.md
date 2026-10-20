# Learnings

What the data checks and experiments found, grouped by topic. Each experiment records:
- what was run, and on which data;
- the results;
- examples wherever values differ;
- how to run it again.

Every section carries a `Tags:` line. To find every section on a subject, search for its tag, e.g. `rg -n "tag-price" docs/LEARNINGS.md`.

## Tags

| Tag | Use for |
|---|---|
| `tag-massive-etrade-comparison` | **Topic: Massive and ETrade feed comparisons.** Massive WebSocket bars (`agg_1s`) compared with the E*TRADE watchlist feed (`intraday_feed`) |
| `tag-massive-data-anomaly` | **Topic: Data quality issues in Massive REST and WebSocket feeds** |
| `tag-timestamps` | What a timestamp means: lags and delays |
| `tag-price` | The price at a moment: last trade, matches and mismatches |
| `tag-volume` | Running and daily volume |
| `tag-vwap` | VWAP fields and rebuilt VWAPs |
| `tag-open-close` | The open, the close, auctions and end-of-day values |

## Adding a learning

- **Where it goes.** A new experiment goes under its topic as a `###` section named `<YYYY-MM> <what was checked> (sessions …)`. A new topic needs its own `##` section and a new `tag-…` in the table above.
- **Tags.** Every topic section and every section under it starts with a `Tags:` line: the topic tag first, then the subject tags.
- **What to record:**
  - the run date and the data used;
  - the script and the command;
  - the result tables;
  - **examples wherever values differ**: symbol, time, both values, and what explains the gap;
  - caveats and follow-ups.
- **Changed findings.** When a later run changes an earlier finding, edit the finding and say what changed and when.

## Massive data anomalies

Tags: `tag-massive-data-anomaly`

Issues found in Massive price and volume data that affect training and backtesting.

### 2026-10 REST vs WebSocket comparison (sessions 2026-10-14 to 10-16)

Tags: `tag-massive-data-anomaly` `tag-price` `tag-volume` `tag-vwap`

**What was run**
- REST 1-second bars (na/massive1s, 2-year backfill completed 2026-10-20) compared second-by-second with the WebSocket delayed feed on the three days where both are available: 2026-10-14, 10-15, 10-16.
- Script: custom DuckDB comparison at /tmp/rest_vs_ws.py (read-only, no fetches).
- Data read from NAS: `agg_1s/date=D/source={massive_rest,massive_ws}/day-D.parquet`.

**TL;DR**
- Second-by-second price: 99.23% within 0.5 basis points (**0.5 bp = 0.005%**, not 0.5%). Excellent agreement.
- Second-by-second volume: 88.81% exact match. Differences are rounding: REST rounds to nearest share; WebSocket uses `floor(fractional_volume)`. Relative error <0.2%.
- **Both feeds have corrupt price data** (prices >$700k on individual stock bars), which contaminates aggregate VWAP and must be filtered before training.

**Good news: matched seconds**
| Metric | Result |
|---|---|
| Matching pairs (both have a bar) | 21.6M of 31.6M pairs (68%) |
| Price exact match | 98.94% |
| Price within 0.5 bp | 99.23% |
| Volume exact match (REST dv vs WS v) | 88.81% |
| OHLC per day (O/H/L/C all match) | 99% ± 0.2% |

**Rounding explanation: why 88.81% exact volume match**
- REST `volume_i_dec`: exact volume from the API (includes fractional shares). REST's `volume_i` rounds to nearest share.
- WebSocket `volume_i`: defined as `floor(fractional_volume)`, not rounded.
- The 11.19% difference is rounding error: average relative difference 0.137%, max observed <1 share on most bars.
- **Recommendation:** use `volume_i_dec` in both sources for training; they match at 88.81% and mean relative error is 0.137%.

**Data quality issue: corrupt prices**
Both REST and WebSocket files contain erroneous price values:
- REST 14–16 Oct: prices range from $0.03 to $790,335 (AAPL at one point shows $790,335; clearly unadjusted split or Massive glitch).
- WebSocket 14–16 Oct: prices range from $0.00 to $1,000,000.02 (similar outliers).
- The 99% price match rate holds for the valid bars. The corrupt data corrupts aggregate statistics: per-day VWAP calculations are unreliable.
- These outliers are most likely:
  - Split-unadjusted bars (unlikely given both sources are split-adjusted as of fetch date per Massive docs).
  - Massive data glitches (temporary bid/ask or test values).
- **Action:** filter prices outside ±5% of the daily bar's close or use a rolling median before training. Do not use raw per-second prices for aggregations until this is resolved.

**Coverage: REST has fewer bars than WebSocket**
| Date | REST bars | WS bars | Both | REST-only | WS-only |
|---|---|---|---|---|---|
| 2026-10-14 | 7.2M | 10.1M | 0 | 7.2M | 10.1M |
| 2026-10-15 | 7.6M | 11.0M | 10.2M | 0.9M | 0.8M |
| 2026-10-16 | 7.0M | 10.3M | 10.2M | 0.8M | 0.1M |

Oct 14 has no overlap (WebSocket started 13:35 UTC, REST at 13:00 UTC). The feeds diverge at session start, suggesting either:
- The WebSocket feed was not yet at full capture rate at 09:35 ET on 14 Oct.
- REST and WS capture different subsets of trades (odd lots, after-hours, etc.).

**Examples of corrupt prices**
Both feeds show prices that are clearly not real:
- AAPL at 790,335 (normal ~$190–$250)
- Generic stock showing $1,000,000.02 (absurd)
- These appear scattered through the data, not at any particular time-of-day pattern.

**Implications for training**
1. The 99.23% price match on valid bars is genuine and trustworthy.
2. Volume (use `volume_i_dec`) is trustworthy.
3. Do not use raw price values for aggregate calculations (daily VWAP, returns) without filtering.
4. For intraday model training on return or price move targets, filter outliers (prices >3σ from rolling median, or outside ±5% of session OHLC).

**Re-running**
```bash
cd python
uv run python -I /tmp/rest_vs_ws.py
# Requires both REST and WS files on the NAS at /Volumes/Backups/MTap-Data/agg_1s/
```

---

## Massive and ETrade feed comparisons

Tags: `tag-massive-etrade-comparison`

These sections compare the two intraday feeds stored on the NAS (`/Volumes/Backups/MTap-Data`).

**E*TRADE watchlist snapshots** (`intraday_feed`, source `etrade_watchlist`)
- Each row is a poll of a watchlist page.
- `price` is the last regular-session trade.
- `volume_cum` is the day's running volume, including extended hours.
- `ts_utc` is the time the scraper received the response. E*TRADE sends no quote time.
- Bid, ask and `day_open` exist only on the Experimental list (606 symbols, polled about every 2 s). The MidCap list has 422 symbols, polled about every 20 s.

**Massive delayed WebSocket per-second bars** (`agg_1s`, source `massive_ws`)
- Bars arrive 15 minutes late.
- `ts_utc` is the bar start, and each bar covers one second.
- There is a bar only for seconds with trades.
- Fields: OHLC; `volume_i` (v); the running volume `volume_cum` (av); `vwap_i` (vw); and `vwap_session` (a).

### 2026-10 feed-quality check (sessions 2026-10-14 to 10-16)

Tags: `tag-massive-etrade-comparison` `tag-timestamps` `tag-price` `tag-volume` `tag-vwap` `tag-open-close`

**What was run**
- **First run:** 2026-10-18.
- **Re-run:** 2026-10-19, adding check B2 (where the differing prices come from, section 2a). The re-run reproduced every earlier number.
- **Comparison:** price, volume and VWAP at matching timestamps across three sessions. Massive's grouped daily bars (unadjusted) are the reference for the open, the close and the day VWAP.
- **Script:** [`python/scripts/check_feed_quality.py`](../python/scripts/check_feed_quality.py). It only reads data. It writes CSVs to `/tmp/feed_quality/` and prints a summary.

1 bp (basis point) is 0.01%, which is 1 cent on a $100 stock.

#### Short answers

Tags: `tag-massive-etrade-comparison` `tag-timestamps` `tag-price` `tag-volume` `tag-vwap` `tag-open-close`

| Question | Answer |
|---|---|
| Is the Massive bar time the trade time, despite the 15-min delay? | **Yes.** Prices match E*TRADE best when the Massive bar ended about 1 s before the E*TRADE poll. Shifted 15 minutes, they almost never match. The closing auction lands in the 16:00:00 bar, not at 16:15. |
| How far apart are the prices? | Once the E*TRADE lag is allowed for, **90–95% of polls match exactly** (to 0.5 bp) for the top 300. 99% are within 4–5 bp. Small caps: 97–98% exact, 99% within about 12 bp. |
| Where do the differing prices come from? | **Real trades a second or two away.** For 80–87% of the polls that differ, the E*TRADE price is a Massive trade from within 2 s. The feeds disagree about *which* trade was the latest, not about prices. Within 10 s the share is 94–98%. The only large gaps (median 111–117 bp) are at the open, before a stock's first regular-session trade, when E*TRADE still shows the previous close. See section 2a. |
| Volume? | The median difference is **0.002–0.006%**, and 90% of polls are within 0.04–0.11%. E*TRADE reads very slightly higher. |
| VWAP? | E*TRADE has no VWAP. Rebuilt from its polls the way the intraday model does, it agrees with Massive to a **median 0.2–0.4 bp** over 5- and 60-minute windows for the top 300. MidCap and small caps: 0.6–2.6 bp. Session-to-date: 1–2.4 bp. |
| Open and close? | **Both feeds' opens equal the daily open 100% of the time.** The daily close is the **closing-auction price**. Massive has it in its closing-auction bar (16:00:00–16:00:05) for 88–98% of symbols. E*TRADE shows it 30 s after the close for 83–87% of the top 300 and 98–100% of MidCap. |
| Is the WebSocket VWAP the day's average or the last second's? | **The feed carries both.** `vwap_session` (`a`) is the running day VWAP, and it matches a cumulative calculation to 0.001–0.02 bp. `vwap_i` (`vw`) covers only that second. At the last bar of the day, `a` is 0.5–1.3 bp from the daily VWAP and `vw` is 37–77 bp away. |

#### What was compared

Tags: `tag-massive-etrade-comparison`

**Data.** Both feeds were read from the NAS.

| Date | E*TRADE polls | Massive bars | Notes |
|---|---|---|---|
| 2026-10-14 | 3.52M | 2.04M | Massive starts at 09:35, so the analysis starts at 09:36:27. No open, no session VWAP. |
| 2026-10-15 | 3.51M | 2.19M | Massive covers 09:01–16:06. |
| 2026-10-16 | 3.51M | 2.07M | Massive covers 09:01–16:06. |

No coverage gaps were found on any date. A gap would be either of these:
- a second with no Massive bar for any symbol, lasting more than 3 s;
- an E*TRADE poll gap of more than 10 s.

**Universe.** 382 common stocks, ranked by market cap. Market cap = the latest QTCO shares × the latest close, both from the E*TRADE fundamentals store.

| Tier | Symbols | Watchlist | E*TRADE poll interval |
|---|---|---|---|
| `top100` | 100 | Experimental (with bid/ask, `day_open`) | ~2 s |
| `top101_300` | 200 | Experimental | ~2 s |
| `midcap_10b` | 41 | MidCap, ≥ $10B | ~20 s |
| `small` | 41 | MidCap, < $2B | ~20 s |

Every small cap is on the 20 s MidCap list. `midcap_10b` is there to separate the effect of the poll interval from the effect of company size.

**Matching rule.** Each E*TRADE poll at time *t* is matched to the latest Massive bar that ended by *t − L*. *L* is the lag: how much later E*TRADE stamps its data than the moment the data describes.

#### 1. Timestamps: does the Massive bar time mean trade time?

Tags: `tag-massive-etrade-comparison` `tag-timestamps`

**Lag scan.** The table shows the share of polls whose price matches Massive exactly, for each assumed lag *L*.
- It counts only polls where E*TRADE's volume rose since the previous poll. Without that filter, idle stocks match at every lag.
- All three dates are pooled.

| L (poll − Massive bar end) | top100 | top101_300 | midcap_10b | small |
|---|---|---|---|---|
| −15 min (bar time = delivery time) | 0.018 | 0.018 | 0.032 | 0.061 |
| −60 s | 0.073 | 0.086 | 0.146 | 0.303 |
| −1 s | 0.593 | 0.719 | 0.867 | 0.912 |
| 0 | 0.713 | 0.815 | 0.951 | 0.969 |
| +0.5 s | 0.801 | 0.877 | **0.968** | **0.979** |
| +1 s | 0.889 | 0.934 | 0.938 | 0.961 |
| **+1.25 s** | **0.897** | **0.937** | 0.915 | 0.945 |
| +1.5 s | 0.877 | 0.921 | 0.894 | 0.931 |
| +2 s | 0.780 | 0.851 | 0.855 | 0.905 |
| +4 s | 0.530 | 0.639 | 0.734 | 0.822 |
| +60 s | 0.075 | 0.088 | 0.148 | 0.303 |

- **The curve has a single sharp peak about 1 s after the bar.** If Massive stamped bars with the time it delivered them, 15 min after the trade, the peak would sit at −15 min. There the match rate is only 2–6%.
- **Best lag:** 1250 ms for the Experimental list and 250–500 ms for MidCap. Every date gave the same answer. The Experimental response is larger (606 rows, about 0.2 MB, against MidCap's 422 rows), which probably explains its longer delay.
- **Small caps** still match about 30% of the time at ±60 s, because their price often doesn't change for a minute.
- **Volume crossing.** For each poll, the check finds when Massive's running volume first reaches E*TRADE's.
  - For `top100`, that is a median 0.5–0.9 s before the poll (IQR −2.2 to +1.75 s), consistent with the lag above.
  - For sparser symbols, the crossing shows up only at the next trade, so the estimate is biased negative: a median of −2.4 to −4 s for MidCap.
- **Auctions.** Both auctions show up at trade times:
  - Close: in the closing window, the biggest jump in Massive's running volume `av` is the closing auction. It lands in the 16:00:00–16:00:05 bars for 79–82% of symbols, most often at 16:00:00. For 77–81% of symbols it is also the biggest jump of the day.
  - Open: the biggest jump in the opening window lands in the 09:30 minute for 83–86% of symbols, most often at 09:30:01.

**The 15-minute delay itself isn't stored, so it can't be measured from the data.** No receive time is recorded on the rows. The two ends of the file are consistent with exactly 15 minutes:

| | Wall clock (ET) | Massive bar time |
|---|---|---|
| Collector connects (the 06:16 PT start job) | 09:16 | first bar 09:01 |
| Collector stops | 16:21 | last bar 16:06 |

The collector heartbeat's `delay_s` shows the delay live. Measuring it over time would need a `recv_ms` column in `agg_1s` (see Follow-ups).

#### 2. Price at a timestamp

Tags: `tag-massive-etrade-comparison` `tag-price`

Pooled regular-session polls. Each cell gives the range across the three dates.

| Tier | Lag | Exact match | \|ΔP\| p90 (bp) | \|ΔP\| p99 (bp) | Massive bar age, median (s) | Massive price inside E*TRADE bid/ask ±1¢ |
|---|---|---|---|---|---|---|
| top100 | 0 | 0.72–0.74 | 2.0–2.4 | 6.9–8.7 | 1.7–1.9 | 0.94–0.95 |
| top100 | 1.25 s | **0.90–0.91** | 0.37–0.46 | 4.0–5.1 | 1.8–1.9 | 0.97 |
| top101_300 | 0 | 0.83–0.84 | 1.5–1.9 | 7.4–9.4 | 3.9–4.2 | 0.97 |
| top101_300 | 1.25 s | **0.94–0.95** | 0 | 3.9–4.8 | 3.9–4.2 | 0.98 |
| midcap_10b | 0 | **0.95–0.96** | 0 | 6.9–9.3 | 5.4–6.0 | — |
| small | 0 | **0.97** | 0 | 11.2–12.5 | 7.7–8.8 | — |

- **The polls that don't match are mostly timing artefacts, not wrong prices.** Once the lag is allowed for, the p90 difference is 0–0.5 bp. Section 2a shows where each differing price comes from.
- **The remaining 1% tail** is 4–5 bp for large caps and about 12 bp for small caps. That is roughly one tick on a fast-moving or low-priced name.
- **Bid/ask.** Massive's price is inside E*TRADE's own bid/ask (±1¢) 97–98% of the time. E*TRADE's last price is inside its own bid/ask 98% of the time. By that yardstick the two feeds are equally consistent.
- **"Bar age"** is how old the matched Massive bar is. Small caps trade rarely, so their bar is often several seconds old. The price is still correct, because no trade has happened since.

#### 2a. Where the differing prices come from

Tags: `tag-massive-etrade-comparison` `tag-price` `tag-timestamps`

**Method.** Check B2 was added on 2026-10-19. It takes every poll whose price differs from Massive at the best lag (1.25 s for Experimental, 0.5 s for MidCap) and looks for that price in Massive around the moment the poll describes. Each poll gets the first of these categories that applies:

1. the close of a bar up to 2 s **later** than the matched bar;
2. the close of a bar up to 2 s **earlier**;
3. inside the low–high range of a bar up to 2 s away;
4. printed 2–10 s away;
5. printed 10–120 s away;
6. the previous session's close;
7. not printed within 2 minutes.

**Share of the differing polls in each category** (all three dates pooled):

| Where the E*TRADE price shows up in Massive | top100 | top101_300 | midcap_10b | small |
|---|---|---|---|---|
| Polls that differ, out of all polls | 9.5% (301,935 of 3.19M) | 5.1% (326,361 of 6.38M) | 3.2% (3,953 of 123k) | 2.1% (2,577 of 123k) |
| 1. Close of a bar ≤ 2 s later | 35.1% | 42.3% | 54.2% | 57.5% |
| 2. Close of a bar ≤ 2 s earlier | 12.4% | 10.5% | 7.9% | 7.3% |
| 3. Inside a bar ≤ 2 s away | 39.8% | 27.1% | 20.5% | 15.6% |
| **Within 2 s (1–3)** | **87.3%** | **79.9%** | **82.6%** | **80.5%** |
| 4. Printed 2–10 s away | 11.1% | 15.5% | 12.0% | 13.6% |
| **Within 10 s (1–4)** | **98.4%** | **95.3%** | **94.5%** | **94.1%** |
| 5. Printed 10–120 s away | 1.5% | 4.5% | 5.4% | 5.7% |
| 6. Previous session's close | 0.10% (300 polls) | 0.14% (461 polls) | — | — |
| 7. Not printed within 2 min | 7 polls | 98 polls | 1 poll | 6 polls |

**Size of the difference**: median \|ΔP\| in bp, with p90 in brackets. Like the bid/ask table below, these are medians of the three daily values.

| Category | top100 | top101_300 | midcap_10b | small |
|---|---|---|---|---|
| 1. Later close | 1.5 (4.6) | 1.9 (5.9) | 3.7 (10.2) | 8.0 (24.5) |
| 2. Earlier close | 1.5 (4.4) | 1.9 (5.7) | 4.0 (10.7) | 8.5 (24.6) |
| 3. Inside a bar | 1.4 (4.2) | 1.9 (6.2) | 3.8 (12.3) | 8.5 (24.3) |
| 4. 2–10 s away | 1.8 (5.5) | 2.1 (6.4) | 3.6 (12.3) | 7.4 (21.8) |
| 5. 10–120 s away | 2.4 (6.9) | 2.7 (8.3) | 3.5 (13.8) | 6.6 (19.6) |
| 6. Previous close | **111 (307)** | **117 (268)** | — | — |
| 7. Not printed | 3.5 (3.9) | 4.1 (13.2) | 0.8 (1 poll) | 11.3 (11.3) |

**Which price agrees with E*TRADE's own bid/ask** (Experimental list only). Share of polls whose price is inside E*TRADE's bid/ask ±1¢:

| Category | E*TRADE price: top100 / top101_300 | Massive price: top100 / top101_300 |
|---|---|---|
| 1. Later close | 99.5% / 99.6% | 89% / 91% |
| 2. Earlier close | 98% / 99% | 89% / 92% |
| 3. Inside a bar | 99% / 99% | 84% / 87% |
| 4. 2–10 s away | 96% / 97% | 90% / 93% |
| 5. 10–120 s away | 91% / 94% | 91% / 93% |
| 6. Previous close | **7% / 31%** | 93% / 90% |
| 7. Not printed | 75% / 85% | 100% / 94% |

**Examples.**
- Times are ET.
- The matched bar is the latest Massive bar that ended at least 1.25 s (Experimental) or 0.5 s (MidCap) before the poll.
- Massive bars are written as *bar start: close [low–high]*.
- Bid–ask is shown for the Experimental list only.
- `price_mismatch_examples.csv` holds the full sample: 5 per tier, category and date, 349 rows, each with every Massive bar within 3 s.

| Cat. | Symbol, date, poll time | E*TRADE price (bid–ask) | Massive at the matched bar | ΔP (bp) | Where the E*TRADE price is in Massive |
|---|---|---|---|---|---|
| 1 | BAC 10-16 09:41:11.340 | 52.32 (52.32–52.33) | 52.325 (09:41:09 bar) | −1.0 | Close of the next bar, 09:41:10: 52.32 [52.31–52.32]. E*TRADE already had that second's trade, so this poll's delay was 0.3–1.3 s, not 1.25 s. |
| 1 | C 10-14 11:29:15.231 | 123.21 (123.19–123.23) | 123.255 (11:29:12) | −3.7 | Close of the 11:29:13 bar: 123.21 [123.21–123.225]. That bar ended only 19 ms after the moment the poll is assumed to describe. |
| 1 | BRBR (small) 10-16 15:59:58.362 | 9.425 | 9.43 (15:59:56) | −5.3 | Close of the 15:59:58 bar. One half-cent tick on a $9 stock is 5.3 bp. |
| 2 | CSCO 10-16 15:55:48.550 | 107.24 (107.23–107.25) | 107.23 (15:55:46) | +0.9 | Close of the 15:55:45 bar, the one before the matched bar. E*TRADE was still showing the previous second's trade. |
| 2 | GT (small) 10-16 15:44:39.856 | 4.255 | 4.25 (15:44:38) | +11.8 | Close of the 15:44:37 bar. Half a cent on a $4 stock is 11.8 bp. |
| 3 | INTC 10-16 11:04:10.196 | 92.145 (92.14–92.15) | 92.155 (11:04:07) | −1.1 | The low of the 11:04:08 bar, 92.1491 [92.145–92.15]. E*TRADE caught a trade from the middle of that second. A 1-second bar keeps only the second's last trade as its close. |
| 3 | AI (small) 10-16 14:56:56.648 | 11.17 | 11.175 (14:56:55) | −4.5 | Inside the matched bar itself, 11.175 [11.17–11.175]. 11.17 traded in that second, but the bar closed at 11.175. |
| 4 | JPM 10-16 13:59:58.546 | 338.16 (338.11–338.19) | 338.11 (13:59:56) | +1.5 | Close of the 13:59:54 bar, 2.3 s earlier. Massive then printed 338.11, 338.09 and 338.04; E*TRADE's last price had not caught up. |
| 4 | BBAI (small) 10-16 14:59:45.634 | 2.15 | 2.155 (14:59:44) | −23.2 | Close of a bar 6.1 s earlier. Half a cent on a $2 stock is 23 bp. |
| 5 | ANET 10-16 12:23:40.618 | 179.875 (179.72–179.82) | 179.77 (12:23:38) | +5.8 | In a bar 16 s away. E*TRADE's price is above its own ask, so E*TRADE's last-trade field is the stale one. |
| 5 | CNXC (small) 10-16 11:59:39.675 | 27.755 | 27.73 (11:59:38) | +9.0 | In a bar 40 s away. |
| 6 | MRVL 10-16 09:30:00.500 | 241.24 (235.10–235.61) | 235.25 (09:29:20, pre-market) | +254.6 | 241.24 is the 10-15 close. The poll describes 09:29:59.25, before the bell, so E*TRADE's last regular-session trade is still yesterday's close. |
| 6 | KO 10-16 09:30:06.656 | 92.00 (92.74–93.19) | 93.02 (09:29:56, pre-market) | −109.7 | The 10-15 close. Massive's first bar after the bell is at 09:30:06 (93.015), so KO had not opened yet at the moment this poll describes. |
| 6 | FTNT 10-15 09:30:02.313 | 184.05 (179.50–181.05) | 180.00 (09:30:00) | +225.0 | The 10-14 close. Massive already has a regular-session trade at 09:30:00, so here E*TRADE was 1–2 s late to switch. |
| 6 | UPS 10-16 09:31:45.994 | 97.83 (98.03–98.79) | 98.64 (09:29:55, pre-market) | −82.1 | The 10-15 close. Massive has no UPS trade between 09:29:55 and this poll: UPS had not traded in the regular session yet. |
| 7 | GEV 10-16 15:59:52.693 | 920.76 (920.29–920.81) | 920.44 (15:59:50) | +3.5 | Never printed. Massive went from 920.44 [919.95–920.44] straight to 920.82 [920.82–920.82] at 15:59:51. |
| 7 | RY 10-15 12:58:53.289 | 195.84 (195.71–195.79) | 195.75 (12:58:50) | +4.6 | Never printed, and above E*TRADE's own ask. |
| 7 | MLYS (small) 10-15 12:31:21.591 | 21.62 | 21.64 (12:30:49) | −9.2 | Never printed. Massive had no trade for 30+ s, while E*TRADE showed 21.62 on two polls 21 s apart. |

What this means:
- **Categories 1–3: neither feed is wrong.** Both prices are real trades from the same second or the next one.
  - E*TRADE's delay varies from poll to poll. It is sometimes fresher than 1.25 s (category 1) and sometimes staler (category 2), so no fixed shift can line every poll up.
  - A 1-second bar cannot show a trade from the middle of its second (category 3).
  - The differences are one tick: 1.4–1.9 bp for the top 300, about 4 bp for MidCap ≥ $10B, about 8 bp for small caps. Small caps look worse in bp only because their prices are lower.
- **Categories 4–5: E*TRADE's last price is the stale one.** In category 5, E*TRADE's price falls outside its own bid/ask more often than in categories 1–3 (6–9% against 0.4–2%). Its quote had moved on, but its last-trade field had not.
- **Category 6 comes from E*TRADE's definition of price, not an error.** E*TRADE's price is the last regular-session trade, so until a stock trades after the bell it is the previous session's close. Massive's price at that moment is a pre-market trade.
  - These are the only large differences in the whole comparison.
  - About 360–400 polls a session for the top 300 (0.01% of polls), almost all in the first seconds after 09:30.
  - A late open can stretch this to minutes: UPS on 10-16 was still at the previous close at 09:31:46.
  - E*TRADE's own bid/ask sits near Massive's price, not near E*TRADE's.
  - There were no such polls on 10-14, because that day's analysis starts at 09:36.
- **Category 7 is rare:** 112 polls in three sessions, out of 9.8M. The cause was not checked. One possibility is trades that Massive leaves out of bar OHLC, such as odd lots (see section 6).

#### 3. Volume at a timestamp

Tags: `tag-massive-etrade-comparison` `tag-volume`

This is the relative difference of E*TRADE `volume_cum` against Massive `av`. The lag is 1.25 s for the Experimental list. For MidCap it is 0, the computed lag nearest its 250–500 ms best.

| Tier | Signed median | \|diff\| median | \|diff\| p90 |
|---|---|---|---|
| top100 | +0.0004% to +0.001% | 0.002% | 0.04–0.05% |
| top101_300 | +0.002% to +0.003% | 0.003–0.004% | 0.05–0.06% |
| midcap_10b | +0.001% to +0.003% | 0.003–0.006% | 0.05–0.08% |
| small | +0.002% to +0.003% | 0.003–0.005% | 0.08–0.11% |

- During the session the two volumes are effectively the same.
- Against Massive's fractional-share volume (`dav`, from 10-15 on), the numbers are the same to the reported precision.
- Volume is a weaker lag signal than price: its signed difference stays within 0.01% from −2 s to +2 s.

#### 4. VWAP

Tags: `tag-massive-etrade-comparison` `tag-vwap`

E*TRADE has no VWAP, so the check rebuilds it the way the intraday model's panel does: price × volume change between polls (`export.load_feed` → `series_for_day` → `window_vwap`).

Massive's VWAP was taken two ways:
- **Cumulative:** from the change in `a × av`, using the bars at each end of the window.
- **Bars:** Σ(vw·v) / Σv over the bars in the window.

Checkpoints are 10:00, 11:00, …, 15:00 and 15:59:59. Massive's values are shifted by the measured lag. All three dates are pooled; session windows exist only on 10-15 and 10-16.

| Tier | Window | n | E*TRADE vs Massive cumulative, median \|bp\| | p90 | E*TRADE vs bars, median | Bars vs cumulative, median | Bar volume / window volume |
|---|---|---|---|---|---|---|---|
| top100 | session | 1323 | **1.26** | 7.0 | 1.81 | 0.82 | 0.93 |
| top100 | trailing 5 min | 2099 | **0.31** | 1.3 | 0.27 | 0.17 | 0.92 |
| top100 | trailing 60 min | 1989 | **0.22** | 2.3 | 0.36 | 0.20 | 0.92 |
| top101_300 | session | 1897 | 1.01 | 6.1 | 1.78 | 0.78 | 0.91 |
| top101_300 | trailing 5 min | 4200 | 0.43 | 1.6 | 0.38 | 0.28 | 0.89 |
| top101_300 | trailing 60 min | 3871 | 0.24 | 1.5 | 0.46 | 0.31 | 0.89 |
| midcap_10b | session | 364 | 2.36 | 10.9 | 3.89 | 1.18 | 0.92 |
| midcap_10b | trailing 5 min | 860 | 1.21 | 4.7 | 0.92 | 0.53 | 0.89 |
| midcap_10b | trailing 60 min | 790 | 0.62 | 3.6 | 0.96 | 0.43 | 0.89 |
| small | session | 350 | 2.05 | 12.3 | 2.60 | 1.18 | 0.94 |
| small | trailing 5 min | 855 | 2.62 | 11.3 | 1.62 | 1.37 | 0.90 |
| small | trailing 60 min | 788 | 1.15 | 5.1 | 1.33 | 0.55 | 0.91 |

- **For the top 300, the model's VWAP inputs are within a fraction of a basis point of Massive.** For MidCap and small caps they are within about 1–2.6 bp.
- **The slower polling costs more than company size does:**
  - Going from 2 s to 20 s polls (`top100` → `midcap_10b`) makes the error 2–4× larger.
  - Going from large to small companies on the same 20 s list (`midcap_10b` → `small`) adds up to another 2× on the trailing windows.
- **Session-to-date differs more than the trailing windows.** Session windows start at the open. A likely cause, not checked: one E*TRADE poll interval absorbs the opening-auction volume at a non-auction price.
- **Summed bars are not the same as the cumulative figure.** The bars hold only 89–94% of the window's volume (see section 6). For VWAP, use the cumulative `a`/`av` form, not Σ(vw·v)/Σv.

#### 5. Open, close and end of day vs Massive daily bars

Tags: `tag-massive-etrade-comparison` `tag-open-close` `tag-price` `tag-volume`

Each cell is the share of symbol-days within 0.5 bp of the daily bar, with the median \|bp\| in brackets. `day_open` exists only on the Experimental list.

| | top100 | top101_300 | midcap_10b | small |
|---|---|---|---|---|
| E*TRADE `day_open` vs daily open | 1.00 | 1.00 | — | — |
| Massive `op` vs daily open | 1.00 | 1.00 | 1.00 | 1.00 |
| E*TRADE price at first poll ≥ 16:00:30 vs daily close | 0.83–0.87 (0) | 0.83–0.85 (0) | 0.98–1.00 | 1.00 |
| Massive last bar before 16:00 vs daily close | 0.14–0.26 (1.4–2.3) | 0.19–0.23 (1.6–1.9) | 0.07–0.32 (2.1–2.6) | 0.20–0.29 (6.7–7.2) |
| Massive closing-auction bar vs daily close | 0.94–0.98 | 0.96–0.97 | 0.88–0.98 | 0.95 |

- **The daily close is the closing-auction price, not the last trade before 16:00.** Massive's auction bar is the bar in [16:00, 16:05) with the largest jump in `av`.
- **When E*TRADE misses the close** (top 300, 144 symbol-days, median miss 2.8 bp, p90 7.3 bp):
  - 44% show the last trade from before 16:00. E*TRADE had not yet updated to the auction price 30 s after the close.
  - 1.4% equal Massive's auction bar, where that bar itself differs from the daily close.
  - The rest match neither. Of the three checked below, one is a trade a second earlier, and two aren't in Massive's last 3 s before the bell.

  Examples:

  | Symbol, date | E*TRADE at first poll ≥ 16:00:30 | Massive last trade before 16:00 | Massive closing-auction bar | Daily close | What E*TRADE shows |
  |---|---|---|---|---|---|
  | PG 10-16 | 159.32 | 159.32 | 159.35 | 159.35 | The last trade before the bell |
  | TT 10-14 | 471.49 | 471.49 | 471.29 | 471.29 | The last trade before the bell |
  | ORLY 10-15 | 88.48 | 88.48 | 88.50 | 88.50 | The last trade before the bell |
  | MRK 10-15 | 145.35 | 145.29 | 145.19 | 145.19 | The close of Massive's 15:59:58 bar, one second earlier |
  | MMM 10-15 | 165.23 | 165.31 | 165.33 | 165.33 | Not in Massive's last 3 s before the bell |
  | CBRE 10-14 | 128.83 | 128.89 | 128.87 | 128.87 | Not in Massive's last 3 s before the bell |

- **At 16:05, E*TRADE's volume is ahead of Massive's `av`.** The median gap is 0–1% and the p90 is up to 6%.
  - During the session the two agree to 0.005%, so this happens only after the close.
  - Its cause was not established.
  - In every example below, E*TRADE sits between Massive's `av` and the final daily volume:

  | Symbol, date | E*TRADE volume at 16:05 | Massive `av` at 16:05 | E*TRADE ahead by | Daily volume |
  |---|---|---|---|---|
  | TD 10-14 | 1.90M | 0.94M | +103% | 2.42M |
  | ALL 10-16 | 3.31M | 2.47M | +34% | 3.50M |
  | PG 10-16 | 11.61M | 9.15M | +27% | 13.53M |
  | XEL 10-16 | 6.22M | 5.91M | +5.3% | 6.28M |
  | MO 10-16 | 9.06M | 8.75M | +3.6% | 9.65M |

- **Daily volume.** Massive's last `av` of the day, around 16:06, is 97–99% of the daily volume. The daily volume runs on to 20:00.

#### 6. Massive's two VWAP fields

Tags: `tag-massive-etrade-comparison` `tag-vwap` `tag-volume`

These checks use Massive data only.

**The running-VWAP test.** If `a` is the running VWAP over `av`, it can be predicted from the previous bar: (previous `a` × previous `av` + `vw` × `v`) / `av`. The test uses "clean" bars, where the change in `av` equals the bar's `v`; these are 43–52% of bars.

| Tier | Prediction error, median (bp) | p99 (bp) | Gap between `vw` and `a`, median (bp) |
|---|---|---|---|
| top100 | 0.001 | 0.01–0.02 | 65–71 |
| top101_300 | 0.002 | 0.02–0.03 | 62–70 |
| midcap_10b | 0.004 | 0.08–0.10 | 65–83 |
| small | 0.010–0.018 | 0.30–0.34 | 108–153 |

The prediction is exact up to rounding. `a` is stored to 4 decimals, which is a bigger step in bp for low-priced stocks. So:
- **`vwap_session` (`a`) is the day's running VWAP**, over all volume in `av` since the day began, including pre-market.
- **`vwap_i` (`vw`) covers that second only.**

**The test you proposed.** Compare the last bar's two VWAPs with the daily VWAP. Values are median \|bp\|. "`a` closer" is the share of symbols where `a` is nearer the daily VWAP than `vw` is.

| Point | `vw` vs daily VWAP | `a` vs daily VWAP | `a` closer | `vw` vs that bar's close | `a` vs that bar's close |
|---|---|---|---|---|---|
| Last bar in file (~16:06) | 37–77 | **0.5–1.3** | 98–100% | 0 | 37–96 |
| Last bar before 16:00 | 37–76 | 8–20 | 95–100% | 0.1–0.8 | 51–103 |

- At the end of the file, `a` is within about 1 bp of the official daily VWAP.
- At 15:59:59 it is 8–20 bp off. That gap is the closing auction and the after-hours trades, which the daily VWAP includes and `a` has not seen yet.
- `vw` sits on the bar's own close, which is what a one-second VWAP should do.

**Other points about the bars**
- **`vw` is often outside the bar's [low, high].** It falls inside for only 56–77% of bars. Of the bars where it falls outside:
  - 87–98% are flat (o = h = l);
  - 67–90% have an average trade size under 100 shares.

  Odd-lot trades count toward `vw` and `v` but don't set OHLC. Don't assume low ≤ `vw` ≤ high.
- **Summed bar volume is not day volume.** Bars hold only 78–86% of the growth in `av`. The rest is mostly prints that appear in `av` but in no bar's `v`:

  | Period | Share of the day's `av` growth outside bars |
  |---|---|
  | 16:00:00–16:00:59 (closing auction) | 9–14% |
  | Regular session | 3.5–6% |
  | After 16:01 | 0.2–5% |
  | Pre-market and the 09:30:00 bar | ≈ 0 |

  Use differences of `av` for volume, not sums of `v`.

#### Caveats

Tags: `tag-massive-etrade-comparison`

- **Sample size.** There are only three sessions, and 10-14 is partial: Massive starts at 09:35, so that day has no open and no session VWAP.
- **Session VWAP sample.** It needs a Massive bar between 09:01 and 09:30 for the starting point. That leaves fewer MidCap and small-cap windows (n ≈ 350).
- **Size and poll rate are tied together.** Small caps are only on the 20 s MidCap list. `midcap_10b` gives the poll-rate control, but there is no large-cap tier at 20 s polls beyond those 41 symbols.
- **Volume definitions differ.** E*TRADE's `volume_cum` includes extended hours. Massive's `av` counts prints that no bar's `v` includes. The two agree intraday, but the definitions are not identical.
- **What "exact match" means.** A match is a price within 0.5 bp of Massive's at the matched bar. A poll whose price changed within the same second can fail to match even when both feeds are right; section 2a measures this.
- **B2 category order.** Each poll gets the first category that applies, so a price that is both a nearby close and inside a bar counts as a close.
- **B2 direction.** The 2–120 s distances in categories 4–5 have no direction (earlier or later). Only the close matches carry a sign.

#### Follow-ups (not done)

Tags: `tag-massive-etrade-comparison`

- Record a receive time (`recv_ms`) for each Massive bar, or log the heartbeat's `delay_s` for each file, so the 15-min delay can be measured.
- Optionally shift E*TRADE timestamps back by about 1.25 s (Experimental) or 0.5 s (MidCap) when aligning the two feeds.
- Check how the intraday panel (`series_for_day`) treats polls before a stock's first regular-session trade, where E*TRADE's price is still the previous close (section 2a, category 6). A return from such a poll to the first real trade would be a jump of about 1–3%.
- Category 7 (about 110 polls in three sessions): check whether these prices are trades that Massive leaves out of bar OHLC, such as odd lots.
- Find out why E*TRADE's volume runs ahead of Massive's `av` at 16:05 (section 5).

#### Re-running

Tags: `tag-massive-etrade-comparison`

```
cd python
MTAP_DATA_ROOT=/Volumes/Backups/MTap-Data uv run python scripts/check_feed_quality.py --dates 2026-10-14 2026-10-15 2026-10-16
#   without --dates: every date with both feeds, so pass --dates to reproduce these numbers
#   [--top 300] [--small-max-mm 2000]
```

- **`MTAP_DATA_ROOT`** must point at the mounted NAS. Otherwise the script reads the repo-local data root.
- **Run time** is several minutes for three dates.
- **API key.** The script needs the Massive API key for the grouped-daily calls: 2 per date, one for the date and one for the previous session's close.
- **Output** goes to `/tmp/feed_quality/`:

  | File | Contents |
  |---|---|
  | `lag_scan.csv` | Lag scan |
  | `volume_crossing.csv` | Volume-crossing times |
  | `cross_times.csv` | Opening and closing auction times |
  | `per_symbol_day.csv`, `per_tier.csv` | Price and volume |
  | `price_mismatches.csv` | B2: where differing prices show up, per date, tier and category |
  | `price_mismatch_examples.csv` | B2: sampled differing polls, with the Massive bars around each |
  | `vwap_windows.csv` | VWAP windows |
  | `daily.csv` | Open, close, end of day |
  | `massive_semantics.csv`, `massive_last_bar.csv` | VWAP field checks |
  | `coverage_gaps.csv` | Only written when there are gaps |
