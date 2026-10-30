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
| `tag-sourcing-host` | **Topic: Where the sourcing runs.** The iMac, the NAS containers, later a VM: what differs between hosts |
| `tag-ml-daily` | **Topic: Daily model experiments.** Training runs of the daily model (`ml/tft_vwap`) compared on the same test days |
| `tag-chartiq` | **Topic: Live pre-market sources.** E*TRADE ChartIQ, the watchlist feed and Massive's snapshot before the open: what each gives, how fast |
| `tag-edgar-results` | **Topic: Results release dates (SEC EDGAR).** Which 8-Ks are earnings releases, which session they date, and how that was checked |

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

## Where the sourcing runs

Tags: `tag-sourcing-host`

The iMac ran all the sourcing until 2026-10-25; the NAS containers take over from
2026-10-26 ([nas-migration.md](nas-migration.md)). These sections record what differs
when the same code runs on another host.

### 2026-10 NAS trial (2026-10-24, a Saturday)

Tags: `tag-sourcing-host`

Run with the runbook's trial steps ([deploy/synology/README.md](../deploy/synology/README.md))
into `MTap-Trial`, with only the `etrade` role (no market: login, polling and the file flow,
not market data).

**E*TRADE login: refused, then passed.**

- **Refused.** E*TRADE runs Akamai Bot Manager (the `_abck` and `bm_sz` cookies). With
  puppeteer's Chrome for Testing 127, headed on Xvfb, about ten logins stayed on the login
  page with "We're unable to process your request". A login typed by hand in the same
  noVNC window failed too. A normal browser on the same home IP logged in, so the block was
  on the browser, not the IP or the account.
- **What a page could read** (a probe through `Browser.ts`, against example.com):

  | | iMac (headless, macOS, CfT 127) | NAS before (headed, Linux, CfT 127) | NAS after (headed, Linux, Chrome 155) |
  | --- | --- | --- | --- |
  | `navigator.userAgentData` brands | `[]` | `[]` | Google Chrome 155, Chromium 155, platform Linux |
  | window outer vs inner (1280×800) | 1200×955 | 945×1060 (outer smaller than inner) | 1280×885 |
  | WebGL | none | none | none |
  | stealth `user-agent-override` | on | overridden by `page.setUserAgent` | off |

- **Why the hints were empty.** `page.setUserAgent(ua)` without metadata drops the client
  hints. Skipping only that call was worse: stealth's `user-agent-override` then claimed
  Windows (`Win32`, a Windows UA) on a Linux browser.
- **The fix**, for headed runs only:
  - no UA override, and stealth's `user-agent-override` evasion off;
  - Google Chrome 155 (`PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome`);
  - a fresh profile (the flagged one is `MTap-State/etrade/chrome-profile.cft127-2026-10-24`).

  The next login reached "confirm your identity" at once. After the texted code was entered
  in noVNC, it polled.
- **The iMac** is unchanged: headless, with the override. It logs in with the same empty
  hints, so its aged profile (and macOS) probably carry it.

**The E*TRADE file flow matches the store's counts.** Two runs (20:41 and 20:50 PT) wrote two
segments, of 60,578 and 84,266 rows. `intraday summary` read 144,844 rows from the two
uploaded files: 135,138 Experimental (606 symbols) and 9,706 MidCap (422). The restart at
20:50 needed no login: the profile's session was still valid.

**Massive daily files match MTap-Data apart from the fetch time.**
`python3 python/scripts/compare_daily_roots.py /Volumes/Backups/MTap-Trial /Volumes/Backups/MTap-Data`
compares the NAS's fetches (made 2026-10-25 03:56Z) with the iMac's (each the next
morning, or the 10-18 backfill), by ticker and field.

| file | tickers in both | differ | what differs |
| --- | --- | --- | --- |
| daily_summary 10-12, 10-13, 10-16 | 12,584–12,630 | 14–17 | all OHLCV fields, by a whole factor: reverse splits |
| daily_summary 10-14 to 10-22 | 12,584–12,670 | 1,068–2,901 | mostly `volume` and `transactions`, by a little: late corrections; plus the split rows |
| extended_hours 10-12 to 10-16 | 5,161–5,204 | 11–54 | split rows; 105–115 tickers only in MTap-Data (the backfill used the full universe, delisted stocks included) |
| extended_hours 10-19 to 10-22 | 5,158–5,198 | 0–70 | single ticks and minute counts; QETA only on the NAS |

**Examples:**

| ticker | file | iMac's value | NAS's value | why |
| --- | --- | --- | --- | --- |
| SKIN | daily_summary 10-12 close | 0.525 | 10.5 | ×20: a reverse split after the iMac's fetch; Massive returns adjusted bars |
| AMTD | daily_summary 10-12 close | 0.4173 | 25.038 | ×60, the same |
| MBRX | extended_hours 10-12 pre-market price | 0.4498 | 8.996 | ×20, the same |
| AMT | daily_summary 10-14 volume / trades | 3,118,771.32 / 53,672 | 3,118,770.59 / 53,671 | a late correction |
| SLSN | daily_summary 10-20 volume / low | 18,407.44 / 0.869 | 18,795.44 / 0.851 | late prints added after the iMac's 06:40 fetch |
| INTU | extended_hours 10-19 after-hours volume / bars | 265,510.66 / 432 | 260,610.66 / 431 | a later revision of the minute bars |

The test tickers MTEST, MTEST.A, NTEST and NTEST.H appear only in the NAS's daily files for
10-14 to 10-22. Massive listed them after the iMac's fetch.

So the host makes no difference; the fetch time does. A rebuild of history from a later
fetch carries the split adjustments (`split_factor` in `daily_bars` handles that) and
Massive's corrections.

**Massive WebSocket from the NAS.** `ws-test` (120 s) authenticated and subscribed to `A.*`.
Massive restarted mid-test: a 1012 close, then two HTTP 503s. The client reconnected after
1, 2 and 4 s and re-subscribed. No bars, as expected on a Saturday.

**Login with the password only (step 9, 21:20–21:31).**

- **First `start`:** it reused the saved session; the 21:20 browser went straight to
  polling.
- **Log-off by hand:** after a log-off in the window, the watchlist calls returned 401. The
  collector's recovery navigation was aborted (`ERR_ABORTED`), and it logged `CRITICAL:
  Session recovery FAILED`.
  - Then it stopped polling and stayed up, idle (`EtradeService.handleSessionExpired` →
    `stopMonitoring`). So supervisord (and launchd on the iMac) did not restart it.
- **After `stop` and `start`:** `Login attempt 1 of 3` → `Login successful` → `No phone
  verification needed`, then polling. The Chrome profile keeps the device trust across a
  log-off, so a weekday start should need no code.

**Sunday 10-25: logins after the containers ran overnight.**

- **06:02, 70 failed logins:** the trial containers were still up at the 22:10 power-off.
  `restart: unless-stopped` brought them back at boot, and every login stayed on the login
  page.
  - **Cause:** Chrome had saved the login typed by hand in the window on 10-24. Chrome
    filled it, and puppeteer's typing appended to it, so the ID and password were doubled.
  - **A red herring:** the quotes in `~/.mtap.env`. Compose (v2.20) strips matching quotes
    from `env_file` values; the selftest showed the username with no quotes.
  - **A filled password reads as empty** to the page until the first keystroke commits it.
    So clearing the field before typing missed it: the field held 20 characters after 10
    were typed. `Authentication.typeInto` now checks the length, clears the value that is
    now visible and types once more; it never clicks "Log on" with the wrong length.
  - **What fixed it today:** deleting the saved password in the window
    (`chrome://password-manager`) and turning off "Offer to save passwords". The next
    attempt (11:42) logged in with no code.
- **A log-off gave a refresh loop:** after a log-off the session cookies stay, so the
  navigation step of the recovery declared success on the login page and repeated.
  - **Fix:** the step now needs a page that is not the login page, and it runs at most
    once a minute. If the recovery fails, the collector exits and the restart loop starts
    a fresh one.
- **"Profile in use on another computer":** Chrome's `SingletonLock` names the host and
  pid. A rebuilt container has a new hostname, so the old container's lock blocked the
  profile.
  - A plain `stop` and `start` reuses the container (the same hostname), and Chrome clears
    its own stale lock. Only `build` or a config change makes a new container.
  - **Fix:** `run-etrade.sh` removes the lock at container start when it names another
    host.

**The cutover build (11:50–12:30).** On the trial, a new container logged
`removing a stale Chrome profile lock from host 43e4feedf158`, then reused the session with
no login. Chrome does not remove its lock when the container is stopped (SIGTERM), so every
new container finds one. Then the production `daily` wrote 10-23 (12,557 rows; 10-22 had
12,605).

**Not covered** (no market): a full day's volume, a 09:00–16:21 ET massive-ws window, and the
evening compact on real data. The Monday 10-26 checks in the runbook cover them.


## Results release dates (SEC EDGAR)

Tags: `tag-edgar-results`

Earnings windows in the daily panel (`is_earnings_day`, `sessions_to_earnings`, the 8-K windows of
[ml/FEATURES.md](../ml/FEATURES.md)) are dated by the company's **results announcement**, not the
quarter end and not the 10-Q/10-K filing. The source is `sourcing-py etrade fetch-edgar` →
`sec_edgar_filings` and `results_announcements` in `state/etrade/fundamentals.duckdb`. Checks
were inline DuckDB queries (commit 13dc536, 2026-10-10); the quarter-end lag was added on
2026-10-27.

### 2026-10 Which 8-Ks are releases, and which session they date (EDGAR 2021-10 → 2026-10)

Tags: `tag-edgar-results` `tag-timestamps`

**Short answers:**
- **Only 8-K item 2.02 ("Results of Operations and Financial Condition") is used.** The filer ticks
  it, and it covers quarterly and annual results alike (there is no separate annual item). Every
  other 8-K item (9.01 exhibits, 7.01 Reg FD, 8.01 other events, 5.02 officers, 1.01 agreements …)
  is ignored. 8-K/A amendments are not releases.
- **The date is the announcement:** the EDGAR acceptance time, mapped to the first session that
  can trade on it (accepted before 16:00 ET → that session; at or after 16:00 → the next one). It
  is **not** the quarter end, which is 4–7 weeks earlier, and not the 10-Q/10-K filing, which is
  later for 30–40% of quarters.
- The price check confirms it: the move peaks on the mapped session t0, 66% of the time.

**Selection.** Snapshot of 148,765 8-Ks; 30.5% carry item 2.02.

| item | 8-Ks | | item | 8-Ks |
|---|---|---|---|---|
| 9.01 exhibits | 119,354 | | 5.02 officers | 26,419 |
| **2.02 results** | **45,443** | | 1.01 agreements | 22,079 |
| 7.01 Reg FD | 39,656 | | 5.07 votes | 12,459 |
| 8.01 other | 35,153 | | 2.03 debt | 8,192 |

- 2.02 comes with 9.01 (the press release exhibit) in 67.5%, with 7.01 + 9.01 in 18.4%, with 8.01 +
  9.01 in 4.5%, alone in 1.7%.
- 2.02s per company per year: 4 for 2,475 companies; 5 for 442 (a prelim or update); 0–3 for 584
  (new listings, delistings, or foreign filers); 6+ for ~170.
- **Scheduled vs unscheduled.** The scheduled release of a quarter is the last 2.02 before the
  first 10-Q/10-K within −1…+50 sessions (`REPORT_MAX`). The rest (prelims, guidance updates) are
  **unscheduled** and get post-event windows only, because they can't be known in advance. Counts:
  42,726 scheduled (3,720 symbols), 2,835 unscheduled.
- Timing of the releases: after the close 26,074, before the open 17,686, during market hours 1,801.

**Release vs quarter end vs filing** (2.02 matched to the first 10-Q/10-K filed −1…+75 days after it
with a period end before it; releases since 2024-01; added 2026-10-27):

| days after the quarter end | n | P5 | P25 | median | P75 | P95 |
|---|---|---|---|---|---|---|
| 2.02 release, quarterly (10-Q) | 27,311 | 18 | 29 | **36** | 38 | 45 |
| 2.02 release, annual (10-K) | 10,702 | 13 | 34 | **51** | 63 | 88 |
| 10-Q filed | 27,311 | 25 | 32 | 37 | 40 | 45 |
| 10-K filed | 10,702 | 41 | 52 | 58 | 72 | 90 |

- The 8-K's own `report_date` is the event date (= the filing date in 88–91%), not the period end
  (0.2–0.3%). The period end comes only from the 10-Q/10-K.
- 10-Q/10-K filing date minus the latest 2.02 (E*TRADE's filing dates match EDGAR's in 99.88%,
  n = 32,069): 10-Q same day 57.8%, 1–7 days 12%, 8–30 days 12.1%, 31–100 days 4.0%, no 2.02 13.8%;
  10-K same day 42.9%, 8–30 days 23.7%, no 2.02 16.2%. **So the old filing-dated windows were
  wrong for ~30–40% of quarters**, which is why the panel switched to 2.02.

Examples (ET):

| stock | quarter end | 2.02 accepted | t0 session | 10-Q/10-K filed |
|---|---|---|---|---|
| AAPL | 2025-09-27 | 2025-10-30 16:30 | 10-31 | 10-K 2025-10-31 |
| AAPL | 2025-12-27 | 2026-01-29 16:30 | 01-30 | 10-Q 2026-01-30 |
| JPM | 2025-09-30 | 2025-10-14 06:30 | 10-14 | 10-Q 2025-11-04 (3 weeks later) |
| JPM | 2025-12-31 | 2026-01-13 06:41 | 01-13 | 10-K 2026-02-13 (a month later) |
| MSFT | 2025-12-31 | 2026-01-28 16:04 | 01-29 | 10-Q 2026-01-28 (same day) |
| NVDA | 2026-01-25 | 2026-02-25 16:31 | 02-26 | 10-K 2026-02-25 |
| WMT | 2026-01-31 | 2026-02-19 06:59 | 02-19 | 10-K 2026-03-13 (3 weeks later) |

**Validation against price moves.** Median |close-to-close move| by session offset from t0,
scheduled releases with an acceptance time (n ≈ 10,250):

| offset | t−3 | t−2 | t−1 | **t0** | t+1 | t+2 | t+3 |
|---|---|---|---|---|---|---|---|
| median move | 1.63% | 1.65% | 1.81% | **4.86%** | 2.38% | 2.00% | 1.79% |

- t0 is 5.6× a normal day, and 49% of t0 moves are over 5%.
- The same after-close releases dated by their filing date instead: t0 1.93%, t+1 5.13%. That is
  the one-session error the acceptance-time rule fixes.
- Unscheduled releases: t0 3.68%. Old E*TRADE 10-Q dates: t0 2.71%, t+1 2.69% (the move smeared
  over two days).
- Which of t−1/t0/t+1 has the biggest move (liquid stocks, > $20M/day, n = 4,814): t0 66.0%, t−1
  14.5%, t+1 19.4%. Before the open 67.0%; after the close 65.6%; market hours (n = 75) 53.3%.

**Acceptance time: use the header page, not the JSON.** The submissions JSON
`acceptanceDateTime` is unreliable and depends on the filer agent: AAPL 16:30:28 ET on 2026-07-30
reads `2026-07-31T00:30:28Z` (+8 h, not +4), AAPL 2026-01-29 +10 h, JPM 06:30:38 reads 14:30:38Z;
Alphabet 21:15:54 reads 01:15:54Z (right). AAPL's own-prefix filings give an impossible time in 399
of 459. The `<ACCEPTANCE-DATETIME>` in `-index-headers.html` is ET and right in every spot check, so
that is what is stored (`accepted_at`, ET).

**Coverage** (3,763 panel symbols): 79% have 2.02s, 3% only partly, 18% none.
- ADRs (DR): 252 of 265 have none. Foreign private issuers file 6-K, not 8-K. (ADRs are out of
  the panel since 2026-10-27.)
- Common stocks: 422 of 3,498 have none: 323 companies file 10-Qs but never tick 2.02, 35 tickers
  aren't in SEC's ticker list (BK → BNY), 7 are banks that file elsewhere (OZK, PFBC). These
  have no earnings windows (every day is Normal).
- **Point-in-time estimate** (`days_to_results_est`, `--earnings pit`): the next release is
  expected at the year-ago release + 364 days (same weekday), counted as Eve while up to
  `OVERDUE_MAX` = 10 sessions overdue. It uses only releases made by the day. Against the real
  dates it is exact 37.5%, within ±1 session 53.4%, within ±5 83.6% (n = 23,781).
- Wider CS universe: 5,056 symbols, 71.1% with a 2.02.

**Fetch cost.** EDGAR allows 10 requests/s. A first run is ~2 h at 8/s (5,358 CIK submissions ~11
min, then one header page per 2.02, 45k pages ~95 min); a daily run only fetches new filings. The
2026-10-27 5-year load (51,115 pages) ran with `EDGAR_WORKERS=16 EDGAR_REQUESTS_PER_SEC=9`.

**How to run it again.**
```sql
-- releases per session offset: results_announcements joined to daily_bars on (symbol, t0 ± k)
SELECT kind, timing, count(*) FROM results_announcements GROUP BY ALL;
-- release vs quarter end: 8-Ks with list_contains(string_split(items, ','), '2.02'), joined to the
-- first 10-Q/10-K of the same cik filed −1…+75 days later with report_date < the 8-K's filing_date
```

**Caveats and follow-ups.**
- Releases on the item 2.02 checkbox only: a company that announces in a 7.01/8.01 8-K without
  ticking 2.02 is missed.
- Incremental daily-index fetch (instead of per-CIK submissions) is still open.

## Live pre-market sources

Tags: `tag-chartiq`

What the live sources give before and at the open, for the same-morning forecasts of the daily
model: E*TRADE ChartIQ (`getChartiqFeed`), the E*TRADE watchlist feed, and Massive's snapshot and
minute aggregates. Probed by hand on 2026-10-27 with the collector's token (read-only, no login).
The design they led to is in [ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q14.

### 2026-10 ChartIQ, watchlist and Massive before the open (session 2026-10-27)

Tags: `tag-chartiq` `tag-price` `tag-volume` `tag-timestamps`

**Short answers:**
- **ChartIQ gives prices, not volume.** Its minute-bucket volume is about half of Massive's, and
  the share moves by stock and by day. Prices agree. It is used for prices only.
- **The watchlist's bid/ask mid is the live pre-market price** (within ~20 bp of Massive); its
  `price` stays at the previous close until the first regular trade.
- **Massive's delayed minute bars reach 09:30 by 09:45** for every stock, fast enough (120
  calls/s) to give the full pre-market with training's own definitions.
- So the 09:30 forecast uses prices only, and the 09:45 forecast the full pre-market from
  Massive. Training uses Massive only.

**ChartIQ (`POST getChartiqFeed`, bearer token, origin/referer etrade.wallst.com):**
- Bars are fixed **86.4 s buckets** whatever `dataInterval` says: OHLC and volume, no VWAP.
- A call returns about 2 sessions whatever `dateStart` says (TSLA: 1,200 bars, 127 KB, 0.48 s);
  filter by time yourself. The `Second` period is empty; `Day` works (23 bars, 0.28 s).
- Sessions: pre 04:00–09:30, regular 09:30–16:00, post 16:00–20:00 ET.
- **The updated agreement's token is real-time:** `isRealTimeQuote: true`, newest bucket 20 s
  old. The old token was about 15 min behind.
- **Several symbols per call:** prices for 20 symbols in one call (3.0 s, 2 MB). The first symbol
  sits at the top level of each bar, the rest are nested under their XID. **Only the first symbol
  keeps its volume**, in every layout tried (price/volume pairs, all prices then all volumes,
  volume only, lowercase `volume`).
- The XIDs from `symbol-lookup` match the feed's own `symbolInfo` for all 20 (205778 → MSFT …).
- **Throughput, one symbol per call, 20 symbols:**

  | at a time | wall time | calls/s | all returned |
  | --- | --- | --- | --- |
  | 4 | 3.4 s | ~6 | yes |
  | 8 | 3.7 s | ~5 | yes |
  | 16 | 2.0 s | ~10 | yes |

  Median latency 0.5 s, no 429 at 16.

**ChartIQ volume vs Massive** (ChartIQ ÷ Massive, 10-27):

| stock | pre-market buckets | regular buckets | Day bar |
| --- | --- | --- | --- |
| TSLA | 0.62 | 0.59 | 0.99 |
| AAPL | 0.53 | 0.52 | 1.00 |
| META | 0.61 | 0.49 | 1.00 |
| AMD | 0.58 | 0.48 | 1.00 |
| F | 0.66 | 0.58 | 1.00 |

- The buckets carry only part of the prints. Scaling by the previous day's share still left
  errors of −7% to −43%, so no learned factor is used.
- Prices agree because any sample of trades sits at the same level: the last price is the same,
  and the typical-price VWAP over the buckets differed by 0.1–28 bp.
- Whether the Day bar is consolidated *during* pre-market is not checked (and no longer needed).

**Watchlist feed before the open** (NAS `intraday_feed`, 1,028 symbols, at 09:28:59 ET):

| | TSLA | AAPL | AMD | F | META |
| --- | --- | --- | --- | --- | --- |
| `price` (all = previous close) | 391.62 | 343.91 | 531.28 | 11.58 | 630.56 |
| bid/ask mid vs Massive's last pre-market trade | +6 bp | −8 bp | +17 bp | −21 bp | −2 bp |
| `volume_cum` ÷ Massive pre-market minute volume to 09:28 | 1.09 | 1.10 | 1.14 | 1.02 | 1.03 |

- The bid/ask mid is on the Experimental list only (606 symbols).
- `volume_cum` is live pre-market volume for 1,011 symbols, 2–14% above Massive's minute bars
  (Massive's bars leave out some prints, `tag-massive-etrade-comparison`). Good for ranking, not
  a model input.
- `day_open` is null until the open; then `price` turns live.

**Massive:**
- **Full-market snapshot** (`/v2/snapshot/locale/us/markets/stocks/tickers`): 13,271 tickers in
  one call, 2.2 s. Per ticker `day` (o, h, l, c, v, vw), `min` (latest minute, `av` running
  volume), `prevDay`, `todaysChangePerc`. 15 min delayed; `lastTrade` is null.
- **Minute aggregates, one stock per call:** 120 calls/s at 64 at a time (median 0.42 s, p95
  0.48 s, no throttling); 35/s at 16. ~2k liquid stocks in ~17 s, ~3.7k panel stocks in ~30 s.
- **Same-day daily bars** were served by 17:20 ET on 10-27 (status `DELAYED`), not only after
  midnight ET.

**Caveats and follow-ups:**
- One morning, a handful of stocks. S1 (one live morning, read-only) checks: what the snapshot's
  `day`/`min` hold at 09:29:30; the shortlist's live prices vs Massive's last pre-market trade the
  next day (accept if the median gap is under 10 bp on liquid names); when `day_open` and
  ChartIQ's first regular bucket appear; and `premarket full` at 64 at a time with no 429.
- Check once that Massive's same-day daily bars don't change after 20:00 ET (tonight's copy vs
  tomorrow's).

## Daily model experiments

Tags: `tag-ml-daily`

Training runs of the daily model (`ml/tft_vwap`) compared with each other. The setup, the
parameters and the candidate inputs are in [ml/FEATURES.md](../ml/FEATURES.md), sections 12–17.
This topic keeps what each comparison found.

### 2026-10 5-year retrain series R0–R8 (R0, R1, R1a, R1c, R6c done; R1c is the base; R4 running)

Tags: `tag-ml-daily`

The series that tests the new inputs and the same-morning forecasts on 5 years. The reasons are
in [ml/MODEL-STRATEGY.md](../ml/MODEL-STRATEGY.md) Q11–Q17; the inputs in
[ml/FEATURES.md](../ml/FEATURES.md).
- **Panels** (schema 4, 4,279,044 rows, 4,356 CS stocks, 2021-10-21 → 2026-10-26):
  - R0: `out/tft/exp5y/panel.parquet` (exported 10-28 03:08Z, `1c55e0b`).
  - R1 on: `out/tft/exp5y-r1/panel.parquet` (10-28 17:20Z). Same rows, and every column R0
    uses is identical; only the new groups differ. **The R0 panel's macro and after-hours
    inputs were all 0**: the yields, inflation and refetched extended-hours files were on the
    NAS but had not been loaded into the DuckDB before the export (`ingest-reference`,
    `ingest-eod --extended-only`, `derive-features` fixed it). R0 doesn't use them, so R0 is
    unaffected.
  - Also fixed on the way: CPI for 2025-10 was never published (shutdown), so every December
    2025 session had `cpi_yoy` = 0. A missing month now carries the previous one for up to 2
    months (`CPI_CARRY`), and `core_cpi_mom` across the gap is the per-month rate.
- **Testing:** 15 walk-forward folds of 63 sessions, 2023-01-20 → 2026-10-26, each with a
  21-session early-stop block in front and at least 252 training sessions before fold 0
  (`--n-folds 15 --fold-len 63 --early-stop-len 21 --min-train-sessions 252 --test-end 1713`).
  Fold 0 trains on about 1.2 years, fold 14 on about 4.7. Ablations and the pre-market bake-off
  use the 7 odd folds (1, 3, …, 13).
- **Trained models** are not in git (`ml/runs/` holds 3 GB of checkpoints and predictions);
  they are copied to the NAS at `PProjects/MTap/ml/runs/`.

| run | inputs | folds | result |
| --- | --- | --- | --- |
| R0 `20261028-030847Z-return-4438sym` | 5-year baseline, no new groups | 0–14 | skill vs naive 26.6%; beats production by 6.2% on shared days (CI clear of 0) |
| R1 `20261028-172535Z-return-4438sym` | R0 + market/sector + macro + after-hours + session shape | 0–14 | worse than R0: loss +2.5% (CI clear of 0), worse in 7 folds, better in none (section below) |
| R1a `20261028-214331Z-return-4438sym` | R1 − macro | 0–6 (paused after fold 6, 10-28 16:32, to start R1c) | worse than R0: +0.000209 (+0.000116 … +0.000318), worse in folds 1 and 5, better in none; no clear difference from R1; P10–P90 74.9% vs 79.6%; market's encoder weight rises to 17–49% |
| R1c `20261028-233305Z-return-4438sym` | R1 − macro − market (after-hours, session shape) | 0–14 | **better than R0:** −0.58%, R0 − R1c +0.000046 (+0.000021 … +0.000071); earnings days +0.000339 (+0.000287 … +0.000387); better in 7 folds, worse in none; skill 27.0%. **The base** for R4 on (section below) |
| R6c `20261028-030847Z-return-4438sym-event` | event overlay on R0: band scale from yesterday's market, sector-vs-market and 2y/10y yield moves beyond 2σ | 1–14 | no gain: −0.000002 (−0.000004 … −0.000000), R0 slightly better; coefficients flip sign between folds (section below) |
| R4 `20261029-040858Z-return-4438sym` | A: R1c + pre-market `optional` (25% night, 25% prices only, 50% everything) | 7 odd | done 10-29 00:06. Night state ties R1c (+0.10%, CI spans 0; earnings days worse); 09:30 and 09:45 states lose to R6p/R6f (section below) |
| R4p `20261029-090243Z-return-4438sym` | B: R1c + prices always on, no `issue_type` | 7 odd | 0.70% better than R4~px (CI clear), 0.35% worse than R6p (CI clear) |
| R4f `20261029-122046Z-return-4438sym` | B: R1c + prices and full always on, no `issue_type` | 7 odd | done 10-29 08:17 PT: 09:45 loss 0.007938, 1.0% better than R4, 0.55% worse than R6f (both CIs clear); P5–P95 89.8% |
| R6p, R6f `20261028-233305Z-return-4438sym-overlay-px`, `-overlay-full` | C: HGB overlays on R1c (prices; everything) | 7 odd | **leading**: R6p 1.05% better than R4~px, R6f 1.55% better than R4 (CIs clear) |
| R7, R7f | plain HGB per quantile, R1c inputs; R7f + pre-market | 0–14; 7 odd | queued |
| R5 | the final model(s), the winning way per forecast | 0–14 | |
| R8 | screener on R5's out-of-fold forecasts vs naive ATR and others | 7 odd | |

**How each is judged:**
- `tft-vwap-compare` against R0, and against production `20261005-025909Z` on the shared dates
  (2025-09 → 2026-08): paired per (symbol, date) pinball difference with a bootstrap CI, per
  fold, earnings vs other days, coverage per quantile.
- **R1's groups:** variable-selection weights per group and fold, and R1 vs R0 per fold. A
  drop-one run only if R1 loses somewhere or a group has near-zero weight (macro first).
- **Pre-market, per forecast** (night, 09:30, 09:45), on the 7 odd folds and rows with pre-market
  data: night R4 vs the base (R1c); 09:30 R4 vs R4p vs R6p; 09:45 R4 vs R4f vs R6f. Lowest pinball wins
  when its CI is clear; otherwise fewer models (A, then C, then B).
- **Screener (R8):** kept only if its top 20 beat the naive ATR top 20 with a CI clear of 0.

### 2026-10 R0, the 5-year baseline (test sessions 2023-01-20 to 2026-10-26)

Tags: `tag-ml-daily`

**The run.** `20261028-030847Z-return-4438sym`, from `uv run tft-vwap --panel
../out/tft/exp5y/panel.parquet --earnings exact --n-folds 15 --test-end 1713` (folds 63/21/252).
- Stock inputs, plus exact earnings windows and series breaks; no market, macro, after-hours,
  session-shape or pre-market inputs.
- 15 folds, 4–13 epochs each (early stop, patience 3), 3.9 h of training in all. Paused
  twice and resumed (`--resume ... --num-workers 8`); resumed folds match the others.

**Pooled test** (3,169,270 forecasts, 4,218 stocks):

| measure | R0 |
| --- | --- |
| pinball loss | 0.007884 (naive 0.010738): skill 26.6% |
| P5–P95 coverage (aim 90%) | 88.9% (5.5% below P5, 5.6% above P95) |
| direction of `q50` | 59.1% |
| MAPE skill vs naive | 5.1% |

**By fold.** Skill is steady across regimes, from 24.4% to 28.4%. The 2023 folds are the
weakest: they train on only 1.2–1.7 years.

| fold | test sessions | skill | P5–P95 | direction |
| --- | --- | --- | --- | --- |
| 0 | 2023-01-20 → 04-20 | 26.4% | 90.8% | 60.4% |
| 1 | 2023-04-21 → 07-21 | 24.4% | 89.9% | 57.0% |
| 2 | 2023-07-24 → 10-19 | 24.4% | 88.2% | 56.8% |
| 3 | 2023-10-20 → 2024-01-22 | 26.5% | 88.7% | 59.5% |
| 4 | 2024-01-23 → 04-22 | 26.2% | 87.8% | 59.3% |
| 5 | 2024-04-23 → 07-23 | 27.5% | 89.2% | 59.4% |
| 6 | 2024-07-24 → 10-21 | 26.1% | 90.0% | 57.8% |
| 7 | 2024-10-22 → 2025-01-23 | 27.3% | 87.6% | 60.2% |
| 8 | 2025-01-24 → 04-24 (April sell-off) | 26.8% | 88.0% | 60.5% |
| 9 | 2025-04-25 → 07-25 | 25.5% | 92.6% | 58.5% |
| 10 | 2025-07-28 → 10-23 | 26.8% | 87.1% | 59.0% |
| 11 | 2025-10-24 → 2026-01-26 | 28.4% | 90.7% | 60.8% |
| 12 | 2026-01-27 → 04-27 | 27.4% | 87.0% | 59.3% |
| 13 | 2026-04-28 → 07-28 | 27.6% | 88.1% | 59.0% |
| 14 | 2026-07-29 → 10-26 | 26.8% | 87.5% | 58.8% |

**Variable importance** (encoder, mean over folds): stock inputs 0.81, earnings 0.19 (from 0.12
in fold 0 to 0.31 in fold 3).

**R0 vs production** `20261005-025909Z-return-3712sym` (`tft-vwap-compare`, matched on
(symbol, date): 756,726 forecasts on 240 days, 2025-09 → 2026-08):

| | rows | R0 | production | difference (95% CI) |
| --- | --- | --- | --- | --- |
| all | 756,726 | 0.008292 | 0.008845 | −0.000550 (−0.000754 … −0.000389), −6.2% |
| earnings days | 58,370 | | | −0.001420 (−0.001799 … −0.001122) |
| other days | 698,356 | | | −0.000484 (−0.000685 … −0.000325) |

| R0 fold | rows | R0 | production | difference (95% CI) |
| --- | --- | --- | --- | --- |
| 10 | 116,961 | 0.007468 | 0.007591 | −0.000123 (−0.000216 … −0.000023) |
| 11 | 197,023 | 0.007569 | 0.007911 | −0.000342 (−0.000511 … −0.000206) |
| 12 | 200,340 | 0.008798 | 0.010068 | −0.001269 (−0.001985 … −0.000731) |
| 13 | 200,951 | 0.008800 | 0.009132 | −0.000332 (−0.000492 … −0.000191) |
| 14 | 41,451 | 0.009140 | 0.009510 | −0.000370 (−0.000481 … −0.000257) |

- Coverage on the shared rows: R0 P10–P90 77.8%, P5–P95 88.3%; production 74.6%, 86.6%.

**What it shows:**
- R0 beats production in every shared fold, with every CI clear of 0. The gain is largest on
  earnings days (2.6× the other days' gain per forecast), where production has no earnings inputs.
- R0 is also better calibrated (P5–P95 88.3% vs 86.6%), though still a little narrow against 90%.
- R0 is the reference for R1 and the rest of the series. A switch of production to R0 is open
  (MODEL-STRATEGY open decision 4).
- `ml/runs/20261028-030847Z-return-4438sym/compare.json` holds the full comparison.

### 2026-10 R1, all always-available inputs, vs R0 (test sessions 2023-01-20 to 2026-10-26)

Tags: `tag-ml-daily`

**Run:** `20261028-172535Z-return-4438sym`. Same panel rows and folds as R0, plus
`--market --macro --session-shape --after-hours`, 8 workers. 10-28 10:25 → 14:43 PT.

**Overall** (3,169,270 forecasts matched on (symbol, date), 945 days):

| | rows | R1 | R0 | difference (95% CI) |
| --- | --- | --- | --- | --- |
| all | 3,169,270 | 0.008086 | 0.007884 | +0.000200 (+0.000142 … +0.000262) |
| earnings days | 238,489 | | | +0.000192 (+0.000120 … +0.000262) |
| other days | 2,930,781 | | | +0.000198 (+0.000138 … +0.000260) |

- **Coverage:** R1 P10–P90 75.7%, P5–P95 87.2%; R0 78.9%, 88.9%. R1's bands are too narrow.
- **Per fold:**
  - R1 is worse in folds 0, 1, 5, 11, 12, 13 and 14, by the most in fold 0 (+0.0011) and fold 12
    (+0.0006, the April 2026 regime).
  - No fold is clearly better. Fold 3 is the only one that leans R1's way, and its CI spans 0.
- **Early stopping:** R1 stops at 4–6 epochs in every fold. R0 used the same 4–6 in folds 0–8 but
  7–13 in folds 9–14. With more training years R0 keeps improving; R1 does not.
- **Encoder weight per group (%), per fold:**

  | fold | base | earnings | market | macro | session shape | after-hours |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0 | 18 | 7 | 32 | 32 | 4 | 7 |
  | 1 | 20 | 9 | 43 | 12 | 5 | 12 |
  | 5 | 10 | 6 | 22 | 14 | 2 | 46 |
  | 7 | 23 | 6 | 18 | 19 | 4 | 30 |
  | 12 | 30 | 12 | 31 | 14 | 6 | 7 |
  | 14 | 27 | 10 | 27 | 11 | 10 | 15 |
  | range, all 15 | 10–58 | 4–17 | 15–43 | 7–32 | 2–16 | 5–46 |

**What it shows:**
- The inputs are correct (market return vs SPY correlation 0.945; CPI and yields filled). The loss
  comes from how the model uses them.
- Market and macro take a third to two-thirds of the encoder weight. Each has one value per day,
  shared by every stock, so the model can use them to tell training days apart (memorising
  dates) rather than learning something that carries to new days. The weight is largest in
  fold 0, which has the least training data, and that is R1's worst fold.
- After-hours swings from 5% to 46% between folds, so it is not a stable signal either.
- Next: the drop-one runs **R1a** (R1 − macro) and **R1c** (R1 − macro − market), same folds,
  queued on this Mac (`logs/r1-variants.sh`). R0 stays the reference and the source of the
  intraday handoff table until one of them beats it.
- `ml/runs/20261028-172535Z-return-4438sym/compare.json` holds the full comparison.

### 2026-10-29 Direction and move accuracy per model, vs one-input rules

Same common rows as the section below. Direction = sign of P50 vs the actual move (rows where both
are nonzero). Big moves = |actual| > 2%. Confident calls = forecast P(up) ≥ 65% or ≤ 35%.
Rank IC = Spearman correlation per day between P50 and the actual move across stocks, averaged over
days (1 = perfect ordering, 0 = none). Scripts: `ml/scripts_direction.py`, `ml/scripts_direction_naive.py`.

| model | rows | direction | big moves | confident calls (share / hit) | median \|P50 err\| | rank IC |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | night, folds 1–14 | 59.0% | 65.6% | 30% / 68.2% | 1.26% | 0.30 |
| R1 | night, folds 1–14 | 58.7% | 65.5% | 38% / 65.4% | 1.29% | 0.30 |
| R1c | night, folds 1–14 | 59.6% | 66.7% | 32% / 68.3% | 1.26% | 0.31 |
| R6c | night, folds 1–14 | 59.0% | 65.6% | 30% / 68.2% | 1.26% | 0.30 |
| R0 | pre-market rows, odd folds | 60.1% | 66.4% | 31% / 70.3% | 1.48% | 0.31 |
| R1c | pre-market rows | 60.7% | 67.6% | 35% / 70.0% | 1.48% | 0.33 |
| R4~night | pre-market rows | 60.4% | 67.0% | 31% / 70.2% | 1.48% | 0.32 |
| R4~px 09:30 | pre-market rows | 70.1% | 80.5% | 58% / 78.7% | 1.26% | 0.53 |
| R4p 09:30 | pre-market rows | 69.9% | 80.3% | 61% / 78.1% | 1.26% | 0.53 |
| R6p 09:30 | pre-market rows | 70.8% | 81.1% | 60% / 79.3% | 1.24% | 0.54 |
| R4 09:45 | pre-market rows | 70.1% | 80.5% | 58% / 78.6% | 1.26% | 0.53 |
| R4f 09:45 | pre-market rows | 70.3% | 80.7% | 60% / 78.8% | 1.26% | 0.53 |
| R6f 09:45 | pre-market rows | 70.8% | 81.1% | 60% / 79.4% | 1.24% | 0.54 |

**One-input rules on the pre-market rows** (forecast direction = sign of that input):

| rule | known at | direction | big moves | rank IC |
| --- | --- | --- | --- | --- |
| open vs previous VWAP | 09:30 | 70.7% | 81.0% | 0.53 |
| last pre-market trade vs previous VWAP | 09:30 | 67.1% | 76.8% | 0.44 |
| pre-market VWAP vs previous VWAP | 09:30 | 66.5% | 76.0% | 0.43 |
| yesterday's close vs its VWAP | night | 61.3% | 67.7% | 0.33 |
| yesterday's after-hours VWAP vs its VWAP | night | 61.2% | 67.7% | 0.32 |

What it shows:
- **No model beats the simplest rule on direction.** At 09:30 "today moves the way the open moved"
  scores 70.7% (models 69.9–70.8%), rank IC 0.53 (models 0.53–0.54). At night "today moves the
  way yesterday closed vs its VWAP" scores 61.3%, above every night model (59.0–60.7%).
- So the models' gain over R0 (skill 27% → 40%) is mostly **band width and calibration** (how far,
  with what uncertainty), not a better direction call. Pinball rewards both; direction ignores size.
- The direction a model is "sure" about is better: 68–70% at night, 78–79% in the morning, on
  30–60% of the rows.
- For the screener (R8) the relevant measure is rank IC and the realised range of the top 20, not
  the direction hit rate.

### 2026-10-29 All models so far vs R0: skill, coverage, calibration (common rows)

Each table scores every model on the same (symbol, date) rows. Skill = 1 − pinball / naive (naive =
every quantile at a 0 move). "vs R0" = pinball change against R0, CI = 95% bootstrap over days.
Bias = 0.5 − mean PIT in points (> 0 optimistic). |q err| = |observed share below quantile − level|
over the 19 quantiles, in points. Script: `ml/scripts_compare_vs_r0.py night|bake`.

**Night forecasts, folds 1–14** (2.95M rows, 882 days; fold 0 left out because R6c starts at fold 1,
so R1c reads −0.79% here vs −0.58% on folds 0–14):

| model | skill | vs R0 (CI) | P5–P95 | P10–P90 | direction | bias (z) | mean / max \|q err\| | grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 26.6% | — | 88.7% | 78.7% | 59.0% | −0.25 (−15) | 0.40 / 0.97 | excellent |
| R1 | 25.3% | +1.72% (+1.12 … +2.39) | 87.5% | 76.1% | 58.7% | −0.75 (−45) | 1.41 / 2.84 | fair |
| R1c | 27.2% | −0.79% (−1.10 … −0.49) | 88.6% | 78.6% | 59.6% | +0.14 (+8) | 0.57 / 0.95 | good |
| R6c | 26.6% | +0.02% (+0.00 … +0.05) | 88.8% | 78.8% | 59.0% | −0.26 (−16) | 0.39 / 0.94 | excellent |
| R1a (folds 1–6 only; R0 there: 25.8%, 1.68) | 23.6% | +2.97% (+1.51 … +4.50) | 86.5% | 75.4% | 57.4% | −1.94 (−76) | 2.29 / 3.95 | poor |

**Pre-market bake-off, odd folds, rows with full pre-market data** (812k rows, 441 days; more active
stocks, so the losses are higher than above):

| model | forecast | skill | vs R0 (CI) | P5–P95 | P10–P90 | direction | bias (z) | mean / max \|q err\| | grade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | night | 26.8% | — | 88.0% | 77.9% | 60.1% | −1.33 (−42) | 1.46 / 2.59 | fair |
| R1c | night | 27.6% | −1.07% (−1.43 … −0.75) | 87.9% | 77.8% | 60.7% | −0.64 (−20) | 0.82 / 2.01 | good |
| R4~night | night | 27.4% | −0.83% (−1.36 … −0.45) | 86.9% | 76.7% | 60.4% | −1.55 (−48) | 1.79 / 3.14 | fair |
| R4~px | 09:30 | 39.2% | −16.9% (−17.7 … −15.6) | 89.0% | 79.5% | 70.1% | −1.25 (−39) | 1.35 / 2.08 | fair |
| R4p | 09:30 | 39.6% | −17.5% (−18.4 … −16.1) | 88.0% | 77.4% | 69.9% | −0.87 (−27) | 1.18 / 2.08 | fair |
| R6p | 09:30 | 39.8% | −17.8% (−18.8 … −16.6) | 90.8% | 81.0% | 70.8% | −0.91 (−28) | 0.99 / 1.72 | good |
| R4 | 09:45 | 39.3% | −17.1% (−17.9 … −15.8) | 88.8% | 79.2% | 70.1% | −1.36 (−42) | 1.47 / 2.20 | fair |
| R4f | 09:45 | 39.9% | −17.9% (−18.8 … −16.6) | 89.8% | 79.9% | 70.3% | −1.00 (−31) | 1.09 / 2.04 | fair |
| R6f | 09:45 | 40.3% | −18.4% (−19.4 … −17.2) | 90.8% | 81.1% | 70.8% | −0.92 (−29) | 1.00 / 1.74 | good |

Grade (proposed 10-29, not in the code): mean |q err| ≤ 0.5 pts excellent, ≤ 1.0 good, ≤ 2.0 fair,
> 2.0 poor. The code's own label (`calibrated` when |bias z| < 2) marks every model optimistic or
pessimistic here: with 0.8–3M rows, a bias of 0.03 points is already "significant".

What it shows:
- **Pre-market is the big lever:** skill 27% → 39–40%, 17–18% lower loss than R0, direction
  60% → 70%. The choice between R4, R4p and R6p/R6f is worth 0.3–1.5% on top of that.
- **All models are slightly pessimistic** (the actual lands a little above the forecast) except
  R1c overall, which is slightly optimistic (+0.14 points).
- **The overlays are the best calibrated** of the morning models, and the only ones near 90% P5–P95.
- On the active-stock rows R0 is only "fair"; R1c is the better-calibrated night model there.

### 2026-10 Pre-market bake-off, first results: R4, R4p, R4f, R6p, R6f (7 odd folds, test sessions in folds 1–13)

Rows with that pre-market data, matched on (symbol, date); 441 test days. Diff = A − B pinball per
day (negative: A better), 95% bootstrap CI over days.

| forecast | A | B | loss A | loss B | A − B (95% CI) | earnings days A − B (CI) | P5–P95 A / B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| night | R1c | R4~night | 0.007863 | 0.007871 | −0.000008 (−0.000049 … +0.000031) | +0.000097 (+0.000031 … +0.000164), R4~night better | 89.2 / 88.5% |
| 09:30 | R6p | R4~px | 0.007953 | 0.008037 | −0.000105 (−0.000134 … −0.000077) | −0.000418 | 90.8 / 89.0% |
| 09:30 | R4~px | R4p | 0.008037 | 0.007981 | +0.000065 (+0.000029 … +0.000103), R4p better | +0.000319 | 89.0 / 88.0% |
| 09:30 | R6p | R4p | 0.007953 | 0.007981 | −0.000040 (−0.000072 … −0.000009) | −0.000099 (−0.000221 … +0.000045) | 90.8 / 88.0% |
| 09:45 | R6f | R4 | 0.007894 | 0.008019 | −0.000144 (−0.000173 … −0.000118) | −0.000578 | 90.8 / 88.8% |
| 09:45 | R4f | R4 | 0.007938 | 0.008019 | −0.000089 (−0.000120 … −0.000058) | −0.000330 (−0.000412 … −0.000250) | 89.8 / 88.8% |
| 09:45 | R6f | R4f | 0.007894 | 0.007938 | −0.000056 (−0.000080 … −0.000033) | −0.000248 (−0.000357 … −0.000140) | 90.8 / 89.8% |

Per fold (A − B):

| fold | night R1c − R4~night | 09:30 R6p − R4~px | 09:30 R6p − R4p | 09:45 R6f − R4 | 09:45 R6f − R4f |
| --- | --- | --- | --- | --- | --- |
| 1 | +0.000021 | −0.000476 | −0.000271 | −0.000512 | −0.000112 |
| 3 | +0.000165 | −0.000214 | −0.000046 | −0.000253 | −0.000155 |
| 5 | −0.000105 | −0.000083 | −0.000010 | −0.000113 | −0.000084 |
| 7 | +0.000013 | −0.000001 | +0.000046 | −0.000078 | −0.000043 |
| 9 | −0.000128 | +0.000087 | −0.000060 | +0.000052 | −0.000086 |
| 11 | −0.000038 | −0.000031 | +0.000068 | −0.000059 | +0.000022 |
| 13 | +0.000014 | −0.000014 | −0.000005 | −0.000049 | +0.000069 (R4f better, CI clear) |

What it shows:
- **Masking costs nothing at night.** R4's night state ties R1c overall and is slightly better on
  earnings days. One masked TFT could serve the night forecast, but R1c stays (no clear gain).
- **The overlays win the morning forecasts.** R6p and R6f beat the masked TFT by 1.0–1.5%. Most of
  the gain is in the early folds (fold 1: about 6%), where a TFT trained on ~1 year has seen
  little pre-market. The gap narrows with more training data, and in folds 7–13 R6p vs R4p is mixed
  (R4p better in folds 7 and 11). The overlays also widen the bands: P5–P95 90.8% vs 88–89%,
  closer to the 90% target.
- **A dedicated TFT (R4p) beats the masked one at 09:30** by 0.70%, so masking costs the
  morning states. R4p runs without `issue_type` and R4 with it (Q27), which Q23 expects to be
  inside the noise.
- **R4f (dedicated 09:45 TFT, 10-29) is second at 09:45.** It beats R4 by 1.0% (CI clear; worse
  only in fold 9) and loses to R6f by 0.55% (CI clear), more on earnings days (−0.000248 vs −0.000036 on other days). The same
  pattern as 09:30: R6f's lead is in folds 1–9; in folds 11–13 R4f is level or better (fold 13 CI
  clear). With more training years the dedicated TFT catches up, so R5's choice should weigh the
  late folds.
- R7 (HGB, night) and R7f (HGB + pre-market) came next: see the R7 entry above.

### 2026-10-29 R5, the single 09:30+ model, and rolling recalibration (20 sessions)

Tags: `tag-ml-daily`

**Runs.** R5 `ml/runs/20261029-211059Z-gbm-4438sym` (09:45) and `…~px` (09:30), one HGB fit
(`--premarket optional --open --vol --tail-iter 600`, folds 0–14, 3.18M forecasts, skill 38.3%),
14:10 → 18:26 PT with recal, tables, R8 and the NAS backup (`ml/runs/queue-r5.sh`). Then
`tft-vwap-recal --sessions 20 --by pm,width` → `…-cal20s` and `…~px-cal20s`. Scratch scripts:
`/tmp/mtap/rolling_recal.py`, `block_calib.py`, `calib_table.py`.

**1. Pinball: R5 is the best model so far** (`tft-vwap-compare`, all rows of folds 3–13):

| pair | pinball | difference (95% CI) | folds |
| --- | --- | --- | --- |
| R5-cal vs R7f-cal (09:45) | 0.006713 vs 0.006958 | −0.000245 (−0.000284 … −0.000208), 3.5% | better in every fold |
| R5~px-cal vs R6p-cal (09:30) | 0.006694 vs 0.007003 | −0.000309 (−0.000353 … −0.000268), 4.4% | better in every fold |
| R5-cal20s vs R7f-cal | 0.006729 vs 0.006958 | −0.000229 (−0.000272 … −0.000189) | better in every fold |
| R5~px-cal20s vs R6p-cal | 0.006710 vs 0.007003 | −0.000294 (−0.000341 … −0.000248); earnings days −0.000441 | better in every fold |
| R5-cal20s vs R5 (folds 0–14) | 0.006696 vs 0.006671 | +0.000025 (+0.000012 … +0.000039): the recal costs 0.4% | no fold clear |

- vs R7 (night) 11–15% better per fold. 09:30 vs 09:45 differs by ~0.3%.
- The gain over R7f/R6p is largest on rows without pre-market (pinball 0.00512 vs 0.00564 /
  0.00574), from `--open`.

**2. Calibration** (dashboard `calib.summary`, folds 3–13):

| run | rows | pinball | mean / max \|q err\| | ECE | P5–P95 / P10–P90 | bias | grade |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R5 | all 1,265,370 | 0.006674 | 0.79 / 1.26 | 1.27 | 88.5 / 78.4 | −0.64 | good |
| R5-cal (4 folds) | all | 0.006675 | 0.57 / 0.94 | 1.37 | 90.5 / 80.7 | −0.53 | good |
| **R5-cal20s** | all | 0.006691 | **0.18 / 0.31** | 1.16 | 89.5 / 79.5 | +0.06 | excellent |
| R5~px-cal20s | all | 0.006710 | 0.18 / 0.31 | 1.16 | 89.5 / 79.5 | +0.06 | excellent |
| R7f-cal | all | 0.006921 | 0.35 / 0.54 | 1.00 | 90.0 / 80.1 | +0.33 | excellent |
| R6p-cal | all | 0.007003 | 0.47 / 0.89 | 1.34 | 89.9 / 79.9 | +0.45 | excellent |
| R5-cal20s | pm = 1, 716,862 | 0.007877 | 0.17 / 0.30 | 1.03 | 89.6 / 79.6 | +0.12 | excellent |
| R7f-cal | pm = 1 | 0.007901 | 0.10 / 0.22 | 0.97 | 90.0 / 80.0 | +0.05 | excellent |
| R5-cal20s | pm = 0, 548,508 | 0.005140 | 0.21 / 0.40 | 1.44 | 89.4 / 79.3 | −0.01 | excellent |
| R7f-cal | pm = 0 | 0.005640 | 0.74 / 1.16 | 1.09 | 90.1 / 80.2 | +0.70 | good |

Folds 1–14 pooled (2.97M rows): R5-cal20s 0.19 / 0.33, excellent.

**3. Why the fold-window recal fails on R5: its error drifts.** Mean |q err| and bias per fold:

| fold | R5 | R5-cal | R5-cal20s |
| --- | --- | --- | --- |
| 1 | 1.78 (−1.28) | 0.84 | 0.85 (+0.64) |
| 2 | 2.85 (+2.54) poor | 4.00 poor | 0.61 |
| 3 | 1.59 (−1.17) | 1.29 | 0.55 |
| 5 | 0.47 (−0.24) | 1.05 | 0.69 |
| 7 | 1.25 (+0.97) | 1.08 | 0.20 |
| 9 | 2.34 (−1.88) poor | 2.96 poor | 1.15 (+1.08) fair |
| 12 | 1.60 (−1.40) | 0.81 | 0.34 |
| 14 | 1.24 (+1.17) | 2.08 poor | 0.30 |

R5-cal20s is good or excellent in 13 of 14 folds (fold 9 fair). The bias flips sign from one
quarter to the next, so last quarter's shift is often the wrong way; the last 20 sessions follow it.

**4. Window length** (scratch `rolling_recal.py`, MIN_ROWS 500, all rows of folds 3–13; the
21-session block columns are means over 18 blocks of the same rows):

| window | pinball | mean / max \|q err\| | block mean \|q err\| | blocks good+ |
| --- | --- | --- | --- | --- |
| none | 0.006711 | 0.78 / 1.26 | 2.43 | 22% |
| 4 folds | 0.006713 | 0.57 | 2.40 | 22% |
| 5 sessions | 0.006807 | 0.66 | 1.18 | 50% |
| 10 | 0.006756 | 0.34 / 0.54 | 1.37 | 50% |
| **20** | 0.006729 | **0.18 / 0.32** | 2.03 | 22% |
| 40 | 0.006723 | | 2.74 | 6% |
| 60 | 0.006722 | 0.17 / 0.28 | 2.68 | 6% |
| 120 | 0.006715 | 0.54 | | |
| R7f-cal (reference) | 0.006958 | 0.36 | 2.48 | 17% |

- 60 matches 20 pooled but is uneven (fold 3: 2.24 vs 0.55). 20 is chosen.
- **Within one month every model is off by about 2 pts** (block column): the whole market's mood
  in a month shifts all stocks together, faster than any window learns it. Only 5–10 sessions
  help there, at 0.7–1.4% pinball. The run-level (pooled) grade is excellent; a single month
  in the dashboard will often read fair.

**5. R8 screener on R5~px-cal** (`ml/runs/screener/20261030-011637Z-…~px-cal`, 819 days, top 20,
mean range from the open): screener 0.1249, naive ATR 0.1096, TFT width 0.1215, random 0.0282.
Screener − ATR +0.0153 (0.0142 … 0.0163), screener − width +0.0034 (0.0029 … 0.0040); above the
universe median on 99.2% of days. **Kept** (Q15's rule: beat ATR with a CI clear of 0).
R8b, the same screener on the chosen R5~px-cal20s (`ml/runs/screener/20261030-033910Z-…~px-cal20s`, 882 days): top 20 range 12.46% vs ATR 10.90% (+1.56 pts, CI 1.45–1.66), vs TFT width 12.10% (+0.35, CI 0.30–0.40), above the universe median on 99.3% of days; its 09:45 re-rank 12.50%. Kept.

**Verdict.** R5 + rolling 20-session recal is the candidate single model for 09:30 and 09:45,
best in pinball and in pooled calibration. Production switch: the user's call (MODEL-STRATEGY Q36).

### 2026-10-29 Calibration first: walk-forward recalibration, averaging, open and volatility inputs (R5 queued)

Tags: `tag-ml-daily`

**Why.** The user set the goal (10-29): quantiles whose probabilities hold, as the dashboard
reports them (mean/max |q err|, grade, ECE, coverage, bias), from one model for 09:30 and after;
the night forecast is dropped. All figures are the dashboard's own `calib.summary` on rows every
compared run has. Scripts: `/tmp/mtap/calib_table.py`, `ens_run.py`, `feat_probe.py` (scratch,
not in the repo).

**1. Every model is slightly pessimistic, the same way in most folds.** Bias −0.7 to −1.2 pts
(the actual lands above P50 more often than half the time), P5–P95 a little under 90%. A steady
error can be learned from earlier folds.

**2. Walk-forward recalibration** (`tft-vwap-recal`, new). For test fold k each quantile moves by
the tau-quantile of (actual − P_tau) over the `window` folds before k (never fold k), per group:
with or without full pre-market × P5–P95 width tercile. Folds 3–13, the 716,862 rows with full
pre-market:

| run | pinball | mean / max \|q err\| | ECE | P5–P95 / P10–P90 | bias | grade |
| --- | --- | --- | --- | --- | --- | --- |
| R7f | 0.007902 | 0.78 / 1.08 | 1.17 | 89.4 / 79.3 | −0.72 | good |
| **R7f-cal** | 0.007901 | **0.10 / 0.22** | **0.97** | 90.0 / 80.0 | +0.05 | excellent |
| R6f | 0.007907 | 0.94 / 1.55 | 1.61 | 90.7 / 80.9 | −0.86 | good |
| R6f-cal | 0.007908 | 0.16 / 0.37 | 1.31 | 90.0 / 80.1 | +0.15 | excellent |
| R4f | 0.007941 | 1.24 / 2.17 | 1.45 | 89.9 / 80.2 | −1.15 | fair |
| R4f-cal | 0.007950 | 0.87 / 1.55 | 1.45 | 90.8 / 81.5 | −0.57 | good |
| R6p | 0.007965 | 0.91 / 1.51 | 1.58 | 90.6 / 80.9 | −0.84 | good |
| R6p-cal | 0.007967 | 0.16 / 0.40 | 1.30 | 90.0 / 80.0 | +0.16 | excellent |
| R4p | 0.007962 | 1.10 / 1.89 | 1.28 | 88.3 / 78.3 | −0.89 | fair |
| R4p-cal | 0.007971 | 0.89 / 1.94 | 1.59 | 91.0 / 82.1 | +0.70 | good |

- Pinball loss is unchanged (±0.1%): the shift moves the bands, not their information.
- It works on the HGB and the overlays, much less on the TFTs (R4f/R4p): their errors change sign
  from fold to fold, so the last 4 folds don't predict the next.
- **Variants on R7f** (mean / max |q err|):

  | variant | all 1.27M rows | rows without pre-market | rows with it |
  | --- | --- | --- | --- |
  | none | 0.89 / 1.11 | 1.05 | 0.77 |
  | one group, window 4 | 0.39 / 0.59 | 0.46 | 0.36 |
  | width only, window 4 | 0.41 / 0.58 | 0.42 | 0.49 |
  | pm, window 2 | 0.25 / 0.40 | 0.51 | 0.10 |
  | pm × width, window 2 | 0.26 / 0.41 | 0.45 | 0.13 |
  | **pm × width, window 4** | | 0.86 | **0.10 / 0.21** |
  | pm × width, window 6 | 0.41 | 0.86 | 0.13 |

  The pre-market group is needed (the two kinds of rows are biased differently). Window 4 is best
  on the pre-market rows, the ones that matter at 09:45; window 2 on the rest.

**3. Averaging two models** (quantile by quantile, then recal):

| run | pinball | mean / max \|q err\| | ECE | grade |
| --- | --- | --- | --- | --- |
| R7f+R6f | 0.007878 | 0.87 / 1.33 | 1.45 | good |
| R7f+R6f-cal | 0.007877 | 0.12 / 0.21 | 1.21 | excellent |
| R7f+R4f | **0.007845** | 1.03 / 1.69 | 1.38 | fair |
| R7f+R4f-cal | 0.007846 | 0.46 / 0.83 | 1.08 | excellent |

Averaging with the TFT lowers the loss 0.7%, but it is two models (one a TFT) and less well
calibrated than R7f-cal. Not used: the user wants one model, calibration first.

**4. Inputs and iterations for the HGB** (`/tmp/mtap/feat_probe.log`, folds 13 and 5, one HGB per
level, 200 iterations; gain in pinball vs the R7f inputs):

| added | P5 | P50 | P95 | on rows without pre-market |
| --- | --- | --- | --- | --- |
| past volatility (`--vol`) | 1.9–2.2% | 0% | 1.5–1.9% | 0.8–2.4% |
| the open, every stock (`--open`) | 2.8–3.4% | 2.5–4.8% | 2.0–6.3% | 6.9–13.3% |
| both | **4.8–5.5%** | 2.5–4.8% | **3.9–7.8%** | 9.3–14.6% |

- **The open:** the panel's `open_pct_prev_vwap` is 0 when a stock had no pre-market trades,
  although the open is known at 09:30 for every stock. `--open` adds the panel's
  `gap_open_vs_prev_vwap_pct` of day d (clipped ±30%). On rows with pre-market it adds little
  (0.2–0.7%), since `open_pct_prev_vwap` already has it.
- **Iterations:** P50 is best at 200–400 and gets worse by 1,000; P5/P95 keep improving to
  1,000 (0.6–1.1% over 200). Hence `--tail-iter 600` for P15 and below and P85 and above.
  63 leaves or learning rate 0.1: no gain.

**5. R5, queued** (`ml/runs/queue-r5.sh`, 14:10 PT). One HGB, `--premarket optional`: every
training row with full pre-market appears twice, as it is and with the full group blank, so one
fit serves 09:45 (run dir) and 09:30 (`~px`). Plus `--open --vol --tail-iter 600`, folds 0–14.
Then recal of both states, the calibration tables vs R7f-cal / R6f-cal / R6p-cal, compare, and R8
(screener) on R5~px-cal. Results: the R5 entry above.

### 2026-10 R7 and R7f, plain gradient boosting (no TFT), vs R1c and the 09:45 models

Tags: `tag-ml-daily`

**Runs.** R7 `ml/runs/20261029-153450Z-gbm-4438sym` (folds 0–14, R1c's inputs, no pre-market) and
R7f `ml/runs/20261029-170229Z-gbm-4438sym` (7 odd folds, + the full pre-market group always on).
`tft-vwap-gbm`: one sklearn `HistGradientBoostingRegressor(loss="quantile")` per quantile, 200
iterations, on the last-session inputs plus lags 1, 2, 5 and 20 of the move; quantiles sorted.
R7 08:34 → ~10:00 PT, R7f → 10:52 PT, on the CPU.

**R7 vs R1c (night)**, 3,169,270 forecasts on (symbol, date), 945 days:

| | R1c | R7 |
| --- | --- | --- |
| pinball loss | 0.0078386 | 0.0077343 (−1.3%) |
| skill vs naive | 27.2% | 28.0% |
| P5–P95 coverage | 88.4% | 88.8% |
| direction of P50 | 59.5% | 60.1% |
| mean \|q err\| (calibration) | 0.67 pts | 0.80 pts |

- Paired R1c − R7: +0.0001036 (+0.0000735 … +0.0001334), R7 better. Earnings days +0.000534,
  other days +0.0000711, both CIs clear.
- Per fold, R7's gain (%): 5.46, 1.39, 2.0, 3.3, 1.39, 2.59, −0.21, 0.52, −0.13, 1.03, 0.99, 0.36,
  0.74, 0.43, 0.47 (folds 0–14). The gain is largest where the TFT has the least training data
  and shrinks to about 0.4–1% in the late folds.

**R7f on the bake-off rows** (812,131 rows with full pre-market, 441 days; diff = rival − R7f,
positive = R7f better):

| vs | rival loss | R7f loss | diff (95% CI) | gain |
| --- | --- | --- | --- | --- |
| R4 | 0.0080187 | 0.0078916 | +0.000149 (+0.00012 … +0.00018) | 1.6% |
| R4f | 0.0079383 | 0.0078916 | +0.000060 (+0.000037 … +0.000087) | 0.76% |
| R6f | 0.0078944 | 0.0078916 | +0.0000048 (−0.0000099 … +0.000019) | tie |

- vs R4: earnings days +0.000629.
- vs R4f, per fold (×1e-6): f1 +103, f3 +168, f5 +87, f7 +39 (CI spans 0), f9 +119, f11 −18
  (spans 0), f13 −76 (CI clear, R4f better). The same pattern as R6f vs R4f: the dedicated TFT
  catches up in the late folds.
- vs R6f: no fold is clear either way. R7f is better on earnings days (+0.000050, +0.000021 …
  +0.000079) and level on other days (−0.0000023, CI spans 0). P5–P95 R6f 90.8%, R7f 89.3%.
- R7f's own rows (1,486,039, every odd-fold row): skill 36.2%, P5–P95 89.4%, direction 65.6%.
- Files: `compare_vs_r4full.json`, `compare_vs_r4f.json` and `compare_vs_r6f.json` in R7f's run.

**What it shows:**
- **On these inputs the TFT is not worth its cost.** A model that trains in minutes on the CPU
  beats R1c at night by 1.3% and ties TFT + overlay at 09:45. Most of the TFT's input is the
  sequence of past days, and lags 1, 2, 5 and 20 of the move carry most of what it learns from it.
- **The early folds favour HGB most** (fold 0: 5.5%): trees need less data. Over 4+ years of
  training the gap narrows, so R5 should weigh the late folds, as at 09:45.
- **Fewer models wins the tie at 09:45:** R7f is one model, R6f is R1c plus an overlay. R6f's
  bands sit closer to the 90% target; R7f is better on earnings days. R5 decides between them.
- Open: an HGB at 09:30 (R7p) to settle 09:30, and whether R5 is HGB-based. (Scoreboard,
  ml/MODEL-STRATEGY.md.)

### 2026-10 R1c, R0 + session shape + after-hours, vs R0 (test sessions 2023-01-20 to 2026-10-26)

Tags: `tag-ml-daily`

**Run:** `20261028-233305Z-return-4438sym`. R1's panel and folds with `--session-shape
--after-hours` only (no market, no macro), 8 workers. 10-28 16:33 → 21:08 PT.

**Overall** (3,169,270 forecasts matched on (symbol, date), 945 days; R0 − R1c, positive = R1c
better):

| | rows | R0 | R1c | difference (95% CI) |
| --- | --- | --- | --- | --- |
| all | 3,169,270 | 0.007884 | 0.007839 | +0.000046 (+0.000021 … +0.000071), −0.58% |
| earnings days | 238,489 | | | +0.000339 (+0.000287 … +0.000387) |
| other days | 2,930,781 | | | +0.000020 (−0.000005 … +0.000046) |

| measure | R0 | R1c |
| --- | --- | --- |
| skill vs naive | 26.6% | 27.0% |
| P5–P95 / P10–P90 coverage | 88.9% / 78.9% | 88.4% / 78.4% |
| median P10–P90 width | 5.11% | 5.04% |
| median \|P50 error\| | 1.262% | 1.262% |
| direction of P50 | 59.1% | 59.5% |
| epochs per fold | 4–13 | 4–10 |

- **Per fold:** R1c is clearly better in folds 1, 2, 7, 9, 10, 12 and 14 and clearly worse in none.
  Fold 0 (2023-Q1, the least training data) leans R0's way: −0.000188 (−0.000399 … +0.0000002).
- **Encoder weight per group (%):** after-hours 6–44, session shape 5–24, earnings 10–44, base
  29–68. After-hours is highest in folds 2, 6, 8, 13 and 14 (32–44).

**What it shows:**
- R1c passed Q21 rule 1 outright (overall CI clear of 0), so it is the base for R4 onward
  (`logs/r4-queue.log`, `logs/r4-base.txt`).
- **The gain is on earnings days:** 17× the other days' gain per forecast, where the other days'
  CI spans 0. The after-hours inputs see the reaction to results released after the close,
  which R0 only sees a day later.
- The median error is unchanged; the gain is in the shape of the band. The bands are 1.4%
  narrower and coverage is 0.5 points lower, so R1c is a little more overconfident than R0.
- Without the day-level groups, there is no sign of memorising: R1c trains about as long as R0
  (mean 6.4 vs 6.8 epochs), where R1 stopped at 4–6 in every fold. Its after-hours weight still
  swings between folds, but no fold is worse for it.
- `ml/runs/20261028-233305Z-return-4438sym/compare_vs_r0.json` holds the full comparison.

### 2026-10 R6c, a market/macro event overlay on R0 (test folds 1–14)

Tags: `tag-ml-daily`

**What it is** (`tft-vwap-event-overlay`, `ml/tft_vwap/event_overlay.py`, ml/MODEL-STRATEGY.md
Q22, Q24). It keeps R0's median and scales its band: `P50 + s × (q − P50)`, with
`s = exp(Σ b_i f_i)`.
- Each `f_i` is how far one of yesterday's day-level moves went beyond 2 standard deviations of
  its past 63 sessions. The moves are the market, the stock's sector vs the market, and the 10y
  and 2y yields. On ordinary days f = 0, so s = 1 and R0 passes through unchanged.
- Fitted per fold on R0's out-of-fold forecasts of the earlier folds only (400 rows per day,
  pinball loss). Fitting takes 12 s for all folds.
- 16% of rows have an event; the market feature fires on 53 test days.

**Results vs R0** (2.95M forecasts, 882 days; negative = R0 better):

| variant | A − B per day (95% CI) | folds R0 better / R6c better | P10–P90 coverage |
| --- | --- | --- | --- |
| event only | −0.000002 (−0.000004 … −0.000000) | 1 / 0 | 78.8% (R0 78.7%) |
| + one band scale for all days (`--calibrate`) | −0.000004 (−0.000007 … −0.000002) | 5 / 2 | 77.8% |
| + calibrate + median shift (`--shift`) | −0.000011 (−0.000020 … −0.000004) | 3 / 1 | 77.7% |

On the event rows themselves (event-only variant):

| feature | rows | days | coverage R0 → R6c | loss R0 → R6c |
| --- | --- | --- | --- | --- |
| market | 178k | 53 | 75.8% → 76.4% | 0.008843 → 0.008852 |
| sector vs market | 145k | 303 | 77.1% → 78.3% | 0.008252 → 0.008250 |
| 10y yield | 137k | 41 | 79.0% → 79.3% | 0.008134 → 0.008165 |
| 2y yield | 150k | 45 | 78.2% → 78.8% | 0.008101 → 0.008125 |
| any (16% of rows) | | | 77.3% → 77.8% (quiet rows 78.9%) | 0.008330 → 0.008341 |

**What it shows:**
- **The coefficients are not stable.** The market coefficient is −0.73 in fold 1 and −0.63 in
  fold 2, then +0.01 … +0.10 from fold 5 on. Yields flip sign between folds. With 40–50 event
  days per series in the training folds, there is no repeatable effect to learn.
- **R0 already reacts to yesterday's big days** through the stocks' own moves (Q22: its band is
  11% wider after a big market day). Widening it further gains a little coverage but loses
  pinball, so the extra width lands on the wrong rows.
- **Calibration and shift make it worse.** One scale for all days is fitted to the earlier folds,
  which were wider than the later ones. The median shift overfits fold 0–2's few big days.
- **What's left.** The misses are on the days the market itself moves (64.7% coverage, Q22),
  which aren't in yesterday's data. Scheduled release days (CPI, jobs, FOMC) are the only
  known-ahead stand-in. They need the release calendar (Q24), not yet fetched.
- Kept as a tool, not as a model. R0 stays the base.

### 2026-10 Release days in R0 (CPI, jobs report, FOMC), test folds 0–14

Tags: `tag-ml-daily`

A pre-check for the release-calendar overlay (ml/MODEL-STRATEGY.md Q25). It uses dates typed in
by hand (FOMC and jobs 2023–2026, CPI 2023–2025); there is no fetched calendar yet.

| day type | days | R0 P10–P90 coverage | loss | mean \|market move\| | top-10% market days |
| --- | --- | --- | --- | --- | --- |
| CPI | 34 | 74.3% | 0.008716 | 0.88% | 5 of 95 |
| jobs report | 41 | 73.8% | 0.008775 | 1.08% | 11 of 95 |
| FOMC day | 30 | 78.3% | 0.007804 | 0.84% | 4 of 95 |
| day after FOMC | 30 | 74.8% | 0.008977 | 0.88% | 6 of 95 |
| other days | 811 | 79.5% | 0.007768 | 0.65% | |

**Band scale on release days only** (folds 5–14, the same scale for every stock):

| release | days | ×1.0 (R0) | ×1.1 | ×1.15 | ×1.2 | ×1.3 |
| --- | --- | --- | --- | --- | --- | --- |
| jobs: coverage / loss (bp) | 27 | 74.6% / 88.42 | 78.4% / 88.17 | 80.2% / 88.15 | 81.7% / 88.19 | 84.5% / 88.46 |
| CPI: coverage / loss (bp) | 19 | 75.7% / 86.64 | 79.6% / 86.51 | 81.3% / 86.55 | 82.9% / 86.65 | 85.7% / 87.05 |

- **Walk-forward fit** (one scale per type, earlier folds only): stable from fold 4. Jobs about
  +16%, CPI +10–15%, day after FOMC +9–20%, FOMC day about 0.
- **Overall vs R0:** +0.0000001 per day, CI (−0.0000029 … +0.0000034). No measurable change.
- **The misses.** 26% of rows miss the band on release days, against 21% on other days, and they
  overshoot it by more (median 22% of the band width, against 18%). They are big moves in either
  direction, so a wider band fixes coverage but barely changes the pinball loss.

### 2026-10 pit earnings and series breaks vs production (test sessions 2025-09-02 to 2026-08-14)

Tags: `tag-ml-daily`

**The runs** (2026-10-11), on the panel to 2026-10-06 (schema 3, 3,763 stocks):
- **Production** `20261005-025909Z-return-3712sym`: no earnings inputs, no series breaks.
- **Retrain** `20261011-173109Z-return-3785sym`, from `uv run tft-vwap --earnings pit
  --test-end 1663`.
  - It has the same four folds as production (FEATURES.md section 12).
  - Its 3,785 symbols are the 3,763 stocks plus 22 earlier segments (`SYMBOL~1`). For the
    comparison these map back to the ticker.
- The two runs are scored on the (stock, forecast day) rows both forecast.

**Results.** These leave out the 22 split tickers, 863,201 rows:

| measure | production | retrain |
| --- | --- | --- |
| pinball loss | reference | −2.74%. The 95% interval over test days excludes 0, and the retrain is lower on 62% of days. |
| skill vs the naive forecast | 0.239 | 0.260 |
| P5–P95 coverage (aim 90%) | 86.65% | 88.11% |
| P25–P75 coverage (aim 50%) | 43.7% | 46.9% |
| direction of `q50` | 57.8% | 58.8% |

- **By fold:** −0.5%, −0.0%, −8.3% and −1.1%. Production's fold 2 was weak.
- **By results window** (the forecast day's window):
  - Earnings_Day, 12,726 rows: −3.0%, and P5–P95 coverage 42.3% → 49.3%.
  - Earnings_Eve −2.3%; Post_Earnings_Reaction −3.5%.
  - In fold 3, Earnings_Day is −6.6%, with coverage 46% → 61%.
- **With the 22 split tickers included** the change is −4.0%, much of it from AKTS.

**Examples by stock** (mean pinball over the shared rows, in moves):

| stock | rows | production | retrain | change | why |
| --- | --- | --- | --- | --- | --- |
| AKTS | 130 | 0.5725 | 0.0106 | −98% | A missed reverse split: production's encoder read a fake 10x jump. |
| BNY | 148 | 0.0170 | 0.0019 | −89% | A closed-end fund held the ticker before BNY Mellon. The breaks split it. |
| IXHL | 240 | 0.0151 | 0.0153 | +1.7% | Not split. It has the largest move left, +529%. |
| AAPL | 240 | 0.0041 | 0.0039 | −5.6% | |
| NVDA | 240 | 0.0063 | 0.0060 | −4.3% | |

**Examples on results days** (moves from the previous session's VWAP):

| stock, forecast day | actual | production P5 … P95 | retrain P5 … P95 | |
| --- | --- | --- | --- | --- |
| MSFT 2026-07-30 | +13.0% | −4.1% … +2.0% | −8.5% … +7.5% | Both miss. The retrain is closer: pinball 0.066 → 0.060. |
| NVDA 2026-05-21 | −1.1% | −2.4% … +4.5% | −7.9% … +8.5% | Both hold. The retrain's band is about twice as wide. |
| ADBE 2026-06-12 | −7.6% | −5.3% … +4.5% | −9.3% … +9.7% | Only the retrain holds. |
| AMZN 2026-07-31 | +12.3% | −4.0% … +1.6% | −5.5% … +2.4% | Both miss, leaning down. The retrain is slightly worse: 0.063 → 0.065. |

**What it shows:**
- The earnings inputs widen the band on results days, but not enough: 49% coverage against
  90%. At the cutoff the model can't know which way the surprise goes, so a lean in the band
  can turn out wrong, as with AMZN.
- Series breaks remove fake jumps that ruined a few stocks' forecasts (AKTS, BNY).
- The two changes went in together, so their separate shares are not measured.

**To run it again:**
- The dashboard's Compare runs tab gives the pooled metrics, with "Only rows present in every
  chosen run" ticked.
- The per-day interval, the split exclusion and the per-window numbers came from a scratch
  script that is not in the repo (FEATURES.md section 13, "Improving the testing").

### 2026-10 first full run, fold 0 calibration (test sessions 2025-09-02 to 11-24)

Tags: `tag-ml-daily`

Run `20261004-231546Z-return-3712sym` (2026-10-04), the first full-panel run, before the
earnings inputs. Fold 0 has 213k test forecasts, graded with `tft-vwap-report`.
- **Overall the probabilities are well calibrated:** calls made at a given probability come true
  about that often.
- **Downside calls at 70–80% are about 6 points overconfident.**
- **Per-stock bias labels at 60 test days are about chance level:** 170 of 3,517 stocks are past
  |z| > 2, close to the 5% expected from noise alone. One fold is too few days per stock to
  call a stock biased.

### 2026-10 live grades of production (forecasts for 2026-10-05 and 10-06)

Tags: `tag-ml-daily`

Production `20261005-025909Z-return-3712sym --fold 3`, 3,679 stocks each day, graded from
`actuals.parquet` (moves vs the previous session's VWAP):

| forecast day (from) | P10–P90 coverage (aim 80%) | above P90 / below P10 | `q50` direction right | bias (negative = pessimistic) | skill vs naive |
| --- | --- | --- | --- | --- | --- |
| 10-05 (10-02) | 85.1% | 10.0% / 4.9% | 58.9% | −0.023 | 24.7% |
| 10-06 (10-05) | 85.6% | 9.7% / 4.7% | 52.6% | −0.054 | 21.1% |

- The live bands were wider than needed (85% inside against 80%). In testing they were a little
  narrow (production: 86.7% inside P5–P95 against 90%), so these two days were probably calmer
  than the test blocks.
- **The model leans pessimistic:** twice as many moves land above the band as below, and the
  bias is negative both days.
- Two days are not a sample: the next grades decide whether the lean is real.
- No grades after 10-06: the panel on this Mac ends there, so the 10-07 forecast is not graded
  yet.

## First live morning on the NAS (2026-10-26, 06:10–08:00 PT)

Tags: `tag-sourcing-host`

A read-only check of the feed, the NAS load and dropped data, against the iMac's 10-23.

- **Dropped data: none found.**
  - massive-ws: 0 dropped frames, 0 bad messages, 0 disconnects; 21 five-minute files.
  - E*TRADE store: 0 bad entries, 0 trimmed entries, lag 0.
- **Poll cadence matches the iMac.**

  | list | NAS 10-26 h10 | iMac 10-23 h10 |
  | --- | --- | --- |
  | Experimental: median gap, max gap | 2.1 s, 5.0 s | 2.0 s, 6.0 s |
  | MidCap: median gap, max gap | 22.2 s, 39.7 s | 21.3 s, 43.0 s |
  | polls per hour (Experimental) | 1,629 | 1,696 |

  No gap over 120 s on either day. Rows per full hour: 1.05M (NAS) against 1.09M (iMac).
- **NAS load:** CPU busy about 1% of 8 threads, no iowait or pressure stalls, 2.5 GB memory
  used, network 149 KB/s in. Redis holds about 918 MB at 3,253 stream entries (282 KB each),
  so the 4,000-entry cap is about 1.1 GB, under the 2 GB limit.
- **Login:** the first attempt at 06:10 was refused (session restore redirected to the login
  page); the retry succeeded with no code. The `/login/sar` lines every 30 s are the
  session's keepalive, not errors.
