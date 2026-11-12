# Live recommendations to MBin: plan to production mode

Status: plan, created 2026-11-01, **status refreshed 2026-11-12** (§0 = every pending item, for review). Nothing
below is built unless marked done. The contract is
`MBin/docs/contracts/` (protocol v1.0-draft, phase 1); the model design is
[intraday-design.md](intraday-design.md); the probe result is in LEARNINGS `tag-ml-intraday`
("Step 0 probe: results"). Where things run today: [STATUS.md](STATUS.md).

**Goal.** Every RECOMMENDATION MTap sends is valid (it already is) **and** real in every field: a fresh
price, real reference data, and a `score` that is a calibrated P(+target before −stop) from a model
with measured skill. Then the same with a scheduled start, grading, and limits, so it can run unattended.

## 0. Pending for the MBin integration (status 2026-11-12, for review)

Checked against the code, git and MBin's docs on 2026-11-12. MBin's last commit is 2026-11-07 (c2036c0), so
nothing on MBin's side has moved since the 2026-11-07 hand-off (REVIEW.md, "Next, MBin").

**What works today:** the contract link (Mac contract Redis `127.0.0.1:6380`, REVIEW item 14) and the
recommender's phase 1 protocol: schema-checked RECOMMENDATION / CONTRACT_STATUS / RESYNC_REQUEST, the outcome
consumer, the audit, and sending in every `trading_state`. It is all started by hand, and candidates are
hand-submitted (`recommender submit`). On MBin's side, M3, M4 (Signals intake, TRADING_PARAMETERS, outcomes)
and M5 (the trading slice, paper and sandbox) are done; M6 (switch to real MTap) is not. **No real
recommendation has gone to MBin yet:** there is no scorer, no live price and no reference data.

**A. Waiting on MBin** (in REVIEW.md; MTap has nothing to do until MBin acts)
| what | MBin's step | then MTap |
| --- | --- | --- |
| REVIEW items 15–18 (answered by MTap 2026-11-07) | accept or counter; set each to `agreed` | nothing, unless MBin counters |
| M13, RECOMMENDATION v2 (answered + confirmed 2026-11-07) | move `proposed/` into `schemas/` and `fixtures/`, plus the registry, `consumes` and fixture-test change; set M13 `agreed` | P13 |
| Item 15's schema edits (`canceled` outcome, non-final `price_moved` / `entry_refused`, `paused` in TRADING_PARAMETERS) | edit `recommendation_outcome.v1` and `trading_parameters.v1` with fixtures, then tell MTap the new hashes | P14 (until then an unknown outcome is `validation_failed` and is not stored) |
| M6, switch to real MTap | stop the replayer, rerun the M4 / M5 checks on real recommendations | needs P5 (or P16) running, so that real recommendations exist |

**B. Waiting on the user's decision**
| what | choice | notes |
| --- | --- | --- |
| P2 deploy: open the NAS Redis read-only to the Mac | OK to deploy the 2026-11-01 draft (§3, ~45 min, outside market hours, plus a DSM firewall rule only the user can add) | decided A on 2026-11-01; drafted (commit 5a37885), not deployed |
| P15: how MBin filters on the Moves (±X%) predictions | option 1, a calibrated ladder P(centre ≥ X) as `events`, recommended; or option 2, raw quantiles plus a calibration table | MODEL-STRATEGY Q42; then a proposal in `MBin/docs/contracts/proposed/` |
| What `score` means for the intraday v2 model | the pocket model sizes moves but doesn't call direction (Q41, Q42). Its `score` would be a base rate (`baseline.climatology`, tier 3), with the forecast block carrying the size predictions | decides whether P16 sends BUYs at all, or only forecasts |
| Item 13 (phase 2: an active exit's `ttl_ms` ≥ 60 000 in the schema) | MTap's call; not needed for the MVP | open since 2026-11-01 |

**C. MTap to build** (no outside dependency except as noted)
| # | item | state on 2026-11-12 |
| --- | --- | --- |
| P3 | reference data on every rec (`name`, `market_cap`, `cap_band`, `sector` / `industry`) | **not built.** The recommender only derives `cap_band` when a candidate brings `market_cap` |
| P4 | selection limits | **partly built.** TTL clamped to `ttl.max_ms` and the close; a repeat `rec_id` is dropped. **Not built:** one live rec per symbol per day; ≤ 20 per minute; `hints.min_probability`; no send before the first TRADING_PARAMETERS |
| P5 | stopgap `recommender pick SYMBOL` (base-rate score) | **not built** (the CLI has `run`, `submit`, `status`, `check-schemas`). Needs P2 (or option B's 15-min price), P3 and P4 |
| P6 | grading from market data, plus slippage logged separately | **not built.** Needs P2 |
| P7 | launchd start on the Mac, trading days ~06:00–13:30 PT | **not built.** MBin starts the contract Redis with its own tool |
| P13 | the v2 builder: vendor v2 and pin its hash; `forecast` from the daily run files, then from the intraday run file; `score_readout` / `probability_readout`; `max_entry_pct` (¼ of the target); `skill` only at ≥ 60 days and ≥ 1,000 events | waits for M13 (A) |
| P14 | re-pin the outcome / parameters schemas after MBin's item 15 edits | waits for MBin (A) |
| P16 | **live scorer for intraday v2** (new; replaces P8 / P9): write the IA1 quantile columns per minute for the live set, then the extremes as each window closes, so the Moves filter and grading run live | **not designed.** Needs P2 (fresh E*TRADE prices) and the user's call on the model (Phase 2 GBM, fold A noS + enc120) |
| P17 | after M6: raise the protocol from `1.0-draft` to `1.0` with MBin in REVIEW | after M6 |

**D. Later (phase 2):** P12 (EXIT_RECOMMENDATION, POSITIONS_SNAPSHOT, item 13).

**Done since 2026-11-01:** P1 (bounce check); P11 (the env-auth bus change and the docs are on main, 5a37885 /
ef8acef); the recommender sends in every `trading_state` (90fdf38); MBin's M4 and M5.

## 1. Where it stands (2026-11-01; superseded by §0 for status)

**Done:**
- `sourcing-py recommender` publishes on MBin's contract Redis on the Mac (`127.0.0.1:6380`, user `mtap`,
  REVIEW item 14 agreed). It sends schema-checked RECOMMENDATION, CONTRACT_STATUS and RESYNC_REQUEST,
  answers MBin's RESYNC, reads RECOMMENDATION_OUTCOME (group `mtap-outcomes`) and keeps an audit file.
- Redis credentials come from env `REDIS_USERNAME` / `REDIS_PASSWORD` in both buses; MTap's secret is
  `~/.config/mtap/contract.env`.
- The Step 0 probe (above).

**What the probe says about the scorer:**
- The **size** of the next hour's move is predictable (+6.3% Brier skill on touch ±0.5σ / 60 min).
- The **direction** is predictable only with fresh data (d = 0: rank IC +0.043, top-20 +0.07σ). On
  15-min-delayed data it is ≈ 0. So the planned Stage A full-market ranker on delayed Massive data fails
  its go rule. A live scorer has to run on stocks MTap prices in real time (the E*TRADE lists, ~2 s polls).
- **P1 (done 2026-11-01): the d = 0 direction edge was bid–ask bounce.** Measured from the next minute's
  open or VWAP, rank IC falls from +0.034 to +0.006 … +0.009 and the top-20 spread from +0.41σ to +0.05σ
  (CI touching 0). Size skill is unchanged. So by §4's rule the scorer stops at the base rate (P5); P8 and
  P9 are not started (LEARNINGS "P1 bounce check").
- A BUY's `score` is P(target before stop). Its skill is the "up first" number (+4.2% over the base
  rate on 2026), smaller than the touch skill.

## 2. What remains

| # | Item | Why | Owner | Size | Depends on |
| --- | --- | --- | --- | --- | --- |
| P1 | **Done 2026-11-01: it was bounce, no direction edge.** **Bounce check** on the probe: targets from the next minute's open (and d = 1), rank IC / top-20 / up-first at d = 0 again | decides whether a live scorer has any direction edge | MTap ml | ~2 h run (features cached in `/tmp/mtap/intraday_probe/`) | — |
| P2 | **Live price on the Mac**: the recommender (and later the scorer) reads fresh prices for any symbol it sends | `data_ts_utc` / `price_ref` must be seconds old; a 15-min-old price with a 5-min TTL is `stale_on_arrival` | MTap | ~2 h | **user decision** (§3) |
| P3 | **Reference data on every rec**: `name`, `market_cap` (Massive shares × previous close), `cap_band`, `sector` / `industry` (TRBC copy in DuckDB, Massive `sic_description` fallback) | MBin's Console filters; promised in REVIEW item 3 | MTap | ~2 h | — |
| P4 | **Selection limits in the recommender**: one live rec per symbol per day, ≤ 20 per minute, `hints.min_probability`, no send without TRADING_PARAMETERS, TTL clamped to `ttl.max_ms` and the close before sending | REVIEW item 1; the probe counts ~1,600 signal keys a day on the pipeline | MTap | ~1 h | — |
| P5 | **Stopgap scorer `recommender pick SYMBOL [--target --stop]`**: real price (P2), reference data (P3), `score` = the base rate of +target before −stop for that time of day × σ × liquidity group (from the 5 y minute bars), `strategy_id: baseline.climatology` | real in every field except the edge; lets MBin's M4–M6 run on real recommendations before the model exists | MTap | ~½ day | P2, P3, P4 |
| P6 | **Grading**: every published rec checked against market data (from `price_ref` at `data_ts_utc`: which of +target / −stop is touched first, within the horizon), daily report, calibration of `score`; never from MBin's `closed` (that is MBin's execution); entry slippage `filled.price − price_ref` logged separately (REVIEW item 17 (d)) | protocol §5 audit; the only proof the probabilities hold | MTap | ~½ day | P2 |
| P7 | **Scheduled start on the Mac**: launchd job for the recommender on trading days ~06:00–13:30 PT, restart on failure, logs, `caffeinate`; the contract Redis started by MBin's tool | today both start by hand | MTap (recommender), MBin (contract Redis) | ~1 h each | — |
| P8 | **Not going ahead (P1: no direction edge).** **Live scorer v0** (if P1 holds): d = 0 model on the live-polled stocks, features stitched from Massive (to t − 15) + E*TRADE polls (last 15 min, §3F), writes `mtap:int:REC_CANDIDATE`; trained with the training-time seam emulated (1 s bars on a 2 s grid) | the real `score`; replaces P5's base rate under `strategy_id: intraday.pressure` | MTap ml | ~1–2 weeks | P1, P2, P6 |
| P9 | **Shadow period**: P8 publishes for ≥ 10 sessions with grading before the operator acts on it; promote only if graded calibration and up-first skill hold live | live data differs from the backtest (feed gaps, the seam) | MTap + operator | 2 weeks elapsed | P8 |
| P10 | **MBin: M3–M5 done (2026-11-01 / 11-07); M6 (switch to real MTap) pending.** **MBin side**: TRADING_PARAMETERS on the contract Redis, Signals intake (M4), sandbox order path (M5), switch to real MTap (M6) | without parameters MTap can't clamp TTLs; without intake nothing reaches the Console | MBin | MBin's MVP plan | — |
| P11 | **Done (on main: 5a37885, ef8acef).** **Commit and push** the env-auth bus change and these docs | before the live tests | user says when | minutes | — |
| P13 | **The full forecast on every rec (contract M13, proposed 2026-11-07)**: RECOMMENDATION v2 adds a typed `forecast` block (daily VWAP band P5–P95, intraday pocket quantiles per duration bin, event probabilities with their base rates, out-of-sample skill with CIs, live track record, model label, overlay variants) and `observed`; `score` / `probability` unchanged. Draft in `MBin/docs/contracts/proposed/`, protocol §8.1.1. MBin agreed and MTap confirmed (2026-11-07). The builder must set `score_readout` / `probability_readout` (the events `score` / `probability` come from), `max_entry_pct` (default ¼ of `target_pct`), and send a `skill` entry only when its test has ≥ 60 days and ≥ 1 000 events (REVIEW items 16–18). Readout ids are fixed per event definition across model labels. | the prediction must reach the operator undiluted, not as one number | MTap (schema), MBin (consume + Console) | ~½ day MTap after MBin moves v2 into `schemas/` | MBin moving v2 |
| P14 | **Contract items 15–18 follow-ups**: re-pin `recommendation_outcome.v1` (new `canceled`, a non-final `price_moved` outcome) and `trading_parameters.v1` (`paused`) when MBin changes them. Until the re-pin, an unknown outcome is `validation_failed` and not stored. | MTap's tooling must never drop MBin's feedback | MTap | ~½ h each | MBin's schema edits |
| P15 | **Moves (±X%) filters for MBin (MODEL-STRATEGY Q42, added 2026-11-12)**: option 1, a calibrated ladder P(centre ≥ X) for X = 0.5 / 1 / 1.5 / 2 / 3% per window and side as `forecast.events` entries (e.g. kind `centre`, with `level` and `base_rate`), recommended; or option 2, the raw `cm_up` / `cm_dn` / `mass_up` quantiles plus a daily calibration table, so MBin computes any X | the user wants to set ±X% / min-P filters in MBin on live predictions | MTap (proposal + builder), MBin (consume + filters) | proposal ~2 h; builder part of P13 | **user picks 1 or 2**; M13 agreed |
| P16 | **Live scorer for intraday v2 (added 2026-11-12, replaces P8 / P9)**: the IA1-style pocket quantiles per minute for the live set, the extremes once each window closes, written in the run-file columns so the dashboard Moves tab, P15 and grading (P6) work unchanged; shadow ≥ 10 sessions before the operator acts | the model now exists (intraday v2 Phase 1); nothing scores it live | MTap ml | ~1–2 weeks, not designed | P2, the user's model call (Phase 2 GBM / fold A check) |
| P17 | **Protocol `1.0-draft` → `1.0`** with MBin in REVIEW | MBin's M6 check | MTap + MBin | minutes | M6 |
| P12 | **Phase 2**: EXIT_RECOMMENDATION, POSITIONS_SNAPSHOT (held symbols on the fast list), REVIEW item 13 | not needed for the MVP | MTap + MBin | later | P8 |

## 3. Decision: the live price on the Mac (P2)

**Decided 2026-11-01 (user): A.** The Mac reads live prices from the E*TRADE watchlist feed on the NAS
Redis; later the Massive feeds also come from the NAS the same way. Not built yet: it means a NAS
compose change (publish 6379 to the LAN, `requirepass`, a read-only ACL user `mac` limited to
`mtap:evt:*` reads, the firewall rule for 192.168.1.216 only, the secret in the NAS secrets file and
`~/.config/mtap/nas-redis.env` on the Mac), so it waits for a session that can restart the etrade profile
outside market hours.

**Drafted 2026-11-01 22:20, not deployed** (deploy/container/compose.yml redis service, jobs.sh, synology/host.env):
the redis service writes its ACL file at start from `REDIS_PASSWORD` (user `default`, used by the store,
collector and jobs through common/bus) and `MAC_REDIS_PASSWORD` (user `mac`: only XREAD / XRANGE / XREVRANGE /
XLEN / XINFO STREAM on `mtap:evt:*`). Published on `MTAP_REDIS_PUBLISH` (NAS: 192.168.1.100:6379); jobs.sh forces
127.0.0.1 when either password is missing, so it is never on the LAN without AUTH. Tested on a local
redis-server 7.2: no AUTH refused; `default` reads / writes / consumer groups; `mac` reads only, and is refused
XADD, XREADGROUP, XACK, XTRIM, XGROUP, DEL, GET, FLUSHALL, CONFIG and keys outside `mtap:evt:*`.
Deploy (~45 min, after the 13:15 evening job): generate both passwords into `~/.mtap.env` on the NAS (no quotes)
and the `mac` one into `~/.config/mtap/nas-redis.env` (600); DSM firewall: allow 6379 from 192.168.1.216 only
(user, in DSM); `push.sh`; queue `build` (the NAS code predates the env-auth bus change); queue `stop`, `start`;
check the store and collector heartbeats and `redis-cli --user mac` from the Mac; queue `stop`. Then the Mac's
reader for the latest price per symbol (part of P5, ~1 h).

E*TRADE prices (~2 s old) are on the NAS Redis, which the Mac can't reach. Massive's full market is 15 min
delayed.
- **(A, recommended) Read-only access from the Mac to the NAS Redis**, with AUTH and a read-only ACL user,
  the port allowed only from the Mac's address. ~2 s old prices for the E*TRADE list stocks, and the same
  access the live scorer needs anyway (P8). Opens a NAS port on the LAN: needs the user's explicit OK.
- **(B) Massive snapshot from the Mac**: nothing opened, but prices are ~15 min old; TTL ≥ 20 min, and
  only good for the P5 stopgap, never for the scorer (the probe: no direction edge at d = 15).

## 4. Order

1. **Tonight / before 2026-11-02's open:** P11 (when the user says), P1 started (detached run).
2. **2026-11-02:** P3, P4, P2 (after the decision), P5 → live tests with `pick` on real symbols; P7.
3. **After P1:** if the d = 0 edge survives the bounce check, P6 then P8 then P9. If it doesn't, the intraday
   scorer stops at the base rate (P5) and the write-up goes to LEARNINGS / MODEL-STRATEGY; the next try is a
   different input (e.g. order-book or 1 s flow), not more of the same.

## 5. Should the live scorer start now?

Not the full scorer. Start with P1, the bounce check (a few hours). It decides whether the scorer's
only direction edge, the d = 0 one, is real or a bid–ask artefact; building the live feature stitching
(the costly part, §3F) before that risks weeks on an edge that isn't there. In parallel, P2–P5 and P7 are
needed either way (any scorer needs a fresh price, reference data, limits and a scheduled start), and they
make tomorrow's live tests real except for the edge.

## Change log
| date | change |
| --- | --- |
| 2026-11-01 | Created after the Step 0 probe and REVIEW item 14. |
| 2026-11-07 | P13 added: contract item M13 (RECOMMENDATION v2 forecast block) proposed to MBin; v1 unchanged. |
| 2026-11-07 | M13 confirmed; REVIEW items 15–18 answered. MTap's views: rank in tiers (skilled / skill unknown / no edge), then by lift. Supersede only by the same or a better tier. Stop and target anchored to `price_ref`. Entry cap `max_entry_pct`. Grading from market data only. P6 and P13 updated, P14 added. Fix: the recommender withheld candidates outside `NORMAL`, against item 12; it now sends them in every state. |
| 2026-11-12 | §0 added: every pending item by owner (waiting on MBin / the user / MTap to build / phase 2), checked against the code and both repos. MBin unchanged since 2026-11-07. P10: M3–M5 done, M6 pending. P11 done. P4 partly built (TTL clamp, repeat drop). P15 (Moves ladder, Q42), P16 (intraday v2 live scorer) and P17 (protocol 1.0) added. |
| 2026-11-01 | P2 decided: A (NAS Redis, read only, from the Mac). P1 built and started: `--parts p1` in `ml/scripts_intraday_probe.py`; five arms on the same rows (d0 / d1 inputs × entry at the last close / minute i's open / minute i's VWAP with outcomes from i + 1). Resumable: rerun `/tmp/mtap/intraday_probe/run_p1.sh`; finished months (`rows_p1/`) and (fold, arm) files (`p1/`) are skipped, files are written via a temp name. |
