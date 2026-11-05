# Live recommendations to MBin: plan to production mode

Status: plan, 2026-11-01. Nothing below is built unless marked done. The contract is
`MBin/docs/contracts/` (protocol v1.0-draft, phase 1); the model design is
[intraday-design.md](intraday-design.md); the probe result is in LEARNINGS `tag-ml-intraday`
("Step 0 probe: results"). Where things run today: [STATUS.md](STATUS.md).

**Goal.** Every RECOMMENDATION MTap sends is valid (it already is) **and** real in every field: a fresh
price, real reference data, and a `score` that is a calibrated P(+target before −stop) from a model
with measured skill. Then the same with a scheduled start, grading, and limits, so it can run unattended.

## 1. Where it stands (2026-11-01)

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
| P6 | **Grading**: every published rec checked against market data (target / stop touched first, within the horizon), daily report, calibration of `score` | protocol §5 audit; the only proof the probabilities hold | MTap | ~½ day | P2 |
| P7 | **Scheduled start on the Mac**: launchd job for the recommender on trading days ~06:00–13:30 PT, restart on failure, logs, `caffeinate`; the contract Redis started by MBin's tool | today both start by hand | MTap (recommender), MBin (contract Redis) | ~1 h each | — |
| P8 | **Not going ahead (P1: no direction edge).** **Live scorer v0** (if P1 holds): d = 0 model on the live-polled stocks, features stitched from Massive (to t − 15) + E*TRADE polls (last 15 min, §3F), writes `mtap:int:REC_CANDIDATE`; trained with the training-time seam emulated (1 s bars on a 2 s grid) | the real `score`; replaces P5's base rate under `strategy_id: intraday.pressure` | MTap ml | ~1–2 weeks | P1, P2, P6 |
| P9 | **Shadow period**: P8 publishes for ≥ 10 sessions with grading before the operator acts on it; promote only if graded calibration and up-first skill hold live | live data differs from the backtest (feed gaps, the seam) | MTap + operator | 2 weeks elapsed | P8 |
| P10 | **MBin side**: TRADING_PARAMETERS on the contract Redis, Signals intake (M4), sandbox order path (M5), switch to real MTap (M6) | without parameters MTap can't clamp TTLs; without intake nothing reaches the Console | MBin | MBin's MVP plan | — |
| P11 | **Commit and push** the env-auth bus change and these docs | before the live tests | user says when | minutes | — |
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
| 2026-11-01 | P2 decided: A (NAS Redis, read only, from the Mac). P1 built and started: `--parts p1` in `ml/scripts_intraday_probe.py`; five arms on the same rows (d0 / d1 inputs × entry at the last close / minute i's open / minute i's VWAP with outcomes from i + 1). Resumable: rerun `/tmp/mtap/intraday_probe/run_p1.sh`; finished months (`rows_p1/`) and (fold, arm) files (`p1/`) are skipped, files are written via a temp name. |
