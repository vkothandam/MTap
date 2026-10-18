"""One-off quality check: E*TRADE watchlist snapshots vs Massive WebSocket 1-second bars.

Both feeds land under DATA_ROOT (the NAS):
    intraday_feed/date=D/source=etrade_watchlist/*.parquet   E*TRADE polls (~2 s Experimental, ~20 s MidCap)
    agg_1s/date=D/source=massive_ws/*.parquet                Massive per-second bars (15-min delayed feed)

E*TRADE rows carry only the time the poll came back. Massive rows carry the bar start `s`; the
bar covers [s, s+1 s). Each poll is matched to the latest Massive bar that had ended by poll − L,
where L is an assumed lag (positive = the poll describes an earlier moment than its timestamp).

Checks, per tier (by market cap) and watchlist, over the regular session:
    A  are Massive timestamps event times? which L lines the feeds up? (lag scan, volume
       crossing, time of the opening / closing cross)
    B  price at a timestamp: |ΔP| bp, exact matches, staleness, inside E*TRADE's bid/ask
    C  volume at a timestamp: E*TRADE volume vs Massive av / dav
    D  VWAP over the session / trailing 5 / 60 min at checkpoints: E*TRADE rebuilt the way the
       intraday panel does it (export.series_for_day + window_vwap) vs Massive
    E  open, close and volume at close + 5 min vs grouped daily
    F  Massive VWAP semantics: is `a` (vwap_session) the running day VWAP and `vw` (vwap_i) the
       second's? dollar identity, vw inside [l, h], last bar vs daily vw, volume outside bars

Read-only; writes CSVs to /tmp/feed_quality/ and prints a summary.

    MTAP_DATA_ROOT=/Volumes/Backups/MTap-Data uv run python scripts/check_feed_quality.py \
        [--dates 2026-10-15 2026-10-16] [--top 300] [--small-max-mm 2000]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from sourcing_py.common import config, http
from sourcing_py.intraday import export

ET = ZoneInfo("America/New_York")
OUT = Path("/tmp/feed_quality")
DENSE_LAGS_MS = list(range(-2000, 4001, 250))
LAGS_MS = [-900_000, -60_000, *DENSE_LAGS_MS, 60_000]
MATCH_REL = 0.5e-4  # a price "matches" when equal to 4 dp or within 0.5 bp
WARMUP_MS = 60_000  # skip the first minute of Massive data (10-14 started mid-session)
MASSIVE_GAP_MS = 3_000  # no bar for any universe symbol this long = a feed outage
CHECKPOINTS = [time(h) for h in range(10, 16)] + [time(15, 59, 59)]
WINDOWS = {"session": None, "trail_5m": 300, "trail_60m": 3600}
MATCH = f"(abs(p - c) < 0.00005 OR abs(p / c - 1) <= {MATCH_REL})"

_MKTCAP_SQL = """
WITH shares AS (
    SELECT symbol, value AS qtco,
           row_number() OVER (PARTITION BY symbol ORDER BY fiscal_end DESC) rn
    FROM fundamentals WHERE line_item = 'QTCO' AND value IS NOT NULL
),
px AS (
    SELECT symbol, close,
           row_number() OVER (PARTITION BY symbol ORDER BY date DESC) rn
    FROM daily_bars WHERE close IS NOT NULL
)
SELECT s.symbol, s.qtco * p.close AS mktcap_mm
FROM shares s
JOIN px p USING (symbol)
JOIN symbols y USING (symbol)
WHERE s.rn = 1 AND p.rn = 1
  AND y.fundamentals_status = 'ok'          -- real operating stock, not an ETF/fund/index
"""

_POLL_METRICS = f"""
    count(*) AS polls,
    avg({MATCH}::INT) AS price_match,
    median(abs(p / c - 1) * 1e4) AS dp_bp_med,
    quantile_cont(abs(p / c - 1) * 1e4, 0.9) AS dp_bp_p90,
    quantile_cont(abs(p / c - 1) * 1e4, 0.99) AS dp_bp_p99,
    median(stale_ms) / 1000 AS stale_s_med,
    avg(CASE WHEN bid > 0 AND ask > 0 THEN (c BETWEEN bid - 0.01 AND ask + 0.01)::INT END) AS m_in_bidask,
    avg(CASE WHEN bid > 0 AND ask > 0 THEN (p BETWEEN bid - 0.01 AND ask + 0.01)::INT END) AS e_in_bidask,
    median(ev / av - 1) AS vol_rel_med,
    median(abs(ev / av - 1)) AS vol_abs_rel_med,
    quantile_cont(abs(ev / av - 1), 0.9) AS vol_abs_rel_p90,
    median(ev / dav - 1) AS vol_rel_dec_med
"""


def _mask(s: str) -> str:
    return re.sub(r"apiKey=[A-Za-z0-9]+", "apiKey=***", s)


def _get(client, cfg, path, **params):
    try:
        return http.get_json(client, f"{cfg['base_url']}{path}", params={**params, "apiKey": cfg["api_key"]})
    except Exception as exc:  # noqa: BLE001 — report and carry on
        print("  !", _mask(str(exc))[:200])
        return None


def _et(ms) -> str | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, ET).strftime("%H:%M:%S.%f")[:-3]


def _ms(day: date, t: time) -> int:
    return int(datetime.combine(day, t, tzinfo=ET).timestamp() * 1000)


def _q(con, sql: str) -> pa.Table:
    return con.execute(sql).to_arrow_table()


def _days(root: Path) -> list[date]:
    def have(sub):
        return {p.name.split("=", 1)[1] for p in (root / sub).glob("date=*") if any(p.glob("*/*.parquet"))}
    return [date.fromisoformat(d) for d in sorted(have("agg_1s") & have("intraday_feed"))]


def universe(con, root: Path, day: date, top: int, small_max_mm: float) -> dict[str, str]:
    """symbol → tier, from watchlist membership on `day` and market cap."""
    wl = dict(con.execute(f"""
        SELECT symbol, any_value(json_extract_string(source_meta, '$.watchlist'))
        FROM read_parquet('{root}/intraday_feed/date={day}/source=*/*.parquet', union_by_name=true)
        GROUP BY symbol""").fetchall())
    db = duckdb.connect(str(config.etrade_config()["db_path"]), read_only=True)
    try:
        caps = dict(db.execute(_MKTCAP_SQL).fetchall())
    finally:
        db.close()
    exp = sorted((s for s, w in wl.items() if w == "Experimental" and s in caps), key=lambda s: -caps[s])
    tiers = {s: ("top100" if i < 100 else f"top101_{top}") for i, s in enumerate(exp[:top])}
    for s, w in wl.items():
        if w == "MidCap" and s in caps:
            if caps[s] >= 10_000:
                tiers[s] = "midcap_10b"
            elif caps[s] < small_max_mm:
                tiers[s] = "small"
    return tiers


def grouped_daily(client, cfg, day: date) -> pa.Table:
    res = (_get(client, cfg, f"/v2/aggs/grouped/locale/us/market/stocks/{day}", adjusted="false") or {})
    rows = res.get("results") or []
    return pa.table({"symbol": [r["T"] for r in rows],
                     **{f"d_{k}": [float(r[k]) if r.get(k) is not None else None for r in rows]
                        for k in ("o", "c", "v", "vw")}})


def load(con, root: Path, day: date) -> None:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE m AS
        WITH r AS (
            SELECT symbol, epoch_ms(ts_utc) AS ms, price AS c, open_i AS o, high_i AS h, low_i AS l,
                   volume_i AS v, volume_cum AS av, volume_cum_dec AS dav, vwap_i AS vw,
                   vwap_session AS a, day_open AS op, avg_trade_size AS z
            FROM read_parquet('{root}/agg_1s/date={day}/*/*.parquet', union_by_name=true)
            WHERE symbol IN (SELECT symbol FROM u)
            QUALIFY row_number() OVER (PARTITION BY symbol, epoch_ms(ts_utc) ORDER BY volume_cum DESC) = 1)
        SELECT *, ms + 1000 AS end_ms,
               lag(av) OVER w AS pav, lag(a) OVER w AS pa
        FROM r WINDOW w AS (PARTITION BY symbol ORDER BY ms)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE e AS
        WITH r AS (
            SELECT symbol, epoch_ms(ts_utc) AS ms, price AS p, volume_cum AS ev, bid, ask, day_open AS eop,
                   json_extract_string(source_meta, '$.watchlist') AS wl
            FROM read_parquet('{root}/intraday_feed/date={day}/source=*/*.parquet', union_by_name=true)
            WHERE symbol IN (SELECT symbol FROM u)
            QUALIFY row_number() OVER (PARTITION BY symbol, epoch_ms(ts_utc)
                                       ORDER BY volume_cum DESC NULLS LAST) = 1)
        SELECT r.*, u.tier, coalesce(ev > lag(ev) OVER (PARTITION BY symbol ORDER BY ms), false) AS moved
        FROM r JOIN u USING (symbol)""")


def coverage(con, day: date, lo: int, open_ms: int, close_ms: int) -> pa.Table:
    """Massive outages (no bar for any universe symbol) and E*TRADE poll gaps, regular session."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE mgap AS
        WITH s AS (SELECT DISTINCT ms FROM m WHERE ms >= {lo} AND ms < {close_ms}),
             g AS (SELECT lag(ms) OVER (ORDER BY ms) AS g0, ms AS g1 FROM s)
        SELECT g0 + 1000 AS gap_start, g1 AS gap_end FROM g WHERE g1 - g0 > {MASSIVE_GAP_MS}""")
    t = _q(con, f"""
        SELECT 'massive' AS feed, NULL::VARCHAR AS wl, gap_start, gap_end FROM mgap
        UNION ALL
        SELECT 'etrade', wl, g0, g1 FROM (
            WITH s AS (SELECT DISTINCT wl, ms FROM e WHERE ms >= {open_ms} AND ms < {close_ms}),
                 g AS (SELECT wl, lag(ms) OVER (PARTITION BY wl ORDER BY ms) AS g0, ms AS g1 FROM s),
                 st AS (SELECT wl, median(g1 - g0) AS step FROM g GROUP BY wl)
            SELECT g.* FROM g JOIN st USING (wl) WHERE g1 - g0 > greatest(10000, 3 * step))
        ORDER BY 1, 2, 3""")
    gs, ge = t["gap_start"].to_pylist(), t["gap_end"].to_pylist()
    return pa.table({"date": [str(day)] * t.num_rows, "feed": t["feed"], "wl": t["wl"],
                     "start_et": [_et(x) for x in gs], "end_et": [_et(x) for x in ge],
                     "seconds": [(b - a) / 1000 for a, b in zip(gs, ge)]})


def lag_scan(con, day: date) -> pa.Table:
    """Check A(1): price-match rate and volume difference at each assumed lag, on polls whose volume moved."""
    out = []
    for lag in LAGS_MS:
        out.append(_q(con, f"""
            SELECT '{day}' AS date, tier, wl, {lag} AS lag_ms, count(*) AS polls,
                   avg({MATCH}::INT) AS price_match,
                   median(abs(ev / av - 1)) AS vol_abs_rel, median(ev / av - 1) AS vol_signed_rel
            FROM (SELECT *, ms - {lag} AS tq FROM eok WHERE moved) e
            ASOF JOIN m ON e.symbol = m.symbol AND e.tq >= m.end_ms
            GROUP BY ALL"""))
    return pa.concat_tables(out)


def best_lag(scan: pa.Table) -> int:
    """The dense-grid lag with the highest poll-weighted match rate on the Experimental watchlist."""
    rows = [r for r in scan.to_pylist() if r["lag_ms"] in DENSE_LAGS_MS]
    exp = [r for r in rows if r["wl"] == "Experimental"] or rows
    score: dict[int, list[float]] = {}
    for r in exp:
        s = score.setdefault(r["lag_ms"], [0.0, 0.0])
        s[0] += (r["price_match"] or 0) * r["polls"]
        s[1] += r["polls"]
    return max(score, key=lambda k: score[k][0] / score[k][1])


def crossing(con, day: date, open_ms: int, close_ms: int) -> tuple[pa.Table, pa.Table]:
    """Check A(2)/(3): when Massive volume first reaches each poll's E*TRADE volume; when the crosses print."""
    cross = _q(con, f"""
        SELECT '{day}' AS date, tier, wl, count(*) AS polls, avg((m.ms IS NULL)::INT) AS never_reached,
               quantile_cont(e.ms - m.end_ms, 0.25) / 1000 AS lag_s_q25,
               median(e.ms - m.end_ms) / 1000 AS lag_s_med,
               quantile_cont(e.ms - m.end_ms, 0.75) / 1000 AS lag_s_q75
        FROM (SELECT * FROM eok WHERE moved) e
        ASOF LEFT JOIN m ON e.symbol = m.symbol AND e.ev <= m.av
        GROUP BY ALL ORDER BY tier, wl""")
    jumps = _q(con, f"""
        SELECT m.symbol, any_value(u.tier) AS tier,
               arg_max(ms, av - pav) AS day_max_jump_ms,
               arg_max(ms, av - pav) FILTER (WHERE ms >= {open_ms} - 60000 AND ms < {open_ms} + 300000) AS open_jump_ms,
               arg_max(ms, av - pav) FILTER (WHERE ms >= {close_ms} - 60000 AND ms < {close_ms} + 300000) AS close_jump_ms
        FROM m JOIN u USING (symbol) WHERE pav IS NOT NULL GROUP BY m.symbol""")
    cols = {k: [_et(x) for x in jumps[f"{k}_ms"].to_pylist()] for k in ("day_max_jump", "open_jump", "close_jump")}
    jumps = pa.table({"date": [str(day)] * jumps.num_rows, "symbol": jumps["symbol"], "tier": jumps["tier"],
                      **{f"{k}_et": v for k, v in cols.items()}})
    return cross, jumps


def polls(con, day: date, lag: int) -> tuple[pa.Table, pa.Table]:
    """Checks B and C at one lag: per symbol-day and per tier (pooled over polls)."""
    con.execute(f"""CREATE OR REPLACE TEMP TABLE j AS
        SELECT e.symbol, e.tier, e.wl, p, ev, bid, ask, c, av, dav, e.tq - m.end_ms AS stale_ms
        FROM (SELECT *, ms - {lag} AS tq FROM eok) e
        ASOF JOIN m ON e.symbol = m.symbol AND e.tq >= m.end_ms""")
    head = f"'{day}' AS date, {lag} AS lag_ms"
    per_sym = _q(con, f"SELECT {head}, symbol, tier, wl, {_POLL_METRICS} FROM j GROUP BY ALL ORDER BY tier, symbol")
    per_tier = _q(con, f"SELECT {head}, tier, wl, {_POLL_METRICS} FROM j GROUP BY ALL ORDER BY tier, wl")
    return per_sym, per_tier


def vwaps(con, day: date, tiers: dict[str, str], lag: int, open_ms: int, close_ms: int) -> pa.Table:
    """Check D: E*TRADE VWAP rebuilt like the intraday panel vs Massive, at the same physical instants."""
    feed = export.load_feed(day)
    feed = feed.filter(pc.is_in(feed["symbol"], value_set=pa.array(list(tiers))))
    series, _ = export.series_for_day(feed, open_ms, close_ms)
    mt = _q(con, "SELECT symbol, end_ms, v, vw, av, a FROM m ORDER BY symbol, ms")
    sym = mt["symbol"].to_numpy(zero_copy_only=False)
    arr = {k: np.nan_to_num(mt[k].to_numpy(zero_copy_only=False).astype(float)) for k in ("end_ms", "v", "vw", "av", "a")}
    starts = np.flatnonzero(np.concatenate([[True], sym[1:] != sym[:-1]])) if len(sym) else np.array([], int)
    bounds = dict(zip(sym[starts], zip(starts, np.concatenate([starts[1:], [len(sym)]]))))
    gaps = con.execute("SELECT gap_start, gap_end FROM mgap").fetchall()
    cps = np.array([_ms(day, t) for t in CHECKPOINTS])
    rows = []
    for s in series:
        if s.symbol not in bounds:
            continue
        i0, i1 = bounds[s.symbol]
        end, v, vw, av, a = (arr[k][i0:i1] for k in ("end_ms", "v", "vw", "av", "a"))
        cpv, cv = np.concatenate([[0.0], np.cumsum(vw * v)]), np.concatenate([[0.0], np.cumsum(v)])
        for name, w in WINDOWS.items():
            lo = np.full(len(cps), open_ms) if w is None else cps - w * 1000
            e_vwap, e_vol = export.window_vwap(s, lo, cps)
            ok = export.gap_free(s, lo, cps, export.LONG_GAP_S if w is None or w >= 3600 else s.gap_short_s)
            ia, ib = np.clip(export.at(s, lo), 0, None), export.at(s, cps)
            xa, xb = s.T[ia] - lag, s.T[ib] - lag  # the moments the E*TRADE endpoints describe
            ka, kb = np.searchsorted(end, xa, "right") - 1, np.searchsorted(end, xb, "right") - 1
            for k in range(len(cps)):
                if not ok[k] or ka[k] < 0 or kb[k] <= ka[k] or np.isnan(e_vwap[k]):
                    continue
                if any(g0 < xb[k] and g1 > xa[k] for g0, g1 in gaps):
                    continue
                m_vol = av[kb[k]] - av[ka[k]]
                bars_vol = cv[kb[k] + 1] - cv[ka[k] + 1]
                m_cum = (a[kb[k]] * av[kb[k]] - a[ka[k]] * av[ka[k]]) / m_vol if m_vol > 0 else np.nan
                m_bars = (cpv[kb[k] + 1] - cpv[ka[k] + 1]) / bars_vol if bars_vol > 0 else np.nan
                rows.append({
                    "date": str(day), "symbol": s.symbol, "tier": tiers[s.symbol], "wl": s.watchlist,
                    "window": name, "checkpoint": CHECKPOINTS[k].isoformat(),
                    "e_vwap": float(e_vwap[k]), "m_vwap_cum": m_cum, "m_vwap_bars": m_bars,
                    "e_vs_cum_bp": (e_vwap[k] / m_cum - 1) * 1e4, "e_vs_bars_bp": (e_vwap[k] / m_bars - 1) * 1e4,
                    "bars_vs_cum_bp": (m_bars / m_cum - 1) * 1e4,
                    "e_vol": float(e_vol[k]), "m_vol": m_vol, "bars_vol": bars_vol,
                })
    return pa.Table.from_pylist(rows)


def daily(con, day: date, lag: int, close_ms: int) -> pa.Table:
    """Check E: open, close and volume at close + 5 min, E*TRADE vs Massive vs grouped daily."""
    late = close_ms + 300_000
    return _q(con, f"""
        WITH eo AS (
            SELECT symbol, any_value(tier) AS tier, any_value(wl) AS wl,
                   arg_min(eop, ms) FILTER (WHERE eop IS NOT NULL) AS e_open,
                   arg_min(p, ms) FILTER (WHERE ms >= {close_ms} + 30000) AS e_close,
                   arg_max(ev, ms) FILTER (WHERE ms <= {late}) AS e_vol_late,
                   max(ms) FILTER (WHERE ms <= {late}) AS e_late_ms
            FROM e GROUP BY symbol),
        mo AS (
            SELECT symbol,
                   arg_max(op, ms) FILTER (WHERE op IS NOT NULL) AS m_open,
                   arg_max(c, ms) FILTER (WHERE ms < {close_ms}) AS m_last_reg,
                   arg_max(c, av - pav) FILTER (WHERE ms >= {close_ms} AND ms < {close_ms} + 300000) AS m_cross,
                   arg_max(av, ms) FILTER (WHERE end_ms <= {late} - {lag}) AS m_av_late,
                   arg_max(dav, ms) FILTER (WHERE end_ms <= {late} - {lag}) AS m_dav_late,
                   arg_max(av, ms) AS m_av_last
            FROM m GROUP BY symbol)
        SELECT '{day}' AS date, eo.*, mo.* EXCLUDE (symbol), d.* EXCLUDE (symbol),
               (e_open / d_o - 1) * 1e4 AS e_open_bp, (m_open / d_o - 1) * 1e4 AS m_open_bp,
               (e_close / d_c - 1) * 1e4 AS e_close_bp, (m_last_reg / d_c - 1) * 1e4 AS m_last_reg_bp,
               (m_cross / d_c - 1) * 1e4 AS m_cross_bp,
               e_vol_late / m_av_late - 1 AS e_vs_m_vol_late, e_vol_late / m_dav_late - 1 AS e_vs_mdec_vol_late,
               m_av_last / d_v AS m_av_last_over_daily
        FROM eo LEFT JOIN mo USING (symbol) LEFT JOIN d USING (symbol) ORDER BY tier, symbol""")


def semantics(con, day: date, open_ms: int, close_ms: int) -> tuple[pa.Table, pa.Table]:
    """Check F: is `a` the running day VWAP and `vw` the bar's? Massive data only."""
    in_range = "(vw BETWEEN l - 1e-6 AND h + 1e-6)"
    clean = "(av - pav = v AND pav > 0 AND v > 0)"
    pred = "((pa * pav + vw * v) / av)"  # the next `a` if it is the running VWAP over `av`
    bars = _q(con, f"""
        SELECT '{day}' AS date, tier, count(*) AS bars, avg({clean}::INT) AS clean_share,
               median(abs({pred} / a - 1) * 1e4) FILTER (WHERE {clean}) AS cum_pred_err_bp,
               quantile_cont(abs({pred} / a - 1) * 1e4, 0.99) FILTER (WHERE {clean}) AS cum_pred_err_bp_p99,
               median(abs(vw / a - 1) * 1e4) FILTER (WHERE {clean}) AS vw_vs_a_bp,
               avg({in_range}::INT) AS vw_in_range,
               avg((o = h AND h = l)::INT) AS flat_share,
               avg((o = h AND h = l)::INT) FILTER (WHERE NOT {in_range}) AS out_flat,
               avg((z < 100)::INT) FILTER (WHERE NOT {in_range}) AS out_oddlot,
               avg((o = h AND h = l AND z < 100)::INT) FILTER (WHERE NOT {in_range}) AS out_flat_oddlot,
               sum(av - pav) AS av_growth, sum(v) / sum(av - pav) AS in_bars_share,
               sum(av - pav - v) FILTER (WHERE ms < {open_ms}) / sum(av - pav) AS out_pre,
               sum(av - pav - v) FILTER (WHERE ms >= {open_ms} AND ms < {open_ms} + 1000) / sum(av - pav) AS out_open_bar,
               sum(av - pav - v) FILTER (WHERE ms >= {open_ms} + 1000 AND ms < {close_ms}) / sum(av - pav) AS out_regular,
               sum(av - pav - v) FILTER (WHERE ms >= {close_ms} AND ms < {close_ms} + 60000) / sum(av - pav) AS out_close_min,
               sum(av - pav - v) FILTER (WHERE ms >= {close_ms} + 60000) / sum(av - pav) AS out_after
        FROM m JOIN u USING (symbol) WHERE pav IS NOT NULL GROUP BY tier ORDER BY tier""")
    last = _q(con, f"""
        WITH x AS (
            SELECT symbol, 'last_regular' AS point, arg_max(vw, ms) AS vw, arg_max(a, ms) AS a, arg_max(c, ms) AS c
            FROM m WHERE ms < {close_ms} GROUP BY symbol
            UNION ALL
            SELECT symbol, 'last_bar', arg_max(vw, ms), arg_max(a, ms), arg_max(c, ms) FROM m GROUP BY symbol)
        SELECT '{day}' AS date, x.symbol, u.tier, x.point, x.vw, x.a, x.c, d.d_vw,
               abs(x.vw / d.d_vw - 1) * 1e4 AS vw_vs_daily_bp, abs(x.a / d.d_vw - 1) * 1e4 AS a_vs_daily_bp,
               abs(x.vw / x.c - 1) * 1e4 AS vw_vs_close_bp, abs(x.a / x.c - 1) * 1e4 AS a_vs_close_bp
        FROM x JOIN u USING (symbol) JOIN d USING (symbol) ORDER BY u.tier, x.symbol, x.point""")
    return bars, last


def _show(title: str, t: pa.Table) -> None:
    print(f"\n== {title} ==")
    cols = t.column_names
    rows = [["" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.values()]
            for r in t.to_pylist()]
    w = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(r)))


def _write(name: str, tables: list[pa.Table]) -> pa.Table | None:
    tables = [t for t in tables if t is not None and t.num_rows]
    if not tables:
        return None
    t = pa.concat_tables(tables, promote_options="permissive")
    pcsv.write_csv(t, OUT / f"{name}.csv")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", type=date.fromisoformat)
    ap.add_argument("--top", type=int, default=300, help="Experimental symbols by market cap")
    ap.add_argument("--small-max-mm", type=float, default=2000.0, help="MidCap small-cap ceiling, $ millions")
    args = ap.parse_args()
    root = config.data_root()
    if not (root / "agg_1s").is_dir():
        sys.exit(f"no agg_1s under {root}; set MTAP_DATA_ROOT (e.g. /Volumes/Backups/MTap-Data)")
    days = args.dates or _days(root)
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    tiers = universe(con, root, days[-1], args.top, args.small_max_mm)
    con.register("u_arrow", pa.table({"symbol": list(tiers), "tier": list(tiers.values())}))
    con.execute("CREATE TEMP TABLE u AS SELECT * FROM u_arrow")
    _show("universe", _q(con, "SELECT tier, count(*) AS symbols FROM u GROUP BY tier ORDER BY tier"))

    cfg = config.massive_config()
    res: dict[str, list[pa.Table]] = {}
    with http.get_client(timeout=60) as client:
        for day in days:
            print(f"\n######## {day}")
            open_ms, close_ms = export._session_ms(day)
            con.register("d_arrow", grouped_daily(client, cfg, day))
            con.execute("CREATE OR REPLACE TEMP TABLE d AS SELECT * FROM d_arrow")
            load(con, root, day)
            first_m = con.execute("SELECT min(ms) FROM m").fetchone()[0]
            if first_m is None:
                print("  no Massive bars for the universe; skipped")
                continue
            lo = max(open_ms, first_m + WARMUP_MS)
            gaps = coverage(con, day, lo, open_ms, close_ms)
            res.setdefault("coverage_gaps", []).append(gaps)
            con.execute(f"""CREATE OR REPLACE TEMP TABLE eok AS SELECT * FROM e
                WHERE ms >= {lo} AND ms < {close_ms}
                  AND NOT EXISTS (SELECT 1 FROM mgap g WHERE e.ms BETWEEN g.gap_start AND g.gap_end + 5000)""")
            n = con.execute("SELECT (SELECT count(*) FROM e), (SELECT count(*) FROM eok), (SELECT count(*) FROM m)").fetchone()
            print(f"  E*TRADE polls {n[0]:,} (analysed {n[1]:,} from {_et(lo)} ET), Massive bars {n[2]:,}, "
                  f"gaps: {gaps.num_rows}")
            scan = lag_scan(con, day)
            lag = best_lag(scan)
            print(f"  best lag (Experimental, poll − bar end): {lag} ms")
            res.setdefault("lag_scan", []).append(scan)
            cross, jumps = crossing(con, day, open_ms, close_ms)
            res.setdefault("volume_crossing", []).append(cross)
            res.setdefault("cross_times", []).append(jumps)
            for L in sorted({0, lag}):
                per_sym, per_tier = polls(con, day, L)
                res.setdefault("per_symbol_day", []).append(per_sym)
                res.setdefault("per_tier", []).append(per_tier)
            res.setdefault("vwap_windows", []).append(vwaps(con, day, tiers, lag, open_ms, close_ms))
            res.setdefault("daily", []).append(daily(con, day, lag, close_ms))
            bars, last = semantics(con, day, open_ms, close_ms)
            res.setdefault("massive_semantics", []).append(bars)
            res.setdefault("massive_last_bar", []).append(last)

    out = {k: _write(k, v) for k, v in res.items()}
    summarize(con, out)
    print(f"\nCSVs: {OUT}/")


def summarize(con, out: dict[str, pa.Table | None]) -> None:
    for k, t in out.items():
        if t is not None:
            con.register(k, t)
    if out.get("coverage_gaps") is not None:
        _show("coverage gaps (regular session)", _q(con, """
            SELECT date, feed, wl, count(*) AS gaps, sum(seconds) AS seconds, max(seconds) AS longest
            FROM coverage_gaps GROUP BY ALL ORDER BY ALL"""))
    if out.get("lag_scan") is None:
        return
    _show("A1 lag scan: price-match rate on moved polls, all dates (lag = poll − Massive bar end)", _q(con, """
        PIVOT (SELECT lag_ms, tier || '/' || wl AS g, sum(price_match * polls) / sum(polls) AS r
               FROM lag_scan GROUP BY ALL) ON g USING first(r) ORDER BY lag_ms"""))
    _show("A1 lag scan: median signed E*TRADE/Massive volume − 1, all dates", _q(con, """
        PIVOT (SELECT lag_ms, tier || '/' || wl AS g, median(vol_signed_rel) AS r FROM lag_scan GROUP BY ALL)
        ON g USING first(r) ORDER BY lag_ms"""))
    _show("A1 best lag per date and tier", _q(con, f"""
        SELECT date, tier, wl, arg_max(lag_ms, price_match) AS best_lag_ms, max(price_match) AS match_at_best
        FROM lag_scan WHERE lag_ms BETWEEN {DENSE_LAGS_MS[0]} AND {DENSE_LAGS_MS[-1]} GROUP BY ALL ORDER BY ALL"""))
    _show("A2 volume crossing: poll time − end of the bar where Massive av first ≥ E*TRADE volume (s)",
          _q(con, "SELECT * FROM volume_crossing ORDER BY ALL"))
    _show("A3 when the biggest av jumps print (Massive bar start, ET)", _q(con, """
        SELECT date, count(*) AS symbols,
               avg((day_max_jump_et >= '16:00:00' AND day_max_jump_et < '16:00:05')::INT) AS day_max_at_1600,
               avg((close_jump_et >= '16:00:00' AND close_jump_et < '16:00:05')::INT) AS close_at_1600,
               avg((open_jump_et >= '09:30:00' AND open_jump_et < '09:30:05')::INT) AS open_at_0930,
               avg((open_jump_et >= '09:30:00' AND open_jump_et < '09:31:00')::INT) AS open_in_0930_min,
               mode(left(close_jump_et, 8)) AS close_mode, mode(left(open_jump_et, 8)) AS open_mode
        FROM cross_times GROUP BY date ORDER BY date"""))
    _show("B/C price and volume at a timestamp, per tier (pooled polls)", _q(con, """
        SELECT date, lag_ms, tier, wl, polls, price_match, dp_bp_med, dp_bp_p90, dp_bp_p99, stale_s_med,
               m_in_bidask, e_in_bidask, vol_rel_med, vol_abs_rel_med, vol_abs_rel_p90, vol_rel_dec_med
        FROM per_tier ORDER BY date, lag_ms, tier"""))
    if out.get("vwap_windows") is not None:
        _show("D VWAP: E*TRADE (panel method) vs Massive, |bp| over symbols × checkpoints", _q(con, """
            SELECT tier, "window", count(*) AS n,
                   median(abs(e_vs_cum_bp)) AS e_vs_cum_med, quantile_cont(abs(e_vs_cum_bp), 0.9) AS e_vs_cum_p90,
                   median(abs(e_vs_bars_bp)) AS e_vs_bars_med, median(abs(bars_vs_cum_bp)) AS bars_vs_cum_med,
                   median(e_vol / m_vol - 1) AS e_vol_rel_med, median(bars_vol / m_vol) AS bars_vol_share
            FROM vwap_windows GROUP BY ALL ORDER BY "window", tier"""))
    _show("E open / close / late volume vs grouped daily (median |bp|, share within 0.5 bp)", _q(con, """
        SELECT date, tier, count(*) AS n,
               median(abs(e_open_bp)) AS e_open, avg((abs(e_open_bp) <= 0.5)::INT) AS e_open_ok,
               median(abs(m_open_bp)) AS m_open, avg((abs(m_open_bp) <= 0.5)::INT) AS m_open_ok,
               median(abs(e_close_bp)) AS e_close, avg((abs(e_close_bp) <= 0.5)::INT) AS e_close_ok,
               median(abs(m_last_reg_bp)) AS m_last_reg, avg((abs(m_last_reg_bp) <= 0.5)::INT) AS m_last_reg_ok,
               median(abs(m_cross_bp)) AS m_cross, avg((abs(m_cross_bp) <= 0.5)::INT) AS m_cross_ok,
               median(e_vs_m_vol_late) AS e_vs_m_vol, median(e_vs_mdec_vol_late) AS e_vs_mdec_vol,
               median(m_av_last_over_daily) AS m_av_over_daily
        FROM daily GROUP BY ALL ORDER BY ALL"""))
    _show("F Massive bars: is `a` the running VWAP over `av`? (clean bars: Δav = v)",
          _q(con, "SELECT * EXCLUDE (av_growth) FROM massive_semantics ORDER BY ALL"))
    _show("F Massive last bar vs grouped daily vw (median |bp|; share where `a` is closer than `vw`)", _q(con, """
        SELECT date, tier, point, count(*) AS n,
               median(vw_vs_daily_bp) AS vw_vs_daily, median(a_vs_daily_bp) AS a_vs_daily,
               avg((a_vs_daily_bp < vw_vs_daily_bp)::INT) AS a_closer,
               median(vw_vs_close_bp) AS vw_vs_close, median(a_vs_close_bp) AS a_vs_close
        FROM massive_last_bar GROUP BY ALL ORDER BY ALL"""))


if __name__ == "__main__":
    main()
