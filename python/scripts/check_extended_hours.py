"""One-off accuracy check: where do pre-market / after-hours prices and volumes come from?

Compares, for a sample of common stocks over a few recent sessions:
    grouped daily   /v2/aggs/grouped/locale/us/market/stocks/{date}   (what daily_bars holds)
    open-close      /v1/open-close/{T}/{date}                         (preMarket / afterHours prices)
    1-min and 30-min aggregates /v2/aggs/ticker/{T}/range/{n}/minute/{from}/{to}

Questions it answers:
    - is grouped `v` the regular session (09:30–16:00 ET) or the whole day (04:00–20:00)?
    - does open-close preMarket / afterHours match the first pre-market / last after-hours bar?
    - does the closing cross land in the 16:00 bar (its volume, its close vs the official close)?
    - how much extended-hours volume sits within 30 min of the session (09:00–09:30, 16:00–16:30)?
    - do 30-min bars sum to the same session volumes as 1-min bars?

Read-only; writes /tmp/extended_hours_check.csv and prints a summary.

    uv run python scripts/check_extended_hours.py --dates 2026-10-12 2026-10-13 2026-10-14
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from datetime import datetime, time
from zoneinfo import ZoneInfo

import httpx

from sourcing_py.common import config, http

ET = ZoneInfo("America/New_York")
LIQUID = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "JPM", "XOM"]
MID = ["ROKU", "ETSY", "CROX", "DKS", "FIVE", "BILL"]
OUT = "/tmp/extended_hours_check.csv"


def _mask(s: str) -> str:
    return re.sub(r"apiKey=[A-Za-z0-9]+", "apiKey=***", s)


def _get(client, cfg, path, **params):
    try:
        return http.get_json(client, f"{cfg['base_url']}{path}", params={**params, "apiKey": cfg["api_key"]})
    except Exception as exc:  # noqa: BLE001 — report and carry on
        print("  !", _mask(str(exc))[:200])
        return None


def _aggs(client, cfg, ticker, n, d0, d1):
    out, url = [], f"{cfg['base_url']}/v2/aggs/ticker/{ticker}/range/{n}/minute/{d0}/{d1}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": cfg["api_key"]}
    while url:
        try:
            page = http.get_json(client, url, params=params)
        except Exception as exc:  # noqa: BLE001
            print("  !", _mask(str(exc))[:200])
            break
        out += page.get("results") or []
        nxt = page.get("next_url")
        url, params = (str(httpx.URL(nxt).copy_merge_params({"apiKey": cfg["api_key"]})), None) if nxt else (None, None)
    for b in out:
        b["et"] = datetime.fromtimestamp(b["t"] / 1000, ET)
    return out


def _split(bars, day):
    """Session sums for one day from bars labelled by their start time."""
    bars = [b for b in bars if b["et"].date().isoformat() == day]
    t = lambda b: b["et"].time()
    pre = [b for b in bars if t(b) < time(9, 30)]
    reg = [b for b in bars if time(9, 30) <= t(b) < time(16)]
    post = [b for b in bars if t(b) >= time(16)]
    v = lambda bs: sum(b["v"] for b in bs)
    at = lambda hh, mm: next((b for b in bars if t(b) == time(hh, mm)), None)
    return {
        "bars": len(bars), "v_pre": v(pre), "v_reg": v(reg), "v_post": v(post), "v_all": v(bars),
        "v_pre_last30": v([b for b in pre if t(b) >= time(9)]),
        "v_post_first30": v([b for b in post if t(b) < time(16, 30)]),
        "first_pre_open": pre[0]["o"] if pre else None, "first_pre_time": t(pre[0]).isoformat() if pre else None,
        "last_post_close": post[-1]["c"] if post else None,
        "last_post_time": t(post[-1]).isoformat() if post else None,
        "reg_open": reg[0]["o"] if reg else None, "reg_last_close": reg[-1]["c"] if reg else None,
        "bar0930_o": (b := at(9, 30)) and b["o"], "bar1600_v": (b := at(16, 0)) and b["v"],
        "bar1600_c": (b := at(16, 0)) and b["c"], "bar1559_v": (b := at(15, 59)) and b["v"],
        "bar1601_v": (b := at(16, 1)) and b["v"],
    }


def _pick_small(grouped, n=6):
    """Low-volume names from grouped daily (common-stock check done by the caller)."""
    rows = sorted((r for r in grouped if r.get("v", 0) > 20_000 and r.get("c", 0) > 1
                   and r["T"].isalpha() and len(r["T"]) <= 4), key=lambda r: r["v"])
    step = max(1, len(rows) // 400)
    return [r["T"] for r in rows[: 400 * step : step]][: n * 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    args = ap.parse_args()
    cfg, dates = config.massive_config(), args.dates
    with http.get_client(timeout=60) as client:
        grouped = {d: {r["T"]: r for r in (_get(client, cfg, f"/v2/aggs/grouped/locale/us/market/stocks/{d}",
                                                  adjusted="true") or {}).get("results") or []}
                   for d in dates}
        small = []
        for t in _pick_small(list(grouped[dates[-1]].values())):
            ref = _get(client, cfg, "/v3/reference/tickers", ticker=t, market="stocks") or {}
            if any(r.get("type") == "CS" for r in ref.get("results") or []):
                small.append(t)
            if len(small) == 6:
                break
        tickers = LIQUID + MID + small
        print(f"tickers ({len(tickers)}): {' '.join(tickers)}")
        rows = []
        for tk in tickers:
            m1 = _aggs(client, cfg, tk, 1, dates[0], dates[-1])
            m30 = _aggs(client, cfg, tk, 30, dates[0], dates[-1])
            for d in dates:
                g = grouped[d].get(tk) or {}
                oc = _get(client, cfg, f"/v1/open-close/{tk}/{d}", adjusted="true") or {}
                s1, s30 = _split(m1, d), _split(m30, d)
                rows.append({
                    "ticker": tk, "date": d, "tier": "liquid" if tk in LIQUID else "mid" if tk in MID else "small",
                    "g_o": g.get("o"), "g_c": g.get("c"), "g_v": g.get("v"),
                    "oc_open": oc.get("open"), "oc_close": oc.get("close"), "oc_v": oc.get("volume"),
                    "oc_pre": oc.get("preMarket"), "oc_after": oc.get("afterHours"),
                    **{f"m1_{k}": v for k, v in s1.items()},
                    **{f"m30_{k}": s30[k] for k in ("bars", "v_pre", "v_reg", "v_post", "v_all", "bar1600_v")},
                })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _summary(rows)
    print(f"\nrows: {OUT}")


def _ratio(rows, a, b):
    xs = [r[a] / r[b] for r in rows if r.get(a) and r.get(b)]
    return f"median {st.median(xs):.4f}  min {min(xs):.4f}  max {max(xs):.4f}  (n={len(xs)})" if xs else "n/a"


def _match(rows, a, b, tol=0.005):
    xs = [(r[a], r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
    hit = sum(abs(x - y) <= tol * max(abs(y), 1e-9) for x, y in xs)
    return f"{hit}/{len(xs)} within {tol:.1%}"


def _summary(rows):
    print("\n== volume: which window does grouped `v` cover? ==")
    print("grouped v / 1-min regular 09:30–16:00 :", _ratio(rows, "g_v", "m1_v_reg"))
    print("grouped v / 1-min regular + 16:00 bar :",
          _ratio([{**r, "x": (r["m1_v_reg"] or 0) + (r["m1_bar1600_v"] or 0)} for r in rows], "g_v", "x"))
    print("grouped v / 1-min whole day 04–20     :", _ratio(rows, "g_v", "m1_v_all"))
    print("open-close volume / grouped v         :", _ratio(rows, "oc_v", "g_v"))
    print("30-min whole day / 1-min whole day    :", _ratio(rows, "m30_v_all", "m1_v_all"))
    print("30-min regular / 1-min regular        :", _ratio(rows, "m30_v_reg", "m1_v_reg"))
    print("\n== prices ==")
    print("open-close preMarket vs first pre-market bar open :", _match(rows, "oc_pre", "m1_first_pre_open"))
    print("open-close afterHours vs last after-hours close   :", _match(rows, "oc_after", "m1_last_post_close"))
    print("grouped open vs 09:30 bar open                    :", _match(rows, "g_o", "m1_bar0930_o"))
    print("grouped close vs 16:00 bar close                  :", _match(rows, "g_c", "m1_bar1600_c"))
    print("grouped close vs 15:59 bar close                  :", _match(rows, "g_c", "m1_reg_last_close"))
    print("\n== closing cross: is the 16:00 bar unusually large? ==")
    print("16:00 bar v / 15:59 bar v :", _ratio(rows, "m1_bar1600_v", "m1_bar1559_v"))
    print("16:00 bar v / 16:01 bar v :", _ratio(rows, "m1_bar1600_v", "m1_bar1601_v"))
    print("16:00 bar v / all post v  :", _ratio(rows, "m1_bar1600_v", "m1_v_post"))
    print("\n== extended hours near the session ==")
    print("pre 09:00–09:30 / all pre    :", _ratio(rows, "m1_v_pre_last30", "m1_v_pre"))
    print("post 16:00–16:30 / all post  :", _ratio(rows, "m1_v_post_first30", "m1_v_post"))
    print("extended / regular volume    :",
          _ratio([{**r, "x": (r["m1_v_pre"] or 0) + (r["m1_v_post"] or 0)} for r in rows], "x", "m1_v_reg"))
    print("first pre-market bar times   :", sorted({r["m1_first_pre_time"] for r in rows if r["m1_first_pre_time"]})[:8])
    print("last after-hours bar times   :", sorted({r["m1_last_post_time"] for r in rows if r["m1_last_post_time"]})[-8:])
    for tier in ("liquid", "mid", "small"):
        sub = [r for r in rows if r["tier"] == tier]
        print(f"\n[{tier}] grouped v / 1-min regular: {_ratio(sub, 'g_v', 'm1_v_reg')}"
              f"\n[{tier}] grouped v / 1-min all    : {_ratio(sub, 'g_v', 'm1_v_all')}"
              f"\n[{tier}] afterHours match        : {_match(sub, 'oc_after', 'm1_last_post_close')}")


if __name__ == "__main__":
    main()
