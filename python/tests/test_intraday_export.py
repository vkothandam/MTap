"""export-tft: the intraday panel. Windows, targets, masks and the leakage rule."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sourcing_py.intraday import export, feed

DAY = date(2025, 12, 19)  # a full session: 14:30 – 21:00 UTC
OPEN_MS = int(datetime(2025, 12, 19, 14, 30, tzinfo=UTC).timestamp() * 1000)
CLOSE_MS = int(datetime(2025, 12, 19, 21, 0, tzinfo=UTC).timestamp() * 1000)


def make_series(ms, price, vol, day_open=None, prev_close=100.0):
    n = len(ms)
    s, why = export.build_series(
        "AAA", "Experimental", np.asarray(ms, np.int64), np.asarray(price, float),
        np.asarray(vol, float), np.full(n, np.nan if day_open is None else day_open),
        np.full(n, prev_close), OPEN_MS, CLOSE_MS)
    assert s is not None, why
    return s


def poll_feed(symbols, seed=0, until_ms=None, every_s=2, perturb_after_ms=None, shift_days=0):
    """Synthetic polls from 09:20 to 16:01 ET (or `until_ms`) for each symbol."""
    rng = np.random.default_rng(seed)
    start = OPEN_MS - 10 * 60 * 1000
    end = until_ms or CLOSE_MS + 60 * 1000
    ts = np.arange(start, end, every_s * 1000)
    rows = {k: [] for k in ("symbol", "ts_utc", "price", "volume_cum", "day_open", "prev_close", "source_meta")}
    for j, sym in enumerate(symbols):
        base = 50.0 + 10 * j
        steps = rng.normal(0, 0.0005, len(ts))
        vol_inc = rng.integers(0, 500, len(ts)).astype(float)
        if perturb_after_ms is not None:
            later = ts > perturb_after_ms
            steps[later] = np.random.default_rng(seed + 99).normal(0.001, 0.002, later.sum())
            vol_inc[later] *= 3
        price = base * np.exp(np.cumsum(steps))
        price[ts <= OPEN_MS] = base  # premarket shows the previous close
        vol_inc[ts <= OPEN_MS] = 0
        vol = 1000 + np.cumsum(vol_inc)
        for t, p, v in zip(ts, price, vol):
            rows["symbol"].append(sym)
            rows["ts_utc"].append(datetime.fromtimestamp(t / 1000, UTC) + timedelta(days=shift_days))
            rows["price"].append(float(p))
            rows["volume_cum"].append(float(v))
            rows["day_open"].append(float(base * 1.001) if t > OPEN_MS else None)
            rows["prev_close"].append(base)
            rows["source_meta"].append(json.dumps({"watchlist": "Experimental"}))
    n = len(rows["symbol"])
    cols = {f.name: [None] * n for f in feed.ARROW_SCHEMA}
    cols.update(rows, source=["etrade_watchlist"] * n, kind=["snapshot"] * n)
    return pa.table(cols, schema=feed.ARROW_SCHEMA)


def fake_context(symbols, sector=lambda i: "S1" if i % 2 else "S2"):
    def ctx(day, syms):
        out = {}
        for i, sym in enumerate(symbols):
            base = 50.0 + 10 * i
            out[sym] = {
                "close_1": base, "vwap_1": base * 0.999, "high_1": base * 1.01, "low_1": base * 0.99,
                "f_1": 1.0, "f_day": 1.0, "close_ratio": 1.002, "vwap_pct_prev_day": 0.001,
                "intraday_spread_pct": 0.02, "close_position_pct": 0.001, "avg_vol_10d": 2e6,
                "sector_code": sector(i), "days_since_results": 30.0, "released_at": None,
                "vwap_d_raw": base,
            }
        out["__meta__"] = {"days_since_prev_session": 1}
        return out

    return ctx


@pytest.fixture
def roots(tmp_path, monkeypatch):
    monkeypatch.setenv("MTAP_DATA_ROOT", str(tmp_path / "nas"))
    monkeypatch.setenv("OUT_ROOT", str(tmp_path / "out"))
    return tmp_path


def write_feed(table, day=DAY):
    d = export.nas_dir() / f"date={day.isoformat()}" / "source=etrade_watchlist"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, d / "mbin.parquet")


SYMS = [f"S{i:02d}" for i in range(14)]


def bit(df, b, rows=slice(None)):
    return (df.loc[rows, "tmask"].to_numpy() >> b) & 1 == 1


# --------------------------------------------------------------------------- windows


def test_window_vwap_equals_brute_force():
    rng = np.random.default_rng(1)
    ms = OPEN_MS - 5000 + 2000 * np.arange(400)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 400)))
    vol = 10_000 + np.cumsum(rng.integers(0, 300, 400))
    s = make_series(ms, price, vol, day_open=100.5)
    a = np.array([OPEN_MS + 30_000, OPEN_MS + 100_000])
    b = a + 60_000
    got, gvol = export.window_vwap(s, a, b)
    for k in range(2):
        sel = (s.T[1:] > a[k]) & (s.T[1:] <= b[k])
        dv = np.diff(s.CV)[sel]
        px = (np.diff(s.CPV)[sel])
        assert gvol[k] == pytest.approx(dv.sum())
        assert got[k] == pytest.approx(px.sum() / dv.sum())
        # the same thing from raw polls: each volume change at the mid of the two prices
        i = np.flatnonzero(sel) + 1
        raw_px = 0.5 * (s.P[i - 1] + s.P[i])
        raw_dv = s.CV[i] - s.CV[i - 1]
        assert got[k] == pytest.approx((raw_px * raw_dv).sum() / raw_dv.sum())


def test_opening_cross_is_priced_at_the_day_open():
    ms = [OPEN_MS - 2000, OPEN_MS + 3000, OPEN_MS + 5000]
    s = make_series(ms, [100.0, 101.0, 101.2], [5000, 9000, 9100], day_open=100.6)
    assert s.CV[-1] == 4100  # premarket volume is the baseline, not part of the session
    assert s.CPV[1] == pytest.approx(4000 * 100.6)
    assert s.CPV[2] - s.CPV[1] == pytest.approx(100 * 0.5 * (101.0 + 101.2))


def test_no_premarket_poll_drops_the_series():
    s, why = export.build_series("AAA", "x", np.array([OPEN_MS + 1000]), np.array([1.0]), np.array([1.0]),
                                 np.array([np.nan]), np.array([1.0]), OPEN_MS, CLOSE_MS)
    assert s is None and why == "no_premarket"


def test_gap_free_and_polls_in():
    ms = [OPEN_MS - 1000, OPEN_MS + 1000, OPEN_MS + 3000, OPEN_MS + 63_000, OPEN_MS + 65_000]
    s = make_series(ms, [1, 1, 1, 1, 1], [0, 1, 2, 3, 4])
    a = np.array([OPEN_MS, OPEN_MS + 3000, OPEN_MS + 63_000])
    b = a + 2000
    assert export.gap_free(s, a, b, 30).tolist() == [True, False, True]
    assert export.polls_in(s, np.array([OPEN_MS]), np.array([OPEN_MS + 3000])).tolist() == [2]


# --------------------------------------------------------------------------- the day panel


def test_build_day_targets_masks_and_rows(roots, monkeypatch):
    monkeypatch.setattr(export, "daily_context", fake_context(SYMS))
    write_feed(poll_feed(SYMS))
    t, info = export.build_day(DAY, step_s=15)
    df = t.to_pandas()
    assert info["symbols"] == len(SYMS) and set(df["symbol"]) == set(SYMS)
    assert df["t_s"].min() >= 0 and (df["t_s"] - info["phase_s"]).mod(15).eq(0).all()
    assert (df["tod_bucket"] == df["t_s"] // 900).all()
    assert df.groupby("symbol").size().nunique() == 1
    # every target is a log move from the last price, with its validity bit set
    for b, name in enumerate(export.TARGETS):
        valid = bit(df, b)
        assert df.loc[valid, name].notna().all()
        assert df.loc[~valid, name].isna().all()
    # windows reaching past the close are cut at it: the last row's 60-minute window is valid
    assert bit(df, 3, [df["t_s"].idxmax()]).all()
    row = df[(df["symbol"] == "S00")].iloc[100]
    assert row["y"] == pytest.approx(np.log(row["last"] / row["prev_vwap"]), abs=1e-6)
    assert row["r_close"] == pytest.approx(np.log(1.002) - row["y"], abs=1e-6)


def test_targets_past_the_coverage_end_are_masked(roots, monkeypatch):
    monkeypatch.setattr(export, "daily_context", fake_context(SYMS))
    write_feed(poll_feed(SYMS, until_ms=OPEN_MS + 3 * 3600 * 1000))  # stops at 12:30 ET
    df = export.build_day(DAY, step_s=15)[0].to_pandas()
    end_s = 3 * 3600
    late = df["t_s"] > end_s - 300
    assert not bit(df, 1, late).any()  # no full 5-minute window left
    assert not bit(df, 4).any()  # never covered to the close: no rest of day
    assert bit(df, 5).all()  # the day's VWAP comes from daily_bars


def test_features_never_see_the_future(roots, monkeypatch):
    """Changing every event after t* leaves every input at t ≤ t* unchanged, while the
    targets of windows reaching past t* change."""
    monkeypatch.setattr(export, "daily_context", fake_context(SYMS))
    cut = OPEN_MS + 2 * 3600 * 1000
    write_feed(poll_feed(SYMS))
    a = export.build_day(DAY, step_s=15)[0].to_pandas()
    write_feed(poll_feed(SYMS, perturb_after_ms=cut))
    b = export.build_day(DAY, step_s=15)[0].to_pandas()
    key = ["symbol", "t_s"]
    a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    cut_s = (cut - OPEN_MS) // 1000
    before = a["t_s"] <= cut_s
    inputs = [c for g in ("observed_reals", "known_reals", "static_reals") for c in export.FEATURES[g]]
    inputs = [c for c in inputs if c not in ("expected_vol_remaining",)]
    for col in inputs + ["last", "ref_session", "ref_trail_5m"]:
        np.testing.assert_allclose(a.loc[before, col], b.loc[before, col], rtol=1e-6, atol=1e-7, err_msg=col)
    near = before & (a["t_s"] > cut_s - 300)
    assert not np.allclose(a.loc[near, "r_5m"].fillna(0), b.loc[near, "r_5m"].fillna(0))


def test_group_context_excludes_self(roots, monkeypatch):
    monkeypatch.setattr(export, "daily_context", fake_context(SYMS))
    write_feed(poll_feed(SYMS))
    df = export.build_day(DAY, step_s=15)[0].to_pandas()
    at = df[df["t_s"] == df["t_s"].unique()[200]]
    t5 = at["L_trail_5"] - at["y"]
    for _, r in at.head(3).iterrows():
        others = t5[at["symbol"] != r["symbol"]]
        assert r["mkt_trail_5"] == pytest.approx(others.mean(), abs=1e-6)
        same = at[(at["sector_code"] == r["sector_code"]) & (at["symbol"] != r["symbol"])]
        assert r["sec_trail_5"] == pytest.approx((same["L_trail_5"] - same["y"]).mean(), abs=1e-6)


def test_basis_mismatch_drops_the_symbol(roots, monkeypatch):
    def ctx(day, syms):
        out = fake_context(SYMS)(day, syms)
        out["S00"]["close_1"] *= 2  # a split the feed's previous close doesn't show
        return out

    monkeypatch.setattr(export, "daily_context", ctx)
    write_feed(poll_feed(SYMS))
    t, info = export.build_day(DAY, step_s=15)
    assert "S00" not in set(t.column("symbol").to_pylist())
    assert info["dropped"]["basis_mismatch"] == 1 and info["basis_mismatch"] == ["S00"]


def test_released_today_turns_on_at_acceptance(roots, monkeypatch):
    accepted = datetime(2025, 12, 19, 11, 0)  # noqa: DTZ001 - naive New York time, as in the DB

    def ctx(day, syms):
        out = fake_context(SYMS)(day, syms)
        out["S03"]["released_at"] = accepted
        return out

    monkeypatch.setattr(export, "daily_context", ctx)
    write_feed(poll_feed(SYMS))
    df = export.build_day(DAY, step_s=15)[0].to_pandas()
    s3 = df[df["symbol"] == "S03"]
    acc_s = 90 * 60
    assert (s3.loc[s3["t_s"] < acc_s, "released_today"] == 0).all()
    assert (s3.loc[s3["t_s"] >= acc_s, "released_today"] == 1).all()
    assert (df.loc[df["symbol"] != "S03", "released_today"] == 0).all()


def test_typical_curve_uses_only_days_before(roots):
    export._write_profile(date(2025, 12, 18), np.full(101, 0.3), True)
    export._write_profile(date(2025, 12, 22), np.full(101, 0.9), True)
    curve, used = export.typical_for(DAY)
    assert used == ["2025-12-18"] and np.allclose(curve, 0.3)
    curve, used = export.typical_for(date(2025, 12, 17))
    assert used == [] and np.allclose(curve, export.default_typical())


def test_export_writes_panel_and_meta(roots, monkeypatch):
    monkeypatch.setattr(export, "daily_context", fake_context(SYMS))
    write_feed(poll_feed(SYMS))
    write_feed(poll_feed(SYMS, seed=3, shift_days=3), day=DAY + timedelta(days=3))  # 2025-12-22
    meta = export.export(step_s=15)
    assert meta["dates"] == ["2025-12-19", "2025-12-22"]
    on_disk = json.loads((export.panel_root() / "panel.meta.json").read_text())
    assert on_disk["tmask_bits"]["r_close"] == 5 and on_disk["features"] == export.FEATURES
    assert on_disk["days"]["2025-12-22"]["typical_from"] == ["2025-12-19"]
    part = pq.read_table(export.panel_root() / "date=2025-12-22" / "part.parquet")
    assert part.num_rows == on_disk["days"]["2025-12-22"]["rows"]
    assert not list(export.panel_root().rglob(".*tmp"))
