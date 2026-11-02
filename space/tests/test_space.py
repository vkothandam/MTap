import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import sessions, synthetic_bars, utc_ms, write_bars

from mtap_space import bars as B
from mtap_space import library, match, pockets
from mtap_space.library import Library

PKG = Path(__file__).parents[1] / "mtap_space"


def test_et_minutes_follows_dst():
    # 09:30 New York is 13:30 UTC in summer (EDT) and 14:30 UTC in winter (EST)
    for day in ("2025-10-31", "2025-11-03", "2026-03-06", "2026-03-09", "2026-07-01"):
        t = utc_ms(pd.Timestamp(day), 570)
        assert B.et_minutes(np.array([t]))[0] % 1440 == 570, day
    summer = (pd.Timestamp("2026-07-01 13:30", tz="UTC").value // 1_000_000)
    winter = (pd.Timestamp("2026-01-05 14:30", tz="UTC").value // 1_000_000)
    assert (B.et_minutes(np.array([summer, winter])) % 1440 == 570).all()


def test_read_month_keeps_the_regular_session(tmp_path):
    raw = synthetic_bars(sessions("2025-10-01", "2025-10-31"), tickers=["AAA", "BBB"])
    write_bars(tmp_path, raw)
    daily, bars = B.read_month(tmp_path / "month=2025-10")
    assert bars["mi"].min() == 0 and bars["mi"].max() == 389
    assert daily.groupby("tk").day.nunique().eq(23).all()   # October 2025 has 23 sessions
    t = pd.to_datetime(raw.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
    mod = t.dt.hour * 60 + t.dt.minute
    reg = raw[(mod >= 570) & (mod < 960)]
    assert len(bars["v"]) == len(reg)          # pre-market and after-hours bars are dropped
    assert np.isclose(daily.dv.sum(), reg.v.sum(), rtol=1e-5)
    first = reg[reg.ticker == "AAA"].iloc[0]
    assert np.isclose(daily[daily.tk == "AAA"].open.iloc[0], first.o)


def test_month_folder_holds_its_month_only(tmp_path):
    # the file is named after its quarter, but only the folder's month is in it
    b = synthetic_bars(sessions("2025-10-27", "2025-11-07"), tickers=["AAA"])
    write_bars(tmp_path, b)
    months = [p.name for p in B.month_dirs(tmp_path)]
    assert months == ["month=2025-10", "month=2025-11"]
    d10, _ = B.read_month(tmp_path / "month=2025-10")
    d11, _ = B.read_month(tmp_path / "month=2025-11")
    assert len(d10) == 5 and len(d11) == 5   # the DST change (Nov 2) loses no session


def test_dense_carries_the_price_forward():
    bars = {"sd": np.array([0, 0, 0], np.int32), "mi": np.array([0, 2, 5], np.int16),
            "h": np.array([11, 12, 13], np.float32), "l": np.array([9, 10, 11], np.float32),
            "c": np.array([10, 11, 12], np.float32), "v": np.array([5, 6, 7], np.float32),
            "vw": np.array([10, np.nan, 12], np.float32)}
    A = B.dense([(bars, np.array([0]))], 1)
    assert A["V"][0, :6].tolist() == [5, 0, 6, 0, 0, 7]
    assert A["C"][0, :6].tolist() == [10, 10, 11, 11, 11, 12]
    assert A["VW"][0, 2] == 11   # a missing VWAP falls back to the close
    assert A["H"][0, 3] == 11    # an empty minute's high is the carried price


def test_prior_rolling_uses_only_earlier_days():
    x = np.arange(30, dtype=float)
    start = np.zeros(30, np.int64)
    m = library._prior(x, start, "mean")
    assert np.isnan(m[:15]).all()
    assert m[20] == np.mean(x[0:20]) and m[29] == np.mean(x[9:29])
    s = library._prior(x, start, "std")
    assert np.isclose(s[25], np.std(x[5:25], ddof=1))
    # a new ticker starts over
    start2 = np.r_[np.zeros(20, np.int64), np.full(10, 20)]
    assert np.isnan(library._prior(x, start2, "mean")[25])


def test_library_shape_and_order(lib):
    m = lib.m
    assert len(lib.sessions) == 25
    assert set(np.unique(m["p"])) <= set(library.P.tolist())
    order = np.lexsort((m["e"], m["p"]))
    assert (order == np.arange(len(order))).all()   # sorted by (p, e)
    assert lib.V.shape == (len(lib.sd), 390) and lib.V.dtype == np.float32
    for k in lib.params + library.READINGS:
        assert np.isfinite(m[k]).all(), k
    assert set(lib.spread) == set(library.PARAMS)


def test_parameters_match_a_hand_computation(lib):
    e = int(len(lib.sd) // 2)
    row = lib.moment(e, 90)
    sd = lib.sd.iloc[e]
    V, VW = np.asarray(lib.V[e], float), np.asarray(lib.VW[e], float)
    vws = (V[:90] * VW[:90]).sum() / V[:90].sum()
    assert np.isclose(lib.m["vwap_vs_prev"][row], np.log(vws / sd.prev) / sd.sig, atol=1e-4)
    assert np.isclose(lib.m["open_vs_prev"][row], np.log(sd.open / sd.prev) / sd.sig, atol=1e-4)
    rod = (V[90:] * VW[90:]).sum() / V[90:].sum()
    assert np.isclose(lib.m["rod"][row], np.log(rod / lib.m["now"][row]) / sd.sig, atol=1e-4)
    assert np.isclose(lib.m["day_1"][row], sd.d1 / sd.sig, atol=1e-4)


def test_no_look_ahead(tmp_path):
    """Changing the bars after the decision minute changes only the readings."""
    days = sessions("2025-09-15", "2025-11-26")
    a = synthetic_bars(days, tickers=["AAA", "BBB", "CCC"])
    b = a.copy()
    last = days[-1]
    t = pd.to_datetime(b.t, unit="ms", utc=True).dt.tz_convert("America/New_York")
    after = (t.dt.date == last.date()) & (t.dt.hour * 60 + t.dt.minute >= 570 + 120) \
        & (b.ticker == "AAA")
    b.loc[after, ["o", "h", "l", "c", "vw"]] *= 1.05
    b.loc[after, "v"] *= 3
    libs = []
    for name, bars in (("a", a), ("b", b)):
        built = library.build(write_bars(tmp_path / name, bars), 20, min_dollar=1e5,
                              log=lambda *_: None)
        libs.append(Library.from_memory(built))
    la, lb = libs
    e = la.stockday("AAA", str(last.date()))
    assert e is not None and e == lb.stockday("AAA", str(last.date()))
    for p in (60, 120):
        ra, rb = la.moment(e, p), lb.moment(e, p)
        for k in la.params:
            assert la.m[k][ra] == lb.m[k][rb], (p, k)
        assert la.m["rod"][ra] != lb.m["rod"][rb]
    # parameters later in the day do see the change
    assert la.m["vwap_vs_prev"][la.moment(e, 180)] != lb.m["vwap_vs_prev"][lb.moment(e, 180)]
    # no other stock-day moved
    same = la.m["e"] != e
    assert np.array_equal(la.m["rod"][same], lb.m["rod"][same])


def test_pool_is_prior_sessions_only(lib):
    row = lib.moment(int(np.flatnonzero(lib.sd.s == 20)[0]), 90)
    s0 = int(lib.m["s"][row])
    c = lib.pool(90, s0, 10)
    assert len(c) and (lib.m["s"][c] < s0).all() and (lib.m["s"][c] >= s0 - 10).all()
    assert (lib.m["p"][c] == 90).all()
    wide = lib.pool(90, s0, 10, tod=15)
    assert set(np.unique(lib.m["p"][wide])) == {75, 90, 105}
    tk = lib.tk[lib.m["e"][row]]
    own = lib.pool(90, s0, 10, tk=tk)
    assert (lib.tk[lib.m["e"][own]] == tk).all() and len(own) <= 10


def test_match_against_brute_force(lib):
    row = lib.moment(int(np.flatnonzero(lib.sd.s == 22)[0]), 120)
    q = match.Query(row=row, params=library.DEFAULT_PARAMS, window=20)
    M = match.run(lib, q, 1.5)
    c0 = match.pool(lib, q)
    tol = np.array([1.5 * lib.spread[k] for k in q.params])
    ref = []
    for r in c0:
        z = [abs(lib.m[k][r] - lib.m[k][row]) / t for k, t in zip(q.params, tol)]
        if max(z) < 1:
            ref.append((r, np.prod([1 - x * x for x in z])))
    assert M.n == len(ref) and M.n > 0
    assert np.array_equal(M.rows, [r for r, _ in ref])
    assert np.allclose(M.w, [w for _, w in ref])
    assert row not in M.rows
    # narrower is a subset
    N = match.run(lib, q, 0.5)
    assert set(N.rows) <= set(M.rows)
    # a wider tolerance on one parameter only adds matches
    T = match.run(lib, match.Query(row=row, params=q.params, window=20,
                                   tol={"day_1": 2.0}), 1.5)
    assert set(M.rows) <= set(T.rows)


def test_neff_days():
    s = np.array([0, 0, 0, 0, 1, 2])
    assert np.isclose(match.neff_days(s, np.ones(6)), 36 / 18)
    assert match.neff_days(np.array([5, 5, 5]), np.ones(3)) == 1.0
    assert match.neff_days(np.zeros(0, int), np.zeros(0)) == 0.0


def test_grid_mass_is_the_later_volume(lib):
    row = lib.moment(3, 60)
    e = lib.m["e"][row]
    G = pockets.grid(lib, np.array([row]), np.array([2.0]))
    V = np.asarray(lib.V[e], float)
    assert np.isclose(G.sum(), 2.0 * V[60:].sum() / lib.nv[e])
    two = pockets.grid(lib, np.array([row, row]), np.array([1.0, 1.0]))
    assert np.allclose(two, G)


def test_smooth_matches_a_direct_convolution():
    G = np.zeros((pockets.NX, pockets.NY))
    G[10, 12] = 1.0
    S = pockets.smooth(G, 1.0)
    assert np.isclose(S.sum(), 1.0)
    k = pockets._kernel(1.0)
    r = len(k) // 2
    assert np.isclose(S[10, 12], k[r] ** 2) and np.isclose(S[12, 11], k[r + 2] * k[r + 1])
    # mirrored edges keep the mass
    G[0, 0] = 1.0
    assert np.isclose(pockets.smooth(G, 1.0).sum(), 2.0)


def test_levels_hold_their_mass():
    rng = np.random.default_rng(3)
    Q = pockets.density(rng.gamma(0.5, 1, (pockets.NX, pockets.NY)))
    for a in pockets.LEVELS:
        held = Q[Q >= pockets.level(Q, a)].sum()
        assert a <= held < a + Q.max() + 1e-12
    r = pockets.region(Q)
    assert (r == 0).sum() < (r <= 1).sum() < (r <= 2).sum()


def test_two_bumps_are_two_blobs():
    G = np.zeros((pockets.NX, pockets.NY))
    G[5, 5] = G[30, 25] = 1.0
    Q = pockets.density(G, 1.0)
    assert pockets.blobs(Q >= pockets.level(Q, 0.5)) == 2
    G[5:31, 5] = 1.0
    G[30, 5:26] = 1.0
    Q = pockets.density(G, 1.0)
    assert pockets.blobs(Q >= pockets.level(Q, 0.95)) == 1


def test_readings_and_shrinkage(lib):
    row = lib.moment(int(np.flatnonzero(lib.sd.s == 24)[0]), 90)
    q = match.Query(row=row, params=["vwap_vs_prev", "move_since_open"], window=20)
    c0 = match.pool(lib, q)
    C = pockets.climatology(lib, c0)
    assert np.isclose(C.Q.sum(), 1) and 0 <= C.pu <= 1 and C.n == len(c0)
    assert C.rq[0] <= C.rq[1] <= C.rq[2]
    M = match.run(lib, q, 1.0, c0)
    S = pockets.matched(lib, M, C)
    a = M.shrink
    raw = pockets.read(lib, M.rows, M.w)
    assert np.isclose(S.pu, a * raw.pu + (1 - a) * C.pu)
    assert np.allclose(S.Q, a * raw.Q + (1 - a) * C.Q)
    none = pockets.matched(lib, match.Match(np.zeros(0, int), np.zeros(0), len(c0), 0.0, 0), C)
    assert none.pu == C.pu
    A = pockets.actual(lib, row)
    assert np.isclose(A.sum(), 1)
    cov = pockets.coverage(A, S.Q)
    assert 0 <= cov[0.5] <= cov[0.8] <= cov[0.95] <= 1
    assert pockets.log_score(A, S.Q) < 0


def test_weighted_sample_above_the_cap(lib):
    rows = lib.pool(90, 25, 25)
    full = pockets.read(lib, rows, np.ones(len(rows)), max_rows=10 ** 9)
    capped = pockets.read(lib, rows, np.ones(len(rows)), max_rows=len(rows) // 2)
    assert full.pu == capped.pu    # quantiles use every row
    assert np.abs(full.Q - capped.Q).sum() < 0.3


def test_cell_mass_adds_up_to_the_grid(lib):
    rows = lib.pool(120, 25, 5)[:40]
    w = np.linspace(0.2, 1, len(rows))
    G = pockets.grid(lib, rows, w)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    per = pockets.cell_mass(lib, rows, w, int(i), int(j))
    assert np.isclose(per.sum(), G[i, j])
    assert pockets.cell_of(0, 0) == (0, pockets.NY // 2)
    assert pockets.cell_of(10_000, -99) == (pockets.NX - 1, 0)


def test_split_is_each_moments_volume_above_now(lib):
    rows = lib.pool(120, 25, 5)[:30]
    up, tot = pockets.split(lib, rows)
    for r, u, t in zip(rows, up, tot):
        G = pockets.grid(lib, np.array([r]), np.ones(1))
        assert np.isclose(t, G.sum()) and np.isclose(u, G[:, pockets.NY // 2:].sum())


def test_save_and_load_round_trip(lib, bars_root, tmp_path):
    built = library.build(bars_root, sessions=25, min_dollar=1e6, log=lambda *_: None)
    path = library.save(built, tmp_path / "space")
    assert path.name == f"lib-{built.meta['sessions'][-1]}"
    assert library.find(tmp_path / "space") == [path]
    L = Library(path)
    assert isinstance(L.V, np.memmap)
    assert L.meta["n_moments"] == len(L.m["e"]) == len(lib.m["e"])
    for k in lib.m:
        assert np.array_equal(L.m[k], lib.m[k]), k
    tk = L.stocks().tk.iloc[0]
    days = L.days_of(tk)
    e = L.stockday(tk, days.date.iloc[0])
    assert e == days.e.iloc[0]
    assert L.times_of(e)
    assert L.label(L.moment(e, L.times_of(e)[0])).startswith(tk)
    # building again replaces it
    assert library.save(built, tmp_path / "space") == path


def test_build_until_and_errors(bars_root, tmp_path):
    built = library.build(bars_root, sessions=10, until="2025-10-31", min_dollar=1e6,
                          log=lambda *_: None)
    assert built.meta["sessions"][-1] == "2025-10-31" and len(built.meta["sessions"]) == 10
    with pytest.raises(FileNotFoundError):
        library.build(tmp_path, 10, log=lambda *_: None)


def test_no_banned_imports():
    banned = {"tft_vwap", "sourcing_py", "forecast_dash", "torch", "scipy", "sklearn"}
    for f in PKG.glob("*.py"):
        tree = ast.parse(f.read_text())
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        names |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not {n.split(".")[0] for n in names} & banned, f.name
