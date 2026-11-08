"""intraday_ft: the doc examples (docs/intraday-features.md) and the leakage rules."""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from intraday_ft import build as B
from intraday_ft import data as D
from intraday_ft import features as Fe
from intraday_ft import profile as Pf
from intraday_ft import report as R
from intraday_ft import targets as Tg
from intraday_ft import tft as T
from intraday_ft import windows as W
from intraday_ft.zigzag import zigzag

DOC = Path(__file__).resolve().parents[2] / "docs" / "intraday-features.md"


def synth(n=12, seed=0):
    """Stock-days shaped like build_month's M: dense bars, the per-row reference values, forecasts."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.01, 0.03, n)
    r = rng.normal(0, 1, (n, 390)) * (sig[:, None] / np.sqrt(390)) * 3
    Cf = 50 * np.exp(np.cumsum(r, 1))
    H = Cf * np.exp(np.abs(rng.normal(0, 0.001, (n, 390))))
    L = Cf * np.exp(-np.abs(rng.normal(0, 0.001, (n, 390))))
    V = rng.integers(1, 2000, (n, 390)).astype(np.float64)
    VW = L + (H - L) * rng.uniform(0, 1, (n, 390))
    O = Cf * np.exp(rng.normal(0, 0.0005, (n, 390)))
    prev = 50 * np.exp(rng.normal(0, 0.01, n))
    from scipy.stats import norm
    Q = prev[:, None] * np.exp(sig[:, None] * norm.ppf(D.TAU)[None, :])
    pm = {f"pm{c}_{k}": rng.uniform(49, 51, n) for c in (555, 570) for k in ("v", "pv", "last", "hi", "lo")}
    nvm = np.full((n, 390), 900.0)
    return SimpleNamespace(
        V=V, N=V / 10, VW=VW, Cf=Cf, H=H, L=L, O=O, Q930=Q, Q945=Q * 1.001, fc_prev930=prev, fc_prev945=prev,
        sig=sig, prev=prev, prevclose=prev * 1.001, prevpost=prev * 1.002, open=O[:, 0], nvm=nvm, nnm=nvm / 10,
        typical=nvm.sum(1), sig1m=np.full((n, 390), 0.001), ndol=np.full(n, 5e7), mcap=rng.uniform(1e9, 1e11, n),
        beta60=np.ones(n), results=np.zeros(n), spy_gap=np.zeros(n), spx_y=np.zeros(n), spx_gap=np.zeros(n),
        pm=pm, npm={555: np.full(n, 1e4), 570: np.full(n, 2e4)}, pm570_last=pm["pm570_last"],
        live=np.arange(n) % 3 != 0, sector=np.array(["10", "20", "30"] * (n // 3), object),
        industry=np.array(["101010", "201010", "301010", "101020"] * (n // 4), object))


def perturb(M, c, seed=9):
    """A copy of M with every bar at column >= c replaced by different numbers."""
    rng = np.random.default_rng(seed)
    X = SimpleNamespace(**vars(M))
    for k in ("Cf", "H", "L", "VW", "O", "V", "N"):
        A = getattr(M, k).copy()
        A[:, c:] = A[:, c:] * rng.uniform(0.5, 1.5, A[:, c:].shape)
        setattr(X, k, A)
    return X


def same(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))


# ---------------------------------------------------------------- the feature table

def test_every_feature_is_in_the_doc():
    doc = DOC.read_text()
    names = {w for span in re.findall(r"`([^`]+)`", doc) for w in re.findall(r"[a-z0-9_]+", span)}
    stems = {re.sub(r"_(k\d|s|hl30|\d+|open|lo\d+|hi\d+)$", "", n) for n in names}
    missing = []
    for f in Fe.NAMES:
        base = re.sub(r"(_k\d)?(_s)?(_hl30)?$", "", f)
        base = re.sub(r"_(1|5|15|30|open)$", "", base)
        if f not in names and base not in names and base not in stems:
            missing.append(f)
    assert not missing, missing


def test_feature_names_are_unique_and_s_tier_matches_profile():
    assert len(Fe.NAMES) == len(set(Fe.NAMES))
    s = set(Fe.S_TIER) - {f"{c}_k{k}_s" for k in Fe.ZZ_K for c in ("zz_dir", "zz_len", "zz_age", "zz_pull",
                                                                    "zz_prev", "zz_legs")}
    assert s == set(Pf.S_COLS)


# ---------------------------------------------------------------- leakage: inputs at i use minutes <= i

@pytest.mark.parametrize("c", [1, 15, 61, 389])
def test_minute_inputs_use_only_bars_up_to_their_column(c):
    M = synth()
    X = perturb(M, c)
    for fn in (Fe.chunk0, Fe.chunk1_minute):
        a, b = fn(M), fn(X)
        for k in a:
            assert same(a[k][:, :c], b[k][:, :c]), k
    a = Fe.chunk2(M, M.Cf[0], M.V[0], M.nvm[0], 50.0, 50.0)
    b = Fe.chunk2(X, X.Cf[0], X.V[0], X.nvm[0], 50.0, 50.0)
    for k in a:
        assert same(a[k][:, :c], b[k][:, :c]), k


def test_second_inputs_use_only_seconds_before_the_minute_end():
    rng = np.random.default_rng(1)
    sec = np.sort(rng.choice(np.arange(Pf.OPEN_S, Pf.OPEN_S + Pf.SESSION), 4000, replace=False))
    price = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(sec))))
    vol = rng.integers(1, 500, len(sec)).astype(float)
    P, V = Pf.grid_day(sec, price, vol)
    a = B.second_columns(P, V, 100.0, 0.02, 5e5, np.full(390, 0.001))
    for m in (5, 60, 200):
        P2, V2 = P.copy(), V.copy()
        P2[60 * m:] *= 1.03
        V2[60 * m:] *= 3
        b = B.second_columns(P2, V2, 100.0, 0.02, 5e5, np.full(390, 0.001))
        for k in a:
            assert same(a[k][:m], b[k][:m]), (k, m)


def test_normals_and_sigma_use_prior_sessions_only():
    """Changing day d's bars moves no reference value of day d (only later days)."""
    n_days = 30
    rng = np.random.default_rng(2)
    S = pd.DataFrame({"tk": "A", "day": np.arange(n_days), "dv": rng.uniform(1e6, 2e6, n_days),
                      "dpv": 1e8, "n": 100.0, "open": 50.0, "close": 50.0, "hi": 51.0, "lo": 49.0,
                      "vwap": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))), "post": 50.0})
    A = {"V": rng.uniform(0, 1000, (n_days, 390)).astype(np.float32),
         "N": rng.uniform(0, 100, (n_days, 390)).astype(np.float32),
         "ss": rng.uniform(0, 1e-5, (n_days, 26)).astype(np.float32),
         "cnt": np.full((n_days, 26), 14, np.float32),
         "pmv555": rng.uniform(0, 1e4, n_days).astype(np.float32),
         "pmv570": rng.uniform(0, 1e4, n_days).astype(np.float32)}
    t = pd.DataFrame({"ticker": ["A"], "composite_figi": ["BBG1"], "cik": [None], "delisted": [pd.NaT],
                      "active": [True]})
    R0, X0 = B.reference(S, A, t)
    N0 = B.normals(R0, X0, np.arange(len(R0)))
    d = 25
    S2, A2 = S.copy(), {k: v.copy() for k, v in A.items()}
    S2.loc[d, ["dv", "vwap"]] = [9e6, 80.0]
    for v in A2.values():
        v[d] *= 7
    R1, X1 = B.reference(S2, A2, t)
    N1 = B.normals(R1, X1, np.arange(len(R1)))
    for c in ("sig", "typical", "ndol", "npm555", "npm570"):
        assert same(R0[c][:d + 1], R1[c][:d + 1]), c
    assert not same(R0.sig[d + 1:], R1.sig[d + 1:])  # day d does enter the later days
    for c in N0:
        assert same(N0[c][:d + 1], N1[c][:d + 1]), c
    assert np.isnan(R0.sig[:B.MIN_SESS + 1]).all()  # >= 15 prior changes needed


def test_reference_window_matches_full_history():
    """The build streams months with only WARM months of history before each one. Every reference
    value of the month's days must equal the one computed over the whole history (two stocks, 150 days,
    the last 20 = 'the month', 63 days of window before it)."""
    rng = np.random.default_rng(3)
    n, month, warm = 150, 20, 63
    S = pd.concat([pd.DataFrame({"tk": tk, "day": np.arange(n), "dv": rng.uniform(1e6, 2e6, n), "dpv": 1e8,
                                 "n": 100.0, "open": 50.0, "close": 50.0, "hi": 51.0, "lo": 49.0, "post": 50.0,
                                 "vwap": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))}) for tk in "AB"],
                  ignore_index=True)
    A = {"V": rng.uniform(0, 1000, (2 * n, 390)).astype(np.float32),
         "N": rng.uniform(0, 100, (2 * n, 390)).astype(np.float32),
         "ss": rng.uniform(0, 1e-5, (2 * n, 26)).astype(np.float32),
         "cnt": np.full((2 * n, 26), 14, np.float32),
         "pmv555": rng.uniform(0, 1e4, 2 * n).astype(np.float32),
         "pmv570": rng.uniform(0, 1e4, 2 * n).astype(np.float32)}
    t = pd.DataFrame({"ticker": ["A", "B"], "composite_figi": ["BBG1", "BBG2"], "cik": [None, None],
                      "delisted": [pd.NaT, pd.NaT], "active": [True, True]})
    spy = pd.DataFrame({"day": np.arange(n), "vw": 500 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))})

    def ref(keep):
        R, X = B.reference(S[keep].reset_index(drop=True), {k: v[keep] for k, v in A.items()}, t)
        R["beta60"] = B.beta60(R, spy)
        rows = np.flatnonzero(R.day.to_numpy() >= n - month)
        return R.iloc[rows].reset_index(drop=True), B.normals(R, X, rows)

    Rf, Nf = ref(np.ones(2 * n, bool))
    Rw, Nw = ref((S.day >= n - month - warm).to_numpy())
    assert (Rf[["tk", "day"]].to_numpy() == Rw[["tk", "day"]].to_numpy()).all()
    for c in ("prev", "prevclose", "sig", "ndol", "typical", "npm555", "npm570", "beta60"):
        assert np.allclose(Rf[c], Rw[c], equal_nan=True, rtol=1e-9), c
        assert np.isfinite(Rw[c]).all(), c
    for c in Nf:
        assert np.allclose(Nf[c], Nw[c], equal_nan=True), c


# ---------------------------------------------------------------- targets

def test_bulge_doc_example():
    vw = np.array([100.10, 100.30, 100.20, 99.90, 99.80])
    v = np.array([200e3, 300e3, 100e3, 250e3, 150e3])
    d = np.round(np.log(vw / 100) / 0.02, 2)  # the doc rounds d to 0.05 steps
    pk = Tg.pocket(d[None], v[None])
    assert pk["bulge"][0] == pytest.approx(0.065)
    assert pk["sag"][0] == pytest.approx(0.0275)
    assert pk["mass_up"][0] == pytest.approx(0.60)
    assert pk["cm_up"][0] == pytest.approx(0.108, abs=5e-4)
    assert pk["cm_dn"][0] == pytest.approx(0.069, abs=5e-4)
    assert pk["net"][0] == pytest.approx(0.0375)
    assert pk["mass_up"][0] + pk["mass_dn"][0] + pk["mass_at"][0] == pytest.approx(1)
    assert pk["bulge"][0] * (1000e3 / 800e3) / 5 == pytest.approx(0.01625)  # pi_up 0.016


def test_targets_use_only_bars_after_the_entry():
    M = synth()
    r = np.repeat(np.arange(len(M.sig)), 4)
    i = np.tile([4, 30, 200, 380], len(M.sig))
    Y = Tg.targets(M, r, i)
    for c in (5, 31, 201):
        X = SimpleNamespace(**vars(M))
        for k in ("Cf", "H", "L", "VW", "O", "V"):
            A = getattr(M, k).copy()
            A[:, :c] *= 1.2  # bars up to the decision move; the entry minute c stays
            setattr(X, k, A)
        Z = Tg.targets(X, r, i)
        rows = i + 1 >= c
        for k in Y:
            if k.startswith("occ"):
                continue
            assert same(np.asarray(Y[k])[rows], np.asarray(Z[k])[rows]), k


def test_mass_sums_to_one_and_bins_are_disjoint():
    M = synth()
    r, i = np.arange(len(M.sig)), np.full(len(M.sig), 100)
    for lo, hi in Tg.BINS:
        cols, valid, _ = Tg.bin_columns(i, lo, hi)
        assert cols[0, 0] == 101 + lo and valid.sum(1)[0] == min(hi, 390) - lo if hi < 390 else True
    Y = Tg.targets(M, r, i)
    assert Y["mins_b1"][0] == 5 and Y["mins_b2"][0] == 10 and Y["mins_b5"][0] == 390 - 161
    with np.errstate(invalid="ignore"):
        d = (np.log(M.VW[:, 101:106]) - np.log(Y["p0"])[:, None]) / M.sig[:, None]
    pk = Tg.pocket(d, M.V[:, 101:106])
    assert np.allclose(pk["mass_up"] + pk["mass_dn"] + pk["mass_at"], 1)
    assert np.allclose(pk["bulge"], Y["bulge_b1"])


def test_t50_matches_a_brute_force_scan():
    M = synth()
    r, i = np.arange(len(M.sig)), np.full(len(M.sig), 50)
    Y = Tg.targets(M, r, i)
    for j in r:
        d = (np.log(M.VW[j, 51:111]) - np.log(Y["p0"][j])) / M.sig[j]
        w = M.V[j, 51:111] * np.maximum(d, 0)
        tot, acc = w.sum(), 0.0
        if tot == 0:
            assert np.isnan(Y["t50_up"][j])
            continue
        for m, x in enumerate(w, 1):
            acc += x
            if acc >= 0.5 * tot:
                break
        assert Y["t50_up"][j] == m


def test_entry_is_the_next_open_else_the_held_price():
    M = synth()
    M.O[0, 11] = np.nan
    Y = Tg.targets(M, np.array([0, 1]), np.array([10, 10]))
    assert Y["p0"][0] == M.Cf[0, 10] and Y["p0"][1] == M.O[1, 11]


def test_race_codes():
    M = synth(n=4)
    i = np.full(4, 10)
    lp0 = np.log(M.O[:, 11])
    M.H[0, 15] = np.exp(lp0[0] + 0.6 * M.sig[0])
    M.L[1, 15] = np.exp(lp0[1] - 0.6 * M.sig[1])
    M.H[:2, 11:71] = np.minimum(M.H[:2, 11:71], np.exp(lp0[:2, None] + 0.4 * M.sig[:2, None]))
    M.H[0, 15] = np.exp(lp0[0] + 0.6 * M.sig[0])
    M.L[:2, 11:71] = np.maximum(M.L[:2, 11:71], np.exp(lp0[:2, None] - 0.4 * M.sig[:2, None]))
    M.L[1, 15] = np.exp(lp0[1] - 0.6 * M.sig[1])
    Y = Tg.targets(M, np.arange(4), i)
    assert Y["race"][0] == 0 and Y["race"][1] == 1


def test_occupancy_counts_the_seconds_above_and_below():
    P = np.full(23400, 100.0)
    P[(11 * 60):(11 * 60 + 190)] = 100.5  # 190 of b1's 300 s above
    o = Tg.occupancy(P, np.array([10]), np.array([100.0]))
    assert o["occ_up_b1"][0] == pytest.approx(190 / 300)
    assert o["occ_dn_b1"][0] == 0


def test_persistence_is_known_at_the_decision():
    M = synth()
    r, i = np.arange(len(M.sig)), np.full(len(M.sig), 40)
    a = Tg.persistence(M, r, i)
    b = Tg.persistence(perturb(M, 41), r, i)
    for k in a:
        assert same(a[k], b[k]), k


# ---------------------------------------------------------------- the second tier

def test_value_area_doc_table():
    counts = np.array([200, 400, 700, 900, 600, 450, 200, 150])  # bins -0.4 .. +0.3, now = 0.0 (index 4)
    lo, hi = Pf.value_area(counts, 4, 0.7)
    assert (lo, hi) == (2, 5)
    assert ((lo - 4 - 0.5) * Pf.BIN, (hi - 4 + 0.5) * Pf.BIN) == pytest.approx((-0.25, 0.15))
    lo2, hi2 = Pf.value_area_rows(counts[None], np.array([4]), 0.7)
    assert (lo2[0], hi2[0]) == (2, 5)


def test_value_area_matches_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(200):
        c = rng.integers(0, 50, 15).astype(float)
        if c.sum() == 0:
            continue
        center = int(rng.integers(0, 15))
        lo, hi = Pf.value_area(c, center, 0.7)
        lo2, hi2 = Pf.value_area_rows(c[None], np.array([center]), 0.7)
        assert (lo, hi) == (lo2[0], hi2[0])
        # the band reaches 70%, and the greedy rule never takes more bins than the shortest such band
        assert c[lo:hi + 1].sum() >= 0.7 * c.sum() - 1e-9
        best = min(b - a + 1 for a in range(15) for b in range(a, 15) if c[a:b + 1].sum() >= 0.7 * c.sum() - 1e-9)
        assert hi - lo + 1 >= best
        assert lo <= int(np.argmax(c)) <= hi


def test_profile_doc_example_time_above_and_dwell():
    """The Q-D table as one hour of held seconds: time_above 22%, dwell_now 17%, poc -0.1."""
    counts = [200, 400, 700, 900, 600, 450, 200, 150]
    centers = np.arange(-4, 4) * 0.1
    sig, prev = 0.02, 100.0
    P = np.full(Pf.SESSION, np.nan)
    pos = 0
    for c, x in sorted(zip(counts, centers), key=lambda z: abs(z[1] + 0.05)):  # any order; now last
        if x == 0:
            continue
        P[pos:pos + c] = prev * np.exp(x * sig)
        pos += c
    P[pos:pos + 600] = prev
    pos += 600
    V = np.where(np.isfinite(P), 1.0, 0.0)
    P[pos:] = np.nan
    out = Pf.day_columns(P, V, np.log(prev), sig, 1e6)
    m = pos // 60 - 1
    assert out["time_above"][m] == pytest.approx(800 / 3600, abs=1e-6)
    assert out["dwell_now"][m] == pytest.approx(600 / 3600, abs=1e-6)
    assert out["tpo_poc"][m] == pytest.approx(-0.1)
    assert out["tpo_lo70"][m] == pytest.approx(-0.25) and out["tpo_hi70"][m] == pytest.approx(0.15)


def test_tick_rule_doc_example():
    P = np.log(np.array([100.00, 100.02, 100.02, 100.02, 100.01]))
    V = np.array([2e3, 3e3, 5e3, 0, 1e3])
    dlt, side = Pf.tick_sides(P, V)
    up_vol = V[side > 0].sum()
    dn_vol = V[side < 0].sum()
    assert up_vol == 8e3 and dn_vol == 1e3
    assert np.maximum(dlt, 0).sum() == pytest.approx(np.log(100.02 / 100))
    assert np.maximum(-dlt, 0).sum() == pytest.approx(np.log(100.02 / 100.01))


def test_resistance_doc_example():
    """5-min window, typical 50M: +0.40% up on 300k -> 1.5; -0.20% down on 400k -> 4.0; bal +0.98."""
    n = 300
    dlt, side, V = np.zeros(n), np.zeros(n), np.zeros(n)
    dlt[:4], side[:150], V[:150] = 0.001, 1, 2e3
    dlt[150:152], side[150:], V[150:] = -0.001, -1, 400e3 / 150
    out = Pf.resistance(dlt, side, V, 50e6, ends=np.array([300]))
    assert out["res_up_5"][0] == pytest.approx(1.5)
    assert out["res_dn_5"][0] == pytest.approx(4.0)
    assert out["res_bal_5"][0] == pytest.approx(np.log(4.0 / 1.5))


def test_grid_day_held_price_and_two_second_polls():
    sec = Pf.OPEN_S + np.array([5, 9])
    P, _ = Pf.grid_day(sec, np.array([100.01, 100.03]), np.array([10.0, 20.0]))
    assert np.isnan(P[4]) and (P[5:9] == 100.01).all() and P[9] == 100.03
    P2, V2 = Pf.grid_day(sec, np.array([100.01, 100.03]), np.array([10.0, 20.0]), step=2)
    # polls at :01, :03, :05, ...: the trade at :05 is seen at :05, the one at :09 at :09
    assert np.isnan(P2[4]) and P2[5] == 100.01 and P2[8] == 100.01 and P2[9] == 100.03
    assert V2.sum() == 30 and V2[5] == 10 and V2[9] == 20
    assert np.isfinite(P2[5:]).sum() == np.isfinite(P[5:]).sum()  # the same held seconds


def test_visits_and_linger():
    near = np.zeros(400, bool)
    near[10:40] = near[100:150] = near[160:170] = True  # the gap 150-160 < 30 s: one visit
    n, d = Pf.visits(near)
    assert n == 2 and d == pytest.approx(90 / 2)


# ---------------------------------------------------------------- zig-zag

def test_zigzag_doc_example():
    """sigma_1m 0.10%, k = 2: up to $100.50, back to $100.35 (pull 0.75), up to $100.60, down to $100.38
    (pivot confirmed there)."""
    x = np.log(np.array([[100.0, 100.50, 100.35, 100.60, 100.38]]))
    z = zigzag(x, np.full(x.shape, 0.002), np.array([0.02]))
    assert z["zz_dir"][0].tolist() == [0, 1, 1, 1, -1]
    assert z["zz_len"][0, 1] == pytest.approx(np.log(1.005) / 0.02, abs=1e-4)
    assert z["zz_pull"][0, 2] == pytest.approx(np.log(100.50 / 100.35) / 0.002, abs=1e-3)
    assert z["zz_pull"][0, 3] == 0
    assert z["zz_legs"][0, 4] == 2
    assert z["zz_prev"][0, 4] == pytest.approx(np.log(1.006) / 0.02, abs=1e-4)


def test_zigzag_does_not_repaint():
    rng = np.random.default_rng(4)
    x = np.log(100) + np.cumsum(rng.normal(0, 1e-3, (5, 300)), 1)
    z = zigzag(x, np.full(x.shape, 2e-3), np.full(5, 0.02))
    for t in (10, 77, 250):
        y = x.copy()
        y[:, t:] += rng.normal(0, 5e-3, (5, 300 - t))
        w = zigzag(y, np.full(x.shape, 2e-3), np.full(5, 0.02))
        for k in z:
            assert same(z[k][:, :t], w[k][:, :t]), (k, t)


def test_zigzag_at_steps_equals_the_full_series():
    rng = np.random.default_rng(5)
    x = np.log(100) + np.cumsum(rng.normal(0, 1e-3, (3, 300)), 1)
    at = np.array([0, 9, 59, 150, 299])
    full = zigzag(x, np.full(x.shape, 2e-3), np.full(3, 0.02))
    sub = zigzag(x, np.full(x.shape, 2e-3), np.full(3, 0.02), at=at)
    for k in full:
        assert same(full[k][:, at], sub[k]), k


def test_seconds_rows_equal_one_row_at_a_time():
    rng = np.random.default_rng(6)
    rows, args = [], []
    for j in range(3):
        sec = np.sort(rng.choice(np.arange(Pf.OPEN_S + 200 * j, Pf.OPEN_S + Pf.SESSION), 3000, replace=False))
        price = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(sec))))
        rows.append(Pf.grid_day(sec, price, rng.integers(1, 500, len(sec)).astype(float)))
        args.append((100.0 + j, 0.02 + 0.005 * j, 5e5, np.full(390, 0.001 * (j + 1))))
    P, V = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])
    b = B.seconds_rows(P, V, *(np.array([a[i] for a in args]) for i in range(3)), np.stack([a[3] for a in args]))
    for j in range(3):
        one = B.second_columns(rows[j][0], rows[j][1], *args[j])
        for k in one:
            assert same(one[k], b[k][j]), (k, j)


# ---------------------------------------------------------------- chunk 2 and 3

def test_sector_leave_one_out_doc_example():
    """AAPL's peers MSFT $3.8T +0.6%, NVDA $4.5T +1.2%, ORCL $0.6T -0.8%: sec_mv +0.81%, ew +0.33%."""
    mv = np.array([[2.0], [0.6], [1.2], [-0.8]])  # AAPL first; its own move never enters
    cap = np.array([3.5, 3.8, 4.5, 0.6])
    cw, ew, n_oth = Fe.group_moves(mv, cap, np.zeros(4, int), cap_w=1.0)
    assert cw[0, 0] == pytest.approx(0.809, abs=1e-3)
    assert ew[0, 0] == pytest.approx(1 / 3, abs=1e-3)
    assert n_oth[0] == 3


def test_capped_weights():
    w = Fe.capped_weights(np.array([10.0, 1, 1, 1, 1, 1]), 0.25)
    assert w.max() == pytest.approx(0.25) and w.sum() == pytest.approx(1)


def test_spy_gap_adds_the_dividend_back():
    """2026-09-18: ex-dividend $1.89; a raw gap of -0.17% becomes +0.08%."""
    prevc = 1.89 / 0.0025
    o = prevc * np.exp(-0.0017)
    g = Fe.spy_gap(np.array([o]), np.array([prevc]), np.array([1.89]))
    assert g[0] == pytest.approx(0.08, abs=1e-3)


def test_chunk3_two_premarket_cuts():
    M = synth()
    out = Fe.chunk3(M)
    for k in ("pm_vwap555", "pm_vwap570", "pm_rvol555", "pm_rvol570", "y_close", "gap", "sigma"):
        assert k in out and out[k].shape == (len(M.sig),)


# ---------------------------------------------------------------- reference helpers

def test_stock_ids_carry_a_ticker_change_and_split_a_recycled_ticker():
    t = pd.DataFrame({"ticker": ["FI", "FISV", "ABC", "ABC"],
                      "composite_figi": ["F1", "F1", "A_OLD", "A_NEW"], "cik": [None] * 4,
                      "delisted": [pd.Timestamp("2025-06-01"), pd.NaT, pd.Timestamp("2022-01-01"), pd.NaT],
                      "active": [False, True, False, True]})
    day = D.day_of("2023-01-02")
    ids = D.stock_ids(np.array(["FI", "FISV", "ABC", "ABC", "ZZZ"]), np.array([day, day, day - 600, day, day]), t)
    assert ids.tolist() == ["F1", "F1", "A_OLD", "A_NEW", "tk:ZZZ"]


def test_cap_band_and_exclusion():
    assert D.cap_band(np.array([3e11, 5e10, 5e9, 5e8, 1e8, np.nan])).tolist() == \
        ["mega", "large", "mid", "small", "micro", "unknown"]
    assert D.exclusion("Acme 5.25% Notes due 2031") == "note"
    assert D.exclusion("Acme Inc") is None


def test_eligible_and_live_stand_in():
    S = pd.DataFrame({"day": [1] * 4, "ndol": [5e7, 3e7, 1e7, 9e7], "sig": [0.02] * 4, "prior": [5] * 4,
                      "open": [10.0] * 4, "prev": [10.0] * 4, "typical": [1e6] * 4, "common": [True] * 4})
    e, live = B.eligible(S)
    assert e.tolist() == [True, True, False, True]
    assert live.tolist() == [True, True, False, True]  # LIVE_N = 1000 > 3 eligible


def test_the_package_imports_nothing_else_from_the_repo():
    src = "\n".join(p.read_text() for p in (Path(Fe.__file__).parent).glob("*.py"))
    for mod in ("tft_vwap", "tft_intraday", "sourcing_py", "scripts_intraday_probe"):
        assert not re.search(rf"^\s*(from|import) {mod}\b", src, re.MULTILINE), mod


# ---------------------------------------------------------------- windows, the TFT and the report (smoke)

def fake_month(root, month, n=24, seed=0):
    """A month frame shaped like build_month's output (rows / seq / dec / meta) from synth()."""
    M = synth(n, seed)
    rng = np.random.default_rng(seed)
    d0 = D.day_of(f"{month}-03")
    k = pd.DataFrame({"tk": [f"S{j % 12}" for j in range(n)], "day": d0 + np.arange(n) // 12})
    seq_cols = [c for c in Fe.NAMES if c not in Fe.STATIC + Fe.CATEGORICAL or c == "poll_tier"]
    seq_cols = [c for c in seq_cols if c not in ("pm_vwap", "pm_rvol", "poll_tier")]
    seq = rng.normal(0, 1, (n, 390, len(seq_cols))).astype(np.float16)
    seq[..., seq_cols.index("normal_rvol")] = 1 / 390
    r_ = np.repeat(np.arange(n), 20)
    i_ = np.tile(B.FIRST_DEC + 5 * np.arange(20), n)
    Y = Tg.targets(M, r_, i_)
    dec = pd.DataFrame({"row": r_.astype(np.int32), "i": i_.astype(np.int16)})
    for c, v in Y.items():
        dec[c] = np.asarray(v, np.float32)
    for c, v in Tg.persistence(M, r_, i_).items():
        if c in ("bulge", "sag", "mass_up", "cm_up", "cm_dn", "net"):
            dec[f"pers_{c}"] = np.asarray(v, np.float32)
    for s in ("up", "dn"):
        for b in range(1, Tg.NB + 1):
            dec[f"occ_{s}_b{b}"] = rng.uniform(0, 1, len(dec)).astype(np.float32)
    rows = k.assign(sid=[f"F{j % 12}" for j in range(n)], sector=M.sector, industry=M.industry, live=M.live,
                    cap_band="large", delisted_later=False, prior=100, poll_tier=1)
    for c in W.STATIC_ALL:
        if c != "poll_tier":
            rows[c] = rng.normal(0, 1, n).astype(np.float32)
    rows["sig"], rows["prev"], rows["log_dvol"] = M.sig, M.prev, rng.normal(17, 1, n)
    B.write_month(month, rows, seq, dec, {"seq_cols": seq_cols}, Path(root))


@pytest.fixture
def frames(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "FOLDS", {"B": {"train": ("2023-01", "2023-03"), "test": ("2023-04", "2023-04")}})
    monkeypatch.setattr(W, "VALID_MONTHS", 1)
    for j, m in enumerate(D.months("2023-01", "2023-04")):
        fake_month(tmp_path, m, seed=j)
    return tmp_path


def test_decoder_known_bins():
    share = np.full((2, 390), 1 / 390)
    k = W.decoder_known(np.array([10, 370]), share)
    assert np.allclose(k[0, :, 0], [5 / 60, 10 / 60, 15 / 60, 30 / 60, (390 - 71) / 60])
    assert np.isclose(k[0, 0, 1], 5 / 390) and np.isclose(k[0, 0, 2], (390 - 11) / 390)
    assert list(k[1, :, 3]) == [0, 0, 1, 1, 0]          # 370: b3 (386-400) and b4 run past the close
    assert np.isclose(k[1, 2, 0], 4 / 60) and k[1, 3, 0] == 0   # b3 keeps 386-389; b4 is empty


def test_days_batch_shapes_and_premarket_cut(frames):
    sc = W.fit_scaler(["2023-01", "2023-02"], frames)
    voc = W.fit_vocab(["2023-01", "2023-02"], frames)
    days = W.Days(["2023-01"], sc, voc, frames, keep_frame=True)
    assert len(days) == 24 and days.n_decisions == 480
    b = days.batch([0, 5])
    assert b["x"].shape == (2, 390, len(W.seq_names(sc))) and b["x_dec"].shape == (40, Tg.NB, 4)
    assert b["y_bin"].shape == (40, Tg.NB, len(W.BIN_REAL)) and b["y_row"].shape == (40, len(W.ROW_REAL))
    assert (b["d_unit"] == np.repeat([0, 1], 20)).all() and np.isfinite(b["x"]).all()
    n = len(sc["seq_cols"])
    assert (b["x"][0, :15, n] == days.pm[0][0, 0, 0]).all() and (b["x"][0, 15:, n] == days.pm[0][0, 1, 0]).all()
    f = days.meta(b["d_rows"])
    assert len(f) == 40 and {"date", "minute", "symbol", "real_net60", "real_pers_net"} <= set(f.columns)
    thin = W.Days(["2023-01"], sc, voc, frames, every=3)
    assert thin.n_decisions == 24 * 7                    # 20 decisions 5 min apart -> every 15 min


def test_undefined_targets_are_masked_and_the_loss_stays_finite(frames):
    # The real frames hold pi_* = inf where a bin's normal volume is 0 (and float16 overflows): the first
    # S3 timing check had loss NaN from step 1 because masked targets became float max, not 0.
    import torch

    from intraday_ft import tft as T
    sc = W.fit_scaler(["2023-01", "2023-02"], frames)
    voc = W.fit_vocab(["2023-01", "2023-02"], frames)
    days = W.Days(["2023-01"], sc, voc, frames)
    days.y_bin[:5, 0, W.BIN_REAL.index("pi_up")] = np.inf
    days.y_row[:5, 0] = -np.inf
    b = days.batch([0, 1])
    assert not b["m_bin"][:5, 0, W.BIN_REAL.index("pi_up")].any() and (b["y_bin"][~b["m_bin"]] == 0).all()
    assert (b["y_row"][~b["m_row"]] == 0).all()
    bt = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items() if k != "d_rows"}
    cards = [len(W.CAPS), len(voc["sector"]) + 1, len(voc["industry"]) + 1, len(voc["symbol"]) + 1]
    net = T.Net(len(W.seq_names(sc)), len(W.STATIC_REALS), cards, len(W.BIN_REAL), len(W.ROW_REAL), 8, 2)
    lsc = {"bin": torch.tensor(sc["y_bin_scale"]), "row": torch.tensor(sc["y_row_scale"])}
    assert torch.isfinite(T.loss_fn(net(bt), bt, lsc))


def test_pi_is_undefined_without_normal_volume():
    M = synth()
    M.nvm = np.zeros_like(M.nvm)
    Y = Tg.targets(M, np.arange(3), np.full(3, 30))
    assert np.isnan(Y["pi_up_b1"]).all() and np.isnan(Y["pi_dn_b1"]).all()
    assert np.isfinite(Y["rvol_b1"]).all()


def test_tft_encoder_is_causal_and_quantiles_ordered():
    import torch
    net = T.Net(6, 3, [6, 4, 5, 7], n_bin=2, n_row=3, hidden=8, heads=2, dropout=0.0).eval()
    x = torch.randn(2, 390, 6)
    st, cat = torch.randn(2, 3), torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]])
    with torch.no_grad():
        a = net.encode(x, st, cat)
        x2 = x.clone()
        x2[:, 201:] = torch.randn(2, 189, 6) * 5
        b = net.encode(x2, st, cat)
        assert torch.allclose(a[:, :201], b[:, :201], atol=1e-5) and not torch.allclose(a[:, 201:], b[:, 201:])
        qb, qr, cl = net.head(a[0, [10, 200]], torch.rand(2, Tg.NB, 4))
    assert qb.shape == (2, Tg.NB, 2, T.NQ) and qr.shape == (2, 3, T.NQ) and cl["race"].shape == (2, 3)
    assert (qb.diff(dim=-1) >= 0).all() and (qr.diff(dim=-1) >= 0).all()


def test_encoder_window_mode_equals_explicit_windows():
    # The ablation runs the selection once per day; it must equal encoding each zero-padded window.
    import torch
    net = T.Net(6, 3, [6, 4, 5, 7], n_bin=2, n_row=3, hidden=8, heads=2, dropout=0.0).train(False)
    b = {"x": torch.randn(2, 390, 6), "static": torch.randn(2, 3), "cat": torch.tensor([[1, 2, 3, 4], [0, 1, 0, 2]]),
         "d_unit": torch.tensor([0, 0, 1, 1]), "d_i": torch.tensor([3, 200, 59, 389]),
         "x_dec": torch.rand(4, Tg.NB, 4)}
    L = 60
    with torch.no_grad():
        qb, qr, _ = net(b, L)
        hs = []
        for u, i in zip(b["d_unit"].tolist(), b["d_i"].tolist()):
            xw = torch.zeros(1, L, 6)
            n = min(L, i + 1)
            xw[0, L - n:] = b["x"][u, i + 1 - n:i + 1]
            pad = torch.arange(L)[None] < L - n
            hs.append(net.sequence(*net.select(xw, b["static"][u:u + 1], b["cat"][u:u + 1]), pad)[:, -1])
        qb2, qr2, _ = net.head(torch.cat(hs), b["x_dec"])
    assert torch.allclose(qb, qb2, atol=1e-5) and torch.allclose(qr, qr2, atol=1e-5)


def test_cdf_at_and_overlay_masks():
    q = np.array([[-2.0, -1, 0, 1, 2]])
    assert np.allclose(T.cdf_at(q, 0.0), 0.5) and np.allclose(T.cdf_at(q, 0.5), 0.625)
    # exponential tails: continuous at P10 / P90, monotone, below 5% far out (rare events)
    assert np.allclose(T.cdf_at(q, 2.0), 0.9) and np.allclose(T.cdf_at(q, -2.0), 0.1)
    v = np.linspace(-9, 9, 181)
    c = T.cdf_at(np.repeat(q, len(v), 0), v)
    assert (np.diff(c) >= 0).all() and 0 < c[0] < 0.001 and 0.999 < c[-1] < 1
    assert np.allclose(T.cdf_at(q, 3.0), 1 - 0.1 * np.exp(-np.log(2.5)))
    cat = np.array([[3, 1, 2, 5]] * 4)
    assert (T.mask_overlay(cat, None, ())[:, 1:] == 0).all() and (T.mask_overlay(cat, None, ())[:, 0] == 3).all()
    assert (T.mask_overlay(cat, None, ("sector",))[0] == [3, 1, 0, 0]).all()
    assert (T.mask_overlay(cat, None, T.VARIANTS["all"]) == cat).all()


def test_report_skill_and_rank_ic():
    day = np.repeat(np.arange(10), 5)
    s = R.boot_ratio(day, np.full(50, 0.75), np.full(50, 1.0))
    assert np.isclose(s["skill"], 0.25) and np.isclose(s["lo"], 0.25) and s["days"] == 10
    f = pd.DataFrame({"date": np.repeat(["d1", "d2"], 12), "minute": 30})
    y = np.tile(np.arange(12.0), 2)
    assert np.isclose(R.rank_ic(f, y, y)["mean"], 1.0) and np.isclose(R.rank_ic(f, -y, y)["mean"], -1.0)
    assert np.allclose(R.pinball(np.zeros((1, 5)), np.array([1.0])), np.mean(R.TAUS))


def test_tft_train_score_report_end_to_end(frames):
    # Real frames have undefined realised targets (bins past the close, no trade): the report must
    # cope (event rates as float, not object; the top-20 ranking on the rows it kept).
    for m in ("2023-02", "2023-04"):
        p = frames / "frames" / f"month={m}" / "dec.parquet"
        d = pd.read_parquet(p)
        d.loc[d.index[::7], ["mass_up60", "bulge60", "sag60", "net60"]] = np.nan
        d.to_parquet(p, index=False)
    run = T.main(["--folds", "B", "--cache", str(frames), "--out", str(frames / "runs"), "--accelerator", "cpu",
                  "--hidden", "8", "--heads", "2", "--batch-days", "4", "--steps", "2", "--max-epochs", "1",
                  "--valid-days", "8", "--test-frac", "1.0", "--label", "smoke"])
    assert run == 0
    rd = next((frames / "runs").iterdir())
    fd = rd / "fold_B"
    pred = pd.read_parquet(fd / "intraday_ft_pred.parquet")
    assert {f"{r}__{v}" for r in T.READOUTS for v in T.VARIANTS} <= set(pred.columns)
    assert len(pred) == 24 * 7 and pred.p_mass_up60__all.between(0, 1).all()
    assert np.load(fd / "quantiles.npy").shape == (len(pred), len(T.quantile_names()), T.NQ)
    rep = json.loads((rd / "report.json").read_text())
    assert rep["folds"][0]["rows"] == len(pred) and "net60__pers" in rep["folds"][0]["rank_ic"]
    shape = [p for p in rep["folds"][0]["pinball"] if "hits" in p]
    assert {p["target"] for p in shape} >= {f"{b}_b1" for b in R.SHAPE}
    assert all(len(p["hits"]) == len(R.TAUS) and all(0 <= a <= b <= 1 for a, b in p["hits"]) for p in shape)
    assert all(0 <= p["cover"][0] <= p["cover"][1] <= 1 for p in shape)
    assert "race_auc_side" in rep["folds"][0] and "Pocket shape" in (rd / "report.md").read_text()
    card = (rd / "MODEL.md").read_text()
    assert "label: smoke" in card and "status: done" in card and "rank IC" in card


def test_span_formats_tie_ranges():
    assert R._span([0.10, 0.104]) == "0.10"
    assert R._span([0.0, 0.31]) == "0.00-0.31"
    assert R._span([0.80, 0.913], 100) == "80-91"
