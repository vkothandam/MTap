from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import norm

import scripts_intraday_probe as P


def synth(n=24, seed=0):
    """Stock-days shaped like build_month's output: two days, three sectors, row 0 trades from minute 3."""
    rng = np.random.default_rng(seed)
    sig = rng.uniform(0.01, 0.03, n)
    r = rng.normal(0, 1, (n, 390)) * (sig[:, None] / np.sqrt(390)) * 3
    Cf = 50 * np.exp(np.cumsum(r, 1))
    H = Cf * np.exp(np.abs(rng.normal(0, 0.001, (n, 390))))
    L = Cf * np.exp(-np.abs(rng.normal(0, 0.001, (n, 390))))
    V = rng.integers(0, 2000, (n, 390)).astype(np.float64)
    VW = L + (H - L) * rng.uniform(0, 1, (n, 390))
    Cf[0, :3] = H[0, :3] = L[0, :3] = VW[0, :3] = np.nan
    V[0, :3] = 0
    prev = 50 * np.exp(rng.normal(0, 0.01, n))
    Q = prev[:, None] * np.exp(sig[:, None] * norm.ppf(P.TAU)[None, :])
    pm = {f"pm{c}_{k}": rng.uniform(49, 51, n) for c in (555, 570) for k in ("v", "pv", "last", "hi", "lo")}
    return SimpleNamespace(
        V=V, VW=VW, Cf=Cf, H=H, L=L, Q=Q, tk=np.array([f"S{i}" for i in range(n)], dtype=object),
        day=20000 + np.arange(n) % 2, open=Cf[:, 3] * 0.999, prev=prev, prevclose=prev * 1.001,
        prevhi=prev * 1.02, prevlo=prev * 0.98, sig=sig, nv=V.sum(1) * 0.9, ndol=V.sum(1) * 50,
        r1=rng.normal(0, 0.01, n), r2=rng.normal(0, 0.01, n), r3=rng.normal(0, 0.01, n),
        sec=np.arange(n) % 3, msig=np.full(n, 0.01), ssig=np.full(n, 0.012), mcap=np.full(n, 5e9),
        res_code=np.zeros(n), days_since=np.full(n, 20.0), days_to=np.full(n, 40.0), fc_prev=prev, pm=pm)


CURVE = np.concatenate([[0.0], np.arange(1, 391) / 390])


def perturb(M, c, seed=9):
    """A copy of M with every bar at column >= c replaced by different numbers."""
    rng = np.random.default_rng(seed)
    X = SimpleNamespace(**vars(M))
    for k, scale in (("Cf", 1), ("H", 1), ("L", 1), ("VW", 1), ("V", 0)):
        A = getattr(M, k).copy()
        A[:, c:] = A[:, c:] * rng.uniform(0.5, 1.5, A[:, c:].shape) if scale else rng.integers(0, 9000, A[:, c:].shape)
        setattr(X, k, A)
    return X


def same(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(np.nan_to_num(a), np.nan_to_num(b))


@pytest.mark.parametrize("c", [1, 7, 15, 61, 200, 389])
def test_dense_features_use_only_bars_before_their_column(c):
    M = synth()
    F, G = P.dense_features(M, CURVE), P.dense_features(perturb(M, c), CURVE)
    for k in F:
        assert same(F[k][:, :c + 1], G[k][:, :c + 1]), k
    assert not same(F["px_open"][:, c + 1:], G["px_open"][:, c + 1:])


@pytest.mark.parametrize("d", [0, 15])
def test_gather_ignores_bars_from_decision_minus_delay(d):
    M = synth()
    i = np.array([1, 5, 16, 30, 120, 389] * 4)
    r = np.arange(len(i))
    c = int(np.maximum(i - d, 1).min())
    X = perturb(M, c)
    a = P.gather(M, P.dense_features(M, CURVE), P.fixed_features(M), r, i, d)
    # every row's own cut is >= c, so bars from c on may only move rows whose cut is past c
    b = P.gather(X, P.dense_features(X, CURVE), P.fixed_features(X), r, i, d)
    low = np.maximum(i - d, 0) <= c
    for k in P.FULL:
        assert same(a[k].values[low], b[k].values[low]), k


def test_gather_per_row_cut():
    """Row-by-row: perturbing bars from that row's own cut never changes that row's inputs."""
    M = synth()
    for d in (0, 15):
        for ii in (20, 90, 300):
            r, i = np.arange(len(M.sig)), np.full(len(M.sig), ii)
            X = perturb(M, ii - d)
            a = P.gather(M, P.dense_features(M, CURVE), P.fixed_features(M), r, i, d)
            b = P.gather(X, P.dense_features(X, CURVE), P.fixed_features(X), r, i, d)
            for k in P.FULL:
                assert same(a[k].values, b[k].values), (d, ii, k)


def brute_targets(M, r, i):
    out = {}
    for j, (rr, ii) in enumerate(zip(r, i)):
        p, s = M.Cf[rr, ii - 1], M.sig[rr]
        for n, lev, kind in P.LEVELS:
            L = lev * s if kind == "s" else lev
            for h in P.REC_H:
                hi, lo = M.H[rr, ii:ii + h], M.L[rr, ii:ii + h]
                out.setdefault(f"tu_{n}_{h}", []).append(int(np.nanmax(hi) >= p * np.exp(L)))
                out.setdefault(f"td_{n}_{h}", []).append(int(np.nanmin(lo) <= p * np.exp(-L)))
        fu = fd = None
        for k in range(ii, 390):
            if fu is None and M.H[rr, k] >= p * np.exp(0.5 * s):
                fu = k
            if fd is None and M.L[rr, k] <= p * np.exp(-0.5 * s):
                fd = k
        fu, fd = 999 if fu is None else fu, 999 if fd is None else fd
        out.setdefault("race", []).append(0 if fu < fd else 1 if fd < fu else 2)
        v, vw = M.V[rr, ii:ii + 60], M.VW[rr, ii:ii + 60]
        out.setdefault("volbal", []).append((v * ((vw > p) + 0.5 * (vw == p))).sum() / v.sum())
    return {k: np.array(v) for k, v in out.items()}


def test_targets_match_brute_force():
    M = synth(n=10, seed=3)
    r = np.repeat(np.arange(1, 10), 5)
    i = np.tile([4, 40, 200, 350, 389], 9)
    Y, B = P.targets(M, P.future(M), r, i), brute_targets(M, r, i)
    for k in B:
        assert np.allclose(Y[k], B[k], equal_nan=True), k
    for h in P.HS:
        up = [np.log(np.nanmax(M.H[rr, ii:ii + h]) / M.Cf[rr, ii - 1]) / M.sig[rr] for rr, ii in zip(r, i)]
        assert np.allclose(Y[f"up_{h}"], up), h


def test_targets_ignore_bars_before_the_decision():
    M = synth(n=8, seed=4)
    r, i = np.arange(1, 8), np.array([10, 60, 61, 100, 200, 300, 389])
    Y = P.targets(M, P.future(M), r, i)
    X = SimpleNamespace(**vars(M))
    for k in ("H", "L", "V", "VW"):
        A = getattr(M, k).copy()
        for rr, ii in zip(r, i):
            A[rr, :ii] *= 1.3  # every bar up to and including minute i - 1
        setattr(X, k, A)
    C = M.Cf.copy()
    for rr, ii in zip(r, i):
        C[rr, :ii - 1] *= 1.3  # the price at i - 1 is p_t itself
    X.Cf = C
    Z = P.targets(X, P.future(X), r, i)
    for k in Y:
        assert np.allclose(Y[k], Z[k], equal_nan=True), k


def test_fwd_matches_loop():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3, 390))
    for h in (1, 5, 15, 60, 390):
        got = P.fwd(X, h, np.fmax)
        want = np.array([[X[r, j:j + h].max() for j in range(390)] for r in range(3)])
        assert np.allclose(got, want), h


def test_fire_once_per_window_then_again():
    p = np.array([[0.5, 0.7, 0.7, 0.7, 0.5, 0.7, 0.7, 0.8, 0.9, 0.7]])
    ok = np.ones_like(p, bool)
    assert np.flatnonzero(P.fire(p, ok, 3)[0]).tolist() == [1, 5, 8]
    assert np.flatnonzero(P.fire(p, ok, 390)[0]).tolist() == [1]
    ok[0, 1] = False
    assert np.flatnonzero(P.fire(p, ok, 390)[0]).tolist() == [2]


def test_on_list_hysteresis():
    rank = np.array([[30, 18, 25, 34, 36, 21, 19]])
    assert P.on_list(rank)[0].tolist() == [False, True, True, True, False, False, True]


def test_cap_band():
    caps = [3e12, 200e9, 199e9, 10e9, 2e9, 1.9e9, 250e6, 249e6, np.nan]
    assert P.cap_band(np.array(caps)).tolist() == ["mega", "mega", "large", "large", "mid", "small", "small",
                                                    "micro", ""]


def test_near_extreme_masks_are_point_in_time():
    M = synth()
    F = P.dense_features(M, CURVE)
    E = P.ext_masks(F)
    G = P.ext_masks(P.dense_features(perturb(M, 100), CURVE))
    for k in E:
        assert np.array_equal(E[k][:, :100], G[k][:, :100]), k  # column j of the mask = decision j + 1
    assert all(E[k].any() for k in E)
