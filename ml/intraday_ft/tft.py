"""intraday-ft: a thin Temporal Fusion Transformer on pocket targets (docs/intraday-features.md §6).

Built from pytorch-forecasting's TFT blocks (variable selection, gated residual networks,
interpretable attention), not its dataset/trainer stack: one pass per stock-day over the whole
session, then the state at each decision minute feeds a 5-step decoder, one step per duration bin.
- static VSN over the static reals + embeddings (cap band, sector, industry, symbol) -> four contexts
  (variable selection, enrichment, LSTM h0 / c0);
- encoder VSN per minute -> LSTM -> gate -> static enrichment -> causal attention (minute i sees
  minutes 0 .. i only) -> gate -> position-wise GRN -> gate;
- decoder: the known bin inputs (bin minutes, normal volume share, minutes to close, cut flag) + a
  bin embedding, a GRN with the decision state as context -> P10 / P25 / P50 / P75 / P90 per bin
  target (quantiles kept in order by construction);
- row head on the decision state: the row-level quantiles plus the race and t_peak classes.
Loss = masked pinball per target (each divided by its training IQR) + 0.2 x cross-entropy per class
target. The overlay (sector / industry / symbol) is masked to "unknown" per stock-day in training
(p 0.3 / 0.3 / 0.5); the test rows are scored four ways: none, sec (+sector), ind (+industry),
all (+symbol).

    intraday-ft --folds B --label F1 --about "TFT, whole day, M+S tiers"
    intraday-ft --folds B --no-seconds          # ablation: M tier only
    intraday-ft --folds B --encoder 120         # ablation: the last 120 minutes only

Writes runs/intraday_ft/<stamp>-ft-<folds>/: config.json, MODEL.md, fold_<F>/{model.pt, prep.json,
history.json, intraday_ft_pred.parquet, quantiles.npy, quantiles.json}, then the report.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import (
    GateAddNorm,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
    VariableSelectionNetwork,
)
from torch import nn

import model_card

from . import data as D
from . import features as Fe
from . import targets as Tg
from . import windows as W

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
NQ = len(TAUS)
CLS_W = 0.2
OVERLAY_P = {"sector": 0.3, "industry": 0.3, "symbol": 0.5}
VARIANTS = {"none": (), "sec": ("sector",), "ind": ("sector", "industry"), "all": ("sector", "industry", "symbol")}
READOUTS = ("p_mass_up60", "p_bulge60", "p_sag60", "net60_p10", "net60_p50", "net60_p90", "t50_up_p50",
            "t50_dn_p50", "p_burst_up", "p_burst_dn", "p_race_up", "p_race_dn")
BURST = 10   # t50 <= 10 min = a burst
RUNS = Path("runs/intraday_ft")


# ---------------------------------------------------------------- model

class Net(nn.Module):
    def __init__(self, n_seq, n_static, cards, n_bin, n_row, hidden=32, heads=4, dropout=0.1, var_size=8):
        super().__init__()
        H = hidden
        self.n_bin, self.n_row = n_bin, n_row
        self.seq_keys = [f"s{j}" for j in range(n_seq)]
        self.stat_keys = [f"r{j}" for j in range(n_static)]
        self.cat_keys = [f"c{j}" for j in range(len(cards))]
        self.emb = nn.ModuleList(nn.Embedding(c, var_size) for c in cards)
        self.static_vsn = VariableSelectionNetwork(
            {**dict.fromkeys(self.stat_keys, var_size), **dict.fromkeys(self.cat_keys, var_size)}, H,
            input_embedding_flags=dict.fromkeys(self.cat_keys, True), dropout=dropout)
        self.ctx = nn.ModuleDict({k: GatedResidualNetwork(H, H, H, dropout) for k in ("var", "enrich", "h", "c")})
        self.seq_vsn = VariableSelectionNetwork(dict.fromkeys(self.seq_keys, var_size), H, dropout=dropout,
                                                context_size=H)
        self.lstm = nn.LSTM(H, H, batch_first=True)
        self.post_lstm = GateAddNorm(H, dropout=dropout)
        self.enrich = GatedResidualNetwork(H, H, H, dropout, context_size=H)
        self.attn = InterpretableMultiHeadAttention(heads, H, dropout=dropout)
        self.post_attn = GateAddNorm(H, dropout=dropout)
        self.pos_wise = GatedResidualNetwork(H, H, H, dropout)
        self.pre_out = GateAddNorm(H, dropout=None)
        self.dec_in = nn.Linear(len(W.KNOWN_DEC), H)
        self.bin_emb = nn.Embedding(Tg.NB, H)
        self.dec = GatedResidualNetwork(H, H, H, dropout, context_size=H)
        self.dec_gate = GateAddNorm(H, dropout=dropout)
        self.bin_out = nn.Linear(H, n_bin * NQ)
        self.row = GatedResidualNetwork(H, H, H, dropout)
        self.row_q = nn.Linear(H, n_row * NQ)
        self.cls = nn.ModuleDict({k: nn.Linear(H, n) for k, n in W.ROW_CLASS.items()})

    @staticmethod
    def _quantiles(z):
        """(..., NQ) raw -> ordered quantiles: the median plus cumulative positive steps either side."""
        mid = z[..., 2:3]
        up = torch.cumsum(nn.functional.softplus(z[..., 3:]), -1)
        dn = torch.flip(torch.cumsum(nn.functional.softplus(torch.flip(z[..., :2], [-1])), -1), [-1])
        return torch.cat([mid - dn, mid, mid + up], -1)

    def select(self, x, static, cat):
        """The position-wise part: static contexts and the selected input per minute (B, T, H)."""
        s = {k: static[:, j:j + 1] for j, k in enumerate(self.stat_keys)}
        s.update({k: self.emb[j](cat[:, j]) for j, k in enumerate(self.cat_keys)})
        se, _ = self.static_vsn(s)
        c = {k: g(se) for k, g in self.ctx.items()}
        xs = {k: x[..., j:j + 1] for j, k in enumerate(self.seq_keys)}
        v, _ = self.seq_vsn(xs, c["var"][:, None].expand(-1, x.shape[1], -1))
        return v, c

    def encode(self, x, static, cat):
        """x (B, T, n_seq) -> the state per minute (B, T, H)."""
        return self.sequence(*self.select(x, static, cat))

    def sequence(self, v, c, pad=None):
        """LSTM + causal attention over the selected inputs v (B, T, H); pad (B, T) True = padding."""
        T = v.shape[1]
        o, _ = self.lstm(v, (c["h"][None].contiguous(), c["c"][None].contiguous()))
        o = self.post_lstm(o, v)
        e = self.enrich(o, c["enrich"][:, None].expand(-1, T, -1))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=v.device), 1)[None]
        if pad is not None:
            mask = mask | pad[:, None, :]
        a, _ = self.attn(e, e, e, mask=mask)
        a = self.post_attn(a, e)
        return self.pre_out(self.pos_wise(a), o)

    def head(self, h, x_dec):
        """Decision states h (D, H) and known bin inputs (D, NB, 4) -> bin quantiles (D, NB, n_bin, NQ),
        row quantiles (D, n_row, NQ), class logits."""
        D_ = len(h)
        z = self.dec_in(x_dec) + self.bin_emb.weight[None]
        hc = h[:, None].expand(-1, Tg.NB, -1)
        z = self.dec_gate(self.dec(z, hc), hc)
        qb = self._quantiles(self.bin_out(z).view(D_, Tg.NB, self.n_bin, NQ))
        r = self.row(h)
        qr = self._quantiles(self.row_q(r).view(D_, self.n_row, NQ))
        return qb, qr, {k: m(r) for k, m in self.cls.items()}

    def forward(self, b, enc=None):
        if enc is None:
            Z = self.encode(b["x"], b["static"], b["cat"])
            h = Z[b["d_unit"], b["d_i"]]
        else:   # ablation: each decision re-reads only its last `enc` minutes
            # The selection is position-wise, so it runs once per day (plus one zero minute for the
            # padding) and only the LSTM + attention run per window: ~40x cheaper, same output.
            x0 = torch.cat([b["x"], b["x"].new_zeros(len(b["x"]), 1, b["x"].shape[2])], 1)
            v, c = self.select(x0, b["static"], b["cat"])
            t, pad = _window_index(b["d_i"], enc)
            u = b["d_unit"]
            h = self.sequence(v[u[:, None], t], {k: g[u] for k, g in c.items()}, pad)[:, -1]
        return self.head(h, b["x_dec"])


def _window_index(i, L, T=390):
    """(D, L) minutes of the windows ending at minute i, left-padded with T (the zero minute); pad mask."""
    t = i[:, None] - (L - 1) + torch.arange(L, device=i.device)[None]
    pad = t < 0
    return torch.where(pad, T, t), pad


# ---------------------------------------------------------------- loss

def pinball(q, y, m, scale):
    """Masked mean pinball loss per target, on targets divided by scale. q (..., n, NQ), y / m (..., n)."""
    tau = torch.tensor(TAUS, device=q.device)
    e = (y / scale)[..., None] - q / scale[..., None]
    loss = torch.maximum(tau * e, (tau - 1) * e).mean(-1)
    mf = m.float()
    dims = tuple(range(loss.ndim - 1))
    return (loss * mf).sum(dims) / mf.sum(dims).clamp(min=1)


def loss_fn(out, b, sc):
    qb, qr, cl = out
    lb = pinball(qb, b["y_bin"], b["m_bin"], sc["bin"])
    lr = pinball(qr, b["y_row"], b["m_row"], sc["row"])
    lc = sum(nn.functional.cross_entropy(cl[k], b["y_cls"][:, j], ignore_index=-1)
             if (b["y_cls"][:, j] >= 0).any() else cl[k].sum() * 0
             for j, k in enumerate(W.ROW_CLASS))
    return torch.cat([lb, lr]).mean() + CLS_W * lc


# ---------------------------------------------------------------- batches

def mask_overlay(cat, rng, keep=None):
    """Training: each overlay code -> 0 (unknown) with its OVERLAY_P. Scoring: keep = the overlays
    left on (a VARIANTS entry); the others -> 0. cap_band (column 0) is never masked."""
    cat = cat.copy()
    for j, name in enumerate(W.CATS[1:], 1):
        if keep is None:
            cat[rng.random(len(cat)) < OVERLAY_P[name], j] = 0
        elif name not in keep:
            cat[:, j] = 0
    return cat


def to_dev(b, dev, drop=()):
    out = {}
    for k, v in b.items():
        if k == "x" and drop:
            v = v.copy()
            v[..., list(drop)] = 0.0
        if k in ("d_rows",):
            continue
        out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
    return out


def device(name):
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


# ---------------------------------------------------------------- training

def train_fold(fold, a, run_dir, status):
    tr_m, va_m, te_m = W.split(fold)
    out = run_dir / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    sc = W.fit_scaler(tr_m, a.cache, seed=a.seed)
    voc = W.fit_vocab(tr_m, a.cache)
    W.save_prep(out / "prep.json", sc, voc)
    names = W.seq_names(sc)
    drop = [j for j, c in enumerate(names) if c in Fe.S_TIER] if a.no_seconds else []
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    train = W.Days(tr_m, sc, voc, a.cache, seed=a.seed)
    valid = W.Days(va_m, sc, voc, a.cache, every=a.valid_every, seed=a.seed)
    dev = device(a.accelerator)
    cards = [len(W.CAPS), len(voc["sector"]) + 1, len(voc["industry"]) + 1, len(voc["symbol"]) + 1]
    net = Net(len(names), len(W.STATIC_REALS), cards, len(W.BIN_REAL), len(W.ROW_REAL), a.hidden, a.heads,
              a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lsc = {"bin": torch.tensor(sc["y_bin_scale"], device=dev), "row": torch.tensor(sc["y_row_scale"], device=dev)}
    v_ids = np.sort(rng.choice(len(valid), min(a.valid_days, len(valid)), replace=False))
    best, bad, hist = np.inf, 0, []
    status.start(f"fold {fold} train", a.max_epochs * a.steps)
    for ep in range(a.max_epochs):
        net.train()
        t0, tl = time.time(), []
        for _ in range(a.steps):
            b = train.batch(rng.choice(len(train), a.batch_days, replace=False))
            b["cat"] = mask_overlay(b["cat"], rng)
            bt = to_dev(b, dev, drop)
            loss = loss_fn(net(bt, a.encoder), bt, lsc)
            if not torch.isfinite(loss):   # stop at once rather than train a whole epoch on NaN
                raise RuntimeError(f"fold {fold} epoch {ep + 1}: loss {loss.item()}")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
            status.step(f"epoch {ep + 1} loss {np.mean(tl[-50:]):.4f}")
        vl = evaluate(net, valid, v_ids, a, dev, lsc, drop)
        hist.append({"epoch": ep + 1, "train": float(np.mean(tl)), "valid": vl, "seconds": time.time() - t0})
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(net.state_dict(), out / "model.pt")
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    test = W.Days(te_m, sc, voc, a.cache, every=a.test_every, day_frac=a.test_frac, seed=a.seed, keep_frame=True)
    score(net, test, a, dev, drop, out, status, fold)
    return {"fold": fold, "train": tr_m, "valid": va_m, "test": te_m, "epochs": len(hist), "best_valid": best,
            "train_days": len(train), "test_decisions": test.n_decisions}


@torch.no_grad()
def evaluate(net, days, ids, a, dev, lsc, drop):
    net.eval()
    tot, n = 0.0, 0
    for s in range(0, len(ids), a.batch_days):
        bt = to_dev(days.batch(ids[s:s + a.batch_days]), dev, drop)
        tot += float(loss_fn(net(bt, a.encoder), bt, lsc)) * len(bt["d_i"])
        n += len(bt["d_i"])
    return tot / max(n, 1)


# ---------------------------------------------------------------- scoring

def cdf_at(q, v):
    """P(X <= v) from the 5 quantiles (..., NQ): linear between them, exponential tails outside, each
    with the decay that fits its outer pair (P10 -> P25, P75 -> P90). A flat 0.05 / 0.95 outside can't
    say less than 5% for a rare event: P(bulge60 > 0.5 sigma) is 1.7% (LEARNINGS "Intraday v2 IA1")."""
    q = np.asarray(q, np.float64)
    tau = np.asarray(TAUS)
    s_lo = np.maximum(q[..., 1] - q[..., 0], 1e-6) / np.log(tau[1] / tau[0])
    s_hi = np.maximum(q[..., -1] - q[..., -2], 1e-6) / np.log((1 - tau[-2]) / (1 - tau[-1]))
    with np.errstate(over="ignore"):
        lo_tail = tau[0] * np.exp(-np.maximum(q[..., 0] - v, 0) / s_lo)
        hi_tail = 1 - (1 - tau[-1]) * np.exp(-np.maximum(v - q[..., -1], 0) / s_hi)
    out = np.where(v < q[..., 0], lo_tail, np.where(v >= q[..., -1], hi_tail, np.nan))
    for k in range(NQ - 1):
        lo, hi = q[..., k], q[..., k + 1]
        inside = (v >= lo) & (v < hi)
        out = np.where(inside, tau[k] + (tau[k + 1] - tau[k]) * (v - lo) / np.maximum(hi - lo, 1e-12), out)
    return out


def readouts(qr, cl):
    """The run-file readouts from the row quantiles (D, n_row, NQ) and the class probabilities."""
    r = {t: qr[:, j] for j, t in enumerate(W.ROW_REAL)}
    return {"p_mass_up60": 1 - cdf_at(r["mass_up60"], 0.5), "p_bulge60": 1 - cdf_at(r["bulge60"], 0.5),
            "p_sag60": 1 - cdf_at(r["sag60"], 0.5), "net60_p10": r["net60"][:, 0], "net60_p50": r["net60"][:, 2],
            "net60_p90": r["net60"][:, 4], "t50_up_p50": r["t50_up"][:, 2], "t50_dn_p50": r["t50_dn"][:, 2],
            "p_burst_up": cdf_at(r["t50_up"], BURST), "p_burst_dn": cdf_at(r["t50_dn"], BURST),
            "p_race_up": cl["race"][:, 0], "p_race_dn": cl["race"][:, 1]}


@torch.no_grad()
def predict(net, b, a, dev, drop):
    qb, qr, cl = net(to_dev(b, dev, drop), a.encoder)
    return (qb.float().cpu().numpy(), qr.float().cpu().numpy(),
            {k: torch.softmax(v, -1).float().cpu().numpy() for k, v in cl.items()})


def quantile_names():
    return [f"{t}_b{k}" for k in range(1, Tg.NB + 1) for t in W.BIN_REAL] + list(W.ROW_REAL)


def score(net, days, a, dev, drop, out, status, fold):
    """Blind predictions on the test months, joined to the realised targets afterwards: the run file
    (keys, readouts x 4 overlays, real_*) and the full quantiles of the `all` variant (quantiles.npy)."""
    net.eval()
    rng = np.random.default_rng(a.seed)
    ids = np.arange(len(days))
    frames = []
    Q = np.lib.format.open_memmap(out / "quantiles.npy", "w+", np.float16,
                                  (days.n_decisions, len(quantile_names()), NQ))
    status.start(f"fold {fold} score", (len(ids) + a.batch_days - 1) // a.batch_days)
    for s in range(0, len(ids), a.batch_days):
        b = days.batch(ids[s:s + a.batch_days])
        cols = {"dec_row": b["d_rows"]}
        for v, keep in VARIANTS.items():
            bv = dict(b, cat=mask_overlay(b["cat"], rng, keep))
            qb, qr, cl = predict(net, bv, a, dev, drop)
            cols.update({f"{k}__{v}": np.asarray(x, np.float32) for k, x in readouts(qr, cl).items()})
            if v == "all":
                Q[b["d_rows"]] = np.concatenate([qb.reshape(len(qb), -1, NQ), qr], 1).astype(np.float16)
        frames.append(pd.concat([days.meta(b["d_rows"]), pd.DataFrame(cols)], axis=1))
        status.step()
    Q.flush()
    pd.concat(frames, ignore_index=True).sort_values("dec_row").to_parquet(
        out / "intraday_ft_pred.parquet", index=False, compression="zstd")
    (out / "quantiles.json").write_text(json.dumps({"taus": TAUS, "targets": quantile_names(),
                                                    "variant": "all", "row": "dec_row"}))


# ---------------------------------------------------------------- run

def run(a, card):
    folds = [f.strip() for f in a.folds.split(",")]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = ("-noS" if a.no_seconds else "") + (f"-enc{a.encoder}" if a.encoder else "")
    run_dir = Path(a.out) / f"{stamp}-ft-{''.join(folds)}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({"model": "intraday_ft", **{k: str(v) if isinstance(v, Path)
                                                     else v for k, v in vars(a).items()},
                                                     "folds_def": {f: W.FOLDS[f] for f in folds}}, indent=1))
    text = {"kind": "Intraday FT", "forecast": "intraday-pockets",
            "inputs": ("Thin TFT on pocket targets over 5 duration bins, whole session"
                       + (f" (encoder {a.encoder} min)" if a.encoder else "")
                       + ("; M tier only" if a.no_seconds else "; M + S tiers")
                       + f"; hidden {a.hidden}, {a.heads} heads; frames `{a.cache}`."),
            "folds": "; ".join(f"{f}: train {W.FOLDS[f]['train'][0]}..{W.FOLDS[f]['train'][1]} (last "
                               f"{W.VALID_MONTHS} months = early stop), test {W.FOLDS[f]['test'][0]}.."
                               f"{W.FOLDS[f]['test'][1]}" for f in folds) + "."}
    model_card.write(run_dir, **text, **card)
    steps = [s for f in folds for s in (f"fold {f} train", f"fold {f} score")]
    status = D.Status(steps, out=run_dir)
    res = [train_fold(f, a, run_dir, status) for f in folds]
    (run_dir / "summary.json").write_text(json.dumps(res, indent=1))
    from . import report
    rep = report.build(run_dir, a.cache)
    model_card.write(run_dir, **text, status="done", result=report.card_result(rep))
    print(f"results: {run_dir}")
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(prog="intraday-ft", description=__doc__.split("\n")[0])
    p.add_argument("--folds", default="B", help="comma list of B, A")
    p.add_argument("--cache", type=Path, default=D.CACHE, help="the intraday-ft-build output")
    p.add_argument("--out", type=Path, default=RUNS)
    p.add_argument("--no-seconds", action="store_true", help="ablation: zero the S-tier inputs")
    p.add_argument("--encoder", type=int, default=None, help="ablation: minutes each decision sees (default: "
                   "the whole day so far)")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-days", type=int, default=32, help="stock-days per batch (all their decisions)")
    p.add_argument("--steps", type=int, default=1000, help="batches per epoch")
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--valid-days", type=int, default=2000)
    p.add_argument("--valid-every", type=int, default=3, help="every n-th sampled decision (5 min) in validation")
    p.add_argument("--test-every", type=int, default=3, help="every n-th sampled decision in the test months")
    p.add_argument("--test-frac", type=float, default=0.25, help="share of the test stock-days scored")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--seed", type=int, default=0)
    model_card.add_args(p)
    a = p.parse_args(argv)
    torch.set_float32_matmul_precision("medium")
    run(a, model_card.from_args(a, "intraday-ft", argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
