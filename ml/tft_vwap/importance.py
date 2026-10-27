"""Variable importance from the TFT's variable-selection weights.

The weights (interpret_output) say how much each input is used: per sample, encoder weights
sum to 1 over the encoder variables (averaged over the window), decoder and static likewise.
`variable_importance` averages them over (a subsample of) the test set; `by_group` adds them up
per input group so 'is macro used at all' reads off one number. Weights are usage, not effect
size: a variable can be used and still add nothing out of sample (that needs a drop-one run).
"""

from __future__ import annotations

import torch

from .config import (
    AFTER_HOURS_REALS,
    EARNINGS_COLUMNS,
    MACRO_REALS,
    MARKET_REALS,
    PM_FULL_REALS,
    PM_PX_REALS,
    SESSION_SHAPE_REALS,
)

GROUPS = {"earnings": EARNINGS_COLUMNS, "market": MARKET_REALS, "macro": MACRO_REALS,
          "session_shape": SESSION_SHAPE_REALS, "after_hours": AFTER_HOURS_REALS,
          "premarket_px": PM_PX_REALS, "premarket_full": PM_FULL_REALS}
_OF = {c: g for g, cols in GROUPS.items() for c in cols}


def group_of(name: str) -> str:
    return _OF.get(name, "base")


def variable_importance(model, dataloader, max_batches: int = 20) -> dict:
    """Mean selection weight per variable: {"encoder"|"decoder"|"static": {name: weight}}
    over at most `max_batches` batches of `dataloader` (a cheap subsample of the test set)."""
    model.eval()
    names = {"encoder": list(model.encoder_variables), "decoder": list(model.decoder_variables),
             "static": list(model.static_variables)}
    total = {k: torch.zeros(len(v)) for k, v in names.items()}
    n = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            x = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in x.items()}
            sums = model.interpret_output(model(x), reduction="sum")  # summed over the batch
            for k in total:
                total[k] += sums[f"{k}_variables"].detach().float().cpu()
            n += len(x["decoder_cont"])
    if n == 0:
        return {k: {} for k in names}
    return {k: {nm: float(w) / n for nm, w in zip(names[k], total[k], strict=True)}
            for k in names}


def by_group(importance: dict) -> dict:
    """Sum the weights per input group: {"encoder"|"decoder"|"static": {group: weight}}."""
    out = {}
    for kind, weights in importance.items():
        g: dict[str, float] = {}
        for name, w in weights.items():
            # the target-derived and calendar inputs are 'base'; categoricals too
            g[group_of(name)] = g.get(group_of(name), 0.0) + w
        out[kind] = g
    return out


def pool(per_fold: list[dict]) -> dict:
    """Mean over folds of each fold's by_group table."""
    out: dict[str, dict[str, float]] = {}
    for kind in ("encoder", "decoder", "static"):
        groups = {g for f in per_fold for g in f.get(kind, {})}
        out[kind] = {g: sum(f.get(kind, {}).get(g, 0.0) for f in per_fold) / len(per_fold)
                     for g in sorted(groups)}
    return out
