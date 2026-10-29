"""MODEL.md text for the daily runs (tft-vwap, -gbm, -overlay, -event-overlay): see model_card."""

from __future__ import annotations

from .config import TrainConfig

FORECAST = {"night": "night", "px": "09:30", "full": "09:45"}


def state_of(cfg: TrainConfig) -> str:
    """The scoring state of a run: what is known when its forecast is made."""
    return "night" if cfg.premarket == "off" else cfg.pm_group_names[-1]


def inputs_text(cfg: TrainConfig, model: str) -> str:
    on = [name for flag, name in ((cfg.market, "market/sector"), (cfg.macro, "macro"),
                                  (cfg.session_shape, "session shape"),
                                  (cfg.after_hours, "after-hours")) if flag]
    if cfg.premarket != "off":
        on.append(f"pre-market {cfg.pm_groups} ({cfg.premarket}"
                  + (f", masks full {cfg.pm_mask_full:g} px {cfg.pm_mask_px:g})"
                     if cfg.premarket == "optional" else ")"))
    return (f"{model}, target {cfg.target}, the stock's history + earnings {cfg.earnings}"
            + (" + " + ", ".join(on) if on else ", no added groups")
            + f". Panel `{cfg.panel}`" + (f", {cfg.symbols} random symbols" if cfg.symbols else "")
            + ".")


def folds_text(cfg: TrainConfig, n_folds: int) -> str:
    sel = [k for k in range(n_folds) if cfg.only_folds is None or k in cfg.only_folds]
    return (f"{len(sel)} of {n_folds} walk-forward folds ({', '.join(map(str, sel))}), "
            f"{cfg.fold_len} test sessions each.")
