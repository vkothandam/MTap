"""Training configuration: one dataclass, serialised next to every run's results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

SUPPORTED_SCHEMA_VERSIONS = (3,)

# The model inputs by category; FEATURES.md describes every column.
STATIC_CATEGORICALS = ("symbol", "industry_code", "exchange", "issue_type")
KNOWN_REALS = ("day_of_week", "month", "is_month_end", "is_quarter_end", "sessions_gap_next")
PRICE_VOLUME_REALS = (
    "vwap_pct_prev_day",
    "intraday_spread_pct",
    "close_position_pct",
    "open_position_pct",
    "volume_velocity",
    "transactions_per_volume",
)
FILING_REALS = ("is_10k", "is_10q")
OBSERVED_REALS = PRICE_VOLUME_REALS + FILING_REALS
# Financial-results calendar, dated from the results releases (TrainConfig.earnings):
#   "exact": the pre windows count to the next scheduled release's actual date (announced
#            ahead), falling back to the estimate where it isn't in the panel yet
#   "pit":   strictly point in time, the next release estimated from the year-ago one
# The flags are one-hot from the panel's results_window (exact) / results_window_pit (pit).
EARNINGS_WINDOWS = {
    "Pre_Earnings_Runup": "is_pre_earnings_runup",
    "Earnings_Eve": "is_earnings_eve",
    "Earnings_Day": "is_earnings_day",
    "Post_Earnings_Reaction": "is_post_earnings_reaction",
    "Post_Earnings_Drift": "is_post_earnings_drift",
}
EARNINGS_SOURCES = {"exact": "results_window", "pit": "results_window_pit"}
EARNINGS_REALS = {
    "exact": ("days_since_results", "days_to_results", "days_to_results_is_est",
              *EARNINGS_WINDOWS.values()),
    "pit": ("days_since_results", "days_to_results_est",
            *(f"{c}_pit" for c in EARNINGS_WINDOWS.values())),
}
EARNINGS_MODES = ("exact", "pit", "off")
# the sessions the day counts are clipped to; also their value when unknown (no release yet)
EARNINGS_FAR = 63
EARNINGS_OVERDUE = -10
SENTIMENT_REALS = ("daily_sentiment", "sentiment_volume", "has_news")
# Series breaks (TrainConfig.series_breaks): more than BREAK_GAP sessions without a bar (a
# reused ticker, a relisting) or a one-session VWAP move of BREAK_JUMP x either way (a missed
# reverse split, a bankruptcy share exchange) start a new security under the same ticker
BREAK_GAP = 20
BREAK_JUMP = 10.0

# The model forecasts a series one session ahead. The panel's `target_tomorrow_vwap` lives on
# day t's row, so using it as the series would put tomorrow's price into the encoder; instead
# the series is today's value and the decoder step is t+1, i.e. exactly next-day VWAP.
TARGETS = {
    "vwap": "today_vwap",  # raw price, per-symbol log GroupNormalizer
    "return": "vwap_pct_prev_day",  # prev-session return, per-symbol standard GroupNormalizer
}


@dataclass
class TrainConfig:
    panel: Path = Path("../out/tft/panel.parquet")
    out_dir: Path = Path("runs")
    target: str = "return"
    # "encoder": scale each sample by its own encoder window (tracks trending prices);
    # "group": one center/scale per symbol over the whole training span
    normalizer: str = "encoder"
    use_sentiment: bool = False
    earnings: str = "exact"  # EARNINGS_MODES
    # each series break starts a new series: the part before it is renamed SYMBOL~1 (~2 ...)
    series_breaks: bool = True
    symbols: int | None = None  # random subset of N symbols for fast iteration
    seed: int = 42

    # walk-forward: `n_folds` test blocks of `fold_len` sessions ending at the last session
    # (or at `test_end`, a day_idx, to repeat an earlier run's folds on a longer panel); the
    # `early_stop_len` sessions before each block drive early stopping (never the test).
    n_folds: int = 4
    fold_len: int = 60
    early_stop_len: int = 20
    only_folds: list[int] | None = None  # 0-based subset of folds to run
    test_end: int | None = None

    encoder_length: int = 60
    min_encoder_length: int = 20

    hidden_size: int = 32
    attention_head_size: int = 2
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    gradient_clip_val: float = 0.1
    # P5..P95 in 5-point steps: the predictive distribution that probabilities are read from
    quantiles: list[float] = field(default_factory=lambda: [round(k * 0.05, 2) for k in range(1, 20)])
    # calibration: P(next-day vwap move >= m) for upside moves m > 0 (<= m for downside)
    moves: list[float] = field(default_factory=lambda: [
        round(k * 0.005, 3) for k in (-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)])
    # call levels: screener thresholds and the per-stock ladder (move at each confidence)
    confidences: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.8, 0.9])
    # forecast-probability buckets of the calibration grid (top edge inclusive). The CDF is
    # clamped to the outermost quantiles, so probabilities lie in [0.05, 0.95]; low buckets
    # grade the "unlikely" calls too, so every forecast has a historical comparison.
    prob_buckets: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    embedding_sizes: dict[str, int] = field(
        default_factory=lambda: {"symbol": 16, "industry_code": 4, "exchange": 2, "issue_type": 1}
    )

    batch_size: int = 512
    max_epochs: int = 15
    limit_train_batches: int | None = 500
    patience: int = 3
    accelerator: str = "auto"
    num_workers: int = 0

    def __post_init__(self):
        self.panel, self.out_dir = Path(self.panel), Path(self.out_dir)
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {sorted(TARGETS)}, got {self.target!r}")
        if self.normalizer not in ("encoder", "group"):
            raise ValueError(f"normalizer must be 'encoder' or 'group', got {self.normalizer!r}")
        if self.earnings not in EARNINGS_MODES:
            raise ValueError(f"earnings must be one of {EARNINGS_MODES}, got {self.earnings!r}")
        if 0.5 not in self.quantiles:
            raise ValueError("quantiles must include 0.5 (the point forecast)")
        self.quantiles = sorted(self.quantiles)

    @property
    def target_col(self) -> str:
        return TARGETS[self.target]

    @property
    def observed_reals(self) -> list[str]:
        cols = [c for c in OBSERVED_REALS if c != self.target_col]
        return (cols + list(EARNINGS_REALS.get(self.earnings, ()))
                + (list(SENTIMENT_REALS) if self.use_sentiment else []))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["panel"], d["out_dir"] = str(self.panel.resolve()), str(self.out_dir)
        return d


# scoring-only settings: they never touch the model, so reports use the current defaults
REPORT_FIELDS = frozenset({"moves", "confidences", "prob_buckets"})
# model inputs and data cleaning added later: a run saved before the field existed was
# trained without them
ADDED_INPUTS = {"earnings": "off", "series_breaks": False}


def from_run(run_dir: Path, **overrides) -> TrainConfig:
    """The TrainConfig a run was trained with (fields added since then, and REPORT_FIELDS,
    take the current defaults, except ADDED_INPUTS; non-None `overrides` win)."""
    import json
    from dataclasses import fields

    saved = json.loads((Path(run_dir) / "config.json").read_text())["config"]
    known = {f.name for f in fields(TrainConfig)} - REPORT_FIELDS
    kw = {**ADDED_INPUTS, **{k: v for k, v in saved.items() if k in known}}
    return TrainConfig(**{**kw, **{k: v for k, v in overrides.items() if v is not None}})
