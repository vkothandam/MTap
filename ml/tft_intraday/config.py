"""Intraday training configuration: one dataclass, serialised next to every run's results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

SUPPORTED_SCHEMA_VERSIONS = (1,)

# The panel's targets, in tmask bit order (panel.meta.json tmask_bits): log moves from the last
# price at the decision time t to the VWAP of the next 1/5/15/60 minutes, of t..close, and of
# the whole day (FEATURES-intraday.md).
TARGETS = ("r_1m", "r_5m", "r_15m", "r_60m", "r_rod", "r_close")
# The momentum baseline's reference per target ("the VWAP ahead equals the trailing VWAP")
# and the dashboard's `ref_trail` column.
MOMENTUM_REF = {"r_1m": "ref_trail_1m", "r_5m": "ref_trail_5m", "r_15m": "ref_trail_15m",
                "r_60m": "ref_trail_60m", "r_rod": "ref_trail_60m", "r_close": "ref_session"}
QUANTILES = [round(k * 0.05, 2) for k in range(1, 20)]

# The panel's model inputs (panel.meta.json features), plus the morning daily forecast joined
# by daily_join.py. Group context (market / sector aggregates) can be switched off.
GROUP_REALS = ("mkt_trail_5", "mkt_trail_15", "mkt_breadth", "sec_trail_5", "sec_rvol",
               "sec_breadth", "excess_5", "rank_trail_5", "rank_rvol")
DAILY_REALS = ("has_daily", "d_med", "d_width", "d_skew", "d_p_up", "hist_mae", "hist_cov80",
               "sec_d_med")
EMBEDDING_SIZES = {"sector_code": 6, "watchlist": 1, "day_of_week": 2, "is_half_day": 1,
                   "tod_bucket": 4}


@dataclass
class IntradayConfig:
    panel: Path = Path("../out/tft_intraday/panel")
    out_dir: Path = Path("runs/intraday")
    # The daily run whose morning forecast is joined in. Its fold 1 tests 2025-11-25..2026-02-23,
    # so every backfill day is out of its sample; the later folds trained on December.
    daily_run: Path | None = Path("runs/20261011-173109Z-return-3785sym")
    daily_fold: int = 1
    use_group: bool = True
    symbols: int | None = None  # random subset of N symbols for fast iteration
    seed: int = 42

    # Walk-forward blocks: each list is one fold's test dates. Early stopping uses the dates just
    # before the test, as many as it takes to reach `es_min_rows`; training uses all before that.
    test_blocks: list[list[str]] = field(default_factory=lambda: [
        ["2025-12-29", "2025-12-30"], ["2026-01-12"]])
    es_min_rows: int = 600_000
    only_folds: list[int] | None = None

    encoder_steps: int = 80  # 20 minutes of 15 s steps; day-so-far features carry the rest
    min_encoder_steps: int = 4
    es_stride_s: int = 300  # one early-stop decision per symbol every 5 minutes
    test_stride_s: int = 60

    hidden_size: int = 32
    attention_head_size: int = 2
    hidden_continuous_size: int = 16
    dropout: float = 0.1
    learning_rate: float = 1e-3
    gradient_clip_val: float = 0.1
    quantiles: list[float] = field(default_factory=lambda: list(QUANTILES))

    batch_size: int = 512
    max_epochs: int = 6
    limit_train_batches: int | None = 300
    patience: int = 2
    accelerator: str = "auto"
    num_workers: int = 0
    baselines_only: bool = False  # write momentum forecasts as the model's, no training

    def __post_init__(self):
        self.panel, self.out_dir = Path(self.panel), Path(self.out_dir)
        self.daily_run = None if self.daily_run is None else Path(self.daily_run)
        if self.quantiles != QUANTILES:
            raise ValueError("the dashboard contract needs exactly P5..P95 in 5-point steps")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["panel"], d["out_dir"] = str(self.panel.resolve()), str(self.out_dir)
        d["daily_run"] = None if self.daily_run is None else str(self.daily_run)
        return d
