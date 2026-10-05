"""TFT next-day VWAP model trained on the MTap `panel.parquet` contract.

Self-contained by design: this package reads only `panel.parquet` + `panel.meta.json` and
never imports the sourcing code or its stores, so it can move to its own repo unchanged.
"""
