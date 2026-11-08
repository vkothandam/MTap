"""Intraday model v2: a sequence model on pocket targets over duration bins (docs/intraday-features.md).

    data      read-only loaders: minute bars, the 1 s poll grid, day reference data, forecasts, SPY / SPX
    zigzag    causal zig-zag (no repainting)
    profile   the second tier: time at price, value area, resistance, extremes, grid minute bars
    features  the FEATURES table and one function per chunk
    targets   pocket targets per bin, t50, t_peak, vol_to, the race, the past-15-min pocket
    build     the month cache /tmp/mtap/intraday_ft/frames/month=YYYY-MM/ (intraday-ft-build)
    windows   folds, standardisation, the windows dataset
    tft       the thin TFT, training and the run file (intraday-ft)
    report    scores against the realised targets, with baselines (intraday-ft-report)

Imports none of tft_vwap, tft_intraday or sourcing_py.
"""
