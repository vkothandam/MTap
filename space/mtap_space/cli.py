"""space-build: build the moment library from the minute bars and save it."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from mtap_space import library

DEFAULT_BARS = "/Volumes/Backups/MTap-Data/bars_1m"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="space-build", description=__doc__)
    ap.add_argument("--bars-root", default=DEFAULT_BARS,
                    help="minute bars, month=YYYY-MM/*.parquet (read-only)")
    ap.add_argument("--sessions", type=int, default=90, help="sessions of moments to keep")
    ap.add_argument("--until", help="last session to include (YYYY-MM-DD); default: the newest")
    ap.add_argument("--min-dollar", type=float, default=5e6,
                    help="stocks trading at least this a day over the prior 20 sessions")
    ap.add_argument("--min-price", type=float, default=3.0)
    ap.add_argument("--out", default="../out/space", help="writes <out>/lib-<last session>/")
    a = ap.parse_args(argv)
    t0 = time.time()
    print(f"space-build: {a.sessions} sessions from {a.bars_root}")
    lib = library.build(a.bars_root, a.sessions, a.until, a.min_dollar, a.min_price)
    path = library.save(lib, a.out)
    size = sum(f.stat().st_size for f in Path(path).iterdir()) / 1e9
    print(f"saved {path} ({size:.2f} GB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
