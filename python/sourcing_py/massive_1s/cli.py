"""CLI for the NAS 1-second-bar fetcher's inputs (sourcing_py/massive_1s/prepare.py).

Dispatched from `sourcing_py.__main__` when the first arg is `massive-1s`:

    sourcing-py massive-1s prepare --fromdate D1 --todate D2 [--from-bars] [--name N]

--from-bars takes the stocks from what traded in bars_1m over the range (delisted and
renamed ones under the ticker they traded as) instead of today's list; both add SPY.
--name N writes inputs-N/ instead of inputs/ (for a run queued behind a running one).

Run by nas/deploy.sh; the fetch itself runs on the NAS (nas/README.md).
"""

from __future__ import annotations

import sys
from datetime import date

from ..common.errors import ConfigError
from ..intraday.cli import _parse

_USAGE = "usage:\n  sourcing-py massive-1s prepare --fromdate D1 --todate D2 [--from-bars] [--name N]"


def main(argv: list[str]) -> int:
    from . import prepare as prep

    if not argv or argv[0] != "prepare":
        print(_USAGE, file=sys.stderr)
        return 2
    try:
        params = _parse(argv[1:])
        d0, d1 = date.fromisoformat(params["fromdate"]), date.fromisoformat(params["todate"])
        if d1 < d0:
            raise ValueError(f"todate {d1} is before fromdate {d0}")
        unknown = set(params) - {"fromdate", "todate", "from-bars", "name"}
        if unknown:
            raise ValueError(f"unknown option(s): {', '.join(sorted(unknown))}")
        name = params.get("name")
        if name is True:
            raise ValueError("--name needs a value")
        s = prep.prepare(d0, d1, from_bars=params.get("from-bars") is True, name=name)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}\n{_USAGE}", file=sys.stderr)
        return 2
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    exch = ", ".join(f"{e} {n:,}" for e, n in s["by_exchange"])
    excl = ", ".join(f"{r} {n}" for r, n in s["excluded"]) or "none"
    seen = "traded in bars_1m" if s["from_bars"] else "listed"
    print(f"stocks: {s['kept']:,} kept of {s['listed']:,} {seen} ({exch})")
    if s["from_bars"]:
        print(f"  {s['not_listed_now']:,} kept stock(s) are no longer listed under that ticker (delisted or renamed)")
    print(f"  excluded {s['listed'] - s['kept']}: {excl} (list: {s['dir'] / 'universe.csv'})")
    print(f"sessions: {s['sessions']:,} ({s['first']} .. {s['last']}), {s['half_days']} half day(s)")
    print(f"wrote {s['dir']}/universe.csv and windows.csv")
    return 0
