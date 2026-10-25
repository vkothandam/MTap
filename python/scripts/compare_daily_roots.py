"""One-off check: do two data roots hold the same Massive daily files?

Compares `out/daily_summary/*.jsonl` and `out/extended_hours/*.jsonl` of a candidate root
(e.g. the NAS trial, MTap-Trial) with a reference root (e.g. the iMac's MTap-Data), by
ticker and field. The run metadata (`_ingested_at`, `_run_id`, `fetched_at`) is ignored.
For each file it prints the tickers in both, how many are equal, which fields differ, the
tickers in only one root, and one example of a difference.

Read-only. Used for the NAS trial of 2026-10-24 (docs/LEARNINGS.md, tag-sourcing-host).

    python3 scripts/compare_daily_roots.py /Volumes/Backups/MTap-Trial /Volumes/Backups/MTap-Data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

META = {"_ingested_at", "_run_id", "fetched_at"}


def load(path: Path) -> dict[str, dict]:
    with path.open() as f:
        rows = (json.loads(line) for line in f)
        return {r["ticker"]: {k: v for k, v in r.items() if k not in META} for r in rows}


def compare(candidate: Path, reference: Path) -> None:
    for sub in ("daily_summary", "extended_hours"):
        print("==", sub)
        for cp in sorted((candidate / "out" / sub).glob("*.jsonl")):
            rp = reference / "out" / sub / cp.name
            if not rp.exists():
                print(cp.name, "not in the reference root")
                continue
            c, r = load(cp), load(rp)
            both = c.keys() & r.keys()
            differ = sorted(k for k in both if c[k] != r[k])
            fields: dict[str, int] = {}
            for k in differ:
                for f in c[k].keys() | r[k].keys():
                    if c[k].get(f) != r[k].get(f):
                        fields[f] = fields.get(f, 0) + 1
            only_c, only_r = sorted(c.keys() - r.keys()), sorted(r.keys() - c.keys())
            print(f"{cp.name}: both {len(both)}, equal {len(both) - len(differ)}, differ {len(differ)} {fields}, "
                  f"only candidate {len(only_c)} {only_c[:4]}, only reference {len(only_r)} {only_r[:4]}")
            if differ:
                k = differ[0]
                print("   e.g.", k, {f: (r[k].get(f), c[k].get(f)) for f in fields if r[k].get(f) != c[k].get(f)},
                      "(reference, candidate)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("candidate", type=Path, help="the root to check, e.g. MTap-Trial")
    ap.add_argument("reference", type=Path, help="the root to compare with, e.g. MTap-Data")
    a = ap.parse_args()
    compare(a.candidate, a.reference)


if __name__ == "__main__":
    main()
