"""CLI for the contract recommender (`mtap/recommender`).

Dispatched from `sourcing_py.__main__` when the first arg is `recommender`:

    sourcing-py recommender run [--for-s N]           (the service loop; REDIS_URL)
    sourcing-py recommender submit --file CAND.json   (one candidate onto mtap:int:REC_CANDIDATE)
    sourcing-py recommender status                    (state/recommender/status.json)
    sourcing-py recommender check-schemas             (vendored copy vs MBin's, and the pins)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

from ..common import bus, config
from . import contract, service

_USAGE = (
    "usage:\n"
    "  sourcing-py recommender run [--for-s N]   (publish recs, heartbeat, answer RESYNC)\n"
    "  sourcing-py recommender submit --file CAND.json   (queue one candidate)\n"
    "  sourcing-py recommender status   (last status.json)\n"
    "  sourcing-py recommender check-schemas   (vendored schemas vs MBin + pins)"
)


def _url() -> str:
    return config.intraday_config()["redis_url"]


def check_schemas() -> list[str]:
    """Problems with the vendored schemas: a pin that doesn't match the bytes, or (when
    MBin is checked out next to MTap) bytes that differ from MBin's."""
    out = []
    for name, pin in contract.PINNED.items():
        got = hashlib.sha256(contract.schema_bytes(name)).hexdigest()
        if got != pin:
            out.append(f"{name}: vendored bytes {got[:12]}… != pinned {pin[:12]}…")
        theirs = contract.mbin_schemas_dir() / f"{name}.schema.json"
        if theirs.exists() and theirs.read_bytes() != contract.schema_bytes(name):
            out.append(f"{name}: differs from MBin's {theirs}")
    return out


def main(argv: list[str]) -> int:
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    params = {rest[i].lstrip("-"): rest[i + 1] for i in range(0, len(rest) - 1, 2)}
    if cmd == "run":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        for_s = float(params["for-s"]) if "for-s" in params else None
        return service.run(_url(), stop_after_s=for_s)
    if cmd == "submit" and "file" in params:
        cand = json.loads(Path(params["file"]).read_text())
        entry = bus.publish(_url(), service.CANDIDATE_EVENT, cand, prefix=service.CANDIDATE_PREFIX)
        print(f"queued {cand.get('symbol')} as {entry}")
        return 0
    if cmd == "status":
        path = service.state_dir() / "status.json"
        print(path.read_text() if path.exists() else f"no status yet ({path})")
        return 0
    if cmd == "check-schemas":
        problems = check_schemas()
        for p in problems:
            print(p)
        print("OK" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    print(_USAGE, file=sys.stderr)
    return 2
