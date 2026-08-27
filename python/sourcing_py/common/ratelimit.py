"""Cross-process request spacing, keyed by provider.

Free-tier APIs cap requests per minute *per account*, so the limit is shared across
every program that talks to the same provider. This throttle enforces a minimum
interval between requests using a file lock under state/rate/<provider>.lock — held
across the wait — so concurrent programs queue instead of bursting past the limit.
"""

from __future__ import annotations

import fcntl
import time

from . import config


def throttle(provider: str, min_interval: float) -> float:
    """Block until >= min_interval seconds have passed since the last request for
    `provider` (across all processes). Returns the seconds actually waited."""
    if min_interval <= 0:
        return 0.0

    state_dir = config.repo_root() / "state" / "rate"
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / f"{provider}.lock"

    with open(lock_path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            content = fh.read().strip()
            last = float(content) if content else 0.0
            now = time.time()
            wait = (last + min_interval) - now
            if wait > 0:
                time.sleep(wait)
                now = last + min_interval
            fh.seek(0)
            fh.truncate()
            fh.write(f"{now:.3f}")
            fh.flush()
            return max(0.0, wait)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
