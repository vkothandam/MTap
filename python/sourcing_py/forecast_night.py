"""The night forecast job (`sourcing-py forecast-night --date D`), run once per forecast date.

D is the forecast session (the next one); t is the session before it. The steps:

    1. Massive EOD + extended hours for t   (run massive_grouped_daily, massive-ext gaps)
    2. etrade ingest-eod, fetch-splits, fetch-edgar, derive-features
    3. etrade universe --date D (premarket_live/universe-D.csv), premarket xids (symbol,XID file
       for the NAS, which has no DuckDB)
    4. etrade export-tft --include-inference (panel.next.parquet, pre-market masked = night state)
    5. the ML predict, as a subprocess (`[forecast] predict_cmd`, `--state night`); ml/ and
       sourcing_py never import each other

Claim: MTap-State/forecast/claim-D.json, created with O_EXCL, {host, status running|done,
started_at, heartbeat}. The heartbeat is refreshed every minute. A host that finds `done`, or
`running` with a heartbeat younger than 30 min, stands aside; a stale `running` claim is taken
over (atomic rename of the stale file, so two hosts cannot both take it over).

Earliest start 20:20 ET on t (post-market ends 20:00, Massive's minute bars lag ~15 min); refused
earlier unless --force.
"""

from __future__ import annotations

import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time as _time
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .common import config
from .common.errors import ConfigError

NY = ZoneInfo("America/New_York")


# -- the claim ------------------------------------------------------------------------------
def _now() -> datetime:
    return datetime.now(UTC)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def claim_path(d: date, claim_dir: Path | None = None) -> Path:
    return (claim_dir or config.forecast_config()["claim_dir"]) / f"claim-{d.isoformat()}.json"


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_new(path: Path, data: dict) -> bool:
    """Exclusive create. True if this call created the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data))
        f.flush()
        os.fsync(f.fileno())
    return True


class Claim:
    """A held claim. Use acquire(); then heartbeat via start_heartbeat(); finally done()."""

    def __init__(self, path: Path, host: str, started_at: str) -> None:
        self.path, self.host, self.started_at = path, host, started_at
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _save(self, status: str, now: datetime | None = None) -> None:
        with self._lock:
            tmp = self.path.with_name(f".{self.path.name}.{self.host}.tmp")
            tmp.write_text(json.dumps({"host": self.host, "status": status, "started_at": self.started_at,
                                       "heartbeat": _iso(now or _now())}))
            os.replace(tmp, self.path)

    def beat(self, now: datetime | None = None) -> None:
        self._save("running", now)

    def start_heartbeat(self, every_s: float = 60.0) -> None:
        def loop():
            while not self._stop.wait(every_s):
                try:
                    self.beat()
                except OSError:
                    pass  # a failed beat is retried next minute

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def done(self) -> None:
        self.stop_heartbeat()
        self._save("done")

    def release(self) -> None:
        """A failed run gives the claim up at once (so the next host need not wait 30 min)."""
        self.stop_heartbeat()
        with self._lock:
            self.path.unlink(missing_ok=True)


def acquire(d: date, *, host: str | None = None, now: datetime | None = None, stale_min: int | None = None,
            claim_dir: Path | None = None) -> tuple[Claim | None, str]:
    """(claim, why). claim is None when another host has done it (why = 'done') or is running it
    (why = 'running'); otherwise why = 'new' or 'takeover'."""
    cfg = config.forecast_config()
    stale = timedelta(minutes=cfg["stale_min"] if stale_min is None else stale_min)
    host = host or socket.gethostname()
    now = now or _now()
    path = claim_path(d, claim_dir)
    fresh = {"host": host, "status": "running", "started_at": _iso(now), "heartbeat": _iso(now)}
    if _write_new(path, fresh):
        return Claim(path, host, fresh["started_at"]), "new"
    cur = _read(path)
    if cur and cur.get("status") == "done":
        return None, "done"
    hb = None
    if cur and cur.get("heartbeat"):
        try:
            hb = datetime.fromisoformat(cur["heartbeat"].replace("Z", "+00:00"))
        except ValueError:
            hb = None
    if hb is not None and now - hb < stale:
        return None, "running"
    # stale (or unreadable): take over under a short-lived exclusive lock, re-checking staleness
    # inside it, so of several hosts racing exactly one replaces the claim.
    lock = path.with_name(f".{path.name}.lock")
    try:
        if lock.exists() and _time.time() - lock.stat().st_mtime > 120:
            lock.unlink(missing_ok=True)  # a crashed taker-over
    except OSError:
        pass
    if not _write_new(lock, {"host": host}):
        return None, "running"
    try:
        cur = _read(path)
        if cur and cur.get("status") == "done":
            return None, "done"
        hb = _hb(cur)
        if hb is not None and now - hb < stale:
            return None, "running"
        path.unlink(missing_ok=True)
        if _write_new(path, fresh):
            return Claim(path, host, fresh["started_at"]), "takeover"
        return None, "running"
    finally:
        lock.unlink(missing_ok=True)


def _hb(cur: dict | None) -> datetime | None:
    if cur and cur.get("heartbeat"):
        try:
            return datetime.fromisoformat(cur["heartbeat"].replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


# -- the schedule ---------------------------------------------------------------------------
def previous_session(d: date) -> date:
    from .utils.trading_calendar import sessions

    prior = [s for s in sessions(d - timedelta(days=10), d - timedelta(days=1))]
    if not prior:
        raise ConfigError(f"no session before {d}")
    return prior[-1]


def next_session(after: date) -> date:
    from .utils.trading_calendar import next_trading_day

    return next_trading_day(after + timedelta(days=1))


def earliest_start(d: date, earliest: str | None = None) -> datetime:
    """20:20 ET on the session before d."""
    h, m = (earliest or config.forecast_config()["earliest"]).split(":")
    return datetime.combine(previous_session(d), time(int(h), int(m)), NY).astimezone(UTC)


# -- the steps ------------------------------------------------------------------------------
SOFT = {("etrade", "fetch-splits"), ("etrade", "fetch-edgar"), ("massive-ext", "gaps")}


def steps(d: date, t: date) -> list[list[str]]:
    """The sourcing-py steps (argv after `sourcing-py`) for forecast session d, data day t.
    export-tft --include-inference already writes panel.next.parquet with the pre-market columns
    masked, which is the night state, so no export-next runs here."""
    return [
        ["run", "massive_grouped_daily", "--date", t.isoformat()],
        ["massive-ext", "gaps", "--days", "14", "--max", "10"],
        ["etrade", "ingest-eod", "--days", "10"],
        ["etrade", "fetch-splits"],
        ["etrade", "fetch-edgar"],
        ["etrade", "derive-features"],
        ["etrade", "universe", "--date", d.isoformat()],
        ["etrade", "premarket", "xids"],
        ["etrade", "export-tft", "--include-inference"],
    ]


def predict_argv(run_dir: str, extra: list[str] | None = None) -> list[str]:
    cfg = config.forecast_config()
    if not (run_dir or cfg["run_dir"]):
        raise ConfigError("no model run: set [forecast] run_dir or env MTAP_FORECAST_RUN_DIR")
    return [*shlex.split(cfg["predict_cmd"]), run_dir or cfg["run_dir"], "--state", "night", *(extra or [])]


def run_cmd(argv: list[str], cwd: Path | None = None) -> int:
    print(f"forecast-night >>> {' '.join(argv)}", flush=True)
    return subprocess.run(argv, cwd=cwd).returncode


def run_night(d: date, *, force: bool = False, run_dir: str = "", now: datetime | None = None,
              runner=run_cmd, claim_dir: Path | None = None, host: str | None = None) -> int:
    """0 done (or already done elsewhere), 1 a step failed, 2 refused."""
    now = now or _now()
    t = previous_session(d)
    start = earliest_start(d)
    if now < start and not force:
        print(f"forecast-night: refused, earliest start for {d} is {start.astimezone(NY):%Y-%m-%d %H:%M} ET "
              "(use --force)", file=sys.stderr)
        return 2
    claim, why = acquire(d, host=host, now=now, claim_dir=claim_dir)
    if claim is None:
        print(f"forecast-night {d}: nothing to do ({why} on another host)")
        return 0
    print(f"forecast-night {d}: claim {why} on {claim.host}")
    claim.start_heartbeat(config.forecast_config()["heartbeat_s"])
    try:
        repo = config.repo_root()
        for argv in steps(d, t):
            # The soft steps are retried by the next run; any other failure gives the claim up.
            rc = runner(["sourcing-py", *argv])
            if rc != 0 and tuple(argv[:2]) not in SOFT and argv[0] != "run":
                print(f"forecast-night: step failed ({rc}): {' '.join(argv)}", file=sys.stderr)
                claim.release()
                return 1
            if rc != 0:
                print(f"forecast-night: {' '.join(argv)} failed ({rc}); continuing", file=sys.stderr)
        rc = runner(predict_argv(run_dir), cwd=repo)
        if rc != 0:
            print(f"forecast-night: predict failed ({rc})", file=sys.stderr)
            claim.release()
            return 1
        claim.done()
        return 0
    except BaseException:
        claim.release()
        raise


def main(argv: list[str]) -> int:
    from .__main__ import _parse_params

    try:
        p = _parse_params(argv)
        d = date.fromisoformat(p["date"]) if p.get("date") not in (None, True) else next_session(datetime.now(NY).date())
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}\nusage: sourcing-py forecast-night [--date D] [--force] [--run-dir DIR]", file=sys.stderr)
        return 2
    try:
        return run_night(d, force=bool(p.get("force")),
                         run_dir=p["run-dir"] if p.get("run-dir") not in (None, True) else "")
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
