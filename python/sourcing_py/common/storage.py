"""Storage roots: `sourcing-py storage check|init`, and the guard every NAS writer calls.

MTAP_DATA_ROOT (usually a Synology share under /Volumes) holds the write-once parquet;
MTAP_STATE_ROOT (local SSD) holds everything small or live. See common/config.py.

The marker file DATA_ROOT/.mtap-data-root is created once by `storage init`. When an SMB
share drops, macOS leaves an empty /Volumes/<name> folder behind that is really on the
local disk; writing into it would silently fill the iMac. Writers check the marker first.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from . import config
from .errors import ConfigError

DATA_DIRS = ("intraday_feed", "agg_1s", "bars_1m", "out", "backup")
STATE_DIRS = ("etrade", "logs", "segments", "segments_massive", "intraday", "massive", "rate", "failures")


def data_root_configured() -> bool:
    """True when the data root is explicitly set (not the repo-root fallback)."""
    return config.data_root().resolve() != config.repo_root().resolve()


def marker_path() -> Path:
    return config.data_root() / config.DATA_ROOT_MARKER


def unmounted_volume(path: Path) -> bool:
    """A /Volumes/<name>/… path whose /Volumes/<name> is not a mount point."""
    parts = path.resolve().parts
    if len(parts) < 3 or parts[1] != "Volumes":
        return False
    return not os.path.ismount(Path(*parts[:3]))


def data_root_ready() -> tuple[bool, str]:
    """(ok, reason). The repo-root fallback needs no marker."""
    root = config.data_root()
    if not data_root_configured():
        return True, "data root is the repo (MTAP_DATA_ROOT not set)"
    if unmounted_volume(root):
        return False, f"{root} is under /Volumes but the share is not mounted"
    if not root.is_dir():
        return False, f"{root} does not exist"
    if not marker_path().exists():
        return False, f"marker {marker_path()} missing (run `sourcing-py storage init`)"
    return True, "ok"


def require_data_root() -> Path:
    """The data root, or ConfigError if it is not safe to write to."""
    ok, reason = data_root_ready()
    if not ok:
        raise ConfigError(f"data root not ready: {reason}")
    return config.data_root()


def _writable(path: Path) -> bool:
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".mtap-write-test-"):
            return True
    except OSError:
        return False


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def check() -> int:
    """Print both roots and their health. Exit 1 if either is unusable."""
    cfg = config.storage_config()
    data, state = cfg["data_root"], cfg["state_root"]
    ok = True

    ready, reason = data_root_ready()
    print(f"data root   {data}")
    print(f"  ready     {'yes' if ready else 'NO'} ({reason})")
    if cfg["nas_url"]:
        print(f"  nas url   {cfg['nas_url']}")
    if data.is_dir():
        usage = shutil.disk_usage(data)
        print(f"  writable  {'yes' if _writable(data) else 'NO'}; free {_gb(usage.free)}")
    print(f"  out root  {config.out_root()}")
    ok &= ready

    print(f"state root  {state}")
    if state.is_dir():
        usage = shutil.disk_usage(state)
        writable = _writable(state)
        print(f"  writable  {'yes' if writable else 'NO'}; free {_gb(usage.free)}")
        ok &= writable
    else:
        print("  missing (run `sourcing-py storage init`)")
        ok = False

    pending = _dir_size(state / "segments")
    cap = cfg["max_local_segments_gb"] * 1e9
    flag = "  OVER CAP — is the NAS down?" if pending > cap else ""
    print(f"  segments  {_gb(pending)} not yet uploaded (cap {_gb(cap)}){flag}")
    massive = state / "segments_massive"
    if massive.is_dir():
        print(f"  massive   {_gb(_dir_size(massive))} not yet uploaded "
              f"(collector pauses at {config.massive_ws_config()['max_local_gb']:.0f} GB)")
    return 0 if ok else 1


def init() -> int:
    """Create both roots' folders and the data-root marker."""
    cfg = config.storage_config()
    data, state = cfg["data_root"], cfg["state_root"]
    if data_root_configured() and unmounted_volume(data):
        print(f"error: {data} is under /Volumes but the share is not mounted; mount it first "
              "(scripts/storage/mount-nas.sh)", file=sys.stderr)
        return 1
    for d in STATE_DIRS:
        (state / d).mkdir(parents=True, exist_ok=True)
    for d in DATA_DIRS:
        (data / d).mkdir(parents=True, exist_ok=True)
    if data_root_configured():
        marker_path().write_text("MTap data root. Jobs refuse to write here without this file.\n")
    print(f"initialised data root {data} and state root {state}")
    return check()


def main(argv: list[str]) -> int:
    sub = argv[0] if argv else ""
    if sub == "check":
        return check()
    if sub == "init":
        return init()
    print("usage: sourcing-py storage check|init", file=sys.stderr)
    return 2
