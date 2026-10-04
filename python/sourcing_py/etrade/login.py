"""`etrade login` — bootstrap the browser session via the Node puppeteer service.

Phase 2 (`fetch`) authenticates to the wsod REST API and does **not** log in itself:
it consumes `state/etrade/session.json`. There are two ways to produce that file.

  1. Manual bearer token (PRIMARY / verified). Paste a short-lived
     `authorization: Bearer <token>` copied from a logged-in browser into the session
     file as `{"accessToken": "<token>"}`. This is the proven path the full
     15,886-symbol run used end to end. Tokens are short-lived; on 401/403 the fetch
     aborts cleanly and you re-seed a fresh token and re-run with `--resume`.

  2. Puppeteer login (THIS command; fallback/automation). Launch the long-running Node
     E*TRADE service (`npm run serve -- etrade`), which drives a headless browser through
     the login + SMS/OTP flow and writes `session.json` (cookies + `stk1`/`stk2` request
     headers, plus any bearer token it captured from the page's own XHRs). Needs `.env`
     (ET_USERNAME/ET_PASSWORD, the SMS gateway, OTP host) and a reachable Redis — see
     node/src/services/etrade/README.md.

     CAVEAT: live sampling showed the fundamentals REST API authenticates with a *bearer
     token*; whether the cookie/`stk1`/`stk2` session this service persists is sufficient
     for that API on its own is UNVERIFIED. If `fetch` then returns 401/403, fall back to
     option 1 with a fresh token.

This helper shells out to the Node service, streams its output inline (so you can watch
for the OTP link and complete the challenge), and waits until `session.json` is
(re)written before reporting readiness. The service keeps running afterwards to hold the
session alive — leave it up and run `fetch` in another terminal, or pass `--once` to stop
it as soon as the session lands.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from ..common import config
from ..common.errors import ConfigError


def _load_env_file(path: Path) -> dict[str, str]:
    """Minimal `.env` parser (`KEY=VALUE` lines; ignores blanks, `#` comments, and a
    leading `export `). We load it ourselves and pass it into the subprocess env so the
    Node service sees its credentials regardless of whether npm/tsx auto-loads `.env`."""
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#"):
            continue
        key, sep, val = line.partition("=")
        if not sep:
            continue
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def _session_fresh(path: Path, before_mtime: float | None) -> bool:
    """True once the service has (re)written the session file this run — either it did
    not exist before (now present) or its mtime advanced past our start snapshot."""
    if not path.exists():
        return False
    if before_mtime is None:
        return True
    return path.stat().st_mtime > before_mtime


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def login(*, once: bool = False, timeout: float = 300.0, poll: float = 2.0) -> int:
    """Launch the Node puppeteer login and wait for it to write the session file.

    Returns a shell exit code: 0 once a session is ready (or after a clean Ctrl-C once it
    was ready), non-zero if the login process died early or timed out first.
    """
    repo = config.repo_root()
    node_dir = repo / "node"
    if not (node_dir / "package.json").exists():
        raise ConfigError(
            f"Node project not found at {node_dir} — cannot launch the puppeteer login. "
            "Run the manual-token path instead (see login.py / the README)."
        )
    session_path = config.etrade_config()["session_path"]

    # Merge repo .env OVER the current environment so ET_USERNAME/REDIS_URL/OTP_HOST/… are
    # present for the child even though `npm run` does not itself load a .env file.
    env = dict(os.environ)
    env.update(_load_env_file(repo / ".env"))

    before_mtime = session_path.stat().st_mtime if session_path.exists() else None

    print(f"launching Node E*TRADE login: `npm run serve -- etrade` (cwd={node_dir})")
    print(
        "  Watch the output for the SMS/OTP link and complete the challenge.\n"
        f"  Waiting up to {timeout:.0f}s for {session_path} to be (re)written..."
    )
    try:
        # Inherit stdout/stderr so the login/OTP prompts stream straight to the terminal.
        proc = subprocess.Popen(["npm", "run", "serve", "--", "etrade"], cwd=str(node_dir), env=env)
    except FileNotFoundError as exc:
        raise ConfigError(
            "`npm` not found on PATH — install Node ≥20 and run `cd node && npm install` "
            "(see node/src/services/etrade/README.md)."
        ) from exc

    deadline = time.monotonic() + timeout
    try:
        while True:
            if proc.poll() is not None:  # service exited before producing a session
                print(f"login process exited early (code {proc.returncode}) — no session written.")
                return proc.returncode or 1
            if _session_fresh(session_path, before_mtime):
                print(f"\n✓ session ready at {session_path}")
                print("  next:  uv run sourcing-py etrade fetch --all --resume")
                if once:
                    print("  --once: stopping the login service now.")
                    _terminate(proc)
                    return 0
                print("  leaving the service running to keep the session alive; Ctrl-C to stop.")
                proc.wait()
                return proc.returncode or 0
            if time.monotonic() > deadline:
                print(f"\ntimed out after {timeout:.0f}s with no session; stopping the login service.")
                _terminate(proc)
                return 1
            time.sleep(poll)
    except KeyboardInterrupt:
        print("\ninterrupted — stopping the login service.")
        _terminate(proc)
        return 130
