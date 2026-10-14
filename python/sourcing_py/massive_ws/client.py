"""One Massive WebSocket session in a thread: connect, auth, subscribe, queue raw frames.

The thread does nothing but `recv()` and `put()`, so it keeps up with the server (Massive
drops clients that fall behind); parsing and disk writes happen in the collector. A
session ends on any error or close; the collector starts a new one after a backoff.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time

log = logging.getLogger("massive_ws.client")


class FeedError(Exception):
    """The server refused us (`status` is e.g. auth_failed, max_connections)."""

    def __init__(self, status: str, message: str = ""):
        super().__init__(f"{status}: {message}" if message else status)
        self.status = status


def _connect(url: str, **kw):
    from websockets.sync.client import connect

    return connect(url, **kw)


class Session(threading.Thread):
    def __init__(self, url: str, key: str, channels: str, out: queue.Queue, *, connect=_connect):
        super().__init__(name="massive-ws", daemon=True)
        self.url, self.key, self.channels, self.out = url, key, channels, out
        self._connect = connect
        self._halt = threading.Event()
        self._ws = None
        self.state = "connecting"  # connecting -> subscribed -> closed
        self.error: FeedError | Exception | None = None
        self.subscribed_at: float | None = None
        self.frames = 0
        self.dropped = 0  # frames lost because the queue was full (collector stalled)

    def _expect(self, ws, want: str) -> None:
        """Read status frames until `want`; raise on a refusal."""
        while True:
            for msg in json.loads(ws.recv(timeout=15)):
                if msg.get("ev") != "status":
                    continue
                status = msg.get("status", "")
                if status == want:
                    return
                if status in ("auth_failed", "max_connections", "error") or "fail" in status:
                    raise FeedError(status, msg.get("message", ""))

    def run(self) -> None:
        try:
            with self._connect(self.url, open_timeout=15, max_size=None, ping_interval=20,
                               ping_timeout=20, close_timeout=5) as ws:
                self._ws = ws
                self._expect(ws, "connected")
                ws.send(json.dumps({"action": "auth", "params": self.key}))
                self._expect(ws, "auth_success")
                ws.send(json.dumps({"action": "subscribe", "params": self.channels}))
                self.state, self.subscribed_at = "subscribed", time.monotonic()
                log.info("subscribed to %s at %s", self.channels, self.url)
                while not self._halt.is_set():
                    try:
                        raw = ws.recv(timeout=1)
                    except TimeoutError:
                        continue
                    self.frames += 1
                    try:
                        self.out.put_nowait(raw)
                    except queue.Full:
                        self.dropped += 1
        except Exception as exc:  # noqa: BLE001 — ConnectionClosed, OSError, FeedError, ...
            if not self._halt.is_set():
                self.error = exc
                log.warning("websocket session ended: %s: %s", type(exc).__name__, exc)
        finally:
            self.state = "closed"

    def stop(self) -> None:
        self._halt.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001, S110 — already closing or closed
                pass
        self.join(timeout=10)
