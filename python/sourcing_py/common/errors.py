"""Typed errors so the CLI can tell fatal config problems from per-day failures."""

from __future__ import annotations


class SourceError(Exception):
    """Base for errors raised by the sourcing pipeline."""


class ConfigError(SourceError):
    """Misconfiguration (missing key, bad settings). Fatal — abort the run."""


class FetchError(SourceError):
    """A single fetch failed after retries. Per-day: log it and continue.

    `status_code` carries the HTTP status when the failure was an HTTP error (None for
    transport/timeout failures), so callers can distinguish e.g. an expired-auth 401
    from a transient network error.
    """

    def __init__(self, *args, status_code: int | None = None) -> None:
        super().__init__(*args)
        self.status_code = status_code
