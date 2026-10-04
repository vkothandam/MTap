"""Thin HTTP client wrapper with sane retry/backoff defaults.

Sources should use `get_client()` rather than constructing httpx directly, so retry,
timeout, and rate-limit policy stay consistent across the module.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import httpx

from .errors import FetchError

DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
RETRY_STATUS = {429, 500, 502, 503, 504}


@contextmanager
def get_client(**kwargs) -> Iterator[httpx.Client]:
    """Yield a configured httpx.Client. Extra kwargs pass through to httpx."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    kwargs.setdefault("headers", {"user-agent": "mtap-sourcing/0.1"})
    with httpx.Client(**kwargs) as client:
        yield client


@asynccontextmanager
async def get_async_client(**kwargs) -> AsyncIterator[httpx.AsyncClient]:
    """Yield a configured httpx.AsyncClient for concurrent fetching. Kwargs pass through."""
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)
    kwargs.setdefault("headers", {"user-agent": "mtap-sourcing/0.1"})
    async with httpx.AsyncClient(**kwargs) as client:
        yield client


async def aget_text(client: httpx.AsyncClient, url: str, *, retries: int = DEFAULT_RETRIES, **kwargs):
    """Async twin of get_text: GET with retry/backoff, returning (final_url, text).

    Same policy as get_text (retry the RETRY_STATUS set + transport errors, fail fast on
    other 4xx) but non-blocking, so many fetches can overlap on one event loop.
    """
    last_exc: Exception | None = None
    status: int | None = None
    for attempt in range(retries + 1):
        try:
            resp = await client.get(url, **kwargs)
            if resp.status_code in RETRY_STATUS:
                raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
            resp.raise_for_status()
            return str(resp.url), resp.text
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            if status not in RETRY_STATUS or attempt == retries:
                break
            await asyncio.sleep(2**attempt)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == retries:
                break
            await asyncio.sleep(2**attempt)
    raise FetchError(
        f"GET {url} failed ({'HTTP ' + str(status) if status else 'transport error'}): {last_exc}",
        status_code=status,
    ) from last_exc


def get_text(client: httpx.Client, url: str, *, retries: int = DEFAULT_RETRIES, **kwargs):
    """GET with the same retry/backoff policy as get_json, but return (final_url, text).

    Used for non-JSON payloads (RSS feeds, article HTML). `final_url` is the URL after
    any redirects (httpx follows them when the client was built with follow_redirects=True)
    — needed because Google News RSS links redirect to the publisher.
    """
    last_exc: Exception | None = None
    status: int | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, **kwargs)
            if resp.status_code in RETRY_STATUS:
                raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
            resp.raise_for_status()
            return str(resp.url), resp.text
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            if status not in RETRY_STATUS or attempt == retries:
                break
            time.sleep(2**attempt)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(2**attempt)
    raise FetchError(
        f"GET {url} failed ({'HTTP ' + str(status) if status else 'transport error'}): {last_exc}",
        status_code=status,
    ) from last_exc


def get_json(client: httpx.Client, url: str, *, retries: int = DEFAULT_RETRIES, **kwargs):
    """GET with retry/backoff on transient failures. Returns parsed JSON.

    Only transient failures are retried: the statuses in RETRY_STATUS and transport/
    timeout errors. A non-retryable HTTP status (e.g. 401/403/404) fails fast — retrying
    it just wastes time — and the resulting FetchError carries `status_code`.
    """
    last_exc: Exception | None = None
    status: int | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.get(url, **kwargs)
            if resp.status_code in RETRY_STATUS:
                raise httpx.HTTPStatusError("retryable status", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            if status not in RETRY_STATUS or attempt == retries:
                break  # non-retryable (4xx) fails immediately; retryable stops at the cap
            time.sleep(2**attempt)  # 1s, 2s, 4s
        except httpx.HTTPError as exc:  # transport / timeout — transient, retry
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(2**attempt)
    raise FetchError(
        f"GET {url} failed ({'HTTP ' + str(status) if status else 'transport error'}): {last_exc}",
        status_code=status,
    ) from last_exc
