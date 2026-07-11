from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "tsdr/0.1 (+https://github.com/floens/tsdr)"
_DEFAULT_TIMEOUT = 10.0
_DEFAULT_MAX_BYTES = 8_000_000


class HttpError(Exception):
    """An HTTP fetch failed: network, timeout, bad status, or oversized body."""


def make_client(timeout: float = _DEFAULT_TIMEOUT) -> httpx.Client:
    """A cookie-keeping client for endpoints that need a multi-request handshake."""
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip"},
    )


def get_capped(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
) -> bytes:
    """GET via an existing client, refusing a body larger than max_bytes.

    Raises HttpError on any transport/HTTP failure or when the body exceeds
    max_bytes. httpx exceptions never escape.
    """
    try:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            return _read_capped(response, max_bytes)
    except httpx.HTTPError as e:
        logger.warning("http_fetch_failed url=%s error=%r", url, e)
        raise HttpError(str(e)) from e


def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """Accumulate the streamed body, aborting the moment it exceeds max_bytes so a
    huge or endless response from an untrusted endpoint can't exhaust memory."""
    body = bytearray()
    for chunk in response.iter_bytes():
        body += chunk
        if len(body) > max_bytes:
            raise HttpError(f"response exceeded {max_bytes} bytes: {response.url}")
    return bytes(body)


def http_get(
    url: str, *, timeout: float = _DEFAULT_TIMEOUT, max_bytes: int = _DEFAULT_MAX_BYTES
) -> bytes:
    """Single defensive GET; the caller decides how to handle HttpError."""
    with make_client(timeout) as client:
        return get_capped(client, url, max_bytes=max_bytes)
