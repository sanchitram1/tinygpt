"""Request-size guard and a simple in-memory rate limiter.

Both are deliberately minimal for v1: the body guard is header-based (Cloud
Run and every mainstream HTTP client send Content-Length; chunked API uploads
are rejected), and the rate limiter is per-process sliding-window state, which
is adequate while Cloud Run concurrency is kept conservative.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from .errors import error_body

_API_PREFIX = "/api/"


class BodySizeLimitMiddleware:
    """Reject API request bodies larger than ``max_bytes`` with a stable 413."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith(_API_PREFIX):
            await self.app(scope, receive, send)
            return
        if scope["method"] in ("POST", "PUT", "PATCH"):
            headers = {name.lower(): value for name, value in scope["headers"]}
            content_length = headers.get(b"content-length")
            if content_length is None:
                await _send_json(send, 411, "length_required", "Content-Length is required")
                return
            try:
                length = int(content_length)
            except ValueError:
                await _send_json(send, 400, "invalid_request", "invalid Content-Length")
                return
            if length > self.max_bytes:
                await _send_json(
                    send,
                    413,
                    "request_too_large",
                    f"request body exceeds {self.max_bytes} bytes",
                )
                return
        await self.app(scope, receive, send)


async def _send_json(send, status: int, code: str, message: str) -> None:
    import json

    body = json.dumps(error_body(code, message)).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class SlidingWindowRateLimiter:
    """Per-key request budget over a 60-second sliding window. 0 disables."""

    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0) -> None:
        self.limit = limit_per_minute
        self.window = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and now - events[0] > self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True
