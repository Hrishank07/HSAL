"""
Fixed-window rate limiting.

Runs BEFORE any embedding or LLM call: the point of a gateway-side
limiter is to stop a hot client from burning compute, so it must sit in
front of the expensive path, not behind it.

Two backends:
- InMemoryRateLimiter: single-process, for local mode. Thread-safe.
- RedisRateLimiter: cross-instance, INCR + EXPIRE per window (the
  pattern documented by Redis for fixed-window limiting).

Fixed window is deliberately the first algorithm: simple to reason
about and test. Its known weakness — a burst straddling a window
boundary can briefly see up to 2x the limit — is acceptable here and
documented rather than hidden.
"""

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from hsal.utils.config import settings


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    window_seconds: int
    remaining: int
    retry_after_seconds: int  # 0 when allowed


class RateLimiter(ABC):
    """Abstract rate limiter interface"""

    @abstractmethod
    def check(self, key: str) -> RateLimitResult:
        """Count one request against `key` and report whether it is allowed."""
        pass


class InMemoryRateLimiter(RateLimiter):
    """Single-process fixed-window limiter (local mode)."""

    def __init__(self, limit: int | None = None, window_seconds: int | None = None):
        self.limit = limit or settings.RATE_LIMIT_REQUESTS
        self.window = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS
        self._counts: dict[str, tuple[int, int]] = {}  # key -> (window_id, count)
        self._lock = threading.Lock()

    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        window_id = int(now // self.window)

        with self._lock:
            stored_window, count = self._counts.get(key, (window_id, 0))
            if stored_window != window_id:
                count = 0  # new window: reset
            count += 1
            self._counts[key] = (window_id, count)

        return self._result(count, window_id, now)

    def _result(self, count: int, window_id: int, now: float) -> RateLimitResult:
        if count > self.limit:
            window_end = (window_id + 1) * self.window
            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                window_seconds=self.window,
                remaining=0,
                retry_after_seconds=max(1, int(window_end - now) + 1),
            )
        return RateLimitResult(
            allowed=True,
            limit=self.limit,
            window_seconds=self.window,
            remaining=self.limit - count,
            retry_after_seconds=0,
        )


class RedisRateLimiter(RateLimiter):
    """Cross-instance fixed-window limiter: INCR + EXPIRE per window key."""

    def __init__(self, limit: int | None = None, window_seconds: int | None = None,
                 host: str | None = None, port: int | None = None,
                 db: int | None = None, password: str | None = None):
        import redis  # lazy import: only needed when Redis backend is used

        self.limit = limit or settings.RATE_LIMIT_REQUESTS
        self.window = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS
        self.client = redis.Redis(
            host=host or settings.REDIS_HOST,
            port=port or settings.REDIS_PORT,
            db=db or settings.REDIS_DB,
            password=password or settings.REDIS_PASSWORD,
            decode_responses=True,
        )

    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        window_id = int(now // self.window)
        redis_key = f"hsal:rate:{key}:{window_id}"

        # INCR creates the key at 1 if missing; EXPIRE bounds its lifetime.
        pipe = self.client.pipeline()
        pipe.incr(redis_key)
        pipe.expire(redis_key, self.window)
        count, _ = pipe.execute()

        if count > self.limit:
            window_end = (window_id + 1) * self.window
            return RateLimitResult(
                allowed=False,
                limit=self.limit,
                window_seconds=self.window,
                remaining=0,
                retry_after_seconds=max(1, int(window_end - now) + 1),
            )
        return RateLimitResult(
            allowed=True,
            limit=self.limit,
            window_seconds=self.window,
            remaining=self.limit - count,
            retry_after_seconds=0,
        )
