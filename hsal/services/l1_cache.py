import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from hsal.utils.config import settings


class L1CacheService(ABC):
    """Abstract L1 cache interface"""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Get value from cache"""
        pass

    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set value in cache"""
        pass


class InMemoryL1Cache(L1CacheService):
    """
    In-memory L1 cache with LRU eviction and optional TTL.

    - max_size bounds memory usage (oldest entries evicted first).
    - ttl_seconds expires stale entries; 0 disables expiry.
    """

    def __init__(self, max_size: int | None = None, ttl_seconds: int | None = None):
        self.max_size = max_size or settings.L1_MAX_SIZE
        self.ttl = settings.L1_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        self._cache: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def get(self, key: str) -> str | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self.ttl and time.time() > expires_at:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)  # mark as recently used
        return value

    def set(self, key: str, value: str) -> None:
        expires_at = time.time() + self.ttl if self.ttl else float("inf")
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (expires_at, value)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)  # evict least recently used

    def __len__(self) -> int:
        return len(self._cache)


class RedisL1Cache(L1CacheService):
    """Redis-based L1 cache (for production / cross-instance sharing)"""

    def __init__(self, host: str | None = None, port: int | None = None,
                 db: int | None = None, password: str | None = None,
                 ttl_seconds: int | None = None):
        import redis  # lazy import: only needed when Redis backend is used

        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db or settings.REDIS_DB
        self.password = password or settings.REDIS_PASSWORD
        self.ttl = settings.L1_TTL_SECONDS if ttl_seconds is None else ttl_seconds

        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )

    def get(self, key: str) -> str | None:
        return self.client.get(key)

    def set(self, key: str, value: str) -> None:
        self.client.set(key, value, ex=self.ttl or None)
