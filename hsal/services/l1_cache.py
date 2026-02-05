from abc import ABC, abstractmethod
from typing import Optional
import redis
from hsal.utils.config import settings

class L1CacheService(ABC):
    """Abstract L1 cache interface"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: str) -> None:
        """Set value in cache"""
        pass

class InMemoryL1Cache(L1CacheService):
    """In-memory L1 cache (for local dev/testing)"""
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        self._cache[key] = value

class RedisL1Cache(L1CacheService):
    """Redis-based L1 cache (for production)"""
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 db: Optional[int] = None, password: Optional[str] = None):
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.db = db or settings.REDIS_DB
        self.password = password or settings.REDIS_PASSWORD
        
        self.client = redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            decode_responses=True
        )
    
    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)
    
    def set(self, key: str, value: str) -> None:
        self.client.set(key, value)
