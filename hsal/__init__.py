"""
HSAL - Hybrid Semantic Acceleration Layer
"""

from hsal.core.router import HSALRouter
from hsal.core.types import CacheRequest, CacheResponse, CacheSource
from hsal.services.embedder import EmbedderService, MockEmbedder, OllamaEmbedder, OpenAIEmbedder
from hsal.services.l1_cache import InMemoryL1Cache, L1CacheService, RedisL1Cache
from hsal.services.l2_cache import ChromaL2Cache, L2CacheService
from hsal.services.llm import LLMService, MockLLM, OllamaLLM, OpenAILLM

__all__ = [
    'HSALRouter',
    'CacheRequest',
    'CacheResponse',
    'CacheSource',
    'EmbedderService',
    'OpenAIEmbedder',
    'MockEmbedder',
    'L1CacheService',
    'InMemoryL1Cache',
    'RedisL1Cache',
    'L2CacheService',
    'ChromaL2Cache',
    'LLMService',
    'OpenAILLM',
    'MockLLM',
    'OllamaEmbedder',
    'OllamaLLM',
]
