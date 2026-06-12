from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CacheSource(Enum):
    """Source of the cache response"""
    L1_EXACT = "L1_EXACT"
    L2_SEMANTIC = "L2_SEMANTIC"
    LLM_GENERATED = "LLM_GENERATED"

@dataclass
class CacheRequest:
    """
    Request to the HSAL system.

    cacheable=False bypasses both cache reads and writes — use for
    user-specific, time-sensitive, or PII-bearing prompts where a
    cached answer (or caching the answer) would be wrong.
    """
    prompt: str
    metadata: Optional[dict] = None
    cacheable: bool = True

@dataclass
class CacheResponse:
    """Response from the HSAL system"""
    response: str
    source: CacheSource
    latency_ms: float
    similarity_score: Optional[float] = None  # Only for L2 hits
    
@dataclass
class L2SearchResult:
    """Result from L2 vector search"""
    response: str
    similarity_score: float
    found: bool = True
