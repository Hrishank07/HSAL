from dataclasses import dataclass
from typing import Optional
from enum import Enum

class CacheSource(Enum):
    """Source of the cache response"""
    L1_EXACT = "L1_EXACT"
    L2_SEMANTIC = "L2_SEMANTIC"
    LLM_GENERATED = "LLM_GENERATED"
    MISS = "MISS"

@dataclass
class CacheRequest:
    """Request to the HSAL system"""
    prompt: str
    metadata: Optional[dict] = None

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
