import time
from typing import Optional
from hsal.core.types import CacheRequest, CacheResponse, CacheSource
from hsal.core.hashing import hash_prompt
from hsal.services.embedder import EmbedderService
from hsal.services.l1_cache import L1CacheService
from hsal.services.l2_cache import L2CacheService
from hsal.services.llm import LLMService
from hsal.utils.config import settings

class HSALRouter:
    """
    The Smart Router - Core orchestration logic for HSAL.
    
    Flow:
    1. Check L1 (exact match) - O(1) hash lookup
    2. If miss, check L2 (semantic match) - vector similarity
    3. If L2 hit above threshold, promote to L1
    4. If both miss, call LLM and update both caches
    """
    
    def __init__(
        self,
        l1_cache: L1CacheService,
        l2_cache: L2CacheService,
        embedder: EmbedderService,
        llm: LLMService,
        similarity_threshold: Optional[float] = None,
        promotion_threshold: Optional[float] = None
    ):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.embedder = embedder
        self.llm = llm
        
        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.promotion_threshold = promotion_threshold or settings.PROMOTION_THRESHOLD
    
    def query(self, request: CacheRequest) -> CacheResponse:
        """
        Main entry point for HSAL query.
        Implements the Smart Router algorithm.
        """
        start_time = time.time()
        prompt = request.prompt
        
        # Step 1: L1 Exact Match (Fast Path)
        hash_key = hash_prompt(prompt)
        l1_result = self.l1_cache.get(hash_key)
        
        if l1_result is not None:
            latency_ms = (time.time() - start_time) * 1000
            return CacheResponse(
                response=l1_result,
                source=CacheSource.L1_EXACT,
                latency_ms=latency_ms
            )
        
        # Step 2: L2 Semantic Match (Warm Path)
        embedding = self.embedder.embed(prompt)
        l2_result = self.l2_cache.search(embedding)
        
        if l2_result.found and l2_result.similarity_score >= self.similarity_threshold:
            # Cache hit in L2
            response = l2_result.response
            
            # Step 3: Cache Promotion (if score is high enough)
            if l2_result.similarity_score >= self.promotion_threshold:
                self.l1_cache.set(hash_key, response)
            
            latency_ms = (time.time() - start_time) * 1000
            return CacheResponse(
                response=response,
                source=CacheSource.L2_SEMANTIC,
                latency_ms=latency_ms,
                similarity_score=l2_result.similarity_score
            )
        
        # Step 4: Cold Path - LLM Generation
        response = self.llm.generate(prompt)
        
        # Update both caches
        self.l1_cache.set(hash_key, response)
        self.l2_cache.add(prompt, response, embedding)
        
        latency_ms = (time.time() - start_time) * 1000
        return CacheResponse(
            response=response,
            source=CacheSource.LLM_GENERATED,
            latency_ms=latency_ms
        )
