import threading
import time
from typing import Optional

from hsal.core.types import CacheRequest, CacheResponse, CacheSource
from hsal.core.hashing import hash_prompt, context_fingerprint
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
        promotion_threshold: Optional[float] = None,
        context: Optional[dict] = None
    ):
        """
        context: everything that affects generation output beyond the user
        prompt — model id/version, system prompt, temperature, tool schema,
        tenant id, etc. It is fingerprinted into every cache key (L1) and
        namespace (L2), so answers are never served across configurations.
        """
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.embedder = embedder
        self.llm = llm

        self.similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        self.promotion_threshold = promotion_threshold or settings.PROMOTION_THRESHOLD

        self.context = context
        self._namespace = context_fingerprint(context)

        self._stats_lock = threading.Lock()
        self._counts = {source: 0 for source in CacheSource}
        self._latency_ms = {source: 0.0 for source in CacheSource}

    def query(self, request: CacheRequest) -> CacheResponse:
        """
        Main entry point for HSAL query.
        Implements the Smart Router algorithm.
        """
        start_time = time.time()
        prompt = request.prompt

        # Step 0: Non-cacheable requests bypass caches entirely
        # (no read: a cached answer may be wrong for this request;
        #  no write: its answer must not be served to anyone else)
        if not request.cacheable:
            response = self.llm.generate(prompt)
            return self._respond(response, CacheSource.LLM_GENERATED, start_time)

        # Step 1: L1 Exact Match (Fast Path)
        hash_key = hash_prompt(prompt, self.context, lowercase=settings.L1_LOWERCASE)
        l1_result = self.l1_cache.get(hash_key)

        if l1_result is not None:
            return self._respond(l1_result, CacheSource.L1_EXACT, start_time)

        # Step 2: L2 Semantic Match (Warm Path)
        embedding = self.embedder.embed(prompt)
        l2_result = self.l2_cache.search(embedding, namespace=self._namespace)

        if l2_result.found and l2_result.similarity_score >= self.similarity_threshold:
            # Step 3: Cache Promotion (if score is high enough)
            if l2_result.similarity_score >= self.promotion_threshold:
                self.l1_cache.set(hash_key, l2_result.response)

            return self._respond(
                l2_result.response, CacheSource.L2_SEMANTIC, start_time,
                similarity_score=l2_result.similarity_score
            )

        # Step 4: Cold Path - LLM Generation
        response = self.llm.generate(prompt)

        # Update both caches
        self.l1_cache.set(hash_key, response)
        self.l2_cache.add(prompt, response, embedding, namespace=self._namespace)

        return self._respond(response, CacheSource.LLM_GENERATED, start_time)

    def _respond(
        self,
        response: str,
        source: CacheSource,
        start_time: float,
        similarity_score: Optional[float] = None
    ) -> CacheResponse:
        latency_ms = (time.time() - start_time) * 1000
        with self._stats_lock:
            self._counts[source] += 1
            self._latency_ms[source] += latency_ms
        return CacheResponse(
            response=response,
            source=source,
            latency_ms=latency_ms,
            similarity_score=similarity_score
        )

    def stats(self) -> dict:
        """Hit counts, hit rate, and average latency per tier."""
        with self._stats_lock:
            counts = dict(self._counts)
            latency = dict(self._latency_ms)

        total = sum(counts.values())
        cache_hits = counts[CacheSource.L1_EXACT] + counts[CacheSource.L2_SEMANTIC]
        return {
            "total_queries": total,
            "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
            "by_source": {
                source.value: {
                    "count": counts[source],
                    "avg_latency_ms": round(latency[source] / counts[source], 2)
                    if counts[source] else 0.0
                }
                for source in CacheSource
            }
        }
