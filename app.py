
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hsal import (
    CacheRequest,
    ChromaL2Cache,
    HSALRouter,
    InMemoryL1Cache,
    OllamaEmbedder,
    OllamaLLM,
)
from hsal.guards import InMemoryRateLimiter
from hsal.observability import (
    RATE_LIMITED_TOTAL,
    metrics_response,
    new_request_id,
    record_request,
)
from hsal.utils.config import settings

app = FastAPI(title="HSAL API - Local Semantic Cache")

# Global router instance
# For production, you'd use RedisL1Cache and potentially a proper DI container
l1 = InMemoryL1Cache()
l2 = ChromaL2Cache()
embedder = OllamaEmbedder()
llm = OllamaLLM()

# Everything that affects generation output belongs in the context:
# cached answers are only served within this exact configuration.
router = HSALRouter(l1, l2, embedder, llm, context={
    "provider": "ollama",
    "model": llm.model,
    "embed_model": embedder.model,
})

# Fixed-window limiter, keyed by client IP. Swap in RedisRateLimiter for
# multi-instance deployments. Enable via RATE_LIMIT_ENABLED=true.
rate_limiter = InMemoryRateLimiter() if settings.RATE_LIMIT_ENABLED else None


class QueryRequest(BaseModel):
    prompt: str
    # Set false for user-specific, time-sensitive, or PII-bearing prompts:
    # bypasses cache reads AND writes.
    cacheable: bool = True


class QueryMetadata(BaseModel):
    request_id: str
    path: str  # L1_EXACT | L2_SEMANTIC | LLM_GENERATED
    cacheable: bool
    latency_ms: float
    similarity_score: float | None = None


class QueryResponse(BaseModel):
    response: str
    metadata: QueryMetadata


# Note: deliberately a sync endpoint. router.query() is blocking (embedding +
# LLM calls), so FastAPI runs it in a threadpool instead of blocking the event
# loop, which an `async def` here would do.
@app.post("/query", response_model=QueryResponse)
def query_hsal(request: QueryRequest, http_request: Request):
    request_id = new_request_id()

    # Rate limit BEFORE any embedding/LLM work: the limiter exists to
    # protect compute, so it must run in front of the expensive path.
    if rate_limiter is not None:
        client_key = http_request.client.host if http_request.client else "unknown"
        verdict = rate_limiter.check(client_key)
        if not verdict.allowed:
            RATE_LIMITED_TOTAL.inc()
            record_request(request_id, "RATE_LIMITED", request.cacheable, 0.0)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(verdict.retry_after_seconds)},
                content={
                    "error": "rate_limited",
                    "limit": verdict.limit,
                    "window_seconds": verdict.window_seconds,
                    "retry_after_seconds": verdict.retry_after_seconds,
                    "request_id": request_id,
                },
            )

    try:
        result = router.query(
            CacheRequest(prompt=request.prompt, cacheable=request.cacheable)
        )
    except Exception as e:
        record_request(request_id, "ERROR", request.cacheable, 0.0, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

    record_request(
        request_id,
        result.source.value,
        request.cacheable,
        result.latency_ms,
        result.similarity_score,
    )
    return QueryResponse(
        response=result.response,
        metadata=QueryMetadata(
            request_id=request_id,
            path=result.source.value,
            cacheable=request.cacheable,
            latency_ms=round(result.latency_ms, 2),
            similarity_score=result.similarity_score,
        ),
    )


@app.get("/stats")
def stats():
    """Human-readable summary: hit rates and average latency per tier."""
    return router.stats()


@app.get("/metrics")
def metrics():
    """Prometheus exposition format."""
    payload, content_type = metrics_response()
    return Response(content=payload, media_type=content_type)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "provider": "ollama"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
