from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import Optional
from hsal import (
    HSALRouter,
    CacheRequest,
    InMemoryL1Cache,
    ChromaL2Cache,
    OllamaEmbedder,
    OllamaLLM,
)
from hsal.observability import metrics_response, new_request_id, record_request

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
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    response: str
    metadata: QueryMetadata


# Note: deliberately a sync endpoint. router.query() is blocking (embedding +
# LLM calls), so FastAPI runs it in a threadpool instead of blocking the event
# loop, which an `async def` here would do.
@app.post("/query", response_model=QueryResponse)
def query_hsal(request: QueryRequest):
    request_id = new_request_id()
    try:
        result = router.query(
            CacheRequest(prompt=request.prompt, cacheable=request.cacheable)
        )
    except Exception as e:
        record_request(request_id, "ERROR", request.cacheable, 0.0, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

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
