from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from hsal import (
    HSALRouter,
    CacheRequest,
    InMemoryL1Cache,
    ChromaL2Cache,
    OllamaEmbedder,
    OllamaLLM,
    CacheSource
)

app = FastAPI(title="HSAL API - Local Semantic Cache")

# Global router instance
# For production, you'd use RedisL1Cache and potentially a proper DI container
l1 = InMemoryL1Cache()
l2 = ChromaL2Cache()
embedder = OllamaEmbedder()
llm = OllamaLLM()
router = HSALRouter(l1, l2, embedder, llm)

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    response: str
    source: str
    latency_ms: float
    similarity_score: Optional[float] = None

@app.post("/query", response_model=QueryResponse)
async def query_hsal(request: QueryRequest):
    try:
        result = router.query(CacheRequest(prompt=request.prompt))
        return QueryResponse(
            response=result.response,
            source=result.source.value,
            latency_ms=result.latency_ms,
            similarity_score=result.similarity_score
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "provider": "ollama"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
