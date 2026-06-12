# Hybrid Semantic Acceleration Layer (HSAL) 🗳️

[![CI](https://github.com/Hrishank07/HSAL/actions/workflows/ci.yml/badge.svg)](https://github.com/Hrishank07/HSAL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange.svg)](https://ollama.com/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## 1. What is HSAL? (The Concept)

**HSAL** is a high-performance orchestration layer designed to optimize LLM request pipelines. It addresses a fundamental inefficiency in modern AI architectures: the **Vector Processing Tax**.

### The Problem: The Vector Tax
Modern semantic caches rely on Vector Databases (Chroma, Pinecone, etc.) for everything. While great for semantic similarity, they are overkill for exact matches. Forcing an identical repeat query through embedding generation and vector search is:
1.  **Expensive**: Consumes GPU/CPU cycles and API credits.
2.  **Slow**: Adds dozens of milliseconds of unnecessary latency.
3.  **Redundant**: The logic is probabilistic where it could be deterministic.

### The Solution: Two-Tier Deterministic Caching
HSAL introduces a disciplined, multi-path retrieval strategy:
- **L1 (Fast Path)**: Sub-millisecond, O(1) hash-based lookup for identical queries.
- **L2 (Warm Path)**: Semantic similarity search for paraphrased or fuzzy matches.

---

## 2. Why HSAL? (The Rationale)

We built HSAL because production LLM workloads often follow a Power Law distribution—a small percentage of prompts (instructions, standard greetings, repetitive tasks) make up a large percentage of traffic.

| Tier | Method | Latency | Logic | Benefit |
| :--- | :--- | :--- | :--- | :--- |
| **L1** | Hash Map (Redis/RAM) | **<1 ms** | Deterministic | Zero compute cost, near-zero latency |
| **L2** | Vector DB (Chroma) | **10-30 ms** | Probabilistic | Handles paraphrasing & intent |
| **LLM** | Generation (Ollama) | **~2000 ms** | Generative | Full inference cost |

Savings depend entirely on how repetitive your traffic is. Measured with the included benchmark (`python benchmarks/run_benchmark.py`, 5,000 requests over a 500-prompt pool, mock backends, L1 tier only):

| Workload | L1 Hit Rate | Embedding Calls Saved | LLM Calls Saved |
| :--- | :--- | :--- | :--- |
| Zipfian (power-law repeats) | 90.9% | 90.9% | 90.9% |
| Uniform over fixed pool | 90.0% | 90.0% | 90.0% |
| Every prompt unique | 0% | 0% | 0% |

The honest summary: HSAL saves roughly your traffic's repetition rate, and saves nothing on one-off prompts. Run the benchmark with your own pool/request ratio before assuming numbers.

### HSAL vs. Provider Prompt Caching
OpenAI and Anthropic both offer provider-side **prompt caching**, which caches repeated *input prefixes* (system prompts, long documents) to cut input-token cost and time-to-first-token. That is complementary, not equivalent: prefix caching still runs generation and bills output tokens every time. HSAL is **response caching** — a hit skips generation entirely. Prefix caching makes repeated *contexts* cheaper; HSAL makes repeated *questions* free.

---

## 3. How It Works? (The Logic)

The heart of HSAL is the **Smart Router**. It orchestrates every request through a precise selection flow.

### 3.1 The Cache Key: Prompt + Context Fingerprint
The user prompt alone is **not** a safe cache key. "Summarize this." produces different answers under a different model, system prompt, temperature, or tenant — hashing only the visible prompt would serve those answers interchangeably.

HSAL therefore scopes every cache entry by a **context fingerprint**: a hash of everything that affects generation output (`model`, `system_prompt`, `temperature`, tool schema, tenant id — whatever you pass as the router's `context` dict). The fingerprint is mixed into every L1 key and namespaces every L2 entry, so cached answers are never served across configurations.

```python
router = HSALRouter(l1, l2, embedder, llm, context={
    "model": "llama3.2",
    "system_prompt": SYSTEM_PROMPT,
    "temperature": 0.7,
})
```

Before hashing, prompts are also normalized: whitespace trimmed and collapsed, and lowercased by default. Set `L1_LOWERCASE=false` for case-sensitive workloads (code, SQL, identifiers) where `SELECT * FROM Users` and `select * from users` must not collide.

### 3.2 The Decision Flow
1.  **L1 FAST PATH**: Compute a SHA-256 hash of the normalized prompt. Check the L1 store. If hit, return immediately.
2.  **L2 WARM PATH**: If L1 misses, generate a vector embedding. Query the L2 Vector DB.
    - If a result is found above `SIMILARITY_THRESHOLD` (e.g., 0.85), return it.
3.  **CACHE PROMOTION**: If an L2 hit is exceptionally strong (e.g., > 0.95), HSAL "promotes" it by writing the exact prompt's hash into L1. Future identical requests will now hit the Fast Path directly.
4.  **COLD PATH**: If both fail, trigger the LLM. Once generated, update both L1 and L2 for future queries.

### 3.3 Cache Policy (TTL & LRU)
Cached LLM answers go stale and unbounded caches leak memory, so the L1 cache enforces:
- **LRU eviction**: capped at `L1_MAX_SIZE` entries (default 10,000); least recently used entries are evicted first.
- **TTL expiry**: entries expire after `L1_TTL_SECONDS` (default 1 hour); set to `0` to disable.

The Redis backend applies the same TTL via native key expiry.

### 3.4 Known Trade-off: Semantic False Positives
A semantic cache can return a *wrong* answer for a *similar* question. These pairs can all clear a 0.9 cosine threshold while requiring different answers:

- "Can I refund this order?" vs. "Can I refund this order after 30 days?"
- "Delete user 123" vs. "Delete user 124"
- "Summarize Q1 revenue" vs. "Summarize Q2 revenue"
- "Is this medication safe for adults?" vs. "...for children?"

`SIMILARITY_THRESHOLD` is deliberately configurable per deployment: raise it (0.95+) for factual or precision-sensitive workloads, lower it for FAQ-style traffic where paraphrase tolerance matters more than exactness.

### 3.5 What Should Never Be Cached
Some requests are wrong to cache at *any* threshold. Pass `cacheable: false` to bypass both cache reads and writes for:

- **User-specific data** — "What is *my* account balance?"
- **Time-sensitive questions** — anything whose answer changes between requests
- **Tool/action requests** — "Delete user 123" must execute, not replay
- **PII or private retrieved documents** — cached answers leak across users
- **Legal/medical/financial advice** — staleness has real consequences

Good candidates for caching: generic explanations, public FAQs, deterministic formatting, stable documentation questions.

---

## 4. Local Setup (Ollama)

HSAL is optimized for local development using **Ollama**.

### Prerequisites
1.  **Ollama**: [Download here](https://ollama.com/)
2.  **Models**:
    ```bash
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ```

### Installation
```bash
# Clone and setup
git clone https://github.com/Hrishank07/HSAL.git
cd HSAL
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the System
- **Demo CLI**: `python main.py`
- **FastAPI Server**: `uvicorn app:app --reload`

### Docker (one command, no local Python or Ollama needed)
```bash
docker compose up --build
```
This starts Ollama, pulls `llama3.2` + `nomic-embed-text` on first run, and serves the HSAL API on `localhost:8000`. Models and the Chroma store persist in named volumes.

### API Endpoints
| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/query` | POST | Run a prompt through the L1 → L2 → LLM pipeline |
| `/stats` | GET | Human-readable summary: hit rates, avg latency per tier |
| `/metrics` | GET | Prometheus exposition: request counters, latency histograms |
| `/health` | GET | Liveness check |

Every response carries decision metadata, and each request emits one structured JSON log line — so you can always answer *why* a response came from where it did:

```json
{
  "response": "Python is...",
  "metadata": {
    "request_id": "req_a1b2c3d4e5f6",
    "path": "L1_EXACT",
    "cacheable": true,
    "latency_ms": 0.42,
    "similarity_score": null
  }
}
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"prompt": "What is Python?"}'

# user-specific request: bypass the cache entirely
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
     -d '{"prompt": "What is my account balance?", "cacheable": false}'

curl localhost:8000/stats
```

### Demo Scenarios
Each of these is directly reproducible against the running server and covered by the test suite:

| # | Scenario | Expected Behavior |
| :--- | :--- | :--- |
| 1 | Same prompt twice | 1st: `LLM_GENERATED` (~2s) → 2nd: `L1_EXACT` (<1ms) |
| 2 | Whitespace/case variant of a cached prompt | `L1_EXACT` — normalization maps both to one key |
| 3 | Paraphrase of a cached prompt | `L2_SEMANTIC` with similarity score; promoted to L1 if ≥ 0.95 |
| 4 | `"cacheable": false` request | Always `LLM_GENERATED`; nothing read from or written to either cache |
| 5 | Same prompt, different `context` (model/system prompt) | Cache miss — context fingerprint isolates configurations |
| 6 | Unique prompts only | 0% hit rate — caching honestly buys nothing here |

---

## 5. Testing & CI

The router, hashing, cache, and observability logic are unit-tested against mock backends — no Ollama or ChromaDB required:

```bash
pip install -r requirements-dev.txt
pytest          # 36 tests
ruff check .    # lint
```

CI (GitHub Actions) runs lint, the test suite across Python 3.10–3.12, and a benchmark smoke run on every push and pull request.

---

## 6. Project Roadmap
- **Configurable Failure Policy**: When the embedder or vector DB is unhealthy — fail-open (availability-critical), fail-closed (budget-sensitive; unguarded fail-open can turn a cache outage into an LLM cost explosion), degrade to a cheaper model, or rate-limited fallback.
- **Async Write-Through**: Moving L2 writes to background tasks.
- **Hybrid L1**: Cross-instance L1 using a shared Redis instance.
- **L2 TTL/Eviction**: Age out stale vector entries.
- **Adaptive Thresholds**: Machine-learning-based threshold adjustment.

---

*Note: This Project is intentionally designed as an infrastructure-level optimization, providing immediate performance and cost benefits while remaining transparent to downstream consumers.*
