# Hybrid Semantic Acceleration Layer (HSAL) 🗳️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange.svg)](https://ollama.com/)

## 1. Executive Summary (The Pitch)

Most modern LLM caching strategies rely almost entirely on **Vector Databases** (e.g., Chroma, Pinecone) to capture semantic similarity. While effective for paraphrased queries, this approach imposes a significant and unnecessary **tax on exact matches**.

In current systems, **every request—including identical repeats—is forced through embedding generation and vector search**. This introduces avoidable latency, increases CPU/GPU utilization, and drives up costs. Identical prompts are common, yet systems repeatedly pay the full semantic-processing cost for them.

**HSAL** eliminates this inefficiency by splitting request handling into two deterministic paths:
- **L1: Exact-Match Cache**: A hash-based lookup for identical prompts, returning responses in **<1ms**.
- **L2: Semantic Cache**: A vector database used only when an exact match is unavailable.

---

## 2. The Implementation

### 2.1 The "Smart Router" Algorithm
HSAL orchestrates queries through a tiered transition:

1.  **L1 FAST PATH (Exact Match)**: O(1) SHA-256 hash lookup.
2.  **L2 WARM PATH (Semantic Match)**: Embedding generation + Vector Search.
    - **Cache Promotion**: If a semantic match is found above a high threshold (e.g., 0.95), it is "promoted" to L1 for future sub-millisecond retrieval.
3.  **COLD PATH (LLM Generation)**: Full generation via Ollama/OpenAI, then population of both cache layers.

### 2.2 Performance Goals
- **L1 (Exact)**: <1 ms latency.
- **L2 (Semantic)**: 10–30 ms latency.
- **LLM (Cold)**: ~2000 ms (depends on model).
- **Target**: ≥30% reduction in embedding calls.

---

## 3. Project Structure

```text
hsal/
├── core/           # Routing & Hashing logic
├── services/       # Cache, Embedder, and LLM adapters
└── utils/          # Pydantic Configuration
main.py             # Interactive CLI Demo
app.py              # FastAPI Service
```

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
# Clone the repository
git clone https://github.com/Hrishank07/HSAL.git
cd HSAL

# Setup Virtual Environment
python -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### Running the Demo
```bash
python main.py
```

### Running the API
```bash
uvicorn app:app --reload
```

---

## 5. Notes & Future Roadmap
- **Infrastructure-Level Optimization**: Requires zero changes to application logic.
- **Normalization**: Prompts are normalized before hashing to increase hit rates (whitespace/casing).
- **Circuit Breaker**: HSAL fails open to the LLM if the embedder or cache layers experience high failure rates.
- **Async Support**: Production deployments should move L2 writes to an async queue.

---

*Note: This Project is intentionally designed as an infrastructure-level optimization, providing immediate performance and cost benefits while remaining transparent to downstream consumers.*
