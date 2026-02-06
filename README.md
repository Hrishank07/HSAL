# Hybrid Semantic Acceleration Layer (HSAL) 🗳️

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-orange.svg)](https://ollama.com/)

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

By capturing L1 hits before generating embeddings, HSAL can reduce embedding infrastructure load by **30%-60%** in high-repeat environments.

---

## 3. How It Works? (The Logic)

The heart of HSAL is the **Smart Router**. It orchestrates every request through a precise selection flow.

### 3.1 Pre-processing: Semantic Normalization
Before hashing, prompts undergo normalization:
- Trimming whitespace.
- Lowercasing (optional).
- Removing redundant line breaks.
*This ensures ` "Hello world"` and `"hello world  "` map to the same L1 record.*

### 3.2 The Decision Flow
1.  **L1 FAST PATH**: Compute a SHA-256 hash of the normalized prompt. Check the L1 store. If hit, return immediately.
2.  **L2 WARM PATH**: If L1 misses, generate a vector embedding. Query the L2 Vector DB.
    - If a result is found above `SIMILARITY_THRESHOLD` (e.g., 0.85), return it.
3.  **CACHE PROMOTION**: If an L2 hit is exceptionally strong (e.g., > 0.95), HSAL "promotes" it by writing the exact prompt's hash into L1. Future identical requests will now hit the Fast Path directly.
4.  **COLD PATH**: If both fail, trigger the LLM. Once generated, update both L1 and L2 for future queries.

### 3.3 Circuit Breaker Logic
To ensure stability, HSAL monitors the health of the Embedder and Cache services. If a service experiences persistent failures, HSAL **fails open**, routing traffic directly to the LLM to prevent application downtime.

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

---

## 5. Project Roadmap
- **Async Write-Through**: Moving L2 writes to background tasks.
- **Hybrid L1**: Cross-instance L1 using a shared Redis instance.
- **Adaptive Thresholds**: Machine-learning-based threshold adjustment.

---

*Note: This Project is intentionally designed as an infrastructure-level optimization, providing immediate performance and cost benefits while remaining transparent to downstream consumers.*
