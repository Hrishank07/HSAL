# HSAL - Hybrid Semantic Acceleration Layer

Two-tier caching for LLMs.

## Run Demo

```bash
pip install -r requirements.txt
python main.py
```

## Local Setup with Ollama

1.  **Install Ollama:** Download from [ollama.com](https://ollama.com).
2.  **Pull Models:**
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ```
3.  **Run with Ollama:** Update your configuration or pass `OllamaLLM` and `OllamaEmbedder` to the `HSALRouter`.

