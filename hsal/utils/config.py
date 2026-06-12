
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # App Config
    APP_NAME: str = "HSAL"
    DEBUG: bool = False

    # L1 Cache (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    # L2 Cache (ChromaDB)
    CHROMA_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "hsal_cache"

    # Embeddings & LLM
    OPENAI_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-3.5-turbo"

    # Ollama Config
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    OLLAMA_LLM_MODEL: str = "llama3.2"

    # HSAL Logic
    SIMILARITY_THRESHOLD: float = 0.9  # Threshold for L2 hit
    PROMOTION_THRESHOLD: float = 0.95  # Threshold to promote L2 hit to L1

    # L1 Cache Policy
    L1_MAX_SIZE: int = 10_000   # Max entries for in-memory L1 (LRU eviction)
    L1_TTL_SECONDS: int = 3600  # Entry TTL; 0 disables expiry
    L1_LOWERCASE: bool = True   # Disable for case-sensitive workloads (code, SQL, identifiers)

settings = Settings()
