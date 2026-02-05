from abc import ABC, abstractmethod
from typing import Optional, List
from openai import OpenAI
from hsal.utils.config import settings

class EmbedderService(ABC):
    """Abstract embedder interface"""
    
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        pass

class OpenAIEmbedder(EmbedderService):
    """OpenAI-based embedder"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.EMBEDDING_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

class MockEmbedder(EmbedderService):
    """Mock embedder for testing (returns random vector)"""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
    
    def embed(self, text: str) -> List[float]:
        """Generate mock embedding (hash-based for consistency)"""
        import hashlib
        # Use hash to generate consistent "random" values
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        import random
        random.seed(hash_val)
        return [random.random() for _ in range(self.dimension)]

class OllamaEmbedder(EmbedderService):
    """Ollama-based embedder for local usage"""
    
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        import ollama
        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_EMBED_MODEL
        self.client = ollama.Client(host=self.host)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        response = self.client.embeddings(
            model=self.model,
            prompt=text
        )
        return response['embedding']

