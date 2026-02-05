from abc import ABC, abstractmethod
from typing import List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from hsal.core.types import L2SearchResult
from hsal.utils.config import settings

class L2CacheService(ABC):
    """Abstract L2 cache interface"""
    
    @abstractmethod
    def search(self, embedding: List[float], top_k: int = 1) -> L2SearchResult:
        """Search for similar embeddings"""
        pass
    
    @abstractmethod
    def add(self, prompt: str, response: str, embedding: List[float]) -> None:
        """Add entry to vector cache"""
        pass

class ChromaL2Cache(L2CacheService):
    """ChromaDB-based L2 cache"""
    
    def __init__(self, path: Optional[str] = None, collection_name: Optional[str] = None):
        self.path = path or settings.CHROMA_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def search(self, embedding: List[float], top_k: int = 1) -> L2SearchResult:
        """Search for similar embeddings using cosine similarity"""
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k
        )
        
        # Check if we have results
        if not results['documents'] or not results['documents'][0]:
            return L2SearchResult(response="", similarity_score=0.0, found=False)
        
        # Get top result
        response = results['metadatas'][0][0]['response']
        # ChromaDB returns distance, convert to similarity (1 - distance for cosine)
        distance = results['distances'][0][0]
        similarity = 1.0 - distance
        
        return L2SearchResult(
            response=response,
            similarity_score=similarity,
            found=True
        )
    
    def add(self, prompt: str, response: str, embedding: List[float]) -> None:
        """Add entry to vector database"""
        import uuid
        doc_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[prompt],
            metadatas=[{"response": response}]
        )
