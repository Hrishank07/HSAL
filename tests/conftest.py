from typing import List

import pytest

from hsal import HSALRouter, InMemoryL1Cache, MockEmbedder, MockLLM
from hsal.core.types import L2SearchResult
from hsal.services.l2_cache import L2CacheService


class FakeL2Cache(L2CacheService):
    """In-memory L2 stand-in with a controllable search result."""

    def __init__(self):
        self.entries = []
        self.searched_namespaces = []
        self.next_result = L2SearchResult(response="", similarity_score=0.0, found=False)

    def search(self, embedding: List[float], top_k: int = 1,
               namespace: str = "default") -> L2SearchResult:
        self.searched_namespaces.append(namespace)
        return self.next_result

    def add(self, prompt: str, response: str, embedding: List[float],
            namespace: str = "default") -> None:
        self.entries.append((prompt, response, embedding, namespace))


@pytest.fixture
def l1():
    return InMemoryL1Cache(max_size=100, ttl_seconds=0)


@pytest.fixture
def l2():
    return FakeL2Cache()


@pytest.fixture
def router(l1, l2):
    return HSALRouter(
        l1_cache=l1,
        l2_cache=l2,
        embedder=MockEmbedder(dimension=8),
        llm=MockLLM(),
        similarity_threshold=0.9,
        promotion_threshold=0.95,
    )
