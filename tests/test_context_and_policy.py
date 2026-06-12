from hsal import CacheRequest, HSALRouter, InMemoryL1Cache, MockEmbedder, MockLLM
from hsal.core.hashing import context_fingerprint, hash_prompt, normalize
from hsal.core.types import CacheSource, L2SearchResult

from .conftest import FakeL2Cache


def make_router(l1, l2, context=None):
    return HSALRouter(
        l1_cache=l1,
        l2_cache=l2,
        embedder=MockEmbedder(dimension=8),
        llm=MockLLM(),
        context=context,
    )


# --- context fingerprint ---

def test_fingerprint_is_deterministic_and_order_insensitive():
    a = context_fingerprint({"model": "llama3.2", "temperature": 0.7})
    b = context_fingerprint({"temperature": 0.7, "model": "llama3.2"})
    assert a == b


def test_fingerprint_differs_when_context_differs():
    a = context_fingerprint({"model": "llama3.2"})
    b = context_fingerprint({"model": "llama3.1"})
    assert a != b


def test_no_context_uses_default_namespace():
    assert context_fingerprint(None) == "default"
    assert context_fingerprint({}) == "default"


def test_same_prompt_different_context_yields_different_l1_keys():
    key_a = hash_prompt("Summarize this.", {"system_prompt": "You are a lawyer"})
    key_b = hash_prompt("Summarize this.", {"system_prompt": "You are a pirate"})
    assert key_a != key_b


def test_same_prompt_different_model_does_not_hit_l1():
    """The core safety property: a cached answer from one configuration
    must never be served under another."""
    l1 = InMemoryL1Cache(max_size=10, ttl_seconds=0)

    router_a = make_router(l1, FakeL2Cache(), context={"model": "llama3.2"})
    router_b = make_router(l1, FakeL2Cache(), context={"model": "gpt-4"})

    router_a.query(CacheRequest(prompt="Summarize this."))
    result = router_b.query(CacheRequest(prompt="Summarize this."))

    assert result.source == CacheSource.LLM_GENERATED  # not L1_EXACT


def test_l2_reads_and_writes_are_namespaced():
    l2 = FakeL2Cache()
    router = make_router(InMemoryL1Cache(max_size=10, ttl_seconds=0), l2,
                         context={"model": "llama3.2"})
    ns = context_fingerprint({"model": "llama3.2"})

    router.query(CacheRequest(prompt="q"))

    assert l2.searched_namespaces == [ns]
    assert l2.entries[0][3] == ns


# --- cacheable flag ---

def test_non_cacheable_request_bypasses_cache_read(router, l2):
    l2.next_result = L2SearchResult(response="cached", similarity_score=0.99, found=True)

    result = router.query(CacheRequest(prompt="what is MY balance?", cacheable=False))

    assert result.source == CacheSource.LLM_GENERATED
    assert result.response != "cached"
    assert l2.searched_namespaces == []  # L2 never consulted


def test_non_cacheable_request_is_not_stored(router, l1, l2):
    router.query(CacheRequest(prompt="delete user 123", cacheable=False))

    assert len(l1) == 0
    assert l2.entries == []

    # identical prompt later still goes to the LLM
    result = router.query(CacheRequest(prompt="delete user 123", cacheable=False))
    assert result.source == CacheSource.LLM_GENERATED


# --- normalization policy ---

def test_lowercase_normalization_is_optional():
    assert normalize("SELECT * FROM Users", lowercase=False) == "SELECT * FROM Users"
    assert normalize("SELECT * FROM Users", lowercase=True) == "select * from users"


def test_case_sensitive_prompts_get_distinct_keys_when_lowercase_disabled():
    key_upper = hash_prompt("SELECT * FROM Users", lowercase=False)
    key_lower = hash_prompt("select * from users", lowercase=False)
    assert key_upper != key_lower
