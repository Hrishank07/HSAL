from hsal import CacheRequest
from hsal.core.types import CacheSource, L2SearchResult


def test_cold_path_generates_and_populates_both_caches(router, l1, l2):
    result = router.query(CacheRequest(prompt="What is Python?"))

    assert result.source == CacheSource.LLM_GENERATED
    assert result.response == "Mock response for: What is Python?"
    assert len(l1) == 1                # written to L1
    assert len(l2.entries) == 1        # written to L2
    assert l2.entries[0][0] == "What is Python?"


def test_identical_prompt_hits_l1(router):
    first = router.query(CacheRequest(prompt="What is Python?"))
    second = router.query(CacheRequest(prompt="What is Python?"))

    assert second.source == CacheSource.L1_EXACT
    assert second.response == first.response


def test_normalized_variant_hits_l1(router):
    router.query(CacheRequest(prompt="What is Python?"))
    result = router.query(CacheRequest(prompt="  what is   PYTHON? "))

    assert result.source == CacheSource.L1_EXACT


def test_l2_hit_above_similarity_threshold(router, l2):
    l2.next_result = L2SearchResult(response="cached", similarity_score=0.92, found=True)

    result = router.query(CacheRequest(prompt="some paraphrase"))

    assert result.source == CacheSource.L2_SEMANTIC
    assert result.response == "cached"
    assert result.similarity_score == 0.92


def test_l2_hit_below_promotion_threshold_is_not_promoted(router, l1, l2):
    l2.next_result = L2SearchResult(response="cached", similarity_score=0.92, found=True)

    router.query(CacheRequest(prompt="some paraphrase"))

    assert len(l1) == 0  # 0.92 < promotion threshold 0.95


def test_strong_l2_hit_is_promoted_to_l1(router, l1, l2):
    l2.next_result = L2SearchResult(response="cached", similarity_score=0.97, found=True)

    router.query(CacheRequest(prompt="some paraphrase"))
    assert len(l1) == 1

    # identical prompt now bypasses L2 entirely
    l2.next_result = L2SearchResult(response="", similarity_score=0.0, found=False)
    result = router.query(CacheRequest(prompt="some paraphrase"))
    assert result.source == CacheSource.L1_EXACT
    assert result.response == "cached"


def test_l2_below_similarity_threshold_falls_through_to_llm(router, l2):
    l2.next_result = L2SearchResult(response="wrong answer", similarity_score=0.5, found=True)

    result = router.query(CacheRequest(prompt="unrelated question"))

    assert result.source == CacheSource.LLM_GENERATED
    assert result.response != "wrong answer"


def test_stats_track_counts_and_hit_rate(router):
    router.query(CacheRequest(prompt="q1"))   # cold
    router.query(CacheRequest(prompt="q1"))   # L1 hit
    router.query(CacheRequest(prompt="q2"))   # cold

    stats = router.stats()
    assert stats["total_queries"] == 3
    assert stats["by_source"]["L1_EXACT"]["count"] == 1
    assert stats["by_source"]["LLM_GENERATED"]["count"] == 2
    assert stats["cache_hit_rate"] == round(1 / 3, 4)


def test_latency_is_reported(router):
    result = router.query(CacheRequest(prompt="q"))
    assert result.latency_ms >= 0
