import pytest

from hsal.guards import InMemoryRateLimiter


@pytest.fixture
def clock(monkeypatch):
    """Controllable time for hsal.guards.rate_limiter."""
    import hsal.guards.rate_limiter as rl_mod

    now = [1000.0]
    monkeypatch.setattr(rl_mod.time, "time", lambda: now[0])
    return now


def test_allows_up_to_limit(clock):
    limiter = InMemoryRateLimiter(limit=3, window_seconds=60)
    for expected_remaining in (2, 1, 0):
        result = limiter.check("client_a")
        assert result.allowed
        assert result.remaining == expected_remaining


def test_blocks_over_limit_with_retry_after(clock):
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60)
    limiter.check("client_a")
    limiter.check("client_a")

    result = limiter.check("client_a")
    assert not result.allowed
    assert result.remaining == 0
    assert result.retry_after_seconds >= 1
    # window started at t=960 (1000 // 60 * 60), ends t=1020 -> ~20s + grace
    assert result.retry_after_seconds <= 60


def test_window_rollover_resets_count(clock):
    limiter = InMemoryRateLimiter(limit=1, window_seconds=60)
    assert limiter.check("client_a").allowed
    assert not limiter.check("client_a").allowed

    clock[0] += 60  # next window
    assert limiter.check("client_a").allowed


def test_keys_are_isolated(clock):
    limiter = InMemoryRateLimiter(limit=1, window_seconds=60)
    assert limiter.check("client_a").allowed
    assert not limiter.check("client_a").allowed
    assert limiter.check("client_b").allowed  # unaffected by client_a


def test_result_carries_policy_for_429_body(clock):
    limiter = InMemoryRateLimiter(limit=5, window_seconds=30)
    result = limiter.check("client_a")
    assert result.limit == 5
    assert result.window_seconds == 30
