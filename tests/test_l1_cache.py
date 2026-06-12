from hsal import InMemoryL1Cache


def test_get_set_roundtrip():
    cache = InMemoryL1Cache(max_size=10, ttl_seconds=0)
    cache.set("k", "v")
    assert cache.get("k") == "v"


def test_miss_returns_none():
    cache = InMemoryL1Cache(max_size=10, ttl_seconds=0)
    assert cache.get("missing") is None


def test_lru_eviction_drops_least_recently_used():
    cache = InMemoryL1Cache(max_size=2, ttl_seconds=0)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.get("a")        # touch "a" so "b" is now LRU
    cache.set("c", "3")   # evicts "b"
    assert cache.get("a") == "1"
    assert cache.get("b") is None
    assert cache.get("c") == "3"


def test_max_size_is_enforced():
    cache = InMemoryL1Cache(max_size=3, ttl_seconds=0)
    for i in range(10):
        cache.set(f"k{i}", str(i))
    assert len(cache) == 3


def test_ttl_expires_entries(monkeypatch):
    import hsal.services.l1_cache as l1_mod

    now = [1000.0]
    monkeypatch.setattr(l1_mod.time, "time", lambda: now[0])

    cache = InMemoryL1Cache(max_size=10, ttl_seconds=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"

    now[0] += 61  # advance past TTL
    assert cache.get("k") is None


def test_ttl_zero_never_expires(monkeypatch):
    import hsal.services.l1_cache as l1_mod

    now = [1000.0]
    monkeypatch.setattr(l1_mod.time, "time", lambda: now[0])

    cache = InMemoryL1Cache(max_size=10, ttl_seconds=0)
    cache.set("k", "v")
    now[0] += 10_000_000
    assert cache.get("k") == "v"
