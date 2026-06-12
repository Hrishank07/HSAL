#!/usr/bin/env python3
"""
HSAL cache-effectiveness benchmark.

Runs synthetic traffic through the full router (using mock embedder/LLM,
so no Ollama or Chroma is needed) and reports how many embedding calls
and LLM generations each traffic shape avoids.

Workloads:
- zipfian:    repetition follows a power law (realistic for FAQ/support traffic)
- uniform:    every prompt drawn uniformly from a fixed pool
- one_off:    every prompt unique (worst case - caching cannot help)

Usage:
    python benchmarks/run_benchmark.py [--requests 5000] [--pool 500] [--seed 42]
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hsal import CacheRequest, HSALRouter, InMemoryL1Cache, MockEmbedder, MockLLM
from hsal.core.types import CacheSource, L2SearchResult
from hsal.services.l2_cache import L2CacheService


class CountingEmbedder(MockEmbedder):
    def __init__(self, dimension: int = 16):
        super().__init__(dimension)
        self.calls = 0

    def embed(self, text: str) -> List[float]:
        self.calls += 1
        return super().embed(text)


class CountingLLM(MockLLM):
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        return super().generate(prompt)


class NullL2Cache(L2CacheService):
    """L2 that never hits - isolates the L1 contribution.

    (MockEmbedder vectors are hash-random, so semantic similarity between
    distinct prompts is meaningless here; a real L2 would only add hits.)
    """

    def search(self, embedding, top_k=1, namespace="default"):
        return L2SearchResult(response="", similarity_score=0.0, found=False)

    def add(self, prompt, response, embedding, namespace="default"):
        pass


def zipfian_traffic(n_requests: int, pool_size: int, rng: random.Random) -> List[str]:
    # P(prompt_i) proportional to 1/(i+1) - classic power law
    weights = [1.0 / (i + 1) for i in range(pool_size)]
    ids = rng.choices(range(pool_size), weights=weights, k=n_requests)
    return [f"prompt number {i}" for i in ids]


def uniform_traffic(n_requests: int, pool_size: int, rng: random.Random) -> List[str]:
    return [f"prompt number {rng.randrange(pool_size)}" for _ in range(n_requests)]


def one_off_traffic(n_requests: int, pool_size: int, rng: random.Random) -> List[str]:
    return [f"unique prompt {i}" for i in range(n_requests)]


def run_workload(name: str, prompts: List[str]) -> dict:
    embedder = CountingEmbedder()
    llm = CountingLLM()
    router = HSALRouter(
        l1_cache=InMemoryL1Cache(max_size=100_000, ttl_seconds=0),
        l2_cache=NullL2Cache(),
        embedder=embedder,
        llm=llm,
        context={"benchmark": name},
    )

    for prompt in prompts:
        router.query(CacheRequest(prompt=prompt))

    n = len(prompts)
    stats = router.stats()
    return {
        "workload": name,
        "requests": n,
        "l1_hits": stats["by_source"][CacheSource.L1_EXACT.value]["count"],
        "hit_rate": stats["cache_hit_rate"],
        "embed_calls": embedder.calls,
        "embed_saved_pct": 100.0 * (1 - embedder.calls / n),
        "llm_calls": llm.calls,
        "llm_saved_pct": 100.0 * (1 - llm.calls / n),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--pool", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    workloads = [
        ("zipfian", zipfian_traffic),
        ("uniform", uniform_traffic),
        ("one_off", one_off_traffic),
    ]

    print(f"\nHSAL benchmark - {args.requests} requests, pool of {args.pool} "
          f"distinct prompts, seed {args.seed}")
    print("(L1 exact-match tier only; a semantic L2 tier would add further hits)\n")

    header = (f"{'workload':<10} {'requests':>8} {'L1 hits':>8} {'hit rate':>9} "
              f"{'embed calls':>12} {'embeds saved':>13} {'LLM calls':>10} {'LLM saved':>10}")
    print(header)
    print("-" * len(header))
    for name, gen in workloads:
        r = run_workload(name, gen(args.requests, args.pool, rng))
        print(f"{r['workload']:<10} {r['requests']:>8} {r['l1_hits']:>8} "
              f"{r['hit_rate']:>9.1%} {r['embed_calls']:>12} "
              f"{r['embed_saved_pct']:>12.1f}% {r['llm_calls']:>10} "
              f"{r['llm_saved_pct']:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
