#!/usr/bin/env python3
"""
HSAL Interactive Demo Script
"""

import sys

from hsal import (
    CacheRequest,
    ChromaL2Cache,
    HSALRouter,
    InMemoryL1Cache,
    OllamaEmbedder,
    OllamaLLM,
)


def print_response(prompt: str, response):
    """Pretty print cache response"""
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"SOURCE: {response.source.value}")
    print(f"LATENCY: {response.latency_ms:.2f}ms")

    if response.similarity_score is not None:
        score = response.similarity_score
        status = "✅ SEMANTIC HIT" if score >= 0.9 else "⚠️ LOW SIMILARITY"
        print(f"SIMILARITY SCORE: {score:.4f} ({status})")
        if score >= 0.95:
            print("🚀 STATUS: Promoted to L1 Cache!")

    print("-" * 20)
    print(f"RESPONSE:\n{response.response}")
    print(f"{'='*60}\n")

def main():
    print("🚀 HSAL Interactive Demo - Hybrid Semantic Acceleration Layer")
    print("Type 'exit' or 'quit' to stop.\n")

    # Initialize components
    print("Connecting to local services...")
    l1_cache = InMemoryL1Cache()
    l2_cache = ChromaL2Cache()

    # Use Ollama for real local inference
    try:
        embedder = OllamaEmbedder()
        llm = OllamaLLM()

        # Verify connection implicitly by initializing router
        router = HSALRouter(
            l1_cache=l1_cache,
            l2_cache=l2_cache,
            embedder=embedder,
            llm=llm
        )
        print("✅ Services connected!\n")
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        print("Make sure 'ollama serve' is running and you have pulled llama3.2 and nomic-embed-text.")
        sys.exit(1)

    while True:
        try:
            prompt = input("\n🔍 Enter your prompt: ").strip()

            if not prompt:
                continue

            if prompt.lower() in ('exit', 'quit'):
                print("\nGoodbye! 👋")
                break

            # Query the HSAL Router
            response = router.query(CacheRequest(prompt=prompt))
            print_response(prompt, response)

        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error during query: {e}")

if __name__ == "__main__":
    main()
