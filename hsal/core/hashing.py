import hashlib
import json
import re
from typing import Optional


def normalize(text: str, lowercase: bool = True) -> str:
    """
    Normalize text for consistent hashing.
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    - Lowercase (optional — disable for case-sensitive workloads
      like code, SQL, or identifiers, where 'Users' != 'users')
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    if lowercase:
        text = text.lower()
    return text


def context_fingerprint(context: Optional[dict]) -> str:
    """
    Deterministic fingerprint of the generation context.

    A prompt alone is not a safe cache key: the same prompt produces
    different answers under a different model, system prompt, temperature,
    tool schema, or tenant. Everything that affects the output belongs in
    the fingerprint so cached answers are only served within the exact
    configuration that produced them.
    """
    if not context:
        return "default"
    canonical = json.dumps(context, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def hash_prompt(prompt: str, context: Optional[dict] = None, lowercase: bool = True) -> str:
    """
    Generate the cache key: SHA256 of the normalized prompt,
    scoped by the context fingerprint.
    """
    normalized = normalize(prompt, lowercase=lowercase)
    fingerprint = context_fingerprint(context)
    return hashlib.sha256(f"{fingerprint}:{normalized}".encode("utf-8")).hexdigest()
