import hashlib
import re

def normalize(text: str) -> str:
    """
    Normalize text for consistent hashing.
    - Strip leading/trailing whitespace
    - Collapse multiple spaces to single space
    - Convert to lowercase
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text

def hash_prompt(prompt: str) -> str:
    """
    Generate SHA256 hash of normalized prompt.
    """
    normalized = normalize(prompt)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
