from hsal.core.hashing import hash_prompt, normalize


def test_normalize_strips_and_collapses_whitespace():
    assert normalize("  Hello   World \n") == "hello world"


def test_normalize_lowercases():
    assert normalize("HELLO") == "hello"


def test_equivalent_prompts_hash_identically():
    assert hash_prompt(' "Hello World"  ') == hash_prompt('"hello world"')


def test_different_prompts_hash_differently():
    assert hash_prompt("what is python?") != hash_prompt("what is java?")


def test_hash_is_sha256_hex():
    h = hash_prompt("anything")
    assert len(h) == 64
    int(h, 16)  # raises if not hex
