"""Unit tests for the LLMCache SQLite-backed response cache."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from triage_iq.cache import LLMCache


@pytest.fixture
def cache(tmp_path: Path):
    c = LLMCache(path=tmp_path / "test_cache.sqlite")
    yield c
    c.close()


_MESSAGES = [
    {"role": "system", "content": "You are a judge."},
    {"role": "user", "content": "Score this plan."},
]


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def test_compute_key_is_deterministic():
    k1 = LLMCache.compute_key("groq", "llama-3.3-70b", _MESSAGES, 0.0, 1024)
    k2 = LLMCache.compute_key("groq", "llama-3.3-70b", _MESSAGES, 0.0, 1024)
    assert k1 == k2


def test_compute_key_differs_on_provider():
    k1 = LLMCache.compute_key("groq", "llama", _MESSAGES, 0.0, 1024)
    k2 = LLMCache.compute_key("cohere", "llama", _MESSAGES, 0.0, 1024)
    assert k1 != k2


def test_compute_key_differs_on_model():
    k1 = LLMCache.compute_key("groq", "model-a", _MESSAGES, 0.0, 1024)
    k2 = LLMCache.compute_key("groq", "model-b", _MESSAGES, 0.0, 1024)
    assert k1 != k2


def test_compute_key_differs_on_temperature():
    k1 = LLMCache.compute_key("groq", "m", _MESSAGES, 0.0, 1024)
    k2 = LLMCache.compute_key("groq", "m", _MESSAGES, 0.5, 1024)
    assert k1 != k2


def test_compute_key_differs_on_extra_kwargs():
    k1 = LLMCache.compute_key("cohere", "m", _MESSAGES, response_format={"type": "json"})
    k2 = LLMCache.compute_key("cohere", "m", _MESSAGES)
    assert k1 != k2


def test_compute_key_is_64_hex_chars():
    k = LLMCache.compute_key("groq", "m", _MESSAGES)
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# Float canonicalization (2026-08-27) -- see llm_cache.py's _canonicalize_floats
# docstring for the full mechanism. The 1-ULP case below is the exact one
# reproduced via a three-environment control experiment (local machine: zero
# misses; a container pinned to requirements.lock: reproduced the identical
# CI cassette miss), not a synthetic example.
# ---------------------------------------------------------------------------

def _plan_message(resolution_lower_days: str) -> list[dict]:
    return [
        {"role": "system", "content": "Evaluate the triage plan."},
        {
            "role": "user",
            "content": (
                '{"predicted_component": "test-infra", "component_confidence": 0.582703122181352, '
                f'"expected_resolution_lower_days": {resolution_lower_days}, '
                '"expected_resolution_upper_days": 56.76348724576911}'
            ),
        },
    ]


def test_compute_key_hits_on_1ulp_float_noise():
    """The exact reproduced case: two float64 representations of (approximately) the same
    value, differing only in the last representable digit, must now hash identically."""
    k1 = LLMCache.compute_key("ollama", "qwen3:8b", _plan_message("0.0356567072910697"), 0.0, seed=42)
    k2 = LLMCache.compute_key("ollama", "qwen3:8b", _plan_message("0.03565670729106969"), 0.0, seed=42)
    assert k1 == k2


def test_compute_key_misses_on_materially_different_prediction():
    """A real, non-noise difference (differs at the 2nd decimal, not the 15th) must still
    produce a different key -- canonicalization must not mask an actual regression."""
    k1 = LLMCache.compute_key("ollama", "qwen3:8b", _plan_message("0.0356567072910697"), 0.0, seed=42)
    k2 = LLMCache.compute_key("ollama", "qwen3:8b", _plan_message("0.15"), 0.0, seed=42)
    assert k1 != k2


def test_compute_key_misses_on_changed_prompt_template():
    """A real prompt/template change (not a numeric one) must still produce a different
    key -- canonicalization only touches float literals, never surrounding text."""
    messages_a = _plan_message("0.0356567072910697")
    messages_b = [
        {"role": "system", "content": "Evaluate the triage plan against a DIFFERENT rubric."},
        messages_a[1],
    ]
    k1 = LLMCache.compute_key("ollama", "qwen3:8b", messages_a, 0.0, seed=42)
    k2 = LLMCache.compute_key("ollama", "qwen3:8b", messages_b, 0.0, seed=42)
    assert k1 != k2


def test_compute_key_misses_on_changed_retrieval_result():
    """A different similar-issue number cited (a real retrieval-result change, not
    numeric noise) must still produce a different key."""
    base = _plan_message("0.0356567072910697")
    messages_a = [base[0], {**base[1], "content": base[1]["content"].replace(
        "56.76348724576911}", '56.76348724576911, "similar_issues": [4343]}'
    )}]
    messages_b = [base[0], {**base[1], "content": base[1]["content"].replace(
        "56.76348724576911}", '56.76348724576911, "similar_issues": [9999]}'
    )}]
    k1 = LLMCache.compute_key("ollama", "qwen3:8b", messages_a, 0.0, seed=42)
    k2 = LLMCache.compute_key("ollama", "qwen3:8b", messages_b, 0.0, seed=42)
    assert k1 != k2


# ---------------------------------------------------------------------------
# Miss → set → hit cycle
# ---------------------------------------------------------------------------

def test_get_returns_none_on_miss(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    assert cache.get(key) is None


def test_set_then_get_returns_response(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.set(key, "groq", "m", _MESSAGES, {"content": "hello", "usage": {"prompt_tokens": 10}})
    result = cache.get(key)
    assert result is not None
    assert result["content"] == "hello"
    assert result["usage"]["prompt_tokens"] == 10


def test_set_is_idempotent(cache: LLMCache):
    """INSERT OR IGNORE — second set with same key must not raise."""
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.set(key, "groq", "m", _MESSAGES, {"content": "first"})
    cache.set(key, "groq", "m", _MESSAGES, {"content": "second"})  # must not raise
    result = cache.get(key)
    assert result["content"] == "first"  # original preserved


# ---------------------------------------------------------------------------
# Session stats
# ---------------------------------------------------------------------------

def test_stats_initial_empty(cache: LLMCache):
    st = cache.stats()
    assert st["entries"] == 0
    assert st["session_hits"] == 0
    assert st["session_misses"] == 0
    assert st["session_hit_rate"] == 0.0


def test_stats_after_miss(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.get(key)
    st = cache.stats()
    assert st["session_misses"] == 1
    assert st["session_hits"] == 0


def test_stats_after_hit(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.set(key, "groq", "m", _MESSAGES, {"content": "r"})
    cache.get(key)
    st = cache.stats()
    assert st["session_hits"] == 1
    assert st["entries"] == 1


def test_stats_hit_rate(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.set(key, "groq", "m", _MESSAGES, {"content": "r"})
    cache.get(key)   # hit
    cache.get("miss-key")  # miss
    st = cache.stats()
    assert st["session_hits"] == 1
    assert st["session_misses"] == 1
    assert st["session_hit_rate"] == pytest.approx(0.5)


def test_stats_size_bytes(cache: LLMCache):
    key = LLMCache.compute_key("groq", "m", _MESSAGES)
    cache.set(key, "groq", "m", _MESSAGES, {"content": "x" * 1000})
    st = cache.stats()
    assert st["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Clear operations
# ---------------------------------------------------------------------------

def test_clear_all(cache: LLMCache):
    for i in range(3):
        msgs = [{"role": "user", "content": f"msg {i}"}]
        key = LLMCache.compute_key("groq", "m", msgs)
        cache.set(key, "groq", "m", msgs, {"content": f"r{i}"})
    removed = cache.clear()
    assert removed == 3
    assert cache.stats()["entries"] == 0


def test_clear_provider(cache: LLMCache):
    msgs_a = [{"role": "user", "content": "a"}]
    msgs_b = [{"role": "user", "content": "b"}]
    k_groq = LLMCache.compute_key("groq", "m", msgs_a)
    k_cohere = LLMCache.compute_key("cohere", "m", msgs_b)
    cache.set(k_groq, "groq", "m", msgs_a, {"content": "r1"})
    cache.set(k_cohere, "cohere", "m", msgs_b, {"content": "r2"})
    removed = cache.clear_provider("groq")
    assert removed == 1
    assert cache.stats()["entries"] == 1


def test_clear_model(cache: LLMCache):
    msgs_a = [{"role": "user", "content": "a"}]
    msgs_b = [{"role": "user", "content": "b"}]
    k1 = LLMCache.compute_key("groq", "model-a", msgs_a)
    k2 = LLMCache.compute_key("groq", "model-b", msgs_b)
    cache.set(k1, "groq", "model-a", msgs_a, {"content": "r1"})
    cache.set(k2, "groq", "model-b", msgs_b, {"content": "r2"})
    removed = cache.clear_model("groq", "model-a")
    assert removed == 1
    assert cache.get(k2) is not None


# ---------------------------------------------------------------------------
# Thread safety — real concurrent stress test
# ---------------------------------------------------------------------------

def test_concurrent_writes_do_not_crash(cache: LLMCache):
    """200 unique writes across 8 workers; no DB errors and exact row count after join."""
    TOTAL = 200
    errors: list[Exception] = []
    written_keys: list[str] = []

    def worker(idx: int) -> str:
        msgs = [{"role": "user", "content": f"msg-{idx}"}]
        key = LLMCache.compute_key("groq", "model-concurrent", msgs)
        cache.set(key, "groq", "model-concurrent", msgs, {"content": f"resp-{idx}"})
        result = cache.get(key)
        assert result is not None, f"get after set returned None for idx={idx}"
        return key

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(TOTAL)]
        for f in as_completed(futures):
            try:
                written_keys.append(f.result())
            except Exception as exc:
                errors.append(exc)

    assert not errors, f"Concurrent errors: {errors}"
    assert cache.stats()["entries"] == TOTAL
    assert len(set(written_keys)) == TOTAL, "Key collision — messages not generating unique keys"
