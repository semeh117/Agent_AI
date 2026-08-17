# test/test_extraction_cache.py
"""
test_extraction_cache.py
--------------
Unit-level test of the cache mechanics themselves (get/set/clear),
using a throwaway model — deliberately NOT calling any real LLM or
extraction function. Confirms the caching layer works correctly in
isolation before trusting it inside cv_parser.py / job_parser.py.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel
from core.extraction_cache import get_cached, set_cached, clear_cache, cache_stats


class _DummyResult(BaseModel):
    value: str
    count: int


def main():
    print("Clearing any existing test cache entries first...")
    clear_cache("test")

    text_a = "some sample input text A"
    text_b = "some DIFFERENT sample input text B"

    print("\n1. First lookup (should be a MISS):")
    result = get_cached("test", text_a, _DummyResult)
    print(f"   -> {result}")
    assert result is None, "Expected a cache miss on first lookup"

    print("\n2. Storing a result, then looking it up again (should be a HIT):")
    set_cached("test", text_a, _DummyResult(value="hello", count=42))
    result = get_cached("test", text_a, _DummyResult)
    print(f"   -> {result}")
    assert result is not None and result.value == "hello" and result.count == 42, \
        "Expected the exact stored result back"

    print("\n3. Looking up DIFFERENT text under the same namespace (should be a MISS):")
    result = get_cached("test", text_b, _DummyResult)
    print(f"   -> {result}")
    assert result is None, "Different text must not hit the same cache entry"

    print("\n4. Overwriting the same key with a new value:")
    set_cached("test", text_a, _DummyResult(value="updated", count=99))
    result = get_cached("test", text_a, _DummyResult)
    print(f"   -> {result}")
    assert result.value == "updated" and result.count == 99, \
        "Expected the overwritten value, not the original"

    print("\n5. cache_stats():")
    print(f"   -> {cache_stats()}")

    print("\n6. Clearing 'test' namespace...")
    deleted = clear_cache("test")
    print(f"   -> deleted {deleted} entries")
    result = get_cached("test", text_a, _DummyResult)
    assert result is None, "Expected a miss after clearing the namespace"

    print("\nAll cache mechanics checks passed.")


if __name__ == "__main__":
    main()