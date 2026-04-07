"""Tests for Redis cache — version-aware keys and TTL logic."""

import sys

sys.path.insert(0, ".")

from src.utils.redis_client import get_ttl_seconds, build_cache_key


def test_ttl_far_departure():
    """Flight > 30 days out should have 6-hour TTL."""
    assert get_ttl_seconds(45) == 21600


def test_ttl_mid_departure():
    """Flight 7-30 days out should have 1-hour TTL."""
    assert get_ttl_seconds(15) == 3600


def test_ttl_near_departure():
    """Flight 1-7 days out should have 5-minute TTL."""
    assert get_ttl_seconds(3) == 300


def test_ttl_imminent_departure():
    """Flight < 1 day out should have 1-minute TTL."""
    assert get_ttl_seconds(0) == 60


def test_ttl_decreases_with_proximity():
    """TTL must decrease as departure approaches."""
    assert get_ttl_seconds(60) > get_ttl_seconds(20)
    assert get_ttl_seconds(20) > get_ttl_seconds(5)
    assert get_ttl_seconds(5) > get_ttl_seconds(0)


def test_cache_key_includes_model_version():
    """Cache key must include model version to prevent stale predictions."""
    key_v1 = build_cache_key("JFK-LAX", "2024-07", "1.0.0", 30)
    key_v2 = build_cache_key("JFK-LAX", "2024-07", "2.0.0", 30)

    assert (
        key_v1 != key_v2
    ), "Different model versions must produce different cache keys"


def test_cache_key_includes_route():
    """Cache key must be route-specific."""
    key1 = build_cache_key("JFK-LAX", "2024-07", "1.0.0", 30)
    key2 = build_cache_key("ORD-MIA", "2024-07", "1.0.0", 30)

    assert key1 != key2


def test_cache_key_format():
    """Cache key must follow expected format."""
    key = build_cache_key("JFK-LAX", "2024-07", "1.0.0", 30)
    assert "v1.0.0" in key
    assert "JFK-LAX" in key
    assert "2024-07" in key
