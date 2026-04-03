"""
Redis client singleton.
All Redis access goes through this module.
Handles connection failures gracefully — Redis being down
must never crash the API.
"""

from typing import Optional
import redis
from src.utils.logger import logger
from src.utils.settings import get_settings


def get_redis_client() -> Optional[redis.Redis]:
    """
    Returns a Redis client, or None if connection fails.
    Returning None instead of raising allows callers to
    implement graceful degradation — serve without cache
    rather than returning 500 to the user.
    """
    settings = get_settings()
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        logger.debug("Redis connection established")
        return client
    except redis.ConnectionError as e:
        logger.warning(f"Redis unavailable — caching disabled: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected Redis error: {e}")
        return None


def get_ttl_seconds(days_to_departure: int) -> int:
    """
    TTL derived from booking horizon volatility.
    Not an arbitrary scalar — derived from RM domain knowledge.
    """
    if days_to_departure > 30:
        return 21600  # 6 hours — demand stable far out
    elif days_to_departure > 7:
        return 3600  # 1 hour — moderate volatility
    elif days_to_departure > 1:
        return 300  # 5 minutes — high volatility
    else:
        return 60  # 1 minute — near departure


def build_cache_key(
    route: str,
    date: str,
    model_version: str,
    days_to_departure: int,
) -> str:
    """
    Version-aware cache key.
    Including model_version ensures stale predictions from
    old model versions are never served after retraining.
    Format: pred:v{version}:{route}:{date}:{days_out}
    """
    return f"pred:v{model_version}:{route}:{date}:{days_to_departure}"
