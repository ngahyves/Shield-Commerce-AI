"""Redis cache layer for the /explain endpoint.

Cache keys are versioned by (schema version, model version, transaction_id)
so that a retraining or a prompt change naturally invalidates stale reports.

Every operation is fault-tolerant: a Redis failure logs a warning and returns
None, letting the caller fall through to the compute path. The API never
crashes because of a cache miss.
"""

from typing import Optional
import os

import redis
from pydantic import BaseModel

from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()

CACHE_VERSION = "v1" 


class RedisCache:
    """Thin wrapper around redis.Redis with Pydantic JSON serialization."""

    def __init__(self, url: str, default_ttl: int = 86400) -> None:
        self.client = redis.from_url(url, decode_responses=True)
        self.default_ttl = default_ttl

    def ping(self) -> bool:
        """Return True if Redis is reachable, False otherwise. Never raises."""
        try:
            return self.client.ping()
        except Exception:
            return False

    def get_model(self, key: str, model_cls: type[BaseModel]) -> Optional[BaseModel]:
        """Fetch a cached Pydantic model. Returns None on miss or error."""
        try:
            payload = self.client.get(key)
        except Exception as exc:
            log.warning("cache: get failed for %s — %s", key, exc)
            return None
        if not payload:
            return None
        try:
            return model_cls.model_validate_json(payload)
        except Exception as exc:
            log.warning("cache: deserialization failed for %s — %s", key, exc)
            return None

    def set_model(self, key: str, value: BaseModel, ttl: Optional[int] = None) -> None:
        """Store a Pydantic model as JSON. Silent on error."""
        try:
            self.client.setex(key, ttl or self.default_ttl, value.model_dump_json())
        except Exception as exc:
            log.warning("cache: set failed for %s — %s", key, exc)

    def close(self) -> None:
        """close Redis connection"""
        try:
            self.client.close()
        except Exception:
            pass


def build_explain_key(model_version: str, transaction_id: int) -> str:
    """Build the cache key for app/routes/explain route.

    Format: explain:{schema}:{model}:{transaction_id}
    Example: explain:v1:1.0.0:3433766
    """
    return f"explain:{CACHE_VERSION}:{model_version}:{transaction_id}"