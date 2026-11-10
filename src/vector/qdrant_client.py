"""Lazy singleton for the Qdrant client.

Timeout is set to 60s (not the default 5s) because HNSW indexing competes
with upsert calls and routinely exceeds the default budget on large batches.
"""

import functools
import os
from src.utils.logging_config import ConfigLoader


@functools.lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient

    cfg = ConfigLoader()
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", cfg.get("qdrant", "host", default="localhost")),
        port=int(os.getenv("QDRANT_PORT", cfg.get("qdrant", "port", default=6333))),
        timeout=60,
        prefer_grpc=True,
        grpc_port=int(os.getenv("QDRANT_GRPC_PORT", cfg.get("qdrant", "grpc_port", default=6334))),
    )