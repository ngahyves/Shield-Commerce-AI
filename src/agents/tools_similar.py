"""
Uses standardized numeric feature vectors, not text embeddings. The scaler
is fitted at index time and reloaded here — the query vector and the indexed
vectors share the same feature space, so cosine scores are meaningful.
"""

import pickle
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from qdrant_client.models import FieldCondition, Filter, Range
from sklearn.preprocessing import StandardScaler

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger
from src.vector.indexer import vectorize
from src.vector.qdrant_client import get_client

log = ProjectLogger(__name__).get_logger()


@lru_cache(maxsize=1)
def _get_cfg() -> ConfigLoader:
    return ConfigLoader()

@lru_cache(maxsize=1)
def _get_scaler() -> StandardScaler:
    path = Path(_get_cfg().get("paths", "models")) / "similarity_scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


@tool
def search_similar_cases(query_row: dict, transaction_dt: int, limit: int = 5) -> list[dict]:
    """Find historical fraud cases whose numeric profile matches the current transaction.

    Only confirmed frauds from the training period are indexed. Results are
    filtered to transaction_dt < current, so the agent never sees a fraud
    that happened after the one being investigated.

    Similarity is computed on 20 standardized features (amount, distance,
    hour, null flags, card_id, card_bank_1, customer aggregates, Vesta
    counters/deltas, velocity). Cosine scores on 20-dim standardized vectors
    concentrate in a narrow band — use the following calibration, measured
    empirically on the test set (n=100 queries):
      >0.99   near-identical profile (top 5% of matches)
      0.97-0.99 strong similarity   (median match)
      0.93-0.97 partial similarity
      <0.93   distinct profile (only 10% of queries find a match below 0.93)
    """
    collection = _get_cfg().get("qdrant", "collection", default="shieldcommerce_fraud_cases")
    vector = vectorize(query_row, _get_scaler())

    response = get_client().query_points(
        collection_name=collection,
        query=vector,
        query_filter=Filter(must=[FieldCondition(key="transaction_dt", range=Range(lt=transaction_dt))]),
        limit=limit,
    )

    results = [
        {
            "transaction_id": int(h.payload["transaction_id"]),
            "score": round(float(h.score), 4),
            "transaction_dt": int(h.payload["transaction_dt"]),
            "amount": float(h.payload["amount"]),
            "fraud_type": h.payload.get("fraud_type"),
        }
        for h in response.points
    ]

    top = results[0]["score"] if results else 0.0
    log.info("search_similar_cases: %d results (top=%.3f)", len(results), top)
    return results