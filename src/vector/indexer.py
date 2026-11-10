"""Index confirmed frauds using standardized numeric feature vectors.

Text embeddings (MiniLM, BGE) collapse on short structured inputs — their
latent space is anisotropic and produces cosine similarities of 0.95+ between
unrelated items. Tabular similarity requires a numeric metric.

Approach: standardize 19 numeric features (amount is log-transformed),
store the vectors in Qdrant with cosine distance, filter temporally at query
time. The scaler is fitted on ALL training transactions (not just frauds) so
the standardization reflects the true feature distribution, then saved to disk
for reuse at query time.

All features in VECTOR_FEATURES are numeric (int or float). String columns
(buyer_region, m4, etc.) are excluded — they would break the to_numpy(float)
conversion. card_id alone (13,500 unique values) is the main collision breaker.
"""

import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sklearn.preprocessing import StandardScaler

from src.utils.logger import ProjectLogger
from src.vector.qdrant_client import get_client

log = ProjectLogger(__name__).get_logger()

VECTOR_FEATURES = [
    "amount",
    "distance_buyer_merchant", 
    "hour_of_day",
    "is_null_dist1",
    "is_null_dist2",
    "is_null_addr1",
    "card_id",
    "card_bank_1",
    "uid_amount_mean",
    "uid_amount_std",
    "uid_transaction_count",
    "c1", "c2", "c11", "c13",
    "d2", "d15",
    "v48",
    "velocity_count_1h",
    "velocity_count_24h",
]

VECTOR_DIM = len(VECTOR_FEATURES)
BATCH_SIZE = 256


def _extract_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract and transform vector features. All columns must be numeric."""
    missing = [c for c in VECTOR_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Vector features missing from DataFrame: {missing}. "
            f"Check the Gold column names — some may have been renamed."
        )
    sub = df[VECTOR_FEATURES].copy()
    for c in VECTOR_FEATURES:
        sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
    sub["amount"] = np.log1p(sub["amount"].clip(lower=0.0))
    return sub.to_numpy(dtype=float)


def vectorize(row: dict, scaler: StandardScaler) -> list[float]:
    """Transform one transaction row (dict) into a standardized vector."""
    return scaler.transform(_extract_matrix(pd.DataFrame([row])))[0].tolist()


def recreate_collection(collection_name: str) -> None:
    client = get_client()
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )
    client.create_payload_index(
        collection_name=collection_name,
        field_name="transaction_dt",
        field_schema=PayloadSchemaType.INTEGER,
    )
    log.info("collection recreated: %s (dim=%d)", collection_name, VECTOR_DIM)


def build_index(
    gold_df: pd.DataFrame,
    collection_name: str,
    train_end_dt: int,
    scaler_path: str | Path,
) -> int:
    train = gold_df[gold_df["transaction_dt"] < train_end_dt]
    frauds = train[train["is_fraud"] == 1].copy()

    if frauds.empty:
        log.warning("no frauds to index")
        return 0

    scaler = StandardScaler().fit(_extract_matrix(train))
    scaler_path = Path(scaler_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("scaler fitted on %d train rows, saved: %s", len(train), scaler_path)

    fraud_vectors = scaler.transform(_extract_matrix(frauds))
    client = get_client()

    for start in range(0, len(frauds), BATCH_SIZE):
        batch_df = frauds.iloc[start : start + BATCH_SIZE]
        batch_vecs = fraud_vectors[start : start + BATCH_SIZE]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec.tolist(),
                payload={
                    "transaction_id": int(row["transaction_id"]),
                    "transaction_dt": int(row["transaction_dt"]),
                    "amount": float(row["amount"]),
                    "fraud_type": None,
                },
            )
            for (_, row), vec in zip(batch_df.iterrows(), batch_vecs)
        ]
        client.upsert(collection_name=collection_name, points=points, wait=True)

    log.info("indexing complete: %d points indexed (dim=%d)", len(frauds), VECTOR_DIM)
    return len(frauds)