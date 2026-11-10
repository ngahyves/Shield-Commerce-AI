"""Deterministic tools for the fraud investigation agent.

All tools in this file read from the local Gold Parquet — the same file
used at training time, so there is zero training-serving skew.

Each tool is intentionally narrow:
- get_transaction: fetch one row's features
- get_customer: aggregate the customer's history BEFORE the current transaction
- get_related_transactions: last N transactions in a time window
- get_shap_explanations: top SHAP drivers for one transaction

Every tool that reads a customer history filters on transaction_dt strictly
less than the current transaction. This mirrors the temporal split used at
training and prevents the investigator from seeing future activity.
"""

import functools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600


# ── Singletons ───────────────────────────────────────────────────────
# The Gold Parquet is loaded once and reused across tool calls. lru_cache ensures we pay the cost only the first time.

@functools.lru_cache(maxsize=1)
def _cfg() -> ConfigLoader:
    return ConfigLoader()


@functools.lru_cache(maxsize=1)
def _gold() -> pd.DataFrame:
    path = Path(_cfg().get("paths", "gold_parquet"))
    df = pd.read_parquet(path)
    log.info("gold loaded for tools: %d rows, %d cols", len(df), len(df.columns))
    return df.set_index("transaction_id", drop=False)

@functools.lru_cache(maxsize=1)
def _encoder():
    """Load the target encoder once. Cached."""
    from src.features.target_encoder import TargetEncoderBuilder
    return TargetEncoderBuilder.load(
        Path(_cfg().get("paths", "models")) / "target_encoder.pkl"
    )

@functools.lru_cache(maxsize=1)
def _shap_explainer():
    """Instantiate the SHAP explainer once. Cached.

    Building the TreeExplainer on the LightGBM model costs ~5s. Cached
    across calls, subsequent SHAP computations are ~50-100ms.
    """
    from src.explainability.shap_explainer import ShapExplainer
    return ShapExplainer()


def _clean(val: Any) -> Any:
    """Convert numpy/pandas scalars to native Python for JSON + LLM safety.

    Without this, values like numpy.int64 cannot be serialized to JSON
    and confuse the LLM when they appear in tool output.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ── Tools ────────────────────────────────────────────────────────────

@tool
def get_transaction(transaction_id: int) -> dict:
    """Fetch the raw features of one transaction.

    Used by the investigator agent to understand what the transaction
    looks like: amount, product, device, hour, velocity counters.
    """
    df = _gold()
    if transaction_id not in df.index:
        raise ValueError(f"transaction not found: {transaction_id}")
    return {k: _clean(v) for k, v in df.loc[transaction_id].to_dict().items()}


def get_transaction_frame(transaction_id: int) -> pd.DataFrame:
    """Return raw transaction as a single-row DataFrame, preserving Gold dtypes.

    get_transaction() returns a JSON-safe dict and loses dtypes (NaN -> None,
    float64 -> object on single-row inference), which breaks LightGBM.
    """
    df = _gold()
    try:
        pos = df.index.get_loc(transaction_id)
    except KeyError:
        raise ValueError(f"transaction_id {transaction_id} not found in Gold")
    return df.iloc[pos:pos + 1]

@tool
def get_customer(uid: str, transaction_dt: int) -> dict:
    """Aggregate the customer's history BEFORE the current transaction.

    Only transactions strictly earlier than `transaction_dt` are counted
    """
    df = _gold()

    # All rows for this customer (to confirm the uid exists at all)
    all_rows = df[df["uid"] == uid]
    if all_rows.empty:
        raise ValueError(f"customer not found: {uid}")

    # Only prior transactions are visible to the investigator
    prior = all_rows[all_rows["transaction_dt"] < transaction_dt]

    if prior.empty:
        return {
            "uid": uid,
            "is_new_customer": True,
            "first_seen_days_ago": 0.0,
            "last_activity_days_ago": 0.0,
            "n_transactions": 0,
            "n_frauds_confirmed": 0,
            "avg_amount": 0.0,
            "std_amount": 0.0,
            "max_amount": 0.0,
        }

    amounts = prior["amount"]

    # Days are measured from the current transaction, not from the end of
    # the dataset — so the numbers stay meaningful for any investigation time.
    return {
        "uid": uid,
        "is_new_customer": False,
        "first_seen_days_ago": round((transaction_dt - prior["transaction_dt"].min()) / SECONDS_PER_DAY, 1),
        "last_activity_days_ago": round((transaction_dt - prior["transaction_dt"].max()) / SECONDS_PER_DAY, 1),
        "n_transactions": int(len(prior)),
        "n_frauds_confirmed": int(prior["is_fraud"].sum()),
        "avg_amount": round(float(amounts.mean()), 2),
        "std_amount": round(float(amounts.std(ddof=0)), 2),
        "max_amount": round(float(amounts.max()), 2),
    }


@tool
def get_related_transactions(
    uid: str, transaction_dt: int, hours: int = 24, limit: int = 10
) -> list[dict]:
    """List the most recent transactions of this customer before the current one.

    Only looks backward in time (transaction_dt strictly less than the
    current transaction). This lets the investigator spot burst patterns
    ("3 transactions in the last 30 minutes") without ever seeing future
    activity — the same temporal-leakage guard used in training.
    """
    df = _gold()
    window_start = transaction_dt - hours * SECONDS_PER_HOUR

    subset = df[
        (df["uid"] == uid)
        & (df["transaction_dt"] < transaction_dt)
        & (df["transaction_dt"] >= window_start)
    ].sort_values("transaction_dt", ascending=False).head(limit)

    return [
        {
            "transaction_id": int(r["transaction_id"]),
            "transaction_dt": int(r["transaction_dt"]),
            "amount": float(r["amount"]),
            "hours_before": round((transaction_dt - int(r["transaction_dt"])) / SECONDS_PER_HOUR, 2),
            "is_fraud": int(r["is_fraud"]),
        }
        for _, r in subset.iterrows()
    ]


@tool
def get_shap_explanations(transaction_id: int, top_n: int = 5) -> list[dict]:
    """Return the top SHAP contributors for this transaction."""
    cfg = _cfg()
    df = _gold()
    if transaction_id not in df.index:
        raise ValueError(f"transaction not found: {transaction_id}")

    x_row = _encoder().transform(
        df.loc[[transaction_id]].drop(columns=cfg.get("features", "metadata_cols"))
    )

    factors = _shap_explainer().explain_transaction(x_row, top_n=top_n)
    for f in factors:
        f["direction"] = "positive" if f["shap_value"] >= 0 else "negative"
    return factors


TOOLS = [
    get_transaction,
    get_customer,
    get_related_transactions,
    get_shap_explanations,
]