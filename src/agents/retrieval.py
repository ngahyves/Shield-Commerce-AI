"""Retrieval stage: call the tools and populate the state.

The similarity search uses `vectorize` from src.vector.indexer — the exact
same function used at index time — so query and indexed vectors live in the
same standardized space and cosine scores are comparable
"""

import time

from src.agents.state import AgentState
from src.agents.tools import (
    get_customer,
    get_related_transactions,
    get_shap_explanations,
    get_transaction,
)
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def retrieval_node(state: AgentState) -> dict:
    """Fetch the transaction, then call the tools that depend on it.

    Each tool call is wrapped in try/except: one failing tool does not
    block the others. Errors accumulate in `out["errors"]` and the
    investigator still reasons with the remaining evidence.
    """
    t0 = time.perf_counter()

    tx = get_transaction.invoke({"transaction_id": state["transaction_id"]})
    out: dict = {"transaction": tx}
    errors: list[str] = []

    calls = [
        ("customer", get_customer,
         {"uid": tx["uid"], "transaction_dt": tx["transaction_dt"]}),
        ("related_transactions", get_related_transactions,
         {"uid": tx["uid"], "transaction_dt": tx["transaction_dt"]}),
        ("shap_factors", get_shap_explanations,
         {"transaction_id": state["transaction_id"]}),
    ]

    cfg = ConfigLoader()
    if cfg.get("agent", "enable_similar_search", default=False):
        from src.agents.tools_similar import search_similar_cases

        calls.append(("similar_cases", search_similar_cases, {
            "query_row": tx,
            "transaction_dt": tx["transaction_dt"],
        }))

    for field, tool, inputs in calls:
        try:
            out[field] = tool.invoke(inputs)
            log.info("retrieval: %s ok", field)
        except Exception as exc:
            log.warning("retrieval: %s failed — %s", field, exc)
            errors.append(f"{field}: {exc}")

    if errors:
        out["errors"] = errors

    out["latency_ms"] = {"retrieval": round((time.perf_counter() - t0) * 1000, 1)}
    return out