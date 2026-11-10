"""LangGraph state contract.

AgentState: the single dict that flows between every node.
`total=False` means every field is optional 
"""

from typing import Literal, TypedDict
from typing import Annotated
import operator


class AgentState(TypedDict, total=False):
    # Input provided by the caller
    transaction_id: int
    calibrated_proba: float
    raw_score: float
    model_decision: Literal["approve", "review", "block"]
    model_version: str

    # Router output
    route: Literal["fast_approve", "fast_block", "full_investigation"]

    # Evidence populated by retrieval_node
    transaction: dict
    customer: dict
    related_transactions: list[dict]
    shap_factors: list[dict]
    similar_cases: list[dict]

    # Agent outputs
    investigation: str
    report: dict

    # Observability — one writer (retrieval_node), so no reducer needed
    errors: list[str]
    latency_ms: Annotated[dict[str, float], operator.or_]