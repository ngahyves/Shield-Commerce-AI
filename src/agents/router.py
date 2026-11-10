"""Router node: decide whether a transaction needs full investigation.

- approve → fast_approve  (trust the model, template report, ~0ms)
- block   → fast_block    (trust the model, template report, ~0ms)
- review  → full_investigation (ambiguous zone, needs LLM analysis)

The supervised model is the first filter. The agent is a second level for
the review zone only — where the model isn't confident enough to decide.
"""

from src.agents.state import AgentState
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


ROUTE_MAP = {
    "approve": "fast_approve",
    "block": "fast_block",
    "review": "full_investigation",
}


def router_node(state: AgentState) -> dict:
    """Map model_decision to a graph route."""
    decision = state["model_decision"]
    route = ROUTE_MAP.get(decision, "full_investigation")
    log.info(
        "router: tx=%d decision=%s proba=%.4f → %s",
        state["transaction_id"], decision, state["calibrated_proba"], route,
    )
    return {"route": route}