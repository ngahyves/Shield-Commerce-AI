"""Fast-report node: template-based report for trivial decisions.

 Used when the supervised model is confident enough that a full
agentic investigation would add latency without adding signal. The report
is a valid InvestigationReport (same Pydantic contract), just generated
from a template instead of an LLM.

Does NOT include similar_cases or deep SHAP: those require the retrieval
stage which fast paths skip by design.
"""

import uuid

from src.agents.state import AgentState
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def fast_report_node(state: AgentState) -> dict:
    """Generate a template report for fast_approve or fast_block paths."""
    decision = state["model_decision"]
    proba = state["calibrated_proba"]
    raw = state["raw_score"]

    if decision == "approve":
        reasoning = (
            f"Auto-approved. Calibrated fraud probability is {proba:.2%}, "
            f"well below the review threshold. The model is confident this "
            f"transaction is legitimate."
        )
        actions = []
    elif decision == "block":
        reasoning = (
            f"Auto-blocked. Calibrated fraud probability is {proba:.2%}, "
            f"above the block threshold. The model is highly confident this "
            f"transaction is fraudulent."
        )
        actions = ["block transaction", "notify customer"]
    else:
        # Should not be reachable — the router maps only approve/block to fast paths.
        # Fallback is defensive, not expected.
        reasoning = (
            f"Decision '{decision}' reached the fast path unexpectedly. "
            f"Calibrated probability {proba:.2%}."
        )
        actions = []

    report = {
        "transaction_id": state["transaction_id"],
        "recommendation": decision,
        "confidence": "high",
        "reasoning": reasoning,
        "evidence": [
            {
                "claim": f"Model predicted {proba:.2%} calibrated probability",
                "source": "model",
                "field": "calibrated_proba",
                "value": proba,
            },
            {
                "claim": f"Raw model score before calibration: {raw:.4f}",
                "source": "model",
                "field": "raw_score",
                "value": raw,
            },
        ],
        "similar_cases": [],
        "suggested_actions": actions,
        "calibrated_proba": proba,
        "raw_score": raw,
        "model_decision": decision,
        "model_version": state["model_version"],
        "latency_ms": {"fast_path": 0.0},
        "audit_id": f"fast-{uuid.uuid4()}",
    }

    log.info("fast_report: tx=%d decision=%s", state["transaction_id"], decision)
    return {"report": report}