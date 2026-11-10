"""Reporting agent: turns the investigator's free-text analysis into a
structured InvestigationReport conforming to the Pydantic schema.

Uses LangChain's with_structured_output so the LLM is forced to emit a
valid InvestigationReport — no manual JSON parsing, no regex, no retry loop.
"""

import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm_client import invoke_with_fallback
from src.agents.prompts import (
    REPORTER_SYSTEM_PROMPT,
    format_evidence_for_prompt,
)
from src.agents.schemas import InvestigationReport
from src.agents.state import AgentState
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def reporter_node(state: AgentState) -> dict:
    """Build the structured InvestigationReport from the investigation text."""
    t0 = time.perf_counter()

    investigation = state.get("investigation", "").strip()
    if not investigation:
        log.warning("reporter: no investigation to work with")
        return {"report": _fallback_report(state)}

    user_payload = (
        f"=== EVIDENCE ===\n{format_evidence_for_prompt(state)}\n\n"
        f"=== INVESTIGATION ANALYSIS ===\n{investigation}\n\n"
        f"=== CONTEXT ===\n"
        f"transaction_id: {state['transaction_id']}\n"
        f"calibrated_proba: {state['calibrated_proba']}\n"
        f"raw_score: {state['raw_score']}\n"
        f"model_decision: {state['model_decision']}\n"
        f"model_version: {state['model_version']}\n"
        f"audit_id: investigation-{uuid.uuid4()}\n"
    )

    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(content=user_payload),
    ]

    try:
        report, provider = invoke_with_fallback(
            messages, structured_schema=InvestigationReport
        )
        report_dict = report.model_dump()
        merged_latency = dict(state.get("latency_ms", {}))
        merged_latency["reporter"] = round((time.perf_counter() - t0) * 1000, 1)
        report_dict["latency_ms"] = merged_latency
        log.info(
            "reporter: recommendation=%s confidence=%s in %.0f ms (provider=%s)",
            report.recommendation, report.confidence,
            (time.perf_counter() - t0) * 1000, provider,
        )
        return {"report": report_dict}

    except Exception as exc:
        log.warning("reporter failed — %s", exc)
        return {"report": _fallback_report(state), "errors": [f"reporter: {exc}"]}


def _fallback_report(state: AgentState) -> dict:
    """Defensive fallback if the LLM fails or returns invalid JSON."""
    merged_latency = dict(state.get("latency_ms", {}))

    return {
        "transaction_id": state["transaction_id"],
        "recommendation": state["model_decision"],
        "confidence": "low",
        "reasoning": "Report generation failed. Falling back to the model's decision.",
        "evidence": [{
            "claim": f"Model predicted {state['calibrated_proba']:.2%} calibrated probability",
            "source": "model",
            "field": "calibrated_proba",
            "value": state["calibrated_proba"],
        }],
        "similar_cases": state.get("similar_cases", []),
        "suggested_actions": [],
        "calibrated_proba": state["calibrated_proba"],
        "raw_score": state["raw_score"],
        "model_decision": state["model_decision"],
        "model_version": state["model_version"],
        "latency_ms": merged_latency,
        "audit_id": f"fallback-{uuid.uuid4()}",
    }