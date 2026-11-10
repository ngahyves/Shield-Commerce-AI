"""Investigator agent: analyzes gathered evidence and writes a free-text
investigation to `state["investigation"]`.

The agent does not emit the final report — a separate reporting agent does
that. The split keeps concerns clean: this agent reasons, the reporter
formats the reasoning into the InvestigationReport contract.
"""

import time

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm_client import invoke_with_fallback
from src.agents.prompts import (
    INVESTIGATOR_SYSTEM_PROMPT,
    format_evidence_for_prompt,
)
from src.agents.state import AgentState
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


def investigator_node(state: AgentState) -> dict:
    """Call the LLM with the formatted evidence. Return the analysis text."""
    t0 = time.perf_counter()

    messages = [
        SystemMessage(content=INVESTIGATOR_SYSTEM_PROMPT),
        HumanMessage(content=format_evidence_for_prompt(state)),
    ]

    try:
        response, provider = invoke_with_fallback(messages)
        investigation = response.content.strip()
        finish_reason = response.response_metadata.get("finish_reason", "unknown")
        usage = response.response_metadata.get("token_usage", {})
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        log.info(
            "investigator: %d chars in %.0f ms, finish=%s, tokens=%s, provider=%s",
            len(investigation), elapsed_ms, finish_reason, usage, provider,
        )
        return {
            "investigation": investigation,
            "latency_ms": {"investigator": elapsed_ms},
        }

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        log.warning("investigator failed in %.0f ms — %s", elapsed_ms, exc)
        return {
            "investigation": "",
            "errors": [f"investigator: {exc}"],
        }