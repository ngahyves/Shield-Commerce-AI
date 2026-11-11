"""Q2 - Hallucination detection using Gemini as judge.

For each investigation narrative, Gemini answers 5 binary questions:
    Q1 - Do numeric values cited match the evidence?
    Q2 - Do customer claims match the evidence?
    Q3 - Do similar_cases cited exist in the evidence?
    Q4 - Is the recommendation justified by the evidence?
    Q5 - Is the narrative free of internal contradictions?

Score = number of "false" answers / 5
(0.0 = fully faithful, 1.0 = narrative entirely fabricated)
"""

import ast
import json
import os
import re

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.prompts import format_evidence_for_prompt
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()


JUDGE_PROMPT = """You are evaluating a fraud investigation narrative against the source evidence.

=== EVIDENCE (from tools) ===
{evidence}

=== ANALYST NARRATIVE ===
{narrative}

=== REPORTER REASONING ===
{reasoning}

Answer 5 binary questions about the narrative and reasoning.

Q1 - Do the numeric values cited (amount, velocity, SHAP values, ratios) match the evidence?
Q2 - Do the claims about the customer (new, history, avg amount, transaction count) match the evidence?
Q3 - Do the similar_cases cited (transaction_id, score, amount) exist in the evidence provided?
Q4 - Is the recommendation (approve / review / block) justified by the evidence?
Q5 - Is the narrative free of internal contradictions (e.g. amount 554 in one section and 850 in another)?

Return ONLY valid JSON (no markdown, no code fences):
{{
  "Q1_numeric_values_match": true,
  "Q2_customer_claims_match": true,
  "Q3_similar_cases_exist": true,
  "Q4_recommendation_supported": true,
  "Q5_no_contradictions": true,
"issues": ["<one-line description per failed question>"]
}}

CRITICAL: use ONLY double quotes (") for JSON keys and strings. Never single quotes.
Return valid JSON parsable by json.loads()."""

def _extract_text(content) -> str:
    """Extract text from Gemini content (str or list of content blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)

def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, tolerating single quotes and multiple objects."""
    import logging
    logging.warning("_parse_json RAW: %r", raw[:300])

    text = raw.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # Try strict JSON first
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = ast.literal_eval(text)

    # If literal_eval returned a tuple (multiple JSONs), take the first dict
    if isinstance(result, tuple):
        for item in result:
            if isinstance(item, dict):
                return item
        raise ValueError(f"no dict in tuple: {result!r}")

    if not isinstance(result, dict):
        raise ValueError(f"expected dict, got {type(result).__name__}: {result!r}")

    return result


class HallucinationJudge:
    """Runs a 5-question hallucination check on investigation narratives."""

    def __init__(self) -> None:
        cfg = ConfigLoader()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set — cannot run hallucination judge"
            )
        self.llm = ChatGoogleGenerativeAI(
            model=cfg.get("agent", "llm_model_2", default="gemini-3.5-flash-lite"),
            temperature=0.1,
            max_output_tokens=500,
            google_api_key=api_key,
            request_timeout=30.0,
        )

    def judge(self, state: dict) -> dict:
        """Run the 5-question check on a single investigation state."""
        narrative = state.get("investigation", "").strip()
        reasoning = state.get("report", {}).get("reasoning", "").strip()

        if not narrative:
            return {
                "verdict": None,
                "score": None,
                "issues": ["no narrative to evaluate"],
                "raw": "",
            }

        prompt = JUDGE_PROMPT.format(
            evidence=format_evidence_for_prompt(state),
            narrative=narrative,
            reasoning=reasoning,
        )

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw = _extract_text(response.content)
            verdict = _parse_json(raw)
        except Exception as exc:
            log.warning("Q2 judge failed: %s", exc)
            log.warning("Q2 judge raw content: %r", raw[:500] if 'raw' in dir() else "<no raw>")
            return {
                "verdict": None,
                "score": None,
                "issues": [f"judge exception: {exc}"],
                "raw": raw if 'raw' in dir() else "",
            }

        keys = [
            "Q1_numeric_values_match",
            "Q2_customer_claims_match",
            "Q3_similar_cases_exist",
            "Q4_recommendation_supported",
            "Q5_no_contradictions",
        ]
        n_false = sum(1 for k in keys if verdict.get(k) is False)
        score = n_false / 5.0

        return {
            "verdict": {k: verdict.get(k) for k in keys},
            "score": score,
            "issues": verdict.get("issues", []),
            "raw": raw,
        }

    def run(self, states: dict[int, dict]) -> dict:
        """Judge every investigation state. Key = tx_id, value = state."""
        log.info("Q2 hallucination: judging %d narratives", len(states))

        per_case = {}
        scores = []

        for tx_id, state in states.items():
            result = self.judge(state)
            per_case[tx_id] = result
            if result["score"] is not None:
                scores.append(result["score"])
            log.info(
                "  tx=%d score=%s issues=%d",
                tx_id,
                f"{result['score']:.2f}" if result["score"] is not None else "N/A",
                len(result["issues"]),
            )

        mean_score = sum(scores) / len(scores) if scores else None
        above_half = sum(1 for s in scores if s > 0.5)

        metrics = {
            "mean_score": mean_score,
            "n_judged": len(scores),
            "n_above_0_5": above_half,
            "per_case": per_case,
        }

        log.info(
            "Q2 summary: mean=%.2f over %d cases, %d case(s) with score > 0.5",
            mean_score if mean_score is not None else 0.0,
            len(scores), above_half,
        )
        return metrics
