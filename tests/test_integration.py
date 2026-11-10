"""Integration tests: real components assembled together.

- Test 1: fast path (approve) skips the LLM entirely (mocked as fail)
- Test 2: score_transaction runs the REAL pipeline (Gold + encoder + model + calibrator)
- Test 3: format_evidence_for_prompt produces the 5 blocks the LLM expects

Test 2 is intentionally slow (~5s first run): it loads the 636 MB Gold,
3 pickles, runs TargetEncoder, LightGBM, and the calibrator. This is the
only test that exercises the real scoring chain — the API tests mock it.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.fraud_investigator import (
    _load_scoring_artifacts,
    _map_decision,
    score_transaction,
)
from src.agents.graph import build_graph
from src.agents.prompts import format_evidence_for_prompt


# --- Test 1: fast path skips the LLM --------------------------------

def test_fast_path_skips_llm():
    """approve decision routes to fast_report and never touches the LLM."""
    with patch("src.agents.graph.investigator_node",
               side_effect=AssertionError("investigator_node must not be called")), \
         patch("src.agents.graph.reporter_node",
               side_effect=AssertionError("reporter_node must not be called")):
        graph = build_graph()
        state = graph.invoke({
            "transaction_id": 2987000,
            "calibrated_proba": 0.03,
            "raw_score": 0.02,
            "model_decision": "approve",
            "model_version": "1.0.0",
        })

    assert state["route"] == "fast_approve"
    assert state["report"]["recommendation"] == "approve"
    assert state["report"]["confidence"] == "high"


# --- Test 2: real pipeline (no mocks on scoring) --------------------

MODEL_PATH = Path("data/models/classifier.pkl")
GOLD_PATH = Path("data/processed/gold_export.parquet")


@pytest.mark.skipif(
    not MODEL_PATH.exists() or not GOLD_PATH.exists(),
    reason="model artifacts or Gold Parquet not available",
)
def test_score_transaction_real_pipeline():
    """score_transaction runs the real pipeline without mocks.

    Validates invariants, NOT exact values (which depend on library versions).
    Slower than other tests (~3-5s first run due to Gold + pickle loading).
    """
    raw, calibrated, decision = score_transaction(3433766)

    # 1. All outputs are well-typed and in valid ranges
    assert isinstance(raw, float)
    assert isinstance(calibrated, float)
    assert 0.0 <= raw <= 1.0
    assert 0.0 <= calibrated <= 1.0
    assert decision in ("approve", "review", "block")

    # 2. Calibration changes the raw score (Platt is non-identity)
    assert calibrated != raw

    # 3. For transaction 3433766 (known fraud, scale_pos_weight biases raw up)
    # raw > calibrated. This catches a silent regression where calibrator
    # is not applied or is applied in reverse.
    assert raw > calibrated

    # 4. Decision is consistent with the calibrated score and current thresholds
    arts = _load_scoring_artifacts()
    assert decision == _map_decision(calibrated, arts["thresholds"])


# --- Test 3: evidence formatter contract ----------------------------

def test_format_evidence_includes_all_blocks(mock_agent_state):
    """format_evidence_for_prompt produces the 6 sections the LLM expects.

    If a section is missing, the LLM receives a truncated prompt and the
    report quality silently degrades. This test locks the contract.
    """
    text = format_evidence_for_prompt(mock_agent_state)

    expected_blocks = (
        "TRANSACTION",
        "MODEL OUTPUT",
        "CUSTOMER",
        "RELATED_TRANSACTIONS",
        "SHAP_FACTORS",
        "SIMILAR_CASES",
    )
    for block in expected_blocks:
        assert block in text, f"missing section: {block}"