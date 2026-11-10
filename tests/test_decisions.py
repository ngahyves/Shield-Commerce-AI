"""Tests for the core decision logic: threshold mapping, routing, cache key, evidence schema.

These are pure functions — no I/O, no mocks, no external dependencies.
"""

import pytest
from pydantic import ValidationError

from src.agents.fraud_investigator import _map_decision
from src.agents.router import router_node
from src.agents.schemas import Evidence
from app.cache import build_explain_key


# --- Test 1: threshold -> decision mapping ---------------------------

def test_map_decision_three_zones():
    """A calibrated probability maps to the correct business decision."""
    thresholds = {"block": 0.7346, "review": 0.0412}

    assert _map_decision(0.03, thresholds) == "approve"
    assert _map_decision(0.30, thresholds) == "review"
    assert _map_decision(0.80, thresholds) == "block"


# --- Test 2: decision -> route mapping ------------------------------

def test_router_node_maps_decisions_to_routes():
    """router_node maps each decision to its graph route."""
    base_state = {"transaction_id": 1, "calibrated_proba": 0.5}

    approve = router_node({**base_state, "model_decision": "approve"})
    assert approve["route"] == "fast_approve"

    block = router_node({**base_state, "model_decision": "block"})
    assert block["route"] == "fast_block"

    review = router_node({**base_state, "model_decision": "review"})
    assert review["route"] == "full_investigation"


# --- Test 3: boundary case (>= vs >) --------------------------------

def test_map_decision_includes_boundaries():
    """Values exactly at a threshold are included in the higher zone (>= not >)."""
    thresholds = {"block": 0.5, "review": 0.2}

    # Just below review -> approve
    assert _map_decision(0.19999, thresholds) == "approve"

    # Exactly at review -> review (not approve)
    assert _map_decision(0.2, thresholds) == "review"

    # Exactly at block -> block (not review)
    assert _map_decision(0.5, thresholds) == "block"


# --- Test 4: cache key format --------------------------------------

def test_build_explain_key_format():
    """Cache key includes schema version and model version (invalidation safety)."""
    key = build_explain_key("1.0.0", 3433766)
    assert key == "explain:v1:1.0.0:3433766"


# --- Test 5: EvidenceItem value contract ---------------------------

def test_evidence_item_accepts_scalar_rejects_array():
    """EvidenceItem.value must be a scalar (protects against LLM returning arrays)."""
    # Accepts a boolean (case: is_new_customer)
    item = Evidence(claim="customer is new", source="get_customer",
                       field="is_new_customer", value=True)
    assert item.value is True

    # Accepts a number
    item = Evidence(claim="amount 554", source="get_transaction",
                       field="amount", value=554.0)
    assert item.value == 554.0

    # Rejects a list (case: LLM returning related_transactions as array)
    with pytest.raises(ValidationError):
        Evidence(claim="x", source="y", field="z", value=[1, 2, 3])