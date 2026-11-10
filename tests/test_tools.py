"""Tests for deterministic tools and the reporter fallback.

- _gold() is mocked so tests run without loading the 636 MB Parquet.
- get_customer is a LangChain @tool so we use .invoke({...}).
- get_transaction_frame is a plain function -> direct call.
"""

from unittest.mock import patch

import pytest

from src.agents.tools import get_customer, get_transaction_frame
from src.agents.reporter import _fallback_report


# --- Test 1: unknown transaction_id -> ValueError (404 path) ---------

def test_get_transaction_frame_unknown_id_raises(sample_gold_df):
    """Unknown transaction_id raises ValueError (maps to HTTP 404)."""
    with patch("src.agents.tools._gold", return_value=sample_gold_df):
        with pytest.raises(ValueError, match="not found"):
            get_transaction_frame(9999999)


# --- Test 2: new customer profile -------------------------------

def test_get_customer_new_customer_profile(sample_gold_df):
    """Customer with no prior transactions -> is_new_customer=True."""
    with patch("src.agents.tools._gold", return_value=sample_gold_df):
        # Customer "c" has one tx at dt=432000. Query at dt=432000 -> no prior.
        result = get_customer.invoke({"uid": "c", "transaction_dt": 432000})

    assert result["is_new_customer"] is True
    assert result["n_transactions"] == 0


# --- Test 3: reporter fallback produces valid shape -----------

def test_fallback_report_has_low_confidence(mock_agent_state):
    """_fallback_report produces a valid report with confidence=low."""
    report = _fallback_report(mock_agent_state)

    assert report["confidence"] == "low"
    assert report["recommendation"] == mock_agent_state["model_decision"]
    assert report["audit_id"].startswith("fallback-")
    assert len(report["evidence"]) == 1
    assert report["evidence"][0]["source"] == "model"