# tests/conftest.py
"""Fixtures partagées. Pas de test ici, juste des données réutilisables."""

import pandas as pd
import pytest
from langchain_core.messages import HumanMessage


@pytest.fixture
def sample_gold_df():
    """A tiny fake Gold file (5 lines) to test the tools without loading 636 MB."""
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "uid": ["a", "a", "b", "b", "c"],
        "transaction_dt": [86400, 172800, 259200, 345600, 432000],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "is_fraud": [0, 1, 0, 0, 1],
    }).set_index("transaction_id", drop=False)


@pytest.fixture
def mock_agent_state():
    """A filled agent state—what comes out of retrieval_node."""
    return {
        "transaction_id": 3433766,
        "calibrated_proba": 0.309,
        "raw_score": 0.667,
        "model_decision": "review",
        "model_version": "1.0.0",
        "transaction": {"amount": 554.0},
        "customer": {"is_new_customer": True, "n_transactions": 0},
        "related_transactions": [],
        "shap_factors": [],
        "similar_cases": [],
        "investigation": "test investigation",
        "errors": [],
        "latency_ms": {},
    }

@pytest.fixture
def mock_messages():
    """Minimal LangChain message list for LLM client tests."""
    from langchain_core.messages import HumanMessage
    return [HumanMessage(content="test")]

@pytest.fixture
def api_client():
    """TestClient with heavy lifespan dependencies mocked.

    Prevents loading:
    - 636 MB Gold Parquet (~2s)
    - SHAP TreeExplainer (~7s)
    - 3 pickles (~3s)
    - Qdrant connection

    Tests run in ~100ms per request instead of ~12s startup.
    """
    from unittest.mock import MagicMock, patch
    from fastapi.testclient import TestClient
    from app.main import app

    mock_arts = {
        "model_version": "1.0.0",
        "thresholds": {"block": 0.7346, "review": 0.0412},
        "model_name": "lightgbm",
        "onnx_sessions": None,
    }

    # Qdrant mock: collection is present, so /ready returns OK
    mock_qdrant = MagicMock()
    mock_collection = MagicMock()
    mock_collection.name = "shieldcommerce_fraud_cases"
    mock_qdrant.get_collections.return_value.collections = [mock_collection]

    with patch("src.agents.fraud_investigator._load_scoring_artifacts",
               return_value=mock_arts), \
         patch("src.agents.tools._gold", return_value=pd.DataFrame()), \
         patch("src.agents.tools._shap_explainer", return_value=MagicMock()), \
         patch("src.agents.graph.build_graph", return_value=MagicMock()), \
         patch("app.routes.predict._load_scoring_artifacts",
               return_value=mock_arts), \
         patch("app.routes.explain._load_scoring_artifacts",
               return_value=mock_arts), \
         patch("src.vector.qdrant_client.get_client",
               return_value=mock_qdrant):

        with TestClient(app) as client:
            # Override the cache created by lifespan
            app.state.cache = MagicMock()
            app.state.cache.ping.return_value = True
            app.state.cache.get_model.return_value = None  # default: cache miss
            yield client