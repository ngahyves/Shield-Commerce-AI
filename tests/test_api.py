"""Tests for FastAPI endpoints using TestClient.

All heavy dependencies (Gold, SHAP, pickles, Qdrant) are mocked in the
api_client fixture from conftest.py. Each test runs in ~100ms after the
initial import of app.main (~20s, one-time cost).

Coverage:
- liveness          (GET /health)
- readiness OK      (GET /ready -> 200)
- readiness degraded (GET /ready -> 503 when Qdrant unreachable)
- predict happy      (POST /predict -> 200, no shap_factors)
- predict 404        (POST /predict -> 404 on unknown transaction)
- explain cache hit  (POST /explain -> 200, LLM never called)
"""

from unittest.mock import patch

from app.schemas import EvidenceBundle, ExplainResponse
from src.agents.schemas import Evidence, InvestigationReport


# --- Test 1: liveness ------------------------------------------------

def test_health_returns_ok(api_client):
    """GET /health returns 200 and status=ok."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Test 2: readiness OK --------------------------------------------

def test_ready_returns_200_when_all_ok(api_client):
    """GET /ready returns 200 when all current checks pass."""
    response = api_client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["qdrant"] is True
    # Current /ready does not check Redis — this is documented in health.py
    assert "redis" not in body["checks"]


# --- Test 3: readiness degraded (Qdrant unreachable) -----------------

def test_ready_returns_503_when_qdrant_unreachable(api_client):
    """GET /ready returns 503 when Qdrant is unreachable (graceful degradation)."""
    with patch("src.vector.qdrant_client.get_client",
               side_effect=Exception("qdrant down")):
        response = api_client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["qdrant"] is False


# --- Test 4: predict happy path --------------------------------------

def test_predict_ok(api_client):
    """POST /predict returns 200 with the correct JSON shape (no shap_factors)."""
    with patch("app.routes.predict.score_transaction",
               return_value=(0.667, 0.309, "review")):
        response = api_client.post("/predict/3433766")

    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == 3433766
    assert body["decision"] == "review"
    assert body["calibrated_proba"] == 0.309
    assert body["model_version"] == "1.0.0"
    assert "thresholds" in body
    # SHAP is intentionally excluded from /predict (lives in /explain)
    assert "shap_factors" not in body


# --- Test 5: predict 404 on unknown id -------------------------------

def test_predict_404_on_unknown_id(api_client):
    """POST /predict returns 404 when scoring raises ValueError."""
    with patch("app.routes.predict.score_transaction",
               side_effect=ValueError("transaction_id 9999999 not found")):
        response = api_client.post("/predict/9999999")

    assert response.status_code == 404


# --- Test 6: explain cache hit (no LLM) ------------------------------

def test_explain_cache_hit_skips_llm(api_client):
    """POST /explain returns the cached report and never calls the LLM."""
    cached_response = _make_cached_explain_response()
    api_client.app.state.cache.get_model.return_value = cached_response

    # If investigate_transaction is called, the test fails -> cache short-circuits
    with patch("app.routes.explain.investigate_transaction",
               side_effect=AssertionError("LLM must not be called on cache hit")):
        response = api_client.post("/explain/3433766")

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "full_investigation"
    assert body["report"]["recommendation"] == "review"


# --- Helpers ---------------------------------------------------------

def _make_cached_explain_response() -> ExplainResponse:
    """Build a valid ExplainResponse for the cache-hit test.

    Fields are constrained by InvestigationReport:
    - reasoning : min_length=20
    - evidence  : min_length=1
    """
    evidence = Evidence(
        claim="model output 0.309 calibrated probability",
        source="model",
        field="calibrated_proba",
        value=0.309,
    )
    report = InvestigationReport(
        transaction_id=3433766,
        recommendation="review",
        confidence="medium",
        reasoning="This is a cached investigation report generated earlier for this transaction.",
        evidence=[evidence],
        similar_cases=[],
        suggested_actions=[],
        calibrated_proba=0.309,
        raw_score=0.667,
        model_decision="review",
        model_version="1.0.0",
        latency_ms={},
        audit_id="cached-abc",
    )
    return ExplainResponse(
        report=report,
        investigation="cached investigation",
        evidence_bundle=EvidenceBundle(
            shap_factors=[],
            customer_profile={},
            related_transactions=[],
        ),
        route="full_investigation",
        latency_ms={},
        errors=[],
    )