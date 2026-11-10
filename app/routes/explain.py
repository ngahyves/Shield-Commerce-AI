"""Full agentic investigation endpoint, with Redis caching."""

import time

from fastapi import APIRouter, HTTPException, Request

from app.api_logger import log_decision
from app.cache import build_explain_key
from app.metrics import (
    CACHE_HITS, CACHE_MISSES,
    PREDICTION_LATENCY, PREDICTIONS_TOTAL, TOOL_FAILURES_TOTAL,
)
from app.schemas import EvidenceBundle, ExplainResponse
from src.agents.fraud_investigator import investigate_transaction, _load_scoring_artifacts
from src.utils.logger import ProjectLogger

router = APIRouter(tags=["explain"])
log = ProjectLogger(__name__).get_logger()


@router.post("/explain/{transaction_id}", response_model=ExplainResponse)
def explain(transaction_id: int, request: Request) -> ExplainResponse:
    """Run the full agentic graph on one transaction, with Redis caching.
    """
    t0 = time.perf_counter()
    cache = request.app.state.cache
    model_version = _load_scoring_artifacts()["model_version"]
    cache_key = build_explain_key(model_version, transaction_id)

    cached = cache.get_model(cache_key, ExplainResponse)
    if cached is not None:
        log.info("explain: cache hit for tx=%d (%.0f ms)",
                 transaction_id, (time.perf_counter() - t0) * 1000)
        log_decision(
            endpoint="/explain",
            transaction_id=transaction_id,
            decision=cached.report.recommendation,
            confidence=cached.report.confidence,
            route=cached.route,
            latency_ms=(time.perf_counter() - t0) * 1000,
            cache_hit=True,
            audit_id=cached.report.audit_id,
        )
        CACHE_HITS.labels(endpoint="explain").inc()
        return cached

    try:
        state = investigate_transaction(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception("api: investigation failed for tx=%d", transaction_id)
        raise HTTPException(status_code=500, detail=f"investigation failed: {exc}")

    if not state.get("report"):
        raise HTTPException(status_code=500, detail="agent produced no report")

    PREDICTION_LATENCY.labels(endpoint="explain").observe(time.perf_counter() - t0)
    PREDICTIONS_TOTAL.labels(
        decision=state["report"]["recommendation"],
        route=state.get("route", "unknown"),
    ).inc()
    for err in state.get("errors", []):
        tool = err.split(":")[0].strip()
        TOOL_FAILURES_TOTAL.labels(tool=tool).inc()

    response = ExplainResponse(
        report=state["report"],
        investigation=state.get("investigation", ""),
        evidence_bundle=EvidenceBundle(
            shap_factors=state.get("shap_factors", []),
            customer_profile=state.get("customer", {}),
            related_transactions=state.get("related_transactions", []),
            # similar_cases removed: it already lives in report.similar_cases
        ),
        route=state.get("route", "unknown"),
        latency_ms=state.get("latency_ms", {}),
        errors=state.get("errors", []),
    )

    cache.set_model(cache_key, response)
    CACHE_MISSES.labels(endpoint="explain").inc()

    log.info(
        "api: explain tx=%d route=%s total=%.0f ms (miss)",
        transaction_id, state.get("route"), (time.perf_counter() - t0) * 1000,
    )
    log_decision(
        endpoint="/explain",
        transaction_id=transaction_id,
        decision=response.report.recommendation,
        confidence=response.report.confidence,
        route=response.route,
        latency_ms=(time.perf_counter() - t0) * 1000,
        cache_hit=False,
        audit_id=response.report.audit_id,
        errors=response.errors,
    )
    return response