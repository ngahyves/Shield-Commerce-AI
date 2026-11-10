"""Fast supervised prediction endpoint with SHAP explanations."""

import time

from fastapi import APIRouter, HTTPException

from app.api_logger import log_decision
from app.metrics import MODEL_SCORE_DIST, PREDICTION_LATENCY, PREDICTIONS_TOTAL
from src.agents.fraud_investigator import _load_scoring_artifacts, score_transaction
from app.schemas import PredictResponse
from src.utils.logger import ProjectLogger

router = APIRouter(tags=["predict"])
log = ProjectLogger(__name__).get_logger()


@router.post("/predict/{transaction_id}", response_model=PredictResponse)
def predict(transaction_id: int) -> PredictResponse:
    """Score one transaction. No LLM, no SHAP. ~50-150 ms."""
    t0 = time.perf_counter()
    try:
        raw, calibrated, decision = score_transaction(transaction_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception("api: prediction failed for tx=%d", transaction_id)
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}")

    arts = _load_scoring_artifacts()
    PREDICTIONS_TOTAL.labels(decision=decision, route="predict_only").inc()
    MODEL_SCORE_DIST.observe(calibrated)

    log.info(
        "api: predict tx=%d cal=%.4f decision=%s",
        transaction_id, calibrated, decision,
    )
    log_decision(
        endpoint="/predict",
        transaction_id=transaction_id,
        decision=decision,
        route="predict_only",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    return PredictResponse(
        transaction_id=transaction_id,
        raw_score=raw,
        calibrated_proba=calibrated,
        decision=decision,
        model_name=arts["model_name"],
        model_version=arts["model_version"],
        thresholds=arts["thresholds"],
    )