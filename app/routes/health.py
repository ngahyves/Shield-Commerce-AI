"""Health and readiness endpoints."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.utils.logging_config import ConfigLoader

router = APIRouter(tags=["health"])
cfg = ConfigLoader()


@router.get("/health")
def health() -> dict:
    """Liveness: process is alive. Never touches disk or network."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> JSONResponse:
    """Readiness: all dependencies are reachable."""
    checks: dict[str, bool | str] = {}
    models_dir = Path(cfg.get("paths", "models"))

    for name in ("classifier.pkl", "target_encoder.pkl", "calibrator.pkl",
                 "model_metadata.json"):
        checks[name] = (models_dir / name).exists()

    gold_path = Path(cfg.get("paths", "gold_parquet"))
    checks["gold_parquet"] = gold_path.exists()

    try:
        from src.vector.qdrant_client import get_client
        collection = cfg.get("qdrant", "collection")
        names = [c.name for c in get_client().get_collections().collections]
        checks["qdrant"] = collection in names
    except Exception as exc:
        checks["qdrant"] = False
        checks["qdrant_error"] = str(exc)[:120]

    healthy = all(v is True for v in checks.values() if isinstance(v, bool))
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ready" if healthy else "degraded", "checks": checks},
    )