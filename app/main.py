"""ShieldCommerce API: FastAPI entry point."""

from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager

from fastapi import FastAPI
import os
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.cache import RedisCache
from app.routes import explain, health, predict
from src.utils.logging_config import ConfigLoader
from src.utils.logger import ProjectLogger

log = ProjectLogger(__name__).get_logger()
cfg = ConfigLoader()

# ── Example transaction IDs from the Gold test set ───────────────────
# Kept here so they show up in the Swagger UI description.
# Source: src/scripts/find_review_case.py + manual test runs.

EXAMPLE_IDS = {
    "review_fraud": 3433766,       # is_fraud=1, calibrated=0.309 → full_investigation
    "review_nonfraud": 3361860,    # is_fraud=0, calibrated=0.638 → full_investigation
    "approve": 2987000,            # is_fraud=0, calibrated=0.030 → fast_approve
}

DESCRIPTION = f"""
## ShieldCommerce AI — Fraud Investigation API

Agentic fraud detection: LightGBM classifier + multi-agent LangGraph pipeline.

### Endpoints
- `POST /score/{{transaction_id}}` — fast supervised prediction (~100 ms, no LLM)
- `POST /explain/{{transaction_id}}` — full agentic investigation (6-8 s cold, ~50 ms cache hit)
- `GET /health` — liveness probe
- `GET /ready` — readiness probe (artifacts, Gold Parquet, Qdrant, Redis)

### Example transaction IDs

| ID | `is_fraud` | `calibrated_proba` | Route |
|----|------------|--------------------|-------|
| `{EXAMPLE_IDS["review_fraud"]}` | 1 | 0.048 | `full_investigation` (LLM) |
| `{EXAMPLE_IDS["review_nonfraud"]}` | 0 |  0.082 | `full_investigation` (LLM) |
| `{EXAMPLE_IDS["approve"]}` | 0 |  0.014 | `fast_approve` (template, no LLM) |

**Try in Swagger:**
- `POST /explain/{EXAMPLE_IDS["review_fraud"]}` → full path, ~14 s cold, then ~50 ms (cache)
- `POST /explain/{EXAMPLE_IDS["approve"]}` → fast path, ~100 ms
- `POST /score/{EXAMPLE_IDS["approve"]}` → supervised only, ~100 ms
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload artifacts and initialize Redis."""
    log.info("startup: preloading artifacts")
    try:
        from src.agents.fraud_investigator import _load_scoring_artifacts
        from src.agents.graph import build_graph
        from src.agents.tools import _gold, _shap_explainer

        _load_scoring_artifacts()
        log.info("startup: scoring artifacts loaded")

        _gold()
        log.info("startup: gold parquet loaded")

        build_graph()
        log.info("startup: graph compiled")

        _shap_explainer()
        log.info("startup: shap explainer warmed")
    except Exception as exc:
        log.error("startup: preload failed — %s", exc)

    redis_url = os.getenv("REDIS_URL", cfg.get("cache", "redis_url", default="redis://localhost:6380/0"))
    ttl = cfg.get("cache", "ttl_seconds", default=86400)
    cache = RedisCache(redis_url, default_ttl=ttl)
    if cache.ping():
        log.info("startup: redis connected (%s)", redis_url)
    else:
        log.warning("startup: redis unreachable at %s — running without cache", redis_url)
    app.state.cache = cache

    yield

    cache.close()
    log.info("shutdown")


app = FastAPI(
    title=cfg.get("app", "name"),
    version=cfg.get("app", "version"),
    description=DESCRIPTION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentator initialisation

Instrumentator().instrument(app).expose(
    app,
    include_in_schema=True,
    tags=["Prometheus endpoint"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(explain.router)