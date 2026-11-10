"""Structured JSONL logger for API decisions.

Each call to /score or /explain writes one JSON line to logs/api_decisions.jsonl.
Format is line-delimited JSON so it can be tailed, grepped, or ingested by
Elasticsearch/Loki without parsing the free-text shieldcommerce.log.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_config import ConfigLoader


def _log_path() -> Path:
    cfg = ConfigLoader()
    logs_dir = Path(cfg.get("paths", "logs", default="logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "api_decisions.jsonl"


def log_decision(
    endpoint: str,
    transaction_id: int,
    decision: str,
    route: str,
    latency_ms: float,
    confidence: str | None = None,
    cache_hit: bool = False,
    audit_id: str | None = None,
    errors: list[str] | None = None,
) -> None:
    """Append one decision event to logs/api_decisions.jsonl. Never raises."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "transaction_id": transaction_id,
        "decision": decision,
        "confidence": confidence,
        "route": route,
        "latency_ms": round(latency_ms, 1),
        "cache_hit": cache_hit,
        "audit_id": audit_id,
        "errors": errors or [],
    }
    try:
        with open(_log_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # a logging failure must not break the API