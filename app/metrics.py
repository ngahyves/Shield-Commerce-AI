"""Custom Prometheus metrics."""

from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Number of predictions", ["decision", "route"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end latency",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0),
)
TOOL_FAILURES_TOTAL = Counter(
    "agent_tool_failures_total", "Tool failures during retrieval", ["tool"],
)
MODEL_SCORE_DIST = Histogram(
    "model_calibrated_score", "Distribution of calibrated probabilities",
    buckets=[0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95, 0.99],
)
CACHE_HITS = Counter(
    "cache_hits_total", "Cache hits", ["endpoint"],
)
CACHE_MISSES = Counter(
    "cache_misses_total", "Cache misses", ["endpoint"],
)
