"""
Observability for the HSAL API layer.

- Structured JSON logs: one event per request, machine-parseable.
- Prometheus metrics: request counters by resolution path, latency
  histograms, bypass counters. Exposed at /metrics.

Deliberately lives outside hsal.core: the router stays dependency-free
and returns enough information (source, latency) for the API layer to
do all instrumentation.
"""

import json
import logging
import uuid

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# --- Metrics ---

REQUESTS_TOTAL = Counter(
    "hsal_requests_total",
    "Requests by resolution path",
    ["path"],
)
CACHE_BYPASS_TOTAL = Counter(
    "hsal_cache_bypass_total",
    "Requests that bypassed the cache",
    ["reason"],
)
REQUEST_LATENCY = Histogram(
    "hsal_request_latency_seconds",
    "End-to-end request latency by resolution path",
    ["path"],
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# --- Structured logging ---

logger = logging.getLogger("hsal.requests")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def record_request(
    request_id: str,
    path: str,
    cacheable: bool,
    latency_ms: float,
    similarity_score: float | None = None,
    error: str | None = None,
) -> None:
    """Update metrics and emit one structured JSON log line."""
    REQUESTS_TOTAL.labels(path=path).inc()
    REQUEST_LATENCY.labels(path=path).observe(latency_ms / 1000.0)
    if not cacheable:
        CACHE_BYPASS_TOTAL.labels(reason="caller_opt_out").inc()

    event = {
        "request_id": request_id,
        "path": path,
        "cacheable": cacheable,
        "latency_ms": round(latency_ms, 2),
    }
    if similarity_score is not None:
        event["similarity_score"] = round(similarity_score, 4)
    if error is not None:
        event["error"] = error
    logger.info(json.dumps(event))


def metrics_response() -> tuple:
    """(payload, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
