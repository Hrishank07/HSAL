import json
import logging

import pytest

from hsal.observability import (
    CACHE_BYPASS_TOTAL,
    REQUESTS_TOTAL,
    metrics_response,
    new_request_id,
    record_request,
)


def counter_value(counter, **labels):
    return counter.labels(**labels)._value.get()


@pytest.fixture
def hsal_caplog(caplog):
    """The hsal.requests logger doesn't propagate to root, so attach
    pytest's capture handler to it directly."""
    logger = logging.getLogger("hsal.requests")
    logger.addHandler(caplog.handler)
    yield caplog
    logger.removeHandler(caplog.handler)


def test_request_id_format_and_uniqueness():
    a, b = new_request_id(), new_request_id()
    assert a.startswith("req_") and len(a) == 16
    assert a != b


def test_record_request_increments_path_counter():
    before = counter_value(REQUESTS_TOTAL, path="L1_EXACT")
    record_request("req_test", "L1_EXACT", True, 1.5)
    assert counter_value(REQUESTS_TOTAL, path="L1_EXACT") == before + 1


def test_bypass_counter_only_for_non_cacheable():
    before = counter_value(CACHE_BYPASS_TOTAL, reason="caller_opt_out")
    record_request("req_a", "LLM_GENERATED", True, 5.0)
    assert counter_value(CACHE_BYPASS_TOTAL, reason="caller_opt_out") == before
    record_request("req_b", "LLM_GENERATED", False, 5.0)
    assert counter_value(CACHE_BYPASS_TOTAL, reason="caller_opt_out") == before + 1


def test_log_line_is_valid_json_with_expected_fields(hsal_caplog):
    record_request("req_json", "L2_SEMANTIC", True, 12.34, similarity_score=0.9234)

    event = json.loads(hsal_caplog.records[-1].message)
    assert event == {
        "request_id": "req_json",
        "path": "L2_SEMANTIC",
        "cacheable": True,
        "latency_ms": 12.34,
        "similarity_score": 0.9234,
    }


def test_error_field_logged(hsal_caplog):
    record_request("req_err", "ERROR", True, 0.0, error="boom")
    event = json.loads(hsal_caplog.records[-1].message)
    assert event["error"] == "boom"
    assert event["path"] == "ERROR"


def test_metrics_response_is_prometheus_format():
    payload, content_type = metrics_response()
    assert b"hsal_requests_total" in payload
    assert b"hsal_request_latency_seconds" in payload
    assert "text/plain" in content_type
