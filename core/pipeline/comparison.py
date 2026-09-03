"""Known result metrics and generic flat-value comparison helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

from bench_app.core.pipeline.models import PipelineError
from bench_app.core.result_keys import (
    CLIENT_PREFIX,
    extract_comparison_values,
    is_numeric_value,
)

_SAVED_METRIC_KEYS = (
    "accept_length",
    "decode_speed_toks",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
)

_REQUIRED_RESULT_KEYS = (
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_e2e_latency_ms",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "mean_itl_ms",
)

_MAX_COMPARISON_VALUES = 4096


def result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Select the key SGLang benchmark metrics used by the result panel."""

    keys = (
        "completed",
        "duration",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "p95_ttft_ms",
        "p95_tpot_ms",
        "p95_e2e_latency_ms",
        "accept_length",
    )
    return {key: result.get(key) for key in keys}


def _finite_float(
    value: Any,
    *,
    label: str,
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool):
        raise PipelineError(f"Result metric {label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Result metric {label} must be numeric") from exc
    if not math.isfinite(number):
        raise PipelineError(f"Result metric {label} must be finite")
    return number


def benchmark_result_metrics(result: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract the comparison metrics from one SGLang benchmark result."""

    completed = _finite_float(result.get("completed"), label="completed")
    if completed is None or completed <= 0:
        raise PipelineError("SGLang result has no successful requests")

    metrics: dict[str, float | None] = {
        key: _finite_float(result.get(key), label=key) for key in _REQUIRED_RESULT_KEYS
    }
    accept_length = _finite_float(
        result.get("accept_length"),
        label="accept_length",
        allow_none=True,
    )
    mean_tpot_ms = metrics["mean_tpot_ms"]
    if mean_tpot_ms is None or mean_tpot_ms <= 0:
        raise PipelineError("Result metric mean_tpot_ms must be greater than zero")

    return {
        "accept_length": accept_length,
        "decode_speed_toks": 1000.0 / mean_tpot_ms,
        **metrics,
    }


def comparison_values(result: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one raw result into every comparable key, plus derived ones.

    `decode_speed_toks` is BenchAPP's own derivation rather than an SGLang key,
    so it is injected here to keep the page working off a single flat mapping.
    """

    values = extract_comparison_values(result)
    if len(values) > _MAX_COMPARISON_VALUES:
        raise PipelineError(
            f"Result exposes {len(values)} comparable keys, above the "
            f"{_MAX_COMPARISON_VALUES} supported by one saved record"
        )
    mean_tpot_ms = result.get("mean_tpot_ms")
    if is_numeric_value(mean_tpot_ms) and mean_tpot_ms > 0:
        values[f"{CLIENT_PREFIX}.decode_speed_toks"] = 1000.0 / float(mean_tpot_ms)
    return values


def relative_change_percent(
    current: Any,
    baseline: Any,
) -> float | None:
    """Return the signed relative change used in the comparison table."""

    current_value = _finite_float(current, label="current", allow_none=True)
    baseline_value = _finite_float(baseline, label="baseline", allow_none=True)
    if current_value is None or baseline_value in (None, 0.0):
        return None
    return (current_value - baseline_value) / baseline_value * 100.0


def numeric_change_percent(current: Any, baseline: Any) -> float | None:
    """Relative change for arbitrary flattened values, never raising.

    ``relative_change_percent`` rejects non-numerics because it guards the
    curated metric set. Flattened records also hold strings and booleans, where
    "no percentage" is the answer rather than an error.
    """

    if not (is_numeric_value(current) and is_numeric_value(baseline)):
        return None
    return relative_change_percent(current, baseline)
