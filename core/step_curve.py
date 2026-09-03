"""Build exact per-request speculative decoding acceptance curves.

The only supported source is the request-local raw verify/accept arrays returned
by SGLang's ``speculative_decoding_stats=detailed`` response extension. Client
SSE chunks and algorithm-specific debug dumps are not step-level data sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_DEFAULT_NUM_BINS = 48


@dataclass(frozen=True)
class CurvePoint:
    """One token-position bin in a request curve."""

    token_position: int
    accept_length: float
    verify_length: float


@dataclass(frozen=True)
class RequestCurve:
    """One request's exact verifier steps, binned for plotting."""

    request_id: str
    num_steps: int
    num_tokens: int
    accept_length: float
    mean_verify_length: float
    points: tuple[CurvePoint, ...]


@dataclass(frozen=True)
class AcceptCurves:
    """The best and worst request in one measured benchmark run."""

    best: RequestCurve
    worst: RequestCurve
    verify_token_cap: int
    bin_width: int
    num_requests: int
    num_step_records: int


def accept_curves_to_dict(curves: AcceptCurves | None) -> dict[str, Any] | None:
    """Serialize the small plotted curve projection for a result sidecar."""

    if curves is None:
        return None

    def request_to_dict(curve: RequestCurve) -> dict[str, Any]:
        return {
            "request_id": curve.request_id,
            "num_steps": curve.num_steps,
            "num_tokens": curve.num_tokens,
            "accept_length": curve.accept_length,
            "mean_verify_length": curve.mean_verify_length,
            "points": [
                {
                    "token_position": point.token_position,
                    "accept_length": point.accept_length,
                    "verify_length": point.verify_length,
                }
                for point in curve.points
            ],
        }

    return {
        "best": request_to_dict(curves.best),
        "worst": request_to_dict(curves.worst),
        "verify_token_cap": curves.verify_token_cap,
        "bin_width": curves.bin_width,
        "num_requests": curves.num_requests,
        "num_step_records": curves.num_step_records,
    }


def accept_curves_from_dict(raw: Mapping[str, Any] | None) -> AcceptCurves | None:
    """Deserialize a sidecar curve projection, rejecting malformed data."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("curve projection must be an object or null")

    def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
        return value

    def require_float(value: Any, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be finite")
        return number

    def request_from_dict(value: Any, label: str) -> RequestCurve:
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} must be an object")
        request_id = value.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"{label}.request_id must be non-empty text")
        raw_points = value.get("points")
        if not isinstance(raw_points, list):
            raise ValueError(f"{label}.points must be a list")
        points = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, Mapping):
                raise ValueError(f"{label}.points.{index} must be an object")
            points.append(
                CurvePoint(
                    token_position=require_int(
                        point.get("token_position"),
                        f"{label}.points.{index}.token_position",
                    ),
                    accept_length=require_float(
                        point.get("accept_length"),
                        f"{label}.points.{index}.accept_length",
                    ),
                    verify_length=require_float(
                        point.get("verify_length"),
                        f"{label}.points.{index}.verify_length",
                    ),
                )
            )
        return RequestCurve(
            request_id=request_id,
            num_steps=require_int(value.get("num_steps"), f"{label}.num_steps"),
            num_tokens=require_int(value.get("num_tokens"), f"{label}.num_tokens"),
            accept_length=require_float(
                value.get("accept_length"), f"{label}.accept_length"
            ),
            mean_verify_length=require_float(
                value.get("mean_verify_length"), f"{label}.mean_verify_length"
            ),
            points=tuple(points),
        )

    return AcceptCurves(
        best=request_from_dict(raw.get("best"), "best"),
        worst=request_from_dict(raw.get("worst"), "worst"),
        verify_token_cap=require_int(
            raw.get("verify_token_cap"), "verify_token_cap", minimum=1
        ),
        bin_width=require_int(raw.get("bin_width"), "bin_width", minimum=1),
        num_requests=require_int(raw.get("num_requests"), "num_requests", minimum=1),
        num_step_records=require_int(
            raw.get("num_step_records"), "num_step_records", minimum=1
        ),
    )


@dataclass(frozen=True)
class _Step:
    accept_length: int
    verify_length: int


def _bin_width(num_tokens: int, num_bins: int) -> int:
    return max(1, math.ceil(num_tokens / num_bins))


def _request_steps(
    result: Mapping[str, Any],
) -> tuple[dict[str, list[_Step]], int]:
    response_stats = result.get("speculative_decoding_stats")
    if not (
        isinstance(response_stats, Mapping)
        and response_stats.get("schema_version") == 1
    ):
        return {}, 0

    requests = response_stats.get("requests")
    if not isinstance(requests, list):
        return {}, 0

    grouped: dict[str, list[_Step]] = {}
    num_records = 0
    for request in requests:
        if not isinstance(request, Mapping):
            continue
        request_index = request.get("request_index")
        stats = request.get("stats")
        if (
            not isinstance(request_index, int)
            or isinstance(request_index, bool)
            or request_index < 0
            or not isinstance(stats, Mapping)
            or stats.get("mode") != "detailed"
        ):
            continue
        verify_lengths = stats.get("verify_lengths")
        accept_lengths = stats.get("accept_lengths")
        if (
            not isinstance(verify_lengths, list)
            or not isinstance(accept_lengths, list)
            or len(verify_lengths) != len(accept_lengths)
        ):
            continue
        steps = []
        for verify_len, accept_len in zip(
            verify_lengths, accept_lengths, strict=True
        ):
            if (
                not isinstance(verify_len, int)
                or isinstance(verify_len, bool)
                or verify_len <= 0
                or not isinstance(accept_len, int)
                or isinstance(accept_len, bool)
                or accept_len <= 0
                or accept_len > verify_len
            ):
                continue
            steps.append(_Step(accept_len, verify_len))
        if steps:
            grouped[f"request-{request_index}"] = steps
            num_records += len(steps)
    return grouped, num_records


def build_request_curve(
    steps: Sequence[tuple[int, int] | _Step],
    *,
    request_id: str,
    bin_width: int,
) -> RequestCurve:
    """Bin one request's exact ``(acc_len, verify_len)`` step sequence."""

    normalized = [
        step if isinstance(step, _Step) else _Step(int(step[0]), int(step[1]))
        for step in steps
    ]
    bin_accept: dict[int, int] = {}
    bin_verify: dict[int, int] = {}
    bin_steps: dict[int, int] = {}
    position = 0
    for step in normalized:
        index = position // bin_width
        bin_accept[index] = bin_accept.get(index, 0) + step.accept_length
        bin_verify[index] = bin_verify.get(index, 0) + step.verify_length
        bin_steps[index] = bin_steps.get(index, 0) + 1
        position += step.accept_length

    points = tuple(
        CurvePoint(
            token_position=index * bin_width,
            accept_length=bin_accept[index] / bin_steps[index],
            verify_length=bin_verify[index] / bin_steps[index],
        )
        for index in sorted(bin_steps)
    )
    num_steps = len(normalized)
    total_verify = sum(step.verify_length for step in normalized)
    return RequestCurve(
        request_id=request_id,
        num_steps=num_steps,
        num_tokens=position,
        accept_length=position / num_steps if num_steps else 0.0,
        mean_verify_length=total_verify / num_steps if num_steps else 0.0,
        points=points,
    )


def accept_length_curves(
    result: Mapping[str, Any],
    *,
    num_bins: int = _DEFAULT_NUM_BINS,
) -> AcceptCurves | None:
    """Return exact best/worst request curves, or ``None`` without step data."""

    if num_bins <= 0:
        raise ValueError("num_bins must be greater than zero")
    grouped, num_records = _request_steps(result)
    if not grouped:
        return None

    summaries = [
        (
            request_id,
            steps,
            sum(step.accept_length for step in steps) / len(steps),
        )
        for request_id, steps in grouped.items()
        if steps
    ]
    best_rid, best_steps, _ = max(
        summaries,
        key=lambda item: (item[2], len(item[1]), item[0]),
    )
    worst_rid, worst_steps, _ = min(
        summaries,
        key=lambda item: (item[2], -len(item[1]), item[0]),
    )
    width = _bin_width(
        max(
            sum(step.accept_length for step in best_steps),
            sum(step.accept_length for step in worst_steps),
        ),
        num_bins,
    )
    verify_cap = max(step.verify_length for _, steps, _ in summaries for step in steps)
    return AcceptCurves(
        best=build_request_curve(
            best_steps,
            request_id=best_rid,
            bin_width=width,
        ),
        worst=build_request_curve(
            worst_steps,
            request_id=worst_rid,
            bin_width=width,
        ),
        verify_token_cap=verify_cap,
        bin_width=width,
        num_requests=len(summaries),
        num_step_records=num_records,
    )
