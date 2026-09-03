"""Tests for exact request-local speculative decoding step curves."""

from __future__ import annotations

import pytest

from bench_app.core.step_curve import (
    accept_curves_from_dict,
    accept_curves_to_dict,
    accept_length_curves,
    build_request_curve,
)


def _response_result(*requests: tuple[list[int], list[int]]) -> dict:
    return {
        # Deliberately contradictory client chunks: exact curves must ignore it.
        "itl_token_lens": [[99] for _ in requests],
        "speculative_decoding_stats": {
            "schema_version": 1,
            "num_requests": len(requests),
            "num_requests_with_stats": len(requests),
            "requests": [
                {
                    "request_index": index,
                    "success": True,
                    "stats": {
                        "mode": "detailed",
                        "verify_lengths": verify_lengths,
                        "accept_lengths": accept_lengths,
                    },
                }
                for index, (verify_lengths, accept_lengths) in enumerate(requests)
            ],
        },
    }


def test_curves_use_response_stats_and_pick_acceptance_extremes() -> None:
    result = _response_result(
        ([6, 6, 6], [4, 5, 3]),
        ([6, 6, 6], [1, 1, 2]),
    )

    curves = accept_length_curves(result, num_bins=2)

    assert curves is not None
    assert curves.num_requests == 2
    assert curves.num_step_records == 6
    assert curves.verify_token_cap == 6
    assert curves.bin_width == 6
    assert curves.best.request_id == "request-0"
    assert curves.best.accept_length == pytest.approx(4.0)
    assert curves.best.num_steps == 3
    assert curves.best.num_tokens == 12
    assert curves.worst.request_id == "request-1"
    assert curves.worst.accept_length == pytest.approx(4 / 3)
    assert max(point.accept_length for point in curves.best.points) <= 6


def test_x_axis_advances_by_committed_tokens_and_keeps_verify_length() -> None:
    curve = build_request_curve(
        [(4, 6), (2, 5), (3, 6)],
        request_id="request-7",
        bin_width=4,
    )

    assert curve.num_tokens == 9
    assert curve.num_steps == 3
    assert curve.accept_length == 3.0
    assert curve.mean_verify_length == pytest.approx(17 / 3)
    assert [point.token_position for point in curve.points] == [0, 4]
    assert curve.points[0].accept_length == 4.0
    assert curve.points[0].verify_length == 6.0
    assert curve.points[1].accept_length == 2.5
    assert curve.points[1].verify_length == 5.5


def test_invalid_or_non_detailed_response_stats_are_ignored() -> None:
    assert accept_length_curves({"completed": 1}) is None
    assert (
        accept_length_curves(
            {
                "speculative_decoding_stats": {
                    "schema_version": 1,
                    "requests": [
                        {
                            "request_index": 0,
                            "stats": {
                                "mode": "summary",
                                "verify_lengths": [6],
                                "accept_lengths": [5],
                            },
                        }
                    ],
                }
            }
        )
        is None
    )
    assert (
        accept_length_curves(
            _response_result(([6, 5], [4]),)
        )
        is None
    )


def test_invalid_steps_are_skipped_without_shifting_requests() -> None:
    curves = accept_length_curves(
        _response_result(
            ([6, 4, 3], [5, 5, 0]),
            ([4], [2]),
        )
    )

    assert curves is not None
    assert curves.num_requests == 2
    assert curves.num_step_records == 2
    assert curves.best.request_id == "request-0"
    assert curves.best.num_steps == 1
    assert curves.worst.request_id == "request-1"


def test_curve_projection_round_trip() -> None:
    curves = accept_length_curves(_response_result(([6, 5], [4, 3]),))

    assert curves is not None
    assert accept_curves_from_dict(accept_curves_to_dict(curves)) == curves

    malformed = accept_curves_to_dict(curves)
    assert malformed is not None
    malformed["best"]["request_id"] = ""
    with pytest.raises(ValueError, match="request_id"):
        accept_curves_from_dict(malformed)


def test_invalid_bin_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        accept_length_curves(_response_result(([6], [5]),), num_bins=0)
