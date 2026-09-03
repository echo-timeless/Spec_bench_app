"""Tests for flattening one benchmark result into selectable comparison keys.

The interesting cases all came out of the real `result.jsonl` files sitting in
``benchmark_workspace/``: `server_info` duplicates itself into
`internal_states[0]`, `--request-rate inf` serializes as a bare ``Infinity``,
and `cuda_graph_config.*.bs` is a 67-item list nobody wants as 67 keys.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from bench_app.core.benchmark_pipeline import (
    PipelineError,
    build_saved_benchmark_result,
    comparison_values,
    list_saved_benchmark_results,
    numeric_change_percent,
    save_benchmark_result,
)
from bench_app.core.result_keys import (
    CLIENT_PREFIX,
    SERVER_PREFIX,
    canonical_server_info,
    extract_comparison_values,
    flatten_scalars,
    format_value,
    group_keys,
    group_title,
    is_numeric_value,
    split_key,
)

_WORKSPACE = Path(__file__).resolve().parent.parent / "benchmark_workspace"


def _sample_result() -> dict:
    return {
        "completed": 10,
        "accept_length": 5.19,
        "request_throughput": 0.68,
        "input_throughput": 22273.70,
        "output_throughput": 2789.27,
        "total_throughput": 25062.97,
        "mean_e2e_latency_ms": 34961.04,
        "mean_ttft_ms": 4368.48,
        "mean_tpot_ms": 7.47,
        "mean_itl_ms": 7.49,
        "p99_ttft_ms": 9000.0,
        "request_rate": float("inf"),
        "backend": "sglang-oai-chat",
        "server_info": {
            "status": "ready",
            "tp_size": 4,
            "attention_backend": "flashinfer",
            "internal_states": [
                {
                    "tp_size": 4,
                    "avg_spec_accept_length": 4.27,
                    "memory_usage": {"weight": 12.5},
                    "_runtime_mutations": {"tp_size": 8},
                    "cuda_graph_config": {"decode": {"bs": list(range(67))}},
                    "disable_radix_cache": False,
                    "served_model_name": None,
                }
            ],
        },
    }


def _real_result_files() -> list[Path]:
    return sorted(_WORKSPACE.glob("*/result.jsonl"))


def test_flatten_keeps_only_finite_scalars_under_dotted_keys() -> None:
    values = flatten_scalars(
        {"a": {"b": 1, "c": None}, "d": [10, 20], "e": {"f": {"g": "x"}}},
        "server",
    )

    assert values == {
        "server.a.b": 1,
        "server.d.0": 10,
        "server.d.1": 20,
        "server.e.f.g": "x",
    }


def test_flatten_drops_long_lists_and_private_segments() -> None:
    values = flatten_scalars(
        {
            "short": [1, 2],
            "long": list(range(9)),
            "_bookkeeping": {"tp_size": 8},
            "nested": {"_in_override": 1, "keep": 2},
        }
    )

    assert values == {"short.0": 1, "short.1": 2, "nested.keep": 2}


def test_infinite_values_stay_json_serializable() -> None:
    values = flatten_scalars({"request_rate": float("inf")})

    assert values == {"request_rate": "inf"}
    assert json.dumps(values, allow_nan=False)


def test_canonical_server_info_prefers_internal_state_and_dedups() -> None:
    merged = canonical_server_info(
        {
            "tp_size": 4,
            "status": "ready",
            "internal_states": [{"tp_size": 8, "avg_spec_accept_length": 4.27}],
        }
    )

    # `internal_states[0]` wins on the 445 keys the two levels share, while the
    # six top-level-only keys such as `status` survive.
    assert merged["tp_size"] == 8
    assert merged["avg_spec_accept_length"] == 4.27
    assert merged["status"] == "ready"
    assert "internal_states" not in merged


def test_canonical_server_info_tolerates_missing_or_wrong_shapes() -> None:
    assert canonical_server_info(None) == {}
    assert canonical_server_info({"internal_states": []}) == {}
    assert canonical_server_info({"tp_size": 4, "internal_states": "oops"}) == {
        "tp_size": 4
    }


def test_extract_splits_client_and_server_without_collisions() -> None:
    values = extract_comparison_values(_sample_result())

    assert values[f"{CLIENT_PREFIX}.accept_length"] == pytest.approx(5.19)
    assert values[f"{CLIENT_PREFIX}.request_rate"] == "inf"
    assert f"{SERVER_PREFIX}.tp_size" not in values
    assert values[f"{SERVER_PREFIX}.avg_spec_accept_length"] == pytest.approx(4.27)
    assert values[f"{SERVER_PREFIX}.memory_usage.weight"] == pytest.approx(12.5)
    assert values[f"{SERVER_PREFIX}.disable_radix_cache"] is False
    # None values and SGLang's own bookkeeping never become comparison keys.
    assert f"{SERVER_PREFIX}.served_model_name" not in values
    assert not any(".cuda_graph_config.decode.bs." in key for key in values)
    assert not any("_runtime_mutations" in key for key in values)
    assert f"{CLIENT_PREFIX}.server_info" not in values


def test_output_details_arrays_never_become_comparison_keys() -> None:
    # Two requests is below `_MAX_LIST_ITEMS`, so length alone would let every
    # `--output-details` array through as `client.generated_texts.0`, ...
    values = extract_comparison_values(
        {
            **_sample_result(),
            "input_lens": [11, 12],
            "output_lens": [21, 22],
            "ttfts": [0.1, 0.2],
            "itls": [[0.03, 0.04], [0.05]],
            "itl_token_lens": [[5, 4], [6]],
            "generated_texts": ["hello", "world"],
            "errors": ["", ""],
            "speculative_decoding_stats": {
                "schema_version": 1,
                "requests": [
                    {
                        "request_index": 0,
                        "success": True,
                        "stats": {
                            "mode": "detailed",
                            "verify_lengths": [6],
                            "accept_lengths": [5],
                        },
                    }
                ],
            },
        }
    )

    assert not any(
        key.startswith(
            (
                f"{CLIENT_PREFIX}.input_lens",
                f"{CLIENT_PREFIX}.output_lens",
                f"{CLIENT_PREFIX}.ttfts",
                f"{CLIENT_PREFIX}.itls",
                f"{CLIENT_PREFIX}.itl_token_lens",
                f"{CLIENT_PREFIX}.generated_texts",
                f"{CLIENT_PREFIX}.errors",
                f"{CLIENT_PREFIX}.speculative_decoding_stats",
            )
        )
        for key in values
    )
    assert values[f"{CLIENT_PREFIX}.completed"] == 10


def test_comparison_values_injects_derived_decode_speed() -> None:
    values = comparison_values(_sample_result())

    assert values[f"{CLIENT_PREFIX}.decode_speed_toks"] == pytest.approx(133.8688)


def test_comparison_values_omits_decode_speed_without_usable_tpot() -> None:
    values = comparison_values({"completed": 1, "mean_tpot_ms": 0})

    assert f"{CLIENT_PREFIX}.decode_speed_toks" not in values


def test_group_classification_is_two_level_and_ordered() -> None:
    groups = group_keys(
        [
            f"{CLIENT_PREFIX}.mean_ttft_ms",
            f"{CLIENT_PREFIX}.output_throughput",
            f"{CLIENT_PREFIX}.accept_length",
            f"{CLIENT_PREFIX}.something_new",
            f"{SERVER_PREFIX}.tp_size",
            f"{SERVER_PREFIX}.page_size",
            f"{SERVER_PREFIX}.speculative_algorithm",
        ]
    )

    assert [(group.source, group.title) for group in groups] == [
        (CLIENT_PREFIX, "峰值与投机"),
        (CLIENT_PREFIX, "计数与吞吐"),
        (CLIENT_PREFIX, "延迟"),
        (CLIENT_PREFIX, "其他"),
        (SERVER_PREFIX, "投机解码"),
        (SERVER_PREFIX, "并行与 EP"),
        (SERVER_PREFIX, "内存与 KV"),
    ]
    assert groups[3].keys == (f"{CLIENT_PREFIX}.something_new",)


def test_group_title_and_split_key_round_trip() -> None:
    assert split_key(f"{SERVER_PREFIX}.memory_usage.weight") == (
        SERVER_PREFIX,
        "memory_usage.weight",
    )
    assert group_title(f"{CLIENT_PREFIX}.p99_itl_ms") == (CLIENT_PREFIX, "延迟")
    assert group_title("unknown.key") == ("unknown", "其他")


def test_value_typing_and_formatting() -> None:
    assert is_numeric_value(1) and is_numeric_value(1.5)
    assert not is_numeric_value(True)
    assert not is_numeric_value("4")
    assert format_value(None) == "—"
    assert format_value(True) == "true"
    assert format_value(4) == "4"
    assert format_value(7.4712) == "7.47"
    assert format_value("flashinfer") == "flashinfer"


def test_numeric_change_percent_never_raises_on_config_values() -> None:
    assert numeric_change_percent(110, 100) == pytest.approx(10.0)
    assert numeric_change_percent("flashinfer", "fa3") is None
    assert numeric_change_percent(True, False) is None
    assert numeric_change_percent(1.0, 0) is None
    assert numeric_change_percent(None, 100) is None


def test_saved_record_keeps_every_value_and_only_selects_columns(
    tmp_path: Path,
) -> None:
    record = build_saved_benchmark_result(
        _sample_result(),
        run_id="speed_bench_keys_unit",
        model_name="dsv4",
        configuration="TP=4 | DP=1 | EP=4 | flashinfer",
        configuration_options={"tp_size": 4, "dp_size": 1, "ep_size": 4},
        result_file=tmp_path / "run" / "result.jsonl",
        log_file=tmp_path / "run" / "benchmark.log",
        selected_keys=[
            f"{CLIENT_PREFIX}.mean_ttft_ms",
            f"{SERVER_PREFIX}.tp_size",  # configuration metadata, not a metric
            f"{CLIENT_PREFIX}.mean_ttft_ms",  # duplicate collapses
            "client.does_not_exist",  # unknown key is dropped
        ],
    )

    assert record["schema_version"] == 2
    assert record["selected_keys"] == [
        f"{CLIENT_PREFIX}.mean_ttft_ms",
    ]
    assert record["configuration_options"] == {
        "tp_size": 4,
        "dp_size": 1,
        "ep_size": 4,
    }
    assert record["result_name"] == ("dsv4 + TP=4 | DP=1 | EP=4 | flashinfer")
    # Selecting two columns must not shrink the measurements stored in the record.
    assert len(record["values"]) > len(record["selected_keys"])
    assert record["metrics"]["decode_speed_toks"] == pytest.approx(133.8688)

    saved_path = save_benchmark_result(record, tmp_path / "shared")
    records, warnings = list_saved_benchmark_results(tmp_path / "shared")

    assert warnings == []
    assert saved_path.name == "speed_bench_keys_unit.json"
    assert records[0]["values"] == record["values"]
    assert records[0]["selected_keys"] == record["selected_keys"]
    assert records[0]["result_name"] == record["result_name"]


def test_saved_record_rejects_non_finite_metrics_at_write_time(
    tmp_path: Path,
) -> None:
    record = build_saved_benchmark_result(
        _sample_result(),
        run_id="speed_bench_nan_unit",
        model_name="dsv4",
        configuration="",
        result_file=tmp_path / "run" / "result.jsonl",
        log_file=tmp_path / "run" / "benchmark.log",
    )
    record["values"]["client.broken"] = math.inf

    with pytest.raises(ValueError):
        save_benchmark_result(record, tmp_path / "shared")


def test_schema_v1_records_are_promoted_to_flattened_values(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "speed_bench_v1_unit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_id": "speed_bench_v1_unit",
                "run_id": "speed_bench_v1_unit",
                "saved_at": "2026-08-03T06:40:00+00:00",
                "model_name": "dsv4",
                "configuration": "old record",
                "metrics": {
                    "accept_length": None,
                    "decode_speed_toks": 133.8688,
                    "request_throughput": 0.68,
                    "input_throughput": 22273.7,
                    "output_throughput": 2789.27,
                    "total_throughput": 25062.97,
                    "mean_e2e_latency_ms": 34961.04,
                    "mean_ttft_ms": 4368.48,
                    "mean_tpot_ms": 7.47,
                    "mean_itl_ms": 7.49,
                },
                "source": {
                    "result_file": "/tmp/result.jsonl",
                    "log_file": "/tmp/x.log",
                },
                "context": {},
            }
        ),
        encoding="utf-8",
    )

    records, warnings = list_saved_benchmark_results(shared)

    assert warnings == []
    assert records[0]["schema_version"] == 1
    assert records[0]["result_name"] == "dsv4 + old record"
    assert records[0]["configuration_options"] == {}
    # The 10 v1 metrics become `client.*` keys so the table has one lookup path.
    assert records[0]["values"][f"{CLIENT_PREFIX}.mean_ttft_ms"] == pytest.approx(
        4368.48
    )
    assert f"{CLIENT_PREFIX}.accept_length" not in records[0]["values"]
    assert f"{CLIENT_PREFIX}.decode_speed_toks" in records[0]["selected_keys"]


def test_unknown_schema_version_is_reported_as_a_warning(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "speed_bench_future.json").write_text(
        json.dumps({"schema_version": 99, "run_id": "speed_bench_future"}),
        encoding="utf-8",
    )

    records, warnings = list_saved_benchmark_results(shared)

    assert records == []
    assert len(warnings) == 1


@pytest.mark.skipif(not _real_result_files(), reason="no recorded benchmark run")
def test_real_result_files_flatten_and_serialize() -> None:
    for result_file in _real_result_files():
        raw = json.loads(result_file.read_text(encoding="utf-8").splitlines()[-1])
        values = comparison_values(raw)

        # Real runs may come from a minimal/mocked server_info response or a
        # full SGLang service with hundreds of flattened settings. The schema
        # contract is the presence and serializability of benchmark metrics,
        # not a deployment-dependent number of server keys.
        assert values
        assert json.dumps(values, allow_nan=False)
        assert f"{CLIENT_PREFIX}.mean_tpot_ms" in values
        assert f"{SERVER_PREFIX}.tp_size" not in values
        # Every key lands in exactly one non-empty group.
        groups = group_keys(values)
        assert sum(len(group.keys) for group in groups) == len(values)
        assert all(group.keys for group in groups)


@pytest.mark.skipif(not _real_result_files(), reason="no recorded benchmark run")
def test_configured_default_keys_resolve_against_a_real_record() -> None:
    from bench_app.core.benchmark_pipeline import load_settings

    settings = load_settings()
    values = comparison_values(
        json.loads(_real_result_files()[0].read_text(encoding="utf-8").splitlines()[-1])
    )

    assert settings.default_comparison_keys
    missing = [key for key in settings.default_comparison_keys if key not in values]
    assert missing == []


def test_comparison_values_rejects_an_unbounded_key_explosion() -> None:
    exploded = {f"key_{index}": index for index in range(5000)}

    with pytest.raises(PipelineError):
        comparison_values({"completed": 1, **exploded})
