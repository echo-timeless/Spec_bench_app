"""Flatten one SGLang benchmark result into selectable comparison keys.

`result.jsonl` mixes three very different things in one JSON object: the client
side measurements, the arguments the run was launched with, and a full
`/server_info` snapshot. This module turns that into a flat ``dotted.key ->
scalar`` mapping plus a two-level classification the page renders as a picker.

No Streamlit dependency, so the whole mapping is unit testable against the real
`result.jsonl` files already sitting in ``benchmark_workspace/``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

CLIENT_PREFIX = "client"
SERVER_PREFIX = "server"
CONFIGURATION_ONLY_KEYS = frozenset(
    {
        f"{SERVER_PREFIX}.tp_size",
        f"{SERVER_PREFIX}.dp_size",
        f"{SERVER_PREFIX}.ep_size",
    }
)

# `internal_states[0]` and the `server_info` top level overlap by 445 keys on a
# real record. Longer lists are dropped instead of exploding into hundreds of
# indexed keys; on real records this only sacrifices `cuda_graph_config.*.bs`.
_MAX_LIST_ITEMS = 8

# `--output-details` merges per-request arrays into the same `result.jsonl`
# object. They are per-request detail, never a run-level scalar worth comparing,
# so they are dropped by name rather than by length: `_MAX_LIST_ITEMS` alone
# only hides them while `--num-prompts` stays above 8, so a small smoke run
# would silently save hundreds of `client.generated_texts.N` keys.
_DETAIL_KEYS = frozenset(
    {
        "input_lens",
        "output_lens",
        "ttfts",
        "itls",
        "itl_token_lens",
        "generated_texts",
        "errors",
        "cached_tokens",
        "cached_tokens_details",
        "speculative_decoding_stats",
    }
)

_VALUE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.\[\]-]{1,160}$")


@dataclass(frozen=True)
class KeyGroup:
    """One second-level group of keys shown under a source tab."""

    source: str
    title: str
    keys: tuple[str, ...]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str))


def is_numeric_value(value: Any) -> bool:
    """Report whether a flattened value participates in percentage compare."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_scalar(value: Any) -> Any:
    """Keep records valid JSON by demoting inf/nan to their text form.

    ``--request-rate inf`` really does land in `result.jsonl` as the bare token
    ``Infinity``, which ``json.dumps(allow_nan=False)`` rejects. Storing it as
    ``"inf"`` matches the string the form already holds for that parameter.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _flatten_into(
    value: Any,
    prefix: str,
    out: dict[str, Any],
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_into(child, f"{prefix}{key}." if prefix else f"{key}.", out)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LIST_ITEMS:
            return
        for index, child in enumerate(value):
            _flatten_into(child, f"{prefix}{index}.", out)
        return
    if value is None or not _is_scalar(value):
        return
    key = prefix[:-1] if prefix.endswith(".") else prefix
    if not key or not _VALUE_KEY_PATTERN.fullmatch(key):
        return
    # `_resolved_overrides`, `_runtime_mutations`, `_in_override`, ... are
    # SGLang's own bookkeeping; they are not comparable run configuration.
    if any(segment.startswith("_") for segment in key.split(".")):
        return
    out[key] = _normalize_scalar(value)


def flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into dotted keys, keeping only finite scalars."""

    out: dict[str, Any] = {}
    _flatten_into(value, f"{prefix}." if prefix else "", out)
    return out


def canonical_server_info(server_info: Any) -> dict[str, Any]:
    """Merge `/server_info` with `internal_states[0]`, dropping duplicates.

    `internal_states[0]` wins because it carries the runtime fields
    (`avg_spec_accept_length`, `last_gen_throughput`, `memory_usage`, ...) that
    the `server_info` top level lacks.
    """

    if not isinstance(server_info, Mapping):
        return {}
    states = server_info.get("internal_states")
    first_state = (
        states[0]
        if isinstance(states, (list, tuple))
        and states
        and isinstance(states[0], Mapping)
        else {}
    )
    merged: dict[str, Any] = dict(first_state)
    for key, value in server_info.items():
        if key == "internal_states":
            continue
        merged.setdefault(key, value)
    return merged


def extract_comparison_values(result: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one raw benchmark result into `client.*` / `server.*` keys."""

    client_part = {
        key: value
        for key, value in result.items()
        if key != "server_info" and key not in _DETAIL_KEYS
    }
    values = flatten_scalars(client_part, CLIENT_PREFIX)
    server_values = flatten_scalars(
        canonical_server_info(result.get("server_info")),
        SERVER_PREFIX,
    )
    # TP/DP/EP belong to the user-authored result configuration, not to the
    # benchmark metric picker. They are captured separately at save time.
    for key in CONFIGURATION_ONLY_KEYS:
        server_values.pop(key, None)
    values.update(server_values)
    return values


_OTHER_TITLE = "其他"

# First match wins, so order is meaningful: narrower patterns come first.
_GROUP_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        CLIENT_PREFIX,
        "延迟",
        r"^(mean|median|std|p\d+|max)_(e2e_latency_ms|ttft_ms|tpot_ms|itl_ms)$",
    ),
    (
        CLIENT_PREFIX,
        "峰值与投机",
        r"^(concurrency|accept_length|max_output_tokens_per_s"
        r"|max_concurrent_requests)$",
    ),
    (
        CLIENT_PREFIX,
        "计数与吞吐",
        r"^(duration|completed|total_input\w*|total_output\w*|\w*throughput\w*)$",
    ),
    (
        CLIENT_PREFIX,
        "参数回填",
        r"^(tag|backend|dataset_name|request_rate|max_concurrency"
        r"|sharegpt_output_len|random_\w+)$",
    ),
    (
        SERVER_PREFIX,
        "运行时状态",
        r"^(last_gen_throughput|avg_spec_accept_length"
        r"|effective_max_running_requests_per_dp|status|version"
        r"|max_total_num_tokens|max_req_input_len)$",
    ),
    (
        SERVER_PREFIX,
        "投机解码",
        r"^(speculative_\w+|max_speculative_\w+|decoupled_spec\w*|spec_trace\w*"
        r"|enable_multi_layer_eagle)$",
    ),
    (
        SERVER_PREFIX,
        "并行与 EP",
        r"^(tp_size|dp_size|pp_\w+|ep_\w+|ep_size|dcp_size|attn_cp_\w+|dwdp_size"
        r"|moe_\w+|deepep_\w+|fuseep_\w+|eplb_\w+|elastic_ep_\w+|max_ep_size"
        r"|expert_\w+|nnodes|node_rank|enable_dp_\w+|enable_ep_\w+"
        r"|enable_eplb|enable_prefill_cp|cp_strategy|load_balance_method)$",
    ),
    (
        SERVER_PREFIX,
        "内存与 KV",
        r"^(mem_fraction_static|max_total_tokens|max_running_requests"
        r"|max_queued_requests|page_size|kv_cache_dtype|chunked_prefill_size"
        r"|max_prefill_tokens|swa_\w+|disable_radix_cache|radix_\w+|hicache_\w+"
        r"|hisparse\w*|memory_usage\.\w+|cpu_offload_gb|offload_\w+"
        r"|enable_hierarchical_cache|enable_unified_memory)$",
    ),
    (
        SERVER_PREFIX,
        "计算后端",
        r"^(\w*attention_backend|sampling_backend|grammar_backend|mamba_backend"
        r"|\w*gemm\w*|moe_runner_backend|linear_attn\w*|dsa_\w+"
        r"|radix_cache_backend)$",
    ),
    (
        SERVER_PREFIX,
        "CUDA Graph",
        r"^(cuda_graph\w*|disable_\w*cuda_graph\w*|enable_cudagraph_gc"
        r"|enable_profile_cuda_graph|debug_cuda_graph)$",
    ),
    (
        SERVER_PREFIX,
        "模型与 Tokenizer",
        r"^(model_\w+|tokenizer_\w+|detokenizer_\w+|served_model_name|dtype"
        r"|quantization\w*|context_length|load_format|trust_remote_code"
        r"|revision|weight_version|skip_tokenizer_init)$",
    ),
)
_GROUP_ORDER: dict[str, tuple[str, ...]] = {
    CLIENT_PREFIX: (
        "峰值与投机",
        "计数与吞吐",
        "延迟",
        "参数回填",
        _OTHER_TITLE,
    ),
    SERVER_PREFIX: (
        "运行时状态",
        "投机解码",
        "并行与 EP",
        "内存与 KV",
        "计算后端",
        "CUDA Graph",
        "模型与 Tokenizer",
        _OTHER_TITLE,
    ),
}
_COMPILED_GROUPS = tuple(
    (source, title, re.compile(pattern)) for source, title, pattern in _GROUP_PATTERNS
)


def split_key(key: str) -> tuple[str, str]:
    """Split ``client.mean_ttft_ms`` into ``("client", "mean_ttft_ms")``."""

    source, _, remainder = key.partition(".")
    return source, remainder


def group_title(key: str) -> tuple[str, str]:
    """Return the ``(source, group title)`` this flattened key belongs to."""

    source, remainder = split_key(key)
    for group_source, title, pattern in _COMPILED_GROUPS:
        if group_source == source and pattern.fullmatch(remainder):
            return source, title
    return source, _OTHER_TITLE


def group_keys(keys: Iterable[str]) -> list[KeyGroup]:
    """Classify flattened keys into ordered, non-empty two-level groups."""

    buckets: dict[tuple[str, str], list[str]] = {}
    for key in keys:
        buckets.setdefault(group_title(key), []).append(key)

    groups: list[KeyGroup] = []
    for source, titles in _GROUP_ORDER.items():
        for title in titles:
            bucket = buckets.pop((source, title), None)
            if bucket:
                groups.append(
                    KeyGroup(source=source, title=title, keys=tuple(sorted(bucket)))
                )
    for (source, title), bucket in sorted(buckets.items()):
        groups.append(KeyGroup(source=source, title=title, keys=tuple(sorted(bucket))))
    return groups


def format_value(value: Any) -> str:
    """Render one flattened value for the comparison table."""

    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
