"""Reusable status, curve, and comparison components."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import plotly.graph_objects as go
import streamlit as st

from ...core.pipeline import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_SUCCEEDED,
    PipelineError,
    ResultRecordNotFoundError,
    delete_saved_benchmark_result,
    list_saved_benchmark_results,
    numeric_change_percent,
)
from ...core.result_keys import (
    CLIENT_PREFIX,
    format_value,
    split_key,
)
from ...core.step_curve import AcceptCurves, RequestCurve
from .state import (
    BASELINE_STATE_KEY,
    HISTORY_NOTICE_STATE_KEY,
)

KEY_LABELS = {
    "client.accept_length": "接受长度",
    "client.decode_speed_toks": "解码速度 (toks)",
    "client.request_throughput": "请求吞吐 (req/s)",
    "client.input_throughput": "输入吞吐 (tok/s)",
    "client.output_throughput": "输出吞吐 (tok/s)",
    "client.total_throughput": "总吞吐 (tok/s)",
    "client.mean_e2e_latency_ms": "平均 E2E (ms)",
    "client.mean_ttft_ms": "平均 TTFT (ms)",
    "client.mean_tpot_ms": "平均 TPOT (ms)",
    "client.mean_itl_ms": "平均 ITL (ms)",
}


@st.cache_data(show_spinner=False)
def _load_saved_results_cached(
    saved_results_root: str,
) -> tuple[list[dict], list[str]]:
    """Avoid rescanning shared storage on every unrelated widget rerun."""

    return list_saved_benchmark_results(saved_results_root)


def _show_status(status: str) -> None:
    if status == STATUS_RUNNING:
        st.warning("Running")
    elif status == STATUS_SUCCEEDED:
        st.success("Succeeded")
    elif status == STATUS_STOPPED:
        st.info("Stopped")
    elif status == STATUS_FAILED:
        st.error("Failed")
    else:
        st.info(status)


def _format_metric(
    value: Any,
    baseline: Any,
    *,
    include_change: bool,
    as_percentage: bool,
) -> str:
    """Render one comparison cell for values of any scalar type."""

    formatted = format_value(value)
    if value is None or not include_change:
        return formatted
    if as_percentage:
        change = numeric_change_percent(value, baseline)
        if change is not None:
            return f"{formatted} ({change:+.1f}%)"
    # Config keys have no meaningful percentage; flag the ones that moved.
    if baseline is not None and value != baseline:
        return f"{formatted} ⚠"
    return formatted


def _column_label(key: str) -> str:
    """Shorten a flattened key into a table header."""

    source, remainder = split_key(key)
    label = KEY_LABELS.get(key, remainder or key)
    return label if source == CLIENT_PREFIX else f"⚙ {label}"


def _curve_label(curve: RequestCurve, kind: str) -> str:
    return (
        f"{kind}：请求 {curve.request_id}"
        f"（{curve.num_steps} steps，接受长度 {curve.accept_length:.2f}）"
    )


def _render_accept_curves(curves: AcceptCurves) -> None:
    """Plot the best and worst request's accept length over token position."""

    figure = go.Figure()
    for curve, kind, color in (
        (curves.best, "最好", "#2ca02c"),
        (curves.worst, "最差", "#d62728"),
    ):
        figure.add_trace(
            go.Scatter(
                x=[point.token_position for point in curve.points],
                y=[point.accept_length for point in curve.points],
                customdata=[point.verify_length for point in curve.points],
                mode="lines+markers",
                name=_curve_label(curve, kind),
                line={"color": color},
                hovertemplate=(
                    "输出位置=%{x}<br>平均接受长度=%{y:.2f}"
                    "<br>平均 verify tokens=%{customdata:.2f}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        xaxis_title="输出 token 位置",
        yaxis_title=f"逐 step 接受长度（{curves.bin_width} token 一档）",
        yaxis_range=[0, curves.verify_token_cap + 0.5],
        legend={"orientation": "h", "y": -0.25},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        height=320,
    )
    st.plotly_chart(figure)
    caption = (
        f"来自请求响应中 SGLang 统计的 {curves.num_step_records} 条调度 step，覆盖 "
        f"{curves.num_requests} 个请求；按请求全程平均接受长度选择最好和最差两条。"
        "横轴按每步实际提交的 acc_len 推进；悬浮可查看同一分档的平均 verify_len。"
    )
    st.caption(caption)


def _comparison_rows(
    records: list[dict],
    baseline_run_id: str | None,
    columns: Sequence[str],
) -> list[dict[str, str]]:
    baseline_record = next(
        (record for record in records if record["run_id"] == baseline_run_id),
        None,
    )
    rows: list[dict[str, str]] = []
    for record in records:
        row = {"结果名称": record["result_name"]}
        for key in columns:
            baseline_value = (
                baseline_record["values"].get(key)
                if baseline_record is not None
                else None
            )
            row[_column_label(key)] = _format_metric(
                record["values"].get(key),
                baseline_value,
                include_change=(
                    baseline_record is not None and record["run_id"] != baseline_run_id
                ),
                as_percentage=split_key(key)[0] == CLIENT_PREFIX,
            )
        rows.append(row)
    return rows


def _differing_keys(
    records: list[dict],
    baseline_run_id: str | None,
    candidates: Sequence[str],
) -> list[str]:
    """Keep only the keys where at least one record differs from the baseline."""

    baseline_record = next(
        (record for record in records if record["run_id"] == baseline_run_id),
        None,
    )
    if baseline_record is None:
        return list(candidates)
    return [
        key
        for key in candidates
        if any(
            record["values"].get(key) != baseline_record["values"].get(key)
            for record in records
        )
    ]


def _ordered_keys(keys: Iterable[str], priority: Sequence[str]) -> list[str]:
    """Put the configured default keys first, then the rest alphabetically."""

    remaining = set(keys)
    ordered = [key for key in priority if key in remaining]
    return ordered + sorted(remaining.difference(ordered))


def _saved_record_label(record: dict) -> str:
    saved_time = record["saved_at"].replace("T", " ")[:19]
    return f"{record['result_name']} | {saved_time} | {record['run_id']}"


@st.dialog("确认删除记录", icon="🗑️")
def _confirm_delete_saved_result(
    record: dict,
    saved_results_root: str,
) -> None:
    run_id = record["run_id"]
    st.warning("此操作只删除保存的对比 JSON，且无法从页面恢复。")
    st.markdown(f"**结果名称：** {record['result_name']}")
    st.caption(f"Run ID：`{run_id}`")
    st.info("原始 result.jsonl 和 benchmark.log 不会被删除。")

    cancel_col, confirm_col = st.columns(2)
    if cancel_col.button("取消", width="stretch"):
        st.rerun()
    if confirm_col.button(
        "确认删除 JSON",
        type="primary",
        width="stretch",
    ):
        try:
            deleted_path = delete_saved_benchmark_result(
                run_id,
                saved_results_root,
            )
            notice = f"已删除对比记录：{deleted_path.name}"
        except ResultRecordNotFoundError:
            notice = f"记录 {run_id} 已不存在，列表已重新加载。"
        except (OSError, PipelineError) as exc:
            st.error(f"删除失败：{exc}")
            return

        if st.session_state.get(BASELINE_STATE_KEY) == run_id:
            st.session_state[BASELINE_STATE_KEY] = ""
        st.session_state[HISTORY_NOTICE_STATE_KEY] = notice
        _load_saved_results_cached.clear()
        st.rerun()
