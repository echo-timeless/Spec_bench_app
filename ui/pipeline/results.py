"""Saved-result lookup, history, curves, and comparison page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ...core.pipeline import (
    PipelineError,
    PipelineSettings,
    ResultRecordConflictError,
    build_job_metadata,
    job_metadata_from_saved_result,
    load_or_create_result_view,
    read_log_tail,
    replace_saved_benchmark_result,
    revise_saved_benchmark_result,
    saved_run_artifacts,
)
from ...core.result_keys import format_value
from .components import (
    _column_label,
    _comparison_rows,
    _confirm_delete_saved_result,
    _differing_keys,
    _load_saved_results_cached,
    _ordered_keys,
    _render_accept_curves,
    _saved_record_label,
)
from .state import (
    BASELINE_STATE_KEY,
    COLUMN_STATE_KEY,
    DIFF_ONLY_STATE_KEY,
    HISTORY_ERROR_STATE_KEY,
    HISTORY_NOTICE_STATE_KEY,
    HISTORY_PREVIEW_STATE_KEY,
    HISTORY_SELECTED_STATE_KEY,
    LOG_TAIL_BYTES,
    _load_page_settings,
)


def _render_saved_history_manager(
    saved_records: list[dict], settings: PipelineSettings
) -> None:
    """Render persisted results independently of the in-memory Runner."""

    st.divider()
    st.markdown("#### 已保存的历史结果")
    st.caption(
        "历史结果没有可管理的 PID；这里从 saved_results 恢复记录，并按需读取受管 Run 目录中的"
        "日志与轻量结果视图。缓存缺失或过期时才重新解析原始结果。"
    )
    if not saved_records:
        st.info("当前保存目录中没有可恢复的历史结果。")
        return

    record_by_id = {record["run_id"]: record for record in saved_records}
    if st.session_state.get(HISTORY_SELECTED_STATE_KEY) not in record_by_id:
        st.session_state[HISTORY_SELECTED_STATE_KEY] = saved_records[-1]["run_id"]

    selected_run_id = st.selectbox(
        "选择历史结果",
        options=[record["run_id"] for record in reversed(saved_records)],
        key=HISTORY_SELECTED_STATE_KEY,
        format_func=lambda run_id: _saved_record_label(record_by_id[run_id]),
    )
    record = record_by_id[selected_run_id]
    metadata = job_metadata_from_saved_result(record)
    artifacts = saved_run_artifacts(record, settings.workspace_root)

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric(
        "接受长度",
        format_value(record["metrics"].get("accept_length")),
    )
    summary_col2.metric(
        "解码速度 (tok/s)",
        format_value(record["metrics"].get("decode_speed_toks")),
    )
    summary_col3.metric(
        "输出吞吐 (tok/s)",
        format_value(record["metrics"].get("output_throughput")),
    )
    summary_col4.metric(
        "Mean TTFT (ms)",
        format_value(record["metrics"].get("mean_ttft_ms")),
    )
    st.caption(
        f"结果名称：`{record['result_name']}`　Run ID：`{selected_run_id}`　"
        f"Revision：`{record['revision']}`"
    )
    st.caption(f"首次保存：`{record['saved_at']}`　最近更新：`{record['updated_at']}`")
    st.caption(f"原始结果：`{artifacts.result_file}`")
    st.caption(f"原始日志：`{artifacts.log_file}`")

    preview = st.session_state.get(HISTORY_PREVIEW_STATE_KEY)
    preview_matches = bool(
        isinstance(preview, dict) and preview.get("run_id") == selected_run_id
    )
    load_col, clear_col = st.columns(2)
    load_clicked = load_col.button(
        "加载 / 刷新历史详情",
        key=f"pipeline_load_history_{selected_run_id}",
        disabled=not artifacts.result_file.is_file(),
        use_container_width=True,
        help="优先读取轻量结果视图；缺失或过期时才解析原始 result.jsonl。",
    )
    clear_clicked = clear_col.button(
        "释放已加载详情",
        key=f"pipeline_clear_history_{selected_run_id}",
        disabled=not isinstance(preview, dict),
        use_container_width=True,
        help="从当前浏览器 Session State 释放轻量结果视图。",
    )
    if clear_clicked:
        st.session_state[HISTORY_PREVIEW_STATE_KEY] = None
        st.session_state[HISTORY_ERROR_STATE_KEY] = None
        st.rerun()
    if load_clicked:
        try:
            with st.spinner("正在加载历史结果视图..."):
                result_view = load_or_create_result_view(artifacts.result_file)
            if result_view is None:
                raise PipelineError("原始结果文件为空。")
            preview = {"run_id": selected_run_id, "result_view": result_view}
            st.session_state[HISTORY_PREVIEW_STATE_KEY] = preview
            st.session_state[HISTORY_ERROR_STATE_KEY] = None
            preview_matches = True
        except (OSError, ValueError, PipelineError) as exc:
            st.session_state[HISTORY_PREVIEW_STATE_KEY] = None
            st.session_state[HISTORY_ERROR_STATE_KEY] = {
                "run_id": selected_run_id,
                "message": str(exc),
            }
            preview = None
            preview_matches = False

    if not artifacts.result_file.is_file():
        st.warning(
            "本机受管 workspace 中找不到该历史记录的原始 result.jsonl。"
            "对比行仍可查看和修改标签，但无法恢复曲线或重新解析指标。"
        )
    history_error = st.session_state.get(HISTORY_ERROR_STATE_KEY)
    if (
        isinstance(history_error, dict)
        and history_error.get("run_id") == selected_run_id
    ):
        st.error(f"历史结果加载失败：{history_error.get('message', '')}")

    with st.expander("历史逐 step 接受长度曲线", expanded=preview_matches):
        if not preview_matches or preview is None:
            st.info("点击“加载 / 刷新历史详情”后恢复该 Run 的精确逐 step 曲线。")
        else:
            curves = preview["result_view"].curves
            if curves is None:
                st.warning(
                    "原始结果中没有 detailed 请求统计，无法恢复逐 step 曲线。"
                )
            else:
                _render_accept_curves(curves)

    metadata_prefix = f"pipeline_history_metadata_{selected_run_id}"
    metadata_revision_key = f"{metadata_prefix}_revision"
    if st.session_state.get(metadata_revision_key) != record["revision"]:
        st.session_state[f"{metadata_prefix}_model"] = metadata.model_name
        st.session_state[f"{metadata_prefix}_tp"] = metadata.tp_size
        st.session_state[f"{metadata_prefix}_dp"] = metadata.dp_size
        st.session_state[f"{metadata_prefix}_ep"] = metadata.ep_size
        st.session_state[f"{metadata_prefix}_additional"] = metadata.additional
        st.session_state[metadata_revision_key] = record["revision"]

    st.markdown("##### 历史结果标签")
    st.caption("这里只修改结果名称相关信息；表格展示字段统一由上方“对比列”控制。")
    with st.form(f"pipeline_history_metadata_form_{selected_run_id}"):
        st.text_input("模型", key=f"{metadata_prefix}_model")
        tp_col, dp_col, ep_col = st.columns(3)
        with tp_col:
            st.number_input(
                "TP Size（可选）",
                min_value=1,
                step=1,
                value=None,
                key=f"{metadata_prefix}_tp",
            )
        with dp_col:
            st.number_input(
                "DP Size（可选）",
                min_value=1,
                step=1,
                value=None,
                key=f"{metadata_prefix}_dp",
            )
        with ep_col:
            st.number_input(
                "EP Size（可选）",
                min_value=1,
                step=1,
                value=None,
                key=f"{metadata_prefix}_ep",
            )
        st.text_input("补充说明（可选）", key=f"{metadata_prefix}_additional")
        update_col, reparse_col = st.columns(2)
        update_metadata_clicked = update_col.form_submit_button(
            "更新结果标签",
            use_container_width=True,
        )
        reparse_result_clicked = reparse_col.form_submit_button(
            "重新解析原始结果并更新",
            use_container_width=True,
            disabled=not artifacts.result_file.is_file(),
        )

    if update_metadata_clicked or reparse_result_clicked:
        try:
            updated_metadata = build_job_metadata(
                model_name=st.session_state[f"{metadata_prefix}_model"],
                tp_size=st.session_state[f"{metadata_prefix}_tp"],
                dp_size=st.session_state[f"{metadata_prefix}_dp"],
                ep_size=st.session_state[f"{metadata_prefix}_ep"],
                additional=st.session_state[f"{metadata_prefix}_additional"],
            )
            refreshed_result = None
            if reparse_result_clicked:
                with st.spinner("正在重新解析原始结果并刷新轻量视图..."):
                    refreshed_result = load_or_create_result_view(
                        artifacts.result_file,
                        force=True,
                    )
                if refreshed_result is None:
                    raise PipelineError("原始结果文件为空。")
            revised = revise_saved_benchmark_result(
                record,
                metadata=updated_metadata,
                result=refreshed_result,
            )
            replace_saved_benchmark_result(
                revised,
                settings.saved_results_root,
                expected_revision=record["revision"],
            )
            if refreshed_result is not None:
                st.session_state[HISTORY_PREVIEW_STATE_KEY] = {
                    "run_id": selected_run_id,
                    "result_view": refreshed_result,
                }
            _load_saved_results_cached.clear()
            st.session_state[HISTORY_NOTICE_STATE_KEY] = (
                f"已更新历史结果 {selected_run_id}，表格行将在本次刷新后使用 Revision "
                f"{revised['revision']}。"
            )
            st.rerun()
        except ResultRecordConflictError as exc:
            _load_saved_results_cached.clear()
            st.error(f"更新冲突：{exc}")
        except (OSError, ValueError, PipelineError) as exc:
            st.error(f"历史结果更新失败：{exc}")

    st.markdown("##### 历史 SGLang Benchmark Log")
    historical_log = read_log_tail(artifacts.log_file, max_bytes=LOG_TAIL_BYTES)
    if historical_log:
        st.code(historical_log, language="text")
    else:
        st.info("本机受管 workspace 中没有该历史结果的日志内容。")


def render_results_page() -> None:
    """Render saved-result lookup, history, and comparison."""

    settings = _load_page_settings()
    st.header("保存结果查询与多项对比")
    st.caption(
        "从 saved_results 恢复历史记录、曲线和原始日志，并对多个已保存结果进行统一比较。"
    )
    st.divider()
    history_title_col, history_refresh_col = st.columns([4, 1])
    history_title_col.markdown("### 结果表格")
    refresh_history_clicked = history_refresh_col.button(
        "刷新保存记录",
        use_container_width=True,
        help="清除本机缓存并重新扫描共享保存目录。",
    )
    if refresh_history_clicked:
        _load_saved_results_cached.clear()

    saved_records, saved_record_warnings = _load_saved_results_cached(
        str(settings.saved_results_root)
    )
    history_notice = st.session_state.pop(HISTORY_NOTICE_STATE_KEY, "")
    if history_notice:
        st.success(history_notice)
    st.caption(f"保存目录：`{settings.saved_results_root}`")
    if saved_record_warnings:
        with st.expander(f"有 {len(saved_record_warnings)} 个记录文件无法读取"):
            for warning in saved_record_warnings:
                st.warning(warning)

    if not saved_records:
        st.info("还没有保存记录。保存所选结果后会在这里形成对比表。")
    else:
        record_by_id = {record["run_id"]: record for record in saved_records}
        if BASELINE_STATE_KEY not in st.session_state:
            st.session_state[BASELINE_STATE_KEY] = saved_records[0]["run_id"]
        baseline_run_id = st.session_state[BASELINE_STATE_KEY]
        if baseline_run_id and baseline_run_id not in record_by_id:
            baseline_run_id = ""
            st.session_state[BASELINE_STATE_KEY] = ""

        selected_record = record_by_id.get(baseline_run_id)
        current_baseline_label = (
            _saved_record_label(selected_record)
            if selected_record is not None
            else "不显示百分比"
        )
        with st.popover(
            "对比基准",
            icon="↕️",
            width="stretch",
            help="展开后选择基准，或删除不再需要的对比记录。",
        ):
            st.caption(f"当前：{current_baseline_label}")
            no_compare_col, no_compare_action_col = st.columns([8, 1])
            if no_compare_col.button(
                f"{'✓ ' if not baseline_run_id else ''}不显示百分比",
                key="pipeline_baseline_none",
                type="primary" if not baseline_run_id else "secondary",
                width="stretch",
            ):
                st.session_state[BASELINE_STATE_KEY] = ""
                st.rerun()
            no_compare_action_col.write("")

            for record in saved_records:
                run_id = record["run_id"]
                is_selected = run_id == baseline_run_id
                record_col, delete_col = st.columns([8, 1])
                if record_col.button(
                    f"{'✓ ' if is_selected else ''}{_saved_record_label(record)}",
                    key=f"pipeline_baseline_{run_id}",
                    type="primary" if is_selected else "secondary",
                    width="stretch",
                ):
                    st.session_state[BASELINE_STATE_KEY] = run_id
                    st.rerun()
                if delete_col.button(
                    "🗑️",
                    key=f"pipeline_delete_{run_id}",
                    help=f"删除 {run_id} 的对比 JSON",
                    width="stretch",
                ):
                    _confirm_delete_saved_result(
                        record,
                        str(settings.saved_results_root),
                    )

        available_keys = _ordered_keys(
            {key for record in saved_records for key in record["values"]},
            settings.default_comparison_keys,
        )
        default_columns = _ordered_keys(
            {
                key
                for record in saved_records
                for key in record["selected_keys"]
                if key in record["values"]
            }
            or {
                key
                for key in settings.default_comparison_keys
                if any(key in record["values"] for record in saved_records)
            },
            settings.default_comparison_keys,
        )
        if COLUMN_STATE_KEY not in st.session_state:
            st.session_state[COLUMN_STATE_KEY] = default_columns
        else:
            st.session_state[COLUMN_STATE_KEY] = [
                key
                for key in st.session_state[COLUMN_STATE_KEY]
                if key in available_keys
            ]
        with st.form("pipeline_column_picker_form", border=False):
            st.multiselect(
                f"对比列（可选 {len(available_keys)} 项）",
                options=available_keys,
                key=COLUMN_STATE_KEY,
                format_func=_column_label,
                help=(
                    "这里统一控制结果表格展示的列，可随时增减；记录本身保存了全部键。"
                    "一次可以改多项，改完点“应用对比列”表格才刷新。"
                ),
            )
            st.form_submit_button("应用对比列", type="primary")
        st.caption("列选择只影响当前表格展示；保存记录中的全部可比较字段不会被删除。")
        st.checkbox(
            "只显示与基准不同的列",
            key=DIFF_ONLY_STATE_KEY,
            help="隐藏所有记录取值都与基准一致的列，用于快速定位两轮之间改了什么。",
        )

        columns = list(st.session_state[COLUMN_STATE_KEY])
        hidden_column_count = 0
        if st.session_state[DIFF_ONLY_STATE_KEY] and baseline_run_id:
            differing = _differing_keys(saved_records, baseline_run_id, columns)
            hidden_column_count = len(columns) - len(differing)
            columns = differing

        if not columns:
            st.info("当前没有可显示的对比列，请在上方选择需要对比的键。")
        else:
            st.dataframe(
                pd.DataFrame(
                    _comparison_rows(
                        saved_records,
                        baseline_run_id or None,
                        columns,
                    )
                ),
                use_container_width=True,
                hide_index=True,
            )
        caption = "数值列括号内为 (new - baseline) / baseline * 100%；非数值列与基准不同时标 ⚠"
        if hidden_column_count:
            caption = f"{caption}；已隐藏 {hidden_column_count} 个与基准一致的列"
        st.caption(caption)

    _render_saved_history_manager(saved_records, settings)
