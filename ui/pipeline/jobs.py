"""Process-lifetime benchmark jobs and live-log page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bench_app.core.pipeline import (
    STATUS_RUNNING,
    STATUS_SUCCEEDED,
    BenchmarkRunner,
    PipelineError,
    PipelineSettings,
    ProcessStateError,
    ResultRecordExistsError,
    build_saved_benchmark_result,
    save_benchmark_result,
)
from bench_app.core.pipeline_runtime import get_benchmark_runner
from bench_app.ui.pipeline.components import (
    _load_saved_results_cached,
    _show_status,
)
from bench_app.ui.pipeline.state import (
    LOG_TAIL_BYTES,
    _initialize_job_state,
    _load_page_settings,
)


def _render_job_manager(settings: PipelineSettings, runner: BenchmarkRunner) -> None:
    title_col, refresh_col = st.columns([3, 1])
    title_col.subheader("发压任务与实时日志")
    refresh_col.button(
        "刷新任务",
        use_container_width=True,
        help="轮询全部后台进程，并读取当前所选任务的日志和结果。",
    )

    snapshots = runner.snapshots()
    if not snapshots:
        st.info(
            "当前 Streamlit 进程还没有启动发压任务。请先到“新建发压”页面创建任务；"
            "已保存结果请到“结果查询与对比”页面查看。"
        )
        st.caption("SGLang Benchmark Log：等待任务")
    else:
        snapshot_by_id = {item.run_id: item for item in snapshots}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "任务": item.metadata.result_name or "未命名任务",
                        "服务端": f"{item.host}:{item.port}",
                        "状态": item.status,
                        "PID": item.pid,
                        "运行时间": f"{item.elapsed_seconds:.1f}s",
                    }
                    for item in snapshots
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        selected_key = "pipeline_selected_job"
        if st.session_state.get(selected_key) not in snapshot_by_id:
            st.session_state[selected_key] = snapshots[0].run_id

        def _job_option_label(run_id: str) -> str:
            item = snapshot_by_id[run_id]
            name = item.metadata.result_name or "未命名任务"
            return f"{name} | {item.host}:{item.port} | {item.status} | {run_id}"

        selected_run_id = st.selectbox(
            "选择任务",
            options=[item.run_id for item in snapshots],
            key=selected_key,
            format_func=_job_option_label,
        )
        snapshot = runner.snapshot(selected_run_id)
        if snapshot is None:  # pragma: no cover - concurrent defensive path
            st.warning("所选任务已不可用，请刷新页面。")
            st.stop()

        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        with status_col1:
            st.caption("状态")
            _show_status(snapshot.status)
        status_col2.metric("PID / PGID", f"{snapshot.pid} / {snapshot.pgid}")
        status_col3.metric("运行时间", f"{snapshot.elapsed_seconds:.1f}s")
        status_col4.metric(
            "退出码",
            "—" if snapshot.return_code is None else str(snapshot.return_code),
        )
        st.caption(
            f"服务端：`{snapshot.host}:{snapshot.port}`　Run ID：`{snapshot.run_id}`"
        )
        st.caption(f"日志：`{snapshot.log_file}`")
        st.caption(f"结果：`{snapshot.output_file}`")

        stop_col, command_col = st.columns([1, 2])
        if stop_col.button(
            "停止所选任务",
            key=f"pipeline_stop_{selected_run_id}",
            disabled=snapshot.status != STATUS_RUNNING,
            use_container_width=True,
        ):
            try:
                runner.stop(selected_run_id)
                st.rerun()
            except (OSError, ProcessStateError) as exc:
                st.error(f"停止失败：{exc}")
        with command_col.expander("查看该任务执行的最终命令"):
            st.code(snapshot.command, language="bash")

        saved_result_file = settings.saved_results_root / f"{selected_run_id}.json"
        result_already_saved = saved_result_file.is_file()
        st.markdown("#### 预设结果信息")
        if result_already_saved:
            st.success(f"该任务结果已保存：{snapshot.metadata.result_name}")
            st.caption("可到“结果查询与对比”页面恢复详情或更新该表格行。")
        else:
            st.info(f"任务完成后将以“{snapshot.metadata.result_name}”保存。")
            st.caption("名称和配置已在发压前固化；保存后可在结果页修订说明。")

        result_preview = st.session_state["pipeline_result_preview"]
        preview_matches_selected = bool(
            result_preview is not None and result_preview["run_id"] == selected_run_id
        )
        if snapshot.status == STATUS_SUCCEEDED and not preview_matches_selected:
            try:
                result_view = runner.result_view(selected_run_id)
                if result_view is None:
                    raise PipelineError("进程已成功结束，但结果文件为空或不存在。")
                st.session_state["pipeline_result_preview"] = {
                    "run_id": snapshot.run_id,
                    "result_view": result_view,
                }
                st.session_state["pipeline_result_error"] = ""
                result_preview = st.session_state["pipeline_result_preview"]
                preview_matches_selected = True
            except (OSError, ValueError, PipelineError) as exc:
                st.session_state["pipeline_result_preview"] = None
                st.session_state["pipeline_result_error"] = str(exc)
                result_preview = None
                preview_matches_selected = False

        result_is_ready = bool(
            snapshot.status == STATUS_SUCCEEDED and preview_matches_selected
        )
        metadata = snapshot.metadata
        can_save_result = bool(
            result_is_ready and metadata.model_name and not result_already_saved
        )
        log_title_col, save_button_col = st.columns([3, 1])
        log_title_col.markdown("#### SGLang Benchmark Log")
        save_result_clicked = save_button_col.button(
            "保存所选结果",
            key=f"pipeline_save_{selected_run_id}",
            type="primary",
            use_container_width=True,
            disabled=not can_save_result,
            help=(
                "保存所选成功任务的结构化指标。"
                if can_save_result
                else (
                    "该任务已经保存。"
                    if result_already_saved
                    else (
                        "任务缺少发压前预设的模型名称。"
                        if result_is_ready
                        else "等待所选任务成功结束。"
                    )
                )
            ),
        )
        if save_result_clicked and result_preview is not None:
            try:
                record = build_saved_benchmark_result(
                    result_preview["result_view"],
                    run_id=selected_run_id,
                    model_name=metadata.model_name,
                    configuration=metadata.configuration,
                    configuration_options={
                        "tp_size": metadata.tp_size,
                        "dp_size": metadata.dp_size,
                        "ep_size": metadata.ep_size,
                        "additional": metadata.additional,
                    },
                    result_file=snapshot.output_file,
                    log_file=snapshot.log_file,
                    context=snapshot.command_context,
                    selected_keys=[
                        key
                        for key in settings.default_comparison_keys
                        if key in result_preview["result_view"].values
                    ],
                )
                saved_path = save_benchmark_result(record, settings.saved_results_root)
                _load_saved_results_cached.clear()
                st.success(f"已保存：{saved_path}")
            except ResultRecordExistsError:
                st.warning(f"Run {selected_run_id} 已保存，不会重复添加。")
            except (OSError, PipelineError) as exc:
                st.error(f"保存失败：{exc}")

        result_error = st.session_state["pipeline_result_error"]
        if result_error and snapshot.status == STATUS_SUCCEEDED:
            st.error(f"结果解析失败：{result_error}")
        log_text = runner.log_tail(selected_run_id, max_bytes=LOG_TAIL_BYTES)
        if log_text:
            st.code(log_text, language="text")
        elif snapshot.status == STATUS_RUNNING:
            st.info("任务正在运行；点击“刷新任务”读取最新日志。")
        else:
            st.info("该任务当前没有日志内容。")


def render_jobs_page() -> None:
    """Render process-lifetime benchmark jobs and live logs."""

    settings = _load_page_settings()
    runner = get_benchmark_runner()
    _initialize_job_state()
    st.header("发压任务管理")
    st.caption("查看当前 Streamlit 进程启动的任务、运行状态和实时 benchmark 日志。")
    _render_job_manager(settings, runner)
