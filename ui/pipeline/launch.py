"""Benchmark creation and launch page."""

from __future__ import annotations

import streamlit as st

from bench_app.core.pipeline import (
    PERFORMANCE_FAMILY,
    SPEED_BENCH_SIZES,
    TASK_EVALUATION_FAMILY,
    TASK_GROUPS,
    THROUGHPUT_CATEGORIES,
    BenchmarkParameters,
    CommandValidationError,
    PipelineError,
    generate_command,
    validate_command,
)
from bench_app.core.pipeline.datasets import (
    get_dataset_spec,
    pending_dataset_specs,
    selectable_dataset_specs,
)
from bench_app.core.pipeline_runtime import get_benchmark_runner
from bench_app.ui.pipeline.state import (
    COMMAND_WIDGET_KEY,
    NEW_JOB_METADATA_PREFIX,
    _initialize_job_state,
    _initialize_launch_state,
    _load_page_settings,
    _new_job_metadata,
    _parameter_widget_key,
    _restore_generated_command,
    _save_command_draft,
    _save_parameter,
)


def render_launch_page() -> None:
    """Render the benchmark creation and launch page."""

    settings = _load_page_settings()
    runner = get_benchmark_runner()
    _initialize_launch_state(settings)
    _initialize_job_state()
    st.header("模型发压任务")
    st.caption(
        "每次启动形成一个独立发压任务；不同 Host:Port 的任务可以并行运行。"
        "启动后请到“发压任务与日志”管理进程，到“结果查询与对比”分析保存记录。"
        "支持性能压测和单轮任务数据集发压；"
        "逐 step 数据随请求响应返回，"
        "不会与其他客户端混合；同一目标实例仍限制一个任务，以免并发 workload 污染性能指标。"
    )

    with st.container():
        st.subheader("新建发压任务")
        st.caption(
            "这里是下一项任务的配置模板；已启动任务持有自己的参数快照，不受后续修改影响。"
        )
        with st.container(border=True):
            service_col1, service_col2 = st.columns([2, 1])
            with service_col1:
                host = st.text_input(
                    "Host",
                    key=_parameter_widget_key("host"),
                    on_change=_save_parameter,
                    args=("host",),
                )
            with service_col2:
                port = st.number_input(
                    "Port",
                    min_value=1,
                    max_value=65535,
                    step=1,
                    key=_parameter_widget_key("port"),
                    on_change=_save_parameter,
                    args=("port",),
                )

            family = st.selectbox(
                "测试场景",
                options=(PERFORMANCE_FAMILY, TASK_EVALUATION_FAMILY),
                format_func=lambda value: (
                    "性能压测" if value == PERFORMANCE_FAMILY else "任务评测"
                ),
                key=_parameter_widget_key("benchmark_family"),
                on_change=_save_parameter,
                args=("benchmark_family",),
            )

            data_col1, data_col2 = st.columns(2)
            with data_col1:
                if family == PERFORMANCE_FAMILY:
                    dataset_id = "speed-bench"
                    dataset_variant = st.selectbox(
                        "数据集长度",
                        options=SPEED_BENCH_SIZES,
                        format_func=lambda value: value,
                        key=_parameter_widget_key("dataset_variant"),
                    )
                    dataset_size = dataset_variant
                    task_group = None
                else:
                    task_group = st.selectbox(
                        "任务类型",
                        options=TASK_GROUPS,
                        key=_parameter_widget_key("task_group"),
                    )
                    task_specs = selectable_dataset_specs(task_group=task_group)
                    pending_specs = pending_dataset_specs(task_group=task_group)
                    if pending_specs:
                        st.caption(
                            "暂不支持多轮数据集："
                            + "、".join(spec.label for spec in pending_specs)
                        )
                    dataset_id = st.selectbox(
                        "数据集",
                        options=[spec.dataset_id for spec in task_specs],
                        format_func=lambda value: get_dataset_spec(value).label,
                        key=_parameter_widget_key(f"dataset_id_{task_group}"),
                    )
                    dataset_variant = ""
                    selected_spec = get_dataset_spec(dataset_id)
                    if selected_spec.variants:
                        dataset_variant = st.selectbox(
                            "数据集变体",
                            options=selected_spec.variants,
                            key=_parameter_widget_key(f"dataset_variant_{dataset_id}"),
                        )
                    dataset_size = (
                        "qualitative" if dataset_id == "speed-bench-qualitative" else dataset_id
                    )
            with data_col2:
                category = st.selectbox(
                    "SPEED-Bench 熵分类",
                    options=THROUGHPUT_CATEGORIES,
                    format_func=lambda value: value or "全部",
                    disabled=family != PERFORMANCE_FAMILY,
                    help="仅性能压测的 SPEED-Bench 长度数据支持 low_entropy / mixed / high_entropy。",
                    key=_parameter_widget_key("category"),
                    on_change=_save_parameter,
                    args=("category",),
                )

            request_col1, request_col2, request_col3 = st.columns(3)
            with request_col1:
                num_prompts = st.number_input(
                    "Num Prompts",
                    min_value=1,
                    step=1,
                    key=_parameter_widget_key("num_prompts"),
                    on_change=_save_parameter,
                    args=("num_prompts",),
                )
            with request_col2:
                output_len = st.number_input(
                    "Output Len",
                    min_value=1,
                    step=1,
                    key=_parameter_widget_key("output_len"),
                    on_change=_save_parameter,
                    args=("output_len",),
                )
            with request_col3:
                max_concurrency = st.number_input(
                    "Max Concurrency",
                    min_value=1,
                    step=1,
                    key=_parameter_widget_key("max_concurrency"),
                    on_change=_save_parameter,
                    args=("max_concurrency",),
                )

            with st.expander("高级参数"):
                advanced_col1, advanced_col2, advanced_col3 = st.columns(3)
                with advanced_col1:
                    request_rate = st.text_input(
                        "Request Rate",
                        help="使用 inf 表示尽快发满并发。",
                        key=_parameter_widget_key("request_rate"),
                        on_change=_save_parameter,
                        args=("request_rate",),
                    )
                with advanced_col2:
                    seed = st.number_input(
                        "Seed",
                        step=1,
                        key=_parameter_widget_key("seed"),
                        on_change=_save_parameter,
                        args=("seed",),
                    )
                with advanced_col3:
                    warmup_requests = st.number_input(
                        "Warmup Requests",
                        min_value=0,
                        step=1,
                        key=_parameter_widget_key("warmup_requests"),
                        on_change=_save_parameter,
                        args=("warmup_requests",),
                    )
                extra_request_body = st.text_area(
                    "Extra Request Body (JSON)",
                    height=100,
                    key=_parameter_widget_key("extra_request_body"),
                    on_change=_save_parameter,
                    args=("extra_request_body",),
                )
                output_details = st.checkbox(
                    "输出逐请求明细 (--output-details)",
                    help=(
                        "在 result.jsonl 同一条记录里追加 input_lens / output_lens / "
                        "ttfts / itls / itl_token_lens / generated_texts / errors。"
                        "这些是客户端 SSE/ITL 排查数据，不用于逐 step 接受长度曲线。"
                        "精确曲线由服务端 --speculative-decoding-stats=detailed "
                        "独立返回并自动落盘。"
                    ),
                    key=_parameter_widget_key("output_details"),
                    on_change=_save_parameter,
                    args=("output_details",),
                )

            st.markdown("##### 结果名称与配置")
            metadata_model_col, metadata_additional_col = st.columns(2)
            with metadata_model_col:
                st.text_input(
                    "模型名称",
                    key=f"{NEW_JOB_METADATA_PREFIX}_model",
                    placeholder="例如：DeepSeek-V4-Flash",
                )
            with metadata_additional_col:
                st.text_input(
                    "配置说明（可选）",
                    key=f"{NEW_JOB_METADATA_PREFIX}_additional",
                    placeholder="例如：MTP-2 + CUDA Graph",
                )

            metadata_tp_col, metadata_dp_col, metadata_ep_col = st.columns(3)
            with metadata_tp_col:
                st.number_input(
                    "TP",
                    min_value=1,
                    step=1,
                    value=None,
                    key=f"{NEW_JOB_METADATA_PREFIX}_tp",
                )
            with metadata_dp_col:
                st.number_input(
                    "DP",
                    min_value=1,
                    step=1,
                    value=None,
                    key=f"{NEW_JOB_METADATA_PREFIX}_dp",
                )
            with metadata_ep_col:
                st.number_input(
                    "EP",
                    min_value=1,
                    step=1,
                    value=None,
                    key=f"{NEW_JOB_METADATA_PREFIX}_ep",
                )
            st.caption(
                "最终名称自动组合为“模型名称 + TP/DP/EP + 配置说明”。"
                "保存后仍可在结果查询页修订。"
            )
            metadata = _new_job_metadata()
            if metadata.result_name:
                st.success(f"结果将保存为：{metadata.result_name}")

            generate_clicked = st.button(
                "生成 / 更新发压命令",
                type="primary",
                use_container_width=True,
            )

        if generate_clicked:
            parameters = BenchmarkParameters(
                host=host,
                port=int(port),
                benchmark_family=family,
                dataset_id=dataset_id,
                dataset_variant=dataset_variant,
                dataset_size=dataset_size,
                category=category,
                num_prompts=int(num_prompts),
                output_len=int(output_len),
                max_concurrency=int(max_concurrency),
                request_rate=request_rate,
                seed=int(seed),
                warmup_requests=int(warmup_requests),
                extra_request_body=extra_request_body,
                output_details=bool(output_details),
            )
            st.session_state["pipeline_saved_parameters"] = {
                "host": parameters.host,
                "port": parameters.port,
                "dataset_size": parameters.dataset_size,
                "category": parameters.category,
                "num_prompts": parameters.num_prompts,
                "output_len": parameters.output_len,
                "max_concurrency": parameters.max_concurrency,
                "request_rate": parameters.request_rate,
                "seed": parameters.seed,
                "warmup_requests": parameters.warmup_requests,
                "extra_request_body": parameters.extra_request_body,
                "output_details": parameters.output_details,
            }
            try:
                generated_command, run_id = generate_command(settings, parameters)
                st.session_state["pipeline_generated_command"] = generated_command
                st.session_state["pipeline_command_draft"] = generated_command
                st.session_state[COMMAND_WIDGET_KEY] = generated_command
                if family != PERFORMANCE_FAMILY and category:
                    st.info(
                        "任务评测数据集不使用 SPEED-Bench 熵分类，本次命令已忽略该选项。"
                    )
                st.success(f"已生成命令：{run_id}")
            except (OSError, PipelineError) as exc:
                st.error(f"生成失败：{exc}")

        st.divider()
        st.subheader("最终发压命令")
        command_text = st.text_area(
            "可以继续修改 SGLang 参数；点击发压时执行此处的最终内容。",
            key=COMMAND_WIDGET_KEY,
            on_change=_save_command_draft,
            height=360,
            placeholder="完成上方配置后点击“生成 / 更新发压命令”。",
        )

        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            validate_clicked = st.button(
                "校验命令",
                use_container_width=True,
                disabled=not command_text.strip(),
            )
        with action_col2:
            st.button(
                "恢复生成",
                use_container_width=True,
                disabled=not st.session_state["pipeline_generated_command"],
                on_click=_restore_generated_command,
            )
        with action_col3:
            start_clicked = st.button(
                "开始发压",
                type="primary",
                use_container_width=True,
                disabled=not command_text.strip() or not metadata.model_name,
                help="不同 Host:Port 可同时运行；同一目标实例会被拒绝重复发压。",
            )

        if validate_clicked:
            try:
                validated = validate_command(
                    command_text,
                    settings,
                    require_new_output=True,
                )
                st.success(f"命令校验通过，结果将写入：{validated.output_file}")
            except CommandValidationError as exc:
                for error in exc.errors:
                    st.error(error)

        if start_clicked:
            try:
                started = runner.start(
                    command_text,
                    settings,
                    metadata=metadata,
                )
                st.session_state["pipeline_selected_job"] = started.run_id
                st.session_state["pipeline_result_preview"] = None
                st.session_state["pipeline_result_error"] = ""
                st.success(
                    f"任务 {started.run_id} 已启动：{started.host}:{started.port}，"
                    f"PID={started.pid}"
                )
            except (OSError, PipelineError) as exc:
                st.error(f"启动失败：{exc}")
