"""Session State keys and initialization for Pipeline pages."""

from __future__ import annotations

import streamlit as st

from ...core.pipeline import (
    PipelineError,
    PipelineSettings,
    build_job_metadata,
    load_settings,
)

COMMAND_WIDGET_KEY = "_pipeline_command_widget"
BASELINE_STATE_KEY = "pipeline_comparison_baseline"
HISTORY_NOTICE_STATE_KEY = "pipeline_history_notice"
COLUMN_STATE_KEY = "pipeline_comparison_columns"
DIFF_ONLY_STATE_KEY = "pipeline_diff_only"
HISTORY_SELECTED_STATE_KEY = "pipeline_selected_saved_result"
HISTORY_PREVIEW_STATE_KEY = "pipeline_historical_result_preview"
HISTORY_ERROR_STATE_KEY = "pipeline_historical_result_error"
LOG_TAIL_BYTES = 16 * 1024
NEW_JOB_METADATA_PREFIX = "pipeline_new_job_metadata"


def _parameter_widget_key(name: str) -> str:
    return f"_pipeline_parameter_{name}"


def _save_parameter(name: str) -> None:
    saved = dict(st.session_state["pipeline_saved_parameters"])
    saved[name] = st.session_state[_parameter_widget_key(name)]
    st.session_state["pipeline_saved_parameters"] = saved


def _save_command_draft() -> None:
    st.session_state["pipeline_command_draft"] = st.session_state.get(
        COMMAND_WIDGET_KEY, ""
    )


def _restore_generated_command() -> None:
    generated = st.session_state.get("pipeline_generated_command", "")
    st.session_state["pipeline_command_draft"] = generated
    st.session_state[COMMAND_WIDGET_KEY] = generated


def _new_job_metadata():
    """Build the result label configured before a benchmark starts."""

    return build_job_metadata(
        model_name=st.session_state[f"{NEW_JOB_METADATA_PREFIX}_model"],
        tp_size=st.session_state[f"{NEW_JOB_METADATA_PREFIX}_tp"],
        dp_size=st.session_state[f"{NEW_JOB_METADATA_PREFIX}_dp"],
        ep_size=st.session_state[f"{NEW_JOB_METADATA_PREFIX}_ep"],
        additional=st.session_state[f"{NEW_JOB_METADATA_PREFIX}_additional"],
    )


def _load_page_settings() -> PipelineSettings:
    """Load Pipeline configuration with a page-friendly error message."""

    try:
        settings = load_settings()
    except (OSError, PipelineError) as exc:
        st.error(f"无法加载 Pipeline 配置：{exc}")
        st.stop()

    return settings


def _initialize_launch_state(settings: PipelineSettings) -> None:
    """Initialize state owned by the benchmark creation page."""

    parameter_defaults = {
        "host": settings.host,
        "port": settings.port,
        "benchmark_family": "performance",
        "dataset_id": "speed-bench",
        "dataset_variant": settings.dataset_size
        if settings.dataset_size in {"1k", "2k", "8k", "16k", "32k"}
        else "32k",
        "dataset_size": settings.dataset_size,
        "task_group": "综合",
        "category": settings.category,
        "num_prompts": settings.num_prompts,
        "output_len": settings.output_len,
        "max_concurrency": settings.max_concurrency,
        "request_rate": settings.request_rate,
        "seed": settings.seed,
        "warmup_requests": settings.warmup_requests,
        "extra_request_body": settings.extra_request_body,
        "output_details": settings.output_details,
    }
    saved_parameters = st.session_state.setdefault(
        "pipeline_saved_parameters", parameter_defaults
    )
    if saved_parameters.get("dataset_variant") == "qualitative":
        saved_parameters["dataset_variant"] = "32k"
    for parameter_name, default_value in parameter_defaults.items():
        saved_parameters.setdefault(parameter_name, default_value)
        widget_key = _parameter_widget_key(parameter_name)
        if widget_key not in st.session_state:
            st.session_state[widget_key] = saved_parameters[parameter_name]

    st.session_state.setdefault("pipeline_command_draft", "")
    st.session_state.setdefault("pipeline_generated_command", "")
    st.session_state.setdefault(f"{NEW_JOB_METADATA_PREFIX}_model", "")
    st.session_state.setdefault(f"{NEW_JOB_METADATA_PREFIX}_tp", None)
    st.session_state.setdefault(f"{NEW_JOB_METADATA_PREFIX}_dp", None)
    st.session_state.setdefault(f"{NEW_JOB_METADATA_PREFIX}_ep", None)
    st.session_state.setdefault(f"{NEW_JOB_METADATA_PREFIX}_additional", "")
    if COMMAND_WIDGET_KEY not in st.session_state:
        st.session_state[COMMAND_WIDGET_KEY] = st.session_state[
            "pipeline_command_draft"
        ]


def _initialize_job_state() -> None:
    """Initialize state shared by launch handoff and the job manager."""

    st.session_state.setdefault("pipeline_result_preview", None)
    st.session_state.setdefault("pipeline_result_error", "")
