"""Process-local runtime shared by all SPEED-Bench Streamlit pages."""

from __future__ import annotations

import streamlit as st

from bench_app.core.pipeline.runner import BenchmarkRunner


@st.cache_resource
def get_benchmark_runner() -> BenchmarkRunner:
    """Keep managed benchmark processes alive while users switch pages."""

    return BenchmarkRunner()
