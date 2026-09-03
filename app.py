"""Bench App — speculative decoding calculator and SPEED-Bench workflow.

Entry point for the Streamlit multipage app.
Run with: streamlit run app.py --server.port 8501
"""

import streamlit as st

st.set_page_config(
    page_title="FastDeploy Bench App",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Explicit navigation with custom labels ─────────────────────────────────

pg = st.navigation([
    st.Page("pages/0_usage_guide.py", title="使用说明", icon="📖", default=True),
    st.Page("pages/1_speedup_calculator.py", title="Speedup Calculator", icon="⚡"),
    st.Page("pages/4_benchmark_pipeline.py", title="新建发压", icon="🚀"),
    st.Page("pages/5_benchmark_jobs.py", title="发压任务与日志", icon="🧭"),
    st.Page("pages/6_benchmark_results.py", title="结果查询与对比", icon="📈"),
])

pg.run()
