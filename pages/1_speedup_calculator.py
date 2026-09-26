"""Page 1: Speedup Calculator — interactive speculative decoding speedup predictor."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Streamlit executes pages as scripts, without a package context.
if not __package__:
    _app_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_app_root.parent))
    __package__ = f"{_app_root.name}.pages"

from ..core.speedup import (
    compute_avg_accept_length,
    compute_incremental_table,
    compute_speedup,
)
# ─── Page Config ────────────────────────────────────────────────────────────

st.header("Speedup Calculator")
st.caption("投机解码加速比交互计算器 — 调整参数实时预测理论加速比")

# ─── Input Parameters (side by side) ────────────────────────────────────────

num_heads = st.select_slider(
    "Num Draft Tokens (MTP heads)",
    options=list(range(1, 7)),
    value=3,
    help="投机 token 数量，对应 MTP head 数 / ngram 候选步数",
)

col_params, col_rates = st.columns([3, 2])

with col_params:
    st.subheader("Timing Parameters")

    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        t_target_ms = st.number_input(
            "T_target (ms)",
            min_value=1.0, max_value=500.0, value=76.0, step=1.0,
            help="Baseline 模型 decode 单步耗时",
        )

    with pcol2:
        verify_overhead = st.number_input(
            "Verify Overhead (x)",
            min_value=1.0, max_value=3.0, value=1.2, step=0.01,
            help="验证步相对于 baseline 的耗时膨胀倍率。"
                 "投机解码时 target model 需要一次验证多个候选 token，"
                 "计算量增大导致 T_verify > T_target。"
                 "典型值 1.1~1.5。",
        )

    with pcol3:
        t_draft_per_head = st.number_input(
            "T_draft per token (ms)",
            min_value=0.1, max_value=50.0, value=3.0, step=0.1,
            help="每个 draft token 的推理耗时",
        )

    t_target_verify_ms = t_target_ms * verify_overhead
    t_draft_ms = t_draft_per_head * num_heads

    # Derived metrics
    dcol1, dcol2, dcol3 = st.columns(3)
    dcol1.metric("T_target_verify", f"{t_target_verify_ms:.1f} ms",
                 delta=f"+{t_target_verify_ms - t_target_ms:.1f}ms", delta_color="inverse")
    dcol2.metric("T_draft total", f"{t_draft_ms:.1f} ms")
    dcol3.metric("T_total (verify+draft)", f"{t_target_verify_ms + t_draft_ms:.1f} ms")

with col_rates:
    st.subheader("Accept Rates")
    accept_rates = []
    for i in range(num_heads):
        default_rate = max(0.5, 0.85 - i * 0.06)
        rate = st.slider(
            f"Token {i+1} 接受率",
            0.0, 1.0, default_rate, 0.01,
            key=f"rate_{i}",
            help=f"第 {i+1} 个 draft token 被接受的概率",
        )
        accept_rates.append(rate)

# ─── Results (colored cards) ────────────────────────────────────────────────

st.divider()

avg_accept_len = compute_avg_accept_length(accept_rates)
speedup = compute_speedup(t_target_ms, t_target_verify_ms, t_draft_ms, accept_rates)

# Colored result cards using st.columns + custom markdown
def _speedup_color(val: float) -> str:
    if val >= 1.5:
        return "#0d9e3f"  # strong green
    elif val >= 1.2:
        return "#4caf50"  # green
    elif val >= 1.0:
        return "#ff9800"  # orange
    else:
        return "#f44336"  # red

speedup_color = _speedup_color(speedup)

rcol1, rcol2, rcol3 = st.columns(3)

with rcol1:
    st.markdown(
        f"""<div style="background:#1e1e2e; border-radius:12px; padding:20px; text-align:center; border: 1px solid #333;">
        <div style="color:#aaa; font-size:13px;">Avg Accept Length</div>
        <div style="color:#fff; font-size:32px; font-weight:700;">{avg_accept_len:.3f}</div>
        <div style="color:#888; font-size:12px;">tokens / step</div>
        </div>""",
        unsafe_allow_html=True,
    )

with rcol2:
    st.markdown(
        f"""<div style="background:#1e1e2e; border-radius:12px; padding:20px; text-align:center; border: 2px solid {speedup_color};">
        <div style="color:#aaa; font-size:13px;">Predicted Speedup</div>
        <div style="color:{speedup_color}; font-size:36px; font-weight:700;">{speedup:.3f}x</div>
        <div style="color:#888; font-size:12px;">vs baseline (no speculation)</div>
        </div>""",
        unsafe_allow_html=True,
    )

with rcol3:
    efficiency = (avg_accept_len / num_heads) * 100 if num_heads > 0 else 0
    eff_color = "#4caf50" if efficiency > 60 else "#ff9800" if efficiency > 40 else "#f44336"
    st.markdown(
        f"""<div style="background:#1e1e2e; border-radius:12px; padding:20px; text-align:center; border: 1px solid #333;">
        <div style="color:#aaa; font-size:13px;">Draft Efficiency</div>
        <div style="color:{eff_color}; font-size:32px; font-weight:700;">{efficiency:.0f}%</div>
        <div style="color:#888; font-size:12px;">accept_len / num_draft_tokens</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Incremental Table ──────────────────────────────────────────────────────

with st.expander("Incremental Speedup Table", expanded=True):
    # Build per-step timing (linear scaling assumption for verify/draft)
    t_verify_per_step = [t_target_ms + (t_target_verify_ms - t_target_ms) * (i + 1) / num_heads for i in range(num_heads)]
    t_draft_per_step = [t_draft_per_head * (i + 1) for i in range(num_heads)]

    table_data = compute_incremental_table(t_target_ms, t_verify_per_step, t_draft_per_step, accept_rates)

    if table_data:
        df_table = pd.DataFrame(table_data)
        df_table.columns = ["Num Draft Tokens", "T_verify (ms)", "T_draft (ms)", "T_total (ms)", "Accept Length", "Speedup vs Baseline", "Speedup vs Previous"]

        def _highlight_speedup(val):
            if isinstance(val, (int, float)):
                if val >= 1.5:
                    return "color: #0d9e3f; font-weight: bold"
                elif val >= 1.0:
                    return "color: #4caf50"
                else:
                    return "color: #f44336"
            return ""

        st.dataframe(
            df_table.style
            .format({
                "T_verify (ms)": "{:.1f}",
                "T_draft (ms)": "{:.1f}",
                "T_total (ms)": "{:.1f}",
                "Accept Length": "{:.3f}",
                "Speedup vs Baseline": "{:.3f}",
                "Speedup vs Previous": "{:.3f}",
            })
            .map(_highlight_speedup, subset=["Speedup vs Baseline", "Speedup vs Previous"]),
            use_container_width=True,
        )

# ─── Bar Chart: Current Config ───────────────────────────────────────────────

st.subheader("Current Config: Speedup by Draft Tokens")

if table_data:
    # Bar chart with color gradient + value annotations
    colors = [_speedup_color(row["speedup_vs_baseline"]) for row in table_data]
    fig_bar = go.Figure(data=go.Bar(
        x=[str(row["num_draft_tokens"]) for row in table_data],
        y=[row["speedup_vs_baseline"] for row in table_data],
        marker_color=colors,
        text=[f"{row['speedup_vs_baseline']:.3f}x" for row in table_data],
        textposition="outside",
        textfont=dict(size=14, color="white"),
    ))
    fig_bar.add_hline(y=1.0, line_dash="dash", line_color="gray",
                      annotation_text="baseline", annotation_font_color="gray")

    # Add accept length annotation on each bar
    for row in table_data:
        fig_bar.add_annotation(
            x=str(row["num_draft_tokens"]),
            y=row["speedup_vs_baseline"] * 0.5,
            text=f"len={row['cumulative_accept_len']:.2f}<br>T={row['t_total_ms']:.0f}ms",
            showarrow=False,
            font=dict(size=10, color="white"),
        )

    fig_bar.update_layout(
        title=dict(text="Speedup by Num Draft Tokens (current params)", font=dict(size=16)),
        xaxis_title="Num Draft Tokens",
        yaxis_title="Speedup (x)",
        yaxis=dict(range=[0, max(row["speedup_vs_baseline"] for row in table_data) * 1.3]),
        template="plotly_dark",
        height=420,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── Sensitivity Heatmap: Accept Rate x Verify Overhead ─────────────────────

st.subheader("Sensitivity: Accept Rate x Verify Overhead")
st.caption("固定 draft token 数，观察接受率和验证开销对加速比的影响")

heatmap_heads = st.select_slider(
    "Num Draft Tokens (for this heatmap)",
    options=list(range(1, 7)),
    value=num_heads,
    key="heatmap_heads",
)

rates_range = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
overhead_range = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]

heatmap_data = []
for overhead in overhead_range:
    row = []
    for rate in rates_range:
        uniform_rates = [rate] * heatmap_heads
        t_draft_heatmap = t_draft_per_head * heatmap_heads
        s = compute_speedup(t_target_ms, t_target_ms * overhead, t_draft_heatmap, uniform_rates)
        row.append(s)
    heatmap_data.append(row)

fig_heat = go.Figure(data=go.Heatmap(
    z=heatmap_data,
    x=[f"{r:.0%}" for r in rates_range],
    y=[f"{o:.2f}x" for o in overhead_range],
    colorscale="RdYlGn",
    zmin=0.7,
    zmax=max(2.5, max(max(row) for row in heatmap_data)),
    text=[[f"{v:.2f}" for v in row] for row in heatmap_data],
    texttemplate="%{text}",
    textfont=dict(size=9),
    colorbar=dict(title=dict(text="Speedup", font=dict(size=12))),
    hovertemplate="Accept Rate: %{x}<br>Overhead: %{y}<br>Speedup: %{z:.3f}x<extra></extra>",
))
fig_heat.update_layout(
    title=dict(
        text=f"Accept Rate x Verify Overhead (tokens={heatmap_heads}, T_draft/tok={t_draft_per_head:.1f}ms)",
        font=dict(size=14),
    ),
    xaxis_title="Uniform Accept Rate",
    yaxis_title="Verify Overhead",
    template="plotly_dark",
    height=480,
    margin=dict(t=50, b=40),
)
st.plotly_chart(fig_heat, use_container_width=True)
