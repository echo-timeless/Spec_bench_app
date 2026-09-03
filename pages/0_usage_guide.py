"""使用说明 — 公式说明和使用指南。"""

import streamlit as st

st.title("使用说明")
st.markdown(
    r"""
## 投机解码加速比计算工具

本工具用于计算和预测投机解码（Speculative Decoding）在不同方法和参数下的理论加速比。

---

### 核心计算公式

#### 1. 平均接受长度

$$
\text{avg\_accept\_len} = 1 + r_1 + r_1 \cdot r_2 + r_1 \cdot r_2 \cdot r_3 + \cdots
$$

- 每个 $r_i$ 为第 $i$ 个 draft token 的接受率
- Baseline（无投机）时 avg_accept_len = 1
- 直觉理解：每一步 target model 保底产出 1 个 token，额外被接受的 draft token 累加上去

#### 2. 加速比

$$
\text{speedup} = \frac{T_{\text{target}} \times \text{avg\_accept\_len}}{T_{\text{target\_verify}} + T_{\text{draft}}}
$$

| 符号 | 含义 |
|------|------|
| $T_{\text{target}}$ | Baseline 模型单步推理耗时（无投机时的 decode 单步） |
| $T_{\text{target\_verify}}$ | Target 模型验证步耗时，一般为 $T_{\text{target}}$ 的 1.0~1.5 倍 |
| $T_{\text{draft}}$ | Draft Model（Proposer）推理总耗时 |

**直觉**：分子是"等效产出的 token 数 × baseline 单步时间"（即不用投机需要多久），分母是"实际花费的时间"。

#### 3. 关键参数影响

| 参数 | 增大时对加速比的影响 |
|------|---------------------|
| Accept Rate ↑ | 加速比 ↑ （接受更多 token） |
| Verify Overhead ↑ | 加速比 ↓ （验证更贵） |
| T_draft ↑ | 加速比 ↓ （Draft 开销更大） |
| Num Draft Tokens ↑ | 初期 ↑ 后期 ↓ （边际收益递减） |

---

### 页面说明

| 页面 | 功能 |
|------|------|
| **Speedup Calculator** | 交互式计算加速比，调参后实时反馈 |
| **新建发压** | 配置 workload、预设结果名称并启动 SGLang 发压 |
| **发压任务与日志** | 管理当前 Job、PID/PGID、最终命令和实时日志 |
| **结果查询与对比** | 恢复已保存结果、曲线与日志，修改说明并进行多项对比 |

---

### 快速开始

1. 左侧导航栏选择 **Speedup Calculator**
2. 顶部选择 MTP heads 数量
3. 调整 T_target、Verify Overhead、T_draft per head、各 head 接受率
4. 实时查看加速比结果和热力图

### SPEED-Bench 结果保存与对比

1. 确认服务端以 `--speculative-decoding-stats=detailed` 启动
2. 在“新建发压”的 **结果名称与配置** 区域填写模型以及可选 TP/DP/EP 和补充说明
3. 生成并校验最终命令；每次“开始发压”都会创建一个携带上述命名快照的 Job
4. 到“发压任务与日志”选择 Job，查看进程状态、PID/PGID、最终命令和实时日志
5. 发压成功后点击日志旁的 **保存所选结果**
6. 到“结果查询与对比”选择保存记录；加载轻量结果视图后查看逐 step 曲线
7. 打开“对比基准”弹出面板选择历史记录，其他行会显示相对变化百分比
8. BenchAPP 重启后，仍可在“结果查询与对比”恢复日志和曲线、更新结果标签，
   或重新解析原始结果刷新表格行
9. 点击记录右侧 🗑️ 并二次确认，可只删除对应的对比 JSON

一个 Job 对应一个发压进程和一个服务端。不同 `Host:Port` 可并行；同一端点只能有一个
Running Job，避免两个 workload 互相影响吞吐和延迟指标。逐 step 数据本身按请求隔离。

Pipeline 结果以每个 Run 一个 JSON 文件的方式保存，默认目录为
`benchmark_workspace/saved_results`；多机使用时可在
`benchmark_pipeline.toml` 中将 `saved_results_root` 指向共享存储。
删除对比记录不会删除原始 `result.jsonl` 和 `benchmark.log`。
"""
)
