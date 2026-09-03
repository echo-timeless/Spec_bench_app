# Bench App — 投机解码加速比计算与 SPEED-Bench

## 功能概览

| 页面 | 用途 |
|------|------|
| Speedup Calculator | 交互式计算投机解码理论加速比 |
| 新建发压 | 选择性能压测或任务评测 workload、预设结果名称并启动 SGLang |
| 发压任务与日志 | 管理当前进程中的 Job，查看状态、命令和实时日志并保存成功结果 |
| 结果查询与对比 | 恢复已保存结果、原始日志和曲线，修改说明并进行多项对比 |

## SGLang 依赖：PR #34814

BenchAPP 的逐 step 曲线和请求级统计依赖 SGLang PR
[#34814 — feat: expose per-request speculative decoding stats](https://github.com/sgl-project/sglang/pull/34814)
（分支 `feat/per-request-speculative-decoding-stats`，提交 `622d048d`）。该 PR 提供：

- 服务端开关 `--speculative-decoding-stats {none,summary,detailed}`，默认 `none`。
- `summary` 返回请求级精确计数、接受率、平均接受长度和直方图；`detailed` 额外返回每个
  verification step 的 `verify_lengths` / `accept_lengths`（均含 root/bonus token，为
  stop/grammar 裁剪前的 verifier 原始值），并满足不变量
  `sum(verify_len - 1) == num_verified_draft_tokens`、
  `sum(accept_len - 1) == num_accepted_draft_tokens`。
- `sglang.benchmark.serving` 把统计按请求索引落盘到 `result.jsonl` 顶层
  `speculative_decoding_stats`；失败或缺失统计的请求显式保存 `stats: null`。

该 PR 的代码必须同时存在于两处，缺一不可：

1. **被测 SGLang 服务**：否则服务不识别 `--speculative-decoding-stats`，响应中没有统计。
2. **`benchmark_pipeline.toml` 中 `sglang_repo` 指向的仓库**：BenchAPP Runner 把
   `<sglang_repo>/python` 注入发压子进程的 `PYTHONPATH`，统计的提取和落盘由该仓库的
   `sglang.benchmark.serving` 完成。

注意：PR #34814 不包含 `--output-details` 的 `itl_token_lens` 字段；该字段目前只存在于
原始开发工作区 `/root/paddlejob/inference-public/jianghaitao/sglang` 的未提交修改中。
只使用逐 step 曲线不受影响，需要逐 chunk token 数排查时才需要那份工作区代码。

## 新用户部署（从共享目录复制）

目录在共享存储上，各机器都能访问。每个人复制一份 BenchAPP 到自己的路径本地使用，
互不影响；结果和保存记录默认落在各自副本的 `benchmark_workspace/` 中。

### 第 1 步：复制 BenchAPP

```bash
# 排除维护者的历史结果、旧数据库和字节码缓存，只复制代码与配置
rsync -a \
  --exclude benchmark_workspace/ \
  --exclude backups/ \
  --exclude bench_app.db \
  --exclude '__pycache__/' \
  /root/paddlejob/inference-public/jianghaitao/bench_app/ \
  <你的路径>/bench_app/
```

### 第 2 步：把 PR #34814 拉进自己的 sglang 仓库

```bash
cd <你的 sglang 仓库>

# 方式 A：直接从共享目录的维护者仓库取分支（无需外网）
git fetch /root/paddlejob/inference-public/jianghaitao/sglang \
  feat/per-request-speculative-decoding-stats
git checkout -b speculative-decoding-stats FETCH_HEAD

# 方式 B：从 GitHub 拉 PR
git fetch https://github.com/sgl-project/sglang.git \
  pull/34814/head:speculative-decoding-stats
git checkout speculative-decoding-stats
```

如需保留自己分支上的其他修改，可改用
`git cherry-pick 622d048df89eab7891887f97b6b6833227f15b83`；若基点差异较大出现冲突，
优先用上面 checkout 整分支的方式。

### 第 3 步：修改自己副本中的 `benchmark_pipeline.toml`

```toml
[environment]
python_executable = "<你的 sglang 运行环境 python 绝对路径>"
sglang_repo       = "<你的 sglang 仓库绝对路径>"   # 需已包含 PR #34814
dataset_root      = "..."   # 共享数据集路径，通常保持默认即可
# saved_results_root 保持默认则对比记录各自独立；
# 改成一个大家都能访问的共享目录，即可互相查看和对比保存结果。
```

### 第 4 步：启动

按下方“快速启动”安装依赖并运行。被测 SGLang 服务需以
`--speculative-decoding-stats=detailed` 启动；同一台机器多人各跑一个 BenchAPP 时，
`streamlit run` 使用不同的 `--server.port`。

## 快速启动

```bash
cd bench_app
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

浏览器访问 `http://<机器地址>:8501`。Pipeline 发压前需要先确认：

- SGLang OpenAI Chat 服务已经运行。
- 服务启动时已设置 `--speculative-decoding-stats=detailed`。
- 发压期间目标 SGLang 实例没有其他 workload，以免吞吐和时延相互干扰。
- `benchmark_pipeline.toml` 中的 Python、SGLang 仓库和数据集路径正确。
- BenchAPP 所在用户对 `benchmark_workspace/` 有写权限。

服务开关支持 `none|summary|detailed`，默认 `none`。BenchAPP 曲线需要 `detailed`；`summary`
只返回计数、接受率、平均接受长度和直方图。曲线只读取请求响应中的统计，不读取或控制
DSpark 的算法专用调试记录。

## SPEED-Bench 工作流

### 数据集分类

新建发压页面按使用目的组织数据集：

```text
性能压测
└── SPEED-Bench
    └── 1k / 2k / 8k / 16k / 32k

任务评测（当前仍只做发压和性能统计）
├── 综合：SPEED-Bench Qualitative
├── 数学：GSM8K、MATH-500
├── 代码：HumanEval、MBPP Full、MBPP Sanitized
└── 对话与指令：Alpaca
```

SPEED-Bench throughput 的 `low_entropy`、`mixed`、`high_entropy` 是长度档位下的可选熵筛选。
MT-Bench 和 Spec-Bench 含多轮样本，当前 BenchAPP 尚未支持，因此不会生成发压命令。任务数据集
只发送 prompt，不执行答案比对、代码测试或 LLM Judge；答案和测试用例仅保留在原始数据集文件中。

单轮任务数据集会在首次生成命令时转换成 SGLang 已支持的 OpenAI JSONL，缓存于
`benchmark_workspace/dataset_cache/`，不会修改原始数据集。

页面布局和执行流程：

1. 在“新建发压”配置服务地址、数据集、请求数、输出长度和并发。
2. 在页面直接展示的“结果名称与配置”区域填写模型名称以及可选的 TP/DP/EP 和配置说明。
   页面会展示任务结束后使用的完整结果名称；未填写模型名称时不能开始发压。
3. 点击“生成 / 更新发压命令”，在命令编辑器中检查或继续修改 SGLang 参数。
4. 点击“校验命令”，校验通过后点击“开始发压”。命名配置随 Job 一起固化，不受下一项任务修改影响。
5. 如需并行测试其他服务，修改 Host/Port 后生成新命令并继续启动；不同端点各自形成一个 Job。
6. 到“发压任务与日志”选择 Job，查看状态、PID/PGID、最终命令和实时日志，也可单独停止运行中的 Job。
7. 发压成功后在任务页点击“保存所选结果”；保存直接复用发压前配置的名称，全部可比较键都会
   写入记录，`benchmark_pipeline.toml` 只提供结果表格“对比列”的初始选择。
8. 到“结果查询与对比”打开“对比基准”弹出面板，选择基准并查看各指标相对变化；用表格上方的
   “对比列”增减列，或勾选“只显示与基准不同的列”定位两轮之间改了什么。
9. 在同一结果页的“已保存的历史结果”选择旧 Run；可重新读取受管目录中的日志和逐 step 曲线，
   也可更新结果标签，或强制重新解析原始结果刷新表格行。详情默认复用轻量
   `result_view.json`；只有缓存不存在、失效或强制刷新时才扫描完整 `result.jsonl`。
10. 不再需要某条记录时，点击该记录右侧的 🗑️，在确认弹窗中删除对比 JSON。

Runner 按 Run ID 同时管理多个 Job，不建立公共批次。一个 Job 对应一个发压进程和一个
`Host:Port`；同一端点存在 Running Job 时拒绝再次启动，避免两个 workload 相互污染性能
指标。不同端点可以并行运行，历史结果仍由用户自行选择对比。

运行环境和默认参数位于 `benchmark_pipeline.toml`。页面不会把
`PYTHONPATH` 等环境变量显示在命令编辑器中；这些值由 BenchAPP Runner
在启动进程时注入。发请求所需的 Model、Tokenizer 和 Served Model Name 由 benchmark
从服务端读取，页面不会生成对应命令参数；“结果名称与配置”中的模型只是人工结果标签，
在任务启动前绑定到 Job。

默认结果目录为 `benchmark_workspace/<run-id>/`，每次生成命令都会使用
新的 Run ID，避免 SGLang 追加写入时混入旧结果。首次查看成功结果时会在同一目录按需生成
`result_view.json`，只缓存汇总指标、可比较标量和用于绘图的最好/最差请求曲线，并根据原始文件
大小与修改时间自动失效；完整原始结果不会放入浏览器 Session State。

Pipeline 对比记录默认保存到
`benchmark_workspace/saved_results/<run-id>.json`。每个 Run ID 只能首次创建一个表格行，
后续历史更新始终修改同一份记录。如需多台机器查看同一批记录，将
`benchmark_pipeline.toml` 的 `saved_results_root` 改成各机器都能访问的共享目录；
在其他 BenchAPP 实例中点击“刷新保存记录”即可重新读取。

记录会保存 `result.jsonl` 里全部可比较的标量——客户端指标、发压参数回填，以及去重
拍平后的 `server_info`（真实记录约 369 个键、18 KB）。结果表格只通过顶部“对比列”选择当前
需要展示的字段，增减列不会删除保存数据。历史记录更新使用 Revision 检查和原子替换：更新结果
标签不会改动指标；只有用户明确点击“重新解析原始结果并更新”才会刷新指标。初始对比列由
`benchmark_pipeline.toml` 的 `[comparison] default_keys` 控制，当前是 10 个客户端指标 +
4 个服务端指标。TP/DP/EP 不从服务端结果提取、不进入指标选择器，而是在发压前由用户填写。

保存表中的解码速度按 `1000 / Mean TPOT (ms)` 计算。选择一条记录作为对比基准后，
客户端数值列按 `(当前值 - 基准值) / 基准值 × 100%` 显示相对变化；服务端配置列
（表头带 `⚙`）不算百分比，与基准不同时标 `⚠`。
删除记录只会移除 `saved_results/<run-id>.json`，不会删除该 Run 的原始
`result.jsonl`、`benchmark.log` 或可重建的 `result_view.json`。保存、更新和删除同一个 Run 的
记录使用同一把锁，避免共享目录中的更新/删除竞争。

### 输出逐请求明细（`--output-details`）

高级参数里的“输出逐请求明细”默认关闭。开启后 SGLang 会在同一条 `result.jsonl` 记录里
追加 `input_lens` / `output_lens` / `ttfts` / `itls` / `itl_token_lens` /
`generated_texts` / `errors`，用来做单请求粒度的排查：

- `itls` 是**逐 chunk** 的到达间隔，不是逐 token，也不保证一个 chunk 等于一个 step。
- `itl_token_lens` 与 `itls` 逐项对齐，给出该 chunk 出了几个 token。汇总里的
  `mean_itl_ms` 等于 `sum(itls) / sum(itl_token_lens)`，是逐 token 值；两者相差的倍数
  就是接受长度。这个字段需要本仓库配套的 SGLang（见
  `python/sglang/benchmark/serving.py`），旧版 SGLang 不会写它。
- 不完整 UTF-8 和 `stop_str` 前缀命中会让某个 step 不推 SSE，其 token 并进后面的 chunk；
  因此这些数组只用于客户端 ITL/SSE 排查，不用于逐 step 曲线。
- **不开投机解码时该数组为空**（每个请求 `[]`）。此时一个 chunk 就是一个 token，
  `mean_itl_ms` 走原来的逐 chunk 路径，不需要除数。

600 请求 × 32k 输出实测约 +20 MB，其中 `generated_texts` 占一半。这些数组不会进入
`saved_results/<run-id>.json`，对比记录的键数和体积不变。

### 逐 step 接受长度曲线

该折叠面板位于“结果查询与对比”的历史详情中。保存结果后按需加载轻量结果视图，页面画两条：
**请求全程平均接受长度最高和最低**。
数据来自服务端响应中的精确 `verify_lengths / accept_lengths`，不依赖 `--output-details`。

- SGLang 在每个请求的最终响应中返回统计，benchmark 为全部请求按索引落盘；失败或缺失统计
  的请求显式保存 `stats: null`。
- 横轴按每个 step 实际提交的 `acc_len` 推进；纵轴是分档后的平均 `acc_len`，悬浮提示同时
  展示平均 `verify_len`。
- `verify_length` 包含 root token，`accept_length` 包含 target/bonus token，且是 stop/grammar
  裁剪前的 verification 原始值。
- 缺少 detailed 数据时页面只提示服务开关，不使用 SSE chunk 猜测。

曲线只解释当次运行，不写进 `saved_results/`，也不参与对比表格；实际展示的两条曲线会缓存到
Run 目录的 `result_view.json`。逻辑在 `core/step_curve.py`，可以直接对
`benchmark_workspace/*/result.jsonl` 做单测。

Pipeline 的核心代码按“配置、命令、进程、产物、保存记录、对比”拆分在 `core/pipeline/`，三个
页面分别位于 `ui/pipeline/launch.py`、`jobs.py` 和 `results.py`，公共组件及 Session State 定义
单独存放。旧的 `core/benchmark_pipeline.py` 与 `ui/benchmark_pipeline.py` 只保留导入兼容层。


## 计算公式说明

### 1. 平均接受长度 (Average Accept Length)

$$\text{avg\_accept\_len} = 1 + r_1 + r_1 \cdot r_2 + r_1 \cdot r_2 \cdot r_3 + \cdots$$

- 每个 $r_i$ 是第 $i$ 个 MTP Head 的接受率
- Baseline（无投机）时 avg_accept_len = 1

**示例**：三个 Head 接受率 [0.8, 0.74, 0.67]

| Draft Heads | 累计接受长度 |
|-------------|-------------|
| 0 (baseline) | 1.0 |
| 1 | 1 + 0.8 = 1.8 |
| 2 | 1 + 0.8 + 0.8×0.74 = 2.392 |
| 3 | 1 + 0.8 + 0.592 + 0.397 = 2.789 |

### 2. 加速比 (Speedup Ratio)

$$\text{speedup} = \frac{T_{\text{target}} \times \text{avg\_accept\_len}}{T_{\text{target\_verify}} + T_{\text{draft}}}$$

其中：
- $T_{\text{target}}$ — Baseline 模型单步推理耗时（无投机时的 decode 单步）
- $T_{\text{target\_verify}}$ — Target 模型验证步耗时（通常为 $T_{\text{target}}$ 的 1.0~1.5 倍，因为要处理多个候选 token）
- $T_{\text{draft}}$ — Draft Model（Proposer）推理总耗时

### 3. 逐步加速比表 (Incremental Speedup)

表中 "Num Draft Tokens" 列表示投机 token 数量（即 MTP Head 数 / ngram 候选步数 / suffix 候选步数），每行展示在该配置下的：

| 列名 | 含义 |
|------|------|
| Num Draft Tokens | 投机 token 数量（通用，兼容 MTP/ngram/suffix） |
| T_verify (ms) | 该配置下 Target 验证耗时 |
| T_draft (ms) | 该配置下 Draft 总耗时 |
| T_total (ms) | T_verify + T_draft |
| Accept Length | 累计平均接受长度 |
| Speedup vs Baseline | 相对于无投机的加速比 |
| Speedup vs Previous | 相对于少一个 Head 的边际加速比 |

**边际收益递减**：当 Speedup vs Previous < 1.0 时，增加该 Head 反而降低性能（Draft 开销超过了额外接受 token 的收益）。

### 4. 实际案例（ERNIE5 TP4DP16EP64）

| 配置 | T_verify | T_draft | T_total | Accept Len | Speedup |
|------|----------|---------|---------|------------|---------|
| Baseline | 76ms | — | 76ms | 1.0 | 1.0x |
| 1 Head | 90ms | 8ms | 98ms | 1.8 | 1.40x |
| 2 Heads | 104ms | 14ms | 118ms | 2.39 | 1.54x |
| 3 Heads | 117ms | 22ms | 139ms | 2.79 | 1.53x |

可以看到第 3 个 Head 的边际收益接近持平（1.53 vs 1.54），再多加 Head 可能得不偿失。

## 灵敏度分析

Speedup Calculator 页面提供热力图，展示：
- **X 轴**：均匀接受率（所有 Head 相同）
- **Y 轴**：Verify Overhead 倍率
- **颜色**：对应的加速比

关键发现：
- 接受率 > 70% 且 Verify Overhead < 1.3x 时，加速效果显著
- 接受率 < 60% 时，无论 overhead 多低都很难获得明显加速

## 运行测试

```bash
cd /path/to/fd_dev
python -m pytest bench_app/tests/ -v
```
