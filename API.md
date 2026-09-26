# 评测 HTTP API

服务接收 HTTP 请求，供 Agent 或其他系统直接控制评测。
核心流程：**提交评测 → 轮询状态 → 获取结果**；评测结束后自动保存。

下面提供可以直接粘贴到终端执行的 curl 命令。curl 只是发送 HTTP 请求的工具，不需要导入任何 Python 客户端。
请求体模板已放在 [`examples/evaluation_request.json`](examples/evaluation_request.json)。

**Agent 按这个顺序操作：**

1. 确认 API 地址、被测 SGLang 地址和评测参数；使用第二节的命令生成请求文件及唯一编号。
2. 调用健康检查和参数校验；响应 `ok: false` 时读取 `error.message`，不要继续提交。
3. 提交一次，保存返回的 `data.run_id`。提交后不要重复发送 POST 来查询进度。
4. 用这个编号查询状态；`running` 时继续等待，`succeeded/failed/stopped` 时获取结果。
5. 从结果的 `data.metrics` 读取指标；失败时读取 `data.error` 和 `data.log_tail`。
6. 只有需要提前终止时才调用取消接口。无需调用“保存”，结果自动持久化。

所有查询/取消 URL 都使用同一个评测编号。第二节是可执行示例，第三节是完整参数和响应说明。

## 1. 启动服务与准备配置

在 API 服务器上执行：

```bash
cd /root/paddlejob/inference-public/jianghaitao/Spec_bench_app
python api.py --host 127.0.0.1 --port 8765
```

保持服务运行。也可从父目录运行 `python -m Spec_bench_app.api`。
指定配置使用 `--config /absolute/path/benchmark_pipeline.toml`。

启动前确认 `benchmark_pipeline.toml`：

| 配置 | 含义 |
|---|---|
| environment.python_executable | 可运行 SGLang benchmark 的 Python |
| environment.sglang_repo | 包含 python/sglang 的仓库路径 |
| environment.dataset_root | SPEED-Bench 数据目录，下面的示例需要 throughput_1k/test.jsonl |
| environment.workspace_root | 评测日志、原始结果和持久评测记录目录 |
| environment.saved_results_root | 成功评测自动生成的页面兼容结果记录目录 |

相对路径以配置文件所在目录为基准。
被测 SGLang 必须已经启动；API 只启动 benchmark，不负责部署或启动 SGLang。

下面示例使用两个地址：

- `http://127.0.0.1:8765`：BenchAPP API 地址，接收任务管理请求。
- `127.0.0.1:30131`：被测 SGLang 地址，接收 benchmark 发出的推理请求。

请求体的 host/port 必须能从 **API 服务器** 访问；HTTP URL 必须能从 **调用方** 访问。
跨机器时不能都照抄 127.0.0.1。默认 API 只监听本机且没有认证，适用于受信任的调用环境。

API 与 Streamlit 独立启动，没有依赖；运行中任务不跨进程共享。成功评测使用相同的
saved_results_root 时，保存记录可在页面中查看。

## 2. 终端中完整执行一次评测

### 先解释评测 ID 和 URL

`run_id` 是一次评测的唯一编号，不是文件路径、模型名称或必须使用的固定名称。
编号用来区分多次评测；例如同一模型测两种并发，需要两个不同编号。

下面使用 `RUN_ID` 变量保存本次编号，自动按时间生成，无需手写示例 ID。

- `/api/v1/evaluations/$RUN_ID`：查询这个编号对应评测的状态。
- `/api/v1/evaluations/$RUN_ID/result`：获取同一次评测的指标、曲线和错误信息。
- `/api/v1/evaluations/$RUN_ID/log`：获取同一次评测的日志。
- `/api/v1/evaluations/$RUN_ID:cancel`：停止同一次评测。

URL 里的 `/result` 是接口名称，**不是让你进入某个本地目录**。

### 第一步：终端 A 启动 API

如果 API 已经在运行，跳过这一步，不要重复启动。

```bash
cd /root/paddlejob/inference-public/jianghaitao/Spec_bench_app
python api.py --host 127.0.0.1 --port 8765
```

保持终端 A 运行。后续命令在同一机器的另一个终端 B 执行。

### 第二步：终端 B 设置地址、生成评测编号并创建 JSON 文件

复制下面整个代码块执行。把 host/port 改成实际被测 SGLang 地址，把 model_name 改成你的模型标签。

```bash
API_URL="http://127.0.0.1:8765"
RUN_ID="evaluation-$(date +%Y%m%d-%H%M%S)-$$"
REQUEST_FILE="/tmp/${RUN_ID}.json"

cat > "$REQUEST_FILE" <<JSON
{
  "run_id": "$RUN_ID",
  "parameters": {
    "host": "127.0.0.1",
    "port": 30131,
    "dataset_id": "speed-bench",
    "dataset_variant": "1k",
    "num_prompts": 20,
    "output_len": 128,
    "max_concurrency": 4
  },
  "metadata": {"model_name": "my-model"},
  "timeout_seconds": 1800
}
JSON

echo "本次评测编号：$RUN_ID"
echo "请求文件位置：$REQUEST_FILE"
```

这段命令会在 `/tmp` 创建本次请求文件，并打印其路径和本次编号。完整参数模板也在
`examples/evaluation_request.json`，所有可配置字段见第三节。
未填写的压测参数使用服务端配置默认值。

`cat` 后到独立一行的 `JSON` 之间都是文件内容，最后一行 JSON 必须一并复制。
后面的所有命令继续使用这个终端 B；换了终端需重新设置 API_URL 和实际 RUN_ID。

### 第三步：检查 API 是否可用

```bash
curl -sS "$API_URL/health"
```

正常返回：

```json
{"ok":true,"data":{"status":"ok"},"error":null}
```

### 第四步：校验参数，不启动 benchmark

```bash
curl -sS -X POST "$API_URL/api/v1/evaluations/validate" \
  -H 'Content-Type: application/json' \
  --data-binary "@$REQUEST_FILE"
```

`--data-binary "@$REQUEST_FILE"` 表示读取刚创建的文件，将其 JSON 内容发送给 API。
成功时返回 `ok: true`、`data.valid: true` 和实际生成的 command/argv。
失败时阅读 `error.message` 并修改请求文件，不要继续提交。
校验不进行真实推理，不保证模型服务一定可用。

### 第五步：正式启动评测

```bash
curl -sS -X POST "$API_URL/api/v1/evaluations" \
  -H 'Content-Type: application/json' \
  --data-binary "@$REQUEST_FILE"
```

返回 `ok: true` 且 `data.status: "running"` 表示任务已启动，**不代表已完成**。
HTTP 状态码是 202。响应中的 data.run_id 就是刚才的 RUN_ID。
已有编号不能重复提交；需要新评测时从第二步生成新编号和文件。
响应结构如下，`data.run_id` 的值以本次返回为准：

```json
{
  "ok": true,
  "data": {
    "run_id": "这里是本次返回的实际编号",
    "status": "running",
    "started_at": "2026-09-26T12:00:00+00:00",
    "ended_at": null,
    "return_code": null,
    "error": null
  },
  "error": null
}
```

上面仅说明响应字段，不是新请求模板。后续命令继续使用已设置的 `$RUN_ID`，无需复制响应示例中的说明文字。

### 第六步：查询状态

```bash
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID"
```

查看返回的 `data.status`：

| 值 | 说明 |
|---|---|
| running | 还在运行，稍后再次执行查询 |
| succeeded | 已成功完成并自动保存，获取结果即可 |
| failed | 失败，获取结果查看 error 和 log_tail |
| stopped | 已取消或超时，获取结果查看具体原因 |

如果终端安装了 watch，可以每秒自动查询一次：

```bash
watch -n 1 "curl -sS '$API_URL/api/v1/evaluations/$RUN_ID'"
```

在终端 B 按 Ctrl-C 只退出这个 watch，不会取消评测。

### 第七步：获取结果

等状态不再是 running 后执行：

```bash
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/result"
```

这个接口返回的是完整 JSON 数据；下面是成功响应中部分字段的示意，指标数值只作说明：

```json
{
  "ok": true,
  "data": {
    "run_id": "这里是本次返回的实际编号",
    "status": "succeeded",
    "return_code": 0,
    "error": null,
    "metrics": {"output_throughput": 80.0, "mean_ttft_ms": 10.0},
    "curves": null
  },
  "error": null
}
```

实际响应还有下面这些字段，完整定义见第三节。返回 data.metrics（指标）、data.values（更多统计字段）、data.curves（接受长度曲线）、
data.error（失败原因）、data.log_tail（日志摘要）和 data.artifacts（服务器产物路径）。
未提供 detailed 统计时 curves 为 null；它不是 draft 接受率报告。

如需把 HTTP 响应保存到当前机器的文件：

```bash
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/result" \
  -o "/tmp/${RUN_ID}-response.json"
```

尚在运行时获取结果返回 HTTP 409；应继续查询状态。
不需要调用 save，API 已自动保存。失败任务也能查询结果，HTTP 200 表示查询成功，
评测是否成功要看 data.status。

### 可选操作：查看日志

```bash
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/log?max_bytes=65536"
```

日志在响应的 data.text 字段，max_bytes 表示最多读取末尾多少字节。

### 可选操作：停止评测

只有需要提前停止时执行，不要把它当作获取成功结果的必经步骤：

```bash
curl -sS -X POST "$API_URL/api/v1/evaluations/$RUN_ID:cancel"
```

停止的是 benchmark，不是 SGLang。重复取消或取消已完成的任务不会改写原来的结果。

### 重新打开终端或重启 API 后查询

将下面的值替换成此前打印出来的真实编号：

```bash
API_URL="http://127.0.0.1:8765"
RUN_ID="这里填写此前返回的实际评测编号"
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID"
curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/result"
```

正常完成的任务重启后仍使用相同的接口。产物默认在服务器的：

```text
benchmark_workspace/<实际评测编号>/evaluation.json
benchmark_workspace/<实际评测编号>/benchmark.log
benchmark_workspace/<实际评测编号>/result.jsonl
benchmark_workspace/<实际评测编号>/result_view.json
benchmark_workspace/saved_results/<实际评测编号>.json
```

失败时原始结果/曲线缓存可能不存在；saved_results 记录仅成功时生成。
artifacts 返回服务器路径，不是下载链接；指标、曲线和日志摘要可以直接从 API 响应获取。

## 3. 接口与终端命令速查

以下速查命令使用第二节已设置的 API_URL、RUN_ID、REQUEST_FILE 变量。

| 操作 | 直接在终端执行的命令 |
|---|---|
| 提交 | `curl -sS -X POST "$API_URL/api/v1/evaluations" -H 'Content-Type: application/json' --data-binary "@$REQUEST_FILE"` |
| 查询状态 | `curl -sS "$API_URL/api/v1/evaluations/$RUN_ID"` |
| 获取结果 | `curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/result"` |
| 停止评测 | `curl -sS -X POST "$API_URL/api/v1/evaluations/$RUN_ID:cancel"` |
| 校验 | `curl -sS -X POST "$API_URL/api/v1/evaluations/validate" -H 'Content-Type: application/json' --data-binary "@$REQUEST_FILE"` |
| 查看日志 | `curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/log?max_bytes=65536"` |
| 健康检查 | `curl -sS "$API_URL/health"` |

`curl -sS` 隐藏传输进度但显示连接错误；`-X POST` 指定 POST 请求；
`-H` 指定 JSON 请求头；`--data-binary @文件路径` 将文件内容作为请求体发送。
提交成功返回 HTTP 202，其他成功操作返回 200。

只接收结构化参数，不再接收任意 `command` 字段。validate 会返回生成的命令以便核对。

### 请求体字段

submit 和 validate 使用同一结构，未知字段会返回 422，不会静默忽略。

| 字段 | 类型 | 必填/默认 | 含义 |
|---|---|---|---|
| `run_id` | string | 可选；服务生成 | 以字母、数字、下划线或短横线开头，后续还可含点；不能有 `/`。已有 ID 返回 409 |
| `parameters` | object | 可省略 | 未提供的参数从服务端配置取默认值 |
| `metadata` | object | 必填 | 结果命名和部署标签 |
| `timeout_seconds` | number | 1800 | 从提交启动后计时的服务端最大评测时长，必须有限且大于 0 |

validate 不保留 ID，不占用 endpoint；校验后仍可能因其他请求占用同一端点而提交失败。
未提供 run_id 时，validate 中的预览 ID 与随后 submit 自动生成的 ID 可以不同。
建议调用方在提交前指定唯一 ID，便于在网络错误后查询该任务。

`parameters` 全部支持字段如下：

| 字段 | JSON 类型 | 含义 |
|---|---|---|
| `host` | string | 被测 SGLang 地址 |
| `port` | integer | 被测端口，1–65535 |
| `benchmark_family` | string | `performance` 或 `task_evaluation`，默认 performance |
| `dataset_id` | string | 默认 `speed-bench`；支持的数据集见仓库 README 的数据集说明 |
| `dataset_variant` | string | SPEED-Bench 为 `1k/2k/8k/16k/32k`；无变体的任务数据集传空字符串 |
| `dataset_size` | string | 兼容旧长度字段；未显式提供 dataset_variant 时作为其值，推荐只用 dataset_variant |
| `category` | string | SPEED-Bench 熵分类：空字符串、low_entropy、mixed、high_entropy |
| `num_prompts` | integer | 请求数量，正整数 |
| `output_len` | integer | 请求输出长度，正整数 |
| `max_concurrency` | integer | 最大并发，正整数 |
| `request_rate` | string | 正数的字符串或 `"inf"`；不要写 JSON 数字 Infinity |
| `seed` | integer | 随机种子 |
| `warmup_requests` | integer | 预热请求数量，非负整数 |
| `extra_request_body` | string | JSON 对象的序列化字符串，如 `"{\"temperature\":0}"`；不是嵌套对象 |
| `output_details` | boolean | 是否输出逐请求明细，使用 true/false，不要用字符串 |

除 family/id 外，具体默认值来自 `benchmark_pipeline.toml` 的 `[defaults]`；validate 返回的
`data.request.parameters` 包含补齐后的实际配置。
任务数据集例如 `benchmark_family: "task_evaluation"`、`dataset_id: "gsm8k"`、`dataset_variant: ""`。
它仍然是该数据集的发压，不会评价答案正确率。

`metadata`：`model_name` 为必填非空字符串；`tp_size/dp_size/ep_size` 是可选正整数或 null；
`additional` 是可选说明字符串。所有标签只用于保存结果，不会更改部署。

cancel 无需请求体，或使用 `{}`。log 的 max_bytes 范围为 1–4194304。

### 响应结构

所有接口统一返回，业务内容在 `data` 中：

```json
{"ok": true, "data": {}, "error": null}
```

请求被拒绝时：

```json
{
  "ok": false,
  "data": null,
  "error": {"code": "VALIDATION_ERROR", "message": "metadata.model_name is required"}
}
```

submit、状态查询、cancel 的 `data` 字段一致，例如：

```json
{
  "run_id": "这里是本次返回的实际编号",
  "status": "running",
  "started_at": "2026-09-26T12:00:00+00:00",
  "ended_at": null,
  "return_code": null,
  "error": null
}
```

状态为 `running`、`succeeded`、`failed`、`stopped`。只有结果解析和自动保存完成后才发布
`succeeded`。进程退出码为 0 但结果缺失/损坏时仍是 failed。

result 接口 `data` 包含以下完整字段：

| 字段 | 内容 |
|---|---|
| `schema_version` | 当前为 1 |
| `run_id/status/started_at/ended_at/return_code/error` | 与状态接口相同 |
| `request` | 补齐默认值后的请求，包含实际 run_id |
| `command` | 执行的 benchmark 命令，便于复核 |
| `metrics` | accept_length、decode_speed_toks、request/input/output/total_throughput、mean_e2e_latency_ms、mean_ttft_ms、mean_tpot_ms、mean_itl_ms |
| `values` | 更多可比较的 client.* / server.* 标量字段，具体键取决于 benchmark 输出 |
| `curves` | 已有的最好/最差请求接受长度曲线；没有 detailed 统计时为 null |
| `log_tail` | 结束时的 benchmark 日志尾部（最多 64 KiB） |
| `artifacts` | evaluation、result_jsonl、benchmark_log、result_view、saved_record 的服务端路径；未保存成功时 saved_record 为 null |

`metrics.accept_length` 是每次验证步的平均接受长度，不是 draft 接受率。
API 沿用已有接受长度统计，不生成新的接受率报告。
失败或停止时指标通常为空；若指标已解析但保存失败，可能保留指标，请始终先检查 status。

validate 的 data 为 `valid/request/command/argv`；它检查请求、数据集和执行路径，
不进行推理探测，不保证真实 SGLang 推理一定成功。任务数据集校验可能生成本地转换缓存。
log 的 data 为 `{"run_id":"...","text":"日志内容"}`。

## 4. 失败、取消、超时和重启

| HTTP 状态 | 情况 |
|---|---|
| 400 | 非法 JSON、非对象请求体、非有限 JSON 数值 |
| 404 | 未知路由或没有该评测记录 |
| 409 | 重复 ID、目标 endpoint 正在发压、尚未结束时获取 result |
| 413 | 请求体超过 4 MiB |
| 422 | 请求参数或数据集/命令校验失败 |
| 500 | 存储或其他未预期错误 |
| 503 | 服务正在关闭 |

终态结果中的 `error.code`：

| code | 含义与处理 |
|---|---|
| BENCHMARK_FAILED | benchmark 非零退出；查看 return_code 和 log_tail |
| INVALID_RESULT | 结果缺失、损坏或没有可用指标；查看原始日志 |
| CANCELLED | 主动取消或服务正常退出时停止任务 |
| TIMEOUT | 服务端 timeout_seconds 到期，benchmark 被停止 |
| SAVE_FAILED | 成功解析，但页面兼容记录保存失败；检查存储路径/权限 |
| FINALIZATION_FAILED | 收尾出现未预期异常；查看 API 进程日志 |
| INTERRUPTED | 新进程读取到上次未完成记录；不会接管旧 PID |

- 状态/结果接口的 `ok: true` 表示查询成功，不能替代 `data.status == "succeeded"`。
- 调用方停止轮询或 HTTP 连接断开不会自动取消任务；需要时调用取消接口。
- 服务端 `timeout_seconds` 才会终止 benchmark；停止进程和写结果可能多花几秒。
- Ctrl-C/SIGTERM 正常退出会停止自己的运行任务并保存 stopped 记录；不会停止 SGLang。
- 正常完成的成功、失败、停止结果，重启后仍可查询，无需换接口或解析原始文件。
- kill -9/崩溃可能留下 benchmark；重启后的首次查询会把未收尾记录标记 failed/INTERRUPTED。
  不接管旧进程、不自动重发压，需要人工确认原 benchmark 是否还在运行。
- 同一个 workspace 只运行一个 API 实例；API 和独立 Streamlit 进程之间不共享 endpoint 锁，
  避免同时对同一 SGLang 发压。
- POST 发生网络超时可能已经提交成功。建议调用方提前设置唯一 run_id，按该 ID 查询再决定后续操作。
- 若存储不可写，任何 API 都无法保证结果持久化；此时会返回错误或记录 FINALIZATION_FAILED，
  并在 API 日志输出原因。恢复存储前不要把内存状态当成可跨重启恢复的结果。

## 5. 如何验证实现

从父目录运行：

```bash
python -m pytest Spec_bench_app/tests -q
```

`tests/test_api.py` 使用临时数据集和 fake benchmark，并通过真实本地 HTTP 请求验证：
自动收尾（不轮询也保存）、重启恢复、异常结果、非零退出、取消、超时、并发冲突、移除旧路由，
以及实际 JSON 请求模板 `examples/evaluation_request.json` 的提交和结果查询。
`tests/test_api_commands.py` 直接读取并执行本页的终端命令，验证文档与 API 一致。
测试只监听系统分配的临时端口，结束时关闭监听并检查 benchmark 已结束，不占用 8765。

这些测试不调用真实模型，临时 JSON 和产物在测试临时目录；不是性能或 SGLang 端到端测试。
按本指南第二节向启动的 API 发送请求，才会连接你配置的真实推理服务。
