"""Tests for the isolated SPEED-Bench Pipeline feature."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ..core.pipeline import artifacts as artifacts_module
from ..core.pipeline import results as results_module
from ..core.pipeline import runner as runner_module
from ..core.benchmark_pipeline import (
    STATUS_STOPPED,
    STATUS_SUCCEEDED,
    BenchmarkParameters,
    BenchmarkRunner,
    CommandValidationError,
    PipelineError,
    PipelineSettings,
    ProcessStateError,
    ResultRecordConflictError,
    ResultRecordExistsError,
    benchmark_result_metrics,
    build_configuration_description,
    build_job_metadata,
    build_result_name,
    build_saved_benchmark_result,
    delete_saved_benchmark_result,
    generate_command,
    list_saved_benchmark_results,
    load_or_create_result_view,
    job_metadata_from_saved_result,
    parse_command,
    parse_result_file,
    read_log_tail,
    relative_change_percent,
    replace_saved_benchmark_result,
    result_view_path,
    revise_saved_benchmark_result,
    save_benchmark_result,
    saved_run_artifacts,
    validate_command,
)


def _write_fake_sglang_module(repo: Path) -> None:
    package = repo / "python" / "sglang" / "benchmark"
    package.mkdir(parents=True)
    (repo / "python" / "sglang" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "serving.py").write_text(
        """
import argparse
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--output-file", required=True)
parser.add_argument("--fake-sleep", type=float, default=0.05)
args, _ = parser.parse_known_args()

print("fake benchmark started", flush=True)
time.sleep(args.fake_sleep)
result = {
    "completed": 2,
    "duration": 0.05,
    "request_throughput": 40.0,
    "input_throughput": 100.0,
    "output_throughput": 80.0,
    "total_throughput": 180.0,
    "mean_e2e_latency_ms": 30.0,
    "mean_ttft_ms": 10.0,
    "mean_tpot_ms": 2.0,
    "mean_itl_ms": 2.1,
    "p95_ttft_ms": 10.0,
    "p95_tpot_ms": 2.0,
    "p95_e2e_latency_ms": 30.0,
    "accept_length": 1.5,
}
output = Path(args.output_file)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(result) + "\\n")
print("fake benchmark completed", flush=True)
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture
def pipeline_setup(tmp_path: Path) -> tuple[PipelineSettings, BenchmarkParameters]:
    repo = tmp_path / "sglang"
    _write_fake_sglang_module(repo)

    dataset_root = tmp_path / "datasets"
    dataset_file = dataset_root / "throughput_1k" / "test.jsonl"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text(
        '{"turns":["hello"],"category":"mixed"}\n', encoding="utf-8"
    )

    qualitative_file = dataset_root / "qualitative" / "test.jsonl"
    qualitative_file.parent.mkdir(parents=True)
    qualitative_file.write_text(
        '{"turns":["hello"],"category":"qa"}\n', encoding="utf-8"
    )

    settings = PipelineSettings(
        python_executable=Path(sys.executable).resolve(),
        sglang_repo=repo,
        dataset_root=dataset_root,
        workspace_root=tmp_path / "workspace",
        saved_results_root=tmp_path / "workspace" / "saved_results",
        host="127.0.0.1",
        port=30000,
        ready_timeout=1,
        dataset_size="1k",
        category="mixed",
        num_prompts=2,
        output_len=8,
        max_concurrency=1,
        request_rate="inf",
        seed=42,
        warmup_requests=0,
        extra_request_body='{"temperature":0}',
    )
    parameters = BenchmarkParameters(
        host=settings.host,
        port=settings.port,
        dataset_size=settings.dataset_size,
        category=settings.category,
        num_prompts=settings.num_prompts,
        output_len=settings.output_len,
        max_concurrency=settings.max_concurrency,
        request_rate=settings.request_rate,
        seed=settings.seed,
        warmup_requests=settings.warmup_requests,
        extra_request_body=settings.extra_request_body,
    )
    return settings, parameters


def _wait_for_completion(
    runner: BenchmarkRunner,
    run_id: str | None = None,
    timeout_seconds: float = 5.0,
):
    deadline = time.monotonic() + timeout_seconds
    snapshot = runner.snapshot(run_id)
    while snapshot is not None and snapshot.status == "Running":
        if time.monotonic() >= deadline:
            pytest.fail("fake benchmark did not finish")
        time.sleep(0.02)
        snapshot = runner.snapshot(run_id)
    return snapshot


def test_generate_and_validate_editable_command(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    command, run_id = generate_command(settings, parameters, "speed_bench_unit")

    assert run_id == "speed_bench_unit"
    assert "PYTHONPATH=" not in command
    assert "\\\n" in command
    assert "--speed-bench-category mixed" in command
    assert "--model" not in command
    assert "--tokenizer" not in command
    assert "--served-model-name" not in command

    parsed = parse_command(command)
    assert parsed[:3] == (
        str(settings.python_executable),
        "-m",
        "sglang.benchmark.serving",
    )

    edited = f"{command} \\\n  --disable-tqdm"
    validated = validate_command(edited, settings)
    assert validated.run_id == "speed_bench_unit"
    assert validated.argv[-1] == "--disable-tqdm"
    assert validated.context == {
        "host": "127.0.0.1",
        "port": "30000",
        "dataset_path": str(settings.dataset_root / "throughput_1k" / "test.jsonl"),
        "category": "mixed",
        "num_prompts": "2",
        "output_len": "8",
        "max_concurrency": "1",
        "request_rate": "inf",
        "dataset_name": "speed-bench",
        "dataset_id": "speed-bench",
        "dataset_variant": "1k",
        "benchmark_family": "performance",
    }


def test_qualitative_command_ignores_entropy_category(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    qualitative = BenchmarkParameters(
        **{
            **parameters.__dict__,
            "dataset_size": "qualitative",
            "category": "mixed",
        }
    )

    command, _ = generate_command(settings, qualitative, "qualitative_unit")

    assert "--speed-bench-category" not in command
    assert "/qualitative/test.jsonl" in command


def test_single_turn_task_dataset_generates_openai_command(
    pipeline_setup,
    tmp_path: Path,
) -> None:
    settings, parameters = pipeline_setup
    builtin_root = tmp_path / "builtin"
    source = builtin_root / "gsm8k" / "test.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps({"question": "What is 2+2?", "answer": "4"}) + "\n",
        encoding="utf-8",
    )
    settings = PipelineSettings(
        **{**settings.__dict__, "builtin_dataset_root": builtin_root}
    )
    task = BenchmarkParameters(
        **{
            **parameters.__dict__,
            "benchmark_family": "task_evaluation",
            "dataset_id": "gsm8k",
            "dataset_variant": "",
            "category": "mixed",
        }
    )

    command, _ = generate_command(settings, task, "gsm8k_unit")
    assert "--dataset-name openai" in command
    assert "--sharegpt-output-len" in command
    assert "--speed-bench-category" not in command
    validated = validate_command(command, settings)
    assert validated.context["dataset_id"] == "gsm8k"
    cached = Path(validated.context["dataset_path"])
    assert json.loads(cached.read_text(encoding="utf-8"))["messages"][0]["content"] == (
        "What is 2+2?"
    )


def test_multiturn_task_datasets_are_explicitly_unsupported(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    for dataset_id in ("mt-bench", "spec-bench"):
        task = BenchmarkParameters(
            **{
                **parameters.__dict__,
                "benchmark_family": "task_evaluation",
                "dataset_id": dataset_id,
                "dataset_variant": "",
            }
        )
        with pytest.raises(PipelineError, match="暂不支持"):
            generate_command(settings, task, f"{dataset_id}_unit")


@pytest.mark.parametrize(
    ("dataset_id", "variant", "relative_path", "row", "expected_prompt"),
    [
        (
            "math500",
            "",
            "math500/test.jsonl",
            {"problem": "Solve x+1=2", "answer": "1"},
            "Solve x+1=2",
        ),
        (
            "humaneval",
            "",
            "humaneval/HumanEval.jsonl",
            {"prompt": "def add(a, b):\n", "test": "assert add(1, 2) == 3"},
            "def add(a, b):",
        ),
        (
            "mbpp",
            "full",
            "mbpp/mbpp.jsonl",
            {"text": "Write an add function", "code": "def add(a,b): return a+b"},
            "Write an add function",
        ),
        (
            "mbpp",
            "sanitized",
            "mbpp/sanitized-mbpp.json",
            [{"prompt": "Write a subtract function", "code": "def sub(a,b): return a-b"}],
            "Write a subtract function",
        ),
        (
            "alpaca",
            "",
            "alpaca/alpaca_data.json",
            [{"instruction": "Summarize this", "input": "A short article", "output": "Summary"}],
            "Summarize this\n\nA short article",
        ),
    ],
)
def test_task_dataset_adapters_only_send_prompts(
    pipeline_setup,
    tmp_path: Path,
    dataset_id: str,
    variant: str,
    relative_path: str,
    row,
    expected_prompt: str,
) -> None:
    settings, parameters = pipeline_setup
    builtin_root = tmp_path / "builtin"
    source = builtin_root / relative_path
    source.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json":
        source.write_text(json.dumps(row), encoding="utf-8")
    else:
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
    settings = PipelineSettings(
        **{**settings.__dict__, "builtin_dataset_root": builtin_root}
    )
    task = BenchmarkParameters(
        **{
            **parameters.__dict__,
            "benchmark_family": "task_evaluation",
            "dataset_id": dataset_id,
            "dataset_variant": variant,
            "category": "",
        }
    )

    command, _ = generate_command(
        settings,
        task,
        f"{dataset_id}_{variant or 'default'}_unit",
    )
    validated = validate_command(command, settings)
    cached_row = json.loads(
        Path(validated.context["dataset_path"]).read_text(encoding="utf-8")
    )
    assert cached_row == {
        "messages": [{"role": "user", "content": expected_prompt}]
    }


def test_output_details_flag_is_opt_in_and_survives_the_editor(
    pipeline_setup,
) -> None:
    settings, parameters = pipeline_setup
    assert (
        "--output-details"
        not in generate_command(settings, parameters, "speed_bench_plain")[0]
    )

    detailed = BenchmarkParameters(**{**parameters.__dict__, "output_details": True})
    command, _ = generate_command(settings, detailed, "speed_bench_details")

    # A valueless flag is the one shape `format_command` can mangle, and
    # `validate_command` is what the page runs on the edited text.
    validated = validate_command(command, settings)
    flag_index = validated.argv.index("--output-details")
    assert validated.argv[flag_index + 1] == "--output-file"
    assert validated.run_id == "speed_bench_details"


def test_configuration_description_and_result_name_are_user_metadata() -> None:
    configuration = build_configuration_description(
        tp_size=8,
        dp_size=2,
        ep_size=4,
        additional="flashinfer + W4A8",
    )

    assert configuration == "TP=8 | DP=2 | EP=4 | flashinfer + W4A8"
    assert build_result_name("DeepSeek-V4", configuration) == (
        "DeepSeek-V4 + TP=8 | DP=2 | EP=4 | flashinfer + W4A8"
    )
    assert build_configuration_description(additional="manual") == "manual"
    with pytest.raises(PipelineError, match="TP Size"):
        build_configuration_description(tp_size=0)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda command, settings: command + " ; echo unexpected",
        lambda command, settings: command.replace(
            "sglang.benchmark.serving", "some.other.module"
        ),
        lambda command, settings: command.replace(
            str(settings.workspace_root / "speed_bench_invalid" / "result.jsonl"),
            str(settings.workspace_root.parent / "outside.jsonl"),
        ),
    ],
)
def test_rejects_unsafe_or_unmanaged_commands(pipeline_setup, mutator) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(
        settings,
        parameters,
        "speed_bench_invalid",
    )

    with pytest.raises(CommandValidationError):
        validate_command(mutator(command, settings), settings)


def test_parse_result_file_uses_last_jsonl_record(tmp_path: Path) -> None:
    result_file = tmp_path / "result.jsonl"
    result_file.write_text(
        '{"completed": 1}\n\n{"completed": 2, "request_throughput": 3.5}\n',
        encoding="utf-8",
    )

    result = parse_result_file(result_file)

    assert result == {"completed": 2, "request_throughput": 3.5}


def test_result_view_caches_metrics_and_plotted_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_file = tmp_path / "run" / "result.jsonl"
    result_file.parent.mkdir()
    result = {
        "completed": 1,
        "accept_length": 2.0,
        "request_throughput": 1.0,
        "input_throughput": 2.0,
        "output_throughput": 3.0,
        "total_throughput": 5.0,
        "mean_e2e_latency_ms": 6.0,
        "mean_ttft_ms": 7.0,
        "mean_tpot_ms": 8.0,
        "mean_itl_ms": 9.0,
        "speculative_decoding_stats": {
            "schema_version": 1,
            "requests": [
                {
                    "request_index": 0,
                    "stats": {
                        "mode": "detailed",
                        "verify_lengths": [3, 3],
                        "accept_lengths": [2, 1],
                    },
                }
            ],
        },
    }
    result_file.write_text(json.dumps(result) + "\n", encoding="utf-8")

    view = load_or_create_result_view(result_file)

    assert view is not None
    assert view.metrics["output_throughput"] == pytest.approx(3.0)
    assert view.curves is not None
    assert view.curves.num_requests == 1
    sidecar = result_view_path(result_file)
    assert sidecar.is_file()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert "speculative_decoding_stats" not in payload

    updated = {**result, "output_throughput": 6.0}
    with result_file.open("a", encoding="utf-8") as result_stream:
        result_stream.write(json.dumps(updated) + "\n")
    refreshed = load_or_create_result_view(result_file)
    assert refreshed is not None
    assert refreshed.metrics["output_throughput"] == pytest.approx(6.0)
    monkeypatch.setattr(
        artifacts_module,
        "parse_result_file",
        lambda _path: pytest.fail("valid result view should avoid raw JSON parsing"),
    )
    cached = load_or_create_result_view(result_file)
    assert cached is not None
    assert cached.curves == refreshed.curves


def test_extracts_saved_metrics_and_computes_decode_speed() -> None:
    result = {
        "completed": 10,
        "accept_length": 5.19,
        "request_throughput": 0.68,
        "input_throughput": 22273.70,
        "output_throughput": 2789.27,
        "total_throughput": 25062.97,
        "mean_e2e_latency_ms": 34961.04,
        "mean_ttft_ms": 4368.48,
        "mean_tpot_ms": 7.47,
        "mean_itl_ms": 7.49,
    }

    metrics = benchmark_result_metrics(result)

    assert metrics["accept_length"] == pytest.approx(5.19)
    assert metrics["decode_speed_toks"] == pytest.approx(133.8688)
    assert metrics["mean_tpot_ms"] == pytest.approx(7.47)


def test_saves_one_json_per_run_and_loads_shared_history(tmp_path: Path) -> None:
    result = {
        "completed": 10,
        "accept_length": 5.19,
        "request_throughput": 0.68,
        "input_throughput": 22273.70,
        "output_throughput": 2789.27,
        "total_throughput": 25062.97,
        "mean_e2e_latency_ms": 34961.04,
        "mean_ttft_ms": 4368.48,
        "mean_tpot_ms": 7.47,
        "mean_itl_ms": 7.49,
    }
    record = build_saved_benchmark_result(
        result,
        run_id="speed_bench_saved_unit",
        model_name="sglang dspark 5",
        configuration="TP=8 | DP=1 | EP=8 | flashinfer + W4A8",
        configuration_options={"tp_size": 8, "dp_size": 1, "ep_size": 8},
        result_file=tmp_path / "run" / "result.jsonl",
        log_file=tmp_path / "run" / "benchmark.log",
        context={"max_concurrency": 24},
    )

    saved_path = save_benchmark_result(record, tmp_path / "shared")
    records, warnings = list_saved_benchmark_results(tmp_path / "shared")

    assert saved_path.name == "speed_bench_saved_unit.json"
    assert warnings == []
    assert len(records) == 1
    assert records[0]["model_name"] == "sglang dspark 5"
    assert records[0]["result_name"] == (
        "sglang dspark 5 + TP=8 | DP=1 | EP=8 | flashinfer + W4A8"
    )
    assert records[0]["metrics"]["decode_speed_toks"] == pytest.approx(133.8688)
    with pytest.raises(ResultRecordExistsError):
        save_benchmark_result(record, tmp_path / "shared")


def test_saved_history_resolves_only_managed_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_id = "speed_bench_history_unit"
    run_dir = workspace / run_id
    run_dir.mkdir(parents=True)
    result_file = run_dir / "custom-result.jsonl"
    log_file = run_dir / "benchmark.log"
    result_file.write_text('{"completed": 1}\n', encoding="utf-8")
    log_file.write_text("done\n", encoding="utf-8")

    artifacts = saved_run_artifacts(
        {
            "run_id": run_id,
            "source": {
                "result_file": str(result_file),
                "log_file": str(log_file),
            },
        },
        workspace,
    )
    assert artifacts.result_file == result_file.resolve()
    assert artifacts.log_file == log_file.resolve()

    untrusted = saved_run_artifacts(
        {
            "run_id": run_id,
            "source": {
                "result_file": "/etc/passwd",
                "log_file": "/tmp/unmanaged.log",
            },
        },
        workspace,
    )
    assert untrusted.result_file == (run_dir / "result.jsonl").resolve()
    assert untrusted.log_file == (run_dir / "benchmark.log").resolve()


def test_revises_saved_history_and_atomically_updates_the_table_row(
    tmp_path: Path,
) -> None:
    result = {
        "completed": 10,
        "accept_length": 2.5,
        "request_throughput": 1.0,
        "input_throughput": 2.0,
        "output_throughput": 3.0,
        "total_throughput": 5.0,
        "mean_e2e_latency_ms": 6.0,
        "mean_ttft_ms": 7.0,
        "mean_tpot_ms": 8.0,
        "mean_itl_ms": 9.0,
    }
    record = build_saved_benchmark_result(
        result,
        run_id="speed_bench_revise_unit",
        model_name="old-model",
        configuration="TP=4 | old backend",
        configuration_options={"tp_size": 4, "additional": "old backend"},
        result_file=tmp_path / "workspace" / "speed_bench_revise_unit" / "result.jsonl",
        log_file=tmp_path / "workspace" / "speed_bench_revise_unit" / "benchmark.log",
        selected_keys=["client.mean_tpot_ms"],
    )
    saved_root = tmp_path / "shared"
    save_benchmark_result(record, saved_root)
    loaded, warnings = list_saved_benchmark_results(saved_root)
    assert warnings == []
    assert loaded[0]["revision"] == 1
    assert job_metadata_from_saved_result(loaded[0]).additional == "old backend"

    refreshed_result = {**result, "mean_tpot_ms": 4.0, "output_throughput": 6.0}
    metadata = build_job_metadata(
        model_name="new-model",
        tp_size=8,
        dp_size=2,
        additional="new backend",
    )
    revised = revise_saved_benchmark_result(
        loaded[0],
        metadata=metadata,
        selected_keys=["client.mean_tpot_ms", "client.output_throughput"],
        result=refreshed_result,
    )
    replace_saved_benchmark_result(
        revised,
        saved_root,
        expected_revision=loaded[0]["revision"],
    )

    updated, warnings = list_saved_benchmark_results(saved_root)
    assert warnings == []
    assert updated[0]["revision"] == 2
    assert updated[0]["saved_at"] == loaded[0]["saved_at"]
    assert updated[0]["updated_at"] != loaded[0]["updated_at"]
    assert updated[0]["result_name"] == ("new-model + TP=8 | DP=2 | new backend")
    assert updated[0]["metrics"]["mean_tpot_ms"] == pytest.approx(4.0)
    assert updated[0]["metrics"]["decode_speed_toks"] == pytest.approx(250.0)
    assert updated[0]["selected_keys"] == [
        "client.mean_tpot_ms",
        "client.output_throughput",
    ]

    with pytest.raises(ResultRecordConflictError):
        replace_saved_benchmark_result(
            revised,
            saved_root,
            expected_revision=1,
        )


def test_saved_history_skips_malformed_files(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "broken.json").write_text("{not-json", encoding="utf-8")

    records, warnings = list_saved_benchmark_results(shared)

    assert records == []
    assert len(warnings) == 1
    assert "broken.json" in warnings[0]


def test_delete_saved_result_preserves_original_result_and_log(tmp_path: Path) -> None:
    run_dir = tmp_path / "workspace" / "speed_bench_delete_unit"
    run_dir.mkdir(parents=True)
    result_file = run_dir / "result.jsonl"
    log_file = run_dir / "benchmark.log"
    result_file.write_text('{"completed": 1}\n', encoding="utf-8")
    log_file.write_text("benchmark completed\n", encoding="utf-8")
    record = build_saved_benchmark_result(
        {
            "completed": 1,
            "accept_length": 1.5,
            "request_throughput": 2.0,
            "input_throughput": 3.0,
            "output_throughput": 4.0,
            "total_throughput": 7.0,
            "mean_e2e_latency_ms": 8.0,
            "mean_ttft_ms": 9.0,
            "mean_tpot_ms": 10.0,
            "mean_itl_ms": 11.0,
        },
        run_id="speed_bench_delete_unit",
        model_name="model",
        configuration="config",
        result_file=result_file,
        log_file=log_file,
    )
    saved_root = tmp_path / "shared"
    saved_path = save_benchmark_result(record, saved_root)

    deleted_path = delete_saved_benchmark_result(
        "speed_bench_delete_unit",
        saved_root,
    )
    records, warnings = list_saved_benchmark_results(saved_root)

    assert deleted_path == saved_path
    assert not saved_path.exists()
    assert result_file.read_text(encoding="utf-8") == '{"completed": 1}\n'
    assert log_file.read_text(encoding="utf-8") == "benchmark completed\n"
    assert records == []
    assert warnings == []


def test_delete_waits_for_an_inflight_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "completed": 1,
        "accept_length": 1.5,
        "request_throughput": 2.0,
        "input_throughput": 3.0,
        "output_throughput": 4.0,
        "total_throughput": 7.0,
        "mean_e2e_latency_ms": 8.0,
        "mean_ttft_ms": 9.0,
        "mean_tpot_ms": 10.0,
        "mean_itl_ms": 11.0,
    }
    run_id = "speed_bench_lock_unit"
    saved_root = tmp_path / "shared"
    record = build_saved_benchmark_result(
        result,
        run_id=run_id,
        model_name="model",
        configuration="config",
        result_file=tmp_path / "run" / "result.jsonl",
        log_file=tmp_path / "run" / "benchmark.log",
    )
    saved_path = save_benchmark_result(record, saved_root)
    revised = revise_saved_benchmark_result(
        record,
        metadata=build_job_metadata(model_name="updated-model"),
    )

    replace_entered = threading.Event()
    allow_replace = threading.Event()
    original_replace = results_module.os.replace

    def blocking_replace(source, target):
        replace_entered.set()
        assert allow_replace.wait(timeout=2)
        original_replace(source, target)

    monkeypatch.setattr(results_module.os, "replace", blocking_replace)
    errors: list[BaseException] = []

    def update_record() -> None:
        try:
            replace_saved_benchmark_result(revised, saved_root, expected_revision=1)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def delete_record() -> None:
        try:
            delete_saved_benchmark_result(run_id, saved_root)
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    update_thread = threading.Thread(target=update_record)
    delete_thread = threading.Thread(target=delete_record)
    update_thread.start()
    assert replace_entered.wait(timeout=2)
    delete_thread.start()
    time.sleep(0.05)
    assert delete_thread.is_alive()

    allow_replace.set()
    update_thread.join(timeout=2)
    delete_thread.join(timeout=2)

    assert errors == []
    assert not update_thread.is_alive()
    assert not delete_thread.is_alive()
    assert not saved_path.exists()


def test_relative_change_percent() -> None:
    assert relative_change_percent(133.87, 91.50) == pytest.approx(46.306, rel=1e-3)
    assert relative_change_percent(34961.04, 55670.45) == pytest.approx(
        -37.199,
        rel=1e-3,
    )
    assert relative_change_percent(1.0, 0.0) is None
    assert relative_change_percent(None, 1.0) is None


def test_runner_tracks_pid_log_and_result(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(settings, parameters, "speed_bench_success")
    command = f"{command} \\\n  --fake-sleep 0.05"
    runner = BenchmarkRunner()

    started = runner.start(command, settings)
    assert started.pid > 0
    assert started.pgid == started.pid
    assert started.status == "Running"
    assert started.command_context["max_concurrency"] == "1"

    finished = _wait_for_completion(runner)
    assert finished is not None
    assert finished.status == STATUS_SUCCEEDED
    assert finished.return_code == 0
    assert runner.result()["completed"] == 2
    assert "fake benchmark completed" in runner.log_tail()


def test_large_result_read_does_not_hold_the_runner_global_lock(
    pipeline_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(settings, parameters, "speed_bench_read_lock")
    runner = BenchmarkRunner()
    started = runner.start(command, settings)
    assert _wait_for_completion(runner, started.run_id).status == STATUS_SUCCEEDED

    read_entered = threading.Event()
    allow_read = threading.Event()

    def blocking_parse(_output_file):
        read_entered.set()
        assert allow_read.wait(timeout=2)
        return {"completed": 2}

    monkeypatch.setattr(runner_module, "parse_result_file", blocking_parse)
    result_holder = []
    read_thread = threading.Thread(
        target=lambda: result_holder.append(runner.result(started.run_id))
    )
    read_thread.start()
    assert read_entered.wait(timeout=2)

    # A different Streamlit session can still poll the job while the result
    # file is being parsed outside the Runner's global lock.
    assert runner.snapshot(started.run_id) is not None
    assert read_thread.is_alive()

    allow_read.set()
    read_thread.join(timeout=2)
    assert not read_thread.is_alive()
    assert result_holder == [{"completed": 2}]


def test_stop_wait_does_not_hold_the_runner_global_lock(
    pipeline_setup,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(settings, parameters, "speed_bench_stop_lock")
    command = f"{command} \\\n  --fake-sleep 30"
    runner = BenchmarkRunner()
    started = runner.start(command, settings)
    process = runner._runs[started.run_id].process
    original_wait = process.wait
    wait_entered = threading.Event()
    allow_wait = threading.Event()

    def blocking_wait(timeout=None):
        wait_entered.set()
        assert allow_wait.wait(timeout=2)
        return original_wait(timeout=timeout)

    monkeypatch.setattr(process, "wait", blocking_wait)
    stopped_holder = []
    stop_thread = threading.Thread(
        target=lambda: stopped_holder.append(
            runner.stop(started.run_id, timeout_seconds=0.2)
        )
    )
    stop_thread.start()
    assert wait_entered.wait(timeout=2)

    assert runner.snapshot(started.run_id) is not None
    assert stop_thread.is_alive()

    allow_wait.set()
    stop_thread.join(timeout=2)
    assert not stop_thread.is_alive()
    assert stopped_holder[0].status == STATUS_STOPPED


def test_runner_stops_the_tracked_process_group(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(settings, parameters, "speed_bench_stop")
    command = f"{command} \\\n  --fake-sleep 30"
    runner = BenchmarkRunner()

    started = runner.start(command, settings)
    stopped = runner.stop(timeout_seconds=0.2)

    assert stopped.pid == started.pid
    assert stopped.pgid == started.pgid
    assert stopped.status == STATUS_STOPPED
    assert stopped.return_code is not None


def test_runner_manages_parallel_jobs_by_run_id(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    second_parameters = BenchmarkParameters(
        **{**parameters.__dict__, "host": "model-node-2", "port": 30001}
    )
    first_command, _ = generate_command(settings, parameters, "speed_bench_parallel_a")
    second_command, _ = generate_command(
        settings,
        second_parameters,
        "speed_bench_parallel_b",
    )
    first_command = f"{first_command} \\\n  --fake-sleep 0.2"
    second_command = f"{second_command} \\\n  --fake-sleep 0.2"
    runner = BenchmarkRunner()

    first = runner.start(
        first_command,
        settings,
        metadata=build_job_metadata(model_name="model-a", tp_size=8),
    )
    second = runner.start(
        second_command,
        settings,
        metadata=build_job_metadata(model_name="model-b", dp_size=2),
    )

    snapshots = runner.snapshots()
    assert [snapshot.run_id for snapshot in snapshots] == [
        "speed_bench_parallel_b",
        "speed_bench_parallel_a",
    ]
    assert {snapshot.status for snapshot in snapshots} == {"Running"}
    assert first.pid != second.pid
    assert runner.snapshot(first.run_id).metadata.result_name == "model-a + TP=8"
    assert runner.snapshot(second.run_id).host == "model-node-2"

    assert _wait_for_completion(runner, first.run_id).status == STATUS_SUCCEEDED
    assert _wait_for_completion(runner, second.run_id).status == STATUS_SUCCEEDED
    assert runner.result(first.run_id)["completed"] == 2
    assert "fake benchmark completed" in runner.log_tail(second.run_id)


def test_runner_rejects_parallel_jobs_for_same_endpoint(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    first_command, _ = generate_command(settings, parameters, "speed_bench_endpoint_a")
    second_command, _ = generate_command(settings, parameters, "speed_bench_endpoint_b")
    first_command = f"{first_command} \\\n  --fake-sleep 30"
    runner = BenchmarkRunner()

    started = runner.start(first_command, settings)
    try:
        with pytest.raises(ProcessStateError, match="already used"):
            runner.start(second_command, settings)
    finally:
        runner.stop(started.run_id, timeout_seconds=0.2)


def test_runner_updates_metadata_for_one_job(pipeline_setup) -> None:
    settings, parameters = pipeline_setup
    command, _ = generate_command(settings, parameters, "speed_bench_metadata")
    runner = BenchmarkRunner()
    started = runner.start(command, settings)

    updated = runner.update_metadata(
        started.run_id,
        build_job_metadata(
            model_name="DeepSeek-V4",
            tp_size=8,
            dp_size=2,
            additional="W4A8",
        ),
    )

    assert updated.metadata.result_name == "DeepSeek-V4 + TP=8 | DP=2 | W4A8"
    assert _wait_for_completion(runner, started.run_id).status == STATUS_SUCCEEDED


def test_log_tail_limits_large_logs(tmp_path: Path) -> None:
    log_file = tmp_path / "benchmark.log"
    log_file.write_text("old\n" + ("x" * 100) + "\nlatest\n", encoding="utf-8")

    tail = read_log_tail(log_file, max_bytes=32)

    assert "latest" in tail
    assert "log truncated" in tail
