"""Generate, parse, and validate the final editable benchmark command."""

from __future__ import annotations

import json
import math
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .models import (
    BenchmarkParameters,
    CommandValidationError,
    PipelineError,
    PipelineSettings,
    RUN_ID_PATTERN,
    ValidatedCommand,
)
from .datasets import (
    PERFORMANCE_FAMILY,
    SPEED_BENCH_SIZES,
    TASK_EVALUATION_FAMILY,
    dataset_context,
    resolve_dataset_spec,
)

DATASET_SIZES = (*SPEED_BENCH_SIZES, "qualitative")

THROUGHPUT_CATEGORIES = ("", "low_entropy", "mixed", "high_entropy")

_FIXED_MODULE_ARGS = ("-m", "sglang.benchmark.serving")

_SHELL_MARKERS = ("&&", "||", ";", "`", "$(", ">", "<")

_COMMAND_CONTEXT_OPTIONS = {
    "host": "--host",
    "port": "--port",
    "dataset_path": "--dataset-path",
    "category": "--speed-bench-category",
    "num_prompts": "--num-prompts",
    "output_len": "--speed-bench-output-len",
    "max_concurrency": "--max-concurrency",
    "request_rate": "--request-rate",
}


def new_run_id() -> str:
    """Return a filesystem-safe, practically unique run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"speed_bench_{timestamp}_{uuid.uuid4().hex[:8]}"


def dataset_path(settings: PipelineSettings, dataset_size: str) -> Path:
    """Resolve a configured SPEED-Bench dataset size to its JSONL file."""

    if dataset_size not in DATASET_SIZES:
        raise PipelineError(
            f"Unsupported dataset size {dataset_size!r}; choose from {DATASET_SIZES}"
        )
    if dataset_size == "qualitative":
        return settings.dataset_root / "qualitative" / "test.jsonl"
    return settings.dataset_root / f"throughput_{dataset_size}" / "test.jsonl"


def _selected_dataset(
    settings: PipelineSettings,
    parameters: BenchmarkParameters,
):
    """Resolve new dataset fields while retaining the v1 SPEED-Bench fields."""

    dataset_id = parameters.dataset_id or "speed-bench"
    variant = parameters.dataset_variant
    if dataset_id == "speed-bench" and not variant:
        variant = parameters.dataset_size
    if dataset_id == "speed-bench" and variant == "qualitative":
        dataset_id = "speed-bench-qualitative"
        variant = ""
    if parameters.benchmark_family not in (
        PERFORMANCE_FAMILY,
        TASK_EVALUATION_FAMILY,
    ):
        raise PipelineError(
            f"Unsupported benchmark family {parameters.benchmark_family!r}"
        )
    return resolve_dataset_spec(
        settings,
        dataset_id=dataset_id,
        variant=variant,
    )


def _validate_parameters(
    settings: PipelineSettings, parameters: BenchmarkParameters
) -> None:
    errors: list[str] = []
    if not parameters.host.strip():
        errors.append("Host cannot be empty")
    if not 1 <= parameters.port <= 65535:
        errors.append("Port must be between 1 and 65535")
    try:
        selected_spec, resolved_dataset = _selected_dataset(settings, parameters)
    except PipelineError as exc:
        errors.append(str(exc))
        selected_spec = None
        resolved_dataset = None
    if selected_spec is not None:
        if selected_spec.dataset_id == "speed-bench":
            if parameters.category not in THROUGHPUT_CATEGORIES:
                errors.append(f"Category must be one of {THROUGHPUT_CATEGORIES}")
    for label, value in (
        ("Num prompts", parameters.num_prompts),
        ("Output length", parameters.output_len),
        ("Max concurrency", parameters.max_concurrency),
    ):
        if value <= 0:
            errors.append(f"{label} must be greater than zero")
    if parameters.warmup_requests < 0:
        errors.append("Warmup requests cannot be negative")
    try:
        request_rate = float(parameters.request_rate)
        if math.isnan(request_rate) or request_rate <= 0:
            errors.append("Request rate must be positive or inf")
    except ValueError:
        errors.append("Request rate must be a positive number or inf")
    if parameters.extra_request_body:
        try:
            extra_body = json.loads(parameters.extra_request_body)
            if not isinstance(extra_body, dict):
                errors.append("Extra request body must be a JSON object")
        except json.JSONDecodeError as exc:
            errors.append(f"Extra request body is invalid JSON: {exc.msg}")
    if resolved_dataset is not None and not resolved_dataset.is_file():
        errors.append(f"Dataset not found: {resolved_dataset}")
    if errors:
        raise CommandValidationError(errors)


def build_command_argv(
    settings: PipelineSettings,
    parameters: BenchmarkParameters,
    run_id: str,
) -> list[str]:
    """Build the exact argv represented in the editable command."""

    _validate_parameters(settings, parameters)
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"Invalid run ID: {run_id!r}")

    selected_spec, resolved_dataset = _selected_dataset(settings, parameters)
    output_file = settings.workspace_root / run_id / "result.jsonl"
    argv = [
        str(settings.python_executable),
        *_FIXED_MODULE_ARGS,
        "--backend",
        "sglang-oai-chat",
        "--host",
        parameters.host.strip(),
        "--port",
        str(parameters.port),
        "--ready-check-timeout-sec",
        str(settings.ready_timeout),
    ]
    argv.extend(
        [
            "--dataset-name",
            selected_spec.loader_name,
            "--dataset-path",
            str(resolved_dataset),
        ]
    )
    if selected_spec.source_kind == "speed_bench" and parameters.category:
        argv.extend(["--speed-bench-category", parameters.category])
    if selected_spec.loader_name == "speed-bench":
        output_length_option = "--speed-bench-output-len"
    else:
        output_length_option = "--sharegpt-output-len"
    argv.extend(
        [
            "--num-prompts",
            str(parameters.num_prompts),
            output_length_option,
            str(parameters.output_len),
            "--max-concurrency",
            str(parameters.max_concurrency),
            "--request-rate",
            parameters.request_rate.strip(),
            "--seed",
            str(parameters.seed),
            "--warmup-requests",
            str(parameters.warmup_requests),
        ]
    )
    if parameters.extra_request_body:
        compact_body = json.dumps(
            json.loads(parameters.extra_request_body),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        argv.extend(["--extra-request-body", compact_body])
    if parameters.output_details:
        # Valueless flag; SGLang merges the per-request arrays into the same
        # `result.jsonl` object, so no extra output path is involved.
        argv.append("--output-details")
    argv.extend(["--output-file", str(output_file)])
    return argv


def format_command(argv: Sequence[str]) -> str:
    """Format argv as an editable multi-line shell-style command."""

    if len(argv) < 3:
        return shlex.join(argv)
    segments = [shlex.join(argv[:3])]
    index = 3
    while index < len(argv):
        token = argv[index]
        if token.startswith("--") and index + 1 < len(argv):
            next_token = argv[index + 1]
            if not next_token.startswith("--"):
                segments.append(shlex.join((token, next_token)))
                index += 2
                continue
        segments.append(shlex.quote(token))
        index += 1
    return " \\\n  ".join(segments)


def generate_command(
    settings: PipelineSettings,
    parameters: BenchmarkParameters,
    run_id: str | None = None,
) -> tuple[str, str]:
    """Generate an editable command and return it with its Run ID."""

    actual_run_id = run_id or new_run_id()
    return (
        format_command(build_command_argv(settings, parameters, actual_run_id)),
        actual_run_id,
    )


def parse_command(command: str) -> tuple[str, ...]:
    """Parse a command editor value without invoking a shell."""

    if not command.strip():
        raise CommandValidationError(["Benchmark command cannot be empty"])
    if any(marker in command for marker in _SHELL_MARKERS):
        raise CommandValidationError(
            ["Shell pipes, redirects, chaining, and substitutions are not allowed"]
        )
    normalized = command.replace("\\\r\n", " ").replace("\\\n", " ")
    try:
        argv = tuple(shlex.split(normalized, posix=True))
    except ValueError as exc:
        raise CommandValidationError([f"Cannot parse command: {exc}"]) from exc
    if not argv:
        raise CommandValidationError(["Benchmark command cannot be empty"])
    return argv


def _option_values(argv: Sequence[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    index = 0
    prefix = f"{option}="
    while index < len(argv):
        token = argv[index]
        if token == option:
            value = argv[index + 1] if index + 1 < len(argv) else None
            values.append(value)
            index += 2
            continue
        if token.startswith(prefix):
            values.append(token[len(prefix) :])
        index += 1
    return values


def command_context(command: str | Sequence[str]) -> dict[str, str]:
    """Extract saved workload context from the final executable argv."""

    argv = parse_command(command) if isinstance(command, str) else tuple(command)
    context: dict[str, str] = {}
    for name, option in _COMMAND_CONTEXT_OPTIONS.items():
        values = _option_values(argv, option)
        if values and values[-1] is not None:
            context[name] = str(values[-1])
    output_len = _option_values(argv, "--speed-bench-output-len")
    output_len.extend(_option_values(argv, "--sharegpt-output-len"))
    if output_len and output_len[-1] is not None:
        context["output_len"] = str(output_len[-1])
    dataset_name_values = _option_values(argv, "--dataset-name")
    if dataset_name_values in (["openai"], ["speed-bench"]):
        context["dataset_name"] = dataset_name_values[0]
        if context.get("dataset_path"):
            context.update(
                dataset_context(context["dataset_name"], context["dataset_path"])
            )
    return context


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_command(
    command: str,
    settings: PipelineSettings,
    *,
    require_new_output: bool = True,
) -> ValidatedCommand:
    """Validate the final edited command and return managed paths."""

    argv = parse_command(command)
    errors: list[str] = []

    if len(argv) < 3:
        errors.append("Command is missing the Python module entry point")
    else:
        if not Path(argv[0]).expanduser().is_absolute():
            errors.append("Python executable must use an absolute path")
        executable = Path(argv[0]).expanduser().resolve()
        if executable != settings.python_executable:
            errors.append(f"Python executable must remain {settings.python_executable}")
        if tuple(argv[1:3]) != _FIXED_MODULE_ARGS:
            errors.append(
                "Command entry point must remain: -m sglang.benchmark.serving"
            )

    output_values = _option_values(argv, "--output-file")
    if len(output_values) != 1 or not output_values[0]:
        errors.append("Command must contain exactly one --output-file value")
        output_file = settings.workspace_root / "_invalid" / "result.jsonl"
    else:
        raw_output_file = Path(str(output_values[0])).expanduser()
        if not raw_output_file.is_absolute():
            errors.append("--output-file must use an absolute path")
        output_file = raw_output_file.resolve()

    workspace_root = settings.workspace_root.resolve()
    run_dir = output_file.parent
    if not _is_within(output_file, workspace_root):
        errors.append(f"--output-file must be inside {workspace_root}")
    elif run_dir == workspace_root:
        errors.append("--output-file must use a dedicated run subdirectory")
    if require_new_output and output_file.exists() and output_file.stat().st_size > 0:
        errors.append(
            f"Output already contains a result; generate a new command: {output_file}"
        )

    dataset_values = _option_values(argv, "--dataset-path")
    if len(dataset_values) != 1 or not dataset_values[0]:
        errors.append("Command must contain exactly one --dataset-path value")
    else:
        dataset_file = Path(str(dataset_values[0])).expanduser()
        if not dataset_file.is_absolute():
            errors.append("--dataset-path must use an absolute path")
        resolved_dataset_file = dataset_file.resolve()
        allowed_dataset_roots = (
            settings.dataset_root.resolve(),
            settings.builtin_dataset_root.resolve(),
            (settings.workspace_root / "dataset_cache").resolve(),
        )
        if not any(
            _is_within(resolved_dataset_file, root)
            for root in allowed_dataset_roots
        ):
            errors.append("--dataset-path must be inside a configured dataset root")
        if not resolved_dataset_file.is_file():
            errors.append(f"Dataset not found: {dataset_values[0]}")

    backend_values = _option_values(argv, "--backend")
    if backend_values != ["sglang-oai-chat"]:
        errors.append("--backend must remain sglang-oai-chat")
    dataset_name_values = _option_values(argv, "--dataset-name")
    if len(dataset_name_values) != 1 or dataset_name_values[0] not in {
        "speed-bench",
        "openai",
    }:
        errors.append("--dataset-name must be speed-bench or openai")

    host_values = _option_values(argv, "--host")
    if len(host_values) != 1 or not host_values[0] or not str(host_values[0]).strip():
        errors.append("Command must contain exactly one non-empty --host value")
        host = ""
    else:
        host = str(host_values[0]).strip()

    port_values = _option_values(argv, "--port")
    if len(port_values) != 1 or not port_values[0]:
        errors.append("Command must contain exactly one --port value")
        port = 0
    else:
        try:
            port = int(str(port_values[0]))
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            errors.append("--port must be an integer between 1 and 65535")
            port = 0
    if not settings.python_executable.is_file():
        errors.append(f"Python executable not found: {settings.python_executable}")
    if not (settings.sglang_repo / "python" / "sglang").is_dir():
        errors.append(f"Invalid SGLang repository: {settings.sglang_repo}")

    if errors:
        raise CommandValidationError(errors)

    return ValidatedCommand(
        argv=argv,
        output_file=output_file,
        run_dir=run_dir,
        log_file=run_dir / "benchmark.log",
        run_id=run_dir.name,
        host=host,
        port=port,
        context=command_context(argv),
    )
