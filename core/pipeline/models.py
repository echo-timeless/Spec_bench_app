"""Pipeline data types, status values, errors, and job metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..step_curve import AcceptCurves

STATUS_RUNNING = "Running"
STATUS_SUCCEEDED = "Succeeded"
STATUS_FAILED = "Failed"
STATUS_STOPPED = "Stopped"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class PipelineError(RuntimeError):
    """Base error for benchmark pipeline operations."""


class CommandValidationError(PipelineError):
    """Raised when the editable benchmark command is invalid or unsafe."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


class ProcessStateError(PipelineError):
    """Raised when a process operation is incompatible with current state."""


class ResultRecordExistsError(PipelineError):
    """Raised when a benchmark run has already been saved."""


class ResultRecordNotFoundError(PipelineError):
    """Raised when a saved benchmark record no longer exists."""


class ResultRecordConflictError(PipelineError):
    """Raised when a saved record changed since the page loaded it."""


@dataclass(frozen=True)
class PipelineSettings:
    """Static environment configuration and UI defaults."""

    python_executable: Path
    sglang_repo: Path
    dataset_root: Path
    workspace_root: Path
    saved_results_root: Path
    host: str
    port: int
    ready_timeout: int
    dataset_size: str
    category: str
    num_prompts: int
    output_len: int
    max_concurrency: int
    request_rate: str
    seed: int
    warmup_requests: int
    extra_request_body: str
    output_details: bool = False
    default_comparison_keys: tuple[str, ...] = ()
    builtin_dataset_root: Path = Path(__file__).resolve().parent.parent.parent / "datasets"


@dataclass(frozen=True)
class BenchmarkParameters:
    """Values configured in the upper-left form."""

    host: str
    port: int
    dataset_size: str
    category: str
    num_prompts: int
    output_len: int
    max_concurrency: int
    request_rate: str
    seed: int
    warmup_requests: int
    extra_request_body: str
    output_details: bool = False
    benchmark_family: str = "performance"
    dataset_id: str = "speed-bench"
    dataset_variant: str = ""


@dataclass(frozen=True)
class ValidatedCommand:
    """Parsed command plus paths managed by BenchAPP."""

    argv: tuple[str, ...]
    output_file: Path
    run_dir: Path
    log_file: Path
    run_id: str
    host: str
    port: int
    context: Mapping[str, str]


@dataclass(frozen=True)
class SavedRunArtifacts:
    """Safe local artifact paths associated with one saved result."""

    result_file: Path
    log_file: Path


@dataclass(frozen=True)
class BenchmarkResultView:
    """Small, reusable projection of a potentially very large raw result."""

    metrics: Mapping[str, float | None]
    values: Mapping[str, Any]
    curves: AcceptCurves | None
    source_size: int
    source_mtime_ns: int


@dataclass(frozen=True)
class BenchmarkJobMetadata:
    """User-owned labels attached to one benchmark job and saved result."""

    model_name: str = ""
    tp_size: int | None = None
    dp_size: int | None = None
    ep_size: int | None = None
    additional: str = ""

    @property
    def configuration(self) -> str:
        return build_configuration_description(
            tp_size=self.tp_size,
            dp_size=self.dp_size,
            ep_size=self.ep_size,
            additional=self.additional,
        )

    @property
    def result_name(self) -> str:
        if not self.model_name:
            return ""
        return build_result_name(self.model_name, self.configuration)


@dataclass(frozen=True)
class RunSnapshot:
    """Read-only benchmark job and process state exposed to the page."""

    run_id: str
    command: str
    pid: int
    pgid: int
    status: str
    started_at: datetime
    ended_at: datetime | None
    elapsed_seconds: float
    return_code: int | None
    log_file: Path
    output_file: Path
    host: str
    port: int
    metadata: BenchmarkJobMetadata
    command_context: Mapping[str, str]


def _configuration_size(value: Any, *, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise PipelineError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"{label} must be a positive integer") from exc
    if number <= 0 or number != value:
        raise PipelineError(f"{label} must be a positive integer")
    return number


def build_configuration_description(
    *,
    tp_size: Any = None,
    dp_size: Any = None,
    ep_size: Any = None,
    additional: str = "",
) -> str:
    """Compose the user-owned configuration label; no server data is read."""

    parts = []
    for label, value in (
        ("TP", _configuration_size(tp_size, label="TP Size")),
        ("DP", _configuration_size(dp_size, label="DP Size")),
        ("EP", _configuration_size(ep_size, label="EP Size")),
    ):
        if value is not None:
            parts.append(f"{label}={value}")
    if normalized_additional := additional.strip():
        parts.append(normalized_additional)
    return " | ".join(parts)


def build_result_name(model_name: str, configuration: str) -> str:
    """Build the human-readable saved-result name from user metadata."""

    model = model_name.strip()
    if not model:
        raise PipelineError("Model name cannot be empty")
    config = configuration.strip()
    return f"{model} + {config}" if config else model


def build_job_metadata(
    *,
    model_name: str = "",
    tp_size: Any = None,
    dp_size: Any = None,
    ep_size: Any = None,
    additional: str = "",
) -> BenchmarkJobMetadata:
    """Normalize the editable labels owned by one benchmark job."""

    return BenchmarkJobMetadata(
        model_name=model_name.strip(),
        tp_size=_configuration_size(tp_size, label="TP Size"),
        dp_size=_configuration_size(dp_size, label="DP Size"),
        ep_size=_configuration_size(ep_size, label="EP Size"),
        additional=additional.strip(),
    )


def job_metadata_from_saved_result(
    record: Mapping[str, Any],
) -> BenchmarkJobMetadata:
    """Recover editable user metadata from a saved result.

    New records store ``additional`` explicitly.  Older schema-v2 records only
    have the rendered configuration string, so remove the known TP/DP/EP parts
    and preserve the remainder as the additional description.
    """

    raw_options = record.get("configuration_options")
    options = raw_options if isinstance(raw_options, Mapping) else {}
    tp_size = options.get("tp_size")
    dp_size = options.get("dp_size")
    ep_size = options.get("ep_size")
    explicit_additional = options.get("additional")
    if isinstance(explicit_additional, str):
        additional = explicit_additional
    else:
        known_parts = {
            f"{label}={value}"
            for label, value in (("TP", tp_size), ("DP", dp_size), ("EP", ep_size))
            if value is not None
        }
        configuration = record.get("configuration")
        parts = (
            [part.strip() for part in configuration.split(" | ")]
            if isinstance(configuration, str)
            else []
        )
        additional = " | ".join(
            part for part in parts if part and part not in known_parts
        )

    return build_job_metadata(
        model_name=str(record.get("model_name", "")),
        tp_size=tp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        additional=additional,
    )
