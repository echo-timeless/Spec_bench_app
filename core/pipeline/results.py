"""Build, validate, and atomically mutate saved comparison records."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import _exclusive_file_lock
from .comparison import (
    _SAVED_METRIC_KEYS,
    _finite_float,
    benchmark_result_metrics,
    comparison_values,
)
from .models import (
    BenchmarkJobMetadata,
    BenchmarkResultView,
    PipelineError,
    ResultRecordConflictError,
    ResultRecordExistsError,
    ResultRecordNotFoundError,
    RUN_ID_PATTERN,
    _configuration_size,
    build_result_name,
)
from ..result_keys import CLIENT_PREFIX, CONFIGURATION_ONLY_KEYS

_SAVED_RESULT_SCHEMA_VERSION = 2

_SUPPORTED_SAVED_RESULT_SCHEMA_VERSIONS = (1, 2)


def build_saved_benchmark_result(
    result: Mapping[str, Any] | BenchmarkResultView,
    *,
    run_id: str,
    model_name: str,
    configuration: str,
    configuration_options: Mapping[str, Any] | None = None,
    result_file: str | Path,
    log_file: str | Path,
    context: Mapping[str, Any] | None = None,
    saved_at: datetime | None = None,
    selected_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the versioned record persisted for comparison.

    Every comparable key is stored so later metadata/display-column updates do
    not discard measurements. ``selected_keys`` only carries which columns the
    table shows by default. A lightweight result view avoids retaining the raw
    result in Streamlit Session State.
    """

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"Invalid run ID for saved result: {run_id!r}")
    normalized_model = model_name.strip()
    if not normalized_model:
        raise PipelineError("Model name cannot be empty")
    normalized_configuration = configuration.strip()
    raw_options = configuration_options or {}
    normalized_options = {
        key: value
        for key, label in (
            ("tp_size", "TP Size"),
            ("dp_size", "DP Size"),
            ("ep_size", "EP Size"),
        )
        if (value := _configuration_size(raw_options.get(key), label=label)) is not None
    }
    raw_additional = raw_options.get("additional", "")
    if raw_additional is not None and not isinstance(raw_additional, str):
        raise PipelineError("Additional configuration must be text")
    if normalized_additional := str(raw_additional or "").strip():
        normalized_options["additional"] = normalized_additional

    timestamp = saved_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    if isinstance(result, BenchmarkResultView):
        metrics = dict(result.metrics)
        values = dict(result.values)
    else:
        metrics = benchmark_result_metrics(result)
        values = comparison_values(result)
    normalized_selection = tuple(
        dict.fromkeys(key for key in (selected_keys or ()) if key in values)
    )

    serialized_saved_at = timestamp.astimezone(timezone.utc).isoformat()
    return {
        "schema_version": _SAVED_RESULT_SCHEMA_VERSION,
        "revision": 1,
        "record_id": run_id,
        "run_id": run_id,
        "saved_at": serialized_saved_at,
        "updated_at": serialized_saved_at,
        "model_name": normalized_model,
        "configuration": normalized_configuration,
        "configuration_options": normalized_options,
        "result_name": build_result_name(normalized_model, normalized_configuration),
        "metrics": metrics,
        "values": values,
        "selected_keys": list(normalized_selection),
        "source": {
            "result_file": str(Path(result_file).expanduser().resolve()),
            "log_file": str(Path(log_file).expanduser().resolve()),
        },
        "context": dict(context or {}),
    }


def revise_saved_benchmark_result(
    record: Mapping[str, Any],
    *,
    metadata: BenchmarkJobMetadata,
    selected_keys: Sequence[str] | None = None,
    result: Mapping[str, Any] | BenchmarkResultView | None = None,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the next revision of one saved comparison record.

    Metadata-only updates preserve the existing metrics and flattened values.
    Passing ``result`` reparses the original ``result.jsonl`` and refreshes the
    row's metrics/values while preserving its identity and original save time.
    """

    existing = _validate_saved_benchmark_result(
        dict(record),
        source_name="saved result",
    )
    if not metadata.model_name:
        raise PipelineError("Model name cannot be empty")

    if isinstance(result, BenchmarkResultView):
        values = dict(result.values)
        metrics = dict(result.metrics)
    elif result is not None:
        values = comparison_values(result)
        metrics = benchmark_result_metrics(result)
    else:
        values = dict(existing["values"])
        metrics = dict(existing["metrics"])
    requested_selection = (
        existing["selected_keys"] if selected_keys is None else selected_keys
    )
    normalized_selection = list(
        dict.fromkeys(key for key in requested_selection if key in values)
    )

    timestamp = updated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    configuration_options: dict[str, Any] = {
        key: value
        for key, value in (
            ("tp_size", metadata.tp_size),
            ("dp_size", metadata.dp_size),
            ("ep_size", metadata.ep_size),
        )
        if value is not None
    }
    if metadata.additional:
        configuration_options["additional"] = metadata.additional

    run_id = existing["run_id"]
    return {
        **existing,
        "schema_version": _SAVED_RESULT_SCHEMA_VERSION,
        "revision": existing["revision"] + 1,
        "record_id": run_id,
        "run_id": run_id,
        "updated_at": timestamp.astimezone(timezone.utc).isoformat(),
        "model_name": metadata.model_name,
        "configuration": metadata.configuration,
        "configuration_options": configuration_options,
        "result_name": metadata.result_name,
        "metrics": metrics,
        "values": values,
        "selected_keys": normalized_selection,
    }


def save_benchmark_result(
    record: Mapping[str, Any],
    saved_results_root: str | Path,
) -> Path:
    """Persist a run's initial JSON record using exclusive creation."""

    run_id = str(record.get("run_id", ""))
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"Invalid run ID for saved result: {run_id!r}")
    if int(record.get("schema_version", 0)) != _SAVED_RESULT_SCHEMA_VERSION:
        raise PipelineError("Unsupported saved result schema version")

    root = Path(saved_results_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{run_id}.json"
    lock_path = root / f".{run_id}.lock"
    with _exclusive_file_lock(lock_path):
        try:
            result_stream = target.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise ResultRecordExistsError(
                f"Run {run_id} has already been saved"
            ) from exc

        try:
            with result_stream:
                json.dump(
                    dict(record),
                    result_stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                result_stream.write("\n")
                result_stream.flush()
                os.fsync(result_stream.fileno())
        except Exception:
            try:
                target.unlink()
            except OSError:
                pass
            raise
    return target


def replace_saved_benchmark_result(
    record: Mapping[str, Any],
    saved_results_root: str | Path,
    *,
    expected_revision: int,
) -> Path:
    """Atomically replace one saved record with optimistic revision checking.

    A small per-Run lock file coordinates BenchAPP instances sharing the same
    directory.  The original JSON remains untouched if validation or writing
    fails.
    """

    incoming = _validate_saved_benchmark_result(
        dict(record),
        source_name="updated saved result",
    )
    run_id = incoming["run_id"]
    if incoming["revision"] != expected_revision + 1:
        raise PipelineError(
            "Updated record revision must be exactly one greater than expected"
        )

    root = Path(saved_results_root).expanduser().resolve()
    if not root.is_dir():
        raise ResultRecordNotFoundError(
            f"Saved result directory does not exist: {root}"
        )
    target = root / f"{run_id}.json"
    lock_path = root / f".{run_id}.lock"
    temp_path = root / f".{run_id}.{uuid.uuid4().hex}.tmp"

    with _exclusive_file_lock(lock_path):
        try:
            with target.open(encoding="utf-8") as current_stream:
                current_raw = json.load(current_stream)
        except FileNotFoundError as exc:
            raise ResultRecordNotFoundError(
                f"Saved result for Run {run_id} no longer exists"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PipelineError(
                f"Saved result for Run {run_id} is not valid JSON"
            ) from exc

        current = _validate_saved_benchmark_result(
            current_raw,
            source_name=target.name,
        )
        if current["revision"] != expected_revision:
            raise ResultRecordConflictError(
                f"Run {run_id} was updated by another session; refresh and retry"
            )

        try:
            with temp_path.open("x", encoding="utf-8") as result_stream:
                json.dump(
                    dict(record),
                    result_stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                result_stream.write("\n")
                result_stream.flush()
                os.fsync(result_stream.fileno())
            os.replace(temp_path, target)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
    return target


def delete_saved_benchmark_result(
    run_id: str,
    saved_results_root: str | Path,
) -> Path:
    """Delete only one saved comparison JSON, never its source artifacts."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"Invalid run ID for saved result: {run_id!r}")
    root = Path(saved_results_root).expanduser().resolve()
    if not root.is_dir():
        raise ResultRecordNotFoundError(
            f"Saved result directory does not exist: {root}"
        )
    target = root / f"{run_id}.json"
    lock_path = root / f".{run_id}.lock"
    with _exclusive_file_lock(lock_path):
        try:
            target.unlink()
        except FileNotFoundError as exc:
            raise ResultRecordNotFoundError(
                f"Saved result for Run {run_id} no longer exists"
            ) from exc
    return target


def _validate_saved_benchmark_result(
    raw: Any,
    *,
    source_name: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PipelineError(f"{source_name}: saved result must be a JSON object")
    schema_version = raw.get("schema_version")
    if schema_version not in _SUPPORTED_SAVED_RESULT_SCHEMA_VERSIONS:
        raise PipelineError(f"{source_name}: unsupported schema version")

    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"{source_name}: invalid run ID")
    if not isinstance(raw.get("model_name"), str) or not raw["model_name"].strip():
        raise PipelineError(f"{source_name}: model name is missing")
    if not isinstance(raw.get("configuration"), str):
        raise PipelineError(f"{source_name}: configuration must be text")
    if not isinstance(raw.get("saved_at"), str):
        raise PipelineError(f"{source_name}: saved_at is missing")
    revision = raw.get("revision", 1)
    if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
        raise PipelineError(f"{source_name}: revision must be a positive integer")
    updated_at = raw.get("updated_at", raw["saved_at"])
    if not isinstance(updated_at, str):
        raise PipelineError(f"{source_name}: updated_at must be text")

    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        raise PipelineError(f"{source_name}: metrics are missing")
    normalized_metrics = {
        key: _finite_float(
            metrics.get(key),
            label=f"{source_name}.{key}",
            allow_none=key == "accept_length",
        )
        for key in _SAVED_METRIC_KEYS
    }

    values = raw.get("values")
    selected_keys = raw.get("selected_keys")
    if schema_version == 1:
        # v1 predates flattening: promote its 10 metrics so both schemas share
        # one lookup path, and default the columns to exactly those metrics.
        normalized_values: dict[str, Any] = {
            f"{CLIENT_PREFIX}.{key}": value
            for key, value in normalized_metrics.items()
            if value is not None
        }
        normalized_selection = [f"{CLIENT_PREFIX}.{key}" for key in _SAVED_METRIC_KEYS]
    else:
        if not isinstance(values, dict):
            raise PipelineError(f"{source_name}: values are missing")
        normalized_values = {
            str(key): value
            for key, value in values.items()
            if isinstance(value, (bool, int, float, str))
            and str(key) not in CONFIGURATION_ONLY_KEYS
        }
        if not isinstance(selected_keys, list):
            raise PipelineError(f"{source_name}: selected_keys must be a list")
        normalized_selection = [
            key
            for key in dict.fromkeys(str(item) for item in selected_keys)
            if key in normalized_values
        ]

    raw_options = raw.get("configuration_options") or {}
    if not isinstance(raw_options, dict):
        raise PipelineError(f"{source_name}: configuration_options must be an object")
    normalized_options = {
        key: value
        for key, label in (
            ("tp_size", "TP Size"),
            ("dp_size", "DP Size"),
            ("ep_size", "EP Size"),
        )
        if (value := _configuration_size(raw_options.get(key), label=label)) is not None
    }
    raw_additional = raw_options.get("additional", "")
    if raw_additional is not None and not isinstance(raw_additional, str):
        raise PipelineError(f"{source_name}: additional configuration must be text")
    if normalized_additional := str(raw_additional or "").strip():
        normalized_options["additional"] = normalized_additional

    return {
        **raw,
        "revision": revision,
        "updated_at": updated_at,
        "result_name": build_result_name(raw["model_name"], raw["configuration"]),
        "configuration_options": normalized_options,
        "metrics": normalized_metrics,
        "values": normalized_values,
        "selected_keys": normalized_selection,
    }


def list_saved_benchmark_results(
    saved_results_root: str | Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load valid shared records and report malformed files without blocking."""

    root = Path(saved_results_root).expanduser().resolve()
    if not root.is_dir():
        return [], []

    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for result_path in sorted(root.glob("*.json")):
        try:
            with result_path.open(encoding="utf-8") as result_stream:
                raw = json.load(result_stream)
            records.append(
                _validate_saved_benchmark_result(
                    raw,
                    source_name=result_path.name,
                )
            )
        except (OSError, json.JSONDecodeError, PipelineError) as exc:
            warnings.append(f"{result_path.name}: {exc}")

    records.sort(key=lambda record: (record["saved_at"], record["run_id"]))
    return records, warnings
