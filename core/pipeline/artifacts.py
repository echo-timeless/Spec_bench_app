"""Locate raw artifacts and cache lightweight result views."""

from __future__ import annotations

import fcntl
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

from .comparison import (
    benchmark_result_metrics,
    comparison_values,
)
from .models import (
    BenchmarkResultView,
    PipelineError,
    RUN_ID_PATTERN,
    SavedRunArtifacts,
)
from ..step_curve import (
    accept_curves_from_dict,
    accept_curves_to_dict,
    accept_length_curves,
)

_RESULT_VIEW_SCHEMA_VERSION = 2

_RESULT_VIEW_FILENAME = "result_view.json"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_result_file(output_file: str | Path) -> dict[str, Any] | None:
    """Parse the last non-empty JSON object from SGLang's JSONL output."""

    path = Path(output_file)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    last_line = ""
    with path.open(encoding="utf-8") as result_file:
        for line in result_file:
            if line.strip():
                last_line = line
    if not last_line:
        return None
    parsed = json.loads(last_line)
    if not isinstance(parsed, dict):
        raise PipelineError(f"Unexpected result format in {path}")
    return parsed


def saved_run_artifacts(
    record: Mapping[str, Any],
    workspace_root: str | Path,
) -> SavedRunArtifacts:
    """Resolve a saved record's artifacts without allowing arbitrary reads.

    Saved result files may live on shared storage and therefore contain source
    paths from another machine.  A path is only trusted when it resolves to the
    configured ``workspace_root/<run-id>/`` directory.  Otherwise the local
    canonical path is used; the page can then report that the artifact is not
    available on this machine.
    """

    run_id = str(record.get("run_id", ""))
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise PipelineError(f"Invalid Run ID in saved result: {run_id!r}")

    root = Path(workspace_root).expanduser().resolve()
    run_dir = (root / run_id).resolve()
    source = record.get("source")
    source = source if isinstance(source, Mapping) else {}

    def resolve_source(key: str, default_name: str) -> Path:
        fallback = run_dir / default_name
        raw_path = source.get(key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            return fallback
        candidate = Path(raw_path).expanduser().resolve()
        if candidate.parent == run_dir and _is_within(candidate, root):
            return candidate
        return fallback

    return SavedRunArtifacts(
        result_file=resolve_source("result_file", "result.jsonl"),
        log_file=resolve_source("log_file", "benchmark.log"),
    )


@contextmanager
def _exclusive_file_lock(lock_path: Path):
    """Coordinate one filesystem mutation across BenchAPP processes."""

    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def result_view_path(output_file: str | Path) -> Path:
    """Return the lightweight UI sidecar beside one raw benchmark result."""

    return Path(output_file).expanduser().resolve().parent / _RESULT_VIEW_FILENAME


def _result_view_payload(view: BenchmarkResultView) -> dict[str, Any]:
    return {
        "schema_version": _RESULT_VIEW_SCHEMA_VERSION,
        "source": {
            "size": view.source_size,
            "mtime_ns": view.source_mtime_ns,
        },
        "metrics": dict(view.metrics),
        "values": dict(view.values),
        "curves": accept_curves_to_dict(view.curves),
    }


def _result_view_from_payload(
    raw: Any,
    *,
    source_size: int,
    source_mtime_ns: int,
) -> BenchmarkResultView:
    if not isinstance(raw, Mapping):
        raise ValueError("result view must be a JSON object")
    if raw.get("schema_version") != _RESULT_VIEW_SCHEMA_VERSION:
        raise ValueError("unsupported result view schema")
    source = raw.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("result view source must be an object")
    if source.get("size") != source_size or source.get("mtime_ns") != source_mtime_ns:
        raise ValueError("result view no longer matches the raw result")
    metrics = raw.get("metrics")
    values = raw.get("values")
    if not isinstance(metrics, Mapping) or not isinstance(values, Mapping):
        raise ValueError("result view metrics and values must be objects")
    return BenchmarkResultView(
        metrics=dict(metrics),
        values=dict(values),
        curves=accept_curves_from_dict(raw.get("curves")),
        source_size=source_size,
        source_mtime_ns=source_mtime_ns,
    )


def load_or_create_result_view(
    output_file: str | Path,
    *,
    force: bool = False,
) -> BenchmarkResultView | None:
    """Load a valid sidecar or parse the raw result once and replace it."""

    output = Path(output_file).expanduser().resolve()
    if not output.is_file() or output.stat().st_size == 0:
        return None
    sidecar = result_view_path(output)
    lock_path = sidecar.with_suffix(f"{sidecar.suffix}.lock")
    with _exclusive_file_lock(lock_path):
        source_stat = output.stat()
        if not force and sidecar.is_file():
            try:
                with sidecar.open(encoding="utf-8") as view_stream:
                    return _result_view_from_payload(
                        json.load(view_stream),
                        source_size=source_stat.st_size,
                        source_mtime_ns=source_stat.st_mtime_ns,
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # A stale or partial sidecar is only a cache miss. The raw result
                # remains the source of truth and is reparsed below.
                pass

        # A finished benchmark is stable. The retry only protects callers that
        # inspect a file while another process is appending its final JSONL row.
        result = None
        for _attempt in range(2):
            source_stat = output.stat()
            result = parse_result_file(output)
            parsed_stat = output.stat()
            if (
                source_stat.st_size == parsed_stat.st_size
                and source_stat.st_mtime_ns == parsed_stat.st_mtime_ns
            ):
                source_stat = parsed_stat
                break
        else:
            raise PipelineError("Raw result changed while its view was being built")
        if result is None:
            return None
        view = BenchmarkResultView(
            metrics=benchmark_result_metrics(result),
            values=comparison_values(result),
            curves=accept_length_curves(result),
            source_size=source_stat.st_size,
            source_mtime_ns=source_stat.st_mtime_ns,
        )
        temp_path = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temp_path.open("x", encoding="utf-8") as view_stream:
                json.dump(
                    _result_view_payload(view),
                    view_stream,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                view_stream.write("\n")
                view_stream.flush()
                os.fsync(view_stream.fileno())
            os.replace(temp_path, sidecar)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        return view


def read_log_tail(log_file: str | Path, max_bytes: int = 64 * 1024) -> str:
    """Read only the tail of a potentially large benchmark log."""

    path = Path(log_file)
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        data = stream.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        first_newline = text.find("\n")
        if first_newline >= 0:
            text = text[first_newline + 1 :]
        return f"... log truncated to last {max_bytes // 1024} KiB ...\n{text}"
    return text
