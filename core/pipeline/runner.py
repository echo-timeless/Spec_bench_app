"""Own, monitor, and stop independent benchmark subprocesses."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench_app.core.pipeline.command import validate_command
from bench_app.core.pipeline.models import (
    STATUS_FAILED,
    STATUS_RUNNING,
    STATUS_STOPPED,
    STATUS_SUCCEEDED,
    BenchmarkJobMetadata,
    BenchmarkResultView,
    PipelineSettings,
    ProcessStateError,
    RunSnapshot,
    ValidatedCommand,
)
from bench_app.core.pipeline.artifacts import (
    load_or_create_result_view,
    parse_result_file,
    read_log_tail,
)


@dataclass
class _ManagedRun:
    """Mutable process state private to one Runner instance."""

    command: str
    command_spec: ValidatedCommand
    process: subprocess.Popen
    pid: int
    pgid: int
    started_at: datetime
    metadata: BenchmarkJobMetadata
    stop_requested: bool = False
    ended_at: datetime | None = None
    final_status: str | None = None
    operation_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )

    @property
    def run_id(self) -> str:
        return self.command_spec.run_id

    @property
    def log_file(self) -> Path:
        return self.command_spec.log_file

    @property
    def output_file(self) -> Path:
        return self.command_spec.output_file

    @property
    def host(self) -> str:
        return self.command_spec.host

    @property
    def port(self) -> int:
        return self.command_spec.port


class BenchmarkRunner:
    """Own and monitor independent benchmark jobs started by BenchAPP."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, _ManagedRun] = {}
        self._run_order: list[str] = []

    def _resolve_run_locked(self, run_id: str | None) -> _ManagedRun | None:
        if run_id is not None:
            return self._runs.get(run_id)
        if not self._run_order:
            return None
        return self._runs[self._run_order[-1]]

    def _snapshot_locked(self, run_id: str | None = None) -> RunSnapshot | None:
        current = self._resolve_run_locked(run_id)
        if current is None:
            return None

        return_code = current.process.poll()
        if return_code is None:
            status = STATUS_RUNNING
        else:
            if current.ended_at is None:
                current.ended_at = datetime.now(timezone.utc)
            if current.final_status is None:
                if current.stop_requested:
                    current.final_status = STATUS_STOPPED
                elif return_code == 0:
                    current.final_status = STATUS_SUCCEEDED
                else:
                    current.final_status = STATUS_FAILED
            status = current.final_status

        end = current.ended_at or datetime.now(timezone.utc)
        elapsed = max(0.0, (end - current.started_at).total_seconds())
        return RunSnapshot(
            run_id=current.run_id,
            command=current.command,
            pid=current.pid,
            pgid=current.pgid,
            status=status,
            started_at=current.started_at,
            ended_at=current.ended_at,
            elapsed_seconds=elapsed,
            return_code=return_code,
            log_file=current.log_file,
            output_file=current.output_file,
            host=current.host,
            port=current.port,
            metadata=current.metadata,
            command_context=dict(current.command_spec.context),
        )

    def snapshot(self, run_id: str | None = None) -> RunSnapshot | None:
        """Return one job state; without an ID, return the most recently started."""

        with self._lock:
            return self._snapshot_locked(run_id)

    def snapshots(self) -> tuple[RunSnapshot, ...]:
        """Return every retained job, newest first, while polling completion."""

        with self._lock:
            return tuple(
                snapshot
                for run_id in reversed(self._run_order)
                if (snapshot := self._snapshot_locked(run_id)) is not None
            )

    def start(
        self,
        command: str,
        settings: PipelineSettings,
        *,
        metadata: BenchmarkJobMetadata | None = None,
    ) -> RunSnapshot:
        """Start the final editor command without a shell."""

        validated = validate_command(command, settings, require_new_output=True)
        with self._lock:
            if validated.run_id in self._runs:
                raise ProcessStateError(
                    f"Run {validated.run_id} is already managed; generate a new command"
                )
            endpoint = (validated.host.casefold(), validated.port)
            for run_id in self._run_order:
                snapshot = self._snapshot_locked(run_id)
                if (
                    snapshot is not None
                    and snapshot.status == STATUS_RUNNING
                    and (snapshot.host.casefold(), snapshot.port) == endpoint
                ):
                    raise ProcessStateError(
                        f"Endpoint {validated.host}:{validated.port} is already used by "
                        f"running job {snapshot.run_id} (PID {snapshot.pid})"
                    )

            validated.run_dir.mkdir(parents=True, exist_ok=False)
            environment = os.environ.copy()
            sglang_python = str(settings.sglang_repo / "python")
            existing_pythonpath = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = (
                f"{sglang_python}{os.pathsep}{existing_pythonpath}"
                if existing_pythonpath
                else sglang_python
            )
            environment["PYTHONUNBUFFERED"] = "1"

            try:
                with validated.log_file.open("w", encoding="utf-8") as log_stream:
                    process = subprocess.Popen(
                        list(validated.argv),
                        cwd=str(settings.sglang_repo),
                        env=environment,
                        stdout=log_stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        text=True,
                    )
            except Exception:
                try:
                    validated.log_file.unlink()
                except OSError:
                    pass
                try:
                    validated.run_dir.rmdir()
                except OSError:
                    pass
                raise

            started_at = datetime.now(timezone.utc)
            self._runs[validated.run_id] = _ManagedRun(
                command=command,
                command_spec=validated,
                process=process,
                pid=process.pid,
                pgid=os.getpgid(process.pid),
                started_at=started_at,
                metadata=metadata or BenchmarkJobMetadata(),
            )
            self._run_order.append(validated.run_id)
            created = self._snapshot_locked(validated.run_id)
            if created is None:  # pragma: no cover - defensive
                raise ProcessStateError("Failed to create process snapshot")
            return created

    def stop(
        self,
        run_id: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> RunSnapshot:
        """Stop the tracked process group, escalating to SIGKILL on timeout."""

        with self._lock:
            current = self._resolve_run_locked(run_id)
            if current is None:
                target = f" {run_id}" if run_id else ""
                raise ProcessStateError(f"Benchmark job{target} was not found")
            operation_lock = current.operation_lock

        with operation_lock:
            with self._lock:
                snapshot = self._snapshot_locked(current.run_id)
                if snapshot is None:
                    raise ProcessStateError(
                        f"Benchmark job {current.run_id} was not found"
                    )
                if snapshot.status != STATUS_RUNNING:
                    return snapshot
                current.stop_requested = True
                pid = current.pid
                tracked_pgid = current.pgid
                process = current.process

            try:
                actual_pgid = os.getpgid(pid)
            except ProcessLookupError:
                with self._lock:
                    return self._snapshot_locked(current.run_id) or snapshot
            if actual_pgid != tracked_pgid:
                with self._lock:
                    current.stop_requested = False
                raise ProcessStateError(
                    "Tracked process group no longer matches; refusing to kill it"
                )

            try:
                os.killpg(tracked_pgid, signal.SIGTERM)
                process.wait(timeout=timeout_seconds)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(tracked_pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=timeout_seconds)

            with self._lock:
                current.final_status = STATUS_STOPPED
                current.ended_at = datetime.now(timezone.utc)
                stopped = self._snapshot_locked(current.run_id)
                if stopped is None:  # pragma: no cover - defensive
                    raise ProcessStateError("Failed to update stopped process")
                return stopped

    def update_metadata(
        self,
        run_id: str,
        metadata: BenchmarkJobMetadata,
    ) -> RunSnapshot:
        """Replace only the user-owned labels of one retained job."""

        with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                raise ProcessStateError(f"Benchmark job {run_id} was not found")
            current.metadata = metadata
            snapshot = self._snapshot_locked(run_id)
            if snapshot is None:  # pragma: no cover - defensive
                raise ProcessStateError(f"Benchmark job {run_id} was not found")
            return snapshot

    def log_tail(
        self,
        run_id: str | None = None,
        max_bytes: int = 64 * 1024,
    ) -> str:
        """Return one retained job's recent log output."""

        with self._lock:
            current = self._resolve_run_locked(run_id)
            if current is None:
                return ""
            log_file = current.log_file
        return read_log_tail(log_file, max_bytes=max_bytes)

    def result(self, run_id: str | None = None) -> dict[str, Any] | None:
        """Return one retained job's latest JSONL result when available."""

        with self._lock:
            current = self._resolve_run_locked(run_id)
            if current is None:
                return None
            output_file = current.output_file
            operation_lock = current.operation_lock
        with operation_lock:
            return parse_result_file(output_file)

    def result_view(
        self,
        run_id: str | None = None,
        *,
        force: bool = False,
    ) -> BenchmarkResultView | None:
        """Return the lightweight cached projection for one completed result."""

        with self._lock:
            current = self._resolve_run_locked(run_id)
            if current is None:
                return None
            output_file = current.output_file
            operation_lock = current.operation_lock
        with operation_lock:
            return load_or_create_result_view(output_file, force=force)
