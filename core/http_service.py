"""Persistent evaluation lifecycle shared by the HTTP endpoints."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
import math
import os
import re
import threading
import time
import uuid

from .pipeline import (
    BenchmarkRunner, PipelineError, ProcessStateError, STATUS_RUNNING,
    STATUS_SUCCEEDED, STATUS_STOPPED, default_parameters, generate_command,
    validate_command, build_job_metadata, build_saved_benchmark_result,
    save_benchmark_result, read_log_tail, load_or_create_result_view,
)
from .step_curve import accept_curves_to_dict


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


def object_fields(value, allowed):
    if not isinstance(value, dict):
        raise PipelineError('Expected a JSON object')
    unknown = value.keys() - set(allowed)
    if unknown:
        raise PipelineError(f'Unknown fields: {sorted(unknown)}')
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', value):
        raise PipelineError('Invalid run_id')
    return value


class BenchmarkService:
    """Own benchmark processes; publish success only after automatic persistence."""

    def __init__(self, settings):
        self.settings = settings
        self.runner = BenchmarkRunner()
        self.lock = threading.RLock()
        self.records = {}
        self.deadlines = {}
        self.monitors = []
        self.closing = False

    def path(self, run_id):
        identifier(run_id)
        root = self.settings.workspace_root.resolve()
        directory = root / run_id
        if directory.resolve() != directory:
            raise PipelineError('Evaluation directory must not be a symlink')
        return directory / 'evaluation.json'

    def persist(self, record):
        path = self.path(record['run_id'])
        temporary = path.with_name(f'.evaluation.{uuid.uuid4().hex}.tmp')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                json.dump(record, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def prepare(self, body):
        object_fields(body, ('parameters', 'run_id', 'metadata', 'timeout_seconds'))
        raw_metadata = object_fields(body.get('metadata', {}),
                                     ('model_name', 'tp_size', 'dp_size', 'ep_size', 'additional'))
        for key in ('model_name', 'additional'):
            if key in raw_metadata and not isinstance(raw_metadata[key], str):
                raise PipelineError(f'metadata.{key} must be text')
        metadata = build_job_metadata(**raw_metadata)
        if not metadata.model_name:
            raise PipelineError('metadata.model_name is required')
        timeout = body.get('timeout_seconds', 1800)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise PipelineError('timeout_seconds must be a finite positive number')
        defaults = default_parameters(self.settings)
        values = object_fields(body.get('parameters', {}), asdict(defaults))
        for key, value in values.items():
            expected = type(getattr(defaults, key))
            if type(value) is not expected:
                raise PipelineError(f'parameters.{key} must be {expected.__name__}')
        # Honor the legacy dataset_size alias when no explicit variant is given.
        if 'dataset_size' in values and 'dataset_variant' not in values:
            values = dict(values, dataset_variant=values['dataset_size'])
        parameters = replace(defaults, **values)
        run_id = identifier(body['run_id']) if 'run_id' in body else None
        if run_id and (self.path(run_id).parent.exists() or
                       (self.settings.saved_results_root / f'{run_id}.json').exists()):
            raise ProcessStateError(f'Run ID already exists: {run_id}')
        command, run_id = generate_command(self.settings, parameters, run_id)
        validated = validate_command(command, self.settings)
        path = self.path(run_id)
        if validated.run_dir != path.parent:
            raise PipelineError('Output must use workspace/<run_id>/result.jsonl')
        if path.parent.exists():
            raise ProcessStateError(f'Run directory already exists: {run_id}')
        request = {'run_id': run_id, 'parameters': asdict(parameters),
                   'metadata': asdict(metadata), 'timeout_seconds': timeout}
        return command, validated, metadata, request

    def validate(self, body):
        command, validated, _, request = self.prepare(body)
        return {'valid': True, 'request': request, 'command': command,
                'argv': list(validated.argv)}

    def submit(self, body):
        with self.lock:
            if self.closing:
                raise ApiError(503, 'SHUTTING_DOWN', 'API is shutting down')
            command, _, metadata, request = self.prepare(body)
            snapshot = self.runner.start(command, self.settings, metadata=metadata)
            run_id = snapshot.run_id
            record = {
                'schema_version': 1, 'run_id': run_id, 'status': 'running',
                'started_at': snapshot.started_at.isoformat(), 'ended_at': None,
                'return_code': None, 'error': None, 'request': request,
                'command': command, 'metrics': {}, 'values': {}, 'curves': None,
                'log_tail': '', 'artifacts': {
                    'evaluation': str(self.path(run_id)),
                    'result_jsonl': str(snapshot.output_file),
                    'benchmark_log': str(snapshot.log_file),
                    'result_view': str(snapshot.output_file.parent / 'result_view.json'),
                    'saved_record': None,
                },
            }
            try:
                self.persist(record)
            except Exception:
                self.runner.stop(run_id)
                raise
            self.records[run_id] = record
            self.deadlines[run_id] = time.monotonic() + request['timeout_seconds']
            monitor = threading.Thread(target=self.monitor, args=(run_id,), daemon=True)
            self.monitors.append(monitor)
            monitor.start()
            return self.summary(record)

    def load(self, run_id):
        path = self.path(run_id)
        if run_id in self.records:
            return self.records[run_id]
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
        except FileNotFoundError as exc:
            raise ApiError(404, 'NOT_FOUND', f'Evaluation {run_id} not found') from exc
        if not isinstance(record, dict) or record.get('run_id') != run_id:
            raise ApiError(500, 'INVALID_RECORD', 'Invalid persisted evaluation')
        # A new API process cannot safely reattach to an old PID.
        if record['status'] == 'running':
            record.update(status='failed', ended_at=datetime.now(timezone.utc).isoformat(),
                          error={'code': 'INTERRUPTED', 'message':
                                 'Previous API process exited before finalization; old benchmark is not managed'})
            self.persist(record)
        self.records[run_id] = record
        return record

    def finish(self, record, snapshot, error=None):
        if record['status'] != 'running':
            return record
        result = dict(record)
        result.update(return_code=snapshot.return_code,
                      ended_at=(snapshot.ended_at or datetime.now(timezone.utc)).isoformat(),
                      status='failed', error=error)
        result['artifacts'] = dict(record['artifacts'])
        if snapshot.status == STATUS_STOPPED:
            result['status'] = 'stopped'
            result['error'] = error or {'code': 'CANCELLED', 'message': 'Benchmark was stopped'}
        elif snapshot.status != STATUS_SUCCEEDED:
            result['error'] = error or {'code': 'BENCHMARK_FAILED',
                                       'message': f'Benchmark exited with code {snapshot.return_code}'}
        else:
            try:
                view = load_or_create_result_view(snapshot.output_file)
                if view is None:
                    raise ValueError('Result file is missing or empty')
                result.update(metrics=dict(view.metrics), values=dict(view.values),
                              curves=accept_curves_to_dict(view.curves))
                # Catch non-finite result data before writing any success record.
                json.dumps(result, allow_nan=False)
            except (ValueError, TypeError, PipelineError, OSError) as exc:
                result.update(metrics={}, values={}, curves=None,
                              error={'code': 'INVALID_RESULT', 'message': str(exc)})
            else:
                metadata = snapshot.metadata
                saved = build_saved_benchmark_result(
                    view, run_id=snapshot.run_id, model_name=metadata.model_name,
                    configuration=metadata.configuration, configuration_options=asdict(metadata),
                    result_file=snapshot.output_file, log_file=snapshot.log_file,
                    context=snapshot.command_context, selected_keys=self.settings.default_comparison_keys)
                try:
                    saved_path = save_benchmark_result(saved, self.settings.saved_results_root)
                except (OSError, PipelineError) as exc:
                    result['error'] = {'code': 'SAVE_FAILED', 'message': str(exc)}
                else:
                    result['artifacts']['saved_record'] = str(saved_path)
                    result.update(status='succeeded', error=None)
        result['log_tail'] = read_log_tail(snapshot.log_file)
        self.persist(result)
        self.records[snapshot.run_id] = result
        return result

    def refresh(self, run_id):
        record = self.load(run_id)
        if record['status'] != 'running':
            return record
        snapshot = self.runner.snapshot(run_id)
        if snapshot is None:
            raise ApiError(500, 'INTERNAL_ERROR', 'Managed benchmark disappeared')
        error = None
        if snapshot.status == STATUS_RUNNING:
            if time.monotonic() < self.deadlines[run_id]:
                return record
            snapshot = self.runner.stop(run_id)
            error = {'code': 'TIMEOUT', 'message': 'Evaluation exceeded timeout_seconds'}
        return self.finish(record, snapshot, error)

    def monitor(self, run_id):
        while True:
            with self.lock:
                try:
                    if self.refresh(run_id)['status'] != 'running':
                        return
                except Exception as exc:
                    logging.exception('Evaluation finalization failed: %s', run_id)
                    snapshot = self.runner.snapshot(run_id)
                    if snapshot and snapshot.status == STATUS_RUNNING:
                        self.runner.stop(run_id)
                    record = dict(self.records[run_id])
                    record.update(status='failed', ended_at=datetime.now(timezone.utc).isoformat(),
                                  error={'code': 'FINALIZATION_FAILED', 'message': str(exc)})
                    self.records[run_id] = record
                    try:
                        self.persist(record)
                    except Exception:
                        logging.exception('Cannot persist failed evaluation: %s', run_id)
                    return
            time.sleep(0.1)

    @staticmethod
    def summary(record):
        return {key: record[key] for key in (
            'run_id', 'status', 'started_at', 'ended_at', 'return_code', 'error')}

    def status(self, run_id):
        with self.lock:
            return self.summary(self.refresh(run_id))

    def result(self, run_id):
        with self.lock:
            record = self.refresh(run_id)
            if record['status'] == 'running':
                raise ProcessStateError('Evaluation is still running')
            return dict(record)

    def cancel(self, run_id):
        with self.lock:
            record = self.refresh(run_id)
            if record['status'] == 'running':
                snapshot = self.runner.stop(run_id)
                record = self.finish(record, snapshot)
            return self.summary(record)

    def log(self, run_id, max_bytes=65536):
        if not 1 <= max_bytes <= 4 * 1024 * 1024:
            raise PipelineError('max_bytes must be between 1 and 4194304')
        with self.lock:
            self.load(run_id)
            path = self.path(run_id).parent / 'benchmark.log'
            return {'run_id': run_id, 'text': read_log_tail(path, max_bytes)}

    def shutdown(self):
        with self.lock:
            self.closing = True
            for snapshot in self.runner.snapshots():
                if snapshot.status == STATUS_RUNNING:
                    snapshot = self.runner.stop(snapshot.run_id)
                record = self.records.get(snapshot.run_id)
                if record is not None:
                    self.finish(record, snapshot)
        for monitor in self.monitors:
            monitor.join(timeout=5)
