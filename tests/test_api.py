"""Real HTTP + fake benchmark tests for the compact evaluation lifecycle."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import pytest
from ..api import create_server
from ..core.pipeline import runner as runner_module
from .test_benchmark_pipeline import pipeline_setup

class HTTPFailure(Exception):
    """Test-only assertion helper for HTTP error responses."""

    def __init__(self, status, error):
        super().__init__(error['message'])
        self.status = status
        self.code = error['code']

@contextmanager
def running_server(settings):
    server = create_server(port=0, settings=settings)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()

    def request(method, path, body=None):
        connection = HTTPConnection('127.0.0.1', server.server_port, timeout=10)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None, {'Content-Type': 'application/json'})
            response = connection.getresponse()
            payload = json.loads(response.read())
            if not payload['ok']:
                raise HTTPFailure(response.status, payload['error'])
            return payload['data']
        finally:
            connection.close()
    try:
        yield (request, server)
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

def job(run_id, **extra):
    return {'run_id': run_id, 'metadata': {'model_name': 'fake', 'tp_size': 1}, **extra}

def wait(request, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if request('GET', '/api/v1/evaluations/' + run_id)['status'] != 'running':
            return request('GET', '/api/v1/evaluations/' + run_id + '/result')
        time.sleep(0.02)
    pytest.fail('Evaluation did not finish')

def test_full_workflow_autosaves_without_polling_and_survives_restart(pipeline_setup):
    settings, _ = pipeline_setup
    with running_server(settings) as (request, server):
        assert request('GET', '/health')['status'] == 'ok'
        validated = request('POST', '/api/v1/evaluations/validate', job('first'))
        assert validated['valid']
        assert not (settings.workspace_root / 'first').exists()
        assert request('POST', '/api/v1/evaluations', job('first'))['run_id'] == 'first'
        path = settings.workspace_root / 'first/evaluation.json'
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            record = json.loads(path.read_text())
            if record['status'] != 'running':
                break
            time.sleep(0.02)
        assert record['status'] == 'succeeded'
        result = request('GET', '/api/v1/evaluations/first/result')
        assert result['metrics']['accept_length'] == 1.5
        assert Path(result['artifacts']['saved_record']).is_file()
        assert 'completed' in request('GET', '/api/v1/evaluations/first/log')['text']
        assert request('POST', '/api/v1/evaluations/first:cancel')['status'] == 'succeeded'
        assert request('GET', '/api/v1/evaluations/first/result') == result
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/first')['status'] == 'succeeded'
        assert request('GET', '/api/v1/evaluations/first/result') == result
        assert 'completed' in request('GET', '/api/v1/evaluations/first/log')['text']
        assert request('POST', '/api/v1/evaluations/first:cancel')['status'] == 'succeeded'
        with pytest.raises(HTTPFailure) as exc:
            request('POST', '/api/v1/evaluations', job('first'))
        assert exc.value.status == 409

@pytest.mark.parametrize('output', [None, '{broken', '{}'])
def test_invalid_results_persist_failure(pipeline_setup, output):
    settings, _ = pipeline_setup
    module = settings.sglang_repo / 'python/sglang/benchmark/serving.py'
    module.write_text('import argparse,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument("--output-file");a,_=p.parse_known_args();time.sleep(.05)\n' + (f'Path(a.output_file).write_text({output!r})\n' if output is not None else ''))
    with running_server(settings) as (request, _):
        request('POST', '/api/v1/evaluations', job('bad'))
        result = wait(request, 'bad')
        assert result['status'] == 'failed'
        assert result['error']['code'] == 'INVALID_RESULT'
        assert result['return_code'] == 0
        assert result['artifacts']['saved_record'] is None
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/bad/result') == result

def test_concurrent_conflict_cancel_timeout_shutdown(pipeline_setup):
    settings, _ = pipeline_setup
    module = settings.sglang_repo / 'python/sglang/benchmark/serving.py'
    module.write_text('import time\nprint("waiting",flush=True)\ntime.sleep(30)\n')
    with running_server(settings) as (request, server):

        def submit(name):
            try:
                return request('POST', '/api/v1/evaluations', job(name))
            except HTTPFailure as exc:
                return exc.status
        with ThreadPoolExecutor(2) as pool:
            responses = list(pool.map(submit, ['a', 'b']))
        assert 409 in responses
        run_id = next((r['run_id'] for r in responses if isinstance(r, dict)))
        with pytest.raises(HTTPFailure) as exc:
            request('GET', '/api/v1/evaluations/' + run_id + '/result')
        assert exc.value.status == 409
        assert request('POST', '/api/v1/evaluations/' + run_id + ':cancel')['status'] == 'stopped'
        assert request('POST', '/api/v1/evaluations/' + run_id + ':cancel')['status'] == 'stopped'
        assert request('GET', '/api/v1/evaluations/' + run_id + '/result')['error']['code'] == 'CANCELLED'
        request('POST', '/api/v1/evaluations', job('timeout', timeout_seconds=0.1))
        result = wait(request, 'timeout')
        assert result['status'] == 'stopped'
        assert result['error']['code'] == 'TIMEOUT'
        assert request('POST', '/api/v1/evaluations/timeout:cancel')['error']['code'] == 'TIMEOUT'
        request('POST', '/api/v1/evaluations', job('shutdown'))
    assert server.service.runner.snapshot('shutdown').status == 'Stopped'
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/shutdown/result')['status'] == 'stopped'

@pytest.mark.parametrize('extra', [{'parameters': {'num_prompts': True}}, {'parameters': {'port': '80'}}, {'parameters': {'request_rate': 'nan'}}, {'parameters': {'dataset_id': 'unknown'}}, {'metadata': {'model_name': None}}, {'parameters': {'typo': 1}}, {'run_id': '..'}, {'run_id': 'x/y'}, {'metadata': {'tp_size': 1.5}}, {'timeout_seconds': 0}, {'timeout_seconds': '10'}, {'command': 'echo no'}])
def test_validation_errors_are_422(pipeline_setup, extra):
    settings, _ = pipeline_setup
    with running_server(settings) as (request, _):
        with pytest.raises(HTTPFailure) as exc:
            request('POST', '/api/v1/evaluations/validate', job('invalid') | extra)
        assert exc.value.status == 422
        assert exc.value.code == 'VALIDATION_ERROR'

def test_bad_json_removed_routes_and_unknown_id(pipeline_setup):
    settings, _ = pipeline_setup
    with running_server(settings) as (request, server):
        for raw in ('{', '{"n":NaN}', '[]'):
            connection = HTTPConnection('127.0.0.1', server.server_port)
            connection.request('POST', '/api/v1/evaluations', raw)
            response = connection.getresponse()
            assert response.status == 400
            assert not json.loads(response.read())['ok']
            connection.close()
        for path in ('/api/v1/jobs', '/api/v1/results', '/api/v1/defaults', '/api/v1/evaluations/missing'):
            with pytest.raises(HTTPFailure) as exc:
                request('GET', path)
            assert exc.value.status == 404

def test_failed_process_and_interrupted_record(pipeline_setup):
    settings, _ = pipeline_setup
    module = settings.sglang_repo / 'python/sglang/benchmark/serving.py'
    module.write_text('import time\ntime.sleep(.05)\nraise SystemExit(7)\n')
    with running_server(settings) as (request, _):
        request('POST', '/api/v1/evaluations', job('exit-seven'))
        result = wait(request, 'exit-seven')
        assert result['status'] == 'failed'
        assert result['return_code'] == 7
        assert result['error']['code'] == 'BENCHMARK_FAILED'
        assert request('POST', '/api/v1/evaluations/exit-seven:cancel')['status'] == 'failed'
    path = settings.workspace_root / 'exit-seven/evaluation.json'
    record = json.loads(path.read_text())
    record.update(status='running', ended_at=None, error=None)
    path.write_text(json.dumps(record))
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/exit-seven')['error']['code'] == 'INTERRUPTED'
        assert request('GET', '/api/v1/evaluations/exit-seven/result')['status'] == 'failed'

def test_example_json_is_accepted_by_http(pipeline_setup):
    settings, _ = pipeline_setup
    root = Path(__file__).resolve().parents[1]
    body = json.loads((root / 'examples/evaluation_request.json').read_text())
    body['parameters']['port'] = settings.port
    body['metadata']['model_name'] = 'fake-example'
    with running_server(settings) as (request, _):
        assert request('POST', '/api/v1/evaluations/validate', body)['valid']
        run_id = request('POST', '/api/v1/evaluations', body)['run_id']
        assert wait(request, run_id)['status'] == 'succeeded'
    assert Path(runner_module.__file__).is_relative_to(root)
    completed = subprocess.run([sys.executable, str(root / 'api.py'), '--help'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

def test_curves_are_in_persistent_result_without_original_files(pipeline_setup):
    settings, _ = pipeline_setup
    module = settings.sglang_repo / 'python/sglang/benchmark/serving.py'
    source = module.read_text()
    stats = {'schema_version': 1, 'requests': [{'request_index': 0, 'stats': {'mode': 'detailed', 'verify_lengths': [4, 4], 'accept_lengths': [2, 4]}}]}
    module.write_text(source.replace('output = Path(args.output_file)', f'result["speculative_decoding_stats"] = {stats!r}\noutput = Path(args.output_file)'))
    with running_server(settings) as (request, _):
        request('POST', '/api/v1/evaluations', job('curves'))
        result = wait(request, 'curves')
        assert result['status'] == 'succeeded'
        assert result['curves']['best']['accept_length'] == 3
    (settings.workspace_root / 'curves/result.jsonl').unlink()
    (settings.workspace_root / 'curves/result_view.json').unlink()
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/curves/result') == result

def test_save_failure_is_not_published_as_success(pipeline_setup, monkeypatch):
    settings, _ = pipeline_setup
    from ..core import http_service

    def fail_save(*args, **kwargs):
        raise OSError('fake storage failure')
    monkeypatch.setattr(http_service, 'save_benchmark_result', fail_save)
    with running_server(settings) as (request, _):
        request('POST', '/api/v1/evaluations', job('save-failure'))
        result = wait(request, 'save-failure')
        assert result['status'] == 'failed'
        assert result['error']['code'] == 'SAVE_FAILED'
        assert request('GET', '/api/v1/evaluations/save-failure')['status'] == 'failed'
    with running_server(settings) as (request, _):
        assert request('GET', '/api/v1/evaluations/save-failure/result') == result
