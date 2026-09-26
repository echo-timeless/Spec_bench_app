"""Execute the documented terminal commands against the real HTTP server."""
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

import pytest

from .test_api import running_server
from .test_benchmark_pipeline import pipeline_setup


@pytest.mark.skipif(shutil.which('curl') is None, reason='Documentation uses curl')
def test_documented_commands_complete_and_cancel_evaluations(pipeline_setup, tmp_path):
    settings, _ = pipeline_setup
    document = (Path(__file__).resolve().parents[1] / 'API.md').read_text()
    blocks = re.findall(r'```bash\n(.*?)\n```', document, re.S)

    def block(start):
        return next(value for value in blocks if value.startswith(start))

    environment = dict(os.environ, DOC_TMP=str(tmp_path), NO_PROXY='127.0.0.1')

    def execute(script):
        # Keep documented /tmp files inside this test's own temporary directory.
        script = script.replace('"/tmp/${RUN_ID}', '"${DOC_TMP}/${RUN_ID}')
        completed = subprocess.run(
            ['bash', '-e', '-c', script], env=environment, cwd=tmp_path,
            text=True, capture_output=True, timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout

    def request(script):
        payload = json.loads(execute(script))
        assert payload['ok'], payload
        return payload['data']

    status_command = block('curl -sS "$API_URL/api/v1/evaluations/$RUN_ID"')
    result_command = block('curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/result"')
    cancel_command = block('curl -sS -X POST "$API_URL/api/v1/evaluations/$RUN_ID:cancel"')
    submit_command = block('curl -sS -X POST "$API_URL/api/v1/evaluations"')

    def prepare_request(port):
        creation = block('API_URL=')
        creation = creation.replace('http://127.0.0.1:8765', f'http://127.0.0.1:{port}')
        # The variables printed here are test observations, not a client API.
        output = execute(creation + '\nprintf "\\nDOC_VARS=%s|%s|%s\\n" "$API_URL" "$RUN_ID" "$REQUEST_FILE"')
        values = next(line.removeprefix('DOC_VARS=') for line in output.splitlines()
                      if line.startswith('DOC_VARS='))
        environment.update(zip(('API_URL', 'RUN_ID', 'REQUEST_FILE'), values.split('|')))
        assert json.loads(Path(environment['REQUEST_FILE']).read_text())['run_id'] == environment['RUN_ID']

    with running_server(settings) as (_, server):
        port = server.server_port
        prepare_request(port)
        assert request(block('curl -sS "$API_URL/health"')) == {'status': 'ok'}
        assert request(block('curl -sS -X POST "$API_URL/api/v1/evaluations/validate"'))['valid']
        assert request(submit_command)['run_id'] == environment['RUN_ID']
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = request(status_command)
            if status['status'] != 'running':
                break
            time.sleep(0.02)
        assert status['status'] == 'succeeded'
        result = request(result_command)
        assert result['metrics']['output_throughput'] == 80.0
        assert Path(result['artifacts']['saved_record']).is_file()
        assert 'completed' in request(block('curl -sS "$API_URL/api/v1/evaluations/$RUN_ID/log?'))['text']
        assert request(cancel_command)['status'] == 'succeeded'
        download = next(value for value in blocks if '-response.json' in value)
        execute(download)
        saved_response = json.loads((tmp_path / f"{environment['RUN_ID']}-response.json").read_text())
        assert saved_response['data'] == result

        module = settings.sglang_repo / 'python/sglang/benchmark/serving.py'
        module.write_text('import time\nprint("waiting", flush=True)\ntime.sleep(30)\n')
        prepare_request(port)
        request(submit_command)
        assert request(cancel_command)['status'] == 'stopped'
        assert request(result_command)['error']['code'] == 'CANCELLED'

    assert server.fileno() == -1
    assert all(snapshot.status != 'Running' for snapshot in server.service.runner.snapshots())
    assert not any(thread.is_alive() for thread in server.service.monitors)
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(('127.0.0.1', port), timeout=1)
