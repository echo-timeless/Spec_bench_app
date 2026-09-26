"""HTTP API for submitting and retrieving persistent benchmark evaluations."""
from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import threading
from urllib.parse import unquote, urlsplit, parse_qs

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = Path(__file__).resolve().parent.name

from .core.http_service import BenchmarkService, ApiError
from .core.pipeline import load_settings, PipelineError, ProcessStateError


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


class Handler(BaseHTTPRequestHandler):
    server_version = 'SpecBenchAPI/1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def body(self):
        if self.headers.get('Transfer-Encoding'):
            raise ApiError(400, 'INVALID_JSON', 'Transfer-Encoding is not supported')
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError as exc:
            raise ApiError(400, 'INVALID_JSON', 'Invalid Content-Length') from exc
        if size < 0 or size > 4 * 1024 * 1024:
            raise ApiError(413, 'BODY_TOO_LARGE', 'Body limit is 4 MiB')
        if size == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(size), parse_constant=self.reject_constant)
        except (ValueError, UnicodeError) as exc:
            raise ApiError(400, 'INVALID_JSON', 'Body must be valid finite JSON') from exc
        if not isinstance(value, dict):
            raise ApiError(400, 'INVALID_JSON', 'Body must be a JSON object')
        return value

    @staticmethod
    def reject_constant(value):
        raise ValueError(f'Invalid JSON constant: {value}')

    def dispatch(self):
        service = self.server.service
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.rstrip('/').split('/')]
        method = self.command
        if method == 'GET' and parts == ['', 'health']:
            return 200, {'status': 'ok'}
        if parts[:3] != ['', 'api', 'v1']:
            raise ApiError(404, 'NOT_FOUND', 'Route not found')
        route = parts[3:]
        body = self.body() if method in ('POST', 'PATCH') else {}
        if method == 'POST' and route == ['evaluations', 'validate']:
            return 200, service.validate(body)
        if method == 'POST' and route == ['evaluations']:
            return 202, service.submit(body)
        if len(route) >= 2 and route[0] == 'evaluations':
            run_id = route[1]
            if len(route) == 2 and run_id.endswith(':cancel') and method == 'POST':
                if body:
                    raise PipelineError('cancel body must be empty')
                return 200, service.cancel(run_id[:-7])
            if len(route) == 2 and method == 'GET':
                return 200, service.status(run_id)
            if len(route) == 3 and method == 'GET':
                if route[2] == 'result':
                    return 200, service.result(run_id)
                if route[2] == 'log':
                    try:
                        max_bytes = int(parse_qs(url.query).get('max_bytes', ['65536'])[0])
                    except ValueError as exc:
                        raise PipelineError('max_bytes must be an integer') from exc
                    return 200, service.log(run_id, max_bytes)
        raise ApiError(404, 'NOT_FOUND', 'Route not found')

    def handle_request(self):
        try:
            status, data = self.dispatch()
            payload = {'ok': True, 'data': data, 'error': None}
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=json_default).encode('utf-8')
        except Exception as exc:
            if isinstance(exc, ApiError):
                status, code = exc.status, exc.code
            elif isinstance(exc, ProcessStateError):
                status, code = 409, 'CONFLICT'
            elif isinstance(exc, PipelineError):
                status, code = 422, 'VALIDATION_ERROR'
            else:
                status, code = 500, 'INTERNAL_ERROR'
            encoded = json.dumps({'ok': False, 'data': None, 'error': {'code': code, 'message': str(exc)}}, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except (BrokenPipeError, ConnectionResetError):
            pass

    do_GET = handle_request
    do_POST = handle_request
    do_PATCH = handle_request
    do_DELETE = handle_request

    def log_message(self, format, *args):
        return


class BenchmarkHTTPServer(ThreadingHTTPServer):
    # Closing the server waits for in-flight operations, then stops owned jobs.
    daemon_threads = False

    def __init__(self, address, service):
        self.service = service
        super().__init__(address, Handler)

    def server_close(self):
        super().server_close()
        self.service.shutdown()


def create_server(host='127.0.0.1', port=8765, config_path=None, *, settings=None):
    service = BenchmarkService(settings or (load_settings(config_path) if config_path else load_settings()))
    return BenchmarkHTTPServer((host, port), service)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path)
    args = parser.parse_args(argv)
    server = create_server(args.host, args.port, args.config)
    def stop(signum, frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f'Spec Bench API listening on http://{args.host}:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
