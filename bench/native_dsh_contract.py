#!/usr/bin/env python3
"""Native DSH read-only tool round trip with one local injected HTTP 503.

The temporary loopback proxy forwards to the selected existing Qwen API. It
retains only request counts, hashes, sampling and tool-role counts, never prompt
contents or credentials. The ordinary DSH profile is preserved by a CLI overlay.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

from qualification import append, idle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://100.120.26.16:8888')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--dsh', default='/Users/kesslerio/.local/bin/dsh')
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.records / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + args.profile + '-native-dsh-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    events = directory / 'events.jsonl'
    (directory / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (directory / 'qualification.py').write_bytes(Path(__file__).with_name('qualification.py').read_bytes())
    marker = 'QWEN_NATIVE_' + uuid.uuid4().hex
    fixture = directory / 'read-only-fixture.txt'
    fixture.write_text(marker + '\n')
    lock = threading.Lock()
    observations = []

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            document = json.loads(body)
            with lock:
                attempt = len(observations)
                observation = {'event': 'proxy_request', 'attempt': attempt,
                    'utc': time.time(), 'body_sha256': hashlib.sha256(body).hexdigest(),
                    'model': document.get('model'),
                    'sampling': {k: document.get(k) for k in ('temperature', 'top_p', 'top_k')},
                    'tool_result_messages': sum(m.get('role') == 'tool' for m in document.get('messages', [])),
                    'injected_503': attempt == 0}
                observations.append(observation)
                append(events, observation)
            if attempt == 0:
                payload = b'{"error":{"message":"Controlled qualification retry fixture","type":"server_error"}}'
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Retry-After', '0')
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path != '/v1/chat/completions':
                self.send_error(404)
                return
            request = urllib.request.Request(args.base.rstrip('/') + self.path,
                data=body, headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    self.send_response(response.status)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'text/event-stream'))
                    self.end_headers()
                    for line in response:
                        self.wfile.write(line)
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                append(events, {'event': 'proxy_client_closed', 'attempt': attempt})
            except urllib.error.HTTPError as error:
                self.send_response(error.code)
                self.end_headers()
                self.wfile.write(error.read())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    overlay = directory / 'native-route.patch.yml'
    overlay.write_text('- id: agent-default-model\n  config:\n    provider: john-remote\n    model: qwen3.8-flash-next\n'
        '- id: llm-pi-ai\n  config:\n    providers:\n      john-remote:\n'
        f'        baseURL: "http://127.0.0.1:{server.server_port}/v1"\n')
    question = f'Read only the file {fixture} using your file-reading or shell tool. Do not modify files, browse, or delegate. Reply with exactly the single line contained in that file.'
    command = [args.dsh, '--profile', 'headless', '--patch', str(overlay), question]
    (directory / 'manifest.json').write_text(json.dumps({'profile': args.profile, 'suite': 'native-dsh-contract',
        'command': command, 'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'injected_fault': 'first HTTP request receives 503, later requests reach the existing API'}, indent=2))
    print('RUN_DIRECTORY=' + str(directory), flush=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        before = idle(args.base)
        append(events, {'event': 'run_start', 'utc': time.time(), 'before': before})
        with (directory / 'stdout.txt').open('w') as stdout, (directory / 'stderr.txt').open('w') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600)
        after = idle(args.base)
        output = (directory / 'stdout.txt').read_text()
        passed = result.returncode == 0 and output.strip() == marker and len(observations) >= 3 and any(o['tool_result_messages'] for o in observations)
        append(events, {'event': 'assertion', 'label': 'native-tool-and-503-retry', 'passed': passed,
            'exit_code': result.returncode, 'requests': len(observations), 'after': after})
        append(events, {'event': 'run_finish', 'status': 'complete' if passed else 'failed', 'utc': time.time()})
        if not passed:
            raise RuntimeError('Native DSH compatibility failed; evidence retained')
    except Exception as error:
        append(events, {'event': 'run_finish', 'status': 'failed', 'error': type(error).__name__, 'utc': time.time()})
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
