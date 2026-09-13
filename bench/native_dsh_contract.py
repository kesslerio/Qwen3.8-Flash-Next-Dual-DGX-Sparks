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
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

from qualification import append, idle, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://100.120.26.16:8888')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--dsh', default='/Users/kesslerio/.local/bin/dsh')
    parser.add_argument('--dsh-source', type=Path, default=Path('/Users/kesslerio/projects/tools/deepseek-harness'))
    parser.add_argument('--dsh-config', type=Path, default=Path.home()/'.dsh/profiles/headless/cordis.patch.yml')
    parser.add_argument('--fixture-run', type=Path, help='Reuse a prior native run fixture and its exact task path')
    parser.add_argument('--no-fault', action='store_true', help='Measure ordinary native task completion without injected 503')
    parser.add_argument('--coordinated', action='store_true', help='Parent wave owns idle checks and traffic attribution')
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.records / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + args.profile + '-native-dsh-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    events = directory / 'events.jsonl'
    (directory / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (directory / 'qualification.py').write_bytes(Path(__file__).with_name('qualification.py').read_bytes())
    prior = json.loads((args.fixture_run/'manifest.json').read_text()) if args.fixture_run else {}
    fixture = Path(prior.get('fixture_source', str((args.fixture_run or directory)/'read-only-fixture.txt')))
    marker = fixture.read_text().strip() if args.fixture_run else 'QWEN_NATIVE_' + uuid.uuid4().hex
    (directory/'read-only-fixture.txt').write_text(marker + '\n')
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
                    'injected_503': attempt == 0 and not args.no_fault}
                observations.append(observation)
                append(events, observation)
            if attempt == 0 and not args.no_fault:
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
                started=time.monotonic();usage=None;response_id=None
                with urllib.request.urlopen(request, timeout=600) as response:
                    is_sse='text/event-stream' in response.headers.get('Content-Type','')
                    nonstream=bytearray()
                    self.send_response(response.status)
                    self.send_header('Content-Type', response.headers.get('Content-Type', 'text/event-stream'))
                    self.end_headers()
                    for line in response:
                        if not is_sse:nonstream.extend(line)
                        if line.startswith(b'data:') and line[5:].strip() != b'[DONE]':
                            event=json.loads(line[5:]);usage=event.get('usage') or usage;response_id=event.get('id') or response_id
                        self.wfile.write(line)
                        self.wfile.flush()
                    if nonstream:
                        event=json.loads(nonstream);usage=event.get('usage');response_id=event.get('id')
                append(events, {'event':'proxy_response','attempt':attempt,'usage':usage,'response_id':response_id,'elapsed_s':time.monotonic()-started})
            except (BrokenPipeError, ConnectionResetError):
                append(events, {'event': 'proxy_client_closed', 'attempt': attempt})
            except urllib.error.HTTPError as error:
                self.send_response(error.code)
                self.end_headers()
                self.wfile.write(error.read())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Proxy)
    overlay = directory / 'native-route.patch.yml'
    settings = directory / 'native-settings.json'
    # A saved global default can override the profile's model. Select Qwen only
    # for this isolated invocation; never rewrite the user's saved selection.
    settings.write_text(json.dumps({'agent-default-model': {'provider':'john-remote','model':'qwen3.8-flash-next'}}))
    # DSH replaces this provider entry when applying an overlay. Carry the
    # complete existing route so models, timeouts, sampling and retry policy
    # retain their actual client settings.
    lines=args.dsh_config.read_text().splitlines()
    start=lines.index('      john-remote:')
    end=next((i for i in range(start+1,len(lines)) if re.match(r'^      [^ #]',lines[i])),len(lines))
    route='\n'.join(lines[start:end])+'\n'
    if re.search(r'^\s*(apiKey|token|password)\s*:',route,re.M):
        raise ValueError('Native route must reference credentials through an environment variable')
    route,count=re.subn(r'^        baseURL:.*$',f'        baseURL: "http://127.0.0.1:{server.server_port}/v1"',route,flags=re.M)
    if count!=1:raise ValueError('Expected exactly one route endpoint')
    overlay.write_text('- id: agent-default-model\n  config:\n    provider: john-remote\n    model: qwen3.8-flash-next\n'
        '- id: llm-pi-ai\n  config:\n    providers:\n'+route+
        '- id: settings\n  config:\n    watch: false\n    path: '+json.dumps(str(settings))+'\n')
    question = f'Read only the file {fixture} using your file-reading or shell tool. Do not modify files, browse, or delegate. Reply with exactly the single line contained in that file.'
    command = [args.dsh, '--profile', 'headless', '--patch', str(overlay), question]
    client_revision = subprocess.check_output(['git','-C',str(args.dsh_source),'rev-parse','HEAD'],text=True).strip()
    (directory / 'manifest.json').write_text(json.dumps({'profile': args.profile, 'suite': 'native-dsh-contract',
        'command': command, 'fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
        'client_revision':client_revision,'client_launcher_sha256':hashlib.sha256(Path(args.dsh).read_bytes()).hexdigest(),
        'task_sha256':hashlib.sha256(question.encode()).hexdigest(),
        'client_config_sha256': hashlib.sha256(args.dsh_config.read_bytes()).hexdigest(),
        'fixture_source':str(fixture),
        'injected_fault': 'none' if args.no_fault else 'first HTTP request receives 503, later requests reach the existing API'}, indent=2))
    print('RUN_DIRECTORY=' + str(directory), flush=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        before = metrics(args.base) if args.coordinated else idle(args.base)
        append(events, {'event': 'run_start', 'utc': time.time(), 'before': before})
        task_started=time.monotonic()
        with (directory / 'stdout.txt').open('w') as stdout, (directory / 'stderr.txt').open('w') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=600)
        task_elapsed=time.monotonic()-task_started
        after = metrics(args.base) if args.coordinated else idle(args.base)
        output = (directory / 'stdout.txt').read_text()
        exact_retry = args.no_fault or bool(observations) and any(
            o['body_sha256'] == observations[0]['body_sha256'] for o in observations[1:])
        passed = result.returncode == 0 and output.strip() == marker and exact_retry and len(observations) >= (2 if args.no_fault else 3) and any(o['tool_result_messages'] for o in observations)
        append(events, {'event': 'assertion', 'label': 'native-tool' if args.no_fault else 'native-tool-and-503-retry', 'passed': passed,
            'exit_code': result.returncode, 'requests': len(observations), 'after': after, 'task_elapsed_s':task_elapsed,'includes_cli_startup':True,'fault_injected':not args.no_fault,'exact_retry':exact_retry if not args.no_fault else None})
        if not passed:
            raise RuntimeError('Native DSH compatibility failed; evidence retained')
        append(events, {'event': 'run_finish', 'status': 'complete', 'utc': time.time()})
    except Exception as error:
        append(events, {'event': 'run_finish', 'status': 'failed', 'error': type(error).__name__, 'utc': time.time()})
        raise
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
