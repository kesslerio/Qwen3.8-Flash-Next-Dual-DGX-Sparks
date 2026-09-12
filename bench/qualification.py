#!/usr/bin/env python3
"""Durable, contamination-checked Qwen field qualification. No service mutations.

Raw synthetic fixture outputs stay in the private run directory. Each invocation
has a unique ID; interrupted/failed runs remain recorded rather than overwritten.
SSE events are not tokens: token counts always come from server usage.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request
import uuid

MODEL = 'qwen3.8-flash-next'
PROMPTS = {
    'prose': 'Explain hash maps in detailed prose, including collisions, load factors, open addressing, resizing and concrete tradeoffs. Aim for 1000 words.',
    'code': 'Implement a complete Python LRU cache with get, put, bounded capacity and tests. Explain its concurrency and complexity tradeoffs in comments.',
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, separators=(',', ':')) + '\n')
        f.flush()
        os.fsync(f.fileno())

def fetch(base, path, payload=None, timeout=1200):
    request = urllib.request.Request(base + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=timeout)

def metrics(base):
    with fetch(base, '/metrics', timeout=15) as r:
        text = r.read().decode()
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{')[0].split()[0]
        if name in ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                    'vllm:generation_tokens_total', 'vllm:num_preemptions_total'):
            values[name] = values.get(name, 0) + float(line.rsplit(' ', 1)[1])
    required = ('vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:generation_tokens_total')
    if not all(k in values for k in required):
        raise RuntimeError('Required server counters missing; cannot verify isolation')
    return values

def idle(base, timeout=180):
    end = time.monotonic() + timeout
    consecutive = 0
    while time.monotonic() < end:
        m = metrics(base)
        consecutive = consecutive + 1 if m['vllm:num_requests_running'] == m['vllm:num_requests_waiting'] == 0 else 0
        if consecutive >= 3:
            return m
        time.sleep(1)
    raise RuntimeError('Requests did not drain; leave production traffic alone')

def request(base, payload, progress=None):
    start = time.monotonic()
    row = {'started_utc': time.time(), 'start': start, 'payload_sha256': digest(payload),
           'usage': None, 'first': None, 'last': None, 'finish': None, 'output': '', 'gaps_s': [], 'tool_calls': {}}
    try:
        with fetch(base, '/v1/chat/completions', payload) as response:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                if data == b'[DONE]':
                    break
                event = json.loads(data)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('usage'):
                    row['usage'] = event['usage']
                for choice in event.get('choices', []):
                    delta = choice.get('delta') or {}
                    if any(delta.get(k) for k in ('content', 'reasoning_content', 'reasoning', 'tool_calls')):
                        now = time.monotonic()
                        if row['first'] is None:
                            row['first'] = now
                            if progress:
                                progress.set()
                        if row['last'] is not None:
                            row['gaps_s'].append(now - row['last'])
                        row['last'] = now
                    row['output'] += delta.get('content') or ''
                    for part in delta.get('tool_calls') or []:
                        call = row['tool_calls'].setdefault(str(part['index']), {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if part.get('id'):
                            call['id'] = part['id']
                        for key in ('name', 'arguments'):
                            call['function'][key] += (part.get('function') or {}).get(key) or ''
                    row['finish'] = choice.get('finish_reason') or row['finish']
        if row['usage'] is None or row['finish'] is None:
            raise RuntimeError('Incomplete stream or missing usage')
    except Exception as exc:
        row['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
    row['end'] = time.monotonic()
    row['elapsed_s'] = row['end'] - start
    row['ttft_s'] = row['first'] - start if row['first'] else None
    n = (row['usage'] or {}).get('completion_tokens')
    row['decode_tps_approx'] = (n - 1) / (row['last'] - row['first']) if n and row['last'] and row['last'] > row['first'] else None
    row['max_progress_gap_s'] = max(row['gaps_s'], default=0)
    return row

def payload(prompt, tokens=800, temperature=0, fixed=False):
    p = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
         'max_tokens': tokens, 'temperature': temperature, 'top_p': 0.95, 'top_k': 20,
         'stream': True, 'stream_options': {'include_usage': True},
         'chat_template_kwargs': {'enable_thinking': False}}
    if fixed:
        p.update(ignore_eos=True, min_tokens=tokens)
    return p

def group(base, directory, label, payloads, interference=False):
    before = idle(base)
    append(directory / 'events.jsonl', {'event': 'group_start', 'label': label, 'utc': time.time(), 'before': before})
    progress = threading.Event()
    barrier = threading.Barrier(len(payloads))
    def run(index):
        barrier.wait()
        if interference and index:
            if not progress.wait(90):
                raise RuntimeError('Active answer did not start; interference fixture invalid')
            time.sleep(2)
        result = request(base, payloads[index], progress if index == 0 else None)
        result['index'] = index
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(payloads)) as pool:
        rows = list(pool.map(run, range(len(payloads))))
    after = idle(base)
    total = sum((r['usage'] or {}).get('completion_tokens', 0) for r in rows)
    delta = after['vllm:generation_tokens_total'] - before['vllm:generation_tokens_total']
    valid = all('error' not in r for r in rows) and delta == total
    first = [r['first'] for r in rows if r['first'] is not None]
    last = [r['last'] for r in rows if r['last'] is not None]
    result = {'event': 'group_finish', 'label': label, 'utc': time.time(), 'valid': valid,
              'generation_counter_delta': delta, 'completion_tokens': total,
              'preemption_delta': after.get('vllm:num_preemptions_total', 0) - before.get('vllm:num_preemptions_total', 0),
              'before': before, 'after': after, 'requests': rows,
              'aggregate_e2e_tps': total / (max(r['end'] for r in rows) - min(r['start'] for r in rows)),
              'aggregate_decode_tps_approx': (total - len(rows)) / (max(last) - min(first)) if first and last and max(last) > min(first) else None}
    append(directory / 'events.jsonl', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('requests', 'before', 'after')}), flush=True)
    if not valid:
        raise RuntimeError('Contaminated or failed group retained; rerun with a new run ID')
    return rows

def context(base, target, salt):
    # Non-repeating records; unique salt at the start gives a genuinely cold prefix.
    lines = [f'{salt}\nInventory ledger.\n']
    for i in range(int(target / 18)):
        lines.append(f'Item {i:06d}, depot {hashlib.sha256(str(i).encode()).hexdigest()[:8]}, quantity {(i * 37) % 997}.\n')
    text = ''.join(lines)
    with fetch(base, '/tokenize', {'model': MODEL, 'prompt': text}) as r:
        tokens = json.load(r)['tokens']
    with fetch(base, '/detokenize', {'model': MODEL, 'tokens': tokens[:target]}) as r:
        return json.load(r)['prompt']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://100.120.26.16:8888')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--suite', choices=('decode', 'interference', 'conversation'), required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--concurrencies', default='1,4,6')
    parser.add_argument('--contexts', default='32000,128000,240000')
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.records / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + args.profile + '-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    (directory / 'qualification-source.py').write_bytes(Path(__file__).read_bytes())
    manifest = {'schema': 1, 'run_id': directory.name, 'profile': args.profile, 'suite': args.suite,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'fixtures_sha256': digest(PROMPTS), 'arguments': {k: str(v) for k, v in vars(args).items()}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('RUN_DIRECTORY=' + str(directory), flush=True)
    try:
        for rep in range(args.repeats):
            if args.suite == 'decode':
                for kind, prompt in PROMPTS.items():
                    for c in map(int, args.concurrencies.split(',')):
                        group(args.base, directory, f'{kind}-c{c}-r{rep}', [payload(prompt, fixed=True) for _ in range(c)])
            elif args.suite == 'interference':
                ctx = context(args.base, 128000, uuid.uuid4().hex)
                group(args.base, directory, f'interference-r{rep}', [payload(PROMPTS['prose'], fixed=True), payload(ctx + '\nSummarize the format of this ledger in one sentence.', 100)], True)
            else:
                for size in map(int, args.contexts.split(',')):
                    ctx = context(args.base, size, uuid.uuid4().hex)
                    mid = len(ctx) // 2
                    ctx = 'Audit north=73019.\n' + ctx[:mid] + '\nAudit west=21863.\n' + ctx[mid:] + '\nAudit east=59147.\n'
                    p = payload(ctx + '\nReturn JSON with keys north, west, east and their audit integer values. No other text.', 96)
                    cold = group(args.base, directory, f'cold-{size}-r{rep}', [p])[0]
                    branches = []
                    for branch in range(3):
                        q = json.loads(json.dumps(p))
                        q['messages'] += [{'role': 'assistant', 'content': cold['output']}, {'role': 'user', 'content': f'Add north, west, east and {branch}. Reply with only the integer.'}]
                        branches.append(q)
                    warm = group(args.base, directory, f'warm-branches-{size}-r{rep}', branches)
                    try:
                        passed = json.loads(cold['output']) == {'north': 73019, 'west': 21863, 'east': 59147}
                    except ValueError:
                        passed = False
                    passed = passed and all(r['output'].strip().strip('.') == str(154029 + b) for b, r in enumerate(warm))
                    append(directory / 'events.jsonl', {'event': 'assertion', 'label': f'conversation-{size}-r{rep}', 'passed': passed})
                    # A wrong answer is a retained quality result, not a reason to
                    # omit later context sizes. Contamination/transport still abort.
        append(directory / 'events.jsonl', {'event': 'run_finish', 'status': 'complete', 'utc': time.time()})
    except BaseException as exc:
        append(directory / 'events.jsonl', {'event': 'run_finish', 'status': 'failed', 'error': str(exc)[:300], 'utc': time.time()})
        raise

if __name__ == '__main__':
    main()
