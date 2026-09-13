#!/usr/bin/env python3
"""Retain streamed tools, image input, cancellation and post-cancel isolation."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid

from agent_contract import two_color_png
from qualification import append, fetch, idle, payload, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://100.120.26.16:8888')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--profile', required=True)
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.records / (time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + args.profile + '-compat-' + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    for filename in ('api_compatibility.py', 'qualification.py', 'agent_contract.py'):
        (directory / filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    (directory / 'manifest.json').write_text(json.dumps({'profile': args.profile, 'suite': 'api-compatibility',
        'temperature': 0, 'top_p': .95, 'top_k': 20}, indent=2))
    events = directory / 'events.jsonl'
    print('RUN_DIRECTORY=' + str(directory), flush=True)
    rows = []

    def call(label, body):
        append(directory / 'synthetic-payloads.jsonl', {'label': label, 'payloads': [body]})
        row = request(args.base, body, retain_wire=True)
        append(directory / 'requests.jsonl', {'label': label, **row})
        rows.append(row)
        if row.get('error'):
            raise RuntimeError('Incomplete compatibility response: ' + label)
        return row

    def check(label, passed, **extra):
        append(events, {'event': 'assertion', 'label': label, 'passed': passed, **extra})
        if not passed:
            raise RuntimeError('Compatibility assertion failed: ' + label)

    try:
        before = idle(args.base)
        append(events, {'event': 'run_start', 'utc': time.time(), 'before': before})
        body = payload('Use multiply to calculate 3 times 7. After the tool returns, answer with just the result.', 128)
        tool = {'type': 'function', 'function': {'name': 'multiply', 'description': 'Multiply integers',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}}, 'required': ['a', 'b']}}}
        body.update(tools=[tool], tool_choice='required')
        first = call('tool-call', body)
        calls = list(first['tool_calls'].values())
        check('tool-arguments', len(calls) == 1 and calls[0]['function']['name'] == 'multiply' and json.loads(calls[0]['function']['arguments']) == {'a': 3, 'b': 7})
        body['messages'] += [{'role': 'assistant', 'content': first['output'] or None, 'tool_calls': calls},
            {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '21'}]
        body['tool_choice'] = 'auto'
        final = call('tool-result', body)
        check('tool-result', final['output'].strip().strip('.') == '21' and not final['tool_calls'])
        image = payload('', 96)
        image['messages'][0]['content'] = [{'type': 'text', 'text': 'Name the two solid colors, left then right. Answer with only the color names.'},
            {'type': 'image_url', 'image_url': {'url': two_color_png()}}]
        answer = call('image', image)['output'].lower()
        check('image-colors', 'red' in answer and 'blue' in answer and answer.index('red') < answer.index('blue'))
        after = idle(args.base)
        total = sum(row['usage']['completion_tokens'] for row in rows)
        check('controlled-tool-image-traffic', after['vllm:generation_tokens_total'] - before['vllm:generation_tokens_total'] == total)

        cancel_body = payload('Explain sorting algorithms in detail.', 8000, fixed=True)
        append(directory / 'synthetic-payloads.jsonl', {'label': 'cancel', 'payloads': [cancel_body]})
        seen = False
        with fetch(args.base, '/v1/chat/completions', cancel_body, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                    continue
                event = json.loads(line[5:])
                append(directory / 'cancel-wire.jsonl', event)
                if any(c.get('delta', {}).get('content') for c in event.get('choices', [])):
                    seen = True
                    break
        check('cancel-after-first-content', seen)
        cancelled_at = time.monotonic()
        drained = idle(args.base, timeout=20)
        check('cancel-releases-engine', True, release_s=time.monotonic() - cancelled_at,
            generation_counter_delta=drained['vllm:generation_tokens_total'] - after['vllm:generation_tokens_total'])
        marker = 'FRESH_' + uuid.uuid4().hex
        retry = call('fresh-after-cancel', payload('Reply with only this exact marker: ' + marker, 96))
        check('fresh-request-isolation', retry['output'].strip() == marker)
        ended = idle(args.base)
        check('fresh-request-traffic', ended['vllm:generation_tokens_total'] - drained['vllm:generation_tokens_total'] == retry['usage']['completion_tokens'])
        check('no-preemptions', ended['vllm:num_preemptions_total'] == before['vllm:num_preemptions_total'])
        append(events, {'event': 'run_finish', 'status': 'complete', 'utc': time.time()})
    except Exception as error:
        append(events, {'event': 'run_finish', 'status': 'failed', 'error': type(error).__name__ + ': ' + str(error)[:200], 'utc': time.time()})
        raise


if __name__ == '__main__':
    main()
