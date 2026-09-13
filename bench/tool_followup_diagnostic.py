#!/usr/bin/env python3
"""Replay synthetic failed tool followups and retain the exact server wire data."""
import argparse
import json
from pathlib import Path
import time
import uuid
from qualification import append, digest, fetch, idle, payload, request
from task_qualification import CASES, FILES, TOOL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://100.120.26.16:8888')
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--profile', required=True)
    a = p.parse_args()
    directory = a.records/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+a.profile+'-tool-diagnostic-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, mode=0o700)
    (directory/'manifest.json').write_text(json.dumps({'profile':a.profile,'suite':'tool-diagnostic','baseline_run':a.baseline_run.name},indent=2))
    (directory/'source.py').write_bytes(Path(__file__).read_bytes())
    print('RUN_DIRECTORY='+str(directory),flush=True)
    cases = {c['id']:c for c in CASES}
    for line in (a.baseline_run/'events.jsonl').read_text().splitlines():
        wave = json.loads(line)
        if wave['event'] != 'task_wave': continue
        for task in wave['tasks']:
            if task['passed'] or cases[task['case']]['kind'] != 'repo': continue
            case = cases[task['case']]; first = task['requests'][0]
            body = payload('Read '+case['path']+' with read_file, then answer this question as JSON only: '+case['question']+' Use this JSON shape: '+json.dumps({k:None for k in case['expected']}),384)
            calls = list(first['tool_calls'].values())
            body['messages'] += [{'role':'assistant','content':first['output'] or None,'tool_calls':calls},
                                 {'role':'tool','tool_call_id':calls[0]['id'],'content':FILES[case['path']]}]
            for mode in ('stream-none','stream-auto','stream-without-tools','nonstream-none'):
                b = json.loads(json.dumps(body)); b.update(tools=[TOOL],tool_choice='none',return_token_ids=True)
                if mode == 'stream-auto': b['tool_choice']='auto'
                if mode == 'stream-without-tools': del b['tools']; del b['tool_choice']
                if mode == 'nonstream-none': b['stream']=False; del b['stream_options']
                before = idle(a.base)
                append(directory/'events.jsonl',{'event':'request_start','case':case['id'],'mode':mode,'payload':b,'sha256':digest(b)})
                if b['stream']:
                    result = request(a.base,b,retain_wire=True); usage = result['usage'] or {}
                else:
                    with fetch(a.base,'/v1/chat/completions',b) as response: result=json.load(response)
                    usage = result.get('usage') or {}
                after = idle(a.base)
                valid = after['vllm:generation_tokens_total']-before['vllm:generation_tokens_total'] == usage.get('completion_tokens')
                append(directory/'events.jsonl',{'event':'request_finish','case':case['id'],'mode':mode,'valid':valid,'result':result})
                print(json.dumps({'case':case['id'],'mode':mode,'valid':valid,'output':result.get('output'),'finish':result.get('finish')}),flush=True)
                if not valid: raise RuntimeError('Contaminated diagnostic; raw run retained')
    append(directory/'events.jsonl',{'event':'run_finish','status':'complete'})


if __name__ == '__main__': main()
