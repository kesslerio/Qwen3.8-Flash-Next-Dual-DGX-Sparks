#!/usr/bin/env python3
"""Run a coordinated native DSH task wave; retain task and shared-window timing."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from qualification import append, idle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://100.120.26.16:8888')
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--fixture-run',type=Path,required=True)
    parser.add_argument('--concurrency',type=int,choices=(1,3),default=3)
    a=parser.parse_args()
    os.umask(0o077)
    identity=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+a.profile+'-native-wave-'+uuid.uuid4().hex[:8]
    directory=a.records/identity;directory.mkdir(parents=True)
    for name in ('native_dsh_wave.py','native_dsh_contract.py','qualification.py'):
        (directory/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (directory/'manifest.json').write_text(json.dumps({'profile':a.profile,'suite':'native-dsh-wave',
        'concurrency':a.concurrency,'fixture_run':str(a.fixture_run),'fault_injected':False},indent=2))
    print('RUN_DIRECTORY='+str(directory),flush=True)
    def child(index):
        command=[sys.executable,str(Path(__file__).with_name('native_dsh_contract.py')),'--base',a.base,
            '--records',str(a.records),'--profile',a.profile+'-native-c'+str(a.concurrency)+'-'+str(index),
            '--fixture-run',str(a.fixture_run),'--no-fault','--coordinated']
        result=subprocess.run(command,capture_output=True,text=True,timeout=1000)
        (directory/f'child-{index}.log').write_text(result.stdout+result.stderr)
        path=next((Path(line.split('=',1)[1]) for line in result.stdout.splitlines() if line.startswith('RUN_DIRECTORY=')),None)
        if path is None:raise RuntimeError('Native child did not create a run')
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        assertions=[row for row in events if row['event']=='assertion']
        responses=[row for row in events if row['event']=='proxy_response']
        return {'run':path.name,'exit_code':result.returncode,'assertions':assertions,'responses':responses}
    try:
        before=idle(a.base);started=time.monotonic()
        append(directory/'events.jsonl',{'event':'wave_start','utc':time.time(),'before':before})
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            children=list(pool.map(child,range(a.concurrency)))
        elapsed=time.monotonic()-started;after=idle(a.base)
        responses=[r for c in children for r in c['responses']]
        expected=sum((r.get('usage') or {}).get('completion_tokens',0) for r in responses)
        valid=bool(responses) and all(r.get('usage') for r in responses) and after['vllm:generation_tokens_total']-before['vllm:generation_tokens_total']==expected
        passed=sum(c['exit_code']==0 and bool(c['assertions']) and all(r['passed'] for r in c['assertions']) for c in children)
        append(directory/'events.jsonl',{'event':'native_wave','valid':bool(valid),'passed':passed,'total':len(children),
            'shared_wall_s':elapsed,'generation_counter_delta':after['vllm:generation_tokens_total']-before['vllm:generation_tokens_total'],
            'reported_completion_tokens':expected,'before':before,'after':after,'children':children})
        append(directory/'events.jsonl',{'event':'run_finish','status':'complete' if valid else 'invalid','utc':time.time()})
        if not valid:
            print('Native wave traffic attribution incomplete; retained as invalid',file=sys.stderr)
            return 2
    except Exception as error:
        append(directory/'events.jsonl',{'event':'run_finish','status':'failed','error':type(error).__name__,'utc':time.time()})
        raise


if __name__=='__main__':sys.exit(main())
