#!/usr/bin/env python3
"""Identity-blind objective quality checks for retained controlled task outputs.

This is finite regression qualification, not a general model-quality assessment.
Build removes profile, run, timing and original-grade data from grader inputs.
Grade reads only those inputs; reveal joins labels after grades are persisted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid
from task_qualification import CASES, SANDBOX, json_answer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extra_tests(case):
    """Held-out boundary checks, independent of the three generation-time checks."""
    rng=random.Random(71423)
    result=[]
    if case=='repair-cache':
        result=[[[{}],None],[[{'cacheReadTokens':8}],None],[[{'inputTokens':0,'cacheReadTokens':9}],9]]
        for _ in range(20):
            a,b=rng.randrange(10**9),rng.randrange(10**9)
            result.append([[{'inputTokens':a,'cacheReadTokens':b}],a+b])
    elif case=='repair-chunks':
        for n in (0,1,2,7,19):
            for size in (-2,0,1,3,20):
                values=list(range(n))
                result.append([[values,size],None if size<=0 else [values[i:i+size] for i in range(0,n,size)]])
    elif case=='repair-expiry':
        for created in (-10,0,17,10**8):
            for ttl in (0,1,23):
                for offset in (-.25,0,.25):
                    now=created+ttl+offset
                    result.append([[created,ttl,now],now>=created+ttl])
    elif case=='repair-rates':
        result=[[[[],[0],[1]],None],[[[1],[],[1]],None],[[[1],[0],[]],None],[[[0],[0],[1]],0],[[[1],[3],[2]],None]]
        for n in (1,2,7):
            counts=[rng.randrange(1000) for _ in range(n)]
            starts=[rng.randrange(20) for _ in range(n)]
            ends=[30+rng.randrange(20) for _ in range(n)]
            result.append([[counts,starts,ends],sum(counts)/(max(ends)-min(starts))])
    return result


def evaluate(item):
    case=next(c for c in CASES if c['id']==item['case'])
    rows=item['responses'];answer=rows[-1]['output'] if rows else ''
    result={'id':item['id'],'case':case['id'],'passed':False,'strict_json':False}
    try:
        try:json.loads(answer);result['strict_json']=True
        except ValueError:pass
        if not rows or any(r['transport_error'] for r in rows):raise ValueError('transport failure')
        value=json_answer(answer)
        if case['kind']!='repo' and any(r['tool_calls'] for r in rows):raise ValueError('unexpected tool call')
        if case['kind']=='repair':
            tests=case['tests']+extra_tests(case['id'])
            check=subprocess.run([sys.executable,'-I','-c',SANDBOX],input=json.dumps({'code':value['code'],'tests':tests}),capture_output=True,text=True,timeout=3)
            result.update(passed=check.returncode==0,executable_checks=len(tests))
            if check.returncode:result['error']='executable or restricted-runtime check failed'
        elif case['kind']=='repo':
            expected_call={'path':case['path']}
            protocol=len(rows)>=2 and not rows[-1]['tool_calls']
            for row in rows[:-1]:
                calls=list(row['tool_calls'].values())
                protocol=protocol and len(calls)==1 and calls[0]['function']['name']=='read_file' and json.loads(calls[0]['function']['arguments'])==expected_call
            result.update(passed=protocol and value==case['expected'],tool_protocol=bool(protocol))
        else:
            result['passed']=value=={'values':case['values'],'sum':sum(case['values'])}
    except Exception as exc:
        result['error']=type(exc).__name__
    return result


def build(root,runs):
    directory=root/('blind-'+uuid.uuid4().hex);blind=directory/'anonymous';blind.mkdir(parents=True,mode=0o700)
    items=[];keys={};sources={}
    for run in runs:
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest.get('tool_mode')!='native-auto':raise ValueError('Only native-auto task runs qualify')
        events=[json.loads(x) for x in (run/'events.jsonl').read_text().splitlines()]
        if not any(x.get('event')=='run_finish' and x.get('status')=='complete' for x in events):raise ValueError('Incomplete run')
        waves=[x for x in events if x.get('event')=='task_wave']
        if not waves or any(not x['valid'] for x in waves):raise ValueError('Invalid task wave')
        sources[run.name]={'manifest_sha256':digest(run/'manifest.json'),'events_sha256':digest(run/'events.jsonl')}
        for wave in waves:
            for row in wave['tasks']:
                identity=uuid.uuid4().hex
                keys[identity]={'profile':manifest['profile'],'run':run.name,'repeat':wave['repeat'],'concurrency':wave['concurrency'],'case':row['case']}
                items.append({'id':identity,'case':row['case'],'responses':[{'output':r['output'],'tool_calls':r['tool_calls'],'transport_error':bool(r.get('error'))} for r in row['requests']]})
    random.SystemRandom().shuffle(items)
    (blind/'inputs.json').write_text(json.dumps(items,indent=2))
    (directory/'key.json').write_text(json.dumps({'inputs_sha256':digest(blind/'inputs.json'),'items':keys,'sources':sources},indent=2))
    for name in ('blind_quality.py','task_qualification.py','qualification.py'):
        (blind/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    print(directory)


def grade(blind):
    inputs=blind/'inputs.json';out=blind/'grades.json'
    result={'inputs_sha256':digest(inputs),'grader_sha256':digest(Path(__file__)),'task_checker_sha256':digest(Path(__file__).with_name('task_qualification.py')),'items':[evaluate(x) for x in json.loads(inputs.read_text())]}
    with out.open('x') as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
    print(out)


def reveal(directory):
    key=json.loads((directory/'key.json').read_text());grades=json.loads((directory/'anonymous/grades.json').read_text())
    if key['inputs_sha256']!=grades['inputs_sha256']:raise ValueError('Input hash mismatch')
    if len(grades['items'])!=len(key['items']) or {g['id'] for g in grades['items']}!=set(key['items']):raise ValueError('Incomplete or duplicate grades')
    profiles={}
    for row in grades['items']:
        k=key['items'][row['id']];p=profiles.setdefault(k['profile'],{'passed':0,'total':0,'strict_json':0,'failures':[]})
        p['total']+=1;p['passed']+=bool(row['passed']);p['strict_json']+=bool(row['strict_json'])
        if not row['passed']:p['failures'].append({**k,**row})
    out={'scope':'Identity-blind objective regression checks with held-out repair boundaries; not general ability or human editorial review.','sources':key['sources'],'grades_sha256':digest(directory/'anonymous/grades.json'),'profiles':profiles}
    with (directory/'revealed.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out['profiles'],indent=2))


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--out',type=Path,required=True);b.add_argument('runs',type=Path,nargs='+')
    g=sub.add_parser('grade');g.add_argument('anonymous',type=Path)
    r=sub.add_parser('reveal');r.add_argument('directory',type=Path)
    a=p.parse_args()
    if a.command=='build':build(a.out,a.runs)
    elif a.command=='grade':grade(a.anonymous)
    else:reveal(a.directory)
