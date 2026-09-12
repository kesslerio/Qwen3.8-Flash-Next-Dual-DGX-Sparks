#!/usr/bin/env python3
"""Twelve deterministic repository/tool/repair/synthesis tasks; private results.

Repairs are AST-restricted and evaluated with a tiny builtins allowlist in an
isolated, time-limited Python process. Model output never reaches a shell or the
user's checkout. These diagnostic tasks qualify regressions, not general ability.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from qualification import append, digest, idle, request, payload

FILES = {
    'cache.py': 'def total_prompt(usage):\n    return usage["inputTokens"]\n# inputTokens excludes cached tokens. cacheReadTokens is separately reported.\n',
    'timing.py': 'def ttft(step_start, request_sent, first_payload):\n    return first_payload - step_start\n# Tool work can occur between step_start and request_sent.\n',
    'streams.py': 'def has_progress(delta):\n    return bool(delta.get("content"))\n# The API also streams tool_calls and reasoning_content.\n',
    'batch.py': 'def aggregate(rates):\n    return sum(rates)\n# Requests can start and finish at different times.\n',
}
CASES = [
    {'id':'repo-cache','kind':'repo','path':'cache.py','question':'Which usage field is omitted from the total prompt calculation?', 'expected':{'field':'cacheReadTokens'}},
    {'id':'repo-timing','kind':'repo','path':'timing.py','question':'Which timestamp should replace step_start to measure request-to-first-payload latency?', 'expected':{'timestamp':'request_sent'}},
    {'id':'repo-stream','kind':'repo','path':'streams.py','question':'Which two progress fields are omitted? Return them sorted alphabetically.', 'expected':{'fields':['reasoning_content','tool_calls']}},
    {'id':'repo-batch','kind':'repo','path':'batch.py','question':'Is summing these per-request rates a valid aggregate over a shared wall-clock window?', 'expected':{'valid':False}},
    {'id':'repair-cache','kind':'repair','task':'Implement candidate(usage): total=inputTokens+cacheReadTokens. Return None if either count is missing; zero is valid.', 'tests':[[[{'inputTokens':3,'cacheReadTokens':7}],10],[[{'inputTokens':0,'cacheReadTokens':0}],0],[[{'inputTokens':8}],None]]},
    {'id':'repair-chunks','kind':'repair','task':'Implement candidate(values, size) returning consecutive chunks including the final short chunk. For size <= 0 return None.', 'tests':[[[[1,2,3,4,5],2],[[1,2],[3,4],[5]]],[[[],2],[]],[[[1],0],None]]},
    {'id':'repair-expiry','kind':'repair','task':'Implement candidate(created, ttl, now) returning whether the entry is expired. It expires exactly when now >= created+ttl.', 'tests':[[[10,5,15],True],[[10,5,14],False],[[10,0,10],True]]},
    {'id':'repair-rates','kind':'repair','task':'Implement candidate(counts, starts, ends): sum(counts)/(max(ends)-min(starts)), or None for empty lists or a nonpositive interval.', 'tests':[[[[100,100],[0,5],[10,15]],200/15],[[[],[],[]],None],[[[5],[2],[2]],None]]},
    {'id':'synthesis-0','kind':'synthesis','values':[713,829,491]},
    {'id':'synthesis-1','kind':'synthesis','values':[281,619,937]},
    {'id':'synthesis-2','kind':'synthesis','values':[433,751,127]},
    {'id':'synthesis-3','kind':'synthesis','values':[199,541,863]},
]
TOOL = {'type':'function','function':{'name':'read_file','description':'Read a repository file', 'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path'],'additionalProperties':False}}}
SANDBOX = '''
import ast,json,sys
d=json.load(sys.stdin);code=d['code'];tree=ast.parse(code)
allowed_calls={'candidate','len','range','sum','min','max','list','dict','sorted','enumerate','zip','bool'}
for n in ast.walk(tree):
 if isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.With,ast.AsyncFunctionDef,ast.Global,ast.Nonlocal,ast.Lambda)):
  raise ValueError('unsupported syntax')
 if isinstance(n,ast.Name) and n.id.startswith('__'):raise ValueError('private name')
 if isinstance(n,ast.Attribute) and n.attr not in ('get','append'):raise ValueError('attribute')
 if isinstance(n,ast.Call) and not ((isinstance(n.func,ast.Name) and n.func.id in allowed_calls) or (isinstance(n.func,ast.Attribute) and n.func.attr in ('get','append'))):raise ValueError('call')
ns={'__builtins__':{k:__builtins__.__dict__[k] for k in allowed_calls if k!='candidate'}}
exec(compile(tree,'<candidate>','exec'),ns)
assert all(ns['candidate'](*args)==expected for args,expected in d['tests'])
print('passed')
'''

def json_answer(text):
    text=text.strip()
    if text.startswith('```'):
        text='\n'.join(text.splitlines()[1:-1])
    return json.loads(text)

def task(base, case, temperature=0, tool_budget=2, background=None):
    start=time.monotonic(); rows=[]; result={'case':case['id'],'passed':False}
    cache_salt = uuid.uuid4().hex + uuid.uuid4().hex
    def body(question, maximum):
        p = payload(question, maximum, temperature)
        if background:
            p['messages'].insert(0, {'role':'user','content':'Repository background follows. Treat it as source data for this task.\n'+background})
            p['cache_salt'] = cache_salt
        return p
    try:
        if case['kind']=='repo':
            p=body('Read '+case['path']+' with read_file, then answer this question as JSON only: '+case['question']+' Use this JSON shape: '+json.dumps({k:None for k in case['expected']}),384)
            p.update(tools=[TOOL],tool_choice='required')
            for step in range(tool_budget):
                r=request(base,p,retain_wire=True);rows.append(r)
                calls=list(r['tool_calls'].values())
                if not calls:
                    if step == 0:raise ValueError('required repository read missing')
                    result['passed']=json_answer(r['output'])==case['expected']
                    break
                if len(calls)!=1 or calls[0]['function']['name']!='read_file':raise ValueError('tool protocol')
                args=json.loads(calls[0]['function']['arguments'])
                if args!={'path':case['path']}:raise ValueError('wrong repository path')
                p['messages'] += [{'role':'assistant','content':r['output'] or None,'tool_calls':calls},
                                  {'role':'tool','tool_call_id':calls[0]['id'],'content':FILES[case['path']]}]
                # Preserve the original two-call protocol diagnostic. A larger
                # fixed budget measures natural tool-loop completion and retries.
                p['tool_choice']='none' if tool_budget == 2 else 'auto'
            else:
                result['error']='tool budget exhausted'
            result['model_calls']=len(rows)
        elif case['kind']=='repair':
            r=request(base,body(case['task']+' Return only JSON with a code key containing the Python function. Use no imports.',512));rows.append(r)
            code=json_answer(r['output'])['code']
            check=subprocess.run([sys.executable,'-I','-c',SANDBOX],input=json.dumps({'code':code,'tests':case['tests']}),capture_output=True,text=True,timeout=3)
            result['passed']=check.returncode==0
            result['check_error']=check.stderr[-300:] if check.returncode else None
        else:
            values=case['values']; sections=[]
            for index,value in enumerate(values):
                sections.append(f'Authoritative audit item {index}: {value}.\n')
                sections.extend(f'Unrelated shipment {index}-{i}, quantity {(i*37)%997}.\n' for i in range(350))
            r=request(base,body(''.join(sections)+'\nReturn only JSON with values listing the three authoritative audit items in order and sum containing their total.',256));rows.append(r)
            result['passed']=json_answer(r['output'])=={'values':values,'sum':sum(values)}
        if any(r.get('error') for r in rows):result['passed']=False
    except Exception as exc:
        result['error']=type(exc).__name__+': '+str(exc)[:300]
    result.update(elapsed_s=time.monotonic()-start,requests=rows)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',default='http://100.120.26.16:8888');p.add_argument('--records',type=Path,required=True)
    p.add_argument('--profile',required=True);p.add_argument('--repeats',type=int,default=3);p.add_argument('--concurrencies',default='1,3')
    p.add_argument('--temperature',type=float,default=0)
    p.add_argument('--tool-budget',type=int,default=2)
    p.add_argument('--context-fixtures',type=Path,help='Private fixed repository background: context-32000.txt and context-128000.txt')
    a=p.parse_args();os.umask(0o077)
    if not 2 <= a.tool_budget <= 8:p.error('tool budget must be 2..8')
    backgrounds = {size:(a.context_fixtures/f'context-{size}.txt').read_text() for size in (32000,128000)} if a.context_fixtures else {}
    directory=a.records/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+a.profile+'-tasks-'+uuid.uuid4().hex[:8]);directory.mkdir(parents=True)
    for name in ('task_qualification.py','qualification.py'):
        (directory/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (directory/'manifest.json').write_text(json.dumps({'profile':a.profile,'suite':'tasks','fixtures_sha256':digest({'cases':CASES,'files':FILES}),'repeats':a.repeats,'concurrencies':a.concurrencies,'temperature':a.temperature,'tool_budget':a.tool_budget,'background_sha256':{size:digest(text) for size,text in backgrounds.items()}},indent=2))
    print('RUN_DIRECTORY='+str(directory),flush=True)
    for repeat in range(a.repeats):
        for c in map(int,a.concurrencies.split(',')):
            before=idle(a.base)
            append(directory/'events.jsonl',{'event':'wave_start','repeat':repeat,'concurrency':c,'before':before})
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                futures=[pool.submit(task,a.base,case,a.temperature,a.tool_budget,backgrounds.get(128000 if case['kind']=='synthesis' else 32000)) for case in CASES]
                rows=[]
                for future in concurrent.futures.as_completed(futures):
                    row=future.result();rows.append(row)
                    append(directory/'events.jsonl',{'event':'task_result','repeat':repeat,'concurrency':c,**row})
            after=idle(a.base)
            total=sum((r['usage'] or {}).get('completion_tokens',0) for t in rows for r in t['requests'])
            delta=after['vllm:generation_tokens_total']-before['vllm:generation_tokens_total']
            valid=delta==total and all(not r.get('error') for t in rows for r in t['requests'])
            result={'event':'task_wave','repeat':repeat,'concurrency':c,'valid':valid,'generation_counter_delta':delta,'completion_tokens':total,'passed':sum(t['passed'] for t in rows),'total':len(rows),'tasks':rows}
            append(directory/'events.jsonl',result);print(json.dumps({k:v for k,v in result.items() if k!='tasks'}),flush=True)
            if not valid:
                append(directory/'events.jsonl',{'event':'run_finish','status':'invalid'})
                raise RuntimeError('Contaminated/failed wave retained; use new run ID')
    append(directory/'events.jsonl',{'event':'run_finish','status':'complete'})

if __name__=='__main__':main()
