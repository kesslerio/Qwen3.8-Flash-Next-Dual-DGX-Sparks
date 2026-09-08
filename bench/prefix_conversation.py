#!/usr/bin/env python3
"""Bounded growing-prefix recall and cache-reuse check on the actual service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import time
import urllib.request
import uuid
import context_capacity as capacity


def request(base, body, expected):
    start=time.monotonic(); first=None; text=[]; usage=None; metrics=None; done=False
    with capacity.post(base,'/v1/chat/completions',body,timeout=600) as response:
        for raw in response:
            line=raw.decode().strip()
            if line=='data: [DONE]': done=True;break
            if not line.startswith('data: '): continue
            data=json.loads(line[6:])
            if data.get('error'): raise RuntimeError('Inference returned a stream error')
            usage=data.get('usage') or usage
            metrics=data.get('metrics') or metrics
            for choice in data.get('choices',[]):
                piece=choice.get('delta',{}).get('content') or ''
                if piece:
                    if first is None:first=time.monotonic()-start
                    text.append(piece)
    answer=''.join(text)
    return {'passed':done and all(code in answer for code in expected), 'ttft_seconds':first,
            'elapsed_seconds':time.monotonic()-start,'usage':usage,'metrics':metrics,
            'complete_stream':done}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',default='http://localhost:8888')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prompt-tokens',type=int,default=50000)
    args=parser.parse_args()
    config=argparse.Namespace(base=args.base,model='qwen3.8-flash-next',prompt_tokens=args.prompt_tokens,max_tokens=256,context_limit=500000,hold_output=False)
    body,count,codes=capacity.fixture(config)
    body['max_tokens']=256
    original=body['messages'][0]['content']
    results=[]
    # Each extension changes the prefix's tail and queries older planted values.
    # This exercises the cache resume that identical-prompt tests miss.
    extra=[]
    for turn in range(8):
        code=str(10000+secrets.randbelow(90000))+'-'+uuid.uuid4().hex[:6]
        extra.append(code)
        original += '\n' + ('A later audit entry provides background details. '* (11+turn*23)) + f'\nTurn {turn} audit code: {code}.\n'
        query='\nReturn the alpha, bravo, and charlie access codes, followed by all turn audit codes, exactly as written. No other text.'
        body['messages'][0]['content']=original+query
        result=request(args.base,body,codes+extra)
        result['turn']=turn
        results.append(result)
        args.out.write_text(json.dumps({'results':results},indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if not result['passed']:raise SystemExit('Growing-prefix semantic check failed')
    # Concurrent extensions of the same cache exercise state isolation.
    bodies=[]
    for i in range(6):
        code='BRANCH-'+uuid.uuid4().hex[:10]
        data=json.loads(json.dumps(body))
        data['messages'][0]['content']=original+f'\nBranch code: {code}.\nReturn the alpha access code and this branch code exactly.'
        bodies.append((data,[codes[0],code]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent=list(pool.map(lambda item:request(args.base,*item),bodies))
    args.out.write_text(json.dumps({'results':results,'concurrent':concurrent},indent=2)+'\n')
    print(json.dumps({'concurrent':concurrent}),flush=True)
    raise SystemExit(0 if all(row['passed'] for row in concurrent) else 1)


if __name__=='__main__':main()
