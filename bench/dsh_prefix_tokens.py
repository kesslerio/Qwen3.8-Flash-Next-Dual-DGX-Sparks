#!/usr/bin/env python3
"""Consume transient wire prompts on stdin; persist token counts/hashes only.

Argument: private root containing tokenizer/{tokenizer,tokenizer_config}.json.
Requires tokenizers==0.23.2 and jinja2==3.1.6. Uses verified vLLM role and
tool-argument normalization; exact token-count matches remain an explicit gate.
"""
import json,sys,hashlib
from pathlib import Path
import tokenizers
from jinja2.sandbox import ImmutableSandboxedEnvironment
root=Path(sys.argv[1]);config=json.loads((root/'tokenizer/tokenizer_config.json').read_text());tok=tokenizers.Tokenizer.from_file(str(root/'tokenizer/tokenizer.json'))
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
env.filters['tojson']=lambda x,**kw:json.dumps(x,ensure_ascii=False,**kw)
def fail(message):raise ValueError(message)
env.globals['raise_exception']=fail
template=env.from_string(config['chat_template']);previous={}
def hash_(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
for line in sys.stdin:
 row=json.loads(line)
 if 'error' in row:print(json.dumps(row),flush=True);continue
 try:
  body=row.pop('body')
  for message in body['messages']:
   if message['role']=='developer':message['role']='system'
   for call in message.get('tool_calls') or []:
    if isinstance(call['function'].get('arguments'),str):call['function']['arguments']=json.loads(call['function']['arguments'])
  text=template.render(messages=body['messages'],tools=body.get('tools'),add_generation_prompt=True,**body.get('chat_template_kwargs',{}))
  ids=tok.encode(text,add_special_tokens=False).ids;old=previous.get(row['session']);count=0
  if old:
   for a,b in zip(old['ids'],ids):
    if a!=b:break
    count+=1
  usage=row.pop('usage') or {};expected=sum(usage.get(k,0) for k in ('inputTokens','cacheReadTokens','cacheWriteTokens'))
  row.update(rendered_tokens=len(ids),reported_prompt_tokens=expected,token_count_matches=len(ids)==expected,cached_tokens=usage.get('cacheReadTokens'),prefix_overlap_tokens=count if old else None,previous_prompt_tokens=len(old['ids']) if old else None,tool_schema_sha256=hash_(body.get('tools')),tool_schema_changed=hash_(body.get('tools'))!=old['tools'] if old else None)
  previous[row['session']]={'ids':ids,'tools':hash_(body.get('tools'))}
  print(json.dumps(row),flush=True)
 except Exception as exc:print(json.dumps({'session':row['session'],'seq':row['seq'],'error':type(exc).__name__+': offline tokenization failed','roles':sorted({m['role'] for m in body['messages']})}),flush=True)
