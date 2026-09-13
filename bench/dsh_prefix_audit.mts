/** Offline metadata audit. Pipe stdout directly into dsh_prefix_tokens.py; it contains transient prompt bodies. No network is permitted. Run through the selected DSH checkout tsx and TSX_TSCONFIG_PATH. */
import fs from 'node:fs';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import pathModule from 'node:path';
const repository=pathModule.resolve(process.argv[2]);
const source=(path:string)=>pathToFileURL(repository+'/'+path).href;
const {foldSurface,deriveEventMessage}=await import(source('packages/core/session/src/surface.ts'));
const {toPiContext}=await import(source('packages/llm/llm-pi-ai/src/context.ts'));
const {stream}=await import(source('packages/llm/llm-pi-ai/node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js'));
const inputs=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
for (const path of inputs) {
 const events=execFileSync('zstd',['-dc',path],{maxBuffer:512*1024*1024}).toString().trim().split('\n').map(JSON.parse).filter(e=>Number.isInteger(e.seq));
 let header:any;const selected:any[]=[];
 for(let i=0;i<events.length;i++) {
  const e=events[i];if(e.type==='request/header')header=e.data.header;
  if(e.type==='assistant/message'&&header?.config?.provider?.startsWith('john')&&header.config.model==='qwen3.8-flash-next')selected.push({i,header});
 }
 for(const {i,header} of selected.slice(-20)) {
  const event=events[i];
  try {
   const preceding=events.slice(0,i);const {nodes}=foldSurface(preceding);
   const messages=nodes.map(seq=>deriveEventMessage(preceding[seq])).filter(Boolean);
   const context=toPiContext({...header.config,messages,tools:header.tools});
   const model={id:header.config.model,provider:header.config.provider,api:'openai-completions',baseUrl:'http://127.0.0.1:1/v1',reasoning:true,input:['text','image'],contextWindow:262144,maxTokens:32768,cost:{input:0,output:0,cacheRead:0,cacheWrite:0},compat:{thinkingFormat:'chat-template',chatTemplateKwargs:{enable_thinking:{$var:'thinking.enabled'},preserve_thinking:true,reasoning_effort:{$var:'thinking.effort',omitWhenOff:true}}},thinkingLevelMap:{low:'low',high:'xhigh',xhigh:'xhigh'}};
   let body:any;
   const output=stream(model,context,{apiKey:'unused-offline',reasoningEffort:header.config.reasoningEffort==='off'?undefined:header.config.reasoningEffort,fetch:async()=>{throw Error('Offline audit forbids network')},onPayload:p=>{body=p;throw Error('Offline serialization complete')}});
   await output.result();if(!body)throw Error('Wire serialization failed');
   process.stdout.write(JSON.stringify({session:path.split('/').slice(-2,-1)[0],seq:event.seq,turn:event.data.turn,step:event.data.step,usage:event.data.usage,body})+'\n');
  } catch(error) {process.stdout.write(JSON.stringify({session:path.split('/').slice(-2,-1)[0],seq:event.seq,error:(error instanceof Error ? error.name : 'Error')+': offline reconstruction failed'})+'\n');}
 }
}
