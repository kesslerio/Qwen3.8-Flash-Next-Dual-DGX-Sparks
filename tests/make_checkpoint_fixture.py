#!/usr/bin/env python3
"""Create tiny synthetic BF16 shards for CPU-only converter integration tests."""
import hashlib
import json
from pathlib import Path
import struct
import sys

root=Path(sys.argv[1])
snapshot=root/'hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594'
snapshot.mkdir(parents=True)
weights={}
for index,names in enumerate([
    ['model.language_model.layers.3.self_attn.q_proj.weight','mtp.layers.0.self_attn.q_proj.weight','lm_head.weight'],
    ['model.language_model.norm.weight'],
]):
    filename=f'model-bf16-{index:05d}-of-00002.safetensors'
    header={};data=b''
    for name in names:
        block=struct.pack('<4H',0x3f80,0x3f00,0xbf40,0x0000)*128
        header[name]={'dtype':'BF16','shape':[4,128],'data_offsets':[len(data),len(data)+len(block)]}
        data+=block;weights[name]=filename
    encoded=json.dumps(header).encode();encoded+=b' '*((-len(encoded))%8)
    (snapshot/filename).write_bytes(struct.pack('<Q',len(encoded))+encoded+data)
(snapshot/'config.json').write_text(json.dumps({'text_config':{'num_hidden_layers':48}}))
(snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':weights}))
(root/'source-proof.json').write_text(json.dumps({p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'uid':p.stat().st_uid,'mode':p.stat().st_mode} for p in snapshot.iterdir()},indent=2))
print(snapshot)
