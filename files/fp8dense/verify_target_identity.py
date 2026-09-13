#!/usr/bin/env python3
"""Prove every non-MTP tensor is byte-identical in a draft-only checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

def header(path):
    with path.open('rb') as f:
        length=struct.unpack('<Q',f.read(8))[0]
        return json.loads(f.read(length)), 8+length

def tensor_hash(path, offset, length):
    h=hashlib.sha256()
    with path.open('rb') as f:
        f.seek(offset)
        while length:
            block=f.read(min(length,4*1024*1024))
            if not block:raise ValueError('Truncated tensor')
            h.update(block);length-=len(block)
    return h.hexdigest()

def verify(source, destination):
    maps=[json.loads((root/'model.safetensors.index.json').read_text())['weight_map'] for root in (source,destination)]
    headers={};count=0;hashed=0
    for name, shard in maps[0].items():
        if name.startswith(('mtp.','model.mtp.')):continue
        if name not in maps[1]:raise ValueError('Missing target tensor '+name)
        a,b=source/shard,destination/maps[1][name]
        if not os.path.samefile(a,b):
            entries=[]
            for path in (a,b):
                if path not in headers:headers[path]=header(path)
                h,start=headers[path];entry=h[name];entries.append((entry,start))
            if any(entries[0][0][k]!=entries[1][0][k] for k in ('dtype','shape')):
                raise ValueError('Target dtype/shape changed: '+name)
            hashes=[]
            for path,(entry,start) in zip((a,b),entries):
                begin,end=entry['data_offsets'];hashes.append(tensor_hash(path,start+begin,end-begin))
            if hashes[0]!=hashes[1]:raise ValueError('Target bytes changed: '+name)
            hashed+=1
        count+=1
    extras=[name for name in maps[1] if name not in maps[0] and not name.startswith(('mtp.','model.mtp.'))]
    if extras:raise ValueError('Unexpected target tensors: '+str(extras[:3]))
    return {'target_tensors_verified':count,'target_tensors_hashed':hashed,'target_tensors_shared_inode':count-hashed}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--src',type=Path,required=True);p.add_argument('--dst',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.src,a.dst)))
