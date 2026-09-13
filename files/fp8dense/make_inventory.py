#!/usr/bin/env python3
"""Record a derived local checkpoint inventory after conversion verification."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--source-revision', required=True)
    p.add_argument('--kind', choices=('draft-only','target-fp8'), required=True)
    p.add_argument('--identity-proof', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Inventory already exists; retain the previous build')
    proof=json.loads(a.identity_proof.read_text()) if a.identity_proof else None
    if a.kind == 'draft-only' and (not proof or proof.get('target_tensors_verified',0) <= 0):
        p.error('Draft-only build requires target identity proof')
    siblings=[]
    for path in sorted(a.snapshot.iterdir()):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
            siblings.append({'rfilename':path.name,'size':path.stat().st_size,'lfs':{'sha256':h.hexdigest()}})
    result={'repo_id':a.model,'model':a.model,'sha':a.revision,'source_revision':a.source_revision,
        'kind':a.kind,'created_utc':time.time(),'siblings':siblings,'target_identity_verified':bool(proof),
        'target_identity_proof':proof,'quality_qualified':False,
        'build_source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}}
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'inventory':str(a.output),'files':len(siblings),'kind':a.kind,'quality_qualified':False}))


if __name__=='__main__':main()
