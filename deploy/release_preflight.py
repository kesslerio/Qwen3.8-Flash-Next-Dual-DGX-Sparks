#!/usr/bin/env python3
"""Validate the pinned HF inventory and disk budget without loading a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def check_space(free, missing, reserve):
    if free < missing + reserve:
        raise ValueError(f"Storage deficit: {missing + reserve - free:,} bytes; current service must stay running")


def verify_files(snapshot, manifest, hashes=False):
    total = 0
    for entry in manifest["siblings"]:
        path = snapshot / entry["rfilename"]
        size = entry["size"]
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or wrong-size checkpoint file: {path.name}")
        expected_hash = entry.get("lfs", {}).get("sha256")
        if hashes and expected_hash:
            with path.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"Checksum mismatch: {path.name}")
        total += size
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        files = {entry["rfilename"] for entry in manifest["siblings"]}
        for shard in set(json.loads(index.read_text())["weight_map"].values()):
            if shard not in files or not (snapshot / shard).is_file():
                raise ValueError(f"Unverified indexed shard: {shard}")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--space-only", action="store_true")
    parser.add_argument("--hashes", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.snapshot.name != manifest["sha"]:
        raise SystemExit("Snapshot path must match the pinned HF revision")
    if args.space_only:
        existing = args.snapshot
        while not existing.exists():
            existing = existing.parent
        missing = sum(f["size"] for f in manifest["siblings"]
                      if not (args.snapshot / f["rfilename"]).is_file()
                      or (args.snapshot / f["rfilename"]).stat().st_size != f["size"])
        check_space(shutil.disk_usage(existing).free, missing, 20 * 1024**3)
        print(json.dumps({"missing_bytes": missing, "reserve_bytes": 20 * 1024**3}))
    else:
        print(json.dumps({"revision": manifest["sha"], "verified_bytes": verify_files(args.snapshot, manifest, args.hashes)}))

if __name__ == "__main__":
    main()
