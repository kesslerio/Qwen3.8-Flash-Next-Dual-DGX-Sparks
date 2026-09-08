#!/usr/bin/env python3
"""Record bounded runtime evidence without sending inference work."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 1 or args.interval < 1:
        parser.error("duration and interval must be positive")
    deadline = time.monotonic() + args.seconds
    wanted = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
              "num_preemptions_total", "generation_tokens_total", "prompt_tokens_total",
              "spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total")
    with args.out.open("a", buffering=1) as output:
        while time.monotonic() < deadline:
            row = {"time": time.time()}
            try:
                with urllib.request.urlopen(args.base + "/metrics", timeout=10) as response:
                    body = response.read().decode()
                row["metrics"] = [line for line in body.splitlines()
                                  if not line.startswith("#") and any(line.startswith("vllm:" + name) for name in wanted)]
            except (OSError, ValueError) as error:
                row["error"] = str(error)
            output.write(json.dumps(row) + "\n")
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
