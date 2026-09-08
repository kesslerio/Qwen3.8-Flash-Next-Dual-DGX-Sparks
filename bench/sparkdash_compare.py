#!/usr/bin/env python3
"""Run the installed SparkDash benchmark so prompts and timing match its UI."""
import argparse
import json
from pathlib import Path
import statistics
import time
import urllib.request


def idle_generation_count(base):
    """Wait for settled idle metrics and return the accepted-output counter."""
    deadline = time.monotonic() + 300
    previous = None
    while time.monotonic() < deadline:
        with urllib.request.urlopen(base + "/metrics", timeout=15) as response:
            lines = response.read().decode().splitlines()
        def metric(name):
            values = [float(line.rsplit(" ", 1)[1]) for line in lines
                      if line.startswith("vllm:" + name + "{")]
            if not values:
                raise ValueError("Required metric absent: " + name)
            return sum(values)
        count = metric("generation_tokens_total")
        idle = metric("num_requests_running") == metric("num_requests_waiting") == 0
        if idle and previous == count:
            return count
        previous = count if idle else None
        time.sleep(5)
    raise TimeoutError("Cluster did not become idle; another caller may be active")


def request(url, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method=method), timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default="http://john:5556")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--concurrencies", default="1,2,3")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclusive-base", help="Reject runs with output tokens from competing inference")
    args = parser.parse_args()
    endpoint = args.dashboard + "/api/sparks/john/llm/bench"
    runs = []
    for repetition in range(args.repetitions):
        before = idle_generation_count(args.exclusive_base) if args.exclusive_base else None
        job = request(endpoint, {"port": 8888, "modelId": args.model, "concurrencies": [int(x) for x in args.concurrencies.split(",")], "maxTokens": args.max_tokens})
        print(json.dumps({"repetition": repetition + 1, "benchId": job["benchId"]}), flush=True)
        try:
            deadline = time.monotonic() + 3600
            while job["status"] not in ("completed", "failed", "cancelled", "interrupted"):
                if time.monotonic() > deadline:
                    raise TimeoutError("SparkDash job exceeded one hour; inspect and cancel this exact benchId")
                time.sleep(5)
                job = request(endpoint + "/" + job["benchId"])
        except BaseException:
            # Cancel only the benchmark this invocation created.
            try:
                request(endpoint + "/" + job["benchId"], method="DELETE")
            except Exception as error:
                print(f"Could not cancel {job['benchId']}: {error}", flush=True)
            raise
        runs.append(job)
        args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
        if job["status"] != "completed" or any(wave.get("streamsFailed", 0) for wave in job["results"]):
            raise ValueError("Incomplete benchmark; inspect saved result")
        if args.exclusive_base:
            after = idle_generation_count(args.exclusive_base)
            expected = sum(wave["totalCompletionTokens"] for wave in job["results"])
            job["exclusive_counter_check"] = {"before": before, "after": after, "expected": expected,
                                              "passed": after - before == expected}
            args.out.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
            if after - before != expected:
                raise ValueError("Competing inference contaminated benchmark; saved run is excluded")
        print(json.dumps({"repetition": repetition + 1, "waves": [{k: wave.get(k) for k in ("concurrency", "aggregateDecodeTps", "medianDecodeTps", "medianTtftMs", "streamsOk")} for wave in job["results"]]}), flush=True)
    summary = {}
    for concurrency in [int(x) for x in args.concurrencies.split(",")]:
        waves = [wave for run in runs for wave in run["results"] if wave["concurrency"] == concurrency]
        summary[concurrency] = {"aggregate_tps_median": statistics.median(w["aggregateDecodeTps"] for w in waves), "per_stream_tps_median": statistics.median(w["medianDecodeTps"] for w in waves), "ttft_ms_median": statistics.median(w["medianTtftMs"] for w in waves)}
    args.out.write_text(json.dumps({"runs": runs, "summary": summary}, indent=2) + "\n")
    print(json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
