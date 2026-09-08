#!/usr/bin/env python3
"""Token-counted concurrent long-context semantic validation using longctx fixtures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.request
import uuid
import longctx


def post(base, path, body, timeout=7200):
    return urllib.request.urlopen(urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=timeout)


def payload(model, prompt, maximum):
    return {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": maximum,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}


def token_count(base, body):
    request = {key: body[key] for key in ("model", "messages", "chat_template_kwargs")}
    request["add_generation_prompt"] = True
    with post(base, "/tokenize", request) as response:
        return json.load(response)["count"]


def fixture(args):
    prefix = f"Independent evaluation {uuid.uuid4()}.\n"
    question = "\nReport the alpha, bravo, and charlie secret access codes from this log. Then add the four-digit prefixes before the hyphens in all three codes and report their sum. Answer directly."
    codes = [f"{1000 + secrets.randbelow(9000)}-{uuid.uuid4().hex[:8].upper()}" for _ in longctx.NEEDLES]
    lines = max(10, args.prompt_tokens // 25)
    for _ in range(6):
        log = longctx.build(lines)
        for (_, _, old), new in zip(longctx.NEEDLES, codes):
            log = log.replace(old, new)
        body = payload(args.model, prefix + log + question, args.max_tokens)
        count = token_count(args.base, body)
        if args.prompt_tokens - 100 <= count <= args.prompt_tokens:
            break
        lines = max(10, lines + (args.prompt_tokens - count) // 25)
    if not args.prompt_tokens - 100 <= count <= args.prompt_tokens:
        raise ValueError(f"Fixture failed calibration: {count} vs {args.prompt_tokens}")
    if count + args.max_tokens > args.context_limit:
        raise ValueError("Prompt plus generation reserve exceeds configured context")
    body.update(stream=True, stream_options={"include_usage": True})
    if args.hold_output:
        body.update(min_tokens=args.max_tokens, ignore_eos=True)
    return body, count, codes


def grade(text, codes):
    total = sum(int(code.split("-")[0]) for code in codes)
    return all(value in text for value in codes) and bool(re.search(r"(?<![0-9])" + str(total) + r"(?![0-9])", text.replace(",", "")))


def run(args, body, count, codes, barrier):
    barrier.wait()
    started = time.monotonic()
    started_epoch = time.time()
    first_epoch = None
    first = None
    content = []
    usage = None
    finished = False
    with post(args.base, "/v1/chat/completions", body) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                finished = True
                break
            data = json.loads(line[6:])
            if data.get("error"):
                raise ValueError(data["error"])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece:
                    if first is None:
                        first = time.monotonic()
                        first_epoch = time.time()
                    content.append(piece)
    answer = "".join(content)
    valid = bool(finished and usage and usage["prompt_tokens"] == count and grade(answer[:2000], codes))
    return {"passed": valid, "calibrated_prompt_tokens": count, "usage": usage,
            "started_epoch": started_epoch, "first_token_epoch": first_epoch, "finished_epoch": time.time(),
            "ttft_seconds": first - started if first else None, "elapsed_seconds": time.monotonic() - started,
            "answer": answer[:2000], "expected_codes": codes, "complete_stream": finished}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--prompt-tokens", type=int, default=980000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--context-limit", type=int, default=1000000)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--hold-output", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bodies = [fixture(args) for _ in range(args.concurrency)]
    print(json.dumps({"calibrated": [count for _, count, _ in bodies]}), flush=True)
    barrier = threading.Barrier(args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(run, args, body, count, codes, barrier) for body, count, codes in bodies]
        results = [future.result() for future in futures]
    args.out.write_text(json.dumps({"results": results, "hold_output": args.hold_output}, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    raise SystemExit(0 if all(row["passed"] for row in results) else 1)

if __name__ == "__main__":
    main()
