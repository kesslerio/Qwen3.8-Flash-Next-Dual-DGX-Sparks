#!/usr/bin/env python3
"""Verify actual streamed reasoning on/off, compatible efforts and final answers."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

def probe(base, model, effort):
    body = {"model": model, "messages": [{"role": "user", "content": "Calculate 187 multiplied by 43. Give only the final integer in your answer."}],
            "reasoning_effort": effort, "chat_template_kwargs": {"enable_thinking": effort != "none", "preserve_thinking": True},
            "temperature": 0, "max_tokens": 8192, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    answer, reasoning, usage, done = [], [], None, False
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "): continue
            if line == "data: [DONE]":
                done = True
                break
            data = json.loads(line[6:])
            if data.get("error"): raise ValueError(data["error"])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                answer.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or delta.get("reasoning_content") or "")
    text = "".join(answer).strip()
    reasoning_chars = len("".join(reasoning))
    passed = bool(done and usage and "8041" in text.replace(",", "") and ((reasoning_chars > 0) == (effort != "none")))
    return {"effort": effort, "passed": passed, "reasoning_chars": reasoning_chars, "answer": text,
            "usage": usage, "complete_stream": done, "seconds": time.monotonic() - started}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [probe(args.base, args.model, effort) for effort in ("none", "low", "xhigh")]
    args.out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    print(json.dumps(results), flush=True)
    raise SystemExit(0 if all(r["passed"] for r in results) else 1)

if __name__ == "__main__": main()
