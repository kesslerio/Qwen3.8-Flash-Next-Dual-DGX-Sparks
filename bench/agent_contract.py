#!/usr/bin/env python3
"""Exercise streamed tool calls and a synthetic image through the actual API."""
import argparse
import base64
import json
import struct
import urllib.request
import zlib


def post(base, payload):
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=300)


def streamed(base, payload):
    content, calls, usage = [], {}, None
    with post(base, dict(payload, stream=True, stream_options={"include_usage": True})) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("error"):
                raise ValueError(data["error"])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                content.append(delta.get("content") or "")
                for part in delta.get("tool_calls", []):
                    call = calls.setdefault(part["index"], {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if part.get("id"):
                        call["id"] = part["id"]
                    for key in ("name", "arguments"):
                        call["function"][key] += part.get("function", {}).get(key) or ""
    return "".join(content), list(calls.values()), usage


def two_color_png():
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data) & 0xffffffff)
    row = b"\0" + b"\xff\0\0" * 64 + b"\0\0\xff" * 64
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 128, 128, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(row * 128)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    common = {"model": args.model, "temperature": 0, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}}
    messages = [{"role": "user", "content": "Use multiply to calculate 3 times 7. After the tool returns, answer with just the result."}]
    tool = {"type": "function", "function": {"name": "multiply", "description": "Multiply two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}
    text, calls, usage = streamed(args.base, dict(common, messages=messages, tools=[tool], tool_choice="required"))
    assert len(calls) == 1 and calls[0]["function"]["name"] == "multiply", calls
    values = json.loads(calls[0]["function"]["arguments"])
    assert values == {"a": 3, "b": 7}, values
    messages += [{"role": "assistant", "content": text or None, "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "21"}]
    answer, extra, final_usage = streamed(args.base, dict(common, messages=messages, tools=[tool], tool_choice="none"))
    assert answer.strip().strip(".") == "21" and not extra, answer
    image_answer, _, image_usage = streamed(args.base, dict(common, messages=[{"role": "user", "content": [{"type": "text", "text": "Name the two solid colors, left then right. Answer with only the color names."}, {"type": "image_url", "image_url": {"url": two_color_png()}}]}]))
    colors = image_answer.lower()
    assert "red" in colors and "blue" in colors and colors.index("red") < colors.index("blue"), image_answer
    result = {"tool_round_trip": True, "answer": answer, "image_answer": image_answer, "usage": [usage, final_usage, image_usage]}
    with open(args.out, "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
