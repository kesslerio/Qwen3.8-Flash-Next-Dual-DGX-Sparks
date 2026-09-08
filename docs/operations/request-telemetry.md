# Request performance history

`QWEN_REQUEST_TELEMETRY=true` loads a bounded ASGI observer and enables vLLM's
per-request timing fields, cached-token details and final usage chunks. The observer forwards requests
and responses unchanged. It records token counts, cache reuse when supplied by
vLLM, queue time, first progress, generation rate, reasoning settings, tool-call
presence, completion status and interruptions. A client address is represented
by a stable hash; prompt text, response text, arguments, credentials and headers
are never written by the observer.

First progress includes reasoning or tool output and is distinct from the first
visible answer token. HTTP 200 alone does not establish stream success. A tool
call being present does not establish that the tool succeeded or the answer was
correct. DSH's own session trace remains the source for that investigation.

The API emits `QWEN_REQUEST_START` and `QWEN_REQUEST` records into container logs.
Both rank containers retain at most five 20 MB log files. The head's
`qwen-request-history.service` copies only allowlisted request records into an
owner-only SQLite database under `~/.local/state/qwen-cluster/`, deduplicates
poll overlap and retains seven days across model restarts. Keep this collector
running independently of inference. It does not restart or call the model.

Install the checked-in history unit on the head, reload systemd and enable it:

```sh
sudo install -m 644 deploy/systemd/qwen-request-history.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-request-history.service
python3 deploy/request_history.py --report-hours 1
```

The report groups by client hash and reports the sample count for each field.
Absent metrics remain unknown, not zero. History starts when instrumentation
was deployed; it cannot retrospectively attribute old aggregate counters.
This complements SparkDash's seven-day aggregate history. Avoid interpreting
process-lifetime latency percentiles as percentiles for the last hour.

Timing definitions follow the pinned vLLM source: `time_to_first_token_ms` starts
at scheduling and excludes queue time; `first_progress_ms` includes the wait
observed by the API middleware. Native `tokens_per_second` includes prefill but
excludes queue time. `decode_tokens_per_second` is the reciprocal of mean
inter-token latency; `end_to_end_tokens_per_second` includes the entire observed
request. Compare like metrics rather than treating all three as decode speed.
