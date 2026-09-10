# Conversation latency qualification, 2026-09-08

The earlier short native-context MTP comparison did not establish performance
for long agent conversations. Real traffic exposed repeated prefill costs while
prefix caching was disabled.

## Before the change

For the eight-hour window ending about 15:07 UTC, 340 completed requests
processed 53,202,691 prompt tokens and generated 505,072 output tokens: about
17.5 aggregate output tokens/second across wall time. Requests waited in 78.3%
of samples, with a median queue of six. Median KV usage was 11.69%, with zero
preemptions. These are aggregate observations, not request latency percentiles.

Engine logs showed 1–3 aggregate decode tokens/second during long prefills and
90–100 afterward. Connections on the busiest client traced to DSH. The evidence
supports repeated long prefills and admission queueing as the main latency
explanation; it does not describe every request or all clients equally.

## Cache acceptance

The successful candidate preserved three active slots, an 8192-token batch
budget, the pinned NVFP4 checkpoint/image, MTP3, FP8 KV and 500000 context. It
added both guarded Mamba cache fixes and `align` prefix caching on both ranks.
Startup allocated 4,134,831 shared KV tokens and 34.25 GiB KV memory on the head.
This allocation is not proof of eight simultaneous near-limit sessions.

| Check | Result |
| --- | --- |
| Growing conversation, 50K to 56K prompt tokens | 8/8 recall checks passed |
| First request TTFT | 19.61 seconds |
| Later-turn TTFT | median 5.65 seconds; range 4.13–7.41 |
| Concurrent extensions, six submitted with three active slots | 6/6 recall checks passed |
| 479987-token retrieval and arithmetic | passed; 237.66 seconds TTFT |
| KV preemptions during acceptance | zero |

The roughly 3.5x TTFT difference is initial versus reused context in this
synthetic conversation, not a general production speedup or matched comparison
against the earlier uncached service. A new near-limit prompt remains expensive.
The six concurrent branches had 10–29 second client TTFT; this is a correctness
check, not a concurrency speed claim.

An initial candidate also changed admission to six and the batch budget to
2048. It failed in GDN Triton warmup before any request. Restoring the original
configuration produced a correct response; preserving those scheduler sizes
then allowed the cache candidate to start and pass the checks above. The exact
cause of the first startup failure remains unproven.

## Telemetry and remaining limits

The request observer records metadata only and the independent SQLite collector
retains seven days. Live acceptance exposed suppressed named-logger output; the
observer now writes its allowlisted records directly to the rotated container
log. Regression tests cover logging suppression and response passthrough.
Native vLLM throughput includes prefill but excludes queue time. The report
also computes separate decode-only and full-request rates; see
[request timing definitions](../operations/request-telemetry.md).

The final service became active at about 17:01 UTC with 4,050,561 shared KV
tokens. Streamed tool use, a tool-result round trip, image recognition and native
DSH file-tool cycles on Mac, mama and theshop passed. The retained database
contained 62 completed records across four client hashes at the next readback,
including queue, decode and cache fields. A 10K repeat had no cache hit; the
50K growing conversation recorded 35200–44800 cached tokens, so reuse must be
measured rather than assumed for every prompt size.

The final 50K run passed 8/8 sequential checks (18.0 seconds initial TTFT,
4.77 seconds later-turn median), but only 4/6 concurrent branches passed exact
recall. A diagnostic rerun passed 8/8 sequential and 6/6 concurrent checks
(17.05 seconds initial, 4.79 seconds later median). The initial failed run did
not retain answers, so it cannot establish whether cache state or ordinary
model recall caused those misses. A matched follow-up passed all six cached branches and five of six cold
branches. Each cached branch reused 44800 tokens; unique cache salts gave all
cold branches zero reused tokens. Median concurrent TTFT was 15.89 seconds
cached versus 66.48 seconds cold, in sequentially run batches with three active
slots. The cold miss was an explicit refusal to repeat the synthetic access
codes, demonstrating that this fixture can fail without cache reuse. It does
not retroactively explain the two unrecorded misses. Cache contamination was
not reproduced in either diagnostic run; production answer quality and mixed
workload performance remain separate ongoing qualification limits.
A subsequent reload failed during CUDA graph capture after reading persisted
FlashInfer tuning choices. Moving those choices aside allowed cold tuning and
graph capture to succeed. The launcher now uses an unmounted per-container
FlashInfer tuning directory, preserving compiled kernels but refreshing tuning
choices on each new container. This is a mitigation supported by that cold
startup, not a proven root cause or upstream fix. During deployment, replacing
the launcher in place while its shell was running caused a separate shell read
error and restart after that successful boot. Future launcher replacement must
be atomic or occur after the launcher exits.

Thirty-one focused tests passed. The requested Luna max/Gemini medium panel found no
actionable P0 issues in the initial bundle, separate logging
correction, startup mitigation, or stream-close reporting correction. The correction also passed its targeted regression test and the
complete focused suite. A long mixed-use
soak, simultaneous near-limit capacity and reboot/failure-injection remain
unqualified. Tool presence is not proof of tool success or answer quality.

Commands: `python3 -m unittest discover -s tests -q`, `bash -n start.sh`,
`python3 bench/prefix_conversation.py --out <private-result.json>`, and
`python3 bench/context_capacity.py --prompt-tokens 480000 --context-limit 500000
--max-tokens 256 --concurrency 1 --out <private-result.json>`.

Upstream references: [cache fixes and replication](https://github.com/vllm-project/vllm/issues/54173#issuecomment-5450408591),
[matching Triton symptom](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/issues/46).
The snapshot-selection and weight-integrity additions were reviewed; this
profile already pins the exact revision and uses its existing inventory
preflight. NVIDIA-specific MTP metadata and unvalidated draft quantization were
not needed for the pinned RadixArk checkpoint.
