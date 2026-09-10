## Current 500K deployment (2026-09-08)

The user selected 500,000 total tokens instead of the original 1M target.
The live NVFP4/MTP3 server now advertises 500,000, with YaRN factor 2,
three active sequences, 4,304,597 shared KV tokens (counted once across TP2),
34.08 GiB KV memory per node, and 65.54 GiB model/draft allocation.
This is configured capacity; three simultaneous near-limit requests have
not yet been qualified. Native-context speed results below are not final
500K-profile benchmarks.

An upstream configuration mismatch initially blocked the 1M launch:
dictionary HF overrides were target-only, leaving the same-checkpoint MTP
draft at 262144 while the target used extended rotary scaling. A scoped,
regenerated speculative-config overlay now propagates Qwen Flash YaRN to
the draft. The pinned runtime configuration preflight verifies identical
target/draft limits and rotary parameters without enabling prefix caching.
The corrected 500K server reached healthy API readiness.

DSH web/headless configurations on Mac, mama, and theshop now contain
qwen3.8-flash-next with 500000 context. Fresh native web catalogs and real
LAN web tool cycles passed on all three hosts. Remote web and headless tool cycles also passed on all three hosts.
The near-limit semantic check passed with 479,986 actual prompt tokens,
123 output tokens, correct early/middle/late fact retrieval and synthesis.
Its 238.05-second TTFT included concurrent DSH acceptance traffic and is
not an isolated speed benchmark. Other-client cutover,
recovery/rollback drills and final soak remain incomplete.

# John / Ofus NVFP4 qualification

This report is incomplete until the final NVFP4 runtime, context, throughput and recovery evidence is recorded. Prepared configuration is not deployment proof.

## Before migration (2026-09-07)

Official FP8 model, native 262144 context, TP2+EP, FP8 KV, MTP off, six active sequences, memory utilization 0.835. Shared KV pool: 1485999 tokens. The user's completed SparkDash job used 2048 output tokens, structural prompts, temperature zero, thinking disabled.

| Active streams | Aggregate tok/s | Median per-stream tok/s |
| --- | --- | --- |
| 1 | 23.08 | 23.08 |
| 2 | 41.87 | 21.07 |
| 3 | 53.35 | 17.83 |
| 4 | 63.91 | 16.02 |
| 5 | 68.14 | 13.73 |
| 6 | 75.70 | 12.64 |

Baseline checks passed: 12/12 reasoning/retrieval tasks; streamed multiply tool round-trip; synthetic red/blue image; synthetic red video. A randomized 119979-token prompt retrieved facts at three depths and correctly summed their four-digit prefixes. Response usage matched the tokenized input count exactly; first content arrived after 46.03 seconds. These quality checks ran during checkpoint staging, so their latency is informational rather than a controlled performance result.

Raw evidence is on john under `~/.local/state/qwen-migration/20260907/`, including the original SparkDash job and baseline responses. The current FP8 launch source, environment, image identity, container inspections and overlays are retained there for rollback.

## Qualification commands

Reuse `bench/sparkdash_compare.py` against the installed dashboard for matching prompts, completion-token accounting and UI results. Run one warmup, then three recorded repetitions at matching concurrency, with no competing inference or checkpoint download. Record MTP acceptance-counter deltas using `bench/mtp_accept.py` around each run.

Use `bench/reasoning_check.py` with baseline comparison, `bench/agent_contract.py` for actual streaming/tool/image behavior, and `bench/context_capacity.py` for tokenizer-counted long inputs with randomized planted facts and synthesis. Near-1M validation uses approximately 980000 input tokens plus 8192 output reserve. Concurrent capacity checks use `--hold-output` to retain long contexts during generation. Separately observe peak cache occupancy and preemption counters; completed requests alone do not prove their full contexts coexisted.

## Native NVFP4 MTP3: preliminary live readback

The first candidate started under the persistent supervisor on 2026-09-07. Both ranks resolve the pinned image, and the API reports `RadixArk/Qwen3.8-Flash-Next-NVFP4` at the diagnostic 262144 limit. Runtime logs select `FLASHINFER_CUTLASS` for NVFP4 experts, runtime FP8 PLE embeddings, and an unquantized MTP draft component.

Model loading reports 65.41 GiB per rank, versus the FP8 baseline's 87.83 GiB. Available KV memory is 33.53 GiB; the shared TP pool is **3913624 tokens**, counted once. This native-context allocation does not qualify the 1M profile by itself. All 12 reasoning/retrieval checks passed with no baseline regressions, including the 104088-token retrieval fixture. The streamed tool round trip, red/blue image check, and red-video check passed. Three recorded speed repetitions and the MTP1/MTP0 controls are underway; preliminary runs are not final-profile qualification.

## Native MTP3 completed comparison

Three recorded warm repetitions completed at every concurrency without failed streams. Matched SparkDash settings: 2048 completion tokens, structural prompts, temperature zero, thinking disabled. These measurements use native 262144 context, not the final extended profile.

| Active streams | Median aggregate tok/s | Median per-stream tok/s | Median TTFT ms |
| --- | --- | --- | --- |
| 1 | 56.55 | 56.55 | 134.42 |
| 2 | 78.13 | 45.08 | 227.77 |
| 3 | 100.93 | 35.56 | 295.64 |
| 4 | 122.51 | 33.95 | 266.77 |
| 5 | 141.92 | 32.96 | 292.81 |
| 6 | 163.72 | 31.43 | 269.58 |

The measurement window recorded 129024 generation tokens, 124749 draft tokens and 87460 accepted draft tokens: 70.1% acceptance, 2.103 accepted tokens per draft. Native MTP1 and MTP0 controls remain pending.

## Prefix-cache inspection

Prefix caching remains disabled in this migration. Inspection of the pinned running image confirms both reported faults are present: MambaManager accepts but never uses drop_eagle_block, and the state-resume divisor uses cache_config.block_size. The manager also has a partial-unit search, so the max_length variant of the first fix is required.

The [reporter's replicated fix](https://github.com/vllm-project/vllm/issues/54173#issuecomment-5450408591) uses the same image and checkpoint but a single GPU with PLE offload and automatic KV dtype. It does not qualify the two-rank, FP8-KV, resident-PLE configuration here. Enabling caching requires both fixes plus varied-length growing-history, block-boundary, concurrency, cancellation, and semantic recall tests. Neither identical-prompt speedups nor a healthy startup are sufficient.

## Final result

Pending. Final-profile speed, KV capacity, and 1M-context qualification remain incomplete.

## Benchmark isolation

The first recorded MTP1 attempt (`f3fa5b10-e587-42d8-a508-cadcbe344fae`) was excluded: API access logs show concurrent inference from mama during its low-concurrency waves. Neither GPU showed active thermal throttling during the diagnostic sample; memory-pressure averages were zero. The interrupted attempt is retained privately as contaminated evidence and is not used for selection. Subsequent comparisons use `--exclusive-base http://localhost:8888`, requiring settled idle boundaries and an exact match between server generation-counter growth and the benchmark completion-token total for every repetition. A mismatch fails qualification. Native MTP3 already had an exact 129024-token match over its three recorded repetitions.
