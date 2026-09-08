# John / Ofus deployment

The canonical checkout is `/home/kesslerio/Qwen3.8-Flash-Next-Dual-DGX-Sparks` on john; ofus is its TP2 worker. Both ranks use john's existing read-only NFS export. Do not stop `dspark-nfs` as part of a model restart.

## Profiles and immutable inputs

Use `QWEN_PROFILE=nvfp4-native` for qualification (262144 context, six active sequences), and `QWEN_PROFILE=nvfp4-long` for the final context-first service (500000 context, three active sequences, YaRN factor 2). Profiles override stale values in `.env` and reject the FP8 wrapper overrides. `QWEN_MTP_TOKENS=0|1|3` selects a controlled comparison; 0 is never the delivered default. The three-session admission cap is configured; three simultaneous near-limit requests remain unqualified.

`deploy/profiles/nvfp4-manifest.json` pins the checkpoint revision and source sizes/LFS hashes. The common profile pins the image registry digest. Before launch, the entire inventory must exist at that exact snapshot. Preserve the original FP8 image, snapshot, private environment, launcher and overlay files as a rollback bundle. Private data belongs outside Git.

## Service ownership

The `deploy/systemd/qwen-vllm.service` runs a persistent Python supervisor. Its EnvironmentFile is `/home/kesslerio/.config/qwen-cluster/active.env`, containing `QWEN_PROFILE=nvfp4-native` or `nvfp4-long` and optionally `QWEN_MTP_TOKENS`. The existing `.env` retains network and host settings.

The supervisor owns worker-first startup, model-specific readiness, three startup/recovery attempts per lifetime, and intentional stop. Each attempt permits 300 seconds for worker reachability and has a 1800-second load deadline; the enclosing unit allows all attempts plus backoff. `Restart=no` prevents an outer infinite retry loop. Check `systemctl status qwen-vllm` and `~/.local/state/qwen-cluster/status.json`; API timeouts alone do not restart live ranks. After exhaustion, repair the cause and explicitly start the unit again.

Disable the legacy `deepseek-vllm.service` and `deepseek-vllm-watchdog.timer` before changing the live Qwen service. The legacy watchdog hardcodes DeepSeek and must not be re-enabled. Preserve its backup. A normal stop removes only the two `vllm-fn` containers and leaves NFS running. Worker overlays are regenerated and copied on every startup, including after reboot.

## Rollout and rollback

Stage all weights before draining. Confirm no active generation or queued work, stop Qwen, install the reviewed supervisor unit, select the native profile, reload systemd and start. Only switch to the long profile after native NVFP4/MTP quality and performance qualification. Match both live image IDs, selected revision, model arguments and overlay hashes to the release record.

For rollback, stop the supervisor, restore the original private `.env` and original unit, and restore the saved launcher/overlay bundle if needed. Remove profile environment overrides; the original FP8 wrapper must run without a selected NVFP4 profile. Keep the DeepSeek retry loop and stale watchdog disabled. Reload systemd, start the original Qwen service, and verify `qwen3.8-flash-next-fp8` at 262144 plus a completed request. Never delete the retained FP8 checkpoint as cleanup.

## Acceptance

The implementation plan requires final-profile >=3000000 measured shared KV tokens, useful answers near the selected 500K total context, and repeatable speed improvements. Count the TP cache once. Do not advertise three full-context sessions based on native-context allocation. Record complete benchmark runs, real streamed tool calls, multimodal checks, recovery/rollback drills and the one-hour soak before calling the migration complete.

## Delivered state

The 500K profile passed a 479986-input-token semantic check and real DSH tool cycles on all three client hosts. Boot startup and delayed failure retries are configured. A controlled rollback returned a correct response; reboot and failure-injection drills, simultaneous near-limit qualification, and the mixed-workload soak remain outstanding. Hermes resumed after benchmark traffic ended. See the [qualification report](../benchmarks/john-ofus-nvfp4.md) for measured native-context speed and current-profile limits.

## Conversational prefill and cache reuse

The long profile preserves 8192-token prefill batches and uses patched Mamba `align`
prefix caching. Both fixes in `files/patch_mamba_prefix.py` are required:
exclude MTP-contaminated cache tails in both lookup paths, and resume recurrent
state using the Mamba group's block size. The launcher regenerates both overlays
from the pinned image and applies them to both ranks. Changed anchors fail
before launch. Do not enable caching by adding a bare CLI flag to an unpatched
image.

Use `bench/prefix_conversation.py --out <private-result.json>` to check varied
prefix lengths, older planted values, and concurrent branches. Identical prompt
replays alone cannot validate recurrent-state correctness. The diagnostic native
profile retains caching off for comparison.

Stop and disable the retired service explicitly; disabling alone does not stop
an already-running `Restart=` loop:

```sh
sudo systemctl stop deepseek-vllm.service
sudo systemctl disable deepseek-vllm.service
```

See [request history](request-telemetry.md) for retained per-client latency,
cache, interruption and tool-presence metadata. Use these records alongside
SparkDash's aggregate history to assess real traffic.

See the [real-workload latency report](../benchmarks/real-workload-20260908.md) for cache acceptance and the distinction between decode speed and conversation latency.
