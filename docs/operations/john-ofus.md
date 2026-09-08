# Ownership

Host service ownership and deployment records live in [spark-john-ofus-management](https://github.com/kesslerio/spark-john-ofus-management).
Request telemetry implementation and its guide live in [sparkDash](https://github.com/kesslerio/sparkDash/tree/main/scripts/llm-request-telemetry).

Set `VLLM_REQUEST_TELEMETRY_PATH` to the externally installed observer file to
mount it into both ranks and enable the supported vLLM metrics flags. The recipe
does not install or supervise the collector.
