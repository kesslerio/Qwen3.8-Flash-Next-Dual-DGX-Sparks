#!/usr/bin/env bash
# Build one pinned derived checkpoint. Run outside benchmark/startup windows.
# Set DST_REPO, DST_REVISION and QUANT_KIND; failed snapshots remain for audit.
set -euo pipefail
umask 077
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
IMAGE="vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
SRC_REVISION="7b719225242aacd3dbd3f9407468c2ee9a9d2594"
: "${DST_REPO:?Set a task-owned derived model ID}"
: "${DST_REVISION:?Set an immutable 40-character build revision}"
: "${QUANT_KIND:?Set draft-only or target-fp8}"
[[ $# == 0 ]] || { echo 'Use the explicit build environment; arbitrary converter flags are not accepted.' >&2; exit 2; }
[[ "$DST_REPO" =~ ^kesslerio/Qwen3\.8-Flash-Next-[A-Za-z0-9-]+$ ]] || exit 2
[[ "$DST_REVISION" =~ ^[a-f0-9]{40}$ ]] || exit 2
case "$QUANT_KIND" in
    draft-only) convert_args=(--draft-only --mtp-dense) ;;
    target-fp8) convert_args=() ;;
    *) exit 2 ;;
esac
source_rel="hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$SRC_REVISION"
derived_rel="hub/models--${DST_REPO%%/*}--${DST_REPO##*/}"
snapshot_rel="$derived_rel/snapshots/$DST_REVISION"
[[ -f "$HF_CACHE_DIR/$source_rel/config.json" && ! -e "$HF_CACHE_DIR/$snapshot_rel" ]] || {
    echo 'Source missing or destination already exists; existing records are preserved.' >&2; exit 2;
}
mkdir -p "$HF_CACHE_DIR/$derived_rel/snapshots"
run() {
    docker run --rm --pull=never --network=none --memory=2g --memory-swap=2g --cpus=4 \
        --user "$(id -u):$(id -g)" --entrypoint python3 \
        -e CUDA_VISIBLE_DEVICES= -e OMP_NUM_THREADS=4 \
        -v "$HF_CACHE_DIR:/hf" -v "$SCRIPT_DIR:/work:ro" "$IMAGE" "$@"
}
run /work/make_fp8_dense_checkpoint.py --src "/hf/$source_rel" --dst "/hf/$snapshot_rel" \
    --link-unchanged-from "/hf/$source_rel" "${convert_args[@]}"
run /work/verify_fp8_dense_checkpoint.py --src "/hf/$source_rel" --dst "/hf/$snapshot_rel"
identity_args=()
if [[ "$QUANT_KIND" == draft-only ]]; then
    identity_rel="$derived_rel/target-identity-$DST_REVISION.json"
    run /work/verify_target_identity.py --src "/hf/$source_rel" --dst "/hf/$snapshot_rel" \
        > "$HF_CACHE_DIR/$identity_rel"
    identity_args=(--identity-proof "/hf/$identity_rel")
fi
run /work/make_inventory.py --snapshot "/hf/$snapshot_rel" --model "$DST_REPO" \
    --revision "$DST_REVISION" --source-revision "$SRC_REVISION" --kind "$QUANT_KIND" \
    --output "/hf/$derived_rel/inventory-$DST_REVISION.json" "${identity_args[@]}"
mkdir -p "$HF_CACHE_DIR/$derived_rel/refs"
printf '%s\n' "$DST_REVISION" > "$HF_CACHE_DIR/$derived_rel/refs/main"
echo "Derived weights verified locally. Runtime dispatch and task quality remain unqualified: $DST_REPO@$DST_REVISION"
