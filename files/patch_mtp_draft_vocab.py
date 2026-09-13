#!/usr/bin/env python3
"""Patch nvidia/mtp.py for reduced-vocabulary drafting (FR-Spec style), TP-aware.

Adapted from MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark (AGPL-3.0-or-later),
which implements this for TP=1 only. This version supports vocab-parallel
lm_heads, which is what this 2-Spark TP=2 deployment needs.

Why it pays here: this checkpoint's vocabulary is 248,320 tokens and the MTP
drafter carries its OWN BF16 ParallelLMHead over all of it (tie_word_embeddings
is false) -- 248320 x 2560 x 2 B = 1.18 GiB, or 0.59 GiB per GPU at TP=2, read
once per draft step. At MTP=3 that is three of the four lm_head reads in an
engine step, ~2.4 GiB of the ~9.2 GiB a single-stream step moves per GPU.

Draft sampling is greedy (the speculator only builds draft_logits for the
"probabilistic" method), so the drafter needs an argmax and nothing else. An
argmax over a frequency-ranked subset is wrong only for tokens outside the
subset. Correct rejection sampling preserves the target distribution while
trading proposal acceptance for bandwidth. Proposal changes consume randomness
differently, so fixed seeds do not promise identical generated sequences.

The reduced head is reached only through get_top_tokens, which the speculator
calls when use_local_argmax_reduction is set. compute_logits keeps the full
head, so every other path keeps full-vocabulary behaviour.

TP notes (the part that differs from upstream): each rank's lm_head holds the
org-vocab rows [org_vocab_start_index, org_vocab_end_index). Each rank slices
its own shard to the draft ids landing in its range, takes a local argmax, and
the winners are reduced across ranks with the same (value, index) all-gather
vLLM's own LogitsProcessor.get_top_tokens uses -- O(batch * 2 * tp_size)
instead of O(batch * vocab_size).

Both the logit scale and the tanh soft cap are monotonic and applied
identically on every rank, so skipping them leaves both the local argmax and
the cross-rank comparison unchanged.

Inputs:  files/mtp_patched.py.orig   (nvidia/mtp.py from the image)
Outputs: files/mtp_patched.py
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, "mtp_patched.py.orig")
OUT = os.path.join(HERE, "mtp_patched.py")

DRAFT_VOCAB_BLOCK = '''

def _attach_draft_vocab(model: nn.Module) -> None:
    """Slice this rank's lm_head shard down to VLLM_MTP_DRAFT_VOCAB's token ids.

    The file is one integer token id per line (see files/build_draft_vocab.py).
    The full head is left in place and untouched; only get_top_tokens below
    reads the slice.
    """
    # Every rank agrees before ANY local early return or balancing collective.
    # CPU metadata collectives remain possible even if the local head is missing.
    from vllm.distributed import get_tp_group
    import hashlib
    group = get_tp_group()
    state = {"ok": False}
    try:
        configured = os.environ.get("VLLM_MTP_DRAFT_VOCAB", "").strip()
        head = getattr(model, "lm_head", None)
        head_weight = getattr(head, "weight", None)
        if not configured:
            state = {"ok": True, "enabled": False}
        else:
            if head_weight is None or head_weight.dim() != 2:
                raise ValueError("Missing two-dimensional draft head")
            if getattr(head, "shard_indices", None) is None:
                raise ValueError("Missing vocabulary shard metadata")
            shard = head.shard_indices
            start = int(shard.org_vocab_start_index)
            end = int(shard.org_vocab_end_index)
            tp_size = int(head.tp_size)
            if tp_size != group.world_size or start < 0 or end <= start or end - start > head_weight.shape[0]:
                raise ValueError("Invalid vocabulary shard bounds or TP size")
            if getattr(model.logits_processor, "scale", 1.0) <= 0:
                raise ValueError("Non-positive logit scale")
            if head_weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError("Reduced vocabulary currently requires an unquantized draft head")
            with open(configured) as handle:
                validated_ids = sorted({int(line) for line in handle if line.strip()})
            vocabulary = int(getattr(head, "org_vocab_size", head_weight.shape[0]))
            if not validated_ids or len(validated_ids) >= vocabulary or min(validated_ids) < 0 or max(validated_ids) >= vocabulary:
                raise ValueError("Invalid draft vocabulary ids")
            if end > vocabulary:
                raise ValueError("Shard exceeds the original vocabulary")
            state = {"ok": True, "enabled": True, "vocabulary": vocabulary,
                     "ids_hash": hashlib.sha256(str(validated_ids).encode()).hexdigest(),
                     "width": int(head_weight.shape[1]), "dtype": str(head_weight.dtype),
                     "balance": os.environ.get("VLLM_MTP_DRAFT_VOCAB_BALANCE", "") not in ("", "0"),
                     "shard": [start, end]}
    except Exception as exc:
        state = {"ok": False, "error": type(exc).__name__}
    states = [None] * group.world_size
    torch.distributed.all_gather_object(states, state, group=group.cpu_group)
    shared = [{k: v for k, v in item.items() if k != "shard"} for item in states]
    if not all(item.get("ok") for item in states) or any(item != shared[0] for item in shared):
        raise RuntimeError("MTP vocabulary preflight disagreed across ranks: " + str(states))
    if not state["enabled"]:
        return
    bounds = sorted(item["shard"] for item in states)
    if bounds[0][0] != 0 or bounds[-1][1] != vocabulary or any(a[1] != b[0] for a, b in zip(bounds, bounds[1:])):
        raise RuntimeError("MTP vocabulary shards do not partition the original vocabulary")
    # Reuse validated values instead of reopening the file after agreement.
    lm_head = head
    weight = head_weight
    shard = head.shard_indices
    org_vocab = vocabulary
    ids = validated_ids

    # Keep only the ids this rank actually owns, as local row offsets.
    local_ids = [i for i in ids if start <= i < end]
    allocated = {"ok": False}
    try:
        rows = torch.tensor([i - start for i in local_ids], dtype=torch.long,
                            device=weight.device)
        draft_weight = weight.data.index_select(0, rows).contiguous()
        draft_ids = torch.tensor(local_ids, dtype=torch.long, device=weight.device)
        model.register_buffer("_draft_lm_head_weight", draft_weight, persistent=False)
        model.register_buffer("_draft_id_to_target_id", draft_ids, persistent=False)
        allocated = {"ok": True}
    except Exception as exc:
        allocated = {"ok": False, "error": type(exc).__name__}
    allocations = [None] * group.world_size
    torch.distributed.all_gather_object(allocations, allocated, group=group.cpu_group)
    if not all(item["ok"] for item in allocations):
        raise RuntimeError("MTP draft slice allocation failed on a rank: " + str(allocations))

    # Optional draft-head STORAGE balancing (VLLM_MTP_DRAFT_VOCAB_BALANCE=1).
    #
    # WHICH tokens are in the draft vocabulary is a quality decision, already made
    # above. WHICH rank stores a given row is purely a bandwidth decision, and the
    # two are independent: get_top_tokens takes a local argmax and reduces with a
    # (value, index) all-gather, so it is correct for ANY row-to-rank assignment as
    # long as _draft_id_to_target_id maps local row -> global id.
    #
    # Slicing by the lm_head's own shard range ties them together, and byte-level
    # BPE puts frequent tokens at low ids, so nearly every draft row lands on rank 0
    # (measured on draft_vocab_32k.txt: 32,406 of 32,768 on rank 0, 362 on rank 1).
    # A decode step waits for the slowest rank, so rank 1 idles at the all-gather
    # while rank 0 reads ~90x more bytes, three times per step.
    #
    # This is NOT build_draft_vocab.py's --balance-shards, which balances the
    # VOCABULARY by padding it with merge-order ids and cost acceptance
    # 56.5% -> 47.8%. Balancing storage keeps the token set identical, so drafts
    # are bit-identical and acceptance is unchanged.
    #
    # Runs after the buffers are registered so it is independent of how the weight
    # slice was produced (bf16 directly, or dequantized from an FP8-dense head).
    if tp_size > 1 and os.environ.get("VLLM_MTP_DRAFT_VOCAB_BALANCE", "") not in ("", "0"):
        # Collectives must be executed by EVERY rank in the same order. A local
        # try/except fallback around them is a deadlock: one rank bailing out to
        # the except branch leaves the others blocked in all_gather forever, and
        # a decode step simply waits. So decide GLOBALLY first, with one
        # all_reduce every rank always reaches, and only then run the gathers.
        group = None
        local_ok = 1
        try:
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            w = model._draft_lm_head_weight
            gid = model._draft_id_to_target_id
            if w.dim() != 2 or gid.dim() != 1 or gid.shape[0] != w.shape[0]:
                local_ok = 0
        except Exception as exc:
            logger.warning("MTP draft vocab: cannot balance storage (%s).", exc)
            local_ok = 0

        agreed = False
        if group is not None:
            try:
                flag = torch.tensor([local_ok], dtype=torch.long, device=weight.device)
                torch.distributed.all_reduce(
                    flag, op=torch.distributed.ReduceOp.MIN, group=group.device_group)
                agreed = bool(int(flag.item()))
            except Exception as exc:
                # The agreement collective itself failed. Every rank calls it, so a
                # failure here is not one-sided; skip balancing everywhere.
                logger.warning(
                    "MTP draft vocab: balance agreement failed (%s); "
                    "keeping the shard-local assignment.", exc)
                agreed = False
            if local_ok and not agreed:
                logger.warning(
                    "MTP draft vocab: another rank cannot balance; all ranks keep "
                    "the shard-local assignment.")

        if agreed:
            # Past this point every rank runs the identical collective sequence.
            rank = int(group.rank_in_group)
            w = model._draft_lm_head_weight
            gid = model._draft_id_to_target_id

            counts_t = torch.zeros(tp_size, dtype=torch.long, device=w.device)
            counts_t[rank] = w.shape[0]
            torch.distributed.all_reduce(counts_t, group=group.device_group)
            counts = [int(c) for c in counts_t.tolist()]
            n_max = max(counts)

            pad_w = w.new_zeros((n_max, w.shape[1]))
            pad_w[: w.shape[0]] = w
            pad_i = torch.full((n_max,), -1, dtype=torch.long, device=w.device)
            pad_i[: gid.shape[0]] = gid

            all_w = group.all_gather(pad_w.unsqueeze(0), dim=0)
            all_i = group.all_gather(pad_i.unsqueeze(0), dim=0)

            keep_w = [all_w[r, : counts[r]] for r in range(tp_size) if counts[r]]
            keep_i = [all_i[r, : counts[r]] for r in range(tp_size) if counts[r]]
            full_w = torch.cat(keep_w, dim=0)
            full_i = torch.cat(keep_i, dim=0)
            order = torch.argsort(full_i)
            full_w, full_i = full_w[order], full_i[order]

            # Postconditions: the reassembled set must be exactly the input ids,
            # with no padding sentinel and no duplicate. A silent violation here
            # would drop or double-count draft candidates on some rank.
            n_total = int(full_i.numel())
            if (n_total != len(ids) or int(full_i.min()) < 0
                    or int(torch.unique(full_i).numel()) != n_total):
                raise RuntimeError(
                    f"MTP draft vocab: balance produced {n_total} ids "
                    f"(expected {len(ids)}, min {int(full_i.min())}, "
                    f"unique {int(torch.unique(full_i).numel())})")

            # Install transactionally: build both buffers, then swap them in
            # together. A failure part-way through must not leave the weight and
            # its id map describing different row sets.
            take = torch.arange(rank, n_total, tp_size, device=w.device)
            new_w = full_w.index_select(0, take).contiguous()
            new_i = full_i.index_select(0, take).contiguous()
            if new_w.shape[0] != new_i.shape[0]:
                raise RuntimeError(
                    f"MTP draft vocab: balanced weight rows {new_w.shape[0]} != "
                    f"id map {new_i.shape[0]}")
            model._draft_lm_head_weight = new_w
            model._draft_id_to_target_id = new_i
            local_ids = [int(i) for i in new_i.tolist()]
            del full_w, full_i, all_w, all_i, pad_w, pad_i
            balanced = True
        else:
            balanced = False
    else:
        balanced = False

    esize = weight.element_size()
    full_gib = weight.numel() * esize / 2**30
    cut_gib = model._draft_lm_head_weight.numel() * esize / 2**30
    logger.info(
        "MTP draft vocab: %d of %d tokens (%.1f%%), %d on this rank of %d "
        "(balanced=%s); draft lm_head shard %.2f -> %.2f GiB per draft step",
        len(ids), org_vocab, 100.0 * len(ids) / org_vocab,
        int(model._draft_id_to_target_id.shape[0]), tp_size, balanced,
        full_gib, cut_gib,
    )

'''

GET_TOP_TOKENS = '''
    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft token ids, over the reduced vocabulary when present.

        Mirrors LogitsProcessor.get_top_tokens: local argmax per rank, then a
        (value, index) all-gather instead of an all-gather of full logits. The
        logit scale and soft cap are monotonic and identical on every rank, so
        skipping them changes neither the local argmax nor the cross-rank
        comparison; _attach_draft_vocab refuses to engage when the scale is not
        positive.
        """
        weight = getattr(self, "_draft_lm_head_weight", None)
        if weight is None:
            return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

        num_tokens = hidden_states.shape[0]
        if weight.shape[0] == 0:
            # This rank owns none of the draft vocabulary: contribute a losing bid.
            local_max_vals = torch.full((num_tokens,), float("-inf"),
                                        dtype=torch.float32, device=hidden_states.device)
            global_indices = torch.zeros((num_tokens,), dtype=torch.long,
                                         device=hidden_states.device)
        else:
            logits = torch.nn.functional.linear(hidden_states.to(weight.dtype), weight)
            local_max_vals, local_max_indices = logits.max(dim=-1)
            global_indices = self._draft_id_to_target_id[local_max_indices]

        tp_size = int(getattr(self.lm_head, "tp_size", 1))
        if tp_size == 1:
            return global_indices.to(torch.int64)

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(num_tokens, tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        top_tokens = gathered[:, :, 1].gather(dim=-1, index=max_rank_idx)
        return top_tokens.squeeze(-1).to(torch.int64)

'''


def patch(edits):
    src = open(ORIG).read()
    for i, (old, new) in enumerate(edits):
        count = src.count(old)
        if count != 1:
            sys.exit(f"mtp_patched: anchor {i} not unique/missing "
                     f"(count={count}):\n{old[:200]}")
        src = src.replace(old, new)
    try:
        ast.parse(src)
    except SyntaxError as exc:
        sys.exit(f"mtp_patched: patched source does not parse: {exc}")
    open(OUT, "w").write(src)
    print("patched mtp_patched.py")


def main():
    if not os.path.isfile(ORIG):
        sys.exit(f"ERROR: missing {ORIG} (start.sh extracts it from the image)")
    if os.path.isfile(OUT) and "_attach_draft_vocab" in open(OUT).read():
        print("mtp_patched.py already patched")
        return
    patch([
        # os, a logger, and the all-gather primitive the reduction needs.
        (
            "from vllm.compilation.decorators import support_torch_compile\n",
            "import os\n\n"
            "from vllm.compilation.decorators import support_torch_compile\n"
            "from vllm.logger import init_logger\n",
        ),
        (
            "from vllm.distributed import get_pp_group\n",
            "from vllm.distributed import get_pp_group\n"
            "from vllm.distributed.communication_op import "
            "tensor_model_parallel_all_gather\n",
        ),
        # Module scope, before the first helper: the MTP class itself sits under
        # a @support_torch_compile decorator, so nothing can go between.
        (
            "def _remap_ignored_layers(\n",
            "logger = init_logger(__name__)\n"
            + DRAFT_VOCAB_BLOCK
            + "\ndef _remap_ignored_layers(\n",
        ),
        # get_top_tokens beside compute_logits, which stays full-vocabulary.
        (
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n",
            "    def compute_logits(\n"
            "        self, hidden_states: torch.Tensor, spec_step_idx: int = 0\n"
            "    ) -> torch.Tensor | None:\n"
            "        return self.logits_processor(self.lm_head, hidden_states)\n"
            + GET_TOP_TOKENS,
        ),
        # Slice once the real weights are in.
        (
            "        return loader.load_weights(remap_weight_names())\n",
            "        loaded = loader.load_weights(remap_weight_names())\n"
            "        _attach_draft_vocab(self)\n"
            "        return loaded\n",
        ),
    ])


if __name__ == "__main__":
    main()
