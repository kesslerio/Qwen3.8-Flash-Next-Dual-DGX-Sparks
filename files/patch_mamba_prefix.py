#!/usr/bin/env python3
"""Pinned vLLM fixes for MTP prefix-cache resume (48375 and 53142).

Reference: starkweatherdigital/qwen3.8-flash-next-nvfp4-recipe patches 40/41.
Generated overlays retain the upstream Apache-2.0 source headers.
"""
import ast
from pathlib import Path

DROP = '''        assert pcp_world_size == 1, "PCP not support mamba now."
        block_hashes = resolve_block_hashes('''
DROP_FIXED = '''        assert pcp_world_size == 1, "PCP not support mamba now."
        # Exclude recurrent state that can include rejected MTP drafts. Bound
        # both the fine-grained and coarse lookup paths, not only the latter.
        if drop_eagle_block:
            max_length = max(0, max_length - kv_cache_spec.block_size)
        block_hashes = resolve_block_hashes('''
SEED = '''            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
            )'''
SEED_FIXED = '''            # A fresh request seeds -1. Cache resumes only follow an earlier
            # batch, which populates the Mamba group's actual block size.
            mamba_bs = (self._mamba_spec.block_size if self._mamba_spec is not None
                        else self.cache_config.block_size)
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_bs
            )'''


def patch(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Mamba prefix patch anchor changed; inspect the pinned image")
    result = source.replace(before, after)
    ast.parse(result)
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name, before, after in [('mamba_manager_patched.py', DROP, DROP_FIXED),
                                ('mamba_hybrid_patched.py', SEED, SEED_FIXED)]:
        (root / name).write_text(patch((root / (name + '.orig')).read_text(), before, after))
