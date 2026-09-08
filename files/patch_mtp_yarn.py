#!/usr/bin/env python3
"""Carry Qwen Flash YaRN into its same-checkpoint MTP draft configuration."""
import ast
from pathlib import Path

ANCHOR = """        if not callable(target_hf_overrides):
            return SpeculativeConfig.hf_config_override
"""
REPLACEMENT = """        if isinstance(target_hf_overrides, dict):
            rope = target_hf_overrides.get("text_config", {}).get("rope_parameters")
            if isinstance(rope, dict) and rope.get("rope_type") == "yarn":
                return functools.partial(
                    SpeculativeConfig._apply_qwen_flash_yarn_override, rope
                )
        if not callable(target_hf_overrides):
            return SpeculativeConfig.hf_config_override
"""
METHOD_ANCHOR = """    @staticmethod
    def compose_draft_hf_overrides("""
METHOD = """    @staticmethod
    def _apply_qwen_flash_yarn_override(rope, hf_config):
        # Qwen's MTP shares the checkpoint and rotary positions with its target.
        # Other architectures must retain the upstream target-only dict policy.
        if getattr(hf_config, "model_type", None) == "qwen4_exp":
            text = hf_config.text_config
            parameters = copy.deepcopy(text.rope_parameters)
            parameters.update(copy.deepcopy(rope))
            text.rope_parameters = parameters
        return SpeculativeConfig.hf_config_override(hf_config)

""" + METHOD_ANCHOR

def patch(source):
    for before, after in [(ANCHOR, REPLACEMENT), (METHOD_ANCHOR, METHOD)]:
        if source.count(before) != 1:
            raise ValueError("MTP YaRN patch anchor changed; inspect the pinned image")
        source = source.replace(before, after)
    ast.parse(source)
    return source

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    source = here / "speculative_yarn_patched.py.orig"
    (here / "speculative_yarn_patched.py").write_text(patch(source.read_text()))
