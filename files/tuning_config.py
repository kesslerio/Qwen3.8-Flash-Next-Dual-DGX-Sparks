#!/usr/bin/env python3
"""Validate a non-secret qualification profile; emit shell-quoted assignments."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex

def render(config):
    allowed = {'id', 'kv_cache_memory_bytes', 'max_num_batched_tokens', 'max_num_seqs',
               'mtp_tokens', 'index_share_for_mtp_iteration', 'draft_sample_method',
               'draft_vocab', 'extra_env', 'async_scheduling'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('Unknown tuning fields')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', config.get('id', '')):
        raise ValueError('Invalid experiment ID')
    result = {'QWEN_EXPERIMENT_ID': config['id'],
              'QWEN_EXPERIMENT_SHA256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    for key, target, minimum, maximum in (
        ('kv_cache_memory_bytes', 'KV_CACHE_MEMORY_BYTES', 16_000_000_000, 40_000_000_000),
        ('max_num_batched_tokens', 'MAX_NUM_BATCHED_TOKENS', 2048, 8192),
        ('max_num_seqs', 'MAX_NUM_SEQS', 1, 8),
        ('mtp_tokens', 'MTP_NUM_SPECULATIVE_TOKENS', 0, 4)):
        if key in config:
            value = config[key]
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError('Invalid ' + key)
            result[target] = str(value)
    spec = {'method': 'mtp', 'num_speculative_tokens': config.get('mtp_tokens', 3)}
    for key in ('index_share_for_mtp_iteration', 'async_scheduling'):
        if key in config and type(config[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    if 'index_share_for_mtp_iteration' in config:
        spec['index_share_for_mtp_iteration'] = config['index_share_for_mtp_iteration']
    method = config.get('draft_sample_method', 'greedy')
    if method not in ('greedy', 'probabilistic'):
        raise ValueError('Unknown draft sampling method')
    spec['draft_sample_method'] = method
    if config.get('draft_vocab'):
        path = Path(config['draft_vocab'])
        if not path.is_absolute() or not path.is_file():
            raise ValueError('Draft vocabulary must be an existing absolute file')
        if method != 'greedy' or not spec['num_speculative_tokens']:
            raise ValueError('Draft vocabulary requires enabled greedy MTP')
        spec['use_local_argmax_reduction'] = True
        result['MTP_DRAFT_VOCAB'] = str(path)
    result['SPEC_CONFIG_JSON'] = json.dumps(spec, separators=(',', ':')) if spec['num_speculative_tokens'] else ''
    if 'async_scheduling' in config:
        result['QWEN_ASYNC_SCHEDULING_ARG'] = '--async-scheduling' if config['async_scheduling'] else '--no-async-scheduling'
    env = config.get('extra_env', {})
    if not isinstance(env, dict):
        raise ValueError('extra_env must be an object')
    for key, value in env.items():
        if not re.fullmatch(r'VLLM_[A-Z0-9_]+', key) or not isinstance(value, str) or '\n' in value or '\0' in value:
            raise ValueError('Invalid VLLM environment entry')
    env = dict(env, VLLM_QUALIFICATION_ID=result['QWEN_EXPERIMENT_ID'], VLLM_QUALIFICATION_SHA256=result['QWEN_EXPERIMENT_SHA256'])
    result['QWEN_EXTRA_ENV_ARGS'] = ' '.join('-e ' + shlex.quote(k + '=' + v) for k, v in sorted(env.items()))
    return '\n'.join('export ' + k + '=' + shlex.quote(v) for k, v in result.items())

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('file', type=Path)
    args = p.parse_args()
    print(render(json.loads(args.file.read_text())))
