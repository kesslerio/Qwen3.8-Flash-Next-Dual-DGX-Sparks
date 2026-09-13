import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('draft', ROOT / 'files/patch_mtp_draft_vocab.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class PreflightTests(unittest.TestCase):
    def exercise(self, remote_bad=False, missing=False, bad_shard=False, overlapping=False):
        calls=[]
        def gather(states, state, **kwargs):
            calls.append(state)
            remote=dict(state)
            if 'shard' in remote:remote['shard']=[0,50] if overlapping else [50,100]
            states[:] = [state, {'ok':False,'error':'OSError'} if remote_bad else remote]
        group=NS(world_size=2,cpu_group=object())
        torch=NS(bfloat16='bf16',float16='f16',float32='f32', distributed=NS(all_gather_object=gather))
        shard=NS() if bad_shard else NS(org_vocab_start_index=0,org_vocab_end_index=50)
        model=NS(lm_head=NS(weight=NS(dim=lambda:2,shape=[50,8],dtype='bf16'),org_vocab_size=100,tp_size=2,shard_indices=shard), logits_processor=NS(scale=1))
        source=m.DRAFT_VOCAB_BLOCK.split('    # Keep only the ids')[0]+'    return True\n'
        env={'nn':NS(Module=object),'torch':torch,'os':m.os}
        exec(source,env)
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('1\n2\n3\n');f.flush()
            with patch.dict('sys.modules', {'vllm.distributed': NS(get_tp_group=lambda:group)}), patch.dict(m.os.environ, {'VLLM_MTP_DRAFT_VOCAB': f.name+'.missing' if missing else f.name}):
                if remote_bad or missing or bad_shard or overlapping:
                    with self.assertRaises(RuntimeError):env['_attach_draft_vocab'](model)
                else:
                    self.assertTrue(env['_attach_draft_vocab'](model))
        self.assertEqual(len(calls),1)

    def test_local_missing_file_still_joins_agreement(self):self.exercise(missing=True)
    def test_remote_failure_stops_locally_valid_rank(self):self.exercise(remote_bad=True)
    def test_agreed_valid_input_continues(self):self.exercise()
    def test_missing_shard_fields_still_joins_agreement(self):self.exercise(bad_shard=True)
    def test_overlapping_shards_fail_before_slicing(self):self.exercise(overlapping=True)
