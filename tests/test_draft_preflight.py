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
    def exercise(self, remote_bad=False, missing=False):
        calls=[]
        def gather(states, state, **kwargs):
            calls.append(state)
            states[:] = [state, {'ok':False,'error':'OSError'} if remote_bad else state]
        group=NS(world_size=2,cpu_group=object())
        torch=NS(bfloat16='bf16',float16='f16',float32='f32', distributed=NS(all_gather_object=gather))
        model=NS(lm_head=NS(weight=NS(dim=lambda:2,shape=[100,8],dtype='bf16'),org_vocab_size=100,shard_indices=NS()), logits_processor=NS(scale=1))
        source=m.DRAFT_VOCAB_BLOCK.split('    # Keep only the ids')[0]+'    return True\n'
        env={'nn':NS(Module=object),'torch':torch,'os':m.os}
        exec(source,env)
        with tempfile.NamedTemporaryFile(mode='w') as f:
            f.write('1\n2\n3\n');f.flush()
            with patch.dict('sys.modules', {'vllm.distributed': NS(get_tp_group=lambda:group)}), patch.dict(m.os.environ, {'VLLM_MTP_DRAFT_VOCAB': f.name+'.missing' if missing else f.name}):
                if remote_bad or missing:
                    with self.assertRaises(RuntimeError):env['_attach_draft_vocab'](model)
                else:
                    self.assertTrue(env['_attach_draft_vocab'](model))
        self.assertEqual(len(calls),1)

    def test_local_missing_file_still_joins_agreement(self):self.exercise(missing=True)
    def test_remote_failure_stops_locally_valid_rank(self):self.exercise(remote_bad=True)
    def test_agreed_valid_input_continues(self):self.exercise()
