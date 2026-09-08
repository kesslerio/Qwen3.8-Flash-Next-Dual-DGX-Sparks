import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('prefix', Path(__file__).resolve().parents[1] / 'files/patch_mamba_prefix.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PrefixTests(unittest.TestCase):
    def test_drop_bounds_fine_and_coarse_search(self):
        text = m.DROP_FIXED.split('        block_hashes =')[0]
        code = compile('def bound(pcp_world_size,drop_eagle_block,max_length,kv_cache_spec):\n'+text+'\n        return max_length\n', '<drop>', 'exec')
        ns = {}; exec(code, ns)
        for length in [0,127,128,159,256,4096]:
            for drop in [True,False]:
                result = ns['bound'](1,drop,length,SimpleNamespace(block_size=128))
                self.assertEqual(result, max(0,length-128) if drop else length)
                self.assertLessEqual(result//32, length//32)

    def test_seed_uses_mamba_not_attention_size(self):
        text = 'def seed(self,req_index,new_req_data):\n'+m.SEED_FIXED
        ns = {};exec(compile(text,'<seed>','exec'),ns)
        values = []
        obj = SimpleNamespace(_mamba_spec=SimpleNamespace(block_size=1600),cache_config=SimpleNamespace(block_size=64),_mamba_state_idx_gpu=[SimpleNamespace(fill_=values.append)])
        ns['seed'](obj,0,SimpleNamespace(num_computed_tokens=3200))
        self.assertEqual(values[-1],1)
        obj._mamba_spec=None
        ns['seed'](obj,0,SimpleNamespace(num_computed_tokens=0))
        self.assertEqual(values[-1],-1)

    def test_drift_fails_before_output(self):
        with self.assertRaisesRegex(ValueError,'anchor changed'):
            m.patch('changed source',m.DROP,m.DROP_FIXED)
