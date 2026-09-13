import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCH=Path(__file__).resolve().parents[1]/'bench'
sys.path.insert(0,str(BENCH))
spec=importlib.util.spec_from_file_location('blind_quality',BENCH/'blind_quality.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class BlindQualityTests(unittest.TestCase):
    def item(self,code):
        return {'id':'opaque','case':'repair-cache','responses':[{'output':json.dumps({'code':code}),'tool_calls':{},'transport_error':False}]}

    def test_valid_repair_and_missing_field_boundary(self):
        correct='def candidate(u):\n a=u.get("inputTokens");b=u.get("cacheReadTokens")\n return None if a is None or b is None else a+b'
        self.assertTrue(m.evaluate(self.item(correct))['passed'])
        incorrect='def candidate(u):\n return u.get("inputTokens",0)+u["cacheReadTokens"] if "cacheReadTokens" in u else None'
        self.assertFalse(m.evaluate(self.item(incorrect))['passed'])

    def test_unsafe_repair_never_runs(self):
        self.assertFalse(m.evaluate(self.item('import os\ndef candidate(u): return 1'))['passed'])

    def test_anonymous_inputs_and_hash_bound_reveal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/'source-run';run.mkdir()
            (run/'manifest.json').write_text(json.dumps({'tool_mode':'native-auto','profile':'SECRET_VARIANT'}))
            case='synthesis-0';values=[713,829,491]
            row={'case':case,'passed':False,'elapsed_s':999,'requests':[{'output':json.dumps({'values':values,'sum':sum(values)}),'tool_calls':{}}]}
            events=[{'event':'task_wave','valid':True,'repeat':0,'concurrency':3,'tasks':[row]},{'event':'run_finish','status':'complete'}]
            (run/'events.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
            with contextlib.redirect_stdout(io.StringIO()):m.build(root/'bundles',[run])
            directory=next((root/'bundles').iterdir());raw=(directory/'anonymous/inputs.json').read_text()
            for secret in ('SECRET_VARIANT','source-run','elapsed_s','passed'):self.assertNotIn(secret,raw)
            with contextlib.redirect_stdout(io.StringIO()):
                m.grade(directory/'anonymous');m.reveal(directory)
            result=json.loads((directory/'revealed.json').read_text())
            self.assertEqual(result['profiles']['SECRET_VARIANT']['passed'],1)
            (directory/'revealed.json').unlink()
            p=directory/'key.json';key=json.loads(p.read_text());key['inputs_sha256']='wrong';p.write_text(json.dumps(key))
            with self.assertRaises(ValueError):m.reveal(directory)


if __name__=='__main__':unittest.main()
