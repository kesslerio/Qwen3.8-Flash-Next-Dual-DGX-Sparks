import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('identity',Path(__file__).resolve().parents[1]/'files/fp8dense/verify_target_identity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class IdentityTests(unittest.TestCase):
    def write(self,path,byte):
        path.mkdir();h=json.dumps({'target.weight':{'dtype':'U8','shape':[1],'data_offsets':[0,1]}}).encode()
        (path/'model.safetensors').write_bytes(struct.pack('<Q',len(h))+h+byte)
        (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'target.weight':'model.safetensors'}}))

    def test_detects_same_shape_wrong_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b';self.write(a,b'a');self.write(b,b'b')
            with self.assertRaises(ValueError):m.verify(a,b)

    def test_accepts_equal_bytes_at_distinct_paths(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a',Path(t)/'b';self.write(a,b'a');self.write(b,b'a')
            self.assertEqual(m.verify(a,b)['target_tensors_hashed'],1)
