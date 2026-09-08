import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "deploy/release_preflight.py"

class PreflightTest(unittest.TestCase):
    def test_missing_and_wrong_shards_fail(self):
        spec = importlib.util.spec_from_file_location("preflight", MODULE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"sha": "a" * 40, "siblings": [{"rfilename": "model.safetensors", "size": 3}]}
            with self.assertRaises(ValueError):
                mod.verify_files(root, manifest)
            (root / "model.safetensors").write_bytes(b"ab")
            with self.assertRaises(ValueError):
                mod.verify_files(root, manifest)
            (root / "model.safetensors").write_bytes(b"abc")
            self.assertEqual(mod.verify_files(root, manifest), 3)

    def test_storage_reserve(self):
        spec = importlib.util.spec_from_file_location("preflight", MODULE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with self.assertRaises(ValueError):
            mod.check_space(99, 80, 20)
        mod.check_space(100, 80, 20)

if __name__ == "__main__":
    unittest.main()
