import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patch_mtp_yarn", root / "files/patch_mtp_yarn.py")
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

class MtpYarnTests(unittest.TestCase):
    def helper(self):
        tree = ast.parse("class Patched:\n" + patcher.METHOD.split(patcher.METHOD_ANCHOR)[0])
        namespace = {"copy": copy, "SpeculativeConfig": SimpleNamespace(hf_config_override=lambda config: config)}
        exec(compile(tree, "<scoped YaRN transform>", "exec"), namespace)
        return namespace["Patched"]._apply_qwen_flash_yarn_override

    def test_qwen_rope_retains_checkpoint_fields_and_does_not_alias_input(self):
        original = {"rope_theta": 10000000, "mrope_section": [11, 11, 10], "rope_type": "default"}
        config = SimpleNamespace(model_type="qwen4_exp", text_config=SimpleNamespace(rope_parameters=original))
        rope = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
        result = self.helper()(rope, config)
        self.assertEqual(result.text_config.rope_parameters["factor"], 4.0)
        self.assertEqual(result.text_config.rope_parameters["mrope_section"], [11, 11, 10])
        self.assertEqual(original["rope_type"], "default")
        rope["factor"] = 8
        self.assertEqual(result.text_config.rope_parameters["factor"], 4.0)

    def test_other_architectures_keep_original_policy(self):
        config = SimpleNamespace(model_type="other")
        self.assertIs(self.helper()({"rope_type": "yarn"}, config), config)
        self.assertFalse(hasattr(config, "text_config"))

    def test_unknown_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "anchor changed"):
            patcher.patch("changed upstream source")

if __name__ == "__main__":
    unittest.main()
