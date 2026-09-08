import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ProfilesTest(unittest.TestCase):
    def load(self, profile, extra=None):
        with tempfile.NamedTemporaryFile(mode="w") as envfile:
            envfile.write("MODEL_ID=stale-fp8\nSKIP_PLE_PATCH=true\nMAX_MODEL_LEN=262144\n")
            envfile.flush()
            env = dict(os.environ, QWEN_ENV_FILE=envfile.name, QWEN_PROFILE=profile, SCRIPT_DIR=str(ROOT))
            env.update(extra or {})
            return subprocess.run(["bash", "-c", 'source "$SCRIPT_DIR/files/load-env.sh" || exit; printf "%s %s %s %s" "$MODEL_ID" "$MAX_MODEL_LEN" "$SKIP_PLE_PATCH" "$MTP_NUM_SPECULATIVE_TOKENS"'], env=env, text=True, capture_output=True)

    def test_long_profile_overrides_stale_fp8(self):
        result = self.load("nvfp4-long")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "RadixArk/Qwen3.8-Flash-Next-NVFP4 1000000 false 3")

    def test_wrapper_cannot_override_selected_profile(self):
        self.assertNotEqual(self.load("nvfp4-long", {"OVERRIDE_MODEL_ID": "fp8"}).returncode, 0)

    def test_native_and_mtp_control(self):
        result = self.load("nvfp4-native", {"QWEN_MTP_TOKENS": "0"})
        self.assertEqual(result.stdout, "RadixArk/Qwen3.8-Flash-Next-NVFP4 262144 false 0")
        self.assertNotEqual(self.load("nvfp4-native", {"QWEN_MTP_TOKENS": "-1"}).returncode, 0)

    def test_long_profile_accepts_mtp2(self):
        result = self.load("nvfp4-long", {"QWEN_MTP_TOKENS": "2"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "RadixArk/Qwen3.8-Flash-Next-NVFP4 1000000 false 2")

    def test_unknown_profile_fails(self):
        self.assertNotEqual(self.load("missing").returncode, 0)
