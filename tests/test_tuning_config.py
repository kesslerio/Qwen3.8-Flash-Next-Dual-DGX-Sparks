import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tuning', ROOT / 'files/tuning_config.py')
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)

class TuningTests(unittest.TestCase):
    def test_render_is_shell_safe(self):
        text = tuning.render({'id': 'safe', 'extra_env': {'VLLM_TEST': "literal $(exit 9) ' quoted"}})
        r = subprocess.run(['bash', '-c', text + '\nprintf "%s" "$QWEN_EXTRA_ENV_ARGS"'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('$(exit 9)', r.stdout)

    def test_reject_invalid_settings(self):
        for config in ({'id': 'x', 'max_num_seqs': 48}, {'id': 'x', 'mtp_tokens': True},
                       {'id': 'x', 'model': 'different'}, {'id': 'x', 'async_scheduling': 'false'},
                       {'id': 'x', 'extra_env': {'HF_TOKEN': 'secret'}}):
            with self.assertRaises(ValueError):
                tuning.render(config)

    def test_probabilistic_cannot_use_reduced_head(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(ValueError):
                tuning.render({'id': 'x', 'draft_vocab': f.name, 'draft_sample_method': 'probabilistic'})

    def test_native_tuning_reaches_shell(self):
        text = tuning.render({'id': 'kv24', 'kv_cache_memory_bytes': 24000000000, 'max_num_seqs': 6, 'mtp_tokens': 4})
        r = subprocess.run(['bash', '-c', text + '\nprintf "%s %s" "$KV_CACHE_MEMORY_BYTES" "$SPEC_CONFIG_JSON"'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn('24000000000', r.stdout)
        self.assertIn('"num_speculative_tokens":4', r.stdout)
