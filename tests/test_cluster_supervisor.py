import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import os
import subprocess
import io
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("supervisor", ROOT / "deploy/qwen_cluster_supervisor.py")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)

class SupervisorTest(unittest.TestCase):
    def test_failed_final_stop_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            envfile = root / "env"
            envfile.write_text("WORKER_USER=test\nWORKER_IP=192.0.2.1\nPORT=8888\nSERVED_MODEL_NAME=qwen\n")
            supervisor.STOP.set()
            try:
                with patch.dict(os.environ, {"QWEN_ENV_FILE": str(envfile), "QWEN_PROFILE": ""}), \
                     patch.object(supervisor, "STATE", root / "state"), \
                     patch.object(supervisor.signal, "signal"), \
                     patch.object(supervisor, "stop_cluster", return_value=False):
                    with self.assertRaisesRegex(RuntimeError, "stop could not be confirmed"):
                        supervisor.main()
                    self.assertIn("failed:", (root / "state/status.json").read_text())
            finally:
                supervisor.STOP.clear()

    def test_long_prefill_timeout_does_not_restart_live_ranks(self):
        policy = supervisor.HealthPolicy("qwen")
        for _ in range(100):
            self.assertEqual(policy.observe(True, True, None), "unready")
        self.assertEqual(policy.observe(True, True, ["qwen"]), "ready")

    def test_transient_worker_loss_then_bounded_failure(self):
        policy = supervisor.HealthPolicy("qwen")
        self.assertEqual(policy.observe(True, False, ["qwen"]), "unready")
        self.assertEqual(policy.observe(True, True, ["qwen"]), "ready")
        for _ in range(2):
            self.assertEqual(policy.observe(True, False, None), "unready")
        self.assertEqual(policy.observe(True, False, None), "recover")

    def test_wrong_model_never_healthy(self):
        policy = supervisor.HealthPolicy("qwen")
        self.assertEqual(policy.observe(True, True, ["deepseek"]), "recover")

    def test_restart_budget_and_intentional_stop(self):
        for intentional in (False, True):
            with self.subTest(intentional=intentional), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                envfile = root / "env"
                envfile.write_text("WORKER_USER=test\nWORKER_IP=192.0.2.1\nPORT=8888\nSERVED_MODEL_NAME=qwen\n")
                supervisor.STOP.clear()
                def failed_launch():
                    if intentional:
                        supervisor.STOP.set()
                    return False
                original_wait = supervisor.STOP.wait
                with patch.dict(os.environ, {"QWEN_ENV_FILE": str(envfile), "QWEN_PROFILE": ""}), \
                     patch.object(supervisor, "STATE", root / "state"), \
                     patch.object(supervisor.signal, "signal"), \
                     patch.object(supervisor, "wait_for_worker", return_value=True), \
                     patch.object(supervisor, "launch", side_effect=failed_launch) as launch, \
                     patch.object(supervisor, "stop_cluster") as stop, \
                     patch.object(supervisor.STOP, "wait", side_effect=lambda seconds: original_wait(0)):
                    self.assertEqual(supervisor.main(), 0 if intentional else 1)
                    self.assertEqual(launch.call_count, 1 if intentional else 3)
                    self.assertTrue(stop.called)
                supervisor.STOP.clear()

    def test_command_timeout_kills_descendant_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "pids"
            with self.assertRaises(subprocess.TimeoutExpired):
                supervisor.run(["bash", "-c", 'echo $$ > "$1"; sleep 60 & echo $! >> "$1"; wait', "test", str(record)], timeout=0.1)
            for pid in record.read_text().splitlines():
                status = Path("/proc") / pid / "stat"
                if status.exists():
                    self.assertEqual(status.read_text().split()[2], "Z", f"live child survived: {pid}")

    def test_model_listing_cannot_hide_engine_failure(self):
        models = io.BytesIO(b'{"data":[{"id":"qwen"}]}')
        failure = urllib.error.HTTPError("http://localhost/health", 503, "engine failed", {}, None)
        with patch.object(supervisor.urllib.request, "urlopen", side_effect=[models, failure]):
            self.assertEqual(supervisor.identities(8888), [])

    def test_engine_health_timeout_remains_unready_without_restart(self):
        models = io.BytesIO(b'{"data":[{"id":"qwen"}]}')
        with patch.object(supervisor.urllib.request, "urlopen", side_effect=[models, TimeoutError("prefill")]):
            identities = supervisor.identities(8888)
        self.assertEqual(supervisor.HealthPolicy("qwen").observe(True, True, identities), "unready")
