from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class StopTest(unittest.TestCase):
    def test_unreachable_worker_still_stops_head_without_removing_nfs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "env").write_text("WORKER_IP=192.0.2.1\n")
            for command, body in {
                "ssh": "#!/bin/bash\nexit 255\n",
                "docker": '#!/bin/bash\nprintf "%s\\n" "$*" >> "$RECORD"\n',
            }.items():
                f = base / command
                f.write_text(body)
                f.chmod(0o755)
            env = dict(os.environ, PATH=str(base) + ":" + os.environ["PATH"], QWEN_ENV_FILE=str(base / "env"), RECORD=str(base / "calls"))
            env.pop("QWEN_PROFILE", None)
            result = subprocess.run(["bash", str(ROOT / "stop.sh")], env=env, capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("rm -f vllm-fn", (base / "calls").read_text())
            self.assertNotIn("nfs", (base / "calls").read_text())

    def test_docker_failure_is_not_reported_as_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "env").write_text("WORKER_IP=192.0.2.1\n")
            for command, body in {"ssh": "#!/bin/bash\nexit 0\n", "docker": "#!/bin/bash\nexit 1\n"}.items():
                f = base / command
                f.write_text(body)
                f.chmod(0o755)
            env = dict(os.environ, PATH=str(base) + ":" + os.environ["PATH"], QWEN_ENV_FILE=str(base / "env"))
            env.pop("QWEN_PROFILE", None)
            result = subprocess.run(["bash", str(ROOT / "stop.sh")], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"Head stop failed", result.stderr)
