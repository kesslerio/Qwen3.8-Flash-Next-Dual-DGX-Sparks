#!/usr/bin/env python3
"""Single owner for both Qwen ranks; API timeouts alone never kill long prefills."""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
STOP = threading.Event()
STATE = Path.home() / ".local/state/qwen-cluster"

class HealthPolicy:
    def __init__(self, model):
        self.model = model
        self.rank_failures = 0

    def observe(self, head, worker, identities):
        if identities is not None and self.model not in identities:
            return "recover"
        if not head or not worker:
            self.rank_failures += 1
            return "recover" if self.rank_failures >= 3 else "unready"
        self.rank_failures = 0
        return "ready" if identities is not None else "unready"


def run(args, timeout=20):
    with subprocess.Popen(args, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)


def notify(state, ready=False):
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "status.tmp"
    tmp.write_text(json.dumps({"state": state, "updated_at": time.time()}) + "\n")
    tmp.replace(STATE / "status.json")
    print(state, flush=True)
    address = os.environ.get("NOTIFY_SOCKET")
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            try:
                sock.connect("\0" + address[1:] if address.startswith("@") else address)
                sock.sendall((("READY=1\n" if ready else "") + "STATUS=" + state).encode())
            except OSError:
                pass


def rank_alive(worker=None):
    command = ["docker", "inspect", "--format", "{{.State.Running}}", "vllm-fn"]
    if worker:
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker,
                   "docker inspect --format '{{.State.Running}}' vllm-fn"]
    try:
        result = run(command)
        return result.returncode == 0 and result.stdout.strip() == "true"
    except subprocess.TimeoutExpired:
        return False


def identities(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
            models = [item["id"] for item in json.load(response)["data"]]
        # /v1/models can succeed while the engine is unhealthy. A definite
        # engine failure differs from a timeout during a long prefill.
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10):
            return models
    except urllib.error.HTTPError as error:
        return [] if error.code >= 500 else None
    except (OSError, ValueError, KeyError):
        return None


def stop_cluster():
    result = run(["bash", "stop.sh"], timeout=120)
    if result.returncode:
        print("Cluster stop could not be fully confirmed; both ranks were attempted", flush=True)
    return result.returncode == 0


def wait_for_worker(worker):
    deadline = time.monotonic() + 300
    while not STOP.is_set() and time.monotonic() < deadline:
        try:
            if run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", worker, "true"], timeout=10).returncode == 0:
                return True
        except subprocess.TimeoutExpired:
            pass
        STOP.wait(5)
    return False


def launch():
    process = subprocess.Popen(["bash", "start.sh", "--launch", "--nfs"], cwd=ROOT,
                               start_new_session=True)
    deadline = time.monotonic() + 1800
    try:
        while process.poll() is None:
            if STOP.wait(1) or time.monotonic() > deadline:
                return False
        return process.returncode == 0
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def supervise():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: STOP.set())
    # Read only the four non-secret values needed for readiness and SSH.
    env = dict(os.environ, SCRIPT_DIR=str(ROOT))
    config = subprocess.run(["bash", "-c", 'source "$SCRIPT_DIR/files/load-env.sh" || exit; printf "%s\n%s\n%s\n%s" "$WORKER_USER" "$WORKER_IP" "$PORT" "$SERVED_MODEL_NAME"'],
                            env=env, text=True, capture_output=True, check=True).stdout.splitlines()
    user, host, port, model = config
    worker = f"{user}@{host}" if user else host
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", worker) or worker.startswith("-"):
        raise ValueError("Invalid worker address")
    port = int(port)
    try:
        # Three attempts over this supervisor lifetime; systemd must not retry forever.
        for attempt in range(1, 4):
            if STOP.is_set():
                return 0
            notify(f"starting attempt {attempt}/3")
            if attempt > 1:
                stop_cluster()
                if STOP.wait(30):
                    return 0
            if not wait_for_worker(worker):
                notify("worker unavailable after startup grace")
                continue
            if not launch():
                continue
            policy = HealthPolicy(model)
            announced = False
            while not STOP.is_set():
                status = policy.observe(rank_alive(), rank_alive(worker), identities(port))
                if status == "recover":
                    notify("rank failure or wrong model; recovering cluster")
                    break
                if status == "ready" and not announced:
                    notify(f"ready: {model}", ready=True)
                    announced = True
                elif status == "unready":
                    notify("unready: checking ranks; live prefills are preserved")
                    announced = False
                STOP.wait(10)
            if STOP.is_set():
                return 0
        notify("failed: restart budget exhausted")
        return 1
    finally:
        if not stop_cluster():
            notify("failed: cluster stop could not be confirmed")
            raise RuntimeError("Cluster stop could not be confirmed")
        if STOP.is_set():
            notify("stopped intentionally")

def main():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return supervise()


if __name__ == "__main__":
    raise SystemExit(main())
