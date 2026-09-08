# Service recovery

`qwen-vllm.service` owns the two-node cluster. The supervisor makes three launch
attempts per run, with a 30-second delay between retries. After an unsuccessful
run or supervisor crash, systemd waits five minutes and starts a fresh run.
Retries continue through prolonged outages. An explicit `systemctl stop` does
not trigger an automatic restart. API timeouts alone do not restart live ranks,
because a long prefill may temporarily delay health responses.

For an existing installation, install `deploy/systemd/qwen-vllm.service.d/20-recovery.conf`
as `/etc/systemd/system/qwen-vllm.service.d/20-recovery.conf`, run
`sudo systemctl daemon-reload`, and `sudo systemctl enable qwen-vllm.service`.
These operations do not restart an active cluster. New installations can use
the base unit, which contains the same recovery policy.

The service reads `/home/kesslerio/.config/qwen-cluster/active.env`; that file
selects the recipe environment/profile. Model identity and API port are read
from the selected configuration rather than hardcoded in the supervisor.
To switch compatible models, update the selected configuration and restart
this same service during a suitable interruption window. Do not enable a
second model service on the same GPUs/port. This recipe's launch/stop scripts
and Docker container name remain recipe-specific: unrelated model runtimes
require a compatible launcher, not merely a different model ID.

Verify effective settings with `systemctl show qwen-vllm.service -p Restart
-p RestartUSec -p UnitFileState -p ActiveState`. Boot enablement and configured
recovery are not evidence of a completed reboot or failure-injection drill.
