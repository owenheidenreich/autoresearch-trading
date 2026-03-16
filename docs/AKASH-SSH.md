# Akash SSH Port Discovery

> **Learned:** March 14, 2026 — after the old container died and a new DSEQ was deployed

## The Problem

Akash Network maps container port 22 (SSH) to a **random high port** on the provider host.
The Akash Console UI sometimes fails to display the forwarded port number, leaving you
unable to SSH into your container.

This happened with DSEQ `25937123` on provider `provider.h100.wdc.hh.akash.pub`.

## The Solution: Parallel Port Scan

Akash providers typically forward to ports in the **30000–33000** range. A fast parallel
socket scan finds which ports are open, then banner-grabbing identifies the SSH service,
and password auth confirms which one is yours.

### Step 1 — Scan for open ports

```python
import socket, concurrent.futures

host = 'provider.h100.wdc.hh.akash.pub'

def check(port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        result = s.connect_ex((host, port))
        s.close()
        return port if result == 0 else None
    except:
        return None

with concurrent.futures.ThreadPoolExecutor(max_workers=200) as ex:
    futures = {ex.submit(check, p): p for p in range(30000, 33001)}
    for f in concurrent.futures.as_completed(futures):
        r = f.result()
        if r:
            print(f'OPEN: {r}')
```

This takes ~2 seconds and found 5 open ports: 30749, 30995, 31838, 32230, 32674.

### Step 2 — Banner-grab to find SSH

```bash
for p in 30749 30995 31838 32230 32674; do
  echo -n "Port $p: "
  echo "" | nc -w 2 provider.h100.wdc.hh.akash.pub $p 2>&1 | head -1
done
```

SSH banners (`SSH-2.0-OpenSSH_*`) appeared on 30749, 30995, and 32230 — multiple
containers run on the same provider host.

### Step 3 — Try your password on each SSH port

```bash
for p in 30749 30995 32230; do
  SSHPASS='autoresearch2026' sshpass -e ssh \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=5 -p $p root@provider.h100.wdc.hh.akash.pub \
    "echo CONNECTED; nvidia-smi --query-gpu=name --format=csv,noheader; hostname" 2>&1
done
```

Port **30995** accepted the password and showed `NVIDIA H100 80GB HBM3`,
hostname `trainer-866bd4d574-fxr6p`. The other ports rejected authentication
(they belong to different tenants' containers).

## Key Facts

- Provider hosts run **many** containers — expect multiple SSH ports open
- Port mapping changes on **every new deployment** (and on container restarts
  that get rescheduled to a different pod)
- The container's **internal** port is always 22; only the external mapping varies
- `nmap` isn't installed on macOS by default; the Python scan above works without it
- Provider port range is typically 30000–33000

## Container Restart Behavior

Akash containers are **ephemeral** — when a pod restarts, **all files in `/root/` are lost**.
This means:
- Uploaded code (train.py, prepare.py, etc.) must be re-uploaded after any restart
- pip-installed packages must be re-installed
- The SSH port mapping may or may not change depending on whether it's a restart
  vs. a reschedule

The `deploy-autoresearch.yaml` startup command installs packages automatically on boot,
but uploaded code/data must be re-sent via SCP.
