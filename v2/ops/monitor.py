"""ART² v2 Monitor: live dashboard for Akash GPU training runs.

Usage:
    python v2/ops/monitor.py              # One-shot status
    python v2/ops/monitor.py --watch      # Auto-refresh every 30s
    python v2/ops/monitor.py --watch 10   # Auto-refresh every 10s

Shows:
- GPU health (utilization, memory, temperature, power)
- Training process status (running/idle, PID, uptime)
- Session state (experiments, best score, streaks, limits)
- Experiment history with scores
- Live log tail from the GPU
- Score trend sparkline

Reads SSH connection info from .deploy-state (written by deploy.sh boot).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPLOY_STATE = os.path.join(PROJECT_ROOT, ".deploy-state")
SSH_PASS = os.environ.get("DEPLOY_SSH_PASS", "autoresearch2026")

W = 70  # dashboard width


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def load_deploy_state() -> dict | None:
    """Load .deploy-state written by deploy.sh boot."""
    if not os.path.exists(DEPLOY_STATE):
        return None
    state = {}
    with open(DEPLOY_STATE) as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                state[k.strip()] = v.strip()
    if state.get("SSH_HOST") and state.get("SSH_PORT"):
        return state
    return None


def ssh_cmd(host: str, port: int, cmd: str, timeout: int = 20) -> str | None:
    """Run a command on the Akash GPU via SSH. Returns stdout or None on failure."""
    env = {**os.environ, "SSHPASS": SSH_PASS}
    ssh_opts = (
        "-o StrictHostKeyChecking=no "
        "-o UserKnownHostsFile=/dev/null "
        "-o LogLevel=ERROR "
        "-o ConnectTimeout=10 "
        "-o ServerAliveInterval=15 "
        "-o ServerAliveCountMax=2 "
        "-o PubkeyAuthentication=no"
    ).split()
    try:
        r = subprocess.run(
            ["sshpass", "-e", "ssh"] + ssh_opts + [
                "-p", str(port), f"root@{host}", f"bash -c {cmd!r}",
            ],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Remote data fetching (single SSH call for efficiency)
# ---------------------------------------------------------------------------

def fetch_remote_data(host: str, port: int) -> dict:
    """Fetch all dashboard data from the Akash GPU in one SSH call."""
    remote_script = (
        # GPU info
        'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit '
        '--format=csv,noheader,nounits 2>/dev/null; '
        'echo "---SEP---"; '

        # Training process
        'PID=$(pgrep -f "[r]un_experiment.py" | head -1); '
        'if [ -n "$PID" ]; then '
        '  UPTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d " "); '
        '  echo "running|$PID|$UPTIME"; '
        'else echo "idle||"; fi; '
        'echo "---SEP---"; '

        # Session state
        'cat /root/v2/.inner_loop_state.json 2>/dev/null || echo "{}"; '
        'echo "---SEP---"; '

        # Results TSV
        'cat /root/v2/results.tsv 2>/dev/null || echo ""; '
        'echo "---SEP---"; '

        # Log tail
        'tail -20 /root/run.log 2>/dev/null || echo "(no log)"; '
    )

    raw = ssh_cmd(host, port, remote_script, timeout=25)
    if not raw:
        return {"ssh_ok": False}

    parts = raw.split("---SEP---")
    data = {"ssh_ok": True}

    # Parse GPU
    gpu_raw = parts[0].strip() if len(parts) > 0 else ""
    if gpu_raw:
        try:
            p = [x.strip() for x in gpu_raw.split(",")]
            data["gpu"] = {
                "name": p[0] if len(p) > 0 else "?",
                "mem_used": float(p[1]) if len(p) > 1 else 0,
                "mem_total": float(p[2]) if len(p) > 2 else 0,
                "util": float(p[3]) if len(p) > 3 else 0,
                "temp": float(p[4]) if len(p) > 4 else 0,
                "power": float(p[5]) if len(p) > 5 else 0,
                "power_limit": float(p[6]) if len(p) > 6 else 0,
            }
        except (ValueError, IndexError):
            data["gpu"] = None
    else:
        data["gpu"] = None

    # Parse process
    proc_raw = parts[1].strip() if len(parts) > 1 else ""
    if proc_raw:
        pp = proc_raw.split("|")
        data["process"] = {
            "status": pp[0] if len(pp) > 0 else "unknown",
            "pid": pp[1] if len(pp) > 1 else "",
            "uptime": pp[2] if len(pp) > 2 else "",
        }
    else:
        data["process"] = {"status": "unknown", "pid": "", "uptime": ""}

    # Parse session state
    state_raw = parts[2].strip() if len(parts) > 2 else "{}"
    try:
        data["session"] = json.loads(state_raw)
    except json.JSONDecodeError:
        data["session"] = {}

    # Parse results TSV
    tsv_raw = parts[3].strip() if len(parts) > 3 else ""
    data["results"] = parse_results_tsv(tsv_raw)

    # Log tail
    data["log_tail"] = parts[4].strip() if len(parts) > 4 else "(no log)"

    return data


def parse_results_tsv(text: str) -> list[dict]:
    """Parse results.tsv text into list of dicts."""
    rows = []
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        row = {}
        for i, col in enumerate(header):
            row[col] = parts[i] if i < len(parts) else ""
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def bar(label: str = "") -> str:
    if label:
        pad = W - len(label) - 4
        return f"== {label} " + "=" * max(pad, 2)
    return "=" * W


def gpu_bar(used: float, total: float, width: int = 30) -> str:
    """ASCII bar for GPU memory usage."""
    if total <= 0:
        return "[?]"
    ratio = used / total
    filled = int(ratio * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {used:.0f}/{total:.0f} MB"


def score_sparkline(results: list[dict], width: int = 40) -> str:
    """ASCII sparkline of scores."""
    scores = []
    for r in results:
        try:
            scores.append(float(r.get("score", 0)))
        except (ValueError, TypeError):
            scores.append(0.0)
    if not scores:
        return "(no data)"
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return "=" * min(len(scores), width)
    blocks = " _.-=*#"
    line = []
    for s in scores[-width:]:
        idx = int((s - mn) / (mx - mn) * (len(blocks) - 1))
        idx = max(0, min(idx, len(blocks) - 1))
        line.append(blocks[idx])
    return "".join(line)


# ---------------------------------------------------------------------------
# Dashboard renderer
# ---------------------------------------------------------------------------

def render_dashboard(data: dict, host: str, port: int):
    """Render the dashboard to stdout."""
    # Clear screen
    print("\033[2J\033[H", end="")

    print(bar())
    print(f"  ART2 v2 Monitor — {host}:{port}")
    print(bar())

    if not data.get("ssh_ok"):
        print("\n  SSH FAILED — cannot reach Akash GPU")
        print(f"  Host: {host}:{port}")
        print(f"  Check: ./v2/ops/deploy.sh ssh")
        print(f"\n{bar()}")
        print(f"  {time.strftime('%H:%M:%S')}")
        print(bar())
        return

    # --- GPU ---
    print(f"\n{bar('GPU')}")
    gpu = data.get("gpu")
    if gpu:
        print(f"  {gpu['name']}")
        print(f"  Util:  {gpu['util']:.0f}%")
        print(f"  Mem:   {gpu_bar(gpu['mem_used'], gpu['mem_total'])}")
        print(f"  Temp:  {gpu['temp']:.0f}C  |  Power: {gpu['power']:.0f}/{gpu['power_limit']:.0f} W")
    else:
        print("  GPU info unavailable")

    # --- Process ---
    print(f"\n{bar('Training Process')}")
    proc = data.get("process", {})
    if proc.get("status") == "running":
        print(f"  Status:  RUNNING")
        print(f"  PID:     {proc['pid']}")
        print(f"  Uptime:  {proc['uptime']}")
    else:
        print(f"  Status:  IDLE (no experiment running)")

    # --- Session State ---
    print(f"\n{bar('Session')}")
    session = data.get("session", {})
    if session:
        exp_count = session.get("experiment_count", 0)
        best_score = session.get("best_score", 0)
        best_exp = session.get("best_experiment_num", 0)
        best_artifact = session.get("best_artifact_id", "")
        streak = session.get("no_improve_streak", 0)
        crash_streak = session.get("crash_streak", 0)
        stopped = session.get("stopped", False)
        stop_reason = session.get("stop_reason", "")
        kept = session.get("kept_count", 0)

        elapsed_h = 0
        if session.get("session_start"):
            elapsed_h = (time.time() - session["session_start"]) / 3600

        print(f"  Experiments: {exp_count} / 50")
        print(f"  Elapsed:     {elapsed_h:.1f}h / 6.0h")
        if isinstance(best_score, (int, float)):
            print(f"  Best Score:  {best_score:.4f} (exp #{best_exp})")
        else:
            print(f"  Best Score:  {best_score}")
        if best_artifact:
            print(f"  Best Model:  {best_artifact}")
        print(f"  Kept: {kept}  |  Reverted: {exp_count - kept}  |  No-improve streak: {streak}")
        if crash_streak > 0:
            print(f"  Crash streak: {crash_streak}")
        if stopped:
            print(f"  ** STOPPED: {stop_reason} **")
    else:
        print("  (no session state yet — no experiments run)")

    # --- Experiment History ---
    results = data.get("results", [])
    print(f"\n{bar(f'Experiments ({len(results)} total)')}")
    if results:
        shown = results[-15:]
        for r in shown:
            exp = r.get("experiment", "?")
            score = r.get("score", "?")
            status = r.get("status", "?")
            desc = r.get("description", "")[:50]

            if status == "keep":
                marker = "+"
            elif status == "crash":
                marker = "X"
            else:
                marker = "-"

            print(f"  {marker} {exp:<16} score={score:<12} [{status}] {desc}")

        # Score trend
        print(f"\n  Trend: {score_sparkline(results)}")

    else:
        print("  (no experiments yet)")

    # --- Log Tail ---
    print(f"\n{bar('Log Tail (last 20 lines)')}")
    log_tail = data.get("log_tail", "(no log)")
    for line in log_tail.split("\n")[-20:]:
        print(f"  {line}")

    # --- Footer ---
    print(f"\n{bar()}")
    print(f"  {time.strftime('%H:%M:%S')}  |  SSH: {host}:{port}")
    print(bar())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ART2 v2 Monitor — Akash GPU dashboard")
    parser.add_argument("--watch", nargs="?", const=30, type=int, metavar="SECS",
                        help="Auto-refresh (default: every 30s)")
    parser.add_argument("--host", help="Override SSH host (default: from .deploy-state)")
    parser.add_argument("--port", type=int, help="Override SSH port")
    args = parser.parse_args()

    # Load connection info
    deploy = load_deploy_state()
    host = args.host or (deploy.get("SSH_HOST") if deploy else None)
    port = args.port or (int(deploy.get("SSH_PORT", 22)) if deploy else 22)

    if not host:
        print("No Akash deployment found.")
        print("Either run './v2/ops/deploy.sh boot' first, or use --host/--port.")
        sys.exit(1)

    if args.watch is not None:
        interval = max(args.watch, 5)
        try:
            while True:
                data = fetch_remote_data(host, port)
                render_dashboard(data, host, port)
                time.sleep(interval)

                # Re-read .deploy-state each cycle to pick up new deployments
                if not args.host:
                    new_deploy = load_deploy_state()
                    if new_deploy:
                        new_host = new_deploy.get("SSH_HOST")
                        new_port = int(new_deploy.get("SSH_PORT", 22))
                        if new_host != host or new_port != port:
                            host, port = new_host, new_port
                            print(f"\n[monitor] Deployment changed: now {host}:{port}")
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
    else:
        data = fetch_remote_data(host, port)
        render_dashboard(data, host, port)


if __name__ == "__main__":
    main()
