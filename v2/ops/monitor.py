#!/usr/bin/env python3
"""ART2 v2 Web Dashboard -- live browser-based training monitor.

Usage:
    python v2/ops/monitor.py                  # Auto-detect remote from .deploy-state
    python v2/ops/monitor.py --port 8420      # Custom port (default: 8420)
    python v2/ops/monitor.py --local          # Local only (no SSH)
    python v2/ops/monitor.py --no-open        # Don't auto-open browser

Opens a browser dashboard with live-updating panels:
  - GPU health gauges (utilization, VRAM, temp, power)
  - Session state (experiments, best score, streaks, limits)
  - Experiment history table with scores
  - Score/metric charts over time
  - Live log tail from the GPU
  - Current train.py source code
  - Score trend sparkline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
SSH_PASS = os.environ.get("DEPLOY_SSH_PASS", "autoresearch2026")
RESULTS_TSV = PROJECT_ROOT / "v2" / "results.tsv"
LAB_NOTEBOOK = PROJECT_ROOT / "v2" / "lab_notebook.md"

# Self-restart on source change
_SELF_PATH = Path(__file__).resolve()
_SELF_HASH_AT_START = hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
_restart_flag = threading.Event()


def _check_self_restart():
    try:
        current_hash = hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
        if current_hash != _SELF_HASH_AT_START:
            print("[monitor] Source changed on disk -- restarting...", flush=True)
            _restart_flag.set()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared state (updated by background poller)
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: dict = {
    "experiments": [],
    "session": {},
    "log_tail": None,
    "gpu": None,
    "train_py": None,
    "train_py_source": "",
    "mode": "initializing",
    "run_state": "unknown",
    "last_fetch": 0,
    "fetch_time": 0,
    "poll_count": 0,
    "ssh_ok": False,
    "lab_notebook": None,
}


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------

def load_deploy_state() -> dict | None:
    if not DEPLOY_STATE.exists():
        return None
    state = {}
    for line in DEPLOY_STATE.read_text().strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            state[k.strip()] = v.strip()
    return state if state.get("SSH_HOST") else None


def _ssh_cmd(host: str, port: int, cmd: str, timeout: int = 20) -> str | None:
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
# Data fetching
# ---------------------------------------------------------------------------

def fetch_remote_data(host: str, port: int) -> dict:
    """Fetch all dashboard data from the Akash GPU in one SSH call."""
    remote_script = (
        # GPU info
        'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit '
        '--format=csv,noheader,nounits 2>/dev/null; '
        'echo "---SEP---"; '

        # Training process (Claude drives the loop, run_experiment.py is what runs)
        'PID=$(pgrep -f "[r]un_experiment" | head -1); '
        'if [ -n "$PID" ]; then '
        '  UPTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d " "); '
        '  echo "running|$PID|$UPTIME"; '
        'else echo "idle||"; fi; '
        'echo "---SEP---"; '

        # Results TSV (remote copy, may lag behind local)
        'cat /root/v2/results.tsv 2>/dev/null || echo ""; '
        'echo "---SEP---"; '

        # Log tail
        'tail -80 /root/run.log 2>/dev/null || echo "(no log)"; '
        'echo "---SEP---"; '

        # train.py source
        'cat /root/v2/train.py 2>/dev/null || echo ""; '
    )

    raw = _ssh_cmd(host, port, remote_script, timeout=25)
    if not raw:
        return {"ssh_ok": False}

    parts = raw.split("---SEP---")
    data: dict = {"ssh_ok": True}

    # Parse GPU
    gpu_raw = parts[0].strip() if len(parts) > 0 else ""
    data["gpu"] = _parse_gpu_csv(gpu_raw)

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

    # Parse results TSV (now part index 2, no more session state section)
    tsv_raw = parts[2].strip() if len(parts) > 2 else ""
    data["experiments"] = _parse_results_tsv(tsv_raw)

    # Log tail
    data["log_tail"] = parts[3].strip() if len(parts) > 3 else None

    # train.py
    data["train_py"] = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None

    return data


def _parse_gpu_csv(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        p = [x.strip() for x in text.split(",")]
        return {
            "name": p[0] if len(p) > 0 else "unknown",
            "mem_used_mb": float(p[1]) if len(p) > 1 else 0,
            "mem_total_mb": float(p[2]) if len(p) > 2 else 0,
            "util_pct": float(p[3]) if len(p) > 3 else 0,
            "temp_c": float(p[4]) if len(p) > 4 else 0,
            "power_w": float(p[5]) if len(p) > 5 else 0,
            "power_limit_w": float(p[6]) if len(p) > 6 else 0,
        }
    except (ValueError, IndexError):
        return None


def _parse_results_tsv(text: str) -> list[dict]:
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
        # Parse score as float
        try:
            row["score_num"] = float(row.get("score", "-999"))
        except (ValueError, TypeError):
            row["score_num"] = -999.0
        # Parse win rate from description (e.g. "wr=53.6%")
        import re
        wr_match = re.search(r'wr=([\d.]+)%', row.get("description", ""))
        row["win_rate"] = float(wr_match.group(1)) if wr_match else None
        rows.append(row)
    return rows


def _enrich_timestamps(experiments: list[dict]) -> None:
    """Add timestamps from manifest.json inside each artifact directory."""
    from datetime import datetime, timezone, timedelta
    artifacts_dir = PROJECT_ROOT / "v2" / "artifacts"
    for exp in experiments:
        exp_name = exp.get("experiment", "")
        manifest = artifacts_dir / exp_name / "manifest.json"
        if manifest.is_file():
            try:
                data = json.loads(manifest.read_text())
                ts_str = data.get("timestamp")
                if ts_str:
                    # GPU records local time ~1hr behind UTC; correct to true UTC
                    dt = datetime.fromisoformat(ts_str).replace(tzinfo=timezone(timedelta(hours=-1)))
                    exp["timestamp"] = dt.timestamp()
                else:
                    exp["timestamp"] = None
            except (OSError, json.JSONDecodeError, ValueError):
                exp["timestamp"] = None
        else:
            exp["timestamp"] = None


def fetch_local_gpu() -> dict | None:
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return _parse_gpu_csv(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def fetch_local_fallback() -> dict:
    """Read local files when SSH is unavailable."""
    data: dict = {"ssh_ok": False}

    # Local results.tsv
    if RESULTS_TSV.exists():
        data["experiments"] = _parse_results_tsv(RESULTS_TSV.read_text())
    else:
        data["experiments"] = []

    # Local train.py
    train_path = PROJECT_ROOT / "v2" / "train.py"
    if train_path.exists():
        data["train_py"] = train_path.read_text()
        data["train_py_source"] = "local: v2/train.py"
    else:
        data["train_py"] = None
        data["train_py_source"] = ""

    # Local GPU
    data["gpu"] = fetch_local_gpu()

    data["session"] = {}
    data["log_tail"] = None
    data["process"] = {"status": "unknown", "pid": "", "uptime": ""}

    return data


def derive_session_from_experiments(experiments: list[dict]) -> dict:
    """Derive session stats from results.tsv rows (replaces .inner_loop_state.json)."""
    if not experiments:
        return {}

    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("status") == "keep")
    reverted = total - kept

    scores = [e["score_num"] for e in experiments if e.get("score_num", -999) > -999]
    best_score = max(scores) if scores else None

    # Find which experiment had the best score
    best_exp = ""
    if best_score is not None:
        for e in experiments:
            if e.get("score_num") == best_score:
                best_exp = e.get("experiment", "")
                break

    # No-improve streak: count consecutive non-keep from end
    streak = 0
    for e in reversed(experiments):
        if e.get("status") == "keep":
            break
        streak += 1

    return {
        "experiment_count": total,
        "kept_count": kept,
        "reverted_count": reverted,
        "best_score": best_score,
        "best_experiment_num": best_exp,
        "no_improve_streak": streak,
    }


def load_lab_notebook() -> str | None:
    if LAB_NOTEBOOK.exists():
        try:
            return LAB_NOTEBOOK.read_text()
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------

def poller_loop(host: str | None, port: int, local_only: bool, interval: int,
                host_pinned: bool = False):
    mode = f"Remote: {host}:{port}" if host else "Local"
    set_state(mode=mode)

    while True:
        _check_self_restart()

        try:
            t0 = time.time()

            # Reload .deploy-state each poll to pick up new deployments
            if not host_pinned and not local_only:
                ds = load_deploy_state()
                if ds:
                    new_host = ds.get("SSH_HOST")
                    new_port = int(ds.get("SSH_PORT", 22))
                else:
                    new_host, new_port = None, 22
                if new_host != host or new_port != port:
                    print(f"[monitor] Deployment changed: {host}:{port} -> {new_host}:{new_port}", flush=True)
                    host, port = new_host, new_port
                new_mode = f"Remote: {host}:{port}" if host else "Local"
                if new_mode != mode:
                    mode = new_mode
                    set_state(mode=mode)

            ssh_ok = False
            log_tail = None
            gpu = None
            train_py = None
            train_py_source = ""
            process = {"status": "unknown", "pid": "", "uptime": ""}

            # Local results.tsv is authoritative (Claude drives the loop locally)
            if RESULTS_TSV.exists():
                experiments = _parse_results_tsv(RESULTS_TSV.read_text())
                _enrich_timestamps(experiments)
            else:
                experiments = []

            # Local train.py is authoritative (Claude edits it locally)
            train_path = PROJECT_ROOT / "v2" / "train.py"
            if train_path.exists():
                train_py = train_path.read_text()
                train_py_source = "local: v2/train.py"

            # Try remote for GPU stats, process status, and log tail
            if host and not local_only:
                remote = fetch_remote_data(host, port)
                ssh_ok = remote.get("ssh_ok", False)
                if ssh_ok:
                    log_tail = remote.get("log_tail")
                    gpu = remote.get("gpu")
                    process = remote.get("process", process)

            # Local GPU fallback (MacBook won't have one, but just in case)
            if not gpu:
                gpu = fetch_local_gpu()

            # Derive session stats from results.tsv
            session = derive_session_from_experiments(experiments)

            # Classify run state
            if process.get("status") == "running":
                run_state = "active"
            elif host and not ssh_ok:
                run_state = "ssh_fail"
            elif not host:
                run_state = "local"
            else:
                run_state = "idle"

            # Lab notebook (always local)
            lab_notebook = load_lab_notebook()

            fetch_time = time.time() - t0
            poll_count = get_state()["poll_count"] + 1

            set_state(
                experiments=experiments,
                session=session,
                log_tail=log_tail,
                gpu=gpu,
                train_py=train_py,
                train_py_source=train_py_source,
                process=process,
                run_state=run_state,
                last_fetch=time.time(),
                fetch_time=fetch_time,
                poll_count=poll_count,
                ssh_ok=ssh_ok,
                lab_notebook=lab_notebook,
            )
        except Exception as e:
            print(f"[poller] Error: {e}", file=sys.stderr)

        time.sleep(interval)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ART2 v2 Training Monitor</title>
<style>
  :root {
    --bg: #0d1117;
    --bg2: #161b22;
    --bg3: #21262d;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --cyan: #58a6ff;
    --orange: #d18616;
    --purple: #bc8cff;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', 'Menlo', monospace;
    font-size: 13px;
    line-height: 1.5;
  }
  header {
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    padding: 6px 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  header h1 {
    font-size: 14px;
    color: var(--cyan);
    font-weight: 700;
    letter-spacing: 1px;
  }
  header .meta {
    color: var(--text-dim);
    font-size: 11px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 8px;
    padding: 8px;
    max-width: 100vw;
    overflow: hidden;
  }
  .full-row { grid-column: 1 / 3; min-width: 0; }
  .panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .panel-header {
    background: var(--bg3);
    padding: 5px 10px;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--cyan);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-body {
    padding: 8px 10px;
    flex: 1;
    overflow: auto;
  }

  /* Row 1: GPU (narrow) + Session (wide) */
  .top-row {
    grid-column: 1 / 3;
    display: flex;
    gap: 8px;
    min-width: 0;
  }
  .top-row .gpu-status { flex: 0 0 240px; min-width: 0; }
  .top-row .session-status { flex: 1; min-width: 0; }

  /* GPU gauges */
  .gauge-row { display: flex; gap: 6px; margin-bottom: 5px; align-items: center; }
  .gauge-label { width: 36px; font-size: 10px; color: var(--text-dim); }
  .gauge-bar { flex: 1; height: 14px; background: var(--bg); border-radius: 3px; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .gauge-value { width: 70px; text-align: right; font-size: 11px; font-weight: 600; }
  .fill-green { background: var(--green); }
  .fill-yellow { background: var(--yellow); }
  .fill-red { background: var(--red); }
  .fill-cyan { background: var(--cyan); }

  /* Stat cards */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 4px; }
  .stat-card {
    background: var(--bg);
    border-radius: 4px;
    padding: 4px 6px;
    text-align: center;
  }
  .stat-card[title] { cursor: help; }
  .stat-value { font-size: 15px; font-weight: 700; }
  .stat-label { font-size: 8px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-green { color: var(--green); }
  .stat-red { color: var(--red); }
  .stat-yellow { color: var(--yellow); }
  .stat-cyan { color: var(--cyan); }
  .stat-purple { color: var(--purple); }

  /* Progress bar */
  .progress-outer {
    width: 100%;
    height: 8px;
    background: var(--bg);
    border-radius: 4px;
    overflow: hidden;
    margin: 6px 0;
  }
  .progress-inner {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--green));
    border-radius: 4px;
    transition: width 0.5s;
  }

  /* Phase indicator */
  .phase-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .phase-active { background: var(--green); }
  .phase-completed { background: var(--text-dim); animation: none; }
  .phase-idle { background: var(--text-dim); animation: none; }
  .phase-ssh_fail { background: var(--red); animation: none; }
  .phase-local { background: var(--yellow); animation: none; }

  .run-state-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-left: 8px;
  }
  .run-state-active { background: rgba(63,185,80,0.2); color: var(--green); }
  .run-state-completed { background: rgba(139,148,158,0.2); color: var(--text-dim); }
  .run-state-idle { background: rgba(139,148,158,0.15); color: var(--text-dim); }
  .run-state-ssh_fail { background: rgba(248,81,73,0.2); color: var(--red); }
  .run-state-local { background: rgba(210,169,34,0.2); color: var(--yellow); }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: left;
    padding: 6px 8px;
    color: var(--text-dim);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: var(--bg2);
  }
  td { padding: 5px 8px; border-bottom: 1px solid var(--bg3); white-space: nowrap; }
  tr:hover { background: var(--bg3); }
  tr.kept td { color: var(--green); }
  tr.crash td { color: var(--red); }
  tr.reverted td { color: var(--text-dim); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.desc { white-space: normal; max-width: 300px; overflow: hidden; text-overflow: ellipsis; color: var(--text-dim); font-size: 11px; }

  /* Charts */
  .charts-panel { grid-column: 1 / 3; }
  .chart-rows { display: flex; flex-direction: column; gap: 6px; }
  .chart-container { display: flex; gap: 8px; height: 80px; }
  .chart-box { flex: 1; position: relative; min-width: 0; }
  .chart-title { font-size: 9px; color: var(--text-dim); margin-bottom: 1px; text-transform: uppercase; letter-spacing: 0.5px; }
  canvas { width: 100% !important; height: 66px !important; }

  /* Bottom: left stack + log/code */
  .left-stack { grid-column: 1 / 2; display: flex; flex-direction: column; gap: 8px; min-width: 0; overflow: hidden; }
  .right-stack { grid-column: 2 / 3; display: flex; flex-direction: column; gap: 8px; min-width: 0; overflow: hidden; }

  /* Code viewer */
  .code-view {
    background: var(--bg);
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.4;
    overflow: auto;
    white-space: pre;
    tab-size: 4;
    max-width: 100%;
    width: 0;
    min-width: 100%;
  }
  .code-view .ln { color: var(--text-dim); user-select: none; display: inline-block; width: 40px; text-align: right; margin-right: 12px; }

  /* Log viewer */
  .log-view {
    background: var(--bg);
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.6;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-all;
    max-width: 100%;
    width: 0;
    min-width: 100%;
  }
  .log-error { color: var(--red); }
  .log-success { color: var(--green); }
  .log-warn { color: var(--yellow); }
  .log-info { color: var(--cyan); }
  .log-dim { color: var(--text-dim); }

  /* Tabs */
  .tab-bar {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
  }
  .tab {
    padding: 6px 14px;
    font-size: 11px;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.2s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .stale-warning {
    background: rgba(248,81,73,0.15);
    color: var(--red);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    animation: pulse 2s ease-in-out infinite;
  }

  /* Lab notebook */
  .notebook-view {
    background: var(--bg);
    border-radius: 4px;
    padding: 10px;
    font-size: 11px;
    line-height: 1.6;
    overflow: auto;
    white-space: pre-wrap;
    max-width: 100%;
    width: 0;
    min-width: 100%;
  }

  /* Responsive */
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .top-row { flex-direction: column; }
    .top-row .gpu-status { flex: none; }
    .full-row, .charts-panel, .left-stack, .right-stack { grid-column: 1 / 2; }
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:12px">
    <h1>ART2 v2 TRAINING MONITOR</h1>
  </div>
  <div class="meta">
    <span id="header-mode"></span> &nbsp;|&nbsp;
    <span id="header-time"></span> &nbsp;|&nbsp;
    <span id="header-poll"></span> &nbsp;|&nbsp;
    <span id="header-fetch"></span>
    <span id="header-stale"></span>
  </div>
</header>

<div class="grid">
  <!-- Row 1: GPU (compact) + Session -->
  <div class="top-row">
    <div class="panel gpu-status">
      <div class="panel-header">GPU <span id="gpu-name" style="font-weight:400"></span></div>
      <div class="panel-body" style="padding:6px 8px">
        <div id="gpu-content">
          <div class="gauge-row">
            <span class="gauge-label">Util</span>
            <div class="gauge-bar"><div class="gauge-fill fill-green" id="gpu-util-bar" style="width:0%"></div></div>
            <span class="gauge-value" id="gpu-util-val">--</span>
          </div>
          <div class="gauge-row">
            <span class="gauge-label">VRAM</span>
            <div class="gauge-bar"><div class="gauge-fill fill-cyan" id="gpu-mem-bar" style="width:0%"></div></div>
            <span class="gauge-value" id="gpu-mem-val">--</span>
          </div>
          <div class="gauge-row">
            <span class="gauge-label">Temp</span>
            <div class="gauge-bar"><div class="gauge-fill fill-yellow" id="gpu-temp-bar" style="width:0%"></div></div>
            <span class="gauge-value" id="gpu-temp-val">--</span>
          </div>
          <div class="gauge-row">
            <span class="gauge-label">Power</span>
            <div class="gauge-bar"><div class="gauge-fill fill-green" id="gpu-power-bar" style="width:0%"></div></div>
            <span class="gauge-value" id="gpu-power-val">--</span>
          </div>
        </div>
      </div>
    </div>
    <div class="panel session-status">
      <div class="panel-header">Session <span id="run-state-badge" class="run-state-badge"></span></div>
      <div class="panel-body" style="padding:6px 10px">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <span class="phase-dot" id="phase-dot"></span>
          <span id="phase-label" style="font-weight:600;font-size:12px"></span>
          <span id="process-info" style="color:var(--text-dim);margin-left:auto;font-size:11px"></span>
        </div>
        <div class="progress-outer">
          <div class="progress-inner" id="exp-progress" style="width:0%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:var(--text-dim);font-size:10px;margin-bottom:6px;">
          <span id="progress-left"></span>
          <span id="progress-right"></span>
        </div>
        <div class="stats-grid">
          <div class="stat-card" title="Experiments kept (beat previous best)"><div class="stat-value stat-green" id="stat-kept">--</div><div class="stat-label">Kept</div></div>
          <div class="stat-card" title="Experiments reverted"><div class="stat-value stat-red" id="stat-reverted">--</div><div class="stat-label">Reverted</div></div>
          <div class="stat-card" title="Total experiments this session"><div class="stat-value stat-cyan" id="stat-total">--</div><div class="stat-label">Total</div></div>
          <div class="stat-card" title="Best composite score achieved"><div class="stat-value stat-green" id="stat-best">--</div><div class="stat-label">Best Score</div></div>
          <div class="stat-card" title="Consecutive experiments without improvement"><div class="stat-value stat-yellow" id="stat-streak">--</div><div class="stat-label">No-Improve</div></div>
          <div class="stat-card" title="Best experiment number"><div class="stat-value stat-purple" id="stat-best-exp">--</div><div class="stat-label">Best Exp</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 2: Charts -->
  <div class="panel full-row charts-panel">
    <div class="panel-header">Score Trend</div>
    <div class="panel-body" style="padding:6px 10px">
      <div class="chart-rows">
        <div class="chart-container">
          <div class="chart-box"><div class="chart-title">Score</div><canvas id="chart-score"></canvas></div>
          <div class="chart-box"><div class="chart-title">Win Rate %</div><canvas id="chart-winrate"></canvas></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 3 Left: Experiments Table -->
  <div class="left-stack">
    <div class="panel">
      <div class="panel-header">Experiments <span id="exp-count" style="font-weight:400;color:var(--text-dim)"></span></div>
      <div class="panel-body" style="padding:0">
        <div style="overflow:auto;max-height:400px;">
          <table id="exp-table">
            <thead>
              <tr>
                <th>#</th><th>Time</th><th>Status</th><th>Score</th><th>Description</th>
              </tr>
            </thead>
            <tbody id="exp-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 3 Right: Code / Log / Notebook tabs -->
  <div class="right-stack">
    <div class="panel" style="flex:1">
      <div class="panel-header">
        <div class="tab-bar" style="border-bottom:none">
          <div class="tab active" data-tab="code" id="code-tab-label">train.py</div>
          <div class="tab" data-tab="notebook">Lab Notebook</div>
        </div>
      </div>
      <div class="panel-body" style="padding:0">
        <div class="tab-content active" id="tab-code">
          <div class="code-view" id="code-content" style="max-height:400px">Loading...</div>
        </div>
        <div class="tab-content" id="tab-notebook">
          <div class="notebook-view" id="notebook-content" style="max-height:400px">Loading...</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// -- Mini chart library (no dependencies) --
function drawChart(canvasId, data, color, opts = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  const mn = opts.min !== undefined ? opts.min : Math.min(...data);
  const mx = opts.max !== undefined ? opts.max : Math.max(...data);
  const range = mx - mn || 1;
  const pad = 4;

  // Grid lines
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = pad + (h - 2*pad) * i / 3;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Area fill
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  for (let i = 0; i < data.length; i++) {
    const x = pad + (w - 2*pad) * i / (data.length - 1 || 1);
    const y = h - pad - (data[i] - mn) / range * (h - 2*pad);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w - pad, h - pad);
  ctx.closePath();
  ctx.fillStyle = color + '15';
  ctx.fill();

  // Line
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad + (w - 2*pad) * i / (data.length - 1 || 1);
    const y = h - pad - (data[i] - mn) / range * (h - 2*pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();

  // Current value label
  const lastVal = data[data.length - 1];
  ctx.fillStyle = color;
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  ctx.fillText(lastVal.toFixed(opts.decimals !== undefined ? opts.decimals : 3), w - 6, 14);

  // Min/max
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.fillText(mn.toFixed(opts.decimals !== undefined ? opts.decimals : 3), w - 6, h - 2);
}

// -- Tab switching --
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    const parent = tab.closest('.panel');
    parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

function escapeHtml(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
}

// -- Highlight train.py --
function highlightPython(code) {
  const lines = code.split('\n');
  return lines.map((line, i) => {
    const ln = `<span class="ln">${String(i+1).padStart(4)}</span>`;
    let hl = escapeHtml(line);
    hl = hl.replace(/(#.*)$/g, '<span style="color:#8b949e">$1</span>');
    hl = hl.replace(/\b(def|class|import|from|return|if|else|elif|for|while|with|as|try|except|finally|raise|yield|lambda|and|or|not|in|is|None|True|False|self|pass|break|continue)\b/g,
      '<span style="color:#ff7b72">$1</span>');
    hl = hl.replace(/("[^"]*"|'[^']*')/g, '<span style="color:#a5d6ff">$1</span>');
    hl = hl.replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, '<span style="color:#79c0ff">$1</span>');
    hl = hl.replace(/^(\s*@\w+)/g, '<span style="color:#d2a8ff">$1</span>');
    return ln + hl;
  }).join('\n');
}

// -- Colorize log --
function colorizeLine(line) {
  if (line.includes('ERROR') || line.includes('CRASH') || line.includes('FAILED')) return `<span class="log-error">${escapeHtml(line)}</span>`;
  if (line.includes('KEPT') || line.includes('keep') || line.includes('IMPROVED')) return `<span class="log-success">${escapeHtml(line)}</span>`;
  if (line.includes('WARNING') || line.includes('revert') || line.includes('REVERT')) return `<span class="log-warn">${escapeHtml(line)}</span>`;
  if (line.includes('===') || line.includes('score') || line.includes('DECISION')) return `<span class="log-info">${escapeHtml(line)}</span>`;
  return `<span class="log-dim">${escapeHtml(line)}</span>`;
}

function fmtDuration(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(0) + 'm';
  return Math.floor(s/3600) + 'h' + String(Math.floor((s%3600)/60)).padStart(2,'0') + 'm';
}

function setGauge(prefix, pct, text) {
  const bar = document.getElementById(prefix + '-bar');
  const val = document.getElementById(prefix + '-val');
  bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  val.textContent = text;
  bar.className = 'gauge-fill ' + (pct > 80 ? 'fill-red' : pct > 50 ? 'fill-yellow' : 'fill-green');
}

// -- Main update loop --
let lastExpCount = 0;
let lastTrainHash = '';
let lastNotebookHash = '';

async function update() {
  try {
    const resp = await fetch('/api/state');
    const data = await resp.json();

    // Header
    const sshTag = data.mode.startsWith('Remote') ? (data.ssh_ok ? ' [SSH OK]' : ' [SSH FAIL]') : '';
    document.getElementById('header-mode').textContent = data.mode + sshTag;
    document.getElementById('header-time').textContent = new Date().toLocaleTimeString();
    document.getElementById('header-poll').textContent = `poll #${data.poll_count}`;
    document.getElementById('header-fetch').textContent = `fetch: ${data.fetch_time.toFixed(1)}s`;

    // Run state badge
    const runState = data.run_state || 'unknown';
    const badge = document.getElementById('run-state-badge');
    const badgeLabels = {active: 'LIVE', completed: 'DONE', idle: 'IDLE', ssh_fail: 'SSH FAIL', local: 'LOCAL'};
    badge.textContent = badgeLabels[runState] || runState;
    badge.className = 'run-state-badge run-state-' + runState;

    // Phase dot
    const phaseDot = document.getElementById('phase-dot');
    phaseDot.className = 'phase-dot phase-' + runState;
    const phaseLabels = {
      active: 'Training Active',
      completed: 'Session Complete',
      idle: 'GPU Idle',
      ssh_fail: 'SSH Connection Failed',
      local: 'Local Mode (no GPU)',
    };
    document.getElementById('phase-label').textContent = phaseLabels[runState] || runState;

    // Process info
    const proc = data.process || {};
    if (proc.status === 'running') {
      document.getElementById('process-info').textContent = `PID ${proc.pid} | uptime ${proc.uptime}`;
    } else {
      document.getElementById('process-info').textContent = '';
    }

    // GPU
    if (data.gpu) {
      const g = data.gpu;
      document.getElementById('gpu-name').textContent = g.name;
      document.getElementById('gpu-content').style.opacity = '1';
      const memPct = g.mem_total_mb > 0 ? (g.mem_used_mb / g.mem_total_mb * 100) : 0;
      const powerPct = g.power_limit_w > 0 ? (g.power_w / g.power_limit_w * 100) : 0;
      setGauge('gpu-util', g.util_pct, g.util_pct.toFixed(0) + '%');
      setGauge('gpu-mem', memPct, `${(g.mem_used_mb/1024).toFixed(1)}/${(g.mem_total_mb/1024).toFixed(0)}GB`);
      setGauge('gpu-temp', Math.min(g.temp_c, 100), g.temp_c.toFixed(0) + 'C');
      setGauge('gpu-power', powerPct, `${g.power_w.toFixed(0)}/${g.power_limit_w.toFixed(0)}W`);
    } else {
      document.getElementById('gpu-name').textContent = data.ssh_ok ? 'waiting...' : '(no GPU data)';
      document.getElementById('gpu-content').style.opacity = '0.3';
    }

    // Session stats
    const session = data.session || {};
    const expCount = session.experiment_count || 0;
    const kept = session.kept_count || 0;
    const reverted = expCount - kept;
    const bestScore = session.best_score;
    const bestExp = session.best_experiment_num || 0;
    const streak = session.no_improve_streak || 0;

    document.getElementById('stat-kept').textContent = kept;
    document.getElementById('stat-reverted').textContent = reverted;
    document.getElementById('stat-total').textContent = expCount;
    document.getElementById('stat-best').textContent = (bestScore != null && bestScore > -5) ? bestScore.toFixed(4) : '--';
    document.getElementById('stat-streak').textContent = streak + '/8';
    document.getElementById('stat-best-exp').textContent = bestExp > 0 ? '#' + bestExp : '--';

    // Progress bar (experiments out of 50)
    const pct = Math.min(100, Math.max(0, expCount / 50 * 100));
    const progressBar = document.getElementById('exp-progress');
    if (session.stopped) {
      progressBar.style.width = '100%';
      progressBar.style.background = 'var(--text-dim)';
      document.getElementById('progress-left').textContent = 'Session stopped: ' + (session.stop_reason || 'unknown');
      document.getElementById('progress-right').textContent = `${expCount}/50 experiments`;
    } else if (runState === 'active') {
      progressBar.style.width = pct + '%';
      progressBar.style.background = '';
      const elapsedH = session.session_start ? ((Date.now()/1000 - session.session_start) / 3600) : 0;
      document.getElementById('progress-left').textContent = `${expCount}/50 experiments | ${elapsedH.toFixed(1)}h/6h`;
      document.getElementById('progress-right').textContent = streak > 0 ? `no-improve streak: ${streak}` : '';
    } else {
      progressBar.style.width = pct + '%';
      progressBar.style.background = '';
      document.getElementById('progress-left').textContent = `${expCount}/50 experiments`;
      document.getElementById('progress-right').textContent = '';
    }

    // Charts
    const exps = data.experiments || [];
    const scores = exps.map(e => e.score_num).filter(s => s > -999);
    if (scores.length >= 1) {
      drawChart('chart-score', scores, '#3fb950', {decimals: 3});
    }
    // Win rate chart
    const winRates = exps.map(e => e.win_rate).filter(w => w != null);
    if (winRates.length >= 1) {
      drawChart('chart-winrate', winRates, '#bc8cff', {decimals: 1, min: Math.min(...winRates) - 1, max: Math.max(...winRates) + 1});
    }

    // Experiments table
    if (exps.length !== lastExpCount) {
      lastExpCount = exps.length;
      document.getElementById('exp-count').textContent = `(${exps.length})`;
      const tbody = document.getElementById('exp-tbody');
      tbody.innerHTML = '';
      for (let i = exps.length - 1; i >= 0; i--) {
        const e = exps[i];
        const tr = document.createElement('tr');
        const status = e.status || '?';
        tr.className = status === 'keep' ? 'kept' : status === 'crash' ? 'crash' : 'reverted';
        const statusDisplay = status === 'keep' ? '+ KEEP' : status === 'crash' ? 'x CRASH' : '~ revert';
        const scoreDisplay = e.score_num > -999 ? e.score_num.toFixed(4) : 'FAIL';
        const timeDisplay = e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '';
        tr.innerHTML = `
          <td class="num">${escapeHtml(e.experiment || '?')}</td>
          <td class="num" style="color:var(--text-dim)">${timeDisplay}</td>
          <td>${statusDisplay}</td>
          <td class="num">${scoreDisplay}</td>
          <td class="desc">${escapeHtml(e.description || '')}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // train.py code
    if (data.train_py) {
      const hash = data.train_py.length + data.train_py.slice(0,100);
      if (hash !== lastTrainHash) {
        lastTrainHash = hash;
        document.getElementById('code-content').innerHTML = highlightPython(data.train_py);
      }
      const src = data.train_py_source || '';
      document.getElementById('code-tab-label').textContent = src ? src : 'train.py';
    } else if (data.poll_count > 2) {
      document.getElementById('code-content').textContent = data.ssh_ok
        ? 'Waiting for train.py on remote...'
        : 'No SSH -- showing local train.py when available';
    }

    // Lab notebook
    if (data.lab_notebook) {
      const nbHash = data.lab_notebook.length.toString();
      if (nbHash !== lastNotebookHash) {
        lastNotebookHash = nbHash;
        document.getElementById('notebook-content').textContent = data.lab_notebook;
      }
    } else if (data.poll_count > 2) {
      document.getElementById('notebook-content').textContent = 'No lab_notebook.md found';
    }

  } catch(e) {
    console.error('Update error:', e);
  }
}

// Poll every 5 seconds
setInterval(update, 5000);
update();

// Redraw charts on resize
window.addEventListener('resize', update);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            state = get_state()
            self.wfile.write(json.dumps(state, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ART2 v2 web dashboard")
    parser.add_argument("--host", type=str, help="Remote SSH hostname (overrides .deploy-state)")
    parser.add_argument("--ssh-port", type=int, help="Remote SSH port (overrides .deploy-state)")
    parser.add_argument("--local", action="store_true", help="Local only (no SSH)")
    parser.add_argument("--port", type=int, default=8420, help="Web server port (default: 8420)")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds (default: 10)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Resolve remote
    remote_host = args.host
    remote_port = args.ssh_port
    host_pinned = args.host is not None
    if not args.local and not remote_host:
        state = load_deploy_state()
        if state:
            remote_host = state.get("SSH_HOST")
            remote_port = int(state.get("SSH_PORT", 22))

    # Start background poller
    poller = threading.Thread(
        target=poller_loop,
        args=(remote_host, remote_port or 22, args.local, args.interval),
        kwargs={"host_pinned": host_pinned},
        daemon=True,
    )
    poller.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    print(f"Dashboard running at {url}")
    print(f"  Mode: {'Remote ' + str(remote_host) + ':' + str(remote_port) if remote_host else 'Local'}")
    print(f"  Poll interval: {args.interval}s")
    print(f"  Press Ctrl+C to stop")

    if not args.no_open:
        import webbrowser
        webbrowser.open(url)

    server.timeout = 0.5
    try:
        while not _restart_flag.is_set():
            server.handle_request()
        # Source changed -- restart
        print("[monitor] Restarting...", flush=True)
        server.server_close()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
