#!/usr/bin/env python3
"""
Autoresearch web dashboard — full browser-based training monitor.

Usage:
  python3 tools/monitor.py                  # Auto-detect remote from .deploy-state
  python3 tools/monitor.py --port 8080      # Custom port
  python3 tools/monitor.py --local results  # Local only (no SSH)

Opens a browser dashboard with live-updating panels:
  - GPU health gauges
  - Experiment history table with all metrics
  - Score/PF/TPD charts over time
  - Full Claude reasoning stream (stream of consciousness)
  - Current best train.py source code
  - Live log tail
  - Session run overview
"""
from __future__ import annotations

import argparse
import json
import html
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
SSH_PASS = "autoresearch2026"

# -- Shared state (updated by background poller) --------------------------

_state_lock = threading.Lock()
_state: dict = {
    "experiments": [],
    "status": None,
    "log_tail": None,
    "gpu": None,
    "train_py": None,
    "mode": "initializing",
    "runs": [],
    "active_run": None,
    "last_fetch": 0,
    "fetch_time": 0,
    "poll_count": 0,
    "ssh_ok": False,
}


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def set_state(**kwargs):
    with _state_lock:
        _state.update(kwargs)


# -- Data fetching ---------------------------------------------------------

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
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
    try:
        r = subprocess.run(
            ["sshpass", "-e", "ssh"] + ssh_opts.split() + [
                "-p", str(port), f"root@{host}", f"bash -c {cmd!r}",
            ],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def fetch_remote(host: str, port: int):
    """Fetch all dashboard data from remote H100 in a single SSH call."""
    remote_script = (
        'RUN=$(cat /root/results/current_run.txt 2>/dev/null | tr -d "\\r\\n"); '
        'if [ -z "$RUN" ]; then echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; echo "---SEP---"; exit 0; fi; '
        'cat /root/results/"$RUN"/experiments.v2.jsonl 2>/dev/null; echo "---SEP---"; '
        'cat /root/results/"$RUN"/status.json 2>/dev/null; echo "---SEP---"; '
        'tail -80 /root/loop.log 2>/dev/null; echo "---SEP---"; '
        'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null; echo "---SEP---"; '
        'cat /root/train.py 2>/dev/null; echo "---SEP---"; '
        'echo "$RUN"'
    )
    raw = _ssh_cmd(host, port, remote_script, timeout=25)
    if not raw:
        return {}, None, None, None, None, None

    parts = raw.split("---SEP---")
    experiments = _parse_jsonl(parts[0] if len(parts) > 0 else "")
    status = _parse_json(parts[1] if len(parts) > 1 else "")
    log_tail = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
    gpu = _parse_gpu_csv(parts[3].strip() if len(parts) > 3 else "")
    train_py = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
    run_name = parts[5].strip() if len(parts) > 5 else None
    if status and run_name:
        status["_run_name"] = run_name
    return experiments, status, log_tail, gpu, train_py, run_name


def fetch_local_run(run_dir: Path):
    experiments = _parse_jsonl_file(run_dir / "experiments.v2.jsonl")
    status = _parse_json_file(run_dir / "status.json")
    if status:
        status["_run_name"] = run_dir.name
    log_tail = None
    for log_candidate in [run_dir / "loop.log", run_dir.parent / "loop.log"]:
        if log_candidate.exists():
            lines = log_candidate.read_text().split("\n")
            log_tail = "\n".join(lines[-80:])
            break
    return experiments, status, log_tail


def fetch_all_local_runs() -> list[dict]:
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir() or not d.name.startswith("run-"):
            continue
        status = _parse_json_file(d / "status.json")
        experiments = _parse_jsonl_file(d / "experiments.v2.jsonl")
        runs.append({"name": d.name, "status": status, "experiments": experiments})
    return runs


def get_active_run_name() -> str | None:
    pointer = RESULTS_ROOT / "current_run.txt"
    if pointer.exists():
        name = pointer.read_text().strip()
        return name if name else None
    return None


# -- Parse helpers ---------------------------------------------------------

def _parse_jsonl(text: str) -> list[dict]:
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_jsonl_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _parse_jsonl(path.read_text())


def _parse_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    return _parse_json(path.read_text())


def _parse_gpu_csv(text: str) -> dict | None:
    text = text.strip()
    if not text:
        return None
    try:
        parts = [p.strip() for p in text.split(",")]
        return {
            "name": parts[0] if len(parts) > 0 else "unknown",
            "mem_used_mb": float(parts[1]) if len(parts) > 1 else 0,
            "mem_total_mb": float(parts[2]) if len(parts) > 2 else 0,
            "util_pct": float(parts[3]) if len(parts) > 3 else 0,
            "temp_c": float(parts[4]) if len(parts) > 4 else 0,
            "power_w": float(parts[5]) if len(parts) > 5 else 0,
            "power_limit_w": float(parts[6]) if len(parts) > 6 else 0,
        }
    except (ValueError, IndexError):
        return None


# -- Background poller -----------------------------------------------------

def _resolve_local_run_dir(local_path: str | None, active_run: str | None) -> Path | None:
    """Find the best local run directory from pointer or active run name."""
    lp = Path(local_path) if local_path else RESULTS_ROOT
    if active_run and (lp / active_run).is_dir():
        return lp / active_run
    if lp.is_dir() and (lp / "current_run.txt").exists():
        rn = (lp / "current_run.txt").read_text().strip()
        if rn and (lp / rn).is_dir():
            return lp / rn
    return lp if lp.is_dir() else None


def poller_loop(host: str | None, port: int, local_path: str | None, interval: int):
    """Background thread that polls remote/local and updates shared state."""
    mode = f"Remote: {host}:{port}" if host else f"Local: {local_path or 'results/'}"
    set_state(mode=mode)

    while True:
        try:
            t0 = time.time()
            runs = fetch_all_local_runs()
            active_run = get_active_run_name()

            experiments = []
            status = None
            log_tail = None
            gpu = None
            train_py = None
            ssh_ok = False

            # Try remote first
            if host:
                experiments, status, log_tail, gpu, train_py, _ = fetch_remote(host, port)
                ssh_ok = bool(experiments or status or gpu)

            # Per-component local fallback — fill in anything remote didn't provide
            run_dir = _resolve_local_run_dir(local_path, active_run)
            if run_dir:
                if not experiments:
                    experiments = _parse_jsonl_file(run_dir / "experiments.v2.jsonl")
                if not status:
                    status = _parse_json_file(run_dir / "status.json")
                    if status:
                        status["_run_name"] = run_dir.name
                if not log_tail:
                    for log_candidate in [run_dir / "loop.log", run_dir.parent / "loop.log"]:
                        if log_candidate.exists():
                            lines = log_candidate.read_text().split("\n")
                            log_tail = "\n".join(lines[-80:])
                            break
                if not train_py:
                    # Try synced best_train.py or train.py from local workspace
                    for tp_candidate in [
                        run_dir / "best_train.py",
                        RESULTS_ROOT.parent / "training" / "best_train.py",
                        RESULTS_ROOT.parent / "training" / "train.py",
                    ]:
                        if tp_candidate.exists():
                            train_py = tp_candidate.read_text()
                            break

            fetch_time = time.time() - t0
            poll_count = get_state()["poll_count"] + 1

            set_state(
                experiments=experiments,
                status=status,
                log_tail=log_tail,
                gpu=gpu,
                train_py=train_py,
                runs=runs,
                active_run=active_run,
                last_fetch=time.time(),
                fetch_time=fetch_time,
                poll_count=poll_count,
                ssh_ok=ssh_ok,
            )
        except Exception as e:
            print(f"[poller] Error: {e}", file=sys.stderr)

        time.sleep(interval)


# -- HTML Dashboard --------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Autoresearch Dashboard</title>
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
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  header h1 {
    font-size: 16px;
    color: var(--cyan);
    font-weight: 700;
    letter-spacing: 1px;
  }
  header .meta {
    color: var(--text-dim);
    font-size: 12px;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    grid-template-rows: auto auto 1fr;
    gap: 12px;
    padding: 12px;
    min-height: calc(100vh - 60px);
  }
  .panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    background: var(--bg3);
    padding: 8px 14px;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--cyan);
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .panel-body {
    padding: 12px 14px;
    flex: 1;
    overflow: auto;
  }
  /* GPU + Status row */
  .gpu-status { grid-column: 1 / 2; }
  .active-run { grid-column: 2 / 4; }
  /* Charts row */
  .charts { grid-column: 1 / 4; min-height: 180px; }
  /* Main content row */
  .experiments { grid-column: 1 / 2; min-height: 400px; }
  .reasoning { grid-column: 2 / 3; min-height: 400px; }
  .right-stack { grid-column: 3 / 4; display: flex; flex-direction: column; gap: 12px; }

  /* GPU gauges */
  .gauge-row { display: flex; gap: 16px; margin-bottom: 8px; align-items: center; }
  .gauge-label { width: 50px; font-size: 11px; color: var(--text-dim); }
  .gauge-bar { flex: 1; height: 18px; background: var(--bg); border-radius: 3px; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 3px; transition: width 0.5s; }
  .gauge-value { width: 80px; text-align: right; font-size: 12px; font-weight: 600; }
  .fill-green { background: var(--green); }
  .fill-yellow { background: var(--yellow); }
  .fill-red { background: var(--red); }
  .fill-cyan { background: var(--cyan); }

  /* Stat cards */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }
  .stat-card {
    background: var(--bg);
    border-radius: 6px;
    padding: 8px 10px;
    text-align: center;
  }
  .stat-value { font-size: 20px; font-weight: 700; }
  .stat-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-green { color: var(--green); }
  .stat-red { color: var(--red); }
  .stat-yellow { color: var(--yellow); }
  .stat-cyan { color: var(--cyan); }

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
  tr.failed td { color: var(--red); }
  tr.reverted td { color: var(--text-dim); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.change { white-space: normal; max-width: 200px; overflow: hidden; text-overflow: ellipsis; color: var(--text-dim); font-size: 11px; }

  /* Reasoning / stream of consciousness */
  .reasoning-entry {
    border-bottom: 1px solid var(--bg3);
    padding: 10px 0;
  }
  .reasoning-entry:last-child { border-bottom: none; }
  .reasoning-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }
  .reasoning-id { font-weight: 700; font-size: 13px; }
  .reasoning-badge {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 10px;
    font-weight: 600;
  }
  .badge-kept { background: rgba(63,185,80,0.15); color: var(--green); }
  .badge-failed { background: rgba(248,81,73,0.15); color: var(--red); }
  .badge-reverted { background: rgba(139,148,158,0.15); color: var(--text-dim); }
  .reasoning-text {
    color: var(--text);
    font-size: 12px;
    line-height: 1.6;
    margin-bottom: 6px;
  }
  .reasoning-change {
    color: var(--cyan);
    font-size: 11px;
    font-style: italic;
  }
  .reasoning-metrics {
    color: var(--text-dim);
    font-size: 11px;
    margin-top: 4px;
  }

  /* Code viewer */
  .code-view {
    background: var(--bg);
    border-radius: 4px;
    padding: 10px;
    font-size: 12px;
    line-height: 1.5;
    overflow: auto;
    white-space: pre;
    max-height: 100%;
    tab-size: 4;
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
  }
  .log-error { color: var(--red); }
  .log-success { color: var(--green); }
  .log-warn { color: var(--yellow); }
  .log-info { color: var(--cyan); }
  .log-dim { color: var(--text-dim); }

  /* Charts */
  .chart-container { display: flex; gap: 16px; height: 150px; }
  .chart-box { flex: 1; position: relative; }
  .chart-title { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
  canvas { width: 100% !important; height: 130px !important; }

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
  .phase-training { background: var(--green); }
  .phase-calling_claude { background: var(--yellow); }
  .phase-evaluating { background: var(--cyan); }
  .phase-error { background: var(--red); animation: none; }
  .phase-completed { background: var(--text-dim); animation: none; }
  .phase-saving { background: var(--purple); }

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

  /* Responsive */
  @media (max-width: 1200px) {
    .grid {
      grid-template-columns: 1fr 1fr;
    }
    .active-run { grid-column: 2 / 3; }
    .charts { grid-column: 1 / 3; }
    .experiments { grid-column: 1 / 3; }
    .reasoning { grid-column: 1 / 2; }
    .right-stack { grid-column: 2 / 3; }
  }

  .stale-warning {
    background: rgba(248,81,73,0.15);
    color: var(--red);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    animation: pulse 2s ease-in-out infinite;
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
  <h1>AUTORESEARCH DASHBOARD</h1>
  <div class="meta">
    <span id="header-mode"></span> &nbsp;|&nbsp;
    <span id="header-time"></span> &nbsp;|&nbsp;
    <span id="header-poll"></span> &nbsp;|&nbsp;
    <span id="header-fetch"></span>
    <span id="header-stale"></span>
  </div>
</header>

<div class="grid">
  <!-- GPU Health -->
  <div class="panel gpu-status">
    <div class="panel-header">GPU Health <span id="gpu-name"></span></div>
    <div class="panel-body">
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

  <!-- Active Run Status -->
  <div class="panel active-run">
    <div class="panel-header">Active Run <span id="run-name" style="font-weight:400;color:var(--text)"></span></div>
    <div class="panel-body">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
        <span class="phase-dot" id="phase-dot"></span>
        <span id="phase-label" style="font-weight:600"></span>
        <span id="exp-id" style="color:var(--text-dim);margin-left:auto"></span>
      </div>
      <div class="progress-outer">
        <div class="progress-inner" id="time-progress" style="width:0%"></div>
      </div>
      <div style="display:flex;justify-content:space-between;color:var(--text-dim);font-size:11px;margin-bottom:10px;">
        <span id="time-elapsed"></span>
        <span id="time-remaining"></span>
      </div>
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value stat-green" id="stat-kept">--</div><div class="stat-label">Kept</div></div>
        <div class="stat-card"><div class="stat-value stat-red" id="stat-failed">--</div><div class="stat-label">Failed</div></div>
        <div class="stat-card"><div class="stat-value stat-cyan" id="stat-total">--</div><div class="stat-label">Total</div></div>
        <div class="stat-card"><div class="stat-value stat-green" id="stat-best">--</div><div class="stat-label">Best Score</div></div>
        <div class="stat-card"><div class="stat-value stat-yellow" id="stat-rate">--</div><div class="stat-label">Exp/hr</div></div>
        <div class="stat-card"><div class="stat-value" id="stat-contract" style="font-size:11px;color:var(--text-dim)">--</div><div class="stat-label">Contract</div></div>
      </div>
      <div class="stats-grid" style="margin-top:4px;">
        <div class="stat-card"><div class="stat-value stat-yellow" id="stat-cost" style="font-size:16px">--</div><div class="stat-label">API Cost</div></div>
        <div class="stat-card"><div class="stat-value stat-cyan" id="stat-cache-hit" style="font-size:16px">--</div><div class="stat-label">Cache Hit %</div></div>
        <div class="stat-card"><div class="stat-value" id="stat-cost-per-exp" style="font-size:16px;color:var(--text-dim)">--</div><div class="stat-label">$/Experiment</div></div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  <div class="panel charts">
    <div class="panel-header">Metric Trends</div>
    <div class="panel-body">
      <div class="chart-container">
        <div class="chart-box">
          <div class="chart-title">Score</div>
          <canvas id="chart-score"></canvas>
        </div>
        <div class="chart-box">
          <div class="chart-title">Profit Factor</div>
          <canvas id="chart-pf"></canvas>
        </div>
        <div class="chart-box">
          <div class="chart-title">Trades/Day</div>
          <canvas id="chart-tpd"></canvas>
        </div>
        <div class="chart-box">
          <div class="chart-title">Win Rate</div>
          <canvas id="chart-wr"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Experiments Table -->
  <div class="panel experiments">
    <div class="panel-header">Experiments <span id="exp-count" style="font-weight:400;color:var(--text-dim)"></span></div>
    <div class="panel-body" style="padding:0">
      <div style="overflow:auto;max-height:500px;">
        <table id="exp-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Result</th>
              <th>Score</th>
              <th>PF</th>
              <th>TPD</th>
              <th>Sharpe</th>
              <th>WR</th>
              <th>SL%</th>
              <th>Hold</th>
              <th>Ruin</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody id="exp-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Stream of Consciousness (Claude Reasoning) -->
  <div class="panel reasoning">
    <div class="panel-header">Stream of Consciousness</div>
    <div class="panel-body" id="reasoning-body" style="padding:8px 14px"></div>
  </div>

  <!-- Right stack: train.py + log -->
  <div class="right-stack">
    <div class="panel" style="flex:1">
      <div class="panel-header">
        <div class="tab-bar" style="border-bottom:none">
          <div class="tab active" data-tab="code">Best train.py</div>
          <div class="tab" data-tab="log">Live Log</div>
          <div class="tab" data-tab="runs">Runs</div>
        </div>
      </div>
      <div class="panel-body" style="padding:0">
        <div class="tab-content active" id="tab-code">
          <div class="code-view" id="code-content" style="max-height:500px">Loading...</div>
        </div>
        <div class="tab-content" id="tab-log">
          <div class="log-view" id="log-content" style="max-height:500px">Waiting for log data...</div>
        </div>
        <div class="tab-content" id="tab-runs">
          <div style="padding:14px;overflow:auto;max-height:500px;">
            <table id="runs-table">
              <thead><tr><th>Run</th><th>Phase</th><th>Exp</th><th>Kept</th><th>Best</th></tr></thead>
              <tbody id="runs-tbody"></tbody>
            </table>
          </div>
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
  ctx.fillText(lastVal.toFixed(opts.decimals !== undefined ? opts.decimals : 2), w - 6, 14);

  // Min/max
  ctx.fillStyle = '#8b949e';
  ctx.font = '9px monospace';
  ctx.fillText(mn.toFixed(opts.decimals !== undefined ? opts.decimals : 2), w - 6, h - 2);
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

// -- PST time --
function nowPST() {
  return new Date().toLocaleString('en-US', {timeZone: 'America/Los_Angeles', hour12: false});
}

function fmtDuration(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(0) + 'm';
  return Math.floor(s/3600) + 'h' + String(Math.floor((s%3600)/60)).padStart(2,'0') + 'm';
}

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
    // Comments first (so they don't get re-highlighted)
    hl = hl.replace(/(#.*)$/g, '<span style="color:#8b949e">$1</span>');
    // Keywords
    hl = hl.replace(/\b(def|class|import|from|return|if|else|elif|for|while|with|as|try|except|finally|raise|yield|lambda|and|or|not|in|is|None|True|False|self|pass|break|continue)\b/g,
      '<span style="color:#ff7b72">$1</span>');
    // Strings (double and single quoted only - avoiding triple quotes in regex)
    hl = hl.replace(/("[^"]*"|'[^']*')/g, '<span style="color:#a5d6ff">$1</span>');
    // Numbers
    hl = hl.replace(/\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g, '<span style="color:#79c0ff">$1</span>');
    // Decorators
    hl = hl.replace(/^(\s*@\w+)/g, '<span style="color:#d2a8ff">$1</span>');
    return ln + hl;
  }).join('\n');
}

// -- Colorize log --
function colorizeLine(line) {
  if (line.includes('ERROR') || line.includes('FAILED')) return `<span class="log-error">${escapeHtml(line)}</span>`;
  if (line.includes('KEPT') || line.includes('kept') || line.includes('IMPROVED')) return `<span class="log-success">${escapeHtml(line)}</span>`;
  if (line.includes('WARNING')) return `<span class="log-warn">${escapeHtml(line)}</span>`;
  if (line.includes('=== Experiment') || line.includes('score:')) return `<span class="log-info">${escapeHtml(line)}</span>`;
  return `<span class="log-dim">${escapeHtml(line)}</span>`;
}

// -- Main update loop --
let lastExpCount = 0;
let lastTrainHash = '';

async function update() {
  try {
    const resp = await fetch('/api/state');
    const data = await resp.json();

    // Header
    const sshTag = data.mode.startsWith('Remote') ? (data.ssh_ok ? ' [SSH OK]' : ' [SSH FAIL — using local]') : '';
    document.getElementById('header-mode').textContent = data.mode + sshTag;
    document.getElementById('header-time').textContent = nowPST();
    document.getElementById('header-poll').textContent = `poll #${data.poll_count}`;
    document.getElementById('header-fetch').textContent = `fetch: ${data.fetch_time.toFixed(1)}s`;

    // Stale check
    const staleEl = document.getElementById('header-stale');
    if (data.status && data.status.updated) {
      const lastUpdate = new Date(data.status.updated + 'Z');
      const ago = (Date.now() - lastUpdate.getTime()) / 1000;
      if (ago > 120) {
        staleEl.innerHTML = `&nbsp;|&nbsp;<span class="stale-warning">STALE: ${fmtDuration(ago)} ago</span>`;
      } else {
        staleEl.textContent = '';
      }
    } else {
      staleEl.textContent = '';
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
      document.getElementById('gpu-name').textContent = data.ssh_ok ? 'waiting...' : 'no SSH';
      document.getElementById('gpu-content').style.opacity = '0.3';
    }

    // Active run
    const st = data.status || {};
    document.getElementById('run-name').textContent = st._run_name || 'none';
    const phase = st.phase || 'unknown';
    const phaseDot = document.getElementById('phase-dot');
    phaseDot.className = 'phase-dot phase-' + phase;
    const phaseLabels = {
      training: 'Training Model', calling_claude: 'Generating Code',
      evaluating: 'Evaluating Results', saving: 'Saving Artifacts',
      completed: 'Completed', error: 'Error', startup: 'Starting Up',
      between_experiments: 'Between Experiments', smoke_check: 'Smoke Check'
    };
    document.getElementById('phase-label').textContent = phaseLabels[phase] || phase;
    document.getElementById('exp-id').textContent = `Experiment #${st.experiment_id || 0}`;

    // Progress
    const timeLeft = st.time_remaining_h || 0;
    const totalH = 8.0;
    const elapsedH = totalH - timeLeft;
    const pct = Math.min(100, Math.max(0, elapsedH / totalH * 100));
    document.getElementById('time-progress').style.width = pct + '%';
    document.getElementById('time-elapsed').textContent = pct.toFixed(0) + '% elapsed';
    document.getElementById('time-remaining').textContent = timeLeft > 0 ? timeLeft.toFixed(1) + 'h left' : 'done';

    // Stats
    document.getElementById('stat-kept').textContent = st.kept || 0;
    document.getElementById('stat-failed').textContent = st.failed || 0;
    document.getElementById('stat-total').textContent = st.total || 0;
    document.getElementById('stat-best').textContent = st.best_score > -999 ? st.best_score.toFixed(4) : '--';
    document.getElementById('stat-contract').textContent = st.contract_checksum || '--';

    // API cost tracking
    const ac = st.api_cost;
    if (ac) {
      document.getElementById('stat-cost').textContent = '$' + (ac.total_cost || 0).toFixed(2);
      document.getElementById('stat-cache-hit').textContent = (ac.cache_hit_pct || 0).toFixed(0) + '%';
      const perExp = st.total > 0 ? (ac.total_cost / st.total).toFixed(3) : '--';
      document.getElementById('stat-cost-per-exp').textContent = perExp !== '--' ? '$' + perExp : '--';
    }

    // Exp rate
    const exps = data.experiments || [];
    if (exps.length >= 2) {
      const t0 = new Date(exps[0].timestamp);
      const t1 = new Date(exps[exps.length-1].timestamp);
      const hrs = (t1 - t0) / 3600000;
      if (hrs > 0) document.getElementById('stat-rate').textContent = (exps.length / hrs).toFixed(1);
    }

    // Charts
    const scored = exps.filter(e => (e.score || -999) > -999);
    if (scored.length > 1) {
      drawChart('chart-score', scored.map(e => e.score || 0), '#3fb950', {decimals: 3});
      drawChart('chart-pf', scored.map(e => e.profit_factor || 0), '#58a6ff', {min: 0, decimals: 2});
      drawChart('chart-tpd', scored.map(e => e.trades_per_day || 0), '#d29922', {min: 0, decimals: 1});
      drawChart('chart-wr', scored.map(e => (e.win_rate || 0) * 100), '#bc8cff', {min: 0, max: 100, decimals: 0});
    }

    // Experiments table
    if (exps.length !== lastExpCount) {
      lastExpCount = exps.length;
      document.getElementById('exp-count').textContent = `(${exps.length})`;
      const tbody = document.getElementById('exp-tbody');
      tbody.innerHTML = '';
      // Show newest first
      for (let i = exps.length - 1; i >= 0; i--) {
        const e = exps[i];
        const tr = document.createElement('tr');
        let result, cls;
        if (e.kept) { result = '+ KEPT'; cls = 'kept'; }
        else if (e.error) { result = 'x ' + (e.failure_type || 'err').slice(0,8); cls = 'failed'; }
        else { result = '~ revert'; cls = 'reverted'; }
        tr.className = cls;

        const score = (e.score || -999) > -999 ? (e.score || 0).toFixed(3) : 'FAIL';
        const pf = e.profit_factor ? e.profit_factor.toFixed(2) : '--';
        const tpd = e.trades_per_day ? e.trades_per_day.toFixed(1) : '--';
        const sharpe = e.trade_sharpe ? e.trade_sharpe.toFixed(2) : '--';
        const wr = e.win_rate ? (e.win_rate * 100).toFixed(0) + '%' : '--';
        const sl = e.stop_loss_rate ? (e.stop_loss_rate * 100).toFixed(0) + '%' : '--';
        const hold = e.avg_hold_bars ? e.avg_hold_bars.toFixed(0) : '--';
        const ruin = e.hit_ruin !== undefined ? (e.hit_ruin ? 'YES' : 'no') : '--';
        const trainTime = e.train_wall_time ? fmtDuration(e.train_wall_time) : '--';

        tr.innerHTML = `
          <td class="num">${e.experiment_id || '?'}</td>
          <td>${result}</td>
          <td class="num">${score}</td>
          <td class="num">${pf}</td>
          <td class="num">${tpd}</td>
          <td class="num">${sharpe}</td>
          <td class="num">${wr}</td>
          <td class="num">${sl}</td>
          <td class="num">${hold}</td>
          <td class="num">${ruin === 'YES' ? '<span style="color:var(--red)">YES</span>' : ruin}</td>
          <td class="num">${trainTime}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // Stream of Consciousness (reasoning from all experiments, newest first)
    const reasoningBody = document.getElementById('reasoning-body');
    if (exps.length !== parseInt(reasoningBody.dataset.count || '0')) {
      reasoningBody.dataset.count = exps.length;
      let html = '';
      for (let i = exps.length - 1; i >= 0; i--) {
        const e = exps[i];
        const reasoning = e._full_reasoning || e.reasoning || '';
        const change = e.change_summary || '';
        const err = e.error || '';
        if (!reasoning && !err) continue;

        let badgeCls, badgeText;
        if (e.kept) { badgeCls = 'badge-kept'; badgeText = 'KEPT'; }
        else if (e.error) { badgeCls = 'badge-failed'; badgeText = e.failure_type || 'FAILED'; }
        else { badgeCls = 'badge-reverted'; badgeText = 'REVERTED'; }

        const score = (e.score || -999) > -999 ? (e.score || 0).toFixed(4) : 'FAIL';
        const metrics = (e.score || -999) > -999
          ? `Score: ${score} | PF: ${(e.profit_factor||0).toFixed(2)} | TPD: ${(e.trades_per_day||0).toFixed(1)} | WR: ${((e.win_rate||0)*100).toFixed(0)}% | Sharpe: ${(e.trade_sharpe||0).toFixed(2)}`
          : '';

        html += `<div class="reasoning-entry">
          <div class="reasoning-header">
            <span class="reasoning-id">#${e.experiment_id || '?'}</span>
            <span class="reasoning-badge ${badgeCls}">${badgeText}</span>
          </div>
          <div class="reasoning-text">${escapeHtml(reasoning || err)}</div>
          ${change ? `<div class="reasoning-change">${escapeHtml(change)}</div>` : ''}
          ${metrics ? `<div class="reasoning-metrics">${metrics}</div>` : ''}
        </div>`;
      }
      reasoningBody.innerHTML = html || '<div style="color:var(--text-dim)">Waiting for experiments...</div>';
    }

    // train.py code
    if (data.train_py) {
      const hash = data.train_py.length + data.train_py.slice(0,100);
      if (hash !== lastTrainHash) {
        lastTrainHash = hash;
        document.getElementById('code-content').innerHTML = highlightPython(data.train_py);
      }
    } else if (data.poll_count > 2) {
      document.getElementById('code-content').textContent = data.ssh_ok
        ? 'Waiting for train.py on remote...'
        : 'No SSH connection — no local train.py found in results/';
    }

    // Log
    if (data.log_tail) {
      const logEl = document.getElementById('log-content');
      const lines = data.log_tail.split('\n');
      logEl.innerHTML = lines.map(colorizeLine).join('\n');
      logEl.scrollTop = logEl.scrollHeight;
    } else if (data.poll_count > 2) {
      document.getElementById('log-content').textContent = data.ssh_ok
        ? 'Waiting for loop.log on remote...'
        : 'No SSH connection — check sync.log in results/ for log data';
    }

    // Runs table
    const runs = data.runs || [];
    if (runs.length) {
      const rtbody = document.getElementById('runs-tbody');
      rtbody.innerHTML = '';
      for (const r of runs.reverse()) {
        const s = r.status || {};
        const re = r.experiments || [];
        const nKept = re.filter(e => e.kept).length;
        const best = re.reduce((m, e) => Math.max(m, e.score || -999), -999);
        const isActive = r.name === data.active_run;
        const tr = document.createElement('tr');
        if (isActive) tr.style.color = 'var(--cyan)';
        tr.innerHTML = `
          <td>${isActive ? '> ' : ''}${r.name}</td>
          <td>${s.phase || '--'}</td>
          <td class="num">${re.length}</td>
          <td class="num">${nKept}</td>
          <td class="num">${best > -999 ? best.toFixed(3) : '--'}</td>
        `;
        rtbody.appendChild(tr);
      }
    }

  } catch(e) {
    console.error('Update error:', e);
  }
}

function setGauge(prefix, pct, text) {
  const bar = document.getElementById(prefix + '-bar');
  const val = document.getElementById(prefix + '-val');
  bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  val.textContent = text;
  // Color based on value
  bar.className = 'gauge-fill ' + (pct > 80 ? 'fill-red' : pct > 50 ? 'fill-yellow' : 'fill-green');
}

// Poll every 5 seconds
setInterval(update, 5000);
update();

// Redraw charts on resize
window.addEventListener('resize', update);
</script>
</body>
</html>"""


# -- HTTP Server -----------------------------------------------------------

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
            # Don't send the full train_py on every poll if it hasn't changed
            self.wfile.write(json.dumps(state, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


# -- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch web dashboard")
    parser.add_argument("--host", type=str, help="Remote H100 hostname (overrides .deploy-state)")
    parser.add_argument("--ssh-port", type=int, help="Remote SSH port (overrides .deploy-state)")
    parser.add_argument("--local", type=str, help="Local results path (skip remote)")
    parser.add_argument("--port", type=int, default=8420, help="Web server port (default: 8420)")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds (default: 10)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Resolve remote
    remote_host = args.host
    remote_port = args.ssh_port
    if not args.local and not remote_host:
        state = load_deploy_state()
        if state:
            remote_host = state.get("SSH_HOST")
            remote_port = int(state.get("SSH_PORT", 22))

    # Start background poller
    poller = threading.Thread(
        target=poller_loop,
        args=(remote_host, remote_port or 22, args.local, args.interval),
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

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.shutdown()


if __name__ == "__main__":
    main()
