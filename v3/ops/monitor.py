#!/usr/bin/env python3
"""ART2 v3 Web Dashboard -- live browser-based RL training monitor.

Usage:
    python v3/ops/monitor.py                  # Auto-detect remote from .deploy-state
    python v3/ops/monitor.py --port 8430      # Custom port (default: 8430)
    python v3/ops/monitor.py --local          # Local only (no SSH)
    python v3/ops/monitor.py --no-open        # Don't auto-open browser

Opens a browser dashboard with live-updating panels:
  - GPU health gauges (utilization, VRAM, temp, power) -- auto-refreshes independently
  - Session state (experiments, best score)
  - Experiment history table with scores
  - Score/metric charts over time
  - Live log tail from the GPU
  - Current train.py source code
  - Lab notebook
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
SSH_PASS = os.environ.get("DEPLOY_SSH_PASS", "autoresearch2026")
RESULTS_TSV = PROJECT_ROOT / "v3" / "results.tsv"
LAB_NOTEBOOK = PROJECT_ROOT / "v3" / "lab_notebook.md"
ARTIFACTS_DIR = PROJECT_ROOT / "v3" / "artifacts"
TRAIN_PY = PROJECT_ROOT / "v3" / "train.py"

# Self-restart on source change + live version hash for browser auto-reload
_SELF_PATH = Path(__file__).resolve()
_SELF_HASH_AT_START = hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
_restart_flag = threading.Event()


def _current_source_hash() -> str:
    """Return the current on-disk hash of this file (for browser reload detection)."""
    try:
        return hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
    except Exception:
        return _SELF_HASH_AT_START


def _check_self_restart():
    try:
        current_hash = _current_source_hash()
        if current_hash != _SELF_HASH_AT_START:
            print("[monitor] Source changed on disk -- restarting...", flush=True)
            _restart_flag.set()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared state (updated by background pollers)
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
    "warnings": [],
    "last_gpu_time": 0,
    "ssh_fail_count": 0,
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
# GPU-only fast poller (independent of main data poller)
# ---------------------------------------------------------------------------

_gpu_lock = threading.Lock()
_gpu_data: dict = {"gpu": None, "last_gpu_time": 0, "ssh_ok": False, "ssh_fail_count": 0}


def get_gpu_state() -> dict:
    with _gpu_lock:
        return dict(_gpu_data)


def _set_gpu_state(**kwargs):
    with _gpu_lock:
        _gpu_data.update(kwargs)


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


def gpu_poller_loop(host_ref: list, port_ref: list, local_only: bool):
    """Dedicated fast GPU poller -- runs every 3s, independent of the main data poll."""
    last_good_gpu = None
    fail_count = 0

    while True:
        gpu = None
        ssh_ok = False

        # Try remote GPU first
        h, p = host_ref[0], port_ref[0]
        if h and not local_only:
            timeout = 8 if fail_count >= 3 else 12
            raw = _ssh_cmd(
                h, p,
                'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit '
                '--format=csv,noheader,nounits 2>/dev/null',
                timeout=timeout,
            )
            if raw:
                gpu = _parse_gpu_csv(raw.strip())
                ssh_ok = True
                fail_count = 0
            else:
                fail_count += 1

        # Local fallback
        if not gpu:
            gpu = fetch_local_gpu()

        if gpu:
            last_good_gpu = gpu
            _set_gpu_state(gpu=gpu, last_gpu_time=time.time(), ssh_ok=ssh_ok, ssh_fail_count=fail_count)
        elif last_good_gpu:
            stale = dict(last_good_gpu)
            stale["_stale"] = True
            _set_gpu_state(gpu=stale, ssh_ok=ssh_ok, ssh_fail_count=fail_count)
        else:
            _set_gpu_state(gpu=None, ssh_ok=ssh_ok, ssh_fail_count=fail_count)

        time.sleep(3)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _parse_results_tsv(text: str) -> list[dict]:
    """Parse v3/results.tsv.

    Columns: experiment_id, created_at, mask, score, total_return_pct,
             max_drawdown_pct, daily_sortino, total_trades, status
    """
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
        # Normalize to common field names
        row["experiment"] = row.get("experiment_id", "").strip()
        try:
            row["score_num"] = float(row.get("score", "-999"))
        except (ValueError, TypeError):
            row["score_num"] = -999.0
        try:
            row["total_return"] = float(row.get("total_return_pct", "0"))
        except (ValueError, TypeError):
            row["total_return"] = 0.0
        try:
            row["max_drawdown"] = float(row.get("max_drawdown_pct", "0"))
        except (ValueError, TypeError):
            row["max_drawdown"] = 0.0
        try:
            row["sortino"] = float(row.get("daily_sortino", "0"))
        except (ValueError, TypeError):
            row["sortino"] = 0.0
        try:
            row["trades"] = int(row.get("total_trades", "0"))
        except (ValueError, TypeError):
            row["trades"] = 0
        rows.append(row)
    return rows


def _exp_sort_key(exp_id: str) -> int:
    m = re.search(r'(\d+)', exp_id)
    return int(m.group(1)) if m else 0


def load_all_experiments(warnings: list[str]) -> list[dict]:
    """Load v3 experiment history from results.tsv and artifacts."""
    tsv_exps: list[dict] = []
    if RESULTS_TSV.exists():
        try:
            tsv_exps = _parse_results_tsv(RESULTS_TSV.read_text())
        except Exception as e:
            warnings.append(f"results.tsv parse failed: {e}")

    # Scan artifact manifests for any not in TSV
    artifact_ids = {e["experiment"] for e in tsv_exps}
    if ARTIFACTS_DIR.is_dir():
        from datetime import datetime
        for entry in sorted(ARTIFACTS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            manifest = entry / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            exp_id = data.get("experiment_id", entry.name)
            if exp_id in artifact_ids:
                continue
            ts = None
            ts_str = data.get("timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str).timestamp()
                except ValueError:
                    pass
            score = data.get("score", -999.0)
            tsv_exps.append({
                "experiment": exp_id,
                "score_num": float(score) if score != -999.0 else -999.0,
                "total_return": data.get("extra", {}).get("total_return_pct", 0),
                "max_drawdown": data.get("extra", {}).get("max_drawdown_pct", 0),
                "sortino": data.get("extra", {}).get("daily_sortino", 0),
                "trades": data.get("extra", {}).get("total_trades", 0),
                "status": "keep" if data.get("promoted") else "fail",
                "timestamp": ts,
            })

    # Enrich timestamps from manifests
    from datetime import datetime
    for exp in tsv_exps:
        if exp.get("timestamp"):
            continue
        ts_str = exp.get("created_at", "")
        if ts_str:
            try:
                exp["timestamp"] = datetime.fromisoformat(ts_str).timestamp()
            except ValueError:
                exp["timestamp"] = None
        else:
            manifest = ARTIFACTS_DIR / exp.get("experiment", "") / "manifest.json"
            if manifest.is_file():
                try:
                    data = json.loads(manifest.read_text())
                    ts_s = data.get("timestamp")
                    if ts_s:
                        exp["timestamp"] = datetime.fromisoformat(ts_s).timestamp()
                except (OSError, json.JSONDecodeError, ValueError):
                    pass

    tsv_exps.sort(key=lambda r: _exp_sort_key(r.get("experiment", "")))
    return tsv_exps


def derive_session(experiments: list[dict]) -> dict:
    if not experiments:
        return {}
    total = len(experiments)
    scored = [e for e in experiments if e.get("score_num", -999.0) > -999.0]
    best_entry = max(scored, key=lambda e: e["score_num"]) if scored else None
    best_score = best_entry["score_num"] if best_entry else None
    best_exp = best_entry.get("experiment", "") if best_entry else ""
    timestamps = [e["timestamp"] for e in experiments if e.get("timestamp")]
    session_start = min(timestamps) if timestamps else None
    return {
        "experiment_count": total,
        "best_score": best_score,
        "best_experiment": best_exp,
        "session_start": session_start,
    }


def fetch_remote_data(host: str, port: int, timeout: int = 25) -> dict:
    """Fetch dashboard data from the Akash GPU in one SSH call."""
    remote_script = (
        # Training process
        'PID=$(pgrep -f "[r]un_experiment" | head -1); '
        'if [ -n "$PID" ]; then '
        '  UPTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d " "); '
        '  echo "running|$PID|$UPTIME"; '
        'else echo "idle||"; fi; '
        'echo "---SEP---"; '

        # Log tail
        'tail -80 /root/run.log 2>/dev/null || echo "(no log)"; '
        'echo "---SEP---"; '

        # Remote train.py
        'cat /root/v3/train.py 2>/dev/null || echo ""; '
    )

    raw = _ssh_cmd(host, port, remote_script, timeout=timeout)
    if not raw:
        return {"ssh_ok": False}

    parts = raw.split("---SEP---")
    data: dict = {"ssh_ok": True}

    # Parse process
    proc_raw = parts[0].strip() if len(parts) > 0 else ""
    if proc_raw:
        pp = proc_raw.split("|")
        data["process"] = {
            "status": pp[0] if len(pp) > 0 else "unknown",
            "pid": pp[1] if len(pp) > 1 else "",
            "uptime": pp[2] if len(pp) > 2 else "",
        }
    else:
        data["process"] = {"status": "unknown", "pid": "", "uptime": ""}

    data["log_tail"] = parts[1].strip() if len(parts) > 1 else None
    data["train_py"] = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None

    return data


def load_lab_notebook() -> str | None:
    if LAB_NOTEBOOK.exists():
        try:
            return LAB_NOTEBOOK.read_text()
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Background poller (main data -- experiments, logs, process)
# ---------------------------------------------------------------------------

def poller_loop(host_ref: list, port_ref: list, local_only: bool, interval: int):
    mode = f"Remote: {host_ref[0]}:{port_ref[0]}" if host_ref[0] else "Local"
    set_state(mode=mode)
    ssh_fail_count = 0

    while True:
        _check_self_restart()

        try:
            t0 = time.time()

            # Reload .deploy-state each poll
            if not local_only:
                ds = load_deploy_state()
                if ds:
                    new_host = ds.get("SSH_HOST")
                    try:
                        new_port = int(ds.get("SSH_PORT", 22))
                    except (ValueError, TypeError):
                        new_port = 22
                else:
                    new_host, new_port = None, 22
                if new_host != host_ref[0] or new_port != port_ref[0]:
                    print(f"[monitor] Deployment changed: {host_ref[0]}:{port_ref[0]} -> {new_host}:{new_port}", flush=True)
                    host_ref[0], port_ref[0] = new_host, new_port
                    ssh_fail_count = 0
                new_mode = f"Remote: {host_ref[0]}:{port_ref[0]}" if host_ref[0] else "Local"
                if new_mode != mode:
                    mode = new_mode
                    set_state(mode=mode)

            ssh_ok = False
            log_tail = None
            train_py = None
            train_py_source = ""
            process = {"status": "unknown", "pid": "", "uptime": ""}
            poll_warnings: list[str] = []

            experiments = load_all_experiments(poll_warnings)

            # Local train.py is authoritative
            if TRAIN_PY.exists():
                train_py = TRAIN_PY.read_text()
                train_py_source = "local: v3/train.py"

            # Try remote for process status and log tail
            host, port = host_ref[0], port_ref[0]
            if host and not local_only:
                ssh_timeout = 15 if ssh_fail_count >= 3 else 25
                remote = fetch_remote_data(host, port, timeout=ssh_timeout)
                ssh_ok = remote.get("ssh_ok", False)
                if ssh_ok:
                    ssh_fail_count = 0
                    log_tail = remote.get("log_tail")
                    process = remote.get("process", process)
                else:
                    ssh_fail_count += 1

            # Get GPU state from the dedicated GPU poller
            gpu_state = get_gpu_state()

            session = derive_session(experiments)

            # Classify run state
            if process.get("status") == "running":
                run_state = "active"
            elif host and not ssh_ok:
                run_state = "ssh_fail"
            elif not host:
                run_state = "local"
            else:
                run_state = "idle"

            lab_notebook = load_lab_notebook()
            fetch_time = time.time() - t0
            poll_count = get_state()["poll_count"] + 1

            set_state(
                experiments=experiments,
                session=session,
                log_tail=log_tail,
                gpu=gpu_state.get("gpu"),
                train_py=train_py,
                train_py_source=train_py_source,
                process=process,
                run_state=run_state,
                last_fetch=time.time(),
                fetch_time=fetch_time,
                poll_count=poll_count,
                ssh_ok=ssh_ok or gpu_state.get("ssh_ok", False),
                lab_notebook=lab_notebook,
                warnings=poll_warnings,
                last_gpu_time=gpu_state.get("last_gpu_time", 0),
                ssh_fail_count=ssh_fail_count,
            )
        except Exception as e:
            print(f"[poller] Error: {e}", file=sys.stderr)

        sleep_time = max(3, interval // 2) if ssh_fail_count > 0 and not local_only else interval
        time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ART2 v3 RL Monitor</title>
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
  header .phase-tag {
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    background: rgba(188,140,255,0.2);
    color: var(--purple);
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-left: 10px;
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
  .top-row .gpu-status { flex: 0 0 260px; min-width: 0; }
  .top-row .session-status { flex: 1; min-width: 0; }

  /* GPU gauges */
  .gauge-row { display: flex; gap: 6px; margin-bottom: 5px; align-items: center; }
  .gauge-label { width: 36px; font-size: 10px; color: var(--text-dim); }
  .gauge-bar { flex: 1; height: 14px; background: var(--bg); border-radius: 3px; overflow: hidden; position: relative; }
  .gauge-fill { height: 100%; border-radius: 3px; transition: width 0.4s ease; }
  .gauge-value { width: 80px; text-align: right; font-size: 11px; font-weight: 600; }
  .fill-green { background: var(--green); }
  .fill-yellow { background: var(--yellow); }
  .fill-red { background: var(--red); }
  .fill-cyan { background: var(--cyan); }
  .gpu-refresh-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    display: inline-block;
    margin-left: 6px;
    transition: opacity 0.3s;
  }
  .gpu-refresh-dot.stale { background: var(--red); animation: pulse 2s ease-in-out infinite; }

  /* Stat cards */
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px; margin-top: 4px; }
  .stat-card {
    background: var(--bg);
    border-radius: 4px;
    padding: 4px 6px;
    text-align: center;
  }
  .stat-value { font-size: 15px; font-weight: 700; }
  .stat-label { font-size: 8px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-green { color: var(--green); }
  .stat-red { color: var(--red); }
  .stat-yellow { color: var(--yellow); }
  .stat-cyan { color: var(--cyan); }
  .stat-purple { color: var(--purple); }

  .phase-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 6px;
    animation: pulse 1.5s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .phase-active { background: var(--green); }
  .phase-idle { background: var(--text-dim); animation: none; }
  .phase-ssh_fail { background: var(--red); animation: none; }
  .phase-local { background: var(--yellow); animation: none; }

  .run-state-badge {
    font-size: 10px; padding: 2px 8px; border-radius: 3px;
    font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-left: 8px;
  }
  .run-state-active { background: rgba(63,185,80,0.2); color: var(--green); }
  .run-state-idle { background: rgba(139,148,158,0.15); color: var(--text-dim); }
  .run-state-ssh_fail { background: rgba(248,81,73,0.2); color: var(--red); }
  .run-state-local { background: rgba(210,169,34,0.2); color: var(--yellow); }

  /* Table */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: left; padding: 6px 8px; color: var(--text-dim);
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; background: var(--bg2);
  }
  td { padding: 5px 8px; border-bottom: 1px solid var(--bg3); white-space: nowrap; }
  tr:hover { background: var(--bg3); }
  tr.kept td { color: var(--green); }
  tr.fail td { color: var(--red); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }

  /* Charts */
  .charts-panel { grid-column: 1 / 3; }
  .chart-rows { display: flex; flex-direction: column; gap: 6px; }
  .chart-container { display: flex; gap: 8px; height: 80px; }
  .chart-box { flex: 1; position: relative; min-width: 0; }
  .chart-title { font-size: 9px; color: var(--text-dim); margin-bottom: 1px; text-transform: uppercase; letter-spacing: 0.5px; }
  canvas { width: 100% !important; height: 66px !important; }

  .left-stack { grid-column: 1 / 2; display: flex; flex-direction: column; gap: 8px; min-width: 0; overflow: hidden; }
  .right-stack { grid-column: 2 / 3; display: flex; flex-direction: column; gap: 8px; min-width: 0; overflow: hidden; }

  .code-view {
    background: var(--bg); border-radius: 4px; padding: 10px;
    font-size: 11px; line-height: 1.4; overflow: auto;
    white-space: pre; tab-size: 4; max-width: 100%; width: 0; min-width: 100%;
  }
  .code-view .ln { color: var(--text-dim); user-select: none; display: inline-block; width: 40px; text-align: right; margin-right: 12px; }

  .log-view {
    background: var(--bg); border-radius: 4px; padding: 10px;
    font-size: 11px; line-height: 1.6; overflow: auto;
    white-space: pre-wrap; word-break: break-all;
    max-width: 100%; width: 0; min-width: 100%;
  }
  .log-error { color: var(--red); }
  .log-success { color: var(--green); }
  .log-warn { color: var(--yellow); }
  .log-info { color: var(--cyan); }
  .log-dim { color: var(--text-dim); }

  .tab-bar { display: flex; gap: 0; border-bottom: 1px solid var(--border); }
  .tab {
    padding: 6px 14px; font-size: 11px; color: var(--text-dim);
    cursor: pointer; border-bottom: 2px solid transparent;
    text-transform: uppercase; letter-spacing: 0.5px; transition: all 0.2s;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--cyan); border-bottom-color: var(--cyan); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .notebook-view {
    background: var(--bg); border-radius: 4px; padding: 10px;
    font-size: 11px; line-height: 1.6; overflow: auto;
    white-space: pre-wrap; max-width: 100%; width: 0; min-width: 100%;
  }

  .warnings-bar {
    background: rgba(210,169,34,0.15); color: var(--yellow);
    padding: 4px 12px; font-size: 11px;
    border-bottom: 1px solid var(--border); display: none;
  }
  .warnings-bar.visible { display: block; }

  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .top-row { flex-direction: column; }
    .top-row .gpu-status { flex: none; }
    .full-row, .charts-panel, .left-stack, .right-stack { grid-column: 1 / 2; }
  }

  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }
</style>
</head>
<body>

<header>
  <div style="display:flex;align-items:center;gap:12px">
    <h1>ART2 v3 RL MONITOR</h1>
    <span class="phase-tag">PURE-RL</span>
  </div>
  <div class="meta">
    <span id="header-mode"></span> &nbsp;|&nbsp;
    <span id="header-time"></span> &nbsp;|&nbsp;
    <span id="header-poll"></span> &nbsp;|&nbsp;
    <span id="header-fetch"></span>
  </div>
</header>
<div class="warnings-bar" id="warnings-bar"></div>

<div class="grid">
  <!-- Row 1: GPU + Session -->
  <div class="top-row">
    <div class="panel gpu-status">
      <div class="panel-header">GPU <span id="gpu-name" style="font-weight:400"></span><span class="gpu-refresh-dot" id="gpu-dot"></span></div>
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
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
          <span class="phase-dot" id="phase-dot"></span>
          <span id="phase-label" style="font-weight:600;font-size:12px"></span>
          <span id="process-info" style="color:var(--text-dim);margin-left:auto;font-size:11px"></span>
        </div>
        <div class="stats-grid">
          <div class="stat-card"><div class="stat-value stat-cyan" id="stat-total">--</div><div class="stat-label">Experiments</div></div>
          <div class="stat-card"><div class="stat-value stat-green" id="stat-best">--</div><div class="stat-label">Best Score</div></div>
          <div class="stat-card"><div class="stat-value stat-purple" id="stat-best-exp">--</div><div class="stat-label">Best Exp</div></div>
          <div class="stat-card"><div class="stat-value stat-yellow" id="stat-elapsed">--</div><div class="stat-label">Elapsed</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 2: Charts -->
  <div class="panel full-row charts-panel">
    <div class="panel-header">Metrics</div>
    <div class="panel-body" style="padding:6px 10px">
      <div class="chart-rows">
        <div class="chart-container">
          <div class="chart-box"><div class="chart-title">Score</div><canvas id="chart-score"></canvas></div>
          <div class="chart-box"><div class="chart-title">Return %</div><canvas id="chart-return"></canvas></div>
          <div class="chart-box"><div class="chart-title">Sortino</div><canvas id="chart-sortino"></canvas></div>
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
                <th>ID</th><th>Time</th><th>Score</th><th>Return</th><th>DD</th><th>Sortino</th><th>Trades</th><th>Status</th>
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
          <div class="tab active" data-tab="log">GPU Log</div>
          <div class="tab" data-tab="code" id="code-tab-label">train.py</div>
          <div class="tab" data-tab="notebook">Lab Notebook</div>
        </div>
      </div>
      <div class="panel-body" style="padding:0">
        <div class="tab-content active" id="tab-log">
          <div class="log-view" id="log-content" style="max-height:400px">Waiting for data...</div>
        </div>
        <div class="tab-content" id="tab-code">
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
// -- Mini chart library --
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

  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = pad + (h - 2*pad) * i / 3;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Zero line if range crosses zero
  if (mn < 0 && mx > 0) {
    const zeroY = h - pad - (0 - mn) / range * (h - 2*pad);
    ctx.strokeStyle = '#8b949e44'; ctx.lineWidth = 1;
    ctx.setLineDash([4,4]);
    ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath(); ctx.moveTo(pad, h - pad);
  for (let i = 0; i < data.length; i++) {
    const x = pad + (w - 2*pad) * i / (data.length - 1 || 1);
    const y = h - pad - (data[i] - mn) / range * (h - 2*pad);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w - pad, h - pad); ctx.closePath();
  ctx.fillStyle = color + '15'; ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = pad + (w - 2*pad) * i / (data.length - 1 || 1);
    const y = h - pad - (data[i] - mn) / range * (h - 2*pad);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();

  // Dots on each data point
  for (let i = 0; i < data.length; i++) {
    const x = pad + (w - 2*pad) * i / (data.length - 1 || 1);
    const y = h - pad - (data[i] - mn) / range * (h - 2*pad);
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI*2);
    ctx.fillStyle = color; ctx.fill();
  }

  const lastVal = data[data.length - 1];
  ctx.fillStyle = color; ctx.font = '11px monospace'; ctx.textAlign = 'right';
  ctx.fillText(lastVal.toFixed(opts.decimals !== undefined ? opts.decimals : 3), w - 6, 14);
  ctx.fillStyle = '#8b949e'; ctx.font = '9px monospace';
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

function colorizeLine(line) {
  if (line.includes('ERROR') || line.includes('CRASH') || line.includes('FAILED')) return `<span class="log-error">${escapeHtml(line)}</span>`;
  if (line.includes('KEPT') || line.includes('keep') || line.includes('IMPROVED') || line.includes('best')) return `<span class="log-success">${escapeHtml(line)}</span>`;
  if (line.includes('WARNING') || line.includes('revert') || line.includes('REVERT')) return `<span class="log-warn">${escapeHtml(line)}</span>`;
  if (line.includes('===') || line.includes('score') || line.includes('update')) return `<span class="log-info">${escapeHtml(line)}</span>`;
  return `<span class="log-dim">${escapeHtml(line)}</span>`;
}

function setGauge(prefix, pct, text) {
  const bar = document.getElementById(prefix + '-bar');
  const val = document.getElementById(prefix + '-val');
  if (!bar || !val) return;
  bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
  val.textContent = text;
  bar.className = 'gauge-fill ' + (pct > 80 ? 'fill-red' : pct > 50 ? 'fill-yellow' : 'fill-green');
}

function fmtDuration(s) {
  if (s < 60) return s.toFixed(0) + 's';
  if (s < 3600) return (s/60).toFixed(0) + 'm';
  return Math.floor(s/3600) + 'h' + String(Math.floor((s%3600)/60)).padStart(2,'0') + 'm';
}

// -- GPU auto-refresh (independent, every 3s) --
let lastGpuUpdate = 0;

async function updateGpu() {
  try {
    const resp = await fetch('/api/gpu');
    const g = await resp.json();
    const dot = document.getElementById('gpu-dot');

    if (g.gpu) {
      const gpu = g.gpu;
      const isStale = gpu._stale === true;
      const gpuAge = g.last_gpu_time > 0 ? Math.round((Date.now()/1000) - g.last_gpu_time) : 0;
      let nameLabel = gpu.name;
      if (isStale && gpuAge > 0) {
        const ageStr = gpuAge < 60 ? gpuAge + 's' : Math.round(gpuAge/60) + 'm';
        nameLabel += ` (stale ${ageStr})`;
      }
      document.getElementById('gpu-name').textContent = nameLabel;
      document.getElementById('gpu-content').style.opacity = isStale ? '0.5' : '1';
      dot.className = 'gpu-refresh-dot' + (isStale ? ' stale' : '');

      const memPct = gpu.mem_total_mb > 0 ? (gpu.mem_used_mb / gpu.mem_total_mb * 100) : 0;
      const powerPct = gpu.power_limit_w > 0 ? (gpu.power_w / gpu.power_limit_w * 100) : 0;
      setGauge('gpu-util', gpu.util_pct, gpu.util_pct.toFixed(0) + '%');
      setGauge('gpu-mem', memPct, `${(gpu.mem_used_mb/1024).toFixed(1)}/${(gpu.mem_total_mb/1024).toFixed(0)}GB`);
      setGauge('gpu-temp', Math.min(gpu.temp_c, 100), gpu.temp_c.toFixed(0) + 'C');
      setGauge('gpu-power', powerPct, `${gpu.power_w.toFixed(0)}/${gpu.power_limit_w.toFixed(0)}W`);
      lastGpuUpdate = Date.now();
    } else {
      const failCount = g.ssh_fail_count || 0;
      let label = failCount > 0 ? `reconnecting (${failCount})...` : '(no GPU)';
      document.getElementById('gpu-name').textContent = label;
      document.getElementById('gpu-content').style.opacity = '0.3';
      dot.className = 'gpu-refresh-dot stale';
    }
  } catch(e) {
    console.error('GPU update error:', e);
  }
}

// GPU refreshes independently every 3 seconds
setInterval(updateGpu, 3000);
updateGpu();

// -- Main dashboard update --
let lastExpHash = '';
let lastTrainHash = '';
let lastNotebookHash = '';
let lastLogHash = '';

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

    // Warnings
    const wBar = document.getElementById('warnings-bar');
    const warns = data.warnings || [];
    if (warns.length > 0) {
      wBar.textContent = warns.join(' | ');
      wBar.classList.add('visible');
    } else {
      wBar.classList.remove('visible');
    }

    // Run state
    const runState = data.run_state || 'unknown';
    const badge = document.getElementById('run-state-badge');
    const badgeLabels = {active: 'TRAINING', idle: 'IDLE', ssh_fail: 'SSH FAIL', local: 'LOCAL'};
    badge.textContent = badgeLabels[runState] || runState;
    badge.className = 'run-state-badge run-state-' + runState;

    const phaseDot = document.getElementById('phase-dot');
    phaseDot.className = 'phase-dot phase-' + runState;
    const phaseLabels = {
      active: 'PPO Training Active',
      idle: 'GPU Idle',
      ssh_fail: 'SSH Connection Failed',
      local: 'Local Mode (no GPU)',
    };
    document.getElementById('phase-label').textContent = phaseLabels[runState] || runState;

    const proc = data.process || {};
    if (proc.status === 'running') {
      document.getElementById('process-info').textContent = `PID ${proc.pid} | uptime ${proc.uptime}`;
    } else {
      document.getElementById('process-info').textContent = '';
    }

    // Session stats
    const session = data.session || {};
    document.getElementById('stat-total').textContent = session.experiment_count || 0;
    const bs = session.best_score;
    document.getElementById('stat-best').textContent = (bs != null && bs > -5) ? bs.toFixed(4) : '--';
    document.getElementById('stat-best-exp').textContent = session.best_experiment || '--';
    const elapsedH = session.session_start ? ((Date.now()/1000 - session.session_start) / 3600) : 0;
    document.getElementById('stat-elapsed').textContent = elapsedH > 0 ? elapsedH.toFixed(1) + 'h' : '--';

    // Charts
    const exps = data.experiments || [];
    const scores = exps.map(e => e.score_num).filter(s => s > -999);
    if (scores.length >= 1) drawChart('chart-score', scores, '#3fb950', {decimals: 3});
    const returns = exps.map(e => e.total_return).filter(r => r !== undefined && r !== null);
    if (returns.length >= 1) drawChart('chart-return', returns, '#58a6ff', {decimals: 2});
    const sortinos = exps.map(e => e.sortino).filter(s => s !== undefined && s !== null);
    if (sortinos.length >= 1) drawChart('chart-sortino', sortinos, '#bc8cff', {decimals: 2});

    // Experiments table
    const expHash = exps.map(e => e.experiment + '|' + e.score_num).join(',');
    if (expHash !== lastExpHash) {
      lastExpHash = expHash;
      document.getElementById('exp-count').textContent = `(${exps.length})`;
      const tbody = document.getElementById('exp-tbody');
      tbody.innerHTML = '';
      for (let i = exps.length - 1; i >= 0; i--) {
        const e = exps[i];
        const tr = document.createElement('tr');
        const status = e.status || '?';
        tr.className = status === 'keep' ? 'kept' : status === 'fail' ? 'fail' : '';
        const scoreDisplay = e.score_num > -999 ? e.score_num.toFixed(4) : '--';
        const retDisplay = e.total_return !== undefined ? (e.total_return * 100).toFixed(1) + '%' : '--';
        const ddDisplay = e.max_drawdown !== undefined ? (e.max_drawdown * 100).toFixed(1) + '%' : '--';
        const sortDisplay = e.sortino !== undefined ? e.sortino.toFixed(2) : '--';
        const timeDisplay = e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}) : '';
        tr.innerHTML = `
          <td class="num">${escapeHtml(e.experiment || '?')}</td>
          <td class="num" style="color:var(--text-dim)">${timeDisplay}</td>
          <td class="num">${scoreDisplay}</td>
          <td class="num">${retDisplay}</td>
          <td class="num">${ddDisplay}</td>
          <td class="num">${sortDisplay}</td>
          <td class="num">${e.trades || '--'}</td>
          <td style="color:${status === 'keep' ? 'var(--green)' : status === 'fail' ? 'var(--red)' : 'var(--text-dim)'}">${status.toUpperCase()}</td>
        `;
        tbody.appendChild(tr);
      }
    }

    // Log
    if (data.log_tail) {
      const logHash = data.log_tail.length + data.log_tail.slice(-100);
      if (logHash !== lastLogHash) {
        lastLogHash = logHash;
        const lines = data.log_tail.split('\n');
        document.getElementById('log-content').innerHTML = lines.map(colorizeLine).join('\n');
        const logEl = document.getElementById('log-content');
        logEl.scrollTop = logEl.scrollHeight;
      }
    } else if (data.poll_count > 2) {
      document.getElementById('log-content').textContent = data.ssh_ok
        ? 'Waiting for log output...'
        : 'No SSH connection -- no live log';
    }

    // train.py
    if (data.train_py) {
      const hash = data.train_py.length + data.train_py.slice(0,100);
      if (hash !== lastTrainHash) {
        lastTrainHash = hash;
        document.getElementById('code-content').innerHTML = highlightPython(data.train_py);
      }
      const src = data.train_py_source || '';
      document.getElementById('code-tab-label').textContent = src ? src : 'train.py';
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

// Main dashboard polls every 5 seconds
setInterval(update, 5000);
update();
window.addEventListener('resize', update);

// -- Auto-reload when monitor.py source changes on disk --
let _knownVersion = null;
async function checkVersion() {
  try {
    const resp = await fetch('/api/version');
    const data = await resp.json();
    if (_knownVersion === null) {
      _knownVersion = data.hash;
    } else if (data.hash !== _knownVersion) {
      console.log('[monitor] Source changed, reloading...');
      // Small delay to let the server restart
      setTimeout(() => location.reload(), 1500);
    }
  } catch(e) {
    // Server might be restarting -- retry on next tick
  }
}
setInterval(checkVersion, 2000);
checkVersion();
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
        elif self.path == "/api/gpu":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            gpu_state = get_gpu_state()
            self.wfile.write(json.dumps(gpu_state, default=str).encode())
        elif self.path == "/api/version":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps({"hash": _current_source_hash()}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ART2 v3 RL web dashboard")
    parser.add_argument("--host", type=str, help="Remote SSH hostname (overrides .deploy-state)")
    parser.add_argument("--ssh-port", type=int, help="Remote SSH port (overrides .deploy-state)")
    parser.add_argument("--local", action="store_true", help="Local only (no SSH)")
    parser.add_argument("--port", type=int, default=8430, help="Web server port (default: 8430)")
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
            try:
                remote_port = int(state.get("SSH_PORT", 22))
            except (ValueError, TypeError):
                remote_port = 22

    # Mutable refs so pollers can track .deploy-state changes
    host_ref = [remote_host]
    port_ref = [remote_port or 22]

    # Start dedicated GPU poller (fast, every 3s)
    gpu_thread = threading.Thread(
        target=gpu_poller_loop,
        args=(host_ref, port_ref, args.local),
        daemon=True,
    )
    gpu_thread.start()

    # Start main data poller
    poller = threading.Thread(
        target=poller_loop,
        args=(host_ref, port_ref, args.local, args.interval),
        daemon=True,
    )
    poller.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    url = f"http://localhost:{args.port}"
    print(f"v3 RL Dashboard running at {url}")
    print(f"  Mode: {'Remote ' + str(remote_host) + ':' + str(remote_port) if remote_host else 'Local'}")
    print(f"  Poll interval: {args.interval}s (GPU: 3s)")
    print(f"  Press Ctrl+C to stop")

    if not args.no_open:
        import webbrowser
        webbrowser.open(url)

    server.timeout = 0.5
    try:
        while not _restart_flag.is_set():
            server.handle_request()
        print("[monitor] Restarting...", flush=True)
        server.server_close()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
