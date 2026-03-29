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

import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
SSH_PASS = "autoresearch2026"
_SELF_PATH = Path(__file__).resolve()
_SELF_HASH_AT_START = hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
_restart_flag = threading.Event()


def _check_self_restart():
    """If monitor.py source changed on disk, signal main thread to re-exec."""
    try:
        current_hash = hashlib.md5(_SELF_PATH.read_bytes()).hexdigest()
        if current_hash != _SELF_HASH_AT_START:
            print(f"[monitor] Source changed on disk — signaling restart...", flush=True)
            _restart_flag.set()
    except Exception:
        pass  # non-fatal

# -- Shared state (updated by background poller) --------------------------

_state_lock = threading.Lock()
_state: dict = {
    "experiments": [],
    "status": None,
    "log_tail": None,
    "gpu": None,
    "train_py": None,
    "mode": "initializing",
    "training_mode": "unknown",  # "opus" or "pbt"
    "run_state": "unknown",  # active, completed, idle, waiting
    "runs": [],
    "active_run": None,
    "prev_run": None,
    "last_fetch": 0,
    "fetch_time": 0,
    "poll_count": 0,
    "ssh_ok": False,
    "pbt": None,  # PBT state when in PBT mode
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
    ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=15 -o ServerAliveCountMax=2"
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
        'cat /root/autoresearch-trading/training/best_train.py 2>/dev/null || cat /root/autoresearch-trading/training/train.py 2>/dev/null; echo "---SEP---"; '
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


def fetch_remote_gpu(host: str, port: int) -> dict | None:
    """Lightweight independent GPU fetch — fallback when compound SSH times out."""
    raw = _ssh_cmd(host, port,
        'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null',
        timeout=8)
    return _parse_gpu_csv(raw.strip()) if raw else None


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
        # Validate: must look like a run name, not an error message
        if name and name.startswith("run-"):
            return name
    # Fallback: find the most recent run- directory
    if RESULTS_ROOT.exists():
        run_dirs = sorted(
            [d for d in RESULTS_ROOT.iterdir() if d.is_dir() and d.name.startswith("run-")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            return run_dirs[0].name
    return None


# -- PBT state loading -----------------------------------------------------

PBT_STATE_PATH = PROJECT_ROOT / "training" / ".pbt_state.json"


def load_pbt_state(experiments: list[dict] | None = None,
                   runs: list[dict] | None = None) -> dict | None:
    """Load PBT state from .pbt_state.json and build a summary for the dashboard.

    Merges PBT experiments from all local runs so data persists across
    deployment teardown/setup cycles.
    """
    if not PBT_STATE_PATH.exists():
        return None
    try:
        raw = json.loads(PBT_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Merge PBT experiments from local runs that belong to THIS PBT run.
    # Parse start time from pbt_run_id (format: pbt-YYYY-MM-DD-HHMMSS) to filter.
    pbt_run_id = raw.get("pbt_run_id", "")
    pbt_start_ts = None
    if pbt_run_id.startswith("pbt-"):
        try:
            pbt_start_ts = datetime.strptime(pbt_run_id[4:], "%Y-%m-%d-%H%M%S")
        except ValueError:
            pass

    all_pbt_experiments = list(experiments or [])
    if runs and pbt_start_ts:
        existing_ids = {e.get("id") for e in all_pbt_experiments}
        for run in runs:
            # Only include runs that started at or after this PBT run
            run_name = run.get("name", "")
            run_ts = None
            if run_name.startswith("run-"):
                try:
                    run_ts = datetime.strptime(run_name[4:], "%Y-%m-%d-%H%M%S")
                except ValueError:
                    pass
            if run_ts and run_ts < pbt_start_ts - timedelta(minutes=5):
                continue
            for e in run.get("experiments", []):
                if e.get("pbt_generation") is not None and e.get("id") not in existing_ids:
                    all_pbt_experiments.append(e)
                    existing_ids.add(e.get("id"))

    generations = raw.get("generations", [])
    summary = {
        "run_id": raw.get("pbt_run_id", ""),
        "generation": raw.get("generation", 0),
        "max_generations": raw.get("max_generations", 0),
        "population_size": raw.get("population_size", 0),
        "focus": raw.get("focus", "all"),
        "current_member": raw.get("current_member", 0),
        "stagnation_count": raw.get("stagnation_count", 0),
        "base_score": raw.get("base_score", 0),
        "best_pbt_score": raw.get("best_pbt_score", 0),
        "time_budget_per_member": raw.get("time_budget_per_member", 300),
        "generations": [],
    }

    for g in generations:
        gen_summary = {
            "generation": g["generation"],
            "members": [],
        }
        scores = []
        for m in g.get("members", []):
            member = {
                "member_id": m["member_id"],
                "score": m.get("score"),
                "role": m.get("role", "baseline" if m["member_id"] == 0 and g["generation"] == 0 else "perturbed"),
                "parent": m.get("parent", ""),
                "anomaly_flags": m.get("anomaly_flags", []),
                "config_summary": _summarize_config(m.get("config", {})),
                "wall_time": m.get("wall_time"),
            }
            # Extract key metrics if available
            metrics = m.get("metrics", {})
            if metrics:
                member["pf"] = metrics.get("profit_factor")
                member["tpd"] = metrics.get("trades_per_day")
                member["wr"] = metrics.get("win_rate")
                member["sharpe"] = metrics.get("trade_sharpe")
                member["stop_rate"] = metrics.get("stop_loss_rate")
            gen_summary["members"].append(member)
            if m.get("score") is not None:
                scores.append(m["score"])

        if scores:
            gen_summary["best_score"] = max(scores)
            gen_summary["worst_score"] = min(scores)
            gen_summary["avg_score"] = sum(scores) / len(scores)
        gen_summary["completed"] = len(scores)
        summary["generations"].append(gen_summary)

    # Build in-progress generation from population + experiments
    current_gen = raw.get("generation", 0)
    population = raw.get("population", [])
    already_has_gen = any(g["generation"] == current_gen for g in summary["generations"])

    if population and not already_has_gen:
        gen_exps = {}
        if all_pbt_experiments:
            for e in all_pbt_experiments:
                if e.get("pbt_generation") == current_gen:
                    gen_exps[e.get("pbt_member")] = e

        gen_summary = {"generation": current_gen, "members": [], "in_progress": True}
        scores = []
        for p in population:
            mid = p["id"]
            exp = gen_exps.get(mid)
            member = {
                "member_id": mid,
                "score": exp.get("score") if exp else None,
                "role": p.get("specialist_type") or ("baseline" if mid == 0 and current_gen == 0 else "perturbed"),
                "config_summary": _summarize_config(p.get("config", {})),
                "anomaly_flags": exp.get("anomaly_flags", []) if exp else [],
                "wall_time": exp.get("wall_time") if exp else None,
            }
            if exp:
                member["pf"] = exp.get("profit_factor")
                member["tpd"] = exp.get("trades_per_day")
                member["wr"] = exp.get("win_rate")
                member["sharpe"] = exp.get("trade_sharpe")
                member["stop_rate"] = exp.get("stop_loss_rate")
            if member["score"] is not None:
                scores.append(member["score"])
            gen_summary["members"].append(member)

        if scores:
            gen_summary["best_score"] = max(scores)
            gen_summary["worst_score"] = min(scores)
            gen_summary["avg_score"] = sum(scores) / len(scores)
        gen_summary["completed"] = len(scores)
        summary["generations"].append(gen_summary)

    return summary


def _summarize_config(config: dict) -> str:
    """Create a short string summarizing PBT config values.

    Shows the most distinctive parameters with abbreviated names.
    """
    if not config:
        return "defaults"
    # Abbreviation map: prefix-strip + short names for readability (v13)
    abbrev = {
        "TRAIN_LR": "lr", "TRAIN_WEIGHT_DECAY": "wd", "TRAIN_DROPOUT": "do",
        "TRAIN_WARMUP_RATIO": "warm", "TRAIN_COOLDOWN_RATIO": "cool",
        "TRAIN_GRAD_CLIP": "gc", "WEIGHT_RECENT_BOOST": "rcnt",
        "WEIGHT_DAY_DIVERSITY": "div", "REG_GATE_ENTROPY": "gent",
        "REG_TEMPORAL_SMOOTH": "tsmth", "WARM_FREEZE_RATIO": "frz",
        "TRAIN_DAY_SEQ_RATIO": "seq",
        "TRAIN_GATE_W": "GATE", "TRAIN_DIR_W": "DIR",
        "TRAIN_PNL_W": "PNL", "TRAIN_CONF_W": "CONF",
        "TRAIN_EXIT_W": "EXIT", "TRAIN_VALUE_W": "VAL",
        "TRAIN_RISK_W": "RISK",
    }
    # Priority keys to show first (most impactful for v13 sniper_loss)
    priority = ["TRAIN_GATE_W", "TRAIN_DIR_W", "TRAIN_PNL_W", "TRAIN_RISK_W",
                "TRAIN_LR", "TRAIN_DROPOUT", "TRAIN_WEIGHT_DECAY"]
    items = []
    shown = set()
    for k in priority:
        v = config.get(k)
        if v is not None:
            name = abbrev.get(k, k.replace("TRAIN_", "").replace("_W", ""))
            items.append(f"{name}={v:.3g}")
            shown.add(k)
    for k, v in config.items():
        if k in shown or v is None:
            continue
        name = abbrev.get(k, k.replace("TRAIN_", ""))
        items.append(f"{name}={v:.2g}")
    return ", ".join(items[:5]) or "defaults"


def detect_training_mode(experiments: list[dict]) -> str:
    """Detect whether experiments are from PBT or Opus-driven mode.

    Primary signal: .pbt_state.json existence (persists across deployments).
    Fallback: check recent experiments for pbt_generation field.
    """
    # Primary signal: local .pbt_state.json exists and has PBT data
    if PBT_STATE_PATH.exists():
        try:
            raw = json.loads(PBT_STATE_PATH.read_text())
            gen = raw.get("generation", 0)
            max_gen = raw.get("max_generations", 0)
            # Active PBT: not yet completed all generations
            if gen < max_gen or raw.get("population"):
                return "pbt"
            # Completed PBT: still show results if generations were recorded
            if raw.get("generations"):
                return "pbt"
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: check recent experiments for pbt_generation field
    if not experiments:
        return "unknown"
    recent = experiments[-3:] if len(experiments) >= 3 else experiments
    if any(e.get("pbt_generation") is not None for e in recent):
        return "pbt"
    return "opus"


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


def fetch_local_gpu() -> dict | None:
    """Try to get GPU info from local nvidia-smi (for local training mode)."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return _parse_gpu_csv(r.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


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
    # Try the active run name first
    if active_run and (lp / active_run).is_dir():
        return lp / active_run
    # Try current_run.txt inside the local path
    if lp.is_dir() and (lp / "current_run.txt").exists():
        rn = (lp / "current_run.txt").read_text().strip()
        if rn and rn.startswith("run-") and (lp / rn).is_dir():
            return lp / rn
    # Fallback: most recent run- directory (NOT lp itself — that's the parent)
    if lp.is_dir():
        run_dirs = sorted(
            [d for d in lp.iterdir() if d.is_dir() and d.name.startswith("run-")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if run_dirs:
            return run_dirs[0]
    return None


def _classify_run_state(status: dict | None, ssh_ok: bool, host: str | None) -> str:
    """Determine the lifecycle state of the current run.

    Returns: 'active', 'completed', 'idle', or 'waiting'.
    """
    if not status:
        # No status at all — either no run exists or SSH failed before any data
        return "waiting" if host else "idle"

    phase = status.get("phase", "")
    time_left = status.get("time_remaining_h", 0)

    # Actively training
    if phase in ("training", "calling_claude", "evaluating", "saving",
                 "startup", "between_experiments", "smoke_check"):
        return "active"

    # Explicitly finished
    if phase in ("completed", "idle"):
        return "completed"

    if phase == "error":
        return "completed"

    # Has time remaining → probably still going
    if time_left and time_left > 0:
        return "active"

    return "completed"


def poller_loop(host: str | None, port: int, local_path: str | None, interval: int, host_pinned: bool = False):
    """Background thread that polls remote/local and updates shared state."""
    # If host_pinned, the user explicitly passed --host so we never reload.
    # Otherwise, re-read .deploy-state each cycle to auto-discover new deployments.

    mode = f"Remote: {host}:{port}" if host else f"Local: {local_path or 'results/'}"
    set_state(mode=mode)
    prev_run = None

    while True:
        # Auto-restart if our source file changed (picks up code edits without manual kill/restart)
        _check_self_restart()

        try:
            t0 = time.time()

            # Reload .deploy-state each poll to pick up new deployments
            if not host_pinned:
                ds = load_deploy_state()
                if ds:
                    new_host = ds.get("SSH_HOST")
                    new_port = int(ds.get("SSH_PORT", 22))
                else:
                    new_host, new_port = None, 22
                if new_host != host or new_port != port:
                    print(f"[monitor] Deployment changed: {host}:{port} → {new_host}:{new_port}", flush=True)
                    host, port = new_host, new_port
                new_mode = f"Remote: {host}:{port}" if host else f"Local: {local_path or 'results/'}"
                if new_mode != mode:
                    mode = new_mode
                    set_state(mode=mode)

            runs = fetch_all_local_runs()
            active_run = get_active_run_name()

            # Detect run transition — log and track
            if active_run != prev_run and prev_run is not None:
                print(f"[monitor] Run changed: {prev_run} → {active_run}", flush=True)
            prev_run = active_run

            experiments = []
            status = None
            log_tail = None
            gpu = None
            train_py = None
            train_py_source = ""
            ssh_ok = False
            prev_gpu = get_state().get("gpu")

            # Try remote first
            if host:
                experiments, status, log_tail, gpu, train_py, _ = fetch_remote(host, port)
                ssh_ok = bool(experiments or status or gpu)
                if train_py:
                    train_py_source = "remote: best_train.py"
                # Independent GPU retry if compound fetch missed it
                if not gpu:
                    gpu = fetch_remote_gpu(host, port)

            # Local GPU fallback — if no GPU from SSH, try local nvidia-smi
            if not gpu:
                gpu = fetch_local_gpu()

            # Preserve last-known GPU data if all fetches failed
            if not gpu and prev_gpu:
                gpu = {**prev_gpu, "_stale": True}
            elif gpu and "_stale" in gpu:
                del gpu["_stale"]

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
                    # Try best_train.py first (the promoted best), then active train.py
                    for tp_candidate in [
                        RESULTS_ROOT.parent / "training" / "best_train.py",
                        run_dir / "best_train.py",
                        RESULTS_ROOT.parent / "training" / "train.py",
                    ]:
                        if tp_candidate.exists():
                            train_py = tp_candidate.read_text()
                            train_py_source = str(tp_candidate.relative_to(PROJECT_ROOT))
                            break

            run_state = _classify_run_state(status, ssh_ok, host)
            training_mode = detect_training_mode(experiments)
            pbt = load_pbt_state(experiments, runs) if training_mode == "pbt" else None
            fetch_time = time.time() - t0
            poll_count = get_state()["poll_count"] + 1

            set_state(
                experiments=experiments,
                status=status,
                log_tail=log_tail,
                gpu=gpu,
                train_py=train_py,
                train_py_source=train_py_source,
                training_mode=training_mode,
                run_state=run_state,
                runs=runs,
                active_run=active_run,
                prev_run=prev_run,
                last_fetch=time.time(),
                fetch_time=fetch_time,
                poll_count=poll_count,
                ssh_ok=ssh_ok,
                pbt=pbt,
            )
        except Exception as e:
            print(f"[poller] Error: {e}", file=sys.stderr)

        time.sleep(interval)


# -- HTML Dashboard --------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ART² Training Monitor</title>
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
  /* Row 1: GPU (narrow) + Active Run (wide) — use flex row */
  .top-row {
    grid-column: 1 / 3;
    display: flex;
    gap: 8px;
    min-width: 0;
  }
  .top-row .gpu-status { flex: 0 0 240px; min-width: 0; }
  .top-row .active-run { flex: 1; min-width: 0; }
  /* Full-width panels */
  .charts-panel { grid-column: 1 / 3; }
  /* Bottom: left stack + reasoning */
  .left-stack { grid-column: 1 / 2; display: flex; flex-direction: column; gap: 8px; min-width: 0; overflow: hidden; }
  .reasoning { grid-column: 2 / 3; }

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
    white-space: pre-wrap;
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

  /* Charts */
  .chart-rows { display: flex; flex-direction: column; gap: 6px; }
  .chart-container { display: flex; gap: 8px; height: 80px; }
  .chart-box { flex: 1; position: relative; min-width: 0; }
  .chart-title { font-size: 9px; color: var(--text-dim); margin-bottom: 1px; text-transform: uppercase; letter-spacing: 0.5px; }
  canvas { width: 100% !important; height: 66px !important; }
  .chart-row-label { font-size: 9px; color: var(--cyan); font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; }

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
  .phase-idle { background: var(--text-dim); animation: none; }
  .phase-saving { background: var(--purple); }
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
  .run-state-waiting { background: rgba(210,169,34,0.2); color: var(--yellow); animation: pulse 2s ease-in-out infinite; }

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

  /* Responsive — single column for narrow windows */
  @media (max-width: 900px) {
    .grid { grid-template-columns: 1fr; }
    .top-row { flex-direction: column; }
    .top-row .gpu-status { flex: none; }
    .full-row, .charts-panel, .left-stack, .reasoning { grid-column: 1 / 2; }
  }

  .stale-warning {
    background: rgba(248,81,73,0.15);
    color: var(--red);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    animation: pulse 2s ease-in-out infinite;
  }

  /* Training mode badge */
  .training-mode-badge {
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 4px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: help;
  }
  .training-mode-opus { background: rgba(88,166,255,0.2); color: var(--cyan); }
  .training-mode-pbt { background: rgba(188,140,255,0.2); color: var(--purple); }
  .training-mode-unknown { background: rgba(139,148,158,0.15); color: var(--text-dim); }

  /* PBT Population Grid */
  .pbt-panel { min-width: 0; overflow: hidden; }
  .pbt-grid-container { display: flex; gap: 16px; }
  .pbt-overview { flex: 0 0 180px; }
  .pbt-overview .stat-card { margin-bottom: 6px; }
  .pbt-gens { flex: 1; overflow-x: auto; }

  /* Compact PBT table */
  .pbt-table { width: 100%; border-collapse: collapse; font-size: 11px; table-layout: fixed; }
  .pbt-table th {
    text-align: left; padding: 4px 8px; color: var(--text-dim);
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border); background: var(--bg2);
    position: sticky; top: 0;
  }
  /* Column widths for fixed layout */
  .pbt-table .col-gen { width: 36px; }
  .pbt-table .col-mem { width: 28px; }
  .pbt-table .col-role { width: 70px; }
  .pbt-table .col-score { width: 56px; }
  .pbt-table .col-pf { width: 40px; }
  .pbt-table .col-tpd { width: 40px; }
  .pbt-table .col-wr { width: 40px; }
  .pbt-table .col-sl { width: 40px; }
  .pbt-table .col-cfg { width: auto; }
  .pbt-table .col-flags { width: 48px; }
  .pbt-table td { padding: 4px 8px; border-bottom: 1px solid var(--bg3); overflow: hidden; text-overflow: ellipsis; }
  .pbt-table tr:hover { background: var(--bg3); }
  .pbt-table .gen-row td {
    background: var(--bg3); font-weight: 700; color: var(--cyan);
    font-size: 11px; padding: 5px 8px; letter-spacing: 0.5px;
    cursor: pointer; user-select: none;
  }
  .pbt-table .gen-row:hover td { background: #2d333b; }
  .pbt-table .gen-row td .arrow { display: inline-block; width: 12px; font-size: 10px; transition: transform 0.15s; }
  .pbt-table .gen-row td .arrow.collapsed { transform: rotate(-90deg); }
  .pbt-table .best-row td { color: var(--green); }
  .pbt-table .running-row td { color: var(--cyan); }
  .pbt-table .pending-row td { opacity: 0.4; }
  .pbt-role {
    font-size: 9px; padding: 1px 5px; border-radius: 3px;
    display: inline-block; font-weight: 600;
  }
  .role-baseline { background: rgba(139,148,158,0.2); color: var(--text-dim); }
  .role-elite { background: rgba(210,169,34,0.2); color: var(--yellow); }
  .role-exploit { background: rgba(63,185,80,0.2); color: var(--green); }
  .role-explore { background: rgba(188,140,255,0.2); color: var(--purple); }
  .role-perturbed { background: rgba(88,166,255,0.15); color: var(--cyan); }
  .role-random { background: rgba(248,81,73,0.15); color: var(--red); }
  .pbt-table td.num, .pbt-table th { white-space: nowrap; }
  .pbt-table .cfg-cell { font-size: 10px; color: var(--text-dim); max-width: 140px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .ts-cell { font-size: 10px; color: var(--text-dim); white-space: nowrap; }

  /* Anomaly flags in experiment table */
  .anomaly-dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block; margin-left: 4px;
  }
  .anomaly-critical { background: var(--red); }
  .anomaly-warn { background: var(--orange); }
  .anomaly-tooltip {
    position: relative;
    cursor: help;
  }
  .anomaly-tooltip:hover::after {
    content: attr(data-tip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg3);
    border: 1px solid var(--border);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
    white-space: nowrap;
    z-index: 10;
    color: var(--text);
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
    <h1>ART² TRAINING MONITOR</h1>
    <span id="training-mode-badge" class="training-mode-badge"></span>
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
  <!-- Row 1: GPU (compact) + Active Run -->
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
    <div class="panel active-run">
      <div class="panel-header">Active Run <span id="run-name" style="font-weight:400;color:var(--text)"></span><span id="run-state-badge" class="run-state-badge"></span></div>
      <div class="panel-body" style="padding:6px 10px">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
          <span class="phase-dot" id="phase-dot"></span>
          <span id="phase-label" style="font-weight:600;font-size:12px"></span>
          <span id="exp-id" style="color:var(--text-dim);margin-left:auto;font-size:11px"></span>
        </div>
        <div class="progress-outer">
          <div class="progress-inner" id="time-progress" style="width:0%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;color:var(--text-dim);font-size:10px;margin-bottom:6px;">
          <span id="time-elapsed"></span>
          <span id="time-remaining"></span>
        </div>
        <div class="stats-grid">
          <div class="stat-card" title="Experiments that beat the previous best score and were promoted"><div class="stat-value stat-green" id="stat-kept">--</div><div class="stat-label">Kept</div></div>
          <div class="stat-card" title="Experiments that crashed or had errors during training"><div class="stat-value stat-red" id="stat-failed">--</div><div class="stat-label">Failed</div></div>
          <div class="stat-card" title="Total experiments run this deployment"><div class="stat-value stat-cyan" id="stat-total">--</div><div class="stat-label">Total</div></div>
          <div class="stat-card" title="Composite score = PF × Sharpe × frequency multiplier × penalties. Higher = better trading. Negative = losing money."><div class="stat-value stat-green" id="stat-best">--</div><div class="stat-label">Best Score</div></div>
          <div class="stat-card" title="How fast experiments are completing"><div class="stat-value stat-yellow" id="stat-rate">--</div><div class="stat-label">Exp/hr</div></div>
          <div class="stat-card" title="Hash of program.md — ensures the training contract hasn't drifted"><div class="stat-value" id="stat-contract" style="font-size:10px;color:var(--text-dim)">--</div><div class="stat-label">Contract</div></div>
        </div>
        <div class="stats-grid" style="margin-top:3px;">
          <div class="stat-card"><div class="stat-value stat-yellow" id="stat-cost" style="font-size:13px">--</div><div class="stat-label">API Cost</div></div>
          <div class="stat-card"><div class="stat-value stat-cyan" id="stat-cache-hit" style="font-size:13px">--</div><div class="stat-label">Cache Hit %</div></div>
          <div class="stat-card"><div class="stat-value" id="stat-cost-per-exp" style="font-size:13px;color:var(--text-dim)">--</div><div class="stat-label">$/Experiment</div></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 2: PBT Population (shown only in PBT mode) -->
  <div class="panel full-row pbt-panel" id="pbt-panel" style="display:none">
    <div class="panel-header">
      PBT Sweep
      <span id="pbt-header-info" style="font-weight:400;color:var(--text-dim);font-size:9px"></span>
    </div>
    <div class="panel-body" style="padding:0">
      <div id="pbt-status-bar" style="padding:5px 10px;font-size:10px;color:var(--text-dim);border-bottom:1px solid var(--border);display:none"></div>
      <div style="overflow:auto;max-height:180px;" id="pbt-gens-container"></div>
    </div>
  </div>

  <!-- Row 3: All 8 charts in one panel, 2 rows of 4 -->
  <div class="panel full-row charts-panel">
    <div class="panel-header">Charts</div>
    <div class="panel-body" style="padding:6px 10px">
      <div class="chart-rows">
        <div class="chart-row-label">Performance</div>
        <div class="chart-container">
          <div class="chart-box"><div class="chart-title">Score</div><canvas id="chart-score"></canvas></div>
          <div class="chart-box"><div class="chart-title">Profit Factor</div><canvas id="chart-pf"></canvas></div>
          <div class="chart-box"><div class="chart-title">Trades/Day</div><canvas id="chart-tpd"></canvas></div>
          <div class="chart-box"><div class="chart-title">Win Rate</div><canvas id="chart-wr"></canvas></div>
        </div>
        <div class="chart-row-label" style="margin-top:4px">Risk &amp; Quality</div>
        <div class="chart-container">
          <div class="chart-box"><div class="chart-title">Sharpe</div><canvas id="chart-sharpe"></canvas></div>
          <div class="chart-box"><div class="chart-title">Stop Loss %</div><canvas id="chart-sl"></canvas></div>
          <div class="chart-box"><div class="chart-title">R:R Ratio</div><canvas id="chart-rr"></canvas></div>
          <div class="chart-box"><div class="chart-title">EV / Trade</div><canvas id="chart-ev"></canvas></div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 4 Left: Experiments + Code/Log stacked -->
  <div class="left-stack">
    <div class="panel">
      <div class="panel-header"><span id="exp-panel-title">Experiments</span> <span id="exp-count" style="font-weight:400;color:var(--text-dim)"></span></div>
      <div class="panel-body" style="padding:0">
        <div style="overflow:auto;max-height:240px;">
          <table id="exp-table">
            <thead>
              <tr id="exp-thead-row">
                <th>#</th><th>Result</th><th>Score</th><th>PF</th><th>TPD</th>
                <th>Sharpe</th><th>WR</th><th>SL%</th><th>Hold</th><th>Flags</th><th>Time</th><th>When (PT)</th>
              </tr>
            </thead>
            <tbody id="exp-tbody"></tbody>
          </table>
        </div>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">
        <div class="tab-bar" style="border-bottom:none">
          <div class="tab active" data-tab="code" id="code-tab-label">train.py</div>
          <div class="tab" data-tab="log">Log</div>
          <div class="tab" data-tab="runs">Runs</div>
        </div>
      </div>
      <div class="panel-body" style="padding:0">
        <div class="tab-content active" id="tab-code">
          <div class="code-view" id="code-content" style="max-height:200px">Loading...</div>
        </div>
        <div class="tab-content" id="tab-log">
          <div class="log-view" id="log-content" style="max-height:200px">Waiting for log data...</div>
        </div>
        <div class="tab-content" id="tab-runs">
          <div style="padding:8px;overflow:auto;max-height:200px;">
            <table id="runs-table">
              <thead><tr><th>Run</th><th>Phase</th><th>Exp</th><th>Kept</th><th>Best</th></tr></thead>
              <tbody id="runs-tbody"></tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Row 4 Right: Reasoning -->
  <div class="panel reasoning">
    <div class="panel-header">Stream of Consciousness</div>
    <div class="panel-body" id="reasoning-body" style="padding:6px 10px"></div>
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

function fmtTimestampPST(ts) {
  if (!ts) return '--';
  try {
    const d = new Date(ts);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleString('en-US', {
      timeZone: 'America/Los_Angeles',
      month: 'short', day: 'numeric',
      hour: 'numeric', minute: '2-digit',
      hour12: true
    });
  } catch(e) { return '--'; }
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

// -- Anomaly classification --
const CRITICAL_ANOMALIES = new Set([
  'metric_inconsistent', 'trades_per_day_extreme', 'cost_realism_low',
  'entry_quality_too_low', 'low_quality_rate_high', 'single_date_specialist'
]);

function classifyAnomaly(flag) {
  // Strip trailing suffixes like _high, _low for matching
  const base = flag.replace(/_(high|low|extreme)$/, '');
  return CRITICAL_ANOMALIES.has(base) ? 'critical' : 'warn';
}

function renderAnomalyFlags(flags) {
  if (!flags || !flags.length) return '';
  const dots = flags.map(f => {
    const cls = classifyAnomaly(f);
    return `<span class="anomaly-tooltip" data-tip="${escapeHtml(f)}"><span class="anomaly-dot anomaly-${cls}"></span></span>`;
  }).join('');
  return dots;
}

// -- PBT generation toggle --
function toggleGen(headerRow) {
  const gen = headerRow.dataset.gen;
  const arrow = headerRow.querySelector('.arrow');
  const table = headerRow.closest('table');
  const members = table.querySelectorAll(`tr[data-gen-member="${gen}"]`);
  const isCollapsed = arrow.classList.toggle('collapsed');
  for (const row of members) {
    row.style.display = isCollapsed ? 'none' : '';
  }
}

// -- Main update loop --
let lastExpCount = 0;
let lastTrainHash = '';
let lastPbtHash = '';

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

    // Training mode badge
    const modeBadge = document.getElementById('training-mode-badge');
    const tm = data.training_mode || 'unknown';
    const modeLabels = {opus: 'SEQUENTIAL', pbt: 'PBT SWEEP', unknown: 'UNKNOWN'};
    const modeTips = {
      opus: 'Sequential mode: AI proposes one experiment at a time, each warm-starting from the best model. Score must improve to keep.',
      pbt: 'Population-Based Training: evolutionary hyperparameter search. N members compete per generation, best survive and mutate.',
      unknown: 'Training mode not detected'
    };
    modeBadge.textContent = modeLabels[tm] || tm;
    modeBadge.title = modeTips[tm] || '';
    modeBadge.className = 'training-mode-badge training-mode-' + tm;

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
      const isLocal = !data.mode.startsWith('Remote');
      document.getElementById('gpu-name').textContent = isLocal ? '(local — no GPU)' : (data.ssh_ok ? 'waiting...' : 'no SSH');
      document.getElementById('gpu-content').style.opacity = '0.3';
    }

    // Active run
    const st = data.status || {};
    const runState = data.run_state || 'unknown';
    document.getElementById('run-name').textContent = st._run_name || data.active_run || 'none';
    const badge = document.getElementById('run-state-badge');
    const badgeLabels = {active: 'LIVE', completed: 'DONE', idle: 'IDLE', waiting: 'WAITING'};
    badge.textContent = badgeLabels[runState] || runState;
    badge.className = 'run-state-badge run-state-' + runState;

    const phase = st.phase || (runState === 'waiting' ? 'waiting' : 'unknown');
    const phaseDot = document.getElementById('phase-dot');
    phaseDot.className = 'phase-dot phase-' + phase;
    const phaseLabels = {
      training: 'Training Model', calling_claude: 'Generating Code',
      evaluating: 'Evaluating Results', saving: 'Saving Artifacts',
      completed: 'Completed', error: 'Error', startup: 'Starting Up',
      between_experiments: 'Between Experiments', smoke_check: 'Smoke Check',
      idle: 'Run Finished', waiting: 'Waiting for Deployment', unknown: 'No Data'
    };
    document.getElementById('phase-label').textContent = phaseLabels[phase] || phase;
    document.getElementById('exp-id').textContent = st.experiment_id ? `Experiment #${st.experiment_id}` : '';

    // Progress
    const timeLeft = st.time_remaining_h || 0;
    const totalH = 8.0;
    const elapsedH = totalH - timeLeft;
    const pct = Math.min(100, Math.max(0, elapsedH / totalH * 100));
    const progressBar = document.getElementById('time-progress');
    if (runState === 'completed' || runState === 'idle') {
      progressBar.style.width = '100%';
      progressBar.style.background = 'var(--text-dim)';
      document.getElementById('time-elapsed').textContent = 'run finished';
      document.getElementById('time-remaining').textContent = `${st.total || 0} experiments`;
    } else if (runState === 'waiting') {
      progressBar.style.width = '0%';
      progressBar.style.background = '';
      document.getElementById('time-elapsed').textContent = 'waiting for deployment...';
      document.getElementById('time-remaining').textContent = '';
    } else {
      progressBar.style.width = pct + '%';
      progressBar.style.background = '';
      document.getElementById('time-elapsed').textContent = pct.toFixed(0) + '% elapsed';
      document.getElementById('time-remaining').textContent = timeLeft > 0 ? timeLeft.toFixed(1) + 'h left' : 'done';
    }

    // Stats
    document.getElementById('stat-kept').textContent = st.kept || 0;
    document.getElementById('stat-failed').textContent = st.failed || 0;
    document.getElementById('stat-total').textContent = st.total || 0;
    document.getElementById('stat-best').textContent = (st.best_score != null && st.best_score > -999) ? st.best_score.toFixed(4) : '--';
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
    const scored = exps.filter(e => e.score != null && e.score > -999);
    if (scored.length >= 1) {
      drawChart('chart-score', scored.map(e => e.score || 0), '#3fb950', {decimals: 3});
      drawChart('chart-pf', scored.map(e => e.profit_factor || 0), '#58a6ff', {min: 0, decimals: 2});
      drawChart('chart-tpd', scored.map(e => e.trades_per_day || 0), '#d29922', {min: 0, decimals: 1});
      drawChart('chart-wr', scored.map(e => (e.win_rate || 0) * 100), '#bc8cff', {min: 0, max: 100, decimals: 0});
      // Row 2: Risk & Quality
      drawChart('chart-sharpe', scored.map(e => e.trade_sharpe || 0), '#f0883e', {decimals: 2});
      drawChart('chart-sl', scored.map(e => (e.stop_loss_rate || 0) * 100), '#f85149', {min: 0, max: 100, decimals: 0});
      drawChart('chart-rr', scored.map(e => e.rr_ratio || 0), '#a371f7', {min: 0, decimals: 2});
      drawChart('chart-ev', scored.map(e => e.ev_per_trade || 0), '#56d364', {decimals: 3});
    }

    // PBT Population Grid
    const pbtPanel = document.getElementById('pbt-panel');
    const isPbt = tm === 'pbt';
    pbtPanel.style.display = isPbt ? '' : 'none';

    if (isPbt && data.pbt) {
      const pbt = data.pbt;
      const pbtHash = JSON.stringify(pbt).length + '_' + (pbt.current_member || 0);
      if (pbtHash !== lastPbtHash) {
        lastPbtHash = pbtHash;

        // Compact header — just the focus area
        document.getElementById('pbt-header-info').textContent = `(${pbt.focus})`;

        // Human-readable status bar
        const totalMembers = pbt.max_generations * pbt.population_size;
        const completedGens = pbt.generations.filter(g => !g.in_progress).length;
        const inProgressDone = pbt.generations.filter(g => g.in_progress).reduce((s,g) => s + g.completed, 0);
        const totalDone = completedGens * pbt.population_size + inProgressDone;
        const remaining = totalMembers - totalDone;
        const budgetMin = (pbt.time_budget_per_member || 300) / 60;
        const etaMin = remaining * budgetMin;
        const etaStr = etaMin >= 60 ? `~${(etaMin/60).toFixed(1)}h` : `~${Math.round(etaMin)}m`;

        const bestPbt = pbt.best_pbt_score > 0 ? pbt.best_pbt_score.toFixed(2) : null;
        const base = pbt.base_score > 0 ? pbt.base_score.toFixed(2) : null;
        const beating = pbt.best_pbt_score > pbt.base_score;

        // Line 1: What's happening + progress
        let line1 = `Generation ${pbt.generation + 1} of ${pbt.max_generations}`;
        line1 += ` &mdash; training member ${pbt.current_member + 1} of ${pbt.population_size}`;
        line1 += ` &nbsp;<span style="color:var(--text-dim)">(${totalDone}/${totalMembers} total, ${etaStr} remaining)</span>`;

        // Line 2: Is it working?
        let line2 = '';
        if (bestPbt && base) {
          if (beating) {
            line2 = `<span style="color:var(--green)">&#10003; Found improvement:</span> best PBT score ${bestPbt} beats baseline ${base}`;
          } else if (pbt.stagnation_count > 0) {
            line2 = `<span style="color:var(--yellow)">&#9888; No improvement yet</span> &mdash; ${pbt.stagnation_count} generation${pbt.stagnation_count > 1 ? 's' : ''} tested, none beat baseline score of ${base}`;
          } else {
            line2 = `Baseline score: ${base} &mdash; searching for better hyperparameters`;
          }
        }

        const statusBar = document.getElementById('pbt-status-bar');
        statusBar.style.display = '';
        statusBar.innerHTML = line1 + (line2 ? '<br>' + line2 : '');

        const container = document.getElementById('pbt-gens-container');
        const allScores = pbt.generations.flatMap(g =>
          g.members.filter(m => m.score != null).map(m => m.score));
        const maxScore = allScores.length ? Math.max(...allScores) : 1;
        const globalBest = maxScore;

        let html = '<table class="pbt-table">';
        html += '<colgroup><col class="col-gen"><col class="col-mem"><col class="col-role"><col class="col-score"><col class="col-pf"><col class="col-tpd"><col class="col-wr"><col class="col-sl"><col class="col-cfg"><col class="col-flags"></colgroup>';
        html += '<thead><tr>';
        html += '<th>Gen</th><th>#</th><th>Role</th><th>Score</th><th>PF</th><th>TPD</th><th>WR</th><th>SL%</th><th>Config</th><th>Flags</th>';
        html += '</tr></thead><tbody>';

        for (const gen of [...pbt.generations].reverse()) {
          // Generation separator row
          const statusTag = gen.in_progress ? ' (in progress)' : '';
          const genStats = gen.best_score != null
            ? `best ${gen.best_score.toFixed(2)} / avg ${gen.avg_score.toFixed(2)} / ${gen.completed}/${pbt.population_size} done`
            : `${gen.completed}/${pbt.population_size} done`;
          html += `<tr class="gen-row" data-gen="${gen.generation}" onclick="toggleGen(this)"><td colspan="10"><span class="arrow">&#9660;</span> Generation ${gen.generation}${statusTag} &mdash; ${genStats}</td></tr>`;

          for (const m of [...gen.members].reverse()) {
            const isRunning = gen.generation === pbt.generation &&
              m.member_id === pbt.current_member && m.score == null;
            const isBest = m.score != null && m.score === globalBest && allScores.length > 1;
            const isPending = m.score == null && !isRunning;
            const role = m.role || 'perturbed';

            let rowCls = '';
            if (isRunning) rowCls = 'running-row';
            else if (isPending) rowCls = 'pending-row';
            else if (isBest) rowCls = 'best-row';

            const scoreColor = m.score != null
              ? (m.score >= (gen.avg_score || 0) ? 'var(--green)' : 'var(--text-dim)')
              : 'var(--text-dim)';
            const scoreText = m.score != null ? m.score.toFixed(2) : (isRunning ? '...' : '--');
            const pfText = m.pf != null ? m.pf.toFixed(1) : '--';
            const tpdText = m.tpd != null ? m.tpd.toFixed(1) : '--';
            const wrText = m.wr != null ? (m.wr * 100).toFixed(0) + '%' : '--';
            const slText = m.stop_rate != null ? (m.stop_rate * 100).toFixed(0) + '%' : '--';
            const config = m.config_summary || 'defaults';
            const flags = (m.anomaly_flags || []).length
              ? m.anomaly_flags.map(f => `<span class="anomaly-tooltip" data-tip="${escapeHtml(f)}"><span class="anomaly-dot anomaly-${classifyAnomaly(f)}"></span></span>`).join('')
              : '';

            html += `<tr class="${rowCls}" data-gen-member="${gen.generation}">`;
            html += `<td class="num">${gen.generation}</td>`;
            html += `<td class="num">${m.member_id}</td>`;
            html += `<td><span class="pbt-role role-${role}">${role}</span></td>`;
            html += `<td class="num" style="font-weight:600;color:${scoreColor}">${scoreText}</td>`;
            html += `<td class="num">${pfText}</td>`;
            html += `<td class="num">${tpdText}</td>`;
            html += `<td class="num">${wrText}</td>`;
            html += `<td class="num">${slText}</td>`;
            html += `<td class="cfg-cell" title="${escapeHtml(config)}">${escapeHtml(config)}</td>`;
            html += `<td>${flags}</td>`;
            html += `</tr>`;
          }
        }
        html += '</tbody></table>';
        container.innerHTML = html;
      }
    }

    // Experiments table — adapt title and columns for mode
    document.getElementById('exp-panel-title').textContent =
      isPbt ? 'PBT Experiments' : (tm === 'opus' ? 'Sequential Experiments' : 'Experiments');
    const theadRow = document.getElementById('exp-thead-row');
    if (isPbt && !theadRow.dataset.pbt) {
      theadRow.dataset.pbt = '1';
      theadRow.innerHTML = '<th>#</th><th>Gen</th><th>Mem</th><th>Result</th><th>Score</th><th>PF</th><th>TPD</th><th>WR</th><th>SL%</th><th>Flags</th><th>Time</th><th>When (PT)</th>';
      lastExpCount = -1; // Force rebuild
    } else if (!isPbt && theadRow.dataset.pbt) {
      delete theadRow.dataset.pbt;
      theadRow.innerHTML = '<th>#</th><th>Result</th><th>Score</th><th>PF</th><th>TPD</th><th>Sharpe</th><th>WR</th><th>SL%</th><th>Hold</th><th>Flags</th><th>Time</th><th>When (PT)</th>';
      lastExpCount = -1;
    }

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
        else if (e.keep_block_reason && e.keep_block_reason.includes('secondary_regression')) {
          result = '&#9888; blocked'; cls = 'failed';
        } else if (e.keep_block_reason && e.keep_block_reason.includes('near_tie')) {
          result = '&#8776; near-tie'; cls = 'reverted';
        } else { result = '~ revert'; cls = 'reverted'; }
        tr.className = cls;

        const score = (e.score || -999) > -999 ? (e.score || 0).toFixed(3) : 'FAIL';
        const pf = e.profit_factor != null ? e.profit_factor.toFixed(2) : '--';
        const tpd = e.trades_per_day != null ? e.trades_per_day.toFixed(1) : '--';
        const wr = e.win_rate != null ? (e.win_rate * 100).toFixed(0) + '%' : '--';
        const sl = e.stop_loss_rate != null ? (e.stop_loss_rate * 100).toFixed(0) + '%' : '--';
        const trainTime = e.wall_time ? fmtDuration(e.wall_time) : '--';
        const whenPST = fmtTimestampPST(e.timestamp);
        const flags = renderAnomalyFlags(e.anomaly_flags);

        const blockTip = e.keep_block_reason ? ` title="${escapeHtml(e.keep_block_reason)}"` : '';

        if (isPbt) {
          const gen = e.pbt_generation != null ? e.pbt_generation : '--';
          const mem = e.pbt_member != null ? e.pbt_member : '--';
          tr.innerHTML = `
            <td class="num">${e.experiment_id || '?'}</td>
            <td class="num">${gen}</td>
            <td class="num">${mem}</td>
            <td${blockTip}>${result}</td>
            <td class="num">${score}</td>
            <td class="num">${pf}</td>
            <td class="num">${tpd}</td>
            <td class="num">${wr}</td>
            <td class="num">${sl}</td>
            <td>${flags}</td>
            <td class="num">${trainTime}</td>
            <td class="num ts-cell">${whenPST}</td>
          `;
        } else {
          const sharpe = e.trade_sharpe != null ? e.trade_sharpe.toFixed(2) : '--';
          const hold = e.avg_hold_bars != null ? e.avg_hold_bars.toFixed(0) + 'b' : '--';
          tr.innerHTML = `
            <td class="num">${e.experiment_id || '?'}</td>
            <td${blockTip}>${result}</td>
            <td class="num">${score}</td>
            <td class="num">${pf}</td>
            <td class="num">${tpd}</td>
            <td class="num">${sharpe}</td>
            <td class="num">${wr}</td>
            <td class="num">${sl}</td>
            <td class="num">${hold}</td>
            <td>${flags}</td>
            <td class="num">${trainTime}</td>
            <td class="num ts-cell">${whenPST}</td>
          `;
        }
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
          ${change && change !== reasoning ? `<div class="reasoning-change">${escapeHtml(change)}</div>` : ''}
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
      const src = data.train_py_source || '';
      document.getElementById('code-tab-label').textContent = src ? src : 'train.py';
    } else if (data.poll_count > 2) {
      document.getElementById('code-content').textContent = data.ssh_ok
        ? 'Waiting for best_train.py on remote...'
        : 'No SSH — no local best_train.py found';
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
    host_pinned = args.host is not None  # True only when user explicitly passes --host
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

    server.timeout = 0.5  # handle_request returns promptly so we can check restart flag
    try:
        while not _restart_flag.is_set():
            server.handle_request()
        # Source changed — restart from main thread (safe, unlike os.execv from daemon)
        print("[monitor] Restarting...", flush=True)
        server.server_close()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
