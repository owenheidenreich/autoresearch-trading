#!/usr/bin/env python3
"""Mechanical inner loop — thin layer for SSH, training, scoring, and keep/revert.

All strategic decisions (what to try, why, when to stop) live in Opus.
This script handles the plumbing:
  init       — create run directory, write initial status.json
  experiment — atomic: validate → upload → train → download → score → keep/revert
  status     — show current run state

Invariant: best_model.pt, best_train.py, and .best_score are always in sync.
  - Model is downloaded to a temp file first, only promoted to best_model.pt on KEEP
  - On REVERT, best_model.pt is unchanged (still matches best_train.py)
  - best_model.pt is uploaded to Akash before each experiment for warm start

Usage (called by Claude Code):
  python3 tools/inner_loop.py init
  python3 tools/inner_loop.py experiment --mutation /tmp/mutation.py --summary "hypothesis"
  python3 tools/inner_loop.py experiment --summary "baseline test"  # no mutation, uses current train.py
  python3 tools/inner_loop.py status
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAIN_PY = TRAINING_DIR / "train.py"
BEST_TRAIN_PY = TRAINING_DIR / "best_train.py"
BEST_MODEL_PT = TRAINING_DIR / "best_model.pt"
BEST_SCORE_FILE = TRAINING_DIR / ".best_score"
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
STATE_FILE = TRAINING_DIR / ".inner_loop_state.json"

sys.path.insert(0, str(TRAINING_DIR))

# Logging to both stdout and loop.log
_log = logging.getLogger("inner_loop")
_log.setLevel(logging.INFO)
_fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(_fmt)
_log.addHandler(_sh)


def _setup_file_logging(run_dir: Path):
    """Add file handler for loop.log in the run directory."""
    log_path = run_dir / "loop.log"
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(_fmt)
    _log.addHandler(fh)


def _sha256_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ---------------------------------------------------------------------------
# SSH/SCP helpers
# ---------------------------------------------------------------------------

def _load_deploy_state() -> dict:
    if not DEPLOY_STATE.exists():
        print("FATAL: No .deploy-state found. Deploy to Akash first.", file=sys.stderr)
        sys.exit(1)
    state = {}
    for line in DEPLOY_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            state[k.strip()] = v.strip()
    missing = [k for k in ["SSH_HOST", "SSH_PORT"] if k not in state]
    if missing:
        print(f"FATAL: .deploy-state missing: {missing}", file=sys.stderr)
        sys.exit(1)
    return state


def _sshpass_env() -> dict:
    """Return env dict with SSHPASS set for sshpass -e mode."""
    env = os.environ.copy()
    env["SSHPASS"] = os.environ.get("SSH_PASS", "autoresearch2026")
    return env

_SSH_COMMON = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
    "-o", "PubkeyAuthentication=no",
]


def _ssh_cmd(deploy: dict, cmd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sshpass", "-e", "ssh"] + _SSH_COMMON + [
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-p", deploy["SSH_PORT"],
            f"root@{deploy['SSH_HOST']}",
            cmd,
        ],
        capture_output=True, text=True, timeout=timeout,
        env=_sshpass_env(),
    )


def _scp_upload(deploy: dict, local: str, remote: str, timeout: int = 120, retries: int = 3):
    """Upload file to Akash via SCP with retry."""
    for attempt in range(retries):
        result = subprocess.run(
            ["sshpass", "-e", "scp"] + _SSH_COMMON + [
                "-P", deploy["SSH_PORT"],
                local, f"root@{deploy['SSH_HOST']}:{remote}"],
            capture_output=True, text=True, timeout=timeout,
            env=_sshpass_env(),
        )
        if result.returncode == 0:
            return
        if attempt < retries - 1:
            _log.info(f"SCP upload attempt {attempt+1} failed, retrying in 3s...")
            time.sleep(3)
    raise RuntimeError(f"SCP upload failed after {retries} attempts: {result.stderr}")


def _scp_download(deploy: dict, remote: str, local: str, timeout: int = 120, retries: int = 3):
    """Download file from Akash via SCP with retry."""
    for attempt in range(retries):
        result = subprocess.run(
            ["sshpass", "-e", "scp"] + _SSH_COMMON + [
                "-P", deploy["SSH_PORT"],
                f"root@{deploy['SSH_HOST']}:{remote}", local],
            capture_output=True, text=True, timeout=timeout,
            env=_sshpass_env(),
        )
        if result.returncode == 0:
            return
        if attempt < retries - 1:
            _log.info(f"SCP download attempt {attempt+1} failed, retrying in 3s...")
            time.sleep(3)
    raise RuntimeError(f"SCP download failed after {retries} attempts: {result.stderr}")


# ---------------------------------------------------------------------------
# Monitor.py-compatible status writing
# ---------------------------------------------------------------------------

def _write_status_json(run_dir: Path, state: dict, phase: str = "idle",
                       last_change: str = "", last_score: float | None = None,
                       last_kept: bool | None = None, time_remaining_h: float = 0.0):
    """Write status.json compatible with monitor.py."""
    total = state.get("total_count", 0)
    kept = state.get("kept_count", 0)
    status = {
        "phase": phase,
        "experiment_id": state.get("experiment_id", 0),
        "best_score": state.get("best_score", -5.0),
        "kept": kept,
        "failed": total - kept,
        "total": total,
        "accept_rate": kept / total if total > 0 else 0.0,
        "time_remaining_h": time_remaining_h,
        "updated": datetime.now(timezone.utc).isoformat(),
        "last_change": last_change,
        "last_score": last_score,
        "last_kept": last_kept,
        "run_name": state.get("run_name", ""),
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2, default=str))


def _append_experiment_jsonl(run_dir: Path, record: dict):
    """Append to experiments.v2.jsonl (monitor.py compatible)."""
    path = run_dir / "experiments.v2.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Initialize a new run."""
    run_name = f"run-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S')}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    _setup_file_logging(run_dir)

    # Write current_run.txt pointer
    (RESULTS_DIR / "current_run.txt").write_text(run_name + "\n")

    # Backup current train.py
    shutil.copy2(TRAIN_PY, run_dir / "train_baseline.py")

    # Read best_score from .best_score file if it exists
    best_score = -5.0
    if BEST_SCORE_FILE.exists():
        try:
            best_score = float(BEST_SCORE_FILE.read_text().strip())
        except (ValueError, OSError):
            pass

    state = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "experiment_id": 0,
        "best_score": best_score,
        "kept_count": 0,
        "total_count": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_state(state)

    # Write initial status.json
    _write_status_json(run_dir, state, phase="idle")

    # Clean up stale .best_snapshot (no longer used)
    snapshot = Path(str(BEST_MODEL_PT) + ".best_snapshot")
    if snapshot.exists():
        snapshot.unlink()
        _log.info("Removed stale .best_snapshot")

    _log.info(f"Initialized run: {run_name} (best_score={best_score})")
    print(json.dumps({"status": "initialized", "run_name": run_name,
                       "run_dir": str(run_dir), "best_score": best_score}))


def cmd_experiment(args):
    """Run a single atomic experiment: validate → upload → train → score → keep/revert."""
    import run_loop

    state = _load_state()
    if not state:
        print(json.dumps({"error": "No active run. Call 'init' first."}))
        sys.exit(1)

    run_dir = Path(state["run_dir"])
    _setup_file_logging(run_dir)

    time_budget = args.time_budget
    change_summary = args.summary or ""
    exp_id = state["experiment_id"] + 1

    # --mutation is optional; omit for baseline experiment using current train.py
    if args.mutation is None:
        mutation_path = str(TRAIN_PY)
        is_baseline = True
    else:
        mutation_path = args.mutation
        is_baseline = False  # will be refined below

    _log.info(f"=== Experiment #{exp_id} starting ===")

    # Create artifact directory
    artifact_dir = run_dir / f"artifacts/exp-{exp_id}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Update status
    _write_status_json(run_dir, state, phase="validating")

    # --- Step 1: Read and validate mutation ---
    if not os.path.exists(mutation_path):
        result = _fail_experiment(state, run_dir, exp_id, "syntax",
                                  f"Mutation file not found: {mutation_path}", change_summary)
        print(json.dumps(result, indent=2))
        return

    new_code = Path(mutation_path).read_text()
    if not is_baseline:
        is_baseline = (Path(mutation_path).resolve() == TRAIN_PY.resolve())

    # Syntax check
    syntax_err = run_loop.validate_syntax(new_code)
    if syntax_err:
        result = _fail_experiment(state, run_dir, exp_id, "syntax", syntax_err, change_summary)
        print(json.dumps(result, indent=2))
        return

    # Safety check
    safety_err = run_loop.validate_safety(new_code)
    if safety_err:
        result = _fail_experiment(state, run_dir, exp_id, "safety", safety_err, change_summary)
        print(json.dumps(result, indent=2))
        return

    _log.info("Validation passed")

    # --- Step 2: Backup and apply mutation ---
    before_hash = hashlib.sha256(TRAIN_PY.read_text().encode()).hexdigest()[:16]
    shutil.copy2(TRAIN_PY, artifact_dir / "train_before.py")

    if not is_baseline:
        TRAIN_PY.write_text(new_code)
        (artifact_dir / "train_candidate.py").write_text(new_code)
        _log.info("Mutation applied to train.py")

    after_hash = hashlib.sha256(TRAIN_PY.read_text().encode()).hexdigest()[:16]

    # --- Step 3: Upload and train on Akash ---
    _write_status_json(run_dir, state, phase="training")

    deploy = _load_deploy_state()

    _log.info("Uploading train.py to Akash...")
    try:
        _scp_upload(deploy, str(TRAIN_PY), "/root/autoresearch-trading/training/train.py")
    except RuntimeError as e:
        _revert_train_py(artifact_dir, is_baseline)
        result = _fail_experiment(state, run_dir, exp_id, "train_crash",
                                  f"Upload failed: {e}", change_summary)
        print(json.dumps(result, indent=2))
        return

    # Upload best_model.pt for warm start (or ensure fresh start)
    if BEST_MODEL_PT.exists():
        _log.info("Uploading best_model.pt for warm start...")
        try:
            _scp_upload(deploy, str(BEST_MODEL_PT), "/root/autoresearch-trading/training/best_model.pt")
        except RuntimeError as e:
            _log.warning(f"Failed to upload best_model.pt (will train from scratch): {e}")
    else:
        _log.info("No local best_model.pt — ensuring fresh start on Akash")
        _ssh_cmd(deploy, "rm -f /root/autoresearch-trading/training/best_model.pt")

    _log.info(f"Training on Akash ({time_budget}s budget)...")
    train_cmd = (
        f"cd /root/autoresearch-trading && "
        f"TIME_BUDGET={time_budget} "
        f"/opt/conda/bin/python -u training/train.py 2>&1"
    )

    timeout = time_budget + 240
    t_start = time.time()
    try:
        ssh_result = _ssh_cmd(deploy, train_cmd, timeout=timeout)
        wall_time = time.time() - t_start
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t_start
        _revert_train_py(artifact_dir, is_baseline)
        result = _fail_experiment(state, run_dir, exp_id, "timeout",
                                  f"Training timed out after {timeout}s", change_summary)
        print(json.dumps(result, indent=2))
        return

    output = ssh_result.stdout + ssh_result.stderr
    (artifact_dir / "train_output.log").write_text(output)

    if ssh_result.returncode != 0:
        tail = "\n".join(output.strip().split("\n")[-50:])
        _revert_train_py(artifact_dir, is_baseline)
        result = _fail_experiment(state, run_dir, exp_id, "train_crash",
                                  f"Exit code {ssh_result.returncode}:\n{tail}", change_summary)
        print(json.dumps(result, indent=2))
        return

    _log.info(f"Training complete in {wall_time:.0f}s")

    # --- Step 4: Parse metrics ---
    _write_status_json(run_dir, state, phase="scoring")
    metrics = run_loop._parse_training_output(output)

    if "error" in metrics:
        _revert_train_py(artifact_dir, is_baseline)
        result = _fail_experiment(state, run_dir, exp_id, metrics.get("error_type", "parse"),
                                  metrics["error"], change_summary)
        print(json.dumps(result, indent=2))
        return

    # Save metrics
    (artifact_dir / "metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "output"}, indent=2, default=str))

    score = metrics.get("score", -999)
    _log.info(f"Score: {score:.4f} (best: {state['best_score']:.4f})")

    # --- Step 5: Download model to temp (only promoted on KEEP) ---
    _log.info("Downloading model candidate...")
    tmp_model = artifact_dir / "model_candidate.pt"
    try:
        _scp_download(deploy, "/root/autoresearch-trading/training/best_model.pt", str(tmp_model))
    except RuntimeError as e:
        _revert_train_py(artifact_dir, is_baseline)
        result = _fail_experiment(state, run_dir, exp_id, "train_crash",
                                  f"Model download failed: {e}", change_summary)
        print(json.dumps(result, indent=2))
        return

    # --- Step 6: Anomaly detection + keep/revert decision ---
    anomaly_flags = run_loop.detect_anomaly_flags(metrics)

    # Single-date specialist detection: model trades on ≤1 date with 3+ trades
    num_trade_dates = metrics.get("num_trade_dates", -1)
    num_trades = metrics.get("num_trades", 0)
    if num_trade_dates != -1 and num_trade_dates <= 1 and num_trades >= 3:
        anomaly_flags.append("single_date_specialist")
        _log.warning(f"ANOMALY: single_date_specialist — {num_trades} trades on {num_trade_dates} date(s)")

    critical_flags = run_loop._critical_anomaly_flags(anomaly_flags)
    best_score = state["best_score"]

    keep = score > best_score and not critical_flags
    reasons = []
    if score <= best_score:
        reasons.append("score_not_improved")
    if critical_flags:
        reasons.append(f"critical_anomalies:{','.join(critical_flags)}")

    state["experiment_id"] = exp_id
    state["total_count"] += 1

    if keep:
        # --- KEEP: promote model + code together (atomic) ---
        _log.info(f"KEPT: score {best_score:.4f} → {score:.4f}")
        state["best_score"] = score
        state["kept_count"] += 1

        shutil.copy2(tmp_model, BEST_MODEL_PT)    # Promote model
        shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)     # Promote code
        BEST_SCORE_FILE.write_text(str(score))     # Update score

        # Record promotion
        run_loop.record_promotion_event({
            "experiment_id": exp_id,
            "run_name": state["run_name"],
            "score": score,
            "profit_factor": metrics.get("profit_factor"),
            "trades_per_day": metrics.get("trades_per_day"),
            "trade_sharpe": metrics.get("trade_sharpe"),
            "stop_loss_rate": metrics.get("stop_loss_rate"),
            "worst_chunk_pf": metrics.get("worst_chunk_pf"),
            "direction_collapse_pct": metrics.get("direction_collapse_pct"),
            "change_summary": change_summary,
        })
    else:
        # --- REVERT: restore code, best_model.pt unchanged (still matches best_train.py) ---
        _log.info(f"REVERTED: score={score:.4f}, reasons: {reasons}")
        _revert_train_py(artifact_dir, is_baseline)

    # --- Loop detection: track consecutive identical revert reasons ---
    revert_key = ",".join(sorted(reasons)) if reasons else ("kept" if keep else "unknown")
    if revert_key == state.get("_last_revert_key", ""):
        state["_consecutive_same_revert"] = state.get("_consecutive_same_revert", 0) + 1
    else:
        state["_consecutive_same_revert"] = 0
    state["_last_revert_key"] = revert_key
    if state["_consecutive_same_revert"] >= 3:
        _log.error(f"LOOP DETECTED: Same revert reason '{revert_key}' 3+ times in a row. "
                   f"Investigate before continuing.")

    _save_state(state)

    # --- Step 7: Write experiment record ---
    exp_record = {
        "schema_version": 2,
        "id": exp_id,
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kept": keep,
        "score": score,
        "best_score": state["best_score"],
        "profit_factor": metrics.get("profit_factor"),
        "trades_per_day": metrics.get("trades_per_day"),
        "trade_sharpe": metrics.get("trade_sharpe"),
        "win_rate": metrics.get("win_rate"),
        "num_trades": metrics.get("num_trades"),
        "stop_loss_rate": metrics.get("stop_loss_rate"),
        "worst_chunk_pf": metrics.get("worst_chunk_pf"),
        "direction_collapse_pct": metrics.get("direction_collapse_pct"),
        "model_exit_rate": metrics.get("model_exit_rate"),
        "avg_hold_bars": metrics.get("avg_hold_bars"),
        "rr_ratio": metrics.get("rr_ratio"),
        "ev_per_trade": metrics.get("ev_per_trade"),
        "hit_ruin": metrics.get("hit_ruin"),
        "anomaly_flags": anomaly_flags,
        "failure_type": "none",
        "train_py_before_hash": before_hash,
        "train_py_after_hash": after_hash,
        "change_summary": change_summary,
        "reasoning": change_summary,
        "wall_time": wall_time,
        "num_steps": metrics.get("num_steps"),
        "training_seconds": metrics.get("training_seconds"),
        "chunk_details": metrics.get("chunk_details", []),
        "trade_diagnostics": metrics.get("trade_diagnostics", ""),
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("output", "trade_diagnostics", "chunk_details",
                                 "parse_summary")},
    }

    _append_experiment_jsonl(run_dir, exp_record)

    # Update status.json
    _write_status_json(run_dir, state, phase="idle",
                       last_change=change_summary, last_score=score, last_kept=keep)

    # Also write history.jsonl for Opus to read
    history_path = run_dir / "history.jsonl"
    with open(history_path, "a") as f:
        f.write(json.dumps(exp_record, default=str) + "\n")

    # --- Output result ---
    result = {
        "status": "kept" if keep else "reverted",
        "experiment_id": exp_id,
        "kept": keep,
        "score": score,
        "best_score": state["best_score"],
        "improvement": score - best_score if keep else None,
        "reasons": reasons if not keep else [],
        "anomaly_flags": anomaly_flags,
        "critical_flags": critical_flags,
        "wall_time": round(wall_time, 1),
        "num_steps": metrics.get("num_steps"),
        "metrics": {
            "profit_factor": metrics.get("profit_factor"),
            "trades_per_day": metrics.get("trades_per_day"),
            "win_rate": metrics.get("win_rate"),
            "stop_loss_rate": metrics.get("stop_loss_rate"),
            "worst_chunk_pf": metrics.get("worst_chunk_pf"),
            "direction_collapse_pct": metrics.get("direction_collapse_pct"),
            "model_exit_rate": metrics.get("model_exit_rate"),
            "num_trades": metrics.get("num_trades"),
            "avg_hold_bars": metrics.get("avg_hold_bars"),
            "rr_ratio": metrics.get("rr_ratio"),
        },
        "trade_diagnostics": metrics.get("trade_diagnostics", ""),
        "chunk_details": metrics.get("chunk_details", []),
    }
    print(json.dumps(result, indent=2))


def _fail_experiment(state: dict, run_dir: Path, exp_id: int,
                     failure_type: str, error: str, change_summary: str) -> dict:
    """Record a failed experiment and return result dict."""
    _log.info(f"FAILED: {failure_type}: {error[:200]}")

    state["experiment_id"] = exp_id
    state["total_count"] += 1
    _save_state(state)

    record = {
        "schema_version": 2,
        "id": exp_id,
        "experiment_id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kept": False,
        "score": None,
        "failure_type": failure_type,
        "error": error[:500],
        "anomaly_flags": [],
        "train_py_before_hash": "",
        "train_py_after_hash": "",
        "change_summary": change_summary,
    }
    _append_experiment_jsonl(run_dir, record)
    _write_status_json(run_dir, state, phase="idle",
                       last_change=f"FAILED: {failure_type}", last_kept=False)

    return {
        "status": "failed",
        "experiment_id": exp_id,
        "kept": False,
        "failure_type": failure_type,
        "error": error,
    }


def _revert_train_py(artifact_dir: Path, is_baseline: bool):
    """Restore train.py from backup."""
    if is_baseline:
        return  # baseline experiments don't modify train.py
    backup = artifact_dir / "train_before.py"
    if backup.exists():
        shutil.copy2(backup, TRAIN_PY)
        _log.info("Reverted train.py from backup")


def cmd_status(args):
    """Show current run state."""
    state = _load_state()
    if not state:
        print(json.dumps({"status": "no_active_run"}))
        return

    print(json.dumps({
        "run_name": state.get("run_name"),
        "experiment_id": state.get("experiment_id"),
        "best_score": state.get("best_score"),
        "kept_count": state.get("kept_count"),
        "total_count": state.get("total_count"),
        "started_at": state.get("started_at"),
    }, indent=2))


# ---------------------------------------------------------------------------
# Shared training helper (used by both cmd_experiment and PBT)
# ---------------------------------------------------------------------------

def _train_on_akash(deploy: dict, time_budget: int, env_overrides: dict | None = None,
                    upload_train_py: bool = True, upload_model: bool = True) -> tuple:
    """Run training on Akash. Returns (output, wall_time, returncode).

    Args:
        deploy: deploy state dict with SSH_HOST/SSH_PORT
        time_budget: seconds for TIME_BUDGET env var
        env_overrides: extra env vars to prepend (e.g., {"TRAIN_GATE_W": "0.6"})
        upload_train_py: whether to upload local train.py first
        upload_model: whether to upload best_model.pt for warm start
    """
    if upload_train_py:
        _scp_upload(deploy, str(TRAIN_PY), "/root/autoresearch-trading/training/train.py")

    if upload_model and BEST_MODEL_PT.exists():
        try:
            _scp_upload(deploy, str(BEST_MODEL_PT), "/root/autoresearch-trading/training/best_model.pt")
        except RuntimeError as e:
            _log.warning(f"Failed to upload best_model.pt (will train from scratch): {e}")
    elif upload_model:
        _ssh_cmd(deploy, "rm -f /root/autoresearch-trading/training/best_model.pt")

    env_str = " ".join(f"{k}={v}" for k, v in (env_overrides or {}).items())
    if env_str:
        env_str += " "

    train_cmd = (
        f"cd /root/autoresearch-trading && "
        f"{env_str}TIME_BUDGET={time_budget} "
        f"/opt/conda/bin/python -u training/train.py 2>&1"
    )

    timeout = time_budget + 240
    t_start = time.time()
    try:
        ssh_result = _ssh_cmd(deploy, train_cmd, timeout=timeout)
        wall_time = time.time() - t_start
    except subprocess.TimeoutExpired:
        wall_time = time.time() - t_start
        return ("Training timed out", wall_time, -1)

    output = ssh_result.stdout + ssh_result.stderr
    return (output, wall_time, ssh_result.returncode)


# ---------------------------------------------------------------------------
# PBT (Population-Based Training) — Phase A
# ---------------------------------------------------------------------------

import math
import random

PBT_STATE_FILE = TRAINING_DIR / ".pbt_state.json"

# Parameter space definitions — tiers match what train.py reads via _env_float/_env_int
_PARAM_SPACE = {
    # Tier 1: Loss Weights
    "TRAIN_GATE_W":              {"default": 0.5,   "lo": 0.1,  "hi": 5.0,  "scale": "log", "tier": 1},
    "TRAIN_DIR_W":               {"default": 2.5,   "lo": 0.1,  "hi": 5.0,  "scale": "log", "tier": 1},
    "TRAIN_PNL_W":               {"default": 0.5,   "lo": 0.01, "hi": 2.0,  "scale": "log", "tier": 1},
    "TRAIN_EXIT_W":              {"default": 0.15,  "lo": 0.01, "hi": 2.0,  "scale": "log", "tier": 1},
    "TRAIN_CONF_W":              {"default": 0.10,  "lo": 0.01, "hi": 1.0,  "scale": "log", "tier": 1},
    "TRAIN_FALSE_ENTRY_PENALTY": {"default": 1.2,   "lo": 1.0,  "hi": 5.0,  "scale": "linear", "tier": 1},
    # Tier 2: Optimizer
    "TRAIN_LR":                  {"default": 2.5e-4, "lo": 1e-5, "hi": 5e-3, "scale": "log", "tier": 2},
    "TRAIN_WEIGHT_DECAY":        {"default": 0.08,  "lo": 0.001, "hi": 0.3,  "scale": "log", "tier": 2},
    "TRAIN_DROPOUT":             {"default": 0.30,  "lo": 0.05, "hi": 0.40, "scale": "linear", "tier": 2},
    "TRAIN_WARMUP_RATIO":        {"default": 0.15,  "lo": 0.0,  "hi": 0.5,  "scale": "linear", "tier": 2},
    "TRAIN_COOLDOWN_RATIO":      {"default": 0.3,   "lo": 0.0,  "hi": 0.8,  "scale": "linear", "tier": 2},
    "TRAIN_GRAD_CLIP":           {"default": 1.0,   "lo": 0.1,  "hi": 5.0,  "scale": "log", "tier": 2},
    # Tier 3: Regularization & Sampling
    "WEIGHT_RECENT_BOOST":       {"default": 0.3,   "lo": 0.0,  "hi": 2.0,  "scale": "linear", "tier": 3},
    "WEIGHT_DAY_DIVERSITY":      {"default": 1.0,   "lo": 0.0,  "hi": 2.0,  "scale": "linear", "tier": 3},
    "REG_GATE_ENTROPY":          {"default": 0.10,  "lo": 0.0,  "hi": 1.0,  "scale": "linear", "tier": 3},
    "REG_TEMPORAL_SMOOTH":       {"default": 0.0,   "lo": 0.0,  "hi": 1.0,  "scale": "linear", "tier": 3},
    "WARM_FREEZE_RATIO":         {"default": 0.0,   "lo": 0.0,  "hi": 0.5,  "scale": "linear", "tier": 3},
    "TRAIN_DAY_SEQ_RATIO":       {"default": 0.85,  "lo": 0.5,  "hi": 1.0,  "scale": "linear", "tier": 3},
    "TRAIN_GATE_LABEL_SMOOTHING": {"default": 0.05, "lo": 0.0,  "hi": 0.2,  "scale": "linear", "tier": 3},
    "TRAIN_DIR_LABEL_SMOOTHING":  {"default": 0.05, "lo": 0.0,  "hi": 0.2,  "scale": "linear", "tier": 3},
    # Tier 1 (Phase C: RWR) — reward-weighted regression
    "TRAIN_RWR_WEIGHT":          {"default": 0.0,  "lo": 0.0,  "hi": 5.0,  "scale": "linear", "tier": 1},
    "TRAIN_DAY_RWR_WEIGHT":      {"default": 0.0,  "lo": 0.0,  "hi": 5.0,  "scale": "linear", "tier": 1},
    # Tier 1 (Phase D: Value Head) — exit intelligence
    "TRAIN_VALUE_W":             {"default": 0.3,  "lo": 0.0,  "hi": 2.0,  "scale": "log", "tier": 1},
    "TRAIN_VALUE_EXIT_THRESH":   {"default": 0.02, "lo": -0.5, "hi": 0.5,  "scale": "linear", "tier": 1},
}


def _build_parameter_space(focus: str) -> dict:
    """Return parameter space dict filtered by focus tier."""
    if focus == "loss_weights":
        return {k: v for k, v in _PARAM_SPACE.items() if v["tier"] == 1}
    elif focus == "regularization":
        return {k: v for k, v in _PARAM_SPACE.items() if v["tier"] in (2, 3)}
    else:  # "all"
        return dict(_PARAM_SPACE)


def _sample_param(spec: dict, rng: random.Random) -> float:
    """Sample a random value for a parameter according to its scale."""
    if spec["scale"] == "log":
        lo_log = math.log(max(spec["lo"], 1e-10))
        hi_log = math.log(spec["hi"])
        return math.exp(rng.uniform(lo_log, hi_log))
    else:
        return rng.uniform(spec["lo"], spec["hi"])


def _perturb_param(value: float, spec: dict, rng: random.Random, strength: str = "standard") -> float:
    """Mutate a parameter value."""
    if strength == "strong" and rng.random() < 0.10:
        return _sample_param(spec, rng)  # 10% chance full random reset

    if spec["scale"] == "log":
        sigma = 0.15 if strength == "standard" else 0.4
        value = value * math.exp(rng.gauss(0, sigma))
    else:
        rng_size = spec["hi"] - spec["lo"]
        sigma = rng_size * 0.05 if strength == "standard" else rng_size * 0.15
        value = value + rng.gauss(0, sigma)

    return max(spec["lo"], min(spec["hi"], value))


def _generate_initial_population(param_space: dict, n: int, rng: random.Random) -> list[dict]:
    """Create N member configs. Member 0 = exact defaults (baseline)."""
    members = []
    # Member 0: baseline defaults
    baseline = {k: spec["default"] for k, spec in param_space.items()}
    members.append(baseline)

    # Members 1..N-1: perturbed from defaults
    for _ in range(n - 1):
        config = {}
        for k, spec in param_space.items():
            if spec["scale"] == "log":
                factor = math.exp(rng.uniform(-0.7, 0.7))  # ~0.5x to 2x
                val = spec["default"] * factor
            else:
                rng_size = spec["hi"] - spec["lo"]
                val = spec["default"] + rng.uniform(-rng_size / 3, rng_size / 3)
            config[k] = max(spec["lo"], min(spec["hi"], val))
        members.append(config)

    return members


def _select_next_generation(results: list[dict], param_space: dict, n: int,
                            stagnation: int, rng: random.Random) -> list[dict]:
    """Evolutionary selection: elite + exploit + explore."""
    # Sort by score descending
    ranked = sorted(results, key=lambda r: r.get("score", -999), reverse=True)

    next_gen = []

    # Elite (member 0): copy best member unchanged
    next_gen.append(dict(ranked[0]["config"]))

    # Anti-stagnation: if no improvement for 2+ gens, inject random members
    inject_random = 2 if stagnation >= 2 else 0

    # Exploit (members 1 to N//2): clone from top-25%, standard mutation
    top_25 = max(1, len(ranked) // 4)
    half = n // 2
    for i in range(1, half):
        if inject_random > 0:
            # Inject fully random member
            config = {k: _sample_param(spec, rng) for k, spec in param_space.items()}
            inject_random -= 1
        else:
            parent = rng.choice(ranked[:top_25])["config"]
            config = {k: _perturb_param(v, param_space[k], rng, "standard")
                      for k, v in parent.items()}
        next_gen.append(config)

    # Explore (members N//2+1 to N-1): clone from top-50%, strong mutation
    top_50 = max(1, len(ranked) // 2)
    for i in range(half, n):
        if inject_random > 0:
            config = {k: _sample_param(spec, rng) for k, spec in param_space.items()}
            inject_random -= 1
        else:
            parent = rng.choice(ranked[:top_50])["config"]
            config = {k: _perturb_param(v, param_space[k], rng, "strong")
                      for k, v in parent.items()}
        next_gen.append(config)

    return next_gen[:n]


def _config_to_env(config: dict) -> dict:
    """Convert a PBT member config to env var overrides (string values)."""
    return {k: f"{v:.6g}" for k, v in config.items()}


def _config_summary(config: dict, param_space: dict) -> str:
    """Short summary of non-default params for experiment logging."""
    diffs = []
    for k, v in config.items():
        default = param_space[k]["default"]
        if abs(v - default) / max(abs(default), 1e-10) > 0.01:  # >1% different
            diffs.append(f"{k.replace('TRAIN_', '')}={v:.4g}")
    return ", ".join(diffs[:8]) or "(baseline defaults)"


def cmd_pbt_init(args):
    """Initialize a PBT population."""
    n = args.population
    gens = args.generations
    focus = args.focus
    time_budget = args.time_budget

    param_space = _build_parameter_space(focus)
    rng = random.Random()

    population = _generate_initial_population(param_space, n, rng)

    # Phase B: Add specialist members
    specialist_types = []  # parallel list: "" for generalists, "morning"/etc for specialists
    for _ in range(n):
        specialist_types.append("")

    if getattr(args, "specialists", False):
        _SPECIALIST_DEFS = [
            ("morning", "0-120"),
            ("midday", "120-240"),
            ("afternoon", "240-390"),
            ("highvol", "VIX>20 days"),
        ]
        for spec_name, spec_desc in _SPECIALIST_DEFS:
            # Each specialist starts from baseline config (member 0)
            spec_config = dict(population[0])
            population.append(spec_config)
            specialist_types.append(spec_name)
        _log.info(f"Specialists added: {[s for s in specialist_types if s]}")

    # Read current best score
    best_score = -5.0
    if BEST_SCORE_FILE.exists():
        try:
            best_score = float(BEST_SCORE_FILE.read_text().strip())
        except (ValueError, OSError):
            pass

    pbt_state = {
        "pbt_run_id": f"pbt-{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H%M%S')}",
        "generation": 0,
        "max_generations": gens,
        "population_size": len(population),
        "focus": focus,
        "time_budget_per_member": time_budget,
        "current_member": 0,
        "stagnation_count": 0,
        "base_score": best_score,
        "best_pbt_score": best_score,
        "best_pbt_config": None,
        "generations": [],
        "population": [{"id": i, "config": pop, "specialist_type": specialist_types[i]}
                       for i, pop in enumerate(population)],
        "parameter_space": {k: {kk: vv for kk, vv in v.items()} for k, v in param_space.items()},
    }

    PBT_STATE_FILE.write_text(json.dumps(pbt_state, indent=2, default=str))

    n_specs = sum(1 for s in specialist_types if s)
    _log.info(f"PBT initialized: {len(population)} members ({n} generalists + {n_specs} specialists), "
              f"{gens} generations, focus={focus}, time_budget={time_budget}s/member, base_score={best_score:.4f}")
    _log.info(f"Parameter space: {len(param_space)} params ({focus})")

    print(json.dumps({
        "status": "pbt_initialized",
        "pbt_run_id": pbt_state["pbt_run_id"],
        "population_size": len(population),
        "generalists": n,
        "specialists": [s for s in specialist_types if s],
        "generations": gens,
        "focus": focus,
        "params": list(param_space.keys()),
        "base_score": best_score,
    }, indent=2))


def cmd_pbt_run(args):
    """Run PBT: iterate through members and generations."""
    import run_loop

    if not PBT_STATE_FILE.exists():
        print(json.dumps({"error": "No PBT state. Run 'pbt-init' first."}))
        sys.exit(1)

    pbt = json.loads(PBT_STATE_FILE.read_text())
    param_space = pbt["parameter_space"]
    deploy = _load_deploy_state()
    rng = random.Random()

    # Ensure inner_loop state exists (for experiment recording)
    state = _load_state()
    if not state:
        print(json.dumps({"error": "No active run. Call 'init' first (for experiment logging)."}))
        sys.exit(1)

    run_dir = Path(state["run_dir"])
    _setup_file_logging(run_dir)

    time_budget = pbt["time_budget_per_member"]

    # Upload train.py + model once at start (all members use same code + weights)
    _log.info("PBT: Uploading train.py and best_model.pt to Akash...")
    _scp_upload(deploy, str(TRAIN_PY), "/root/autoresearch-trading/training/train.py")
    if BEST_MODEL_PT.exists():
        _scp_upload(deploy, str(BEST_MODEL_PT), "/root/autoresearch-trading/training/best_model.pt")

    gen = pbt["generation"]
    max_gens = pbt["max_generations"]

    while gen < max_gens:
        population = pbt["population"]
        gen_results = []
        start_member = pbt["current_member"]

        _log.info(f"=== PBT Generation {gen}/{max_gens}, members {start_member}-{len(population)-1} ===")

        for mi in range(start_member, len(population)):
            member = population[mi]
            config = member["config"]
            env_overrides = _config_to_env(config)
            spec_type = member.get("specialist_type", "")
            if spec_type:
                env_overrides["TRAIN_TOD_FILTER"] = spec_type
            summary = f"PBT gen={gen} member={mi}/{len(population)}"
            if spec_type:
                summary += f" [{spec_type}]"
            summary += f": {_config_summary(config, param_space)}"

            exp_id = state["experiment_id"] + 1
            _log.info(f"  Member {mi}: {_config_summary(config, param_space)}")

            # Train with env overrides (don't re-upload train.py/model each time)
            output, wall_time, returncode = _train_on_akash(
                deploy, time_budget, env_overrides=env_overrides,
                upload_train_py=False, upload_model=False,
            )

            # Parse metrics
            anomaly_flags = []
            if returncode != 0:
                _log.warning(f"  Member {mi}: CRASHED (exit {returncode})")
                score = -999.0
                metrics = {}
            else:
                metrics = run_loop._parse_training_output(output)
                if "error" in metrics:
                    _log.warning(f"  Member {mi}: PARSE ERROR: {metrics['error'][:100]}")
                    score = -999.0
                else:
                    score = metrics.get("score", -999)
                    anomaly_flags = run_loop.detect_anomaly_flags(metrics)
                    critical = run_loop._critical_anomaly_flags(anomaly_flags)
                    if critical:
                        # Log but DON'T disqualify during PBT ranking —
                        # anomalies are expected on fresh-start/short-budget training.
                        # Anomaly check applied at promotion time instead.
                        _log.warning(f"  Member {mi}: ANOMALY {critical}, score={score:.4f} (not disqualified for PBT ranking)")

            _log.info(f"  Member {mi}: score={score:.4f} (wall={wall_time:.0f}s)")

            gen_results.append({
                "member_id": mi,
                "config": config,
                "score": score,
                "wall_time": wall_time,
                "anomaly_flags": anomaly_flags,
                "metrics": {k: v for k, v in metrics.items()
                            if k not in ("output", "trade_diagnostics", "chunk_details", "parse_summary")}
                           if metrics else {},
            })

            # Record as experiment in jsonl (monitor.py compatible)
            state["experiment_id"] = exp_id
            state["total_count"] += 1

            exp_record = {
                "schema_version": 2,
                "id": exp_id,
                "experiment_id": exp_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kept": False,  # updated below if promoted
                "score": score if score > -999 else None,
                "best_score": state["best_score"],
                "profit_factor": metrics.get("profit_factor") if metrics else None,
                "trades_per_day": metrics.get("trades_per_day") if metrics else None,
                "trade_sharpe": metrics.get("trade_sharpe") if metrics else None,
                "win_rate": metrics.get("win_rate") if metrics else None,
                "num_trades": metrics.get("num_trades") if metrics else None,
                "stop_loss_rate": metrics.get("stop_loss_rate") if metrics else None,
                "anomaly_flags": [],
                "failure_type": "none" if returncode == 0 else "train_crash",
                "change_summary": summary,
                "reasoning": summary,
                "wall_time": wall_time,
                "pbt_generation": gen,
                "pbt_member": mi,
            }
            _append_experiment_jsonl(run_dir, exp_record)
            _save_state(state)

            # Save progress (resumable)
            pbt["current_member"] = mi + 1
            PBT_STATE_FILE.write_text(json.dumps(pbt, indent=2, default=str))

        # --- Generation complete: select + promote ---
        valid_results = [r for r in gen_results if r["score"] > -999]
        if not valid_results:
            _log.error(f"Generation {gen}: ALL members failed. Stopping PBT.")
            break

        # Find best generalist and best per-specialist
        generalist_results = [r for r in valid_results
                               if population[r["member_id"]].get("specialist_type", "") == ""]
        specialist_results_by_type = {}
        for r in valid_results:
            st = population[r["member_id"]].get("specialist_type", "")
            if st:
                specialist_results_by_type.setdefault(st, []).append(r)

        best_member = max(valid_results, key=lambda r: r["score"])
        best_generalist = max(generalist_results, key=lambda r: r["score"]) if generalist_results else None
        best_score_gen = best_member["score"]
        _log.info(f"Generation {gen} best: member {best_member['member_id']} "
                  f"score={best_score_gen:.4f} (baseline={pbt['base_score']:.4f})")

        # Log specialist results
        for st, results in specialist_results_by_type.items():
            best_spec = max(results, key=lambda r: r["score"])
            _log.info(f"  Best {st}: member {best_spec['member_id']} score={best_spec['score']:.4f}")

        # Promotion: generalist only → best_model.pt
        promoted = False
        if best_generalist and best_generalist["score"] > pbt["best_pbt_score"]:
            pbt["best_pbt_score"] = best_generalist["score"]
            pbt["best_pbt_config"] = best_generalist["config"]
            pbt["stagnation_count"] = 0

            best_critical = run_loop._critical_anomaly_flags(best_generalist.get("anomaly_flags", []))
            if best_generalist["score"] > state["best_score"] and not best_critical:
                _log.info(f"PROMOTING: PBT generalist score {best_generalist['score']:.4f} > best {state['best_score']:.4f}")

                # Re-train winner to get the model file
                if best_generalist["member_id"] != len(population) - 1:
                    _log.info("Re-training winner to get model weights...")
                    env_overrides = _config_to_env(best_generalist["config"])
                    output, _, rc = _train_on_akash(
                        deploy, time_budget, env_overrides=env_overrides,
                        upload_train_py=False, upload_model=False,
                    )

                tmp_model = run_dir / f"pbt_winner_gen{gen}.pt"
                _scp_download(deploy, "/root/autoresearch-trading/training/best_model.pt", str(tmp_model))
                shutil.copy2(tmp_model, BEST_MODEL_PT)
                shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)
                state["best_score"] = best_generalist["score"]
                state["kept_count"] += 1
                BEST_SCORE_FILE.write_text(str(best_generalist["score"]))
                _save_state(state)
                promoted = True
                _log.info(f"Promoted: best_model.pt updated, score={best_generalist['score']:.4f}")
            elif best_critical:
                _log.warning(f"Score improved ({best_generalist['score']:.4f}) but has critical anomalies — NOT promoting")
        else:
            pbt["stagnation_count"] += 1
            _log.info(f"No improvement (stagnation={pbt['stagnation_count']})")

        # Save specialist models separately
        for st, results in specialist_results_by_type.items():
            best_spec = max(results, key=lambda r: r["score"])
            if best_spec["score"] > -999:
                # Re-train specialist to get its model
                spec_env = _config_to_env(best_spec["config"])
                spec_env["TRAIN_TOD_FILTER"] = st
                if best_spec["member_id"] != len(population) - 1:
                    _log.info(f"Re-training {st} specialist for model save...")
                    _train_on_akash(deploy, time_budget, env_overrides=spec_env,
                                    upload_train_py=False, upload_model=False)
                spec_model = BEST_MODEL_PT.parent / f"best_model_{st}.pt"
                _scp_download(deploy, "/root/autoresearch-trading/training/best_model.pt", str(spec_model))
                _log.info(f"Saved {st} specialist: {spec_model} (score={best_spec['score']:.4f})")

        # Record generation
        pbt["generations"].append({
            "generation": gen,
            "members": gen_results,
            "best_member_id": best_member["member_id"],
            "best_score": best_score_gen,
            "promoted": promoted,
        })

        # --- Evolve next generation ---
        gen += 1
        if gen < max_gens:
            # Separate generalists and specialists for evolution
            old_pop = pbt["population"]
            generalist_results = [r for r in gen_results if old_pop[r["member_id"]].get("specialist_type", "") == ""]
            specialist_results = [r for r in gen_results if old_pop[r["member_id"]].get("specialist_type", "") != ""]
            n_generalists = len(generalist_results)

            # Evolve generalists normally
            next_gen_pop = _select_next_generation(
                generalist_results, param_space, n_generalists,
                pbt["stagnation_count"], rng,
            ) if n_generalists > 0 else []

            new_population = [{"id": i, "config": cfg, "specialist_type": ""}
                              for i, cfg in enumerate(next_gen_pop)]

            # Specialists: keep type, mutate hyperparams from best generalist
            for sr in specialist_results:
                spec_type = old_pop[sr["member_id"]]["specialist_type"]
                # If specialist scored well, keep its config; otherwise inherit from best generalist
                if sr["score"] > -999:
                    base = dict(sr["config"])
                else:
                    base = dict(next_gen_pop[0]) if next_gen_pop else dict(sr["config"])
                # Light perturbation
                for k, spec in param_space.items():
                    base[k] = _perturb_param(base[k], spec, rng, strength="standard")
                idx = len(new_population)
                new_population.append({"id": idx, "config": base, "specialist_type": spec_type})

            pbt["population"] = new_population
            pbt["generation"] = gen
            pbt["current_member"] = 0

            # Re-upload best_model.pt for next generation (all start from same checkpoint)
            if BEST_MODEL_PT.exists():
                _scp_upload(deploy, str(BEST_MODEL_PT), "/root/autoresearch-trading/training/best_model.pt")

        PBT_STATE_FILE.write_text(json.dumps(pbt, indent=2, default=str))

    # --- PBT complete ---
    pbt["generation"] = gen
    PBT_STATE_FILE.write_text(json.dumps(pbt, indent=2, default=str))

    _log.info(f"PBT complete: {gen} generations, best_score={pbt['best_pbt_score']:.4f}")
    if pbt["best_pbt_config"]:
        _log.info(f"Winning config: {_config_summary(pbt['best_pbt_config'], param_space)}")

    print(json.dumps({
        "status": "pbt_complete",
        "generations_run": gen,
        "best_score": pbt["best_pbt_score"],
        "base_score": pbt["base_score"],
        "improvement": pbt["best_pbt_score"] - pbt["base_score"],
        "best_config": pbt.get("best_pbt_config"),
        "promoted": pbt["best_pbt_score"] > pbt["base_score"],
    }, indent=2))


def cmd_pbt_status(args):
    """Show PBT state."""
    if not PBT_STATE_FILE.exists():
        print(json.dumps({"status": "no_pbt_state"}))
        return

    pbt = json.loads(PBT_STATE_FILE.read_text())
    param_space = pbt.get("parameter_space", {})

    summary = {
        "pbt_run_id": pbt["pbt_run_id"],
        "generation": pbt["generation"],
        "max_generations": pbt["max_generations"],
        "population_size": pbt["population_size"],
        "focus": pbt["focus"],
        "current_member": pbt["current_member"],
        "stagnation_count": pbt["stagnation_count"],
        "base_score": pbt["base_score"],
        "best_pbt_score": pbt["best_pbt_score"],
        "best_config_summary": _config_summary(pbt["best_pbt_config"], param_space)
                               if pbt.get("best_pbt_config") else None,
        "generations_completed": len(pbt.get("generations", [])),
    }

    # Add per-generation summaries
    for g in pbt.get("generations", []):
        gen_key = f"gen_{g['generation']}"
        summary[gen_key] = {
            "best_member": g["best_member_id"],
            "best_score": g["best_score"],
            "promoted": g["promoted"],
            "members_run": len(g["members"]),
        }

    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inner loop — mechanical experiment runner")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize a new run")
    sub.add_parser("status", help="Show current state")

    p_exp = sub.add_parser("experiment", help="Run one atomic experiment")
    p_exp.add_argument("--mutation", required=False, default=None,
                        help="Path to mutated train.py (omit for baseline)")
    p_exp.add_argument("--time-budget", type=int, default=300, help="Training time budget (seconds)")
    p_exp.add_argument("--summary", default="", help="Change summary for this experiment")

    p_pbt_init = sub.add_parser("pbt-init", help="Initialize PBT population")
    p_pbt_init.add_argument("--population", type=int, default=6, help="Population size")
    p_pbt_init.add_argument("--generations", type=int, default=3, help="Number of generations")
    p_pbt_init.add_argument("--focus", default="all",
                             choices=["loss_weights", "regularization", "all"],
                             help="Which parameter tier to explore")
    p_pbt_init.add_argument("--time-budget", type=int, default=240,
                             help="Training time budget per member (seconds)")
    p_pbt_init.add_argument("--specialists", action="store_true",
                             help="Add regime-specialist members (morning/midday/afternoon/highvol)")

    p_pbt_run = sub.add_parser("pbt-run", help="Run PBT sweep")

    sub.add_parser("pbt-status", help="Show PBT state")

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "experiment": cmd_experiment,
        "status": cmd_status,
        "pbt-init": cmd_pbt_init,
        "pbt-run": cmd_pbt_run,
        "pbt-status": cmd_pbt_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
