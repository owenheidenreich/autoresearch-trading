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

from __future__ import annotations
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


def _score_config_fingerprint() -> str:
    """Hash _score_config dict from train.py. Changes mean scores are incomparable."""
    import hashlib, re
    try:
        code = TRAIN_PY.read_text()
        # Extract _score_config block
        m = re.search(r'_score_config\s*=\s*\{([^}]+)\}', code)
        if m:
            return hashlib.sha256(m.group(0).encode()).hexdigest()[:16]
    except Exception:
        pass
    return "unknown"


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


def _ssh_cmd(deploy: dict, cmd: str, timeout: int = 600, retries: int = 3) -> subprocess.CompletedProcess:
    """Run command on Akash via SSH with retry on connection failures."""
    for attempt in range(retries):
        result = subprocess.run(
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
        # Exit code 255 = SSH connection failure (auth, network, etc.)
        # Retry only on SSH-level failures, not on command failures
        if result.returncode != 255 or attempt >= retries - 1:
            return result
        delay = 5 * (attempt + 1)  # 5s, 10s
        _log.warning(f"SSH connection failed (attempt {attempt+1}/{retries}), retrying in {delay}s...")
        time.sleep(delay)
    return result


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


def _build_reasoning(summary: str, kept: bool, score: float, best_score: float,
                     metrics: dict, reasons: list, anomaly_flags: list) -> str:
    """Build a rich reasoning string for the Stream of Consciousness panel."""
    parts = []
    # Hypothesis
    parts.append(f"Hypothesis: {summary}")
    # Outcome
    if kept:
        delta = score - best_score
        parts.append(f"KEPT — score {best_score:.4f} → {score:.4f} (+{delta:.4f})")
    else:
        reason_str = ", ".join(reasons) if reasons else "below best"
        parts.append(f"REVERTED — score {score:.4f} vs best {best_score:.4f} ({reason_str})")
    # Key metrics
    pf = metrics.get("profit_factor")
    tpd = metrics.get("trades_per_day")
    wr = metrics.get("win_rate")
    sharpe = metrics.get("trade_sharpe")
    sl = metrics.get("stop_loss_rate")
    rr = metrics.get("rr_ratio")
    if pf is not None:
        metric_parts = [f"PF={pf:.2f}"]
        if tpd is not None:
            metric_parts.append(f"TPD={tpd:.1f}")
        if wr is not None:
            metric_parts.append(f"WR={wr*100:.0f}%")
        if sharpe is not None:
            metric_parts.append(f"Sharpe={sharpe:.2f}")
        if sl is not None:
            metric_parts.append(f"SL={sl*100:.0f}%")
        if rr is not None:
            metric_parts.append(f"R:R={rr:.2f}")
        parts.append("Metrics: " + " | ".join(metric_parts))
    # Anomalies
    if anomaly_flags:
        parts.append(f"Anomalies: {', '.join(anomaly_flags)}")
    return "\n".join(parts)


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
    try:
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError as e:
        _log.error(f"Failed to write experiment record to {path}: {e} — record lost: {record.get('experiment_id', '?')}")


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

    # Check for incomplete promotion from a previous crash
    _promotion_sentinel = BEST_SCORE_FILE.parent / ".promotion_in_progress"
    if _promotion_sentinel.exists():
        _log.warning(
            "Found .promotion_in_progress sentinel — previous promotion may be incomplete. "
            f"Contents: {_promotion_sentinel.read_text().strip()!r}. "
            "Verify best_model.pt, best_train.py, and .best_score are consistent."
        )
        _promotion_sentinel.unlink(missing_ok=True)

    # Read best_score from .best_score file if it exists
    best_score = -5.0
    if BEST_SCORE_FILE.exists():
        try:
            best_score = float(BEST_SCORE_FILE.read_text().strip())
        except (ValueError, OSError) as _e:
            _log.warning(f"Could not parse .best_score (contents: {BEST_SCORE_FILE.read_text()!r}): {_e} — defaulting to -5.0")

    # Validate .best_score matches model checkpoint (catch manual edits)
    if BEST_MODEL_PT.exists() and best_score > -5.0:
        try:
            import torch as _torch
            _ckpt = _torch.load(str(BEST_MODEL_PT), map_location="cpu", weights_only=False)
            _ckpt_score = _ckpt.get('metrics', {}).get('score')
            if _ckpt_score is not None and abs(_ckpt_score - best_score) > 0.5:
                _log.error(
                    f"INTEGRITY: .best_score ({best_score:.4f}) diverges from model checkpoint "
                    f"({_ckpt_score:.4f}). Possible manual edit or stale state. "
                    f"Using checkpoint score as truth."
                )
                best_score = _ckpt_score
                BEST_SCORE_FILE.write_text(str(best_score))
            del _ckpt
        except Exception as _e:
            _log.warning(f"Could not validate .best_score against checkpoint: {_e}")

    state = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "experiment_id": 0,
        "best_score": best_score,
        "kept_count": 0,
        "total_count": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_score_config_fingerprint": _score_config_fingerprint(),
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
        _log.info("No active run — auto-initializing...")
        cmd_init(args)
        state = _load_state()
        if not state:
            print(json.dumps({"error": "Auto-init failed. Check .best_score and training/ directory."}))
            sys.exit(1)

    # Consistency check: if .best_score file disagrees with state, re-sync.
    # This catches stale state after a fresh start that resets .best_score.
    if BEST_SCORE_FILE.exists():
        try:
            _file_score = float(BEST_SCORE_FILE.read_text().strip())
            if abs(_file_score - state["best_score"]) > 0.01 and _file_score < state["best_score"]:
                _log.warning(f".best_score file ({_file_score}) < state best_score ({state['best_score']:.4f}) — "
                             f"fresh start detected, re-initializing...")
                cmd_init(args)
                state = _load_state()
        except (ValueError, OSError):
            pass

    # Score config drift detection: if _score_config changed since last experiment,
    # the stored best_score is incomparable. Auto-reset to -5.0.
    _sc_fingerprint = _score_config_fingerprint()
    _stored_fp = state.get("_score_config_fingerprint")
    if _stored_fp and _sc_fingerprint != _stored_fp:
        _log.warning(
            f"SCORE CONFIG CHANGED (fingerprint {_stored_fp[:8]}→{_sc_fingerprint[:8]}). "
            f"Resetting best_score from {state['best_score']:.4f} to -5.0 — "
            f"old scores are incomparable under new config."
        )
        state["best_score"] = -5.0
        BEST_SCORE_FILE.write_text("-5.0")
        state["_score_config_fingerprint"] = _sc_fingerprint
        _save_state(state)
    elif not _stored_fp:
        # First time: store fingerprint without resetting
        state["_score_config_fingerprint"] = _sc_fingerprint
        _save_state(state)

    _log.info("=" * 60)
    _log.info("TRAINING MODE: SEQUENTIAL (single experiment, warm-start)")
    _log.info("=" * 60)
    state["training_mode"] = "sequential"

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
    _extra_features = os.environ.get("EXTRA_FEATURES", "")
    _ef_export = f"EXTRA_FEATURES='{_extra_features}' " if _extra_features else ""
    train_cmd = (
        f"cd /root/autoresearch-trading && "
        f"{_ef_export}"
        f"TIME_BUDGET={time_budget} "
        f"/opt/conda/bin/python -u training/train.py 2>&1"
    )

    timeout = time_budget + 480  # eval can take 3-4 min on large validation sets
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
        # Fallback: download metrics.json written by train.py (SSH stdout can lose data)
        _log.info("SSH output parse failed — trying metrics.json file fallback...")
        try:
            _metrics_tmp = artifact_dir / "remote_metrics.json"
            _scp_download(deploy, "/root/autoresearch-trading/training/metrics.json", str(_metrics_tmp))
            with open(_metrics_tmp) as _f:
                _remote_metrics = json.load(_f)
            metrics = {"output": output}
            metrics.update(_remote_metrics)
            _log.info(f"Loaded metrics from remote file (score={metrics.get('score', 'N/A')})")
        except Exception as _e:
            _log.warning(f"Metrics file fallback also failed: {_e}")
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

        # Validate downloaded model integrity before committing
        _integrity_ok = False
        try:
            import torch as _torch
            _ckpt = _torch.load(tmp_model, weights_only=False, map_location="cpu")
            _embedded = _ckpt.get("metrics", {}).get("score")
            if _embedded is not None:
                _ratio = _embedded / score if score != 0 else 0
                if _ratio < 0.5:
                    _log.error(
                        f"INTEGRITY FAIL: Downloaded model has embedded score {_embedded:.4f} "
                        f"but training reported {score:.4f} (ratio={_ratio:.2f}). Skipping promotion."
                    )
                else:
                    _log.info(f"Model integrity OK: embedded={_embedded:.4f}, reported={score:.4f}")
                    _integrity_ok = True
            else:
                _log.warning("Model checkpoint has no embedded score — skipping promotion.")
            del _ckpt
        except Exception as _e:
            _log.error(f"Model integrity check failed: {_e} — skipping promotion.")

        if not _integrity_ok:
            _log.warning("REVERTING: model integrity check did not pass, promotion blocked.")
            keep = False
            state["best_score"] = best_score  # restore original
            state["kept_count"] -= 1
        else:
            # Atomic promotion: sentinel guards against partial writes
            sentinel = BEST_SCORE_FILE.parent / ".promotion_in_progress"
            sentinel.write_text(str(score))
            shutil.copy2(tmp_model, BEST_MODEL_PT)    # Promote model
            shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)     # Promote code
            # Atomic score update via temp file + os.rename
            tmp_score = BEST_SCORE_FILE.with_suffix(".tmp")
            tmp_score.write_text(str(score))
            os.rename(str(tmp_score), str(BEST_SCORE_FILE))
            sentinel.unlink(missing_ok=True)

        # Record promotion (with full provenance for reproducibility)
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
            "num_val_days": metrics.get("num_val_days"),
            "num_val_bars": metrics.get("num_val_bars"),
            "num_trades": metrics.get("num_trades"),
            "num_features": metrics.get("num_features"),
            "lookback": metrics.get("lookback"),
            "data_fingerprint": metrics.get("data_fingerprint"),
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
        "reasoning": _build_reasoning(change_summary, keep, score, best_score, metrics, reasons, anomaly_flags),
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


# PBT code removed in v17 simplification.
# Karpathy design: one experiment at a time, one metric, keep/revert.


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

    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "experiment": cmd_experiment,
        "status": cmd_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
