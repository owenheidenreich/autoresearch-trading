#!/usr/bin/env python3
"""ART² — AutoResearch Trading Squared.

Mechanical orchestrator for the outer meta-loop. Handles subprocess management,
data collection, and verification. Strategic decisions are made by Claude Code
(Opus) reading the briefing output.

Designed for Claude Pro (~45 msgs / 5h). Every message counts.

Subcommands:
    python3 tools/art2.py train --minutes 60      # Deploy Akash, run inner loop, download
    python3 tools/art2.py analyze                  # Collect metrics into analysis.json
    python3 tools/art2.py report                   # Generate markdown briefing for Claude Code
    python3 tools/art2.py verify --dates 3         # OOS replay battery + IBKR probe
    python3 tools/art2.py status                   # Current cycle state
    python3 tools/art2.py cycle --minutes 60       # Full: train → analyze → report → await
    python3 tools/art2.py autonomous               # Market-aware loop: train when closed, monitor when open
    python3 tools/art2.py market                   # Check market status + IBKR compatibility
    python3 tools/art2.py init                     # Initialize results/art2/ structure

Pro-optimized flow (3-4 messages per cycle):
    1. User/loop says "run cycle"  → Claude runs: art2.py cycle --minutes 90
    2. [90 min training, 0 messages]
    3. Claude reads briefing.md    → ONE message to analyze + decide + implement
    4. Claude runs: art2.py verify → ONE message to check results
    Total: ~3 messages per 2h cycle = ~7 cycles in a 5h window

Each subcommand is idempotent and checkpoint-based via results/art2/state.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ART2_DIR = RESULTS_DIR / "art2"
CACHE_DIR = Path.home() / ".cache" / "autoresearch-trading"
FEATURES_DIR = CACHE_DIR / "features"
DEPLOY_SH = str(PROJECT_ROOT / "infra" / "deploy.sh")
ET = ZoneInfo("America/New_York")

# US equity market holidays 2025-2027 (NYSE/NASDAQ closed)
US_MARKET_HOLIDAYS = {
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18",
    "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03",
    "2026-05-25", "2026-06-19", "2026-07-03", "2026-09-07",
    "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06",
    "2027-11-25", "2027-12-24",
}

# Early close days (1:00 PM ET) — day before/after major holidays
US_EARLY_CLOSE = {
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-07-02", "2026-11-27", "2026-12-24",
    "2027-11-26",
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.now(ET).strftime("%H:%M:%S ET")
    print(f"[ART²] [{ts}] {msg}", flush=True)


def run_cmd(cmd: list[str], log_path: Path | None = None,
            timeout: int | None = None, env: dict | None = None) -> tuple[int, str]:
    """Run a command, capturing output. Returns (returncode, output)."""
    merged_env = {**os.environ, **(env or {})}
    log(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, cwd=str(PROJECT_ROOT), env=merged_env,
        )
        output = result.stdout + result.stderr
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"$ {' '.join(cmd)}\n\n{output}\n\nExit code: {result.returncode}\n")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        output = f"TIMEOUT after {timeout}s"
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"$ {' '.join(cmd)}\n\n{output}\n")
        return -1, output


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# API Spend Tracking (tier-aware budget guardrails)
# ---------------------------------------------------------------------------

SPEND_TRACKER_PATH = CACHE_DIR / "api_spend_tracker.json"
DEFAULT_TIER_BUDGET = 1000.0  # Tier 3: $1,000/month
AVG_COST_PER_EXPERIMENT = 0.20  # ~$0.20/experiment (Sonnet 4)
AVG_MINUTES_PER_EXPERIMENT = 6  # ~6 min/experiment on H100

# Daemon constants
STOP_SENTINEL = ART2_DIR / "STOP"
PAUSE_SENTINEL = ART2_DIR / "PAUSED"
REVIEW_SENTINEL = ART2_DIR / "REVIEW"
HEARTBEAT_PATH = ART2_DIR / "daemon_heartbeat.json"
ALERTS_PATH = ART2_DIR / "alerts.jsonl"
PID_PATH = ART2_DIR / "daemon.pid"
MAX_CONSECUTIVE_FAILURES = 3
MAX_DAILY_SPEND = 80.0  # $/day safety cap
OPUS_TIMEOUT = 600  # seconds for Claude Code CLI invocation (increased for domain knowledge context)


def _load_spend_tracker(budget: float = DEFAULT_TIER_BUDGET) -> dict[str, Any]:
    """Load or initialize the monthly API spend tracker. Auto-resets on month change."""
    current_month = datetime.now().strftime("%Y-%m")
    tracker = None
    if SPEND_TRACKER_PATH.exists():
        try:
            tracker = json.loads(SPEND_TRACKER_PATH.read_text())
        except Exception:
            pass
    if not tracker or tracker.get("month") != current_month:
        tracker = {
            "month": current_month,
            "tier_budget": budget,
            "sessions": [],
            "total_spend": 0.0,
        }
        _save_spend_tracker(tracker)
    # Allow budget override
    if tracker["tier_budget"] != budget:
        tracker["tier_budget"] = budget
    return tracker


def _save_spend_tracker(tracker: dict[str, Any]) -> None:
    SPEND_TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPEND_TRACKER_PATH, "w") as f:
        json.dump(tracker, f, indent=2, default=str)


def _record_session_spend(run_name: str, cost: float, experiments: int,
                          budget: float = DEFAULT_TIER_BUDGET) -> None:
    """Record a completed session's API spend."""
    tracker = _load_spend_tracker(budget)
    tracker["sessions"].append({
        "run_name": run_name,
        "cost": round(cost, 2),
        "experiments": experiments,
        "timestamp": datetime.now().isoformat(),
    })
    tracker["total_spend"] = round(sum(s["cost"] for s in tracker["sessions"]), 2)
    _save_spend_tracker(tracker)


def _remaining_budget(budget: float = DEFAULT_TIER_BUDGET) -> float:
    tracker = _load_spend_tracker(budget)
    return max(0, tracker["tier_budget"] - tracker["total_spend"])


def _project_session_cost(minutes: int) -> float:
    """Estimate API cost for a session of given duration."""
    est_experiments = minutes / AVG_MINUTES_PER_EXPERIMENT
    return round(est_experiments * AVG_COST_PER_EXPERIMENT, 2)


def load_state() -> dict[str, Any]:
    state_path = ART2_DIR / "state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            pass
    return {
        "cycle": 0,
        "phase": "idle",
        "last_completed_cycle": None,
        "cumulative_cost_usd": 0.0,
        "total_experiments": 0,
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(ET).isoformat()
    write_json(ART2_DIR / "state.json", state)


def cycle_dir(cycle_num: int) -> Path:
    return ART2_DIR / f"cycle-{cycle_num:03d}"


# ---------------------------------------------------------------------------
# Market Hours & IBKR Compatibility
# ---------------------------------------------------------------------------

def is_market_open() -> dict:
    """Check if US equity market is currently open.

    Returns dict with:
        open: bool           — True if market is open right now
        next_open: str       — ISO datetime of next market open (ET)
        next_close: str      — ISO datetime of next market close (ET)
        minutes_to_open: int — minutes until next open (negative if open)
        minutes_to_close: int — minutes until close (negative if closed)
        reason: str          — human-readable status
    """
    from datetime import date as _date
    now = datetime.now(ET)
    today = now.date()
    today_str = today.isoformat()

    result = {"timestamp": now.isoformat(), "open": False}

    # Weekend check
    if today.weekday() >= 5:
        days_to_monday = 7 - today.weekday()
        next_open_date = today + timedelta(days=days_to_monday)
        # Skip holidays on Monday
        while next_open_date.isoformat() in US_MARKET_HOLIDAYS:
            next_open_date += timedelta(days=1)
        next_open = datetime(next_open_date.year, next_open_date.month, next_open_date.day, 9, 30, tzinfo=ET)
        result["reason"] = "weekend"
        result["next_open"] = next_open.isoformat()
        result["minutes_to_open"] = int((next_open - now).total_seconds() / 60)
        return result

    # Holiday check
    if today_str in US_MARKET_HOLIDAYS:
        next_day = today + timedelta(days=1)
        while next_day.isoformat() in US_MARKET_HOLIDAYS or next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_open = datetime(next_day.year, next_day.month, next_day.day, 9, 30, tzinfo=ET)
        result["reason"] = "holiday"
        result["next_open"] = next_open.isoformat()
        result["minutes_to_open"] = int((next_open - now).total_seconds() / 60)
        return result

    # Market hours
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if today_str in US_EARLY_CLOSE:
        market_close = now.replace(hour=13, minute=0, second=0, microsecond=0)
    else:
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

    result["next_close"] = market_close.isoformat()

    if now < market_open:
        result["reason"] = "pre-market"
        result["next_open"] = market_open.isoformat()
        result["minutes_to_open"] = int((market_open - now).total_seconds() / 60)
        result["minutes_to_close"] = int((market_close - now).total_seconds() / 60)
    elif now >= market_close:
        next_day = today + timedelta(days=1)
        while next_day.isoformat() in US_MARKET_HOLIDAYS or next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_open = datetime(next_day.year, next_day.month, next_day.day, 9, 30, tzinfo=ET)
        result["reason"] = "after-hours"
        result["next_open"] = next_open.isoformat()
        result["minutes_to_open"] = int((next_open - now).total_seconds() / 60)
    else:
        result["open"] = True
        result["reason"] = "market open"
        result["minutes_to_close"] = int((market_close - now).total_seconds() / 60)
        result["next_open"] = market_open.isoformat()

    return result


def ibkr_compatibility_gate() -> dict:
    """Verify that the current best model is compatible with IBKR live trading.

    Checks:
    1. best_model.pt exists and loads successfully
    2. Model produces valid action outputs (dry-run forward pass)
    3. Feature parity: training features match live feature list
    4. IBKR connectivity (ib_probe.py)

    Returns dict with pass/fail status and details.
    """
    gate = {"passed": True, "checks": [], "timestamp": datetime.now(ET).isoformat()}

    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    train_py = PROJECT_ROOT / "training" / "best_train.py"

    # Check 1: Model exists
    if not model_path.exists():
        gate["passed"] = False
        gate["checks"].append({"name": "model_exists", "passed": False, "detail": "best_model.pt not found"})
        return gate
    gate["checks"].append({"name": "model_exists", "passed": True, "detail": f"hash={hash_file(model_path)}"})

    # Check 2: Model checkpoint is valid and has correct head shapes
    check2 = {"name": "model_checkpoint", "passed": False}
    rc, output = run_cmd(
        [sys.executable, "-c", f"""
import torch, sys
cp = torch.load('{model_path}', map_location='cpu', weights_only=False)
state = cp.get('model_state_dict') or cp.get('state_dict')
if state is None:
    print("ERROR: no state_dict in checkpoint")
    sys.exit(1)
n_params = sum(p.numel() for p in state.values())
# Check gate head (2 outputs) and direction head (6 outputs) exist
gate_keys = [k for k in state if 'gate' in k and 'weight' in k]
dir_keys = [k for k in state if ('dir' in k or 'direction' in k) and 'weight' in k]
# Find final linear layers by output size
gate_out = None
dir_out = None
for k, v in state.items():
    if v.dim() >= 2:
        if v.shape[0] == 2 and 'gate' in k:
            gate_out = v.shape[0]
        if v.shape[0] == 6 and ('dir' in k or 'direction' in k):
            dir_out = v.shape[0]
assert gate_out == 2, f"No gate head with 2 outputs found"
assert dir_out == 6, f"No direction head with 6 outputs found"
print(f"Checkpoint OK: {{n_params}} params, gate={{gate_out}}, dir={{dir_out}}")
"""],
        timeout=30,
    )
    if rc == 0:
        check2["passed"] = True
        check2["detail"] = output.strip().split("\n")[-1]
    else:
        check2["detail"] = output.strip()[-200:]
        gate["passed"] = False
    gate["checks"].append(check2)

    # Check 3: Feature parity
    check3 = {"name": "feature_parity", "passed": False}
    data_pt = CACHE_DIR / "features" / "data.pt"
    if not data_pt.exists():
        # Fallback: check training/ dir
        data_pt = PROJECT_ROOT / "training" / "data.pt"
    rc, output = run_cmd(
        [sys.executable, "-c", f"""
import torch
data = torch.load('{data_pt}', map_location='cpu', weights_only=False)
# data.pt uses 'features' key (not 'X')
feat = data['features'] if 'features' in data else data.get('X')
if feat is None:
    print("ERROR: no 'features' or 'X' key in data.pt")
    raise SystemExit(1)
n_features = feat.shape[-1]
print(f"data.pt features: {{n_features}}, samples: {{feat.shape[0]}}")
assert n_features == 32, f"Expected 32 features, got {{n_features}}"
# Check val_start_idx exists (needed for train/eval split)
assert 'val_start_idx' in data, "Missing val_start_idx"
print("Feature parity OK")
"""],
        timeout=60,
    )
    if rc == 0:
        check3["passed"] = True
        check3["detail"] = "32 features confirmed"
    else:
        check3["detail"] = output.strip()[-200:]
        gate["passed"] = False
    gate["checks"].append(check3)

    # Check 4: IBKR connectivity
    check4 = {"name": "ibkr_probe", "passed": False}
    rc, output = run_cmd(
        [sys.executable, str(PROJECT_ROOT / "tools" / "ib_probe.py")],
        timeout=60,
    )
    if rc == 0:
        check4["passed"] = True
        check4["detail"] = "IBKR connected"
    else:
        check4["passed"] = False
        check4["detail"] = f"IBKR probe failed (exit {rc})"
        # IBKR down doesn't block training, just flags it
        log("WARNING: IBKR not reachable — model passes other gates but live trading unavailable")
    gate["checks"].append(check4)

    return gate


# ---------------------------------------------------------------------------
# Preflight Validation (before spending AKT/API credits)
# ---------------------------------------------------------------------------

def _preflight_validate() -> bool:
    """Validate train.py, best_train.py, and data.pt before deploying.

    Catches issues that would cause 100% experiment failure:
    - Syntax errors in train.py
    - Missing required fields in data.pt
    - train.py / best_train.py divergence
    - SCORE_* env var readers (safety checker will reject)
    """
    import ast as _ast
    import re as _re

    issues: list[str] = []
    train_py = PROJECT_ROOT / "training" / "train.py"
    best_py = PROJECT_ROOT / "training" / "best_train.py"

    # 1. Syntax check
    for f in [train_py, best_py]:
        if not f.exists():
            issues.append(f"MISSING: {f.name}")
            continue
        try:
            _ast.parse(f.read_text())
        except SyntaxError as e:
            issues.append(f"SYNTAX ERROR in {f.name}: {e}")

    # 2. SCORE_* env var check (safety checker rejects these)
    if train_py.exists():
        code = train_py.read_text()
        if _re.search(r'_env_float\(\s*["\']SCORE_', code):
            issues.append("train.py contains _env_float('SCORE_...') — safety checker will reject")

    # 3. Data contract check
    data_paths = [
        CACHE_DIR / "features" / "data.pt",
        PROJECT_ROOT / "training" / "data.pt",
    ]
    data_pt = None
    for dp in data_paths:
        if dp.exists():
            data_pt = dp
            break

    if data_pt is None:
        issues.append("data.pt not found in any expected location")
    else:
        try:
            import torch
            data = torch.load(data_pt, weights_only=False, map_location="cpu")
            required_fields = [
                'features', 'targets', 'valid_mask', 'dates',
                'call_pnl', 'put_pnl', 'exit_call_label', 'exit_put_label',
                'otm5_call_pnl', 'otm5_put_pnl', 'otm10_call_pnl', 'otm10_put_pnl',
                'call_stopped_pnl', 'put_stopped_pnl',
                'otm5_call_stopped_pnl', 'otm5_put_stopped_pnl',
                'otm10_call_stopped_pnl', 'otm10_put_stopped_pnl',
                'day_boundaries', 'supervision_weight', 'actionable_mask', 'risk_state_mask',
            ]
            missing = [f for f in required_fields if f not in data]
            if missing:
                issues.append(f"data.pt missing fields: {', '.join(missing)}")
            else:
                n_feat = data['features'].shape[1] if len(data['features'].shape) > 1 else 0
                if n_feat != 32:
                    issues.append(f"data.pt has {n_feat} features, expected 32")
                log(f"  data.pt: {data['features'].shape[0]} bars, {n_feat} features ✓")
        except Exception as e:
            issues.append(f"Failed to load data.pt: {e}")

    # Report
    if issues:
        for issue in issues:
            log(f"  ✗ {issue}")
        return False
    else:
        log("  Preflight: all checks passed ✓")
        return True


# ---------------------------------------------------------------------------
# Phase 1: TRAIN
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> bool:
    """Deploy to Akash, run inner loop, download results."""
    state = load_state()
    cycle_num = state["cycle"] + 1
    cdir = cycle_dir(cycle_num)
    cdir.mkdir(parents=True, exist_ok=True)
    log_dir = cdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    state["cycle"] = cycle_num
    state["phase"] = "training"
    save_state(state)

    minutes = args.minutes
    hours = round(minutes / 60, 2)

    if args.dry_run:
        log(f"[dry-run] Would boot Akash, train {minutes} min, stop and download")
        state["phase"] = "trained"
        save_state(state)
        return True

    # --- API BUDGET CHECK ---
    budget = getattr(args, "budget", DEFAULT_TIER_BUDGET)
    remaining = _remaining_budget(budget)
    projected = _project_session_cost(args.minutes)
    log(f"API budget: ${remaining:.2f} remaining (tier ${budget:.0f}/mo), projected session cost: ${projected:.2f}")
    if projected > remaining:
        log(f"WARNING: Projected cost ${projected:.2f} exceeds remaining budget ${remaining:.2f}")
        if projected > remaining * 1.5:
            log("BLOCKED: Would exceed budget by >50%. Reduce --minutes or increase --budget.")
            state["phase"] = "budget_exceeded"
            save_state(state)
            return False
        log("Proceeding with warning — session may be capped by credit exhaustion.")

    # --- PREFLIGHT VALIDATION (before spending AKT) ---
    preflight_ok = _preflight_validate()
    if not preflight_ok:
        log("PREFLIGHT FAILED — fix issues before deploying")
        state["phase"] = "preflight_failed"
        save_state(state)
        return False

    # Back up current model
    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    prev_path = PROJECT_ROOT / "training" / "best_model.pt.prev"
    if model_path.exists():
        shutil.copy2(model_path, prev_path)
        log("Backed up best_model.pt → best_model.pt.prev")

    # Snapshot pre-training metrics
    pre = _snapshot_metrics()
    write_json(cdir / "pre_metrics.json", pre)

    # Calculate AKT needed for the full session upfront.
    # H100 pricing: ~7-8 AKT/hour. Deploy with full amount to avoid underfunded leases.
    deposit_akt = getattr(args, 'deposit_akt', None) or max(5, int(minutes / 60 * 8) + 2)
    log(f"=== TRAIN: Booting Akash ({minutes} min budget, {deposit_akt} AKT) ===")
    rc, output = run_cmd(
        [DEPLOY_SH, "boot"],
        log_path=log_dir / "boot.log",
        timeout=300,
        env={"DEPOSIT_AKT": str(deposit_akt)},
    )
    if rc != 0:
        log(f"Boot FAILED (exit {rc})")
        state["phase"] = "train_failed"
        save_state(state)
        return False

    # Start inner loop (retry once on SSH failure — transient 255 errors are common)
    # deploy.sh start is NON-BLOCKING — it launches the loop on Akash and returns.
    # We must poll deploy.sh status until training completes, then stop to download.
    max_exp = max(5, minutes // 6)  # ~6 min per experiment (API call + training)
    log(f"Starting inner loop ({hours}h, max {max_exp} experiments)...")
    for attempt in range(1, 3):
        rc, output = run_cmd(
            [DEPLOY_SH, "start", "--hours", str(hours), "--max-experiments", str(max_exp)],
            log_path=log_dir / f"start_attempt{attempt}.log",
            timeout=300,
        )
        if rc == 0:
            log("Loop launched on Akash")
            break
        if rc == 255 and attempt == 1:
            log(f"Start failed with SSH error (exit 255) — retrying in 15s...")
            time.sleep(15)
        else:
            log(f"Start FAILED (exit {rc})")
            run_cmd([DEPLOY_SH, "stop", "-y"], log_path=log_dir / "stop_cleanup.log", timeout=300)
            state["phase"] = "train_failed"
            save_state(state)
            return False

    # Poll until training completes (auto-sync pulls results in background)
    # Also detect all-crash patterns and abort early to save API credits.
    deadline = time.time() + (minutes + 15) * 60
    consecutive_failures = 0
    last_total = 0
    log(f"Waiting for training (up to {minutes + 15} min)...")
    while time.time() < deadline:
        time.sleep(60)
        rc, output = run_cmd([DEPLOY_SH, "status"], timeout=30)

        # Check for completion signals
        if "completed" in output.lower():
            log("Training completed!")
            break
        if "No active deployment" in output or "not running" in output.lower():
            log("Deployment ended")
            break

        # Parse experiment progress for early abort.
        # deploy.sh status dashboard format: "Results:    {kept} kept / {failed} failed / {total} total"
        # auto-sync log format: "exp={total} kept={kept} best={score} remaining={h}h"
        import re as _re
        m_dashboard = _re.search(r'Results:\s*(\d+)\s+kept\s*/\s*(\d+)\s+failed\s*/\s*(\d+)\s+total', output)
        m_sync = _re.search(r'exp=(\d+)\s+kept=(\d+).*?remaining=([\d.]+)h', output)
        if m_dashboard:
            kept, failed, total = int(m_dashboard.group(1)), int(m_dashboard.group(2)), int(m_dashboard.group(3))
        elif m_sync:
            total, kept = int(m_sync.group(1)), int(m_sync.group(2))
            failed = total - kept
        else:
            kept = failed = total = -1

        if total >= 0 and total > last_total:
            last_total = total
            # Only count actual CRASHES as failures (not regressions — those are normal)
            if failed > 0 and failed >= total and total >= 3:
                consecutive_failures = total
            elif failed > 0 and failed == last_total:
                consecutive_failures += (total - last_total)  # no reset if still crashing
            else:
                consecutive_failures = 0
            log(f"  Progress: {kept} kept / {failed} failed / {total} total")

        # Early abort: 3+ consecutive CRASHES = systemic issue (not regressions)
        if consecutive_failures >= 3:
            log(f"EARLY ABORT: {consecutive_failures} consecutive crashes — systemic issue")
            break

        # Check sync.log for completion
        sync_log = RESULTS_DIR / "sync.log"
        loop_done = False
        if sync_log.exists():
            tail = sync_log.read_text().split("\n")[-5:]
            for line in tail:
                if "COMPLETED" in line or "LOOP COMPLETED" in line:
                    log("Training completed (sync detected)")
                    loop_done = True
                    break
        if loop_done:
            break
    else:
        log("Training timed out — stopping")

    # Stop and download
    log("Stopping deployment and downloading results...")
    rc, output = run_cmd(
        [DEPLOY_SH, "stop", "-y"],
        log_path=log_dir / "stop.log",
        timeout=600,
    )
    if rc != 0:
        log(f"Stop had issues (exit {rc}) — checking for results")

    # Verify model downloaded
    if model_path.exists():
        age = time.time() - model_path.stat().st_mtime
        log(f"best_model.pt found ({age:.0f}s old)")
        state["phase"] = "trained"
    else:
        log("WARNING: best_model.pt not found — restoring backup")
        if prev_path.exists():
            shutil.copy2(prev_path, model_path)
        state["phase"] = "train_failed"

    # Record API spend from downloaded results
    try:
        run_name = state.get("run_name", "unknown")
        status_json = RESULTS_DIR / run_name / "status.json"
        if status_json.exists():
            status_data = json.loads(status_json.read_text())
            api_cost = status_data.get("api_cost", {})
            cost = api_cost.get("total_cost", 0) if isinstance(api_cost, dict) else float(api_cost or 0)
            exp_count = status_data.get("total_experiments", 0)
            _record_session_spend(run_name, cost, exp_count, budget)
            log(f"Recorded API spend: ${cost:.2f} for {exp_count} experiments")
    except Exception as e:
        log(f"Could not record API spend: {e}")

    save_state(state)
    return state["phase"] == "trained"


# ---------------------------------------------------------------------------
# Phase 2: ANALYZE
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> bool:
    """Collect all metrics for Claude Code to analyze."""
    state = load_state()
    cycle_num = state.get("cycle", 0)
    if cycle_num == 0:
        log("No active cycle — run 'train' first or use 'init'")
        # Allow analyze without training (e.g., analyzing existing results)
        cycle_num = state.get("cycle", 0) or 1
        state["cycle"] = cycle_num
        cdir = cycle_dir(cycle_num)
        cdir.mkdir(parents=True, exist_ok=True)
    else:
        cdir = cycle_dir(cycle_num)

    state["phase"] = "analyzing"
    save_state(state)

    analysis = {}

    # 1. Latest run experiments
    latest_run = _find_latest_run()
    if latest_run:
        experiments = _parse_experiments(latest_run / "experiments.v2.jsonl")
        analysis["latest_run"] = {
            "name": latest_run.name,
            "total_experiments": len(experiments),
            "kept_count": sum(1 for e in experiments if e.get("kept", False)),
            "experiments": experiments[-20:],  # Last 20 for context
        }
        # Score trajectory
        scores = [e.get("score", 0) for e in experiments if e.get("score") is not None]
        if scores:
            analysis["latest_run"]["best_score"] = max(scores)
            analysis["latest_run"]["worst_score"] = min(scores)
            analysis["latest_run"]["mean_score"] = sum(scores) / len(scores)

        # Status.json for cost info
        status = read_json(latest_run / "status.json")
        if status:
            analysis["latest_run"]["api_cost"] = status.get("api_cost")
            analysis["latest_run"]["cache_hit_pct"] = status.get("cache_hit_pct")

        log(f"Latest run: {latest_run.name} ({len(experiments)} experiments)")
    else:
        log("No run directories found")
        analysis["latest_run"] = None

    # 2. Cross-run score history (all promoted models)
    analysis["promoted_history"] = _parse_promoted_history()
    log(f"Promoted models: {len(analysis['promoted_history'])}")

    # 3. Paper trading P&L
    analysis["paper_trades"] = _collect_paper_trades()
    if analysis["paper_trades"]:
        total_pnl = sum(t.get("total_pnl_pct", 0) for t in analysis["paper_trades"])
        log(f"Paper trading: {len(analysis['paper_trades'])} days, {total_pnl:+.1f}% total")
    else:
        log("No paper trading data found")

    # 4. Current system config snapshot
    analysis["config"] = {
        "program_md_hash": hash_file(PROJECT_ROOT / "training" / "program.md"),
        "prepare_py_hash": hash_file(PROJECT_ROOT / "training" / "prepare.py"),
        "train_py_hash": hash_file(PROJECT_ROOT / "training" / "train.py"),
        "best_train_py_hash": hash_file(PROJECT_ROOT / "training" / "best_train.py"),
        "best_model_exists": (PROJECT_ROOT / "training" / "best_model.pt").exists(),
        "data_pt_exists": (FEATURES_DIR / "data.pt").exists(),
        "data_pt_size_mb": round((FEATURES_DIR / "data.pt").stat().st_size / 1e6, 1)
        if (FEATURES_DIR / "data.pt").exists() else None,
    }

    # 5. Anti-gaming checks
    analysis["gaming_checks"] = _check_for_gaming(analysis)

    # 6. Inner loop health
    analysis["inner_loop_health"] = _assess_inner_loop_health(analysis)

    write_json(cdir / "analysis.json", analysis)
    log(f"Analysis written to {cdir / 'analysis.json'}")

    state["phase"] = "analyzed"
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# Phase 5: VERIFY
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> bool:
    """Run OOS replay battery and IBKR probe."""
    state = load_state()
    cycle_num = state.get("cycle", 0) or 1
    cdir = cycle_dir(cycle_num)
    cdir.mkdir(parents=True, exist_ok=True)
    log_dir = cdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    state["phase"] = "verifying"
    save_state(state)

    n_dates = args.dates
    verify_results = {}

    if args.dry_run:
        log(f"[dry-run] Would run replay battery on {n_dates} OOS dates + IBKR probe")
        state["phase"] = "verified"
        save_state(state)
        return True

    # Pick OOS dates (most recent weekdays before today)
    oos_dates = _pick_oos_dates(n_dates)
    if not oos_dates:
        log("WARNING: Could not determine OOS dates — skipping replay battery")
        verify_results["replay_battery"] = {"status": "skipped", "reason": "no dates"}
    else:
        log(f"Running replay battery on {len(oos_dates)} dates: {oos_dates}")
        battery_dir = cdir / "replay_battery"
        battery_dir.mkdir(parents=True, exist_ok=True)

        model_path = PROJECT_ROOT / "training" / "best_model.pt"
        train_py_path = PROJECT_ROOT / "training" / "best_train.py"

        replay_script = PROJECT_ROOT / "training" / "replay.py"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "tools" / "replay_battery.py"),
            "--dates", ",".join(oos_dates),
            "--output", str(battery_dir),
            "--no-download",
        ]
        if model_path.exists():
            cmd.extend(["--model", str(model_path)])
        if train_py_path.exists():
            cmd.extend(["--train-py", str(train_py_path)])

        rc, output = run_cmd(
            cmd,
            log_path=log_dir / "replay_battery.log",
            timeout=600,
        )

        # Read battery summary
        battery_summary = read_json(battery_dir / "battery_summary.json")
        if battery_summary:
            verify_results["replay_battery"] = battery_summary
            log(f"Replay battery: {battery_summary.get('pass_count', '?')}/{battery_summary.get('total', '?')} passed")
        else:
            verify_results["replay_battery"] = {"status": "failed", "exit_code": rc}
            log(f"Replay battery failed (exit {rc})")

    # IBKR probe
    log("Running IBKR connectivity probe...")
    rc, output = run_cmd(
        [sys.executable, str(PROJECT_ROOT / "tools" / "ib_probe.py")],
        log_path=log_dir / "ib_probe.log",
        timeout=60,
    )
    verify_results["ibkr_probe"] = {
        "status": "pass" if rc == 0 else "fail",
        "exit_code": rc,
    }
    log(f"IBKR probe: {'PASS' if rc == 0 else 'FAIL'}")

    # Post-change metrics snapshot
    post = _snapshot_metrics()
    write_json(cdir / "post_metrics.json", post)

    # Compare pre vs post if both exist
    pre = read_json(cdir / "pre_metrics.json")
    if pre and post:
        verify_results["metric_comparison"] = _compare_metrics(pre, post)
        gaming = _detect_gaming(pre, post)
        if gaming:
            verify_results["gaming_alerts"] = gaming
            for alert in gaming:
                log(f"GAMING ALERT: {alert}")

    write_json(cdir / "verify_results.json", verify_results)

    # Log to changes.jsonl
    _log_change(cycle_num, cdir, verify_results)

    state["phase"] = "verified"
    state["last_completed_cycle"] = cycle_num
    save_state(state)
    log(f"Cycle {cycle_num} verification complete")
    return True


# ---------------------------------------------------------------------------
# AUTOMATED DECISION TREE (from art2_contract.md)
# ---------------------------------------------------------------------------

def compute_recommended_action(analysis: dict) -> dict:
    """Automated decision tree per art2_contract.md.

    Returns dict with:
        action: str       — A/B/C/D/E/F letter code
        label: str        — human-readable action name
        reason: str       — why this was chosen
        confidence: str   — high/medium/low
        path: list[str]   — decision tree steps taken
    """
    path = []
    lr = analysis.get("latest_run", {}) or {}
    total = lr.get("total_experiments", 0)
    kept = lr.get("kept_count", 0)
    experiments = lr.get("experiments", [])
    gaming = analysis.get("gaming_checks", [])
    paper = analysis.get("paper_trades", [])

    # Step 0: No data
    if total == 0:
        path.append("Step 0: No experiment data")
        return {
            "action": "F",
            "label": "Fix infrastructure",
            "reason": "No experiments found — training may not have run",
            "confidence": "high",
            "path": path,
        }

    # Step 1: Crash detection — if all experiments crashed
    crash_exps = [e for e in experiments if e.get("failure_type") == "train_crash"]
    if len(crash_exps) == total and total >= 3:
        path.append(f"Step 1: ALL {total} experiments crashed")
        return {
            "action": "F",
            "label": "Fix infrastructure",
            "reason": f"All {total} experiments crashed — systemic bug in train.py",
            "confidence": "high",
            "path": path,
        }

    # Step 1: Accept rate test
    accept_rate = kept / total if total > 0 else 0
    path.append(f"Step 1: Accept rate = {kept}/{total} ({accept_rate:.0%})")

    if accept_rate >= 0.33:
        path.append("Accept rate >= 33% → LET IT COOK")
        return {
            "action": "A",
            "label": "Let it cook",
            "reason": f"Inner loop is productive ({kept}/{total} kept, {accept_rate:.0%} accept rate)",
            "confidence": "high",
            "path": path,
        }

    # Step 2: Gaming detection
    if gaming:
        gaming_indicators = [g for g in gaming if "GAMING" in g.upper()]
        if gaming_indicators:
            path.append(f"Step 2: Gaming detected — {gaming_indicators[0][:80]}")
            return {
                "action": "F",
                "label": "Fix infrastructure (revert gamed baseline)",
                "reason": f"Score-PF divergence detected: {gaming_indicators[0]}",
                "confidence": "high",
                "path": path,
            }

    # Step 2b: Failure pattern — all safety-blocked
    safety_blocked = [e for e in experiments if e.get("failure_type") == "safety_check"]
    if len(safety_blocked) > total * 0.8 and total >= 5:
        path.append(f"Step 2: {len(safety_blocked)}/{total} safety-blocked")
        return {
            "action": "F",
            "label": "Fix infrastructure",
            "reason": f"{len(safety_blocked)}/{total} experiments blocked by safety checker — likely a pattern in train.py",
            "confidence": "high",
            "path": path,
        }

    # Step 3: Paper trading divergence
    if paper and lr.get("best_score", 0) > 0:
        recent_paper = paper[-5:]
        paper_pnl = sum(t.get("total_pnl_pct", 0) for t in recent_paper)
        if paper_pnl < -15:  # >30% divergence from positive backtest
            path.append(f"Step 3: Paper P&L = {paper_pnl:+.1f}% (divergence)")
            return {
                "action": "E",
                "label": "Rebuild data / data quality check",
                "reason": f"Paper trading P&L ({paper_pnl:+.1f}%) diverges significantly from positive backtest scores",
                "confidence": "medium",
                "path": path,
            }

    # Step 3b: Stall detection
    if accept_rate < 0.05 and total >= 10:
        # Check if scores are improving even if not kept
        scores = [e.get("score", -999) for e in experiments if e.get("score") is not None]
        if scores:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            avg_first = sum(first_half) / len(first_half) if first_half else 0
            avg_second = sum(second_half) / len(second_half) if second_half else 0

            if avg_second > avg_first * 1.2 and avg_second > 0:
                path.append(f"Step 3: Scores improving ({avg_first:.2f} → {avg_second:.2f}) but not kept")
                return {
                    "action": "B",
                    "label": "Steer inner loop",
                    "reason": f"Scores trending up ({avg_first:.2f} → {avg_second:.2f}) — steer lab_notebook to reinforce this direction",
                    "confidence": "medium",
                    "path": path,
                }

        path.append(f"Step 3: Stalled — {accept_rate:.0%} accept rate, {total} experiments")
        return {
            "action": "B",
            "label": "Steer inner loop",
            "reason": f"Low accept rate ({accept_rate:.0%}) — redirect inner loop via lab_notebook.md",
            "confidence": "medium",
            "path": path,
        }

    # Step 4: Default — moderate accept rate, steer gently
    path.append(f"Step 4: Moderate progress ({accept_rate:.0%}), default to steering")
    if accept_rate >= 0.15:
        return {
            "action": "A",
            "label": "Let it cook",
            "reason": f"Acceptable progress ({kept}/{total} kept). Inner loop finding improvements.",
            "confidence": "medium",
            "path": path,
        }

    return {
        "action": "B",
        "label": "Steer inner loop",
        "reason": f"Below-average accept rate ({accept_rate:.0%}) — consider adjusting lab_notebook priorities",
        "confidence": "low",
        "path": path,
    }


# ---------------------------------------------------------------------------
# Phase 3: REPLAY — backtest best model, get trade-by-trade metrics
# ---------------------------------------------------------------------------

def cmd_replay(args: argparse.Namespace) -> bool:
    """Run replay backtest on best model, produce trade-level metrics."""
    state = load_state()
    cycle_num = state.get("cycle", 0) or 1
    cdir = cycle_dir(cycle_num)
    cdir.mkdir(parents=True, exist_ok=True)

    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    if not model_path.exists():
        log("No best_model.pt found — skipping replay")
        return False

    log("=== REPLAY: Running backtest on best model ===")
    output_dir = cdir / "replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    rc, output = run_cmd(
        ["python3", str(PROJECT_ROOT / "training" / "replay.py"),
         "--backtest",
         "--model", str(model_path),
         "--output-dir", str(output_dir)],
        log_path=cdir / "logs" / "replay.log",
        timeout=300,
    )

    if rc != 0:
        log(f"Replay failed (exit {rc})")
        return False

    # Parse replay output for key metrics
    replay_metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        for key in ["profit_factor", "trades_per_day", "win_rate", "stop_loss_rate",
                     "trade_sharpe", "max_drawdown", "num_trades", "avg_winner",
                     "avg_loser", "total_return"]:
            if line.lower().startswith(key + ":") or line.lower().startswith(key + " "):
                try:
                    val = float(line.split(":")[-1].strip().rstrip("%"))
                    replay_metrics[key] = val
                except (ValueError, IndexError):
                    pass

    # Also try to parse from trade log CSV if it exists
    trade_csv = output_dir / "backtest_trades.csv"
    if not trade_csv.exists():
        trade_csv = output_dir / "trade_log.csv"
    if trade_csv.exists():
        try:
            trades = []
            with open(trade_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
            if trades:
                pnls = [float(t.get("pnl_pct", 0)) for t in trades if t.get("pnl_pct")]
                winners = [p for p in pnls if p > 0]
                losers = [p for p in pnls if p <= 0]
                stops = sum(1 for t in trades if "stop" in t.get("reason", t.get("exit_reason", "")).lower())

                replay_metrics["num_trades"] = len(trades)
                replay_metrics["win_rate"] = len(winners) / max(len(pnls), 1)
                replay_metrics["stop_loss_rate"] = stops / max(len(trades), 1)
                if winners:
                    replay_metrics["avg_winner"] = sum(winners) / len(winners)
                if losers:
                    replay_metrics["avg_loser"] = sum(losers) / len(losers)
                total_won = sum(winners) if winners else 0
                total_lost = abs(sum(losers)) if losers else 0
                replay_metrics["profit_factor"] = total_won / max(total_lost, 1e-6)

                # Time-of-day analysis
                tod_buckets = {"morning": [], "midday": [], "afternoon": [], "power_hour": []}
                for t in trades:
                    entry = t.get("entry_time", "")
                    pnl = float(t.get("pnl_pct", 0))
                    # Handle HH:MM format (e.g. "08:31") or full timestamp
                    hour_str = entry.split(":")[0][-2:] if entry else ""
                    try:
                        hour = int(hour_str)
                    except ValueError:
                        hour = -1
                    if hour in (9, 10):
                        tod_buckets["morning"].append(pnl)
                    elif hour in (11, 12):
                        tod_buckets["midday"].append(pnl)
                    elif hour in (13, 14):
                        tod_buckets["afternoon"].append(pnl)
                    elif hour == 15:
                        tod_buckets["power_hour"].append(pnl)

                replay_metrics["time_of_day"] = {}
                for bucket, pnls_b in tod_buckets.items():
                    if pnls_b:
                        w = [p for p in pnls_b if p > 0]
                        l = [p for p in pnls_b if p <= 0]
                        pf = sum(w) / max(abs(sum(l)), 1e-6) if l else float("inf")
                        replay_metrics["time_of_day"][bucket] = {
                            "trades": len(pnls_b),
                            "pf": round(pf, 2),
                            "avg_pnl": round(sum(pnls_b) / len(pnls_b), 4),
                        }

                # Hold time distribution
                hold_times = [int(t.get("bars_held", 0)) for t in trades]
                if hold_times:
                    replay_metrics["avg_hold_bars"] = sum(hold_times) / len(hold_times)
                    replay_metrics["max_hold_bars"] = max(hold_times)

        except Exception as e:
            log(f"Warning: could not parse trade CSV: {e}")

    # Save replay metrics
    write_json(cdir / "replay_metrics.json", replay_metrics)
    log(f"Replay complete: {replay_metrics.get('num_trades', '?')} trades, "
        f"PF={replay_metrics.get('profit_factor', '?')}, "
        f"WR={replay_metrics.get('win_rate', '?')}")

    state["phase"] = "replayed"
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# Phase 4: DIAGNOSE — compare training vs replay, find mismatches
# ---------------------------------------------------------------------------

def cmd_diagnose(args: argparse.Namespace) -> bool:
    """Compare training metrics against replay metrics to find mismatches.

    The core question: is training playing the same game as live trading?
    """
    state = load_state()
    cycle_num = state.get("cycle", 0) or 1
    cdir = cycle_dir(cycle_num)

    # Load analysis (training metrics)
    analysis = read_json(cdir / "analysis.json")
    if not analysis:
        log("No analysis.json — run analyze first")
        return False

    # Load replay metrics
    replay = read_json(cdir / "replay_metrics.json")
    if not replay:
        log("No replay_metrics.json — run replay first")
        return False

    log("=== DIAGNOSE: Comparing training vs replay ===")

    diagnosis = {
        "mismatches": [],
        "warnings": [],
        "healthy": [],
        "recommendation": None,
    }

    # Get training metrics from best experiment or latest
    latest = analysis.get("latest_run", {})
    exps = latest.get("experiments", [])
    train_metrics = {}
    if exps:
        # Use best non-crash experiment
        valid_exps = [e for e in exps if e.get("score", -999) > -999]
        if valid_exps:
            best = max(valid_exps, key=lambda e: e.get("score", -999))
            train_metrics = {
                "profit_factor": best.get("profit_factor"),
                "trades_per_day": best.get("trades_per_day"),
                "win_rate": best.get("win_rate"),
                "stop_loss_rate": best.get("stop_loss_rate"),
                "trade_sharpe": best.get("trade_sharpe"),
                "max_drawdown": best.get("max_drawdown"),
                "num_trades": best.get("num_trades"),
                "direction_collapse_pct": best.get("direction_collapse_pct"),
                "worst_chunk_pf": best.get("worst_chunk_pf"),
            }

    # Compare key metrics
    comparisons = [
        ("profit_factor", 0.20, "PF diverges — loss function may not match eval"),
        ("stop_loss_rate", 0.10, "Stop rate diverges — stop implementation mismatch"),
        ("trades_per_day", 0.30, "TPD diverges — gate behavior differs"),
        ("win_rate", 0.15, "Win rate diverges — different trade selection"),
    ]

    for metric, threshold, msg in comparisons:
        tv = train_metrics.get(metric)
        rv = replay.get(metric)
        if tv is not None and rv is not None and tv > 0:
            divergence = abs(tv - rv) / max(abs(tv), 1e-6)
            entry = {"metric": metric, "training": round(tv, 4), "replay": round(rv, 4),
                     "divergence_pct": round(divergence * 100, 1)}
            if divergence > threshold:
                entry["issue"] = msg
                diagnosis["mismatches"].append(entry)
            else:
                diagnosis["healthy"].append(entry)

    # Time-of-day analysis
    tod = replay.get("time_of_day", {})
    if tod:
        power_hour = tod.get("power_hour", {})
        morning = tod.get("morning", {})
        if power_hour.get("pf", 999) < 0.8 and morning.get("pf", 0) > 1.5:
            diagnosis["warnings"].append({
                "type": "power_hour_bleed",
                "detail": f"Morning PF={morning.get('pf')}, Power hour PF={power_hour.get('pf')}",
                "action": "Model bleeds in power hour — check time-of-day features and theta awareness",
            })
        for bucket, data in tod.items():
            if data.get("pf", 999) < 0.5 and data.get("trades", 0) > 5:
                diagnosis["warnings"].append({
                    "type": f"weak_{bucket}",
                    "detail": f"{bucket}: PF={data['pf']}, {data['trades']} trades",
                    "action": f"Consistently losing in {bucket} — consider time-based trade filtering",
                })

    # Experiment pattern analysis
    if exps:
        # Tunnel vision detection
        changes = [e.get("change_summary", "") for e in exps if e.get("change_summary")]
        if len(changes) >= 3:
            # Check if >50% of experiments share a common pattern
            from collections import Counter
            keywords = []
            for c in changes:
                if "STOPPED_PNL" in c.upper():
                    keywords.append("STOPPED_PNL")
                elif "class " in c:
                    keywords.append("NEW_MODULE")
                elif "bias" in c.lower():
                    keywords.append("BIAS_TUNING")
                elif "_env_float" in c:
                    keywords.append("HYPERPARAM")
                else:
                    keywords.append("OTHER")
            counts = Counter(keywords)
            dominant = counts.most_common(1)[0]
            if dominant[1] >= len(changes) * 0.5 and dominant[1] >= 3:
                diagnosis["warnings"].append({
                    "type": "tunnel_vision",
                    "detail": f"Inner loop fixated on {dominant[0]} ({dominant[1]}/{len(changes)} experiments)",
                    "action": "Update lab_notebook dead ends and redirect priorities",
                })

        # Score trend
        scores = [e.get("score", -999) for e in exps if e.get("score", -999) > -999]
        if len(scores) >= 3:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            if sum(second_half)/len(second_half) > sum(first_half)/len(first_half) * 1.1:
                diagnosis["healthy"].append({"type": "score_trend", "detail": "Scores improving over experiments"})
            elif sum(second_half)/len(second_half) < sum(first_half)/len(first_half) * 0.8:
                diagnosis["warnings"].append({
                    "type": "score_declining",
                    "detail": "Scores declining over experiments",
                    "action": "Inner loop may be overcomplicating — consider simpler approaches",
                })

    # Overall recommendation
    if diagnosis["mismatches"]:
        diagnosis["recommendation"] = {
            "action": "FIX_MISMATCH",
            "priority": "P0",
            "detail": "Training/eval mismatch detected. Fix before more training.",
            "mismatches": [m["metric"] for m in diagnosis["mismatches"]],
        }
    elif any(w["type"] == "tunnel_vision" for w in diagnosis["warnings"]):
        diagnosis["recommendation"] = {
            "action": "REDIRECT_INNER_LOOP",
            "priority": "P1",
            "detail": "Inner loop is tunnel-visioning. Update lab_notebook and program.md.",
        }
    elif any(w["type"].startswith("power_hour") or w["type"].startswith("weak_") for w in diagnosis["warnings"]):
        diagnosis["recommendation"] = {
            "action": "CHECK_TIME_FEATURES",
            "priority": "P1",
            "detail": "Time-of-day weakness detected. Check feature coverage and loss weighting.",
        }
    elif not diagnosis["mismatches"] and not diagnosis["warnings"]:
        diagnosis["recommendation"] = {
            "action": "LET_IT_COOK",
            "priority": "P3",
            "detail": "No mismatches or warnings. Continue training with longer session.",
        }
    else:
        diagnosis["recommendation"] = {
            "action": "STEER_AND_CONTINUE",
            "priority": "P2",
            "detail": "Minor issues. Adjust lab_notebook and continue.",
        }

    # Save and print
    write_json(cdir / "diagnosis.json", diagnosis)

    log(f"\nDiagnosis Summary:")
    log(f"  Mismatches: {len(diagnosis['mismatches'])}")
    log(f"  Warnings:   {len(diagnosis['warnings'])}")
    log(f"  Healthy:    {len(diagnosis['healthy'])}")
    log(f"  Recommendation: {diagnosis['recommendation']['action']} ({diagnosis['recommendation']['priority']})")
    if diagnosis["mismatches"]:
        for m in diagnosis["mismatches"]:
            log(f"    ✗ {m['metric']}: train={m['training']} replay={m['replay']} ({m['divergence_pct']}% divergence)")
    if diagnosis["warnings"]:
        for w in diagnosis["warnings"]:
            log(f"    ⚠ {w['type']}: {w['detail']}")

    state["phase"] = "diagnosed"
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# RESEARCH — deep analysis of trade data against domain knowledge
# ---------------------------------------------------------------------------

def cmd_research(args: argparse.Namespace) -> bool:
    """Cross-reference replay trade data against domain knowledge.

    Produces structured findings with hypotheses grounded in 0DTE mechanics.
    This is the "WHY" that the ANALYZE phase's "WHAT" lacks.
    """
    state = load_state()
    cycle_num = state.get("cycle", 0) or 1
    cdir = cycle_dir(cycle_num)

    # Load trade CSV
    trade_csv = cdir / "replay" / "backtest_trades.csv"
    if not trade_csv.exists():
        log("No backtest_trades.csv — skipping research phase")
        return False

    trades = []
    try:
        with open(trade_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(row)
    except Exception as e:
        log(f"Failed to read trade CSV: {e}")
        return False

    if len(trades) < 5:
        log(f"Only {len(trades)} trades — too few for meaningful research")
        return False

    log(f"=== RESEARCH: Analyzing {len(trades)} trades against domain knowledge ===")

    findings = []
    research = {"num_trades": len(trades), "findings": []}

    # --- Helper: safe float parse ---
    def _f(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except (ValueError, TypeError):
            return default

    def _hour(row):
        """Extract entry hour from entry_time (HH:MM format)."""
        entry = row.get("entry_time", "")
        hour_str = entry.split(":")[0][-2:] if entry else ""
        try:
            return int(hour_str)
        except ValueError:
            return -1

    # --- 1. Time-of-Day P&L Decomposition ---
    tod = {
        "morning (9:35-10:30)": [],
        "midday (10:30-13:30)": [],
        "afternoon (13:30-15:30)": [],
        "power_hour (15:30-16:00)": [],
    }
    for t in trades:
        h = _hour(t)
        m_str = t.get("entry_time", "").split(":")
        minute = int(m_str[1]) if len(m_str) > 1 else 0
        pnl = _f(t, "pnl_pct")
        time_val = h * 60 + minute
        if time_val < 630:  # before 10:30
            tod["morning (9:35-10:30)"].append(pnl)
        elif time_val < 810:  # before 13:30
            tod["midday (10:30-13:30)"].append(pnl)
        elif time_val < 930:  # before 15:30
            tod["afternoon (13:30-15:30)"].append(pnl)
        else:
            tod["power_hour (15:30-16:00)"].append(pnl)

    tod_report = []
    for period, pnls in tod.items():
        if pnls:
            w = [p for p in pnls if p > 0]
            l = [p for p in pnls if p <= 0]
            pf = sum(w) / max(abs(sum(l)), 1e-6) if l else float("inf")
            avg = sum(pnls) / len(pnls)
            wr = len(w) / len(pnls) * 100
            tod_report.append({"period": period, "trades": len(pnls),
                               "pf": round(pf, 2), "avg_pnl": round(avg, 2),
                               "win_rate": round(wr, 1)})

    research["time_of_day"] = tod_report

    # Generate time-of-day finding
    morning_pnl = sum(tod["morning (9:35-10:30)"])
    afternoon_pnl = sum(tod["afternoon (13:30-15:30)"]) + sum(tod["power_hour (15:30-16:00)"])
    if afternoon_pnl < 0 and morning_pnl > 0:
        findings.append({
            "category": "time_of_day",
            "finding": f"Morning trades net +{morning_pnl:.1f}% but afternoon trades net {afternoon_pnl:.1f}% — model bleeds in the afternoon",
            "domain_knowledge": "Theta follows 1/sqrt(T): ~$2/hr at open → $30+/hr by 3:30pm. Long options bleed fastest in final 2 hours. Gamma spikes 5-8x from morning to 3pm creating extreme whipsaw.",
            "hypothesis": "Model needs tighter dynamic stops or exit bias after 2pm. Consider sample weighting that emphasizes afternoon exit quality during training.",
        })
    elif all(len(v) == 0 for k, v in tod.items() if "afternoon" in k or "power" in k):
        findings.append({
            "category": "time_of_day",
            "finding": "No afternoon trades — model avoids late session entirely",
            "domain_knowledge": "Charm flows (afternoon) create predictable dealer unwinds. SPX gravitates toward max pain / high-OI strikes after 1:30pm.",
            "hypothesis": "Model may be over-penalizing afternoon entries. Could be missing charm flow trades.",
        })

    # --- 2. Stop-Loss Clustering ---
    stops = [t for t in trades if "stop" in t.get("reason", "").lower()]
    non_stops = [t for t in trades if "stop" not in t.get("reason", "").lower()]
    stop_rate = len(stops) / max(len(trades), 1)
    research["stop_analysis"] = {
        "total_stops": len(stops),
        "stop_rate": round(stop_rate, 3),
    }

    if stops:
        stop_holds = [int(t.get("bars_held", 0)) for t in stops]
        avg_stop_hold = sum(stop_holds) / len(stop_holds)
        early_stops = sum(1 for h in stop_holds if h <= 5)
        late_stops = sum(1 for h in stop_holds if h > 20)

        # Stop time-of-day
        stop_hours = {}
        for t in stops:
            h = _hour(t)
            stop_hours[h] = stop_hours.get(h, 0) + 1

        research["stop_analysis"].update({
            "avg_hold_before_stop": round(avg_stop_hold, 1),
            "early_stops_le5bars": early_stops,
            "late_stops_gt20bars": late_stops,
            "stop_by_hour": stop_hours,
        })

        if stop_rate > 0.35:
            peak_hour = max(stop_hours, key=stop_hours.get) if stop_hours else "?"
            findings.append({
                "category": "stop_loss",
                "finding": f"Stop rate {stop_rate:.0%} is high. {early_stops} early stops (≤5 bars), {late_stops} late stops (>20 bars). Peak hour: {peak_hour}:00",
                "domain_knowledge": "Gamma ATM: 0.02-0.04 morning → 0.10-0.20 by 3pm. High gamma means small SPX moves create large option P&L swings. Dynamic stop should widen with gamma.",
                "hypothesis": f"If stops cluster in afternoon (hour {peak_hour}), gamma spike is eating positions before the trade thesis plays out. If early stops dominate, entry quality is poor — model is entering at bad prices.",
            })

    # --- 3. Exit Quality ---
    exit_reasons = {}
    for t in trades:
        reason = t.get("reason", "unknown").lower()
        if "model" in reason or "gate" in reason:
            r = "model_exit"
        elif "stop" in reason:
            r = "dynamic_stop"
        elif "eod" in reason or "end" in reason:
            r = "end_of_day"
        elif "max_hold" in reason:
            r = "max_hold"
        else:
            r = reason
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "pnls": []}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnls"].append(_f(t, "pnl_pct"))

    exit_summary = {}
    for reason, data in exit_reasons.items():
        pnls = data["pnls"]
        exit_summary[reason] = {
            "count": data["count"],
            "pct": round(data["count"] / len(trades) * 100, 1),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 2),
            "total_pnl": round(sum(pnls), 2),
        }
    research["exit_quality"] = exit_summary

    model_exits = exit_reasons.get("model_exit", {"count": 0, "pnls": []})
    model_exit_rate = model_exits["count"] / max(len(trades), 1)
    if model_exit_rate < 0.3:
        findings.append({
            "category": "exit_quality",
            "finding": f"Only {model_exit_rate:.0%} of exits are model-driven. Most exits are forced (stop/EOD/max_hold).",
            "domain_knowledge": "Exits > Entries — the edge is in exit timing. 'Always take profits off the table' — the model must learn when to exit, not just when to enter.",
            "hypothesis": "Gate head is not learning to exit. EXIT_LOSS_WEIGHT may need increase, or exit label quality may need review.",
        })

    # --- 4. Strike Performance ---
    strike_perf = {}
    for t in trades:
        direction = t.get("direction", "")
        if "ATM" in direction:
            s = "ATM"
        elif "OTM5" in direction:
            s = "OTM5"
        elif "OTM10" in direction:
            s = "OTM10"
        else:
            s = direction
        if s not in strike_perf:
            strike_perf[s] = []
        strike_perf[s].append(_f(t, "pnl_pct"))

    strike_summary = {}
    for strike, pnls in strike_perf.items():
        w = [p for p in pnls if p > 0]
        l = [p for p in pnls if p <= 0]
        pf = sum(w) / max(abs(sum(l)), 1e-6) if l else float("inf")
        strike_summary[strike] = {
            "trades": len(pnls),
            "pf": round(pf, 2),
            "avg_pnl": round(sum(pnls) / max(len(pnls), 1), 2),
            "total_pnl": round(sum(pnls), 2),
        }
    research["strike_performance"] = strike_summary

    otm_total = sum(d["total_pnl"] for k, d in strike_summary.items() if "OTM" in k)
    atm_total = sum(d["total_pnl"] for k, d in strike_summary.items() if k == "ATM")
    if otm_total < -10:
        findings.append({
            "category": "strike_selection",
            "finding": f"OTM trades total P&L: {otm_total:.1f}% vs ATM: {atm_total:.1f}%",
            "domain_knowledge": "ATM preferred — OTM backtest cumulative -601%. ATM has highest gamma, most responsive to directional moves.",
            "hypothesis": "Model should bias toward ATM strikes. Direction head may need stronger ATM bias initialization.",
        })

    # --- 5. Direction Bias ---
    call_pnls = [_f(t, "pnl_pct") for t in trades if "CALL" in t.get("direction", "")]
    put_pnls = [_f(t, "pnl_pct") for t in trades if "PUT" in t.get("direction", "")]
    research["direction"] = {
        "calls": {"trades": len(call_pnls), "total_pnl": round(sum(call_pnls), 2),
                  "avg_pnl": round(sum(call_pnls) / max(len(call_pnls), 1), 2)},
        "puts": {"trades": len(put_pnls), "total_pnl": round(sum(put_pnls), 2),
                 "avg_pnl": round(sum(put_pnls) / max(len(put_pnls), 1), 2)},
    }

    # --- 6. Hold Time vs Profitability ---
    short_holds = [t for t in trades if int(t.get("bars_held", 0)) <= 3]
    long_holds = [t for t in trades if int(t.get("bars_held", 0)) > 30]
    if short_holds:
        short_avg = sum(_f(t, "pnl_pct") for t in short_holds) / len(short_holds)
        if short_avg < -2 and len(short_holds) > 5:
            findings.append({
                "category": "hold_time",
                "finding": f"{len(short_holds)} trades held ≤3 bars with avg P&L {short_avg:.1f}% — scalp-and-lose pattern",
                "domain_knowledge": "Short holds in 0DTE often mean the model exits immediately due to gamma whipsaw. Momentum needs time to develop (5-15 bars typical for a move).",
                "hypothesis": "Gate head may be too jittery — triggering exits on noise. Consider increasing gate label smoothing or adjusting exit labels.",
            })

    # --- 7. Consecutive Loss Analysis ---
    max_streak = 0
    current_streak = 0
    streak_details = []
    for t in trades:
        if _f(t, "pnl_pct") <= 0:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            if current_streak >= 3:
                streak_details.append({
                    "length": current_streak,
                    "date": t.get("date", ""),
                    "hour": _hour(t),
                })
            current_streak = 0
    research["loss_streaks"] = {
        "max_consecutive_losses": max_streak,
        "streaks_ge3": len(streak_details),
    }

    # --- Compile research report ---
    research["findings"] = findings

    # Save JSON
    write_json(cdir / "research.json", research)

    # Generate markdown report
    lines = [f"# Research Findings — Cycle {cycle_num:03d}",
             f"Analyzed {len(trades)} trades from replay backtest.\n"]

    # Time-of-day table
    lines.append("## Time-of-Day Performance")
    lines.append("| Period | Trades | PF | Avg P&L | Win Rate |")
    lines.append("|--------|--------|-----|---------|----------|")
    for r in tod_report:
        lines.append(f"| {r['period']} | {r['trades']} | {r['pf']} | {r['avg_pnl']}% | {r['win_rate']}% |")
    lines.append("")

    # Exit quality table
    lines.append("## Exit Quality")
    lines.append("| Reason | Count | % | Avg P&L | Total P&L |")
    lines.append("|--------|-------|---|---------|-----------|")
    for reason, data in sorted(exit_summary.items(), key=lambda x: -x[1]["count"]):
        lines.append(f"| {reason} | {data['count']} | {data['pct']}% | {data['avg_pnl']}% | {data['total_pnl']}% |")
    lines.append("")

    # Strike performance table
    lines.append("## Strike Performance")
    lines.append("| Strike | Trades | PF | Avg P&L | Total P&L |")
    lines.append("|--------|--------|-----|---------|-----------|")
    for strike, data in sorted(strike_summary.items()):
        lines.append(f"| {strike} | {data['trades']} | {data['pf']} | {data['avg_pnl']}% | {data['total_pnl']}% |")
    lines.append("")

    # Direction bias
    lines.append("## Direction Bias")
    for d in ["calls", "puts"]:
        data = research["direction"][d]
        lines.append(f"- **{d.title()}**: {data['trades']} trades, avg {data['avg_pnl']}%, total {data['total_pnl']}%")
    lines.append("")

    # Findings with hypotheses
    if findings:
        lines.append("## Key Findings & Hypotheses")
        for i, f in enumerate(findings, 1):
            lines.append(f"\n### Finding {i}: {f['category'].replace('_', ' ').title()}")
            lines.append(f"**Observation:** {f['finding']}")
            lines.append(f"**Domain Knowledge:** {f['domain_knowledge']}")
            lines.append(f"**Hypothesis:** {f['hypothesis']}")
    else:
        lines.append("## No significant anomalies detected.")

    lines.append("")

    # Write report
    report_path = cdir / "research.md"
    report_path.write_text("\n".join(lines))

    log(f"Research complete: {len(findings)} findings written to {report_path}")
    state["phase"] = "researched"
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# REPORT — single markdown briefing for Claude Code (Pro-optimized)
# ---------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> bool:
    """Generate a self-contained markdown briefing for Claude Code.

    This is the Pro-plan optimization: instead of Claude Code reading
    analysis.json + art2_notebook.md + state.json + multiple files,
    it reads ONE file (briefing.md) and can decide + act in a single message.
    """
    state = load_state()
    cycle_num = state.get("cycle", 0) or 1
    cdir = cycle_dir(cycle_num)

    # Run analyze first if needed
    analysis_path = cdir / "analysis.json"
    if not analysis_path.exists():
        cmd_analyze(args)

    analysis = read_json(analysis_path)
    if not analysis:
        log("No analysis data — run analyze first")
        return False

    # Read art2_notebook (canonical location: docs/art2-notebook.md)
    notebook_path = PROJECT_ROOT / "docs" / "art2-notebook.md"
    if not notebook_path.exists():
        notebook_path = ART2_DIR / "art2_notebook.md"  # fallback to legacy location
    notebook = notebook_path.read_text() if notebook_path.exists() else "(no notebook yet)"

    # Read lab_notebook
    lab_notebook_path = PROJECT_ROOT / "training" / "lab_notebook.md"
    lab_notebook = lab_notebook_path.read_text() if lab_notebook_path.exists() else "(not found)"

    # Build briefing
    lines = []
    lines.append(f"# ART² Cycle {cycle_num:03d} Briefing")
    lines.append(f"Generated: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}")
    lines.append("")

    # --- Section 1: Inner Loop Results ---
    lines.append("## 1. Inner Loop Results")
    lr = analysis.get("latest_run")
    if lr:
        lines.append(f"- **Run:** {lr.get('name', '?')}")
        lines.append(f"- **Experiments:** {lr.get('total_experiments', 0)}")
        lines.append(f"- **Kept:** {lr.get('kept_count', 0)}")
        lines.append(f"- **Best score:** {lr.get('best_score', '?')}")
        lines.append(f"- **Mean score:** {lr.get('mean_score', 0):.3f}" if lr.get('mean_score') else "")
        api_cost = lr.get("api_cost")
        if api_cost:
            if isinstance(api_cost, dict):
                lines.append(f"- **API cost:** ${api_cost.get('total_cost', 0):.2f} "
                             f"(cache hit: {api_cost.get('cache_hit_pct', 0):.0f}%)")
            elif isinstance(api_cost, (int, float)):
                lines.append(f"- **API cost:** ${api_cost:.2f}")

        # Monthly budget context
        tracker = _load_spend_tracker()
        remaining = _remaining_budget()
        lines.append(f"- **API budget:** ${tracker['total_spend']:.2f} / ${tracker['tier_budget']:.0f} "
                     f"({tracker['month']}, ${remaining:.2f} remaining)")

        # Top kept experiments
        exps = lr.get("experiments", [])
        kept_exps = [e for e in exps if e.get("kept", False)]
        if kept_exps:
            kept_exps.sort(key=lambda e: e.get("score", 0), reverse=True)
            lines.append("")
            lines.append("### Top Kept Experiments")
            lines.append("| # | Score | PF | TPD | Sharpe | SL% | Change |")
            lines.append("|---|-------|-----|-----|--------|-----|--------|")
            for e in kept_exps[:5]:
                lines.append(
                    f"| {e.get('experiment_id', '?')} "
                    f"| {e.get('score', 0):.3f} "
                    f"| {e.get('profit_factor', 0):.2f} "
                    f"| {e.get('trades_per_day', 0):.2f} "
                    f"| {e.get('trade_sharpe', 0):.2f} "
                    f"| {e.get('stop_loss_rate', 0):.1%} "
                    f"| {(e.get('change_summary', '') or '')[:60]} |"
                )

        # Recent failures (last 5 non-kept)
        failed_exps = [e for e in exps if not e.get("kept", False)]
        if failed_exps:
            lines.append("")
            lines.append("### Recent Failures")
            lines.append("| # | Score | Reason | Change |")
            lines.append("|---|-------|--------|--------|")
            for e in failed_exps[-5:]:
                reason = e.get("failure_type", "regression")
                lines.append(
                    f"| {e.get('experiment_id', '?')} "
                    f"| {e.get('score', 0):.3f} "
                    f"| {reason} "
                    f"| {(e.get('change_summary', '') or '')[:60]} |"
                )
    else:
        lines.append("- No run data found")

    lines.append("")

    # --- Section 2: Health Assessment ---
    lines.append("## 2. Health Assessment")
    health = analysis.get("inner_loop_health", {})
    lines.append(f"- **Status:** {health.get('status', 'unknown')}")
    lines.append(f"- **Accept rate:** {health.get('accept_rate', '?')}")
    if health.get("recommendation"):
        lines.append(f"- **Recommendation:** {health['recommendation']}")

    gaming = analysis.get("gaming_checks", [])
    if gaming:
        lines.append("")
        lines.append("### Gaming Alerts")
        for alert in gaming:
            lines.append(f"- {alert}")
    lines.append("")

    # --- Section 2.5: Research Findings ---
    research_path = cdir / "research.md"
    if research_path.exists():
        lines.append("## 2.5 Research Findings (Domain Knowledge Cross-Reference)")
        lines.append(research_path.read_text())
        lines.append("")

    # --- Section 3: Paper Trading (Ground Truth) ---
    lines.append("## 3. Paper Trading (Ground Truth)")
    paper = analysis.get("paper_trades", [])
    if paper:
        lines.append("| Date | Trades | P&L % | Win Rate |")
        lines.append("|------|--------|-------|----------|")
        for pt in paper[-10:]:
            lines.append(
                f"| {pt.get('date', '?')} "
                f"| {pt.get('trade_count', '?')} "
                f"| {pt.get('total_pnl_pct', 0):+.1f}% "
                f"| {pt.get('win_rate', 0):.0%} |"
                if pt.get('win_rate') is not None
                else f"| {pt.get('date', '?')} | {pt.get('trade_count', '?')} | ? | ? |"
            )
        total = sum(t.get("total_pnl_pct", 0) for t in paper)
        lines.append(f"\n**Cumulative paper P&L: {total:+.1f}%**")
    else:
        lines.append("- No paper trading data yet")
    lines.append("")

    # --- Section 4: Previous Cycle Decisions ---
    lines.append("## 4. Previous Decisions")
    changes_path = ART2_DIR / "changes.jsonl"
    if changes_path.exists():
        changes = _parse_experiments(changes_path)
        if changes:
            for c in changes[-3:]:
                lines.append(
                    f"- Cycle {c.get('cycle', '?')}: "
                    f"score {c.get('pre_score', '?')} → {c.get('post_score', '?')}, "
                    f"PF {c.get('pre_pf', '?')} → {c.get('post_pf', '?')}"
                )
                if c.get("gaming_alerts"):
                    for a in c["gaming_alerts"]:
                        lines.append(f"  - {a}")
        else:
            lines.append("- No previous cycles")
    else:
        lines.append("- No previous cycles")
    lines.append("")

    # --- Section 5: Current Inner Loop Guidance ---
    lines.append("## 5. Current Inner Loop Guidance")
    lines.append("### lab_notebook.md (injected into Sonnet's prompt)")
    # Just the key sections, not the whole thing
    for section in ["## What Fails", "## Best Runs", "## Next Priorities", "## What Works"]:
        idx = lab_notebook.find(section)
        if idx >= 0:
            # Extract section until next ## or end
            end_idx = lab_notebook.find("\n## ", idx + len(section))
            if end_idx < 0:
                end_idx = len(lab_notebook)
            chunk = lab_notebook[idx:end_idx].strip()
            lines.append(f"\n{chunk}")
    lines.append("")

    # --- Section 6: Recommended Action (automated decision tree) ---
    recommendation = compute_recommended_action(analysis)
    lines.append("## 6. RECOMMENDED ACTION")
    lines.append("")
    lines.append(f"**→ {recommendation['action']}) {recommendation['label']}** (confidence: {recommendation['confidence']})")
    lines.append(f"")
    lines.append(f"**Reason:** {recommendation['reason']}")
    lines.append("")
    lines.append("Decision tree path:")
    for step in recommendation["path"]:
        lines.append(f"  - {step}")
    lines.append("")
    lines.append("### All Options")
    lines.append("- **A) Let it cook** — Inner loop is making progress. No changes.")
    lines.append("- **B) Steer inner loop** — Edit `training/lab_notebook.md` (Next Priorities section)")
    lines.append("- **C) Change constraints** — Edit `training/program.md` (unlock/modify)")
    lines.append("- **D) Change features** — Edit `training/prepare.py` (add/remove features)")
    lines.append("- **E) Rebuild data** — Run `prepare.py` with new params")
    lines.append("- **F) Fix infrastructure** — Address deploy/API/data issues")
    lines.append("")
    lines.append("Write your decision to `results/art2/cycle-{:03d}/decision.md`".format(cycle_num))
    lines.append("Then run: `python3 tools/art2.py verify --dates 3`")
    lines.append("")

    # --- ART² Notebook (outer loop memory) ---
    lines.append("---")
    lines.append("## Appendix: ART² Notebook (outer loop memory)")
    lines.append(notebook)

    briefing_text = "\n".join(lines)
    briefing_path = cdir / "briefing.md"
    briefing_path.write_text(briefing_text)
    log(f"Briefing written to {briefing_path}")

    # Also print the briefing to stdout so Claude Code sees it directly
    print("\n" + "=" * 70)
    print(briefing_text)
    print("=" * 70 + "\n")

    state["phase"] = "awaiting_strategy"
    save_state(state)
    return True


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Print current ART² state."""
    state = load_state()
    print(json.dumps(state, indent=2))

    # Also show recent cycles
    for i in range(max(1, state.get("cycle", 0) - 4), state.get("cycle", 0) + 1):
        cdir = cycle_dir(i)
        if cdir.exists():
            analysis = read_json(cdir / "analysis.json")
            verify = read_json(cdir / "verify_results.json")
            decision = cdir / "decision.md"
            parts = [f"Cycle {i:03d}:"]
            if analysis and analysis.get("latest_run"):
                lr = analysis["latest_run"]
                parts.append(f"run={lr.get('name', '?')}")
                parts.append(f"exp={lr.get('total_experiments', '?')}")
                parts.append(f"best={lr.get('best_score', '?')}")
            if verify and verify.get("replay_battery"):
                rb = verify["replay_battery"]
                parts.append(f"battery={rb.get('pass_count', '?')}/{rb.get('total', '?')}")
            if decision.exists():
                parts.append("decision=YES")
            print("  " + " | ".join(parts))

    # API budget summary
    budget = getattr(args, "budget", DEFAULT_TIER_BUDGET)
    tracker = _load_spend_tracker(budget)
    remaining = _remaining_budget(budget)
    print(f"\n  API Budget ({tracker['month']}):")
    print(f"    Spent:     ${tracker['total_spend']:.2f} / ${tracker['tier_budget']:.0f}")
    print(f"    Remaining: ${remaining:.2f}")
    print(f"    Sessions:  {len(tracker['sessions'])}")
    est_remaining_exps = int(remaining / AVG_COST_PER_EXPERIMENT) if remaining > 0 else 0
    print(f"    Est. capacity: ~{est_remaining_exps} more experiments")


# ---------------------------------------------------------------------------
# INIT
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    """Initialize the ART² directory structure."""
    ART2_DIR.mkdir(parents=True, exist_ok=True)

    # Create art2_notebook.md if it doesn't exist (canonical: docs/art2-notebook.md)
    notebook_path = PROJECT_ROOT / "docs" / "art2-notebook.md"
    if not notebook_path.exists():
        notebook_path.write_text("""\
# ART² Lab Notebook — Outer Loop Memory

## System
ART² meta-loop wrapping autoresearch inner loop.
Outer loop (Opus) makes strategic changes; inner loop (Sonnet) optimizes train.py.

## Strategic Changes Tried
| Cycle | Change | OOS PF Before | OOS PF After | Verdict |
|-------|--------|---------------|--------------|---------|

## Paper Trading P&L Tracking
| Date | Trades | P&L % | Backtest Expected | Divergence |
|------|--------|-------|-------------------|------------|

## Dead Ends (Strategic Level)
| Change | Cycles | Result |
|--------|--------|--------|

## Current Hypothesis
- TBD: ART² just initialized. First cycle will establish baseline.

## What Works (Outer Loop)
- TBD: No strategic changes yet.
""")
        log(f"Created {notebook_path}")

    # Initialize state
    state = load_state()
    if state.get("cycle", 0) == 0:
        save_state(state)
        log(f"Initialized state.json")

    # Create changes.jsonl if needed
    changes_path = ART2_DIR / "changes.jsonl"
    if not changes_path.exists():
        changes_path.touch()
        log(f"Created {changes_path}")

    log(f"ART² initialized at {ART2_DIR}")


# ---------------------------------------------------------------------------
# CYCLE (full train → analyze → verify)
# ---------------------------------------------------------------------------

def cmd_cycle(args: argparse.Namespace) -> bool:
    """Run a full ART² cycle: train → analyze → replay → diagnose → report.

    The complete autonomous pipeline. Each phase produces structured output
    that feeds the next. After this completes, Claude Code (Opus) reads the
    diagnosis and report to make ONE strategic decision (fix + next session).
    """
    log("=== ART² CYCLE START ===")

    # Phase 1: Train
    if not args.skip_train:
        ok = cmd_train(args)
        if not ok and not args.dry_run:
            log("Training failed — analyzing existing results instead")
    else:
        log("Skipping training (--skip-train)")
        state = load_state()
        state["cycle"] = state.get("cycle", 0) + 1
        state["phase"] = "trained"
        save_state(state)

    # Phase 2: Analyze (collect experiment data)
    ok = cmd_analyze(args)
    if not ok:
        log("Analysis failed")
        return False

    # Phase 3: Replay (backtest best model)
    replay_ok = cmd_replay(args)
    if not replay_ok:
        log("Replay failed or no model — skipping diagnosis")

    # Phase 4: Diagnose (compare training vs replay)
    if replay_ok:
        cmd_diagnose(args)

    # Phase 4.5: Research (deep analysis against domain knowledge)
    if replay_ok:
        cmd_research(args)

    # Phase 5: Report (generate briefing for Claude Code)
    cmd_report(args)

    log("=== ART² CYCLE COMPLETE ===")
    log("Next: Read diagnosis + report, decide fix, launch next session.")
    return True


# ---------------------------------------------------------------------------
# AUTONOMOUS — market-aware master loop
# ---------------------------------------------------------------------------

def cmd_autonomous(args: argparse.Namespace) -> None:
    """Market-aware autonomous loop.

    Behavior changes based on market hours:
    - Market CLOSED: Run training cycles, validate IBKR compatibility after each
    - Market OPEN: Monitor paper trading health, do NOT train
    - Pre-market (<2h to open): Stop training, ensure model is ready

    This is the top-level command for hands-off ART² operation.
    """
    max_cycles = getattr(args, "max_cycles", 0) or 999
    minutes = getattr(args, "minutes", 130)
    cycles_run = 0

    log("=== ART² AUTONOMOUS MODE ===")

    # Initial market check
    market = is_market_open()
    log(f"Market status: {market['reason']}")
    if market.get("next_open"):
        log(f"  Next open: {market['next_open']}")
    if market.get("minutes_to_close"):
        log(f"  Minutes to close: {market['minutes_to_close']}")

    # Initial IBKR compatibility check
    log("Running IBKR compatibility gate...")
    gate = ibkr_compatibility_gate()
    for check in gate["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        log(f"  [{status}] {check['name']}: {check.get('detail', '')}")
    if not gate["passed"]:
        log("WARNING: IBKR compatibility gate FAILED — will train but flag for review")

    while cycles_run < max_cycles:
        market = is_market_open()

        if market["open"]:
            # === MARKET OPEN MODE ===
            log("Market is OPEN — monitoring mode (no training)")
            _monitor_paper_trading()
            # Re-check every 5 minutes during market hours
            time.sleep(300)
            continue

        # Check if market opens within 2 hours — don't start a long training run
        minutes_to_open = market.get("minutes_to_open", 9999)
        if 0 < minutes_to_open <= 120:
            log(f"Market opens in {minutes_to_open} min — too close to start training")
            log("Ensuring best model is deployed and ready for trading...")
            _pre_market_readiness_check()
            # Sleep until market open, then switch to monitoring
            time.sleep(min(minutes_to_open * 60, 300))
            continue

        # === MARKET CLOSED MODE — TRAIN ===
        # Auto-size training: don't exceed time until 2h before market open
        effective_minutes = minutes
        if minutes_to_open < 9999:
            max_safe_minutes = max(30, minutes_to_open - 120)  # 2h buffer
            if effective_minutes > max_safe_minutes:
                log(f"Capping training from {effective_minutes} to {max_safe_minutes} min (market opens in {minutes_to_open} min)")
                effective_minutes = max_safe_minutes

        log(f"=== TRAINING CYCLE {cycles_run + 1}/{max_cycles} ({effective_minutes} min) ===")

        # Create args-like object for cmd_cycle
        cycle_args = argparse.Namespace(
            minutes=effective_minutes,
            deposit_akt=getattr(args, "deposit_akt", None),
            skip_train=False,
            dry_run=getattr(args, "dry_run", False),
            dates=getattr(args, "dates", 3),
        )

        ok = cmd_cycle(cycle_args)
        cycles_run += 1

        if ok:
            # Post-cycle IBKR compatibility gate
            log("Post-cycle IBKR compatibility gate...")
            gate = ibkr_compatibility_gate()
            for check in gate["checks"]:
                status = "PASS" if check["passed"] else "FAIL"
                log(f"  [{status}] {check['name']}: {check.get('detail', '')}")

            if not gate["passed"]:
                log("IBKR GATE FAILED — model may break live trading!")
                log("Checking if previous model is available for rollback...")
                prev = PROJECT_ROOT / "training" / "best_model.pt.prev"
                if prev.exists():
                    log("Previous model available at best_model.pt.prev")
                    log("RECOMMENDATION: Rollback if gate failures are critical (model_inference, feature_parity)")
                    # Don't auto-rollback — let the next cycle's report flag it
                else:
                    log("No previous model available — continuing with current")

        log(f"Cycle {cycles_run} complete. Checking market status for next action...")
        # Brief pause between cycles
        time.sleep(30)

    log(f"=== AUTONOMOUS MODE COMPLETE ({cycles_run} cycles run) ===")


def _monitor_paper_trading() -> None:
    """Check paper trading health during market hours."""
    # Check if paper_live.py is running
    import subprocess as _sp
    try:
        result = _sp.run(
            ["pgrep", "-f", "paper_live.py"],
            capture_output=True, text=True, timeout=5,
        )
        pids = result.stdout.strip()
        if pids:
            log(f"paper_live.py is running (PIDs: {pids.replace(chr(10), ', ')})")
        else:
            log("WARNING: paper_live.py is NOT running during market hours!")
            log("  Start with: python3 tools/paper_live.py")
    except Exception as e:
        log(f"Could not check paper_live.py status: {e}")

    # Check latest audit trail
    live_dir = RESULTS_DIR / "live"
    if live_dir.exists():
        today_str = datetime.now(ET).strftime("%Y-%m-%d")
        audit = live_dir / f"audit-{today_str}.jsonl"
        if audit.exists():
            size = audit.stat().st_size
            # Count lines (trades/events)
            try:
                with open(audit) as f:
                    n_events = sum(1 for _ in f)
                log(f"Today's audit trail: {n_events} events ({size:,} bytes)")
            except Exception:
                log(f"Today's audit trail: {size:,} bytes")
        else:
            log(f"No audit trail yet for {today_str}")


def _pre_market_readiness_check() -> None:
    """Ensure model is ready before market opens."""
    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    if not model_path.exists():
        log("CRITICAL: No best_model.pt — cannot trade!")
        return

    gate = ibkr_compatibility_gate()
    all_passed = all(c["passed"] for c in gate["checks"])
    if all_passed:
        log("Pre-market readiness: ALL CHECKS PASSED")
    else:
        failed = [c["name"] for c in gate["checks"] if not c["passed"]]
        log(f"Pre-market readiness: FAILED checks: {', '.join(failed)}")
        if "ibkr_probe" in failed:
            log("  IBKR not connected — ensure IB Gateway is running on port 4002")


# ---------------------------------------------------------------------------
# DAEMON — persistent autonomous loop with Opus strategic decisions
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _sigterm_handler(signum, frame):
    """Handle SIGTERM gracefully — finish current phase, then exit."""
    global _shutdown_requested
    _shutdown_requested = True
    log("SIGTERM received — will exit after current phase completes")


def _check_pid_lock() -> None:
    """Ensure no other daemon is running. Exit if duplicate."""
    if PID_PATH.exists():
        try:
            old_pid = int(PID_PATH.read_text().strip())
            # Check if process is alive
            os.kill(old_pid, 0)
            log(f"Another daemon is already running (PID {old_pid}). Exiting.")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # PID is stale
            PID_PATH.unlink(missing_ok=True)
        except PermissionError:
            # Process exists but we can't signal it — it's alive
            log(f"Another daemon is already running (PID). Exiting.")
            sys.exit(1)
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))


def _remove_pid_lock() -> None:
    PID_PATH.unlink(missing_ok=True)


def _check_sentinels() -> str:
    """Return 'stop', 'paused', 'review', or 'ok'."""
    if STOP_SENTINEL.exists():
        return "stop"
    if PAUSE_SENTINEL.exists():
        return "paused"
    if REVIEW_SENTINEL.exists():
        return "review"
    return "ok"


def _pause(reason: str) -> None:
    """Create PAUSED sentinel with reason."""
    PAUSE_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_SENTINEL.write_text(f"{reason}\n{datetime.now(ET).isoformat()}\n")
    log(f"PAUSED: {reason}")


def _write_heartbeat(phase: str, cycle: int, budget_remaining: float) -> None:
    """Write heartbeat JSON for external monitoring."""
    data = {
        "pid": os.getpid(),
        "timestamp": datetime.now(ET).isoformat(),
        "phase": phase,
        "cycle": cycle,
        "budget_remaining": round(budget_remaining, 2),
    }
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEARTBEAT_PATH.write_text(json.dumps(data, indent=2))


def _alert(title: str, message: str, critical: bool = False) -> None:
    """Send macOS notification and append to alerts log."""
    log(f"ALERT: {title} — {message}")
    # macOS notification
    sound = ' sound name "Basso"' if critical else ""
    script = f'display notification "{message}" with title "ART²: {title}"{sound}'
    try:
        subprocess.run(["osascript", "-e", script], timeout=5,
                       capture_output=True)
    except Exception:
        pass
    # Append to alerts log
    entry = {
        "timestamp": datetime.now(ET).isoformat(),
        "title": title,
        "message": message,
        "critical": critical,
    }
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _resolve_claude_bin() -> list[str] | None:
    """Find Claude Code CLI. Returns command prefix or None."""
    # Check for direct 'claude' binary
    try:
        result = subprocess.run(["which", "claude"], capture_output=True,
                                text=True, timeout=5)
        if result.returncode == 0:
            path = result.stdout.strip()
            if path:
                log(f"Found claude CLI at: {path}")
                return [path]
    except Exception:
        pass
    # Check npx
    try:
        result = subprocess.run(
            ["npx", "@anthropic-ai/claude-code", "--version"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            ver = result.stdout.strip()
            log(f"Found claude via npx (version {ver})")
            return ["npx", "@anthropic-ai/claude-code"]
    except Exception:
        pass
    log("WARNING: Claude Code CLI not found — daemon will run in auto-only mode")
    return None


def _daily_spend() -> float:
    """Get total API spend for today."""
    tracker = _load_spend_tracker()
    today_str = datetime.now().strftime("%Y-%m-%d")
    return sum(
        s.get("cost", 0)
        for s in tracker.get("sessions", [])
        if s.get("timestamp", "").startswith(today_str)
    )


# NOTE: The strategist prompt is now loaded from docs/art2-opus-system-prompt.md
# by _invoke_opus(). See that function for the full prompt construction.


def _invoke_opus(briefing_path: Path, cycle_dir_path: Path,
                 claude_bin: list[str]) -> dict | None:
    """Invoke Claude Code CLI for strategic decision. Returns parsed decision or None."""
    if not briefing_path.exists():
        log(f"No briefing at {briefing_path}")
        return None

    briefing = briefing_path.read_text()

    # Load persistent system prompt for Opus strategic decisions
    system_prompt_path = PROJECT_ROOT / "docs" / "art2-opus-system-prompt.md"
    system_prompt = ""
    if system_prompt_path.exists():
        system_prompt = system_prompt_path.read_text() + "\n\n"
    else:
        log(f"WARNING: {system_prompt_path} not found — Opus will run without system prompt")

    # Load domain knowledge files so Opus can make informed decisions
    domain_knowledge = ""
    for dk_file in ["0dte-domain-knowledge.md", "pickles-trading-knowledge.md"]:
        dk_path = PROJECT_ROOT / "docs" / dk_file
        if dk_path.exists():
            domain_knowledge += f"\n\n--- {dk_file} ---\n{dk_path.read_text()}\n"
        else:
            log(f"WARNING: {dk_path} not found — Opus will lack domain knowledge")

    prompt = (
        f"{system_prompt}"
        f"\n## DOMAIN KNOWLEDGE\n{domain_knowledge}\n\n"
        "Read this ART² training cycle briefing and make a strategic decision.\n\n"
        f"--- BRIEFING ---\n{briefing}\n--- END BRIEFING ---\n\n"
        "Respond with ONLY a JSON object. No markdown, no explanation outside the JSON."
    )

    cmd = [
        *claude_bin,
        "-p",
        "--model", "opus",
        "--output-format", "json",
    ]

    log("Invoking Claude Code (Opus) for strategic decision...")
    try:
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=OPUS_TIMEOUT, cwd=str(PROJECT_ROOT),
        )
        raw = result.stdout.strip()
        # Save raw response for debugging
        debug_path = cycle_dir_path / "opus_response.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(f"Exit code: {result.returncode}\n\nSTDOUT:\n{raw}\n\nSTDERR:\n{result.stderr}\n")

        if result.returncode != 0:
            log(f"Claude Code exited with code {result.returncode}")
            return None

        # --output-format json wraps response; extract the result text
        try:
            wrapper = json.loads(raw)
            if isinstance(wrapper, dict) and "result" in wrapper:
                raw = wrapper["result"]
            elif isinstance(wrapper, list):
                # Extract text from content blocks
                for block in wrapper:
                    if isinstance(block, dict) and block.get("type") == "text":
                        raw = block["text"]
                        break
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from the response
        # Handle cases where the model wraps JSON in markdown code blocks
        if "```" in raw:
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
            if json_match:
                raw = json_match.group(1).strip()

        decision = json.loads(raw)
        log(f"Opus decision: action={decision.get('action')}, "
            f"rationale={decision.get('rationale', '')[:80]}")
        return decision

    except subprocess.TimeoutExpired:
        log(f"Claude Code timed out after {OPUS_TIMEOUT}s")
        return None
    except json.JSONDecodeError as e:
        log(f"Failed to parse Opus response as JSON: {e}")
        log(f"Raw response (first 200 chars): {raw[:200]}")
        return None
    except Exception as e:
        log(f"Claude Code invocation failed: {e}")
        return None


def _write_chronicle_entry(cycle_num: int, entry_text: str, action: str) -> None:
    """Prepend a dated chronicle entry to docs/project-chronicle.md."""
    chronicle_path = PROJECT_ROOT / "docs" / "project-chronicle.md"
    today = datetime.now(ET).strftime("%Y-%m-%d")
    action_labels = {
        "A": "Let It Cook", "B": "Steering the Inner Loop",
        "C": "Changing Constraints", "D": "Feature Engineering",
        "E": "Data Rebuild", "F": "Infrastructure Fix",
        "G": "Direct Architecture Change",
    }
    title = action_labels.get(action, f"Action {action}")
    header = f"## {today} — Cycle {cycle_num:03d}: {title}"
    new_entry = f"{header}\n\n{entry_text.strip()}\n\n---\n"

    if chronicle_path.exists():
        content = chronicle_path.read_text()
        # Insert after the header block (after first ---)
        marker = "---\n"
        first_sep = content.find(marker)
        if first_sep >= 0:
            insert_at = first_sep + len(marker) + 1  # after the --- and newline
            content = content[:insert_at] + "\n" + new_entry + "\n" + content[insert_at:]
        else:
            content = new_entry + "\n" + content
        chronicle_path.write_text(content)
    else:
        chronicle_path.write_text(
            "# ART² Project Chronicle\n\n"
            "> A human-readable record of what's happening. Most recent first.\n\n---\n\n"
            + new_entry
        )
    log(f"Chronicle updated: {header.strip()[:60]}")


def _write_review(cycle_num: int, decision: dict, chronicle_entry: str) -> None:
    """Write review.md for human review and create REVIEW sentinel."""
    cdir = cycle_dir(cycle_num)
    action = decision.get("action", "A")
    rationale = decision.get("rationale", "")
    changes = decision.get("changes", [])
    minutes = decision.get("next_session_minutes", decision.get("training_minutes", 130))

    review_text = (
        f"# Cycle {cycle_num} — Review Before Next Run\n\n"
        f"**Action:** {action}\n"
        f"**Next training:** {minutes} minutes\n"
        f"**Timestamp:** {datetime.now(ET).isoformat()}\n\n"
        f"---\n\n"
        f"## What Happened (Human Summary)\n\n"
        f"{chronicle_entry.strip()}\n\n"
        f"---\n\n"
        f"## Strategic Decision\n\n"
        f"{rationale}\n\n"
    )
    if changes:
        review_text += "**Changes made:**\n" + "\n".join(f"- {c}" for c in changes) + "\n\n"

    fresh = decision.get("fresh_start", False)
    rebuild = decision.get("rebuild_data", False)
    if fresh or rebuild:
        review_text += "**Flags:**\n"
        if fresh:
            review_text += "- Fresh start (best_model.pt backed up)\n"
        if rebuild:
            review_text += "- Data rebuild (prepare.py will run)\n"
        review_text += "\n"

    review_text += (
        "---\n\n"
        "## What Happens Next\n\n"
        f"The daemon will launch a **{minutes}-minute training session** on Akash H100.\n\n"
        "**Your options:**\n"
        "- **Approve:** Remove `results/art2/REVIEW` to continue\n"
        "- **Redirect:** Edit `training/lab_notebook.md`, `training/program.md`, or `training/train.py` first, then remove REVIEW\n"
        "- **Stop:** Create `results/art2/STOP` to halt the daemon\n"
    )

    (cdir / "review.md").write_text(review_text)
    REVIEW_SENTINEL.parent.mkdir(parents=True, exist_ok=True)
    REVIEW_SENTINEL.write_text(f"cycle_{cycle_num}\n{datetime.now(ET).isoformat()}\n")
    _alert("Review Ready", f"Cycle {cycle_num} decision ready for review", critical=False)
    log(f"REVIEW gate: waiting for human to remove {REVIEW_SENTINEL}")


def _auto_decide(cycle_num: int, rec: dict, analysis: dict) -> int:
    """Auto-decide for high-confidence action A. Returns next training minutes."""
    cdir = cycle_dir(cycle_num)
    cdir.mkdir(parents=True, exist_ok=True)

    lr = analysis.get("latest_run", {}) or {}
    best_score = lr.get("best_score", 0)

    decision_text = (
        f"# Cycle {cycle_num} Decision\n\n"
        f"**Action:** A — Let it cook\n"
        f"**Decided by:** Auto (high-confidence)\n"
        f"**Rationale:** {rec.get('reason', 'Inner loop productive')}\n"
        f"**Decision path:** {' → '.join(rec.get('path', []))}\n"
        f"**Best score:** {best_score}\n"
        f"**Timestamp:** {datetime.now(ET).isoformat()}\n"
    )
    (cdir / "decision.md").write_text(decision_text)

    # Chronicle entry for auto-decide
    lr = analysis.get("latest_run", {}) or {}
    total = lr.get("total_experiments", 0)
    kept = lr.get("kept_experiments", 0)
    chronicle = (
        f"The inner loop is making progress — {kept} of {total} experiments were kept "
        f"with a best score of {best_score:.2f}. No strategic changes needed this cycle. "
        f"Letting the model continue exploring on its current trajectory."
    )
    _write_chronicle_entry(cycle_num, chronicle, "A")

    # Review gate
    _write_review(cycle_num, {
        "action": "A", "rationale": rec.get("reason", "Inner loop productive"),
        "changes": [], "next_session_minutes": 130,
    }, chronicle)

    state = load_state()
    state["phase"] = "decided"
    state["last_action"] = "A"
    state["last_decision_by"] = "auto"
    save_state(state)

    log(f"Auto-decided: A (let it cook) — score={best_score}")
    # Use 130 if improving, 90 if stable
    return 130


def _apply_opus_decision(decision: dict, cycle_num: int) -> int:
    """Apply Opus's strategic decision. Returns next training minutes."""
    import re

    cdir = cycle_dir(cycle_num)
    cdir.mkdir(parents=True, exist_ok=True)

    action = decision.get("action", "A")
    rationale = decision.get("rationale", "")
    repair_desc = decision.get("repair_description", "")
    changes = decision.get("changes", [])
    file_edits = decision.get("file_edits", [])
    minutes = decision.get("next_session_minutes", decision.get("training_minutes", 130))

    # Warn if repair_description missing for code-change actions
    if action not in ("A",) and not repair_desc:
        log(f"WARNING: Opus action {action} missing repair_description — violates repair-over-workaround policy")

    decision_text = (
        f"# Cycle {cycle_num} Decision\n\n"
        f"**Action:** {action}\n"
        f"**Decided by:** Claude Code (Opus)\n"
        f"**Rationale:** {rationale}\n"
        f"**Training minutes:** {minutes}\n"
        f"**Timestamp:** {datetime.now(ET).isoformat()}\n"
    )

    if repair_desc:
        decision_text += f"\n## Repair Description\n\n{repair_desc}\n"
    if changes:
        decision_text += f"\n## Changes\n\n" + "\n".join(f"- {c}" for c in changes) + "\n"

    # --- Apply lab_notebook_edit (legacy action B support) ---
    if action == "B" and decision.get("lab_notebook_edit"):
        nb_path = PROJECT_ROOT / "training" / "lab_notebook.md"
        if nb_path.exists():
            content = nb_path.read_text()
            new_priorities = decision["lab_notebook_edit"]
            pattern = r'(## Next Priorities\n).*?(?=\n## |\Z)'
            replacement = f'\\1{new_priorities}\n'
            new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            if new_content != content:
                nb_path.write_text(new_content)
                log(f"Updated lab_notebook.md Next Priorities")
                decision_text += f"\n**Lab notebook edit:**\n{new_priorities}\n"
            else:
                log("WARNING: Could not find ## Next Priorities section to update")

    # --- Apply file_edits + doc_edits (full autonomy for all actions) ---
    ALLOWED_EDIT_PATHS = {
        "training/program.md", "training/lab_notebook.md",
        "training/train.py", "training/prepare.py",
        "training/replay.py", "training/run_loop.py",
        "docs/art2-opus-system-prompt.md", "docs/art2-notebook.md",
        "docs/ARCHITECTURE.md", "docs/art2.md", "docs/CLAUDE.md",
        "docs/daily-pipeline.md", ".claude/rules/art2-operating-manual.md",
        "tools/art2.py",
    }

    # Merge doc_edits into file_edits for unified processing
    doc_edits = decision.get("doc_edits", [])
    file_edits = file_edits + doc_edits

    if file_edits:
        decision_text += "\n## File Edits Applied\n\n"
        for edit in file_edits:
            edit_path = edit.get("path", "")
            operation = edit.get("operation", "")

            # Safety: only allow edits to whitelisted paths
            if edit_path not in ALLOWED_EDIT_PATHS:
                log(f"REJECTED edit to {edit_path} — not in allowed paths")
                decision_text += f"- **REJECTED** `{edit_path}` (not in whitelist)\n"
                continue

            target = PROJECT_ROOT / edit_path
            if not target.parent.exists():
                log(f"REJECTED edit to {edit_path} — parent directory does not exist")
                continue

            try:
                if operation == "replace_section":
                    section = edit.get("section", "")
                    new_content = edit.get("new_content", "")
                    if not section or not target.exists():
                        log(f"REJECTED replace_section on {edit_path} — no section or file missing")
                        continue
                    content = target.read_text()
                    idx = content.find(section)
                    if idx < 0:
                        log(f"WARNING: Section '{section}' not found in {edit_path}")
                        decision_text += f"- **FAILED** `{edit_path}`: section not found\n"
                        continue
                    # Find next section header (## at same or higher level) or EOF
                    section_level = len(section) - len(section.lstrip("#"))
                    rest = content[idx + len(section):]
                    next_header = re.search(r'\n#{1,' + str(section_level) + r'} ', rest)
                    if next_header:
                        end = idx + len(section) + next_header.start()
                    else:
                        end = len(content)
                    new_file = content[:idx] + section + "\n" + new_content + "\n" + content[end:]
                    target.write_text(new_file)
                    log(f"Applied replace_section to {edit_path}: {section.strip()[:50]}")
                    decision_text += f"- `{edit_path}`: replaced section `{section.strip()[:50]}`\n"

                elif operation == "write_full":
                    new_content = edit.get("new_content", "")
                    target.write_text(new_content)
                    log(f"Wrote full file: {edit_path}")
                    decision_text += f"- `{edit_path}`: full rewrite ({len(new_content)} chars)\n"

                elif operation == "append":
                    new_content = edit.get("new_content", "")
                    with open(target, "a") as f:
                        f.write("\n" + new_content)
                    log(f"Appended to {edit_path}")
                    decision_text += f"- `{edit_path}`: appended {len(new_content)} chars\n"

                else:
                    log(f"Unknown edit operation '{operation}' for {edit_path}")
                    decision_text += f"- **UNKNOWN OP** `{edit_path}`: {operation}\n"

            except Exception as e:
                log(f"ERROR applying edit to {edit_path}: {e}")
                decision_text += f"- **ERROR** `{edit_path}`: {e}\n"

    # --- Handle rebuild_data and fresh_start flags ---
    if decision.get("rebuild_data"):
        log("Opus requested data rebuild — running prepare.py")
        decision_text += "\n**Data rebuild requested.**\n"
        rc, _ = run_cmd(
            ["python3", str(PROJECT_ROOT / "training" / "prepare.py")],
            timeout=600,
        )
        if rc != 0:
            log("WARNING: prepare.py failed during data rebuild")

    if decision.get("fresh_start"):
        model_path = PROJECT_ROOT / "training" / "best_model.pt"
        if model_path.exists():
            import shutil
            bak = str(model_path) + ".bak"
            shutil.move(str(model_path), bak)
            log(f"Fresh start: moved best_model.pt to {bak}")
            decision_text += "\n**Fresh start: best_model.pt moved to .bak**\n"

    (cdir / "decision.md").write_text(decision_text)

    # Chronicle entry from Opus (or auto-generate if missing)
    chronicle = decision.get("chronicle_entry", "")
    if not chronicle:
        chronicle = f"Action {action}: {rationale[:200]}"
        log("WARNING: Opus response missing chronicle_entry — auto-generating from rationale")
    _write_chronicle_entry(cycle_num, chronicle, action)

    # Review gate — human must approve before next training run
    _write_review(cycle_num, decision, chronicle)

    state = load_state()
    state["phase"] = "decided"
    state["last_action"] = action
    state["last_decision_by"] = "opus"
    save_state(state)

    log(f"Applied Opus decision: {action} — {rationale[:80]}")
    return minutes


def cmd_daemon(args: argparse.Namespace) -> None:
    """Persistent autonomous loop with strategic decision-making.

    High-confidence action A → auto-decide (no Opus needed).
    Medium-confidence A or action B → invoke Claude Code (Opus) via CLI.
    Action F or Opus defers → pause and alert user.

    Control:
        touch results/art2/STOP      # graceful exit after current phase
        rm results/art2/PAUSED       # resume after pause
        rm results/art2/REVIEW       # approve decision and continue to next training run
    """
    global _shutdown_requested

    _check_pid_lock()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    claude_bin = None
    if not getattr(args, "auto_only", False):
        claude_bin = _resolve_claude_bin()

    max_cycles = getattr(args, "max_cycles", 50) or 50
    minutes = getattr(args, "minutes", 130)
    budget = getattr(args, "budget", DEFAULT_TIER_BUDGET)
    dry_run = getattr(args, "dry_run", False)
    consecutive_failures = 0
    cycles_completed = 0

    log("=== ART² DAEMON START ===")
    log(f"  Minutes/cycle: {minutes}")
    log(f"  Max cycles: {max_cycles}")
    log(f"  Budget: ${budget:.0f}/month")
    log(f"  Claude CLI: {'available' if claude_bin else 'NOT FOUND (auto-only mode)'}")
    log(f"  Dry run: {dry_run}")

    try:
        while not _shutdown_requested and cycles_completed < max_cycles:
            # 1. Check sentinels
            sentinel = _check_sentinels()
            if sentinel == "stop":
                log("STOP sentinel found. Exiting.")
                break
            if sentinel == "paused":
                log("PAUSED — waiting for user to remove PAUSED file")
                _write_heartbeat("paused", cycles_completed, _remaining_budget(budget))
                time.sleep(300)
                continue
            if sentinel == "review":
                log("REVIEW — decision ready for human review. Remove results/art2/REVIEW to continue.")
                _write_heartbeat("review", cycles_completed, _remaining_budget(budget))
                time.sleep(60)
                continue

            # 2. Budget checks
            remaining = _remaining_budget(budget)
            _write_heartbeat("budget_check", cycles_completed, remaining)

            if remaining < 20:
                _alert("Budget Exhausted", f"${remaining:.2f} remaining", critical=True)
                _pause("budget_exhausted")
                continue

            daily = _daily_spend()
            if daily >= MAX_DAILY_SPEND:
                _alert("Daily Spend Cap", f"${daily:.2f} spent today (cap: ${MAX_DAILY_SPEND})",
                       critical=False)
                _pause("daily_spend_cap")
                continue

            # Auto-size minutes to remaining budget
            projected = _project_session_cost(minutes)
            effective_minutes = minutes
            if remaining < projected * 1.5:
                effective_minutes = max(30, int(
                    remaining / AVG_COST_PER_EXPERIMENT
                    * AVG_MINUTES_PER_EXPERIMENT * 0.5
                ))
                log(f"Budget-constrained: {minutes} → {effective_minutes} min")

            # 3. Market check
            market = is_market_open()
            if market["open"]:
                log(f"Market OPEN — monitoring mode ({market['reason']})")
                _monitor_paper_trading()
                _write_heartbeat("monitoring", cycles_completed, remaining)
                time.sleep(300)
                continue

            minutes_to_open = market.get("minutes_to_open", 9999)
            if 0 < minutes_to_open <= 120:
                log(f"Market opens in {minutes_to_open} min — pre-market readiness check")
                _pre_market_readiness_check()
                _write_heartbeat("pre_market", cycles_completed, remaining)
                time.sleep(min(minutes_to_open * 60, 300))
                continue

            # Auto-size to market window
            if minutes_to_open < 9999:
                max_safe = max(30, minutes_to_open - 120)
                if effective_minutes > max_safe:
                    log(f"Market-constrained: {effective_minutes} → {max_safe} min")
                    effective_minutes = max_safe

            # 4. Run cycle
            log(f"=== DAEMON CYCLE {cycles_completed + 1}/{max_cycles} ({effective_minutes} min) ===")
            _write_heartbeat("training", cycles_completed, remaining)

            cycle_args = argparse.Namespace(
                minutes=effective_minutes,
                deposit_akt=getattr(args, "deposit_akt", None),
                skip_train=False,
                dry_run=dry_run,
                dates=3,
                budget=budget,
            )

            ok = cmd_cycle(cycle_args)

            if not ok:
                consecutive_failures += 1
                log(f"Cycle failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _alert("3 Consecutive Failures",
                           "Pausing daemon — check infrastructure",
                           critical=True)
                    _pause("consecutive_failures")
                continue

            consecutive_failures = 0

            # 5. Make strategic decision
            state = load_state()
            cycle_num = state.get("cycle", 0)
            cdir = cycle_dir(cycle_num)
            analysis = read_json(cdir / "analysis.json")

            if not analysis:
                log("No analysis.json — skipping decision")
                cycles_completed += 1
                continue

            rec = compute_recommended_action(analysis)
            log(f"Recommendation: {rec['action']} ({rec['label']}) — "
                f"confidence={rec['confidence']}")
            log(f"  Path: {' → '.join(rec.get('path', []))}")

            if rec["action"] == "A" and rec["confidence"] == "high":
                _auto_decide(cycle_num, rec, analysis)
            elif claude_bin:
                # ALL actions go through Opus — full autonomy
                decision = _invoke_opus(cdir / "briefing.md", cdir, claude_bin)
                if decision is None:
                    if rec["action"] == "A":
                        log("Opus failed — falling back to auto-decide for action A")
                        _auto_decide(cycle_num, rec, analysis)
                    else:
                        _alert("Opus Failed", f"Could not get strategic decision for action {rec['action']}",
                               critical=False)
                        _pause("opus_failed")
                        continue
                elif decision.get("needs_human"):
                    _alert("Opus Defers",
                           decision.get("rationale", "Needs human review"),
                           critical=False)
                    _pause("opus_defers")
                    continue
                else:
                    effective_minutes = _apply_opus_decision(decision, cycle_num)
            elif rec["action"] == "A":
                _auto_decide(cycle_num, rec, analysis)
            else:
                # No claude CLI available for non-A actions
                _alert(f"Action {rec['action']}: {rec['label']}",
                       f"No Claude CLI for autonomous execution. {rec.get('reason', '')}",
                       critical=(rec["action"] == "F"))
                _pause(f"action_{rec['action']}_no_cli")
                continue

            cycles_completed += 1
            _write_heartbeat("between_cycles", cycle_num, _remaining_budget(budget))
            log(f"Cycle {cycles_completed} complete. Sleeping 30s before next cycle...")
            time.sleep(30)

    finally:
        _remove_pid_lock()
        log(f"=== ART² DAEMON EXIT ({cycles_completed} cycles completed) ===")


# ---------------------------------------------------------------------------
# Helper: metrics & data collection
# ---------------------------------------------------------------------------

def _find_latest_run() -> Path | None:
    runs = sorted(RESULTS_DIR.glob("run-*/experiments.v2.jsonl"),
                  key=lambda p: p.parent.name, reverse=True)
    return runs[0].parent if runs else None


def _parse_experiments(jsonl_path: Path) -> list[dict]:
    experiments = []
    if not jsonl_path.exists():
        return experiments
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return experiments


def _parse_promoted_history() -> list[dict]:
    history_path = RESULTS_DIR / "promoted" / "history.jsonl"
    return _parse_experiments(history_path)


def _collect_paper_trades() -> list[dict]:
    """Collect paper trading summaries from results/live/."""
    live_dir = RESULTS_DIR / "live"
    if not live_dir.exists():
        return []

    summaries = []
    for summary_file in sorted(live_dir.glob("summary-*.txt")):
        date_str = summary_file.stem.replace("summary-", "")
        content = summary_file.read_text()
        entry = {"date": date_str, "raw": content}

        # Try to parse key metrics from summary
        for line in content.split("\n"):
            if "P&L" in line and "%" in line:
                try:
                    pct = float(line.split("%")[0].split()[-1])
                    entry["total_pnl_pct"] = pct
                except (ValueError, IndexError):
                    pass
            if "trades" in line.lower():
                try:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == "trades" and i > 0:
                            entry["trade_count"] = int(parts[i - 1])
                except (ValueError, IndexError):
                    pass

        # Also read trades CSV if available
        csv_path = live_dir / f"trades-{date_str}.csv"
        if csv_path.exists():
            try:
                with open(csv_path, newline="") as f:
                    reader = csv.DictReader(f)
                    trades = list(reader)
                    entry["trade_count"] = len(trades)
                    pnls = []
                    for t in trades:
                        try:
                            pnls.append(float(t.get("pnl_pct", 0)))
                        except (ValueError, TypeError):
                            pass
                    if pnls:
                        entry["total_pnl_pct"] = sum(pnls)
                        entry["win_rate"] = sum(1 for p in pnls if p > 0) / len(pnls) if pnls else 0
            except Exception:
                pass

        summaries.append(entry)
    return summaries


def _snapshot_metrics() -> dict:
    """Capture current model metrics for pre/post comparison."""
    snapshot = {
        "timestamp": datetime.now(ET).isoformat(),
        "best_model_exists": (PROJECT_ROOT / "training" / "best_model.pt").exists(),
        "best_model_hash": hash_file(PROJECT_ROOT / "training" / "best_model.pt"),
    }

    # Read .best_score
    score_file = PROJECT_ROOT / "training" / ".best_score"
    if score_file.exists():
        try:
            snapshot["best_score"] = float(score_file.read_text().strip())
        except (ValueError, OSError):
            pass

    # Latest run best score
    latest = _find_latest_run()
    if latest:
        experiments = _parse_experiments(latest / "experiments.v2.jsonl")
        kept = [e for e in experiments if e.get("score", -999) > 0]
        if kept:
            best = max(kept, key=lambda e: e.get("score", 0))
            snapshot["latest_best_score"] = best.get("score")
            snapshot["latest_best_pf"] = best.get("profit_factor")
            snapshot["latest_best_tpd"] = best.get("trades_per_day")
            snapshot["latest_best_sharpe"] = best.get("trade_sharpe")
            snapshot["latest_stop_loss_rate"] = best.get("stop_loss_rate")
            snapshot["latest_worst_chunk_pf"] = best.get("worst_chunk_pf")

    return snapshot


def _pick_oos_dates(n: int) -> list[str]:
    """Pick N recent weekdays for out-of-sample testing."""
    today = datetime.now(ET).date()
    dates = []
    d = today - timedelta(days=1)
    while len(dates) < n and d > today - timedelta(days=30):
        if d.weekday() < 5:  # weekday
            dates.append(d.isoformat())
        d -= timedelta(days=1)
    return dates


def _check_for_gaming(analysis: dict) -> list[str]:
    """Check for signs of metric gaming."""
    alerts = []
    lr = analysis.get("latest_run")
    if not lr or not lr.get("experiments"):
        return alerts

    experiments = lr["experiments"]
    kept = [e for e in experiments if e.get("score", -999) > 0]

    if len(experiments) > 10 and len(kept) == 0:
        alerts.append("STALL: 0 experiments kept in latest run — inner loop may be stuck")

    # Check for score-PF divergence across kept experiments
    if len(kept) >= 2:
        scores = [e["score"] for e in kept]
        pfs = [e.get("profit_factor", 0) for e in kept]
        if max(scores) > 2 * min(scores) and max(pfs) < 1.2 * min(pfs):
            alerts.append(
                f"GAMING INDICATOR: Score range {min(scores):.2f}-{max(scores):.2f} "
                f"but PF range only {min(pfs):.2f}-{max(pfs):.2f}"
            )

    return alerts


def _assess_inner_loop_health(analysis: dict) -> dict:
    """Assess whether the inner loop is making progress."""
    health = {"status": "unknown"}
    lr = analysis.get("latest_run")
    if not lr:
        return health

    total = lr.get("total_experiments", 0)
    kept = lr.get("kept_count", 0)

    if total == 0:
        health["status"] = "no_data"
        return health

    accept_rate = kept / total if total > 0 else 0
    health["accept_rate"] = round(accept_rate, 3)
    health["total_experiments"] = total
    health["kept_count"] = kept

    if accept_rate == 0 and total > 10:
        health["status"] = "stuck"
        health["recommendation"] = "Inner loop stuck — consider strategic intervention (modify lab_notebook.md or program.md)"
    elif accept_rate < 0.05:
        health["status"] = "struggling"
        health["recommendation"] = "Very low accept rate — inner loop may need fresh direction"
    elif accept_rate < 0.15:
        health["status"] = "normal"
        health["recommendation"] = "Typical accept rate — let inner loop continue"
    else:
        health["status"] = "productive"
        health["recommendation"] = "High accept rate — inner loop is finding improvements, let it cook"

    return health


def _compare_metrics(pre: dict, post: dict) -> dict:
    """Compare pre/post metric snapshots."""
    comparison = {}
    for key in ["best_score", "latest_best_score", "latest_best_pf",
                "latest_best_tpd", "latest_best_sharpe", "latest_stop_loss_rate",
                "latest_worst_chunk_pf"]:
        pre_val = pre.get(key)
        post_val = post.get(key)
        if pre_val is not None and post_val is not None:
            comparison[key] = {
                "before": pre_val,
                "after": post_val,
                "delta": round(post_val - pre_val, 6),
                "improved": post_val > pre_val if key != "latest_stop_loss_rate" else post_val < pre_val,
            }
    return comparison


def _detect_gaming(pre: dict, post: dict) -> list[str]:
    """Detect if a change gamed metrics without real improvement."""
    alerts = []

    pre_score = pre.get("latest_best_score")
    post_score = post.get("latest_best_score")
    pre_pf = pre.get("latest_best_pf")
    post_pf = post.get("latest_best_pf")

    if all(v is not None for v in [pre_score, post_score, pre_pf, post_pf]):
        score_delta = post_score - pre_score
        pf_delta = post_pf - pre_pf
        if score_delta > 0.5 and pf_delta < 0:
            alerts.append(
                f"GAMING: Score up {score_delta:+.2f} but PF down {pf_delta:+.2f} — "
                f"score inflation without real trading improvement"
            )

    return alerts


def _log_change(cycle_num: int, cdir: Path, verify_results: dict) -> None:
    """Append change record to changes.jsonl."""
    decision_path = cdir / "decision.md"
    pre = read_json(cdir / "pre_metrics.json") or {}
    post = read_json(cdir / "post_metrics.json") or {}

    record = {
        "cycle": cycle_num,
        "timestamp": datetime.now(ET).isoformat(),
        "decision_exists": decision_path.exists(),
        "pre_score": pre.get("latest_best_score"),
        "post_score": post.get("latest_best_score"),
        "pre_pf": pre.get("latest_best_pf"),
        "post_pf": post.get("latest_best_pf"),
        "gaming_alerts": verify_results.get("gaming_alerts", []),
        "replay_battery_pass": verify_results.get("replay_battery", {}).get("pass_count"),
        "replay_battery_total": verify_results.get("replay_battery", {}).get("total"),
        "ibkr_probe": verify_results.get("ibkr_probe", {}).get("status"),
    }

    changes_path = ART2_DIR / "changes.jsonl"
    with open(changes_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ART² — AutoResearch Trading Squared meta-orchestrator"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Deploy Akash, run inner loop, download results")
    p_train.add_argument("--minutes", type=int, default=130, help="Training time budget (default: 130 = ~20 experiments)")
    p_train.add_argument("--deposit-akt", type=int, default=None, help="AKT deposit override (default: auto)")
    p_train.add_argument("--budget", type=float, default=DEFAULT_TIER_BUDGET, help=f"Monthly API budget (default: ${DEFAULT_TIER_BUDGET:.0f})")
    p_train.add_argument("--dry-run", action="store_true")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Collect all metrics into analysis.json")
    p_analyze.add_argument("--dry-run", action="store_true")

    # report
    p_report = sub.add_parser("report", help="Generate markdown briefing for Claude Code")
    p_report.add_argument("--dry-run", action="store_true")

    # verify
    p_verify = sub.add_parser("verify", help="OOS replay battery + IBKR probe")
    p_verify.add_argument("--dates", type=int, default=5, help="Number of OOS dates (default: 5)")
    p_verify.add_argument("--dry-run", action="store_true")

    # status
    sub.add_parser("status", help="Show current ART² state")

    # init
    sub.add_parser("init", help="Initialize results/art2/ directory structure")

    # replay
    p_replay = sub.add_parser("replay", help="Backtest best model, produce trade-level metrics")
    p_replay.add_argument("--dry-run", action="store_true")

    # diagnose
    p_diagnose = sub.add_parser("diagnose", help="Compare training vs replay metrics, find mismatches")
    p_diagnose.add_argument("--dry-run", action="store_true")

    # research
    p_research = sub.add_parser("research", help="Deep analysis of trade data against domain knowledge")
    p_research.add_argument("--dry-run", action="store_true")

    # cycle
    p_cycle = sub.add_parser("cycle", help="Full cycle: train → analyze (then await strategy → verify)")
    p_cycle.add_argument("--minutes", type=int, default=130, help="Training time budget (default: 130 = ~20 experiments)")
    p_cycle.add_argument("--deposit-akt", type=int, default=None, help="AKT deposit override (default: auto)")
    p_cycle.add_argument("--skip-train", action="store_true", help="Skip training, analyze existing results")
    p_cycle.add_argument("--dry-run", action="store_true")

    # autonomous — market-aware master loop
    p_auto = sub.add_parser("autonomous", help="Market-aware autonomous loop: train when closed, monitor when open")
    p_auto.add_argument("--minutes", type=int, default=130, help="Training time budget per cycle (default: 130)")
    p_auto.add_argument("--max-cycles", type=int, default=0, help="Max training cycles (0=unlimited)")
    p_auto.add_argument("--deposit-akt", type=int, default=None, help="AKT deposit override (default: auto)")
    p_auto.add_argument("--dry-run", action="store_true")

    # daemon — persistent autonomous loop
    p_daemon = sub.add_parser("daemon", help="Persistent autonomous loop (run via launchd)")
    p_daemon.add_argument("--minutes", type=int, default=130, help="Training time budget per cycle (default: 130)")
    p_daemon.add_argument("--max-cycles", type=int, default=50, help="Max training cycles (default: 50)")
    p_daemon.add_argument("--deposit-akt", type=int, default=None, help="AKT deposit override (default: auto)")
    p_daemon.add_argument("--budget", type=float, default=DEFAULT_TIER_BUDGET, help=f"Monthly API budget (default: ${DEFAULT_TIER_BUDGET:.0f})")
    p_daemon.add_argument("--auto-only", action="store_true", help="Skip Opus, only auto-decide action A")
    p_daemon.add_argument("--dry-run", action="store_true")

    # market — check market status
    sub.add_parser("market", help="Check market open/closed status and IBKR compatibility")

    args = parser.parse_args()

    # Load .env
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value

    if args.command == "train":
        cmd_train(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "replay":
        cmd_replay(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    elif args.command == "research":
        cmd_research(args)
    elif args.command == "cycle":
        cmd_cycle(args)
    elif args.command == "autonomous":
        cmd_autonomous(args)
    elif args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "market":
        market = is_market_open()
        print(json.dumps(market, indent=2))
        print()
        gate = ibkr_compatibility_gate()
        for check in gate["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['name']}: {check.get('detail', '')}")
        print(f"\nIBKR Gate: {'PASSED' if gate['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
