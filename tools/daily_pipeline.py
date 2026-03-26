#!/usr/bin/env python3
"""Daily automation pipeline: data rebuild → paper trading → CSV export.

Training is OFF by default (stable model trades all week). Use --retrain for
weekly retraining sessions that run the full autoresearch loop on Akash.

Usage:
    python3 tools/daily_pipeline.py                          # Daily: rebuild data + trade
    python3 tools/daily_pipeline.py --retrain                # Weekly: rebuild + retrain + trade
    python3 tools/daily_pipeline.py --retrain --training-minutes 120  # Longer session
    python3 tools/daily_pipeline.py --skip-data              # Skip data rebuild
    python3 tools/daily_pipeline.py --dry-run                # Log what would happen
    python3 tools/daily_pipeline.py --export-only 2026-03-20 # Just export CSV for a date

Schedule via launchd:
  Daily (weekdays 2:30 AM PT):  infra/daily_pipeline.plist       → data + trade
  Weekly (Sunday 8 PM PT):      infra/weekly_retrain.plist       → data + retrain + trade
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path.home() / ".cache" / "autoresearch-trading"
DATA_DIR = CACHE_DIR / "data"
FEATURES_DIR = CACHE_DIR / "features"
PIPELINE_DIR = PROJECT_ROOT / "results" / "pipeline"

ET = ZoneInfo("America/New_York")

# Aggregate option caches rebuilt from per-day caches each run.
# SPY/SPX/VIX monolithic caches are now handled by prepare.py's
# _incremental_update() — they append new bars instead of re-downloading.
AGGREGATE_CACHES = [
    DATA_DIR / "spxw_full.pkl",
    DATA_DIR / "spxw_chain_full.pkl",
]


def log(msg: str) -> None:
    ts = datetime.now(ET).strftime("%H:%M:%S ET")
    print(f"[{ts}] {msg}", flush=True)


def run_cmd(cmd: list[str], log_path: Path | None = None, timeout: int | None = None,
            env: dict | None = None) -> tuple[int, str]:
    """Run a command, capturing output. Returns (returncode, output)."""
    merged_env = {**os.environ, **(env or {})}
    log(f"  $ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
            env=merged_env,
        )
        output = result.stdout + result.stderr
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"$ {' '.join(cmd)}\n\n{output}\n\nExit code: {result.returncode}\n")
        return result.returncode, output
    except subprocess.TimeoutExpired as e:
        output = f"TIMEOUT after {timeout}s"
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                f.write(f"$ {' '.join(cmd)}\n\n{output}\n")
        return -1, output


def check_ib_gateway(host: str = "127.0.0.1", port: int = 4002, timeout: float = 5.0) -> bool:
    """TCP check that IB Gateway is reachable."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True
    except (OSError, socket.timeout):
        return False


def check_prerequisites(log_dir: Path) -> dict:
    """Verify environment and return config."""
    config = {}
    errors = []

    # Check .env vars
    for var in ["POLYGON_S3_KEY_ID", "POLYGON_S3_SECRET"]:
        if not os.environ.get(var):
            errors.append(f"Missing env var: {var}")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("WARNING: ANTHROPIC_API_KEY not set — training stage will fail")

    # Check IB Gateway
    if check_ib_gateway():
        log("IB Gateway: reachable on :4002")
        config["ib_available"] = True
    else:
        log("WARNING: IB Gateway not reachable on :4002")
        config["ib_available"] = False

    if errors:
        for e in errors:
            log(f"FATAL: {e}")
        sys.exit(1)

    config["data_pt"] = FEATURES_DIR / "data.pt"
    return config


def find_best_model() -> tuple[Path | None, Path | None]:
    """Find the most recent best_model.pt and best_train.py."""
    # Primary: training/best_model.pt (deployed by deploy.sh stop)
    model = PROJECT_ROOT / "training" / "best_model.pt"
    train_py = PROJECT_ROOT / "training" / "best_train.py"
    if model.exists():
        return model, train_py if train_py.exists() else None

    # Fallback: scan results/run-* directories for the most recent
    results_dir = PROJECT_ROOT / "results"
    runs = sorted(results_dir.glob("run-*/best_model.pt"), key=lambda p: p.parent.name, reverse=True)
    if runs:
        run_dir = runs[0].parent
        model = run_dir / "best_model.pt"
        train_py = run_dir / "best_train.py"
        return model, train_py if train_py.exists() else None

    return None, None


# ---------------------------------------------------------------------------
# Stage 1: Data Rebuild
# ---------------------------------------------------------------------------

def stage_data_rebuild(log_dir: Path, yesterday: str, dry_run: bool = False) -> bool:
    log(f"=== STAGE 1: Data Rebuild (through {yesterday}) ===")

    if dry_run:
        log("  [dry-run] Would delete aggregate caches and rebuild data.pt (incremental)")
        for c in AGGREGATE_CACHES:
            log(f"  [dry-run] rm {c}")
        log(f"  [dry-run] python3 training/prepare.py --use-spx --ib-port 4002 --start 2022-03-14 --end {yesterday}")
        return True

    # Delete aggregate option caches (rebuilt from per-day caches).
    # SPY/SPX/VIX caches are updated incrementally by prepare.py.
    for cache_path in AGGREGATE_CACHES:
        if cache_path.exists():
            cache_path.unlink()
            log(f"  Deleted aggregate cache: {cache_path.name}")

    # Run prepare.py
    rc, output = run_cmd(
        [sys.executable, "training/prepare.py", "--use-spx", "--ib-port", "4002",
         "--start", "2022-03-14", "--end", yesterday],
        log_path=log_dir / "stage1_data_rebuild.log",
        timeout=1800,  # 30 min max
    )

    if rc != 0:
        log(f"  FAILED (exit {rc})")
        # Print last 20 lines of output for debugging
        for line in output.strip().split("\n")[-20:]:
            log(f"  | {line}")
        return False

    # Verify data.pt exists and is fresh
    data_pt = FEATURES_DIR / "data.pt"
    if not data_pt.exists():
        log("  FAILED: data.pt not found after rebuild")
        return False

    age_seconds = time.time() - data_pt.stat().st_mtime
    if age_seconds > 300:  # older than 5 min
        log(f"  WARNING: data.pt is {age_seconds:.0f}s old — may not have been rebuilt")

    size_mb = data_pt.stat().st_size / (1024 * 1024)
    log(f"  data.pt rebuilt: {size_mb:.0f}MB, {age_seconds:.0f}s ago")
    return True


# ---------------------------------------------------------------------------
# Stage 2: Akash Training
# ---------------------------------------------------------------------------

def stage_training(log_dir: Path, time_budget_min: int = 45, dry_run: bool = False) -> bool:
    log(f"=== STAGE 2: Akash Training ({time_budget_min} min budget) ===")
    deploy_sh = str(PROJECT_ROOT / "infra" / "deploy.sh")

    if dry_run:
        log(f"  [dry-run] {deploy_sh} boot")
        log(f"  [dry-run] {deploy_sh} start --hours {time_budget_min / 60:.2f} --max-experiments 10")
        log(f"  [dry-run] {deploy_sh} stop -y")
        return True

    # Boot
    log("  Booting Akash deployment...")
    rc, output = run_cmd(
        [deploy_sh, "boot"],
        log_path=log_dir / "stage2_boot.log",
        timeout=300,  # 5 min
        env={"DEPOSIT_AKT": "10"},
    )
    if rc != 0:
        log(f"  Boot FAILED (exit {rc})")
        for line in output.strip().split("\n")[-10:]:
            log(f"  | {line}")
        return False

    # Start training
    hours = round(time_budget_min / 60, 2)
    log(f"  Starting training loop ({hours}h, max 10 experiments)...")
    rc, output = run_cmd(
        [deploy_sh, "start", "--hours", str(hours), "--max-experiments", "10"],
        log_path=log_dir / "stage2_start.log",
        timeout=300,  # 5 min for upload
    )
    if rc != 0:
        log(f"  Start FAILED (exit {rc})")
        # Try to clean up
        run_cmd([deploy_sh, "stop", "-y"], log_path=log_dir / "stage2_stop_cleanup.log", timeout=300)
        return False

    # Wait for training to finish
    deadline = time.time() + (time_budget_min + 15) * 60  # budget + 15 min buffer
    log(f"  Waiting for training (up to {time_budget_min + 15} min)...")
    while time.time() < deadline:
        time.sleep(60)
        rc, output = run_cmd([deploy_sh, "status"], timeout=30)
        if "loop_phase" in output and '"completed"' in output.lower():
            log("  Training completed!")
            break
        # Check if process is still alive
        if "No active deployment" in output or "not running" in output.lower():
            log("  Training process appears to have ended")
            break
    else:
        log("  Training timed out — proceeding to stop")

    # Stop and download results
    log("  Stopping deployment and downloading results...")
    rc, output = run_cmd(
        [deploy_sh, "stop", "-y"],
        log_path=log_dir / "stage2_stop.log",
        timeout=600,  # 10 min for download + close
    )
    if rc != 0:
        log(f"  Stop had issues (exit {rc}) — checking for results anyway")

    # Verify results
    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    if model_path.exists():
        age = time.time() - model_path.stat().st_mtime
        log(f"  best_model.pt found ({age:.0f}s old)")
        return True
    else:
        log("  WARNING: best_model.pt not found after training")
        return False


def validate_new_model(log_dir: Path) -> bool:
    """Validation gate: compare new model score vs previous best.

    If the new model didn't improve, restore the previous model.
    Returns True if the new model was promoted (or no previous existed).
    """
    model_path = PROJECT_ROOT / "training" / "best_model.pt"
    prev_path = PROJECT_ROOT / "training" / "best_model.pt.prev"

    if not model_path.exists():
        log("  No new model to validate")
        return False

    # Check if we have a score from the training run
    # deploy.sh stop downloads results including experiments.v2.jsonl
    new_score = _get_latest_score()

    if prev_path.exists() and new_score is not None:
        prev_score = _read_prev_score()
        if prev_score is not None and new_score <= prev_score:
            log(f"  New model score ({new_score:.3f}) <= previous ({prev_score:.3f}) — reverting")
            # Restore previous model
            shutil.copy2(prev_path, model_path)
            return False
        else:
            prev_label = f"{prev_score:.3f}" if prev_score is not None else "unknown"
            log(f"  New model score ({new_score:.3f}) > previous ({prev_label}) — promoted!")
    elif new_score is not None:
        log(f"  New model score: {new_score:.3f} (no previous to compare)")
    else:
        log("  Could not determine model score — keeping new model")

    # Save score for next comparison
    _save_score(new_score)
    return True


def _get_latest_score() -> float | None:
    """Read the best score from the most recent run's experiments log."""
    results_dir = PROJECT_ROOT / "results"
    runs = sorted(results_dir.glob("run-*/experiments.v2.jsonl"),
                  key=lambda p: p.parent.name, reverse=True)
    if not runs:
        return None

    best_score = None
    try:
        with open(runs[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = json.loads(line)
                    score = exp.get("score")
                    if score is not None and exp.get("status") == "kept":
                        if best_score is None or score > best_score:
                            best_score = score
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return best_score


def _read_prev_score() -> float | None:
    """Read the previously saved best score."""
    score_file = PROJECT_ROOT / "training" / ".best_score"
    if score_file.exists():
        try:
            return float(score_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _save_score(score: float | None) -> None:
    """Persist the current best score for future comparisons."""
    if score is None:
        return
    score_file = PROJECT_ROOT / "training" / ".best_score"
    score_file.write_text(f"{score:.6f}\n")


# ---------------------------------------------------------------------------
# Stage 3: Paper Trading
# ---------------------------------------------------------------------------

def stage_paper_trading(log_dir: Path, date_str: str, model_path: Path,
                        train_py_path: Path | None, dry_run: bool = False) -> int | None:
    log("=== STAGE 3: Paper Trading ===")

    audit_path = PROJECT_ROOT / "results" / "live" / f"audit-{date_str}.jsonl"
    paper_log = log_dir / "stage3_paper_trading.log"

    cmd = [
        sys.executable, str(PROJECT_ROOT / "tools" / "paper_live.py"),
        "--paper-auto",
        "--model", str(model_path),
        "--audit-path", str(audit_path),
        "--max-minutes", "390",
    ]
    if train_py_path and train_py_path.exists():
        cmd.extend(["--train-py", str(train_py_path)])

    if dry_run:
        log(f"  [dry-run] Would launch: {' '.join(cmd)}")
        log(f"  [dry-run] Audit → {audit_path}")
        return None

    if not check_ib_gateway():
        log("  FATAL: IB Gateway not reachable — cannot start paper trading")
        return None

    log(f"  Launching paper trading (audit → {audit_path.name})...")
    log(f"  Model: {model_path}")
    if train_py_path:
        log(f"  Train.py: {train_py_path}")

    # Launch as background process
    paper_log.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_log, "w") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,  # Detach from parent process group
        )

    pid = proc.pid
    pid_file = log_dir / "paper_trade.pid"
    pid_file.write_text(str(pid))
    log(f"  Paper trading launched (PID {pid})")

    # Wait a bit and verify it's still alive
    time.sleep(10)
    try:
        os.kill(pid, 0)  # Check if process exists
        log("  Process alive after 10s — good")
    except OSError:
        log("  WARNING: Process died within 10s — check log")
        return None

    return pid


# ---------------------------------------------------------------------------
# Stage 4: Export CSV
# ---------------------------------------------------------------------------

def stage_export_csv(date_str: str, dry_run: bool = False) -> Path | None:
    log("=== STAGE 4: Export Trades to CSV ===")

    audit_path = PROJECT_ROOT / "results" / "live" / f"audit-{date_str}.jsonl"

    if not audit_path.exists():
        # Try the default audit.jsonl
        audit_path = PROJECT_ROOT / "results" / "live" / "audit.jsonl"
        if not audit_path.exists():
            log(f"  No audit file found for {date_str}")
            return None

    if dry_run:
        log(f"  [dry-run] python3 tools/export_trades.py {audit_path}")
        return None

    rc, output = run_cmd(
        [sys.executable, str(PROJECT_ROOT / "tools" / "export_trades.py"), str(audit_path)],
        timeout=60,
    )
    print(output)

    csv_path = PROJECT_ROOT / "results" / "live" / f"trades-{date_str}.csv"
    if csv_path.exists():
        log(f"  CSV exported: {csv_path}")
        return csv_path
    else:
        log("  WARNING: CSV not found after export")
        return None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_pipeline_summary(log_dir: Path, stages: dict) -> None:
    summary_path = log_dir / "summary.txt"
    lines = [
        f"Daily Pipeline Summary — {stages.get('date', '?')}",
        f"Run at: {datetime.now(ET).isoformat()}",
        "",
        f"Stage 1 (Data):     {'OK' if stages.get('data') else 'FAILED/SKIPPED'}",
        f"Stage 2 (Training): {'OK' if stages.get('training') else 'FAILED/SKIPPED'}",
        f"Stage 3 (Trading):  PID={stages.get('trading_pid', 'N/A')}",
        f"Stage 4 (CSV):      {stages.get('csv_path') or 'N/A'}",
        f"Stage 5 (IBKR):     {'OK' if stages.get('ibkr_analyze') else 'SKIPPED' if stages.get('ibkr_analyze') is None else 'FAILED'}",
        "",
        f"Model: {stages.get('model_path', 'N/A')}",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    log(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Daily automation pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Skip data rebuild stage")
    parser.add_argument("--retrain", action="store_true",
                        help="Run Akash training (default: skip — use for weekly retraining)")
    parser.add_argument("--skip-training", action="store_true",
                        help="(deprecated, training is now off by default)")
    parser.add_argument("--skip-trading", action="store_true", help="Skip paper trading stage")
    parser.add_argument("--training-minutes", type=int, default=45, help="Training time budget (default: 45)")
    parser.add_argument("--dry-run", action="store_true", help="Log what would happen without executing")
    parser.add_argument("--export-only", metavar="DATE", help="Only export CSV for given date (YYYY-MM-DD)")
    parser.add_argument("--date", help="Override target date (default: today ET)")
    args = parser.parse_args()

    # Load .env if present
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

    now_et = datetime.now(ET)
    today_str = args.date or now_et.strftime("%Y-%m-%d")
    yesterday_str = (datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    # Export-only mode
    if args.export_only:
        stage_export_csv(args.export_only, dry_run=args.dry_run)
        return

    mode = "RETRAIN" if args.retrain else "DAILY"
    log(f"=== Daily Pipeline — {today_str} ({mode}) ===")
    log(f"  Yesterday: {yesterday_str}")
    log(f"  Dry run: {args.dry_run}")
    log(f"  Retrain: {args.retrain}")

    log_dir = PIPELINE_DIR / today_str
    log_dir.mkdir(parents=True, exist_ok=True)

    config = check_prerequisites(log_dir)
    stages: dict = {"date": today_str}

    # Stage 1: Data Rebuild
    if args.skip_data:
        log("Skipping data rebuild (--skip-data)")
        stages["data"] = True
    else:
        stages["data"] = stage_data_rebuild(log_dir, yesterday_str, dry_run=args.dry_run)
        if not stages["data"]:
            log("Data rebuild failed — continuing with existing data.pt")

    # Stage 2: Training (opt-in via --retrain)
    if args.retrain:
        # Back up current model before retraining
        model_backup = PROJECT_ROOT / "training" / "best_model.pt"
        prev_backup = PROJECT_ROOT / "training" / "best_model.pt.prev"
        if model_backup.exists():
            shutil.copy2(model_backup, prev_backup)
            log(f"  Backed up current model → best_model.pt.prev")

        stages["training"] = stage_training(log_dir, args.training_minutes, dry_run=args.dry_run)
        if stages["training"] and not args.dry_run:
            # Validation gate: only keep new model if it improves
            stages["training"] = validate_new_model(log_dir)
        elif not stages["training"]:
            log("Training failed — will use existing model")
            # Restore previous model if training failed
            if prev_backup.exists() and not model_backup.exists():
                shutil.copy2(prev_backup, model_backup)
                log("  Restored previous model from backup")
    else:
        if args.skip_training:
            log("Skipping training (--skip-training is now default behavior)")
        else:
            log("Training skipped (default — use --retrain for weekly retraining)")
        stages["training"] = True

    # Find best model
    model_path, train_py_path = find_best_model()
    stages["model_path"] = str(model_path) if model_path else None
    if model_path:
        log(f"Using model: {model_path}")
    else:
        log("FATAL: No model found — cannot proceed to paper trading")
        write_pipeline_summary(log_dir, stages)
        sys.exit(1)

    # Stage 3: Paper Trading
    if args.skip_trading:
        log("Skipping paper trading (--skip-trading)")
        stages["trading_pid"] = None
    else:
        pid = stage_paper_trading(log_dir, today_str, model_path, train_py_path, dry_run=args.dry_run)
        stages["trading_pid"] = pid

    # Stage 4: CSV Export
    # For the current day, paper trading is still running — schedule export for later
    # If this is an end-of-day run, export now
    if now_et.hour >= 16:
        csv_path = stage_export_csv(today_str, dry_run=args.dry_run)
        stages["csv_path"] = str(csv_path) if csv_path else None
    else:
        log("Paper trading still running — CSV export deferred to post-market")
        log(f"  Run: python3 tools/export_trades.py results/live/audit-{today_str}.jsonl")
        stages["csv_path"] = None

    # Stage 5: IBKR Session Analysis (feedback loop)
    # Analyze yesterday's paper trading session (if audit exists)
    yesterday_audit = PROJECT_ROOT / "results" / "live" / f"audit-{yesterday_str}.jsonl"
    default_audit = PROJECT_ROOT / "results" / "live" / "audit.jsonl"
    audit_to_analyze = yesterday_audit if yesterday_audit.exists() else default_audit
    if audit_to_analyze.exists():
        log(f"Stage 5: Analyzing IBKR session from {audit_to_analyze.name}")
        if not args.dry_run:
            rc, output = run_cmd(
                [sys.executable, str(PROJECT_ROOT / "tools" / "ibkr_analyze.py"),
                 "--audit", str(audit_to_analyze)],
                log_path=log_dir / "stage5_ibkr_analyze.log",
                timeout=60,
            )
            stages["ibkr_analyze"] = rc == 0
            if rc == 0:
                log("  IBKR session analysis complete")
            else:
                log(f"  IBKR analysis failed (rc={rc})")
        else:
            log("  [DRY RUN] Would run ibkr_analyze.py")
    else:
        log("Stage 5: No audit.jsonl to analyze (skipped)")
        stages["ibkr_analyze"] = None

    write_pipeline_summary(log_dir, stages)
    log("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
