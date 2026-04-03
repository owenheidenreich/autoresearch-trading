#!/usr/bin/env python3
"""Feature Tournament: test domain-knowledge features against the current best model.

Each hypothesis adds 1+ features via EXTRA_FEATURES env var, rebuilds data.pt,
uploads to Akash, and runs N experiments (fresh start). Results are compared
against a fresh-start baseline (38 features, no extras).

Usage:
  python3 tools/feature_tournament.py init
  python3 tools/feature_tournament.py run --hypothesis baseline
  python3 tools/feature_tournament.py run --hypothesis vix_roc
  python3 tools/feature_tournament.py run --hypothesis vix_roc,overnight_gap  # combo
  python3 tools/feature_tournament.py status
  python3 tools/feature_tournament.py report
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
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAIN_PY = TRAINING_DIR / "train.py"
BEST_MODEL_PT = TRAINING_DIR / "best_model.pt"
BEST_SCORE_FILE = TRAINING_DIR / ".best_score"
DEPLOY_STATE = PROJECT_ROOT / ".deploy-state"
TOURNAMENT_STATE = TRAINING_DIR / ".tournament_state.json"
TOURNAMENT_DIR = RESULTS_DIR / "tournament"

sys.path.insert(0, str(TRAINING_DIR))

# Available hypotheses (single features)
HYPOTHESES = {
    'baseline':           '',  # no extra features
    'vix_roc':            'vix_roc',
    'overnight_gap':      'overnight_gap',
    'event_day':          'event_day',
    'option_spread_width':'option_spread_width',
    'pc_volume_ratio':    'pc_volume_ratio',
    'gamma_pressure':     'gamma_pressure',
    'iv_percentile':      'iv_percentile',
    'rsi_15min':          'rsi_15min',
}

EXPERIMENTS_PER_HYPOTHESIS = 5
TIME_BUDGET = 300  # 5 min per experiment


def _load_state() -> dict:
    if TOURNAMENT_STATE.exists():
        return json.loads(TOURNAMENT_STATE.read_text())
    return {}


def _save_state(state: dict):
    TOURNAMENT_STATE.write_text(json.dumps(state, indent=2, default=str))


def _load_deploy_state() -> dict:
    if not DEPLOY_STATE.exists():
        print("FATAL: No .deploy-state found. Deploy to Akash first.", file=sys.stderr)
        sys.exit(1)
    state = {}
    for line in DEPLOY_STATE.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            state[k.strip()] = v.strip()
    return state


def _sshpass_env() -> dict:
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


def _scp_upload(deploy: dict, local: str, remote: str, timeout: int = 300):
    for attempt in range(3):
        result = subprocess.run(
            ["sshpass", "-e", "scp"] + _SSH_COMMON + [
                "-P", deploy["SSH_PORT"],
                local, f"root@{deploy['SSH_HOST']}:{remote}"],
            capture_output=True, text=True, timeout=timeout,
            env=_sshpass_env(),
        )
        if result.returncode == 0:
            return
        if attempt < 2:
            time.sleep(3)
    raise RuntimeError(f"SCP upload failed: {result.stderr}")


def _scp_download(deploy: dict, remote: str, local: str, timeout: int = 300):
    for attempt in range(3):
        result = subprocess.run(
            ["sshpass", "-e", "scp"] + _SSH_COMMON + [
                "-P", deploy["SSH_PORT"],
                f"root@{deploy['SSH_HOST']}:{remote}", local],
            capture_output=True, text=True, timeout=timeout,
            env=_sshpass_env(),
        )
        if result.returncode == 0:
            return
        if attempt < 2:
            time.sleep(3)
    raise RuntimeError(f"SCP download failed: {result.stderr}")


def _rebuild_data_pt_on_akash(deploy: dict, extra_features: str) -> str:
    """Rebuild data.pt ON AKASH (H100 has more RAM/CPU than local Mac). Returns status."""
    print(f"  Rebuilding data.pt on Akash with EXTRA_FEATURES='{extra_features}'...")

    # Upload latest prepare.py and run_loop.py first
    print("  Uploading prepare.py + run_loop.py...")
    _scp_upload(deploy, str(TRAINING_DIR / "prepare.py"), "/root/autoresearch-trading/training/prepare.py")
    _scp_upload(deploy, str(TRAINING_DIR / "run_loop.py"), "/root/autoresearch-trading/training/run_loop.py")

    ef_export = f"export EXTRA_FEATURES='{extra_features}' && " if extra_features else ""

    rebuild_cmd = (
        f"cd /root/autoresearch-trading && "
        f"{ef_export}"
        f"/opt/conda/bin/python -u training/prepare.py --skip-download --use-spx 2>&1"
    )

    print("  Running prepare.py on Akash (~10-15 min)...")
    try:
        result = _ssh_cmd(deploy, rebuild_cmd, timeout=1800)
    except subprocess.TimeoutExpired:
        print("  FATAL: data.pt rebuild timed out on Akash (30 min)", file=sys.stderr)
        sys.exit(1)

    if result.returncode != 0:
        output = result.stdout + result.stderr
        print(f"  FAILED to rebuild data.pt on Akash:\n{output[-2000:]}", file=sys.stderr)
        sys.exit(1)

    # Verify data.pt was created
    check = _ssh_cmd(deploy, "python3 -c \"import torch; d=torch.load('/root/.cache/autoresearch-trading/features/data.pt', weights_only=False, map_location='cpu'); print(f'OK:{d[\\\"features\\\"].shape[1]}')\"")
    output = check.stdout.strip()
    if 'OK:' in output:
        n_feat = output.split('OK:')[1].strip()
        print(f"  data.pt rebuilt on Akash: {n_feat} features")
        return n_feat
    else:
        print(f"  FATAL: data.pt verification failed: {output}", file=sys.stderr)
        sys.exit(1)


def _upload_code(deploy: dict):
    """Upload train.py and prepare.py to Akash."""
    print("  Uploading train.py...")
    _scp_upload(deploy, str(TRAIN_PY), "/root/autoresearch-trading/training/train.py")
    print("  Uploading prepare.py...")
    _scp_upload(deploy, str(TRAINING_DIR / "prepare.py"), "/root/autoresearch-trading/training/prepare.py")
    print("  Uploading run_loop.py...")
    _scp_upload(deploy, str(TRAINING_DIR / "run_loop.py"), "/root/autoresearch-trading/training/run_loop.py")


def _run_single_experiment(deploy: dict, extra_features: str, exp_num: int,
                           artifact_dir: Path, warm_model: Path | None) -> dict:
    """Run a single training experiment on Akash. Returns metrics dict."""
    # Upload model for warm start (or ensure fresh start)
    if warm_model and warm_model.exists():
        print(f"    Uploading warm-start model...")
        _scp_upload(deploy, str(warm_model), "/root/autoresearch-trading/training/best_model.pt")
    else:
        print(f"    Fresh start (no model)")
        _ssh_cmd(deploy, "rm -f /root/autoresearch-trading/training/best_model.pt")

    # Set EXTRA_FEATURES on Akash for train.py (which imports prepare.py)
    ef_export = f"export EXTRA_FEATURES='{extra_features}' && " if extra_features else ""

    train_cmd = (
        f"cd /root/autoresearch-trading && "
        f"{ef_export}"
        f"TIME_BUDGET={TIME_BUDGET} "
        f"/opt/conda/bin/python -u training/train.py 2>&1"
    )

    timeout = TIME_BUDGET + 480
    t_start = time.time()
    try:
        ssh_result = _ssh_cmd(deploy, train_cmd, timeout=timeout)
        wall_time = time.time() - t_start
    except subprocess.TimeoutExpired:
        return {"error": f"Timeout after {timeout}s", "wall_time": time.time() - t_start}

    output = ssh_result.stdout + ssh_result.stderr
    (artifact_dir / f"exp{exp_num}_output.log").write_text(output)

    if ssh_result.returncode != 0:
        tail = "\n".join(output.strip().split("\n")[-30:])
        return {"error": f"Exit code {ssh_result.returncode}: {tail}", "wall_time": wall_time}

    # Parse metrics from output
    import run_loop
    metrics = run_loop._parse_training_output(output)

    if "error" in metrics:
        # Fallback: try metrics.json
        try:
            tmp = artifact_dir / f"exp{exp_num}_metrics.json"
            _scp_download(deploy, "/root/autoresearch-trading/training/metrics.json", str(tmp))
            with open(tmp) as f:
                metrics = json.load(f)
        except Exception as e:
            return {"error": f"Parse failed + metrics.json fallback failed: {e}", "wall_time": wall_time}

    metrics["wall_time"] = wall_time
    (artifact_dir / f"exp{exp_num}_metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "output"}, indent=2, default=str))

    # Download model if score is positive
    score = metrics.get("score", -999)
    if score > -5.0:
        try:
            _scp_download(deploy, "/root/autoresearch-trading/training/best_model.pt",
                         str(artifact_dir / f"exp{exp_num}_model.pt"))
        except Exception:
            pass

    return metrics


def cmd_init(args):
    """Initialize tournament state."""
    TOURNAMENT_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "created": datetime.now(timezone.utc).isoformat(),
        "hypotheses": {},
        "phase": "screening",
    }
    _save_state(state)
    print(f"Tournament initialized. State: {TOURNAMENT_STATE}")


def cmd_run(args):
    """Run a hypothesis: rebuild data, upload, run N experiments."""
    hypothesis = args.hypothesis
    n_experiments = args.n or EXPERIMENTS_PER_HYPOTHESIS

    # Determine extra features string
    if hypothesis == 'baseline':
        extra_features = ''
    elif hypothesis in HYPOTHESES:
        extra_features = HYPOTHESES[hypothesis]
    else:
        # Assume it's a comma-separated combo (e.g., "vix_roc,overnight_gap")
        extra_features = hypothesis

    state = _load_state()
    if not state:
        cmd_init(args)
        state = _load_state()

    hyp_key = hypothesis
    if hyp_key not in state["hypotheses"]:
        state["hypotheses"][hyp_key] = {
            "extra_features": extra_features,
            "experiments": [],
            "best_score": -999,
            "best_pf": None,
            "best_wr": None,
            "best_tpd": None,
            "status": "pending",
        }

    hyp = state["hypotheses"][hyp_key]
    artifact_dir = TOURNAMENT_DIR / hyp_key
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TOURNAMENT: {hyp_key}")
    print(f"  Extra features: {extra_features or '(none - baseline)'}")
    print(f"  Experiments: {n_experiments}")
    print(f"{'='*60}\n")

    # Step 1+2: Upload code and rebuild data.pt on Akash
    deploy = _load_deploy_state()
    _upload_code(deploy)
    n_feat = _rebuild_data_pt_on_akash(deploy, extra_features)
    hyp["num_features"] = n_feat

    # Step 3: Run experiments (inner loop with local best tracking)
    local_best_score = -999.0
    local_best_model = None

    for exp_num in range(1, n_experiments + 1):
        print(f"\n--- Experiment {exp_num}/{n_experiments} for '{hyp_key}' ---")

        # Warm start from local best (if any), otherwise fresh
        warm_model = local_best_model
        metrics = _run_single_experiment(deploy, extra_features, exp_num, artifact_dir, warm_model)

        if "error" in metrics:
            print(f"    FAILED: {metrics['error'][:200]}")
            hyp["experiments"].append({
                "exp_num": exp_num, "error": metrics["error"][:500],
                "wall_time": metrics.get("wall_time", 0),
            })
            _save_state(state)
            continue

        score = metrics.get("score", -999)
        pf = metrics.get("profit_factor", 0)
        wr = metrics.get("win_rate", 0)
        tpd = metrics.get("trades_per_day", 0)
        n_trades = metrics.get("num_trades", 0)

        kept = score > local_best_score
        if kept:
            local_best_score = score
            # Check if model was downloaded
            model_path = artifact_dir / f"exp{exp_num}_model.pt"
            if model_path.exists():
                local_best_model = model_path
            print(f"    KEPT: score={score:.4f} PF={pf:.2f} WR={wr:.1%} TPD={tpd:.1f} trades={n_trades}")
        else:
            print(f"    REVERTED: score={score:.4f} < best={local_best_score:.4f}")

        hyp["experiments"].append({
            "exp_num": exp_num,
            "score": score,
            "profit_factor": pf,
            "win_rate": wr,
            "trades_per_day": tpd,
            "num_trades": n_trades,
            "stop_loss_rate": metrics.get("stop_loss_rate"),
            "worst_chunk_pf": metrics.get("worst_chunk_pf"),
            "avg_hold_bars": metrics.get("avg_hold_bars"),
            "model_exit_rate": metrics.get("model_exit_rate"),
            "wall_time": metrics.get("wall_time", 0),
            "kept": kept,
        })

        hyp["best_score"] = local_best_score
        if kept:
            hyp["best_pf"] = pf
            hyp["best_wr"] = wr
            hyp["best_tpd"] = tpd
        hyp["status"] = "done"
        _save_state(state)

    # Copy best model to hypothesis dir
    if local_best_model and local_best_model.exists():
        shutil.copy2(local_best_model, artifact_dir / "best_model.pt")

    print(f"\n{'='*60}")
    print(f"DONE: {hyp_key}")
    print(f"  Best score: {local_best_score:.4f}")
    print(f"  Best PF: {hyp.get('best_pf', 'N/A')}")
    print(f"  Best WR: {hyp.get('best_wr', 'N/A')}")
    print(f"{'='*60}\n")


def cmd_status(args):
    """Show tournament standings."""
    state = _load_state()
    if not state or not state.get("hypotheses"):
        print("No tournament data. Run 'init' first.")
        return

    print(f"\n{'='*80}")
    print(f"FEATURE TOURNAMENT STATUS")
    print(f"{'='*80}")
    print(f"{'Hypothesis':<25} {'Score':>8} {'PF':>6} {'WR':>6} {'TPD':>5} {'Exps':>5} {'Status':>8}")
    print(f"{'-'*80}")

    # Sort by best score descending
    sorted_hyps = sorted(state["hypotheses"].items(),
                         key=lambda x: x[1].get("best_score", -999), reverse=True)

    baseline_score = state["hypotheses"].get("baseline", {}).get("best_score", -999)

    for name, hyp in sorted_hyps:
        score = hyp.get("best_score", -999)
        pf = hyp.get("best_pf")
        wr = hyp.get("best_wr")
        tpd = hyp.get("best_tpd")
        n_exps = len(hyp.get("experiments", []))
        n_kept = sum(1 for e in hyp.get("experiments", []) if e.get("kept"))
        status = hyp.get("status", "pending")

        # Compare to baseline
        marker = ""
        if name != "baseline" and baseline_score > -999 and score > -999:
            if score > baseline_score:
                marker = " +"
            elif score >= baseline_score * 0.90:
                marker = " ~"
            else:
                marker = " X"

        pf_str = f"{pf:.2f}" if pf is not None else "---"
        wr_str = f"{wr:.1%}" if wr is not None else "---"
        tpd_str = f"{tpd:.1f}" if tpd is not None else "---"

        print(f"{name:<25} {score:>8.4f} {pf_str:>6} {wr_str:>6} {tpd_str:>5} {n_kept}/{n_exps:>2}  {status:>6}{marker}")

    print(f"\nLegend: + = beats baseline, ~ = within 10%, X = below 90% baseline")
    print()


def cmd_report(args):
    """Generate detailed tournament report."""
    state = _load_state()
    if not state or not state.get("hypotheses"):
        print("No tournament data.")
        return

    baseline = state["hypotheses"].get("baseline", {})
    baseline_score = baseline.get("best_score", -999)

    print(f"\n{'='*80}")
    print(f"FEATURE TOURNAMENT REPORT")
    print(f"{'='*80}")
    print(f"\nBaseline (38 features, fresh start): score={baseline_score:.4f}")
    print(f"  PF={baseline.get('best_pf', 'N/A')} WR={baseline.get('best_wr', 'N/A')} TPD={baseline.get('best_tpd', 'N/A')}")

    # Rank hypotheses
    winners = []
    losers = []
    inconclusive = []

    for name, hyp in state["hypotheses"].items():
        if name == "baseline":
            continue
        score = hyp.get("best_score", -999)
        if score <= -999 or hyp.get("status") != "done":
            continue

        if baseline_score <= -999:
            inconclusive.append((name, hyp))
        elif score > baseline_score:
            winners.append((name, hyp))
        elif score >= baseline_score * 0.90:
            inconclusive.append((name, hyp))
        else:
            losers.append((name, hyp))

    winners.sort(key=lambda x: x[1]["best_score"], reverse=True)
    inconclusive.sort(key=lambda x: x[1]["best_score"], reverse=True)
    losers.sort(key=lambda x: x[1]["best_score"], reverse=True)

    if winners:
        print(f"\n--- WINNERS (beat baseline) ---")
        for name, hyp in winners:
            delta = hyp["best_score"] - baseline_score
            pct = (delta / abs(baseline_score) * 100) if baseline_score != 0 else 0
            print(f"  {name}: score={hyp['best_score']:.4f} (+{delta:.4f}, +{pct:.1f}%)")
            print(f"    PF={hyp.get('best_pf', 'N/A')} WR={hyp.get('best_wr', 'N/A')} TPD={hyp.get('best_tpd', 'N/A')}")

    if inconclusive:
        print(f"\n--- INCONCLUSIVE (within 10% of baseline) ---")
        for name, hyp in inconclusive:
            print(f"  {name}: score={hyp['best_score']:.4f}")

    if losers:
        print(f"\n--- ELIMINATED (below 90% baseline) ---")
        for name, hyp in losers:
            print(f"  {name}: score={hyp['best_score']:.4f}")

    # Suggest combinations
    if winners:
        combo = ",".join(name for name, _ in winners)
        print(f"\n--- SUGGESTED COMBINATION ---")
        print(f"  Run: python3 tools/feature_tournament.py run --hypothesis '{combo}'")

    print()


def main():
    parser = argparse.ArgumentParser(description="Feature Tournament")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize tournament")

    run_p = sub.add_parser("run", help="Run a hypothesis")
    run_p.add_argument("--hypothesis", required=True, help="Hypothesis name or comma-separated combo")
    run_p.add_argument("--n", type=int, default=None, help=f"Number of experiments (default: {EXPERIMENTS_PER_HYPOTHESIS})")

    sub.add_parser("status", help="Show standings")
    sub.add_parser("report", help="Detailed report")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
