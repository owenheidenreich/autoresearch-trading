#!/usr/bin/env python3
"""Phase 1: ENTROPY_COEFF targeted sweep.

Runs 5 experiments from the same checkpoint with different ENTROPY_COEFF values.
Reports which value produces the best score.
"""
import json, sys, os, time, subprocess, shutil, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PT = TRAINING_DIR / "best_model.pt"
BEST_TRAIN_PY = TRAINING_DIR / "best_train.py"
TRAIN_PY = TRAINING_DIR / "train.py"

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
from inner_loop import (
    _load_deploy_state, _scp_upload, _scp_download, _ssh_cmd,
    _load_state, _save_state,
)
sys.path.insert(0, str(TRAINING_DIR))
import run_loop

SWEEP_VALUES = [0.01, 0.02, 0.04, 0.07, 0.10]
TIME_BUDGET = 300  # 5 min per experiment


def run_sweep():
    deploy = _load_deploy_state()
    state = _load_state()
    if not state:
        print("ERROR: No inner loop state. Run 'inner_loop.py init' first.")
        sys.exit(1)

    run_dir = Path(state["run_dir"])

    # Save the model checkpoint we'll use for all sweep members
    # (don't let KEEP promote during sweep — we compare all from same base)
    sweep_model = TRAINING_DIR / ".sweep_base_model.pt"
    if BEST_MODEL_PT.exists():
        shutil.copy2(BEST_MODEL_PT, sweep_model)
        print(f"Saved sweep base model (score={state['best_score']:.4f})")
    else:
        print("No best_model.pt — sweep will train from scratch")
        sweep_model = None

    results = []

    for i, entropy_val in enumerate(SWEEP_VALUES):
        print(f"\n{'='*60}")
        print(f"SWEEP {i+1}/{len(SWEEP_VALUES)}: ENTROPY_COEFF={entropy_val}")
        print(f"{'='*60}")

        exp_id = state["experiment_id"] + 1
        state["experiment_id"] = exp_id
        artifact_dir = run_dir / f"artifacts/exp-{exp_id}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Restore base model for each sweep member
        if sweep_model and sweep_model.exists():
            shutil.copy2(sweep_model, BEST_MODEL_PT)

        # Upload train.py
        print("  Uploading train.py...")
        _scp_upload(deploy, str(TRAIN_PY), "/root/autoresearch-trading/training/train.py")

        # Upload model
        if BEST_MODEL_PT.exists():
            print("  Uploading base model...")
            _scp_upload(deploy, str(BEST_MODEL_PT), "/root/autoresearch-trading/training/best_model.pt")
        else:
            _ssh_cmd(deploy, "rm -f /root/autoresearch-trading/training/best_model.pt")

        # Train with env override
        train_cmd = (
            f"cd /root/autoresearch-trading && "
            f"TRAIN_ENTROPY_COEFF={entropy_val} TIME_BUDGET={TIME_BUDGET} "
            f"/opt/conda/bin/python -u training/train.py 2>&1"
        )

        print(f"  Training (ENTROPY_COEFF={entropy_val}, {TIME_BUDGET}s)...")
        t_start = time.time()
        timeout = TIME_BUDGET + 240
        try:
            ssh_result = _ssh_cmd(deploy, train_cmd, timeout=timeout)
            wall_time = time.time() - t_start
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {timeout}s")
            results.append({"entropy": entropy_val, "status": "timeout"})
            continue

        output = ssh_result.stdout + ssh_result.stderr
        (artifact_dir / "train_output.log").write_text(output)

        if ssh_result.returncode != 0:
            tail = "\n".join(output.strip().split("\n")[-20:])
            print(f"  CRASH: {tail[:200]}")
            results.append({"entropy": entropy_val, "status": "crash", "error": tail[:200]})
            continue

        print(f"  Training complete in {wall_time:.0f}s")

        # Parse metrics
        metrics = run_loop._parse_training_output(output)
        if "error" in metrics:
            print(f"  PARSE ERROR: {metrics['error'][:200]}")
            results.append({"entropy": entropy_val, "status": "parse_error"})
            continue

        score = metrics.get("score", -999)
        pf = metrics.get("profit_factor", 0)
        wr = metrics.get("win_rate", 0)
        trades = metrics.get("total_trades", 0)
        hit_ruin = metrics.get("hit_ruin", False)
        sl_rate = metrics.get("stop_loss_rate", 0)

        # Save full metrics
        (artifact_dir / "metrics.json").write_text(
            json.dumps({k: v for k, v in metrics.items() if k != "output"}, indent=2, default=str))

        # Download model candidate
        tmp_model = artifact_dir / "model_candidate.pt"
        try:
            _scp_download(deploy, "/root/autoresearch-trading/training/best_model.pt", str(tmp_model))
        except RuntimeError:
            pass

        result = {
            "entropy": entropy_val,
            "status": "ok",
            "score": round(score, 4),
            "pf": round(pf, 2),
            "win_rate": round(wr, 4),
            "trades": trades,
            "hit_ruin": hit_ruin,
            "sl_rate": round(sl_rate, 3),
            "exp_id": exp_id,
            "wall_time": round(wall_time, 0),
        }
        results.append(result)
        print(f"  Score={score:.4f} PF={pf:.2f} WR={wr:.1%} Trades={trades} SL={sl_rate:.0%} Ruin={hit_ruin}")

        # Log to experiments.v2.jsonl
        record = {
            "experiment_id": exp_id,
            "score": score,
            "kept": False,  # sweep doesn't promote
            "metrics": {k: v for k, v in metrics.items() if k != "output"},
            "summary": f"ENTROPY_SWEEP: ENTROPY_COEFF={entropy_val}",
            "sweep": True,
            "entropy_coeff": entropy_val,
        }
        with open(run_dir / "experiments.v2.jsonl", "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    # Save state
    _save_state(state)

    # Clean up
    if sweep_model and sweep_model.exists():
        sweep_model.unlink()

    # Report
    print(f"\n{'='*60}")
    print("ENTROPY SWEEP RESULTS")
    print(f"{'='*60}")
    print(f"{'Entropy':>10} {'Score':>8} {'PF':>6} {'WR':>7} {'Trades':>7} {'SL%':>6} {'Ruin':>5}")
    print("-" * 60)

    ok_results = [r for r in results if r["status"] == "ok"]
    for r in results:
        if r["status"] == "ok":
            print(f"{r['entropy']:>10.3f} {r['score']:>8.4f} {r['pf']:>6.2f} {r['win_rate']:>7.1%} {r['trades']:>7} {r['sl_rate']:>6.0%} {'YES' if r['hit_ruin'] else 'no':>5}")
        else:
            print(f"{r['entropy']:>10.3f} {'FAILED':>8} ({r['status']})")

    if ok_results:
        best = max(ok_results, key=lambda r: r["score"])
        print(f"\nBEST: ENTROPY_COEFF={best['entropy']} (score={best['score']:.4f}, PF={best['pf']:.2f})")

        # Promote best sweep model
        best_artifact = run_dir / f"artifacts/exp-{best['exp_id']}" / "model_candidate.pt"
        if best_artifact.exists() and best["score"] > state["best_score"]:
            shutil.copy2(best_artifact, BEST_MODEL_PT)
            shutil.copy2(TRAIN_PY, BEST_TRAIN_PY)
            state["best_score"] = best["score"]
            _save_state(state)
            print(f"PROMOTED: exp-{best['exp_id']} model → best_model.pt (score {best['score']:.4f})")
        elif best_artifact.exists():
            print(f"NOT PROMOTED: sweep best ({best['score']:.4f}) < current best ({state['best_score']:.4f})")

        # Save recommendation
        rec = {"best_entropy": best["entropy"], "results": results}
        (run_dir / "entropy_sweep.json").write_text(json.dumps(rec, indent=2, default=str))
        print(f"\nRecommendation saved to {run_dir / 'entropy_sweep.json'}")

    return results


if __name__ == "__main__":
    run_sweep()
