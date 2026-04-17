"""Overnight autoresearch runner.

Reads a candidate config JSON, trains and evaluates each variant
against the frozen baseline, applies promotion gates, and writes
a structured report.

Each candidate runs: BC → RL → collect trajectories → AWAC → eval fold 0.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch


def _run_cmd(cmd: list[str], env_extra: dict | None = None, timeout: int = 1800) -> tuple[int, str]:
    """Run a subprocess with merged env vars. Returns (returncode, output)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"


def _extract_metrics_json(output: str) -> dict | None:
    """Extract METRICS_JSON from subprocess output."""
    for line in output.split("\n"):
        if "METRICS_JSON:" in line:
            try:
                return json.loads(line.split("METRICS_JSON:", 1)[1].strip())
            except json.JSONDecodeError:
                pass
    return None


def _evaluate_fold0(agent_path: str, data_path: str, encoder_path: str,
                    env_vars: dict) -> dict:
    """Evaluate an agent on fold 0 test days. Returns metrics dict."""
    env = os.environ.copy()
    env.update(env_vars)

    # Run evaluation inline to avoid subprocess overhead. Uses the same
    # screening-mode API as the rest of the harness: --screen-mode latest is
    # the canonical single-fold triage window (fold 4 of the full CV set),
    # not the earliest window.
    eval_code = f"""
import os
for k, v in {json.dumps(env_vars)}.items():
    os.environ[k] = v

import torch, numpy as np
from collections import defaultdict
from v2.core.walkforward import generate_folds, resolve_fold_indices
from v2.core.features import _FEAT_IDX
from v2.analysis.frontier_study import _load_agent, _day_regime
from v2.replay import replay_sequential

data = torch.load("{data_path}", map_location="cpu", weights_only=False)
all_days = sorted(set(data["dates"]))
folds = generate_folds(all_days)
selected = resolve_fold_indices("latest", total_folds=len(folds))
target_fold = next(f for f in folds if f.fold_idx == selected[0])

agent = _load_agent("{encoder_path}", "{agent_path}")
metrics, trades, episodes = replay_sequential(agent, data, target_fold.test_days, deterministic=True)

entry_sides = []
for ep in episodes:
    entry_sides.extend(ep.get("entry_sides", []))
call_pnls, put_pnls = [], []
for i, t in enumerate(trades):
    if i < len(entry_sides):
        (call_pnls if entry_sides[i] == "call" else put_pnls).append(t.net_pnl_pct)

call_w = sum(p for p in call_pnls if p >= 0)
call_l = sum(abs(p) for p in call_pnls if p < 0)
put_w = sum(p for p in put_pnls if p >= 0)
put_l = sum(abs(p) for p in put_pnls if p < 0)
calls = sum(e["actions"].get(1,0) for e in episodes)
puts = sum(e["actions"].get(2,0) for e in episodes)
total = calls + puts
flips = np.mean([e["side_flips"] for e in episodes])
net = sum(t.net_pnl_pct for t in trades)

import json
result = {{
    "pf": metrics.profit_factor,
    "net_pnl": net,
    "tpd": metrics.trades_per_day,
    "call_pct": 100*calls/max(total,1),
    "flips": flips,
    "call_pf": call_w / max(call_l, 1e-6),
    "put_pf": put_w / max(put_l, 1e-6),
    "wr": metrics.win_rate,
    "n_trades": len(trades),
}}
print("EVAL_JSON:" + json.dumps(result))
"""
    rc, output = _run_cmd(
        [sys.executable, "-c", eval_code],
        env_extra=env_vars,
        timeout=600,
    )
    for line in output.split("\n"):
        if "EVAL_JSON:" in line:
            try:
                return json.loads(line.split("EVAL_JSON:", 1)[1].strip())
            except json.JSONDecodeError:
                pass
    return {"error": f"eval failed (rc={rc})", "output_tail": output[-500:]}


def _apply_gates(result: dict, baseline: dict) -> dict[str, str]:
    """Apply promotion gates. Returns gate_name → PASS/FAIL."""
    gates = {}

    # Primary metric: PF must beat baseline
    if "error" in result:
        return {"all": "FAIL (eval error)"}

    pf_delta = result["pf"] - baseline["pf"]
    gates["pf_improvement"] = "PASS" if pf_delta > 0 else "FAIL"

    # Behavior gates
    gates["call_pct"] = "PASS" if 40 <= result["call_pct"] <= 60 else "FAIL"
    gates["flips"] = "PASS" if result["flips"] <= 0.5 else "FAIL"
    gates["tpd"] = "PASS" if 1.0 <= result["tpd"] <= 3.0 else "FAIL"

    return gates


def run_candidate(
    name: str,
    env_vars: dict,
    hypothesis: str,
    data_path: str,
    encoder_path: str,
    work_dir: str,
    bc_epochs: int = 20,
    rl_epochs: int = 50,
    awac_epochs: int = 100,
    time_budget: int = 600,
) -> dict:
    """Run full pipeline for one candidate: BC → RL → collect → AWAC → eval."""
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  Candidate: {name}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  Env: {env_vars}")
    print(f"{'='*60}")

    bc_path = os.path.join(work_dir, f"{name}_bc.pt")
    rl_path = os.path.join(work_dir, f"{name}_rl.pt")
    traj_dir = os.path.join(work_dir, f"{name}_traj")
    awac_path = os.path.join(work_dir, f"{name}_awac.pt")
    os.makedirs(traj_dir, exist_ok=True)

    result = {
        "name": name,
        "hypothesis": hypothesis,
        "env_vars": env_vars,
        "steps": {},
    }

    # Shared env vars for all steps
    base_env = {
        "ENV_LATE_ENTRY_BAR": "89",
        "ENV_LATE_EXIT_BAR": "95",
        "ENV_DECAY_COEFF": "0.001",
    }
    base_env.update(env_vars)

    # Step 1: BC training
    print(f"  [1/4] BC training...")
    bc_env = {**base_env, "BC_EPOCHS": str(bc_epochs), "SEQ_TIME_BUDGET": str(time_budget)}
    rc, out = _run_cmd(
        [sys.executable, "-m", "v2.train_seq",
         "--data", data_path, "--model", encoder_path,
         "--output", bc_path, "--mode", "bc"],
        env_extra=bc_env, timeout=time_budget + 120,
    )
    bc_metrics = _extract_metrics_json(out)
    result["steps"]["bc"] = {"rc": rc, "metrics": bc_metrics}
    if rc != 0 or not os.path.exists(bc_path):
        result["error"] = f"BC failed (rc={rc})"
        result["output_tail"] = out[-500:]
        return result
    print(f"         BC done: {bc_metrics}")

    # Step 2: RL fine-tuning
    print(f"  [2/4] RL training...")
    rl_env = {
        **base_env,
        "RL_ENTRY_COST": "0.015", "RL_ENTRY_ESCALATION": "0.015",
        "RL_SIDE_IMBALANCE": "0.03", "RL_KL_COEFF": "0.03",
        "RL_LR": "1e-4", "RL_EPOCHS": str(rl_epochs),
        "SEQ_TIME_BUDGET": str(time_budget),
    }
    rc, out = _run_cmd(
        [sys.executable, "-m", "v2.train_seq",
         "--data", data_path, "--model", encoder_path,
         "--bc-checkpoint", bc_path, "--output", rl_path, "--mode", "rl"],
        env_extra=rl_env, timeout=time_budget + 120,
    )
    rl_metrics = _extract_metrics_json(out)
    result["steps"]["rl"] = {"rc": rc, "metrics": rl_metrics}
    if rc != 0 or not os.path.exists(rl_path):
        result["error"] = f"RL failed (rc={rc})"
        result["output_tail"] = out[-500:]
        return result
    print(f"         RL done: {rl_metrics}")

    # Step 3: Collect trajectories
    print(f"  [3/4] Collecting trajectories...")
    collect_env = {
        **base_env,
        "RL_ENTRY_COST": "0.015", "RL_ENTRY_ESCALATION": "0.015",
        "RL_SIDE_IMBALANCE": "0.03",
    }
    rc, out = _run_cmd(
        [sys.executable, "-m", "v2.collect_trajectories",
         "--data", data_path, "--encoder", encoder_path,
         "--checkpoint", rl_path, "--source-label", name,
         "--output-dir", traj_dir, "--fold", "0"],
        env_extra=collect_env, timeout=time_budget + 120,
    )
    result["steps"]["collect"] = {"rc": rc}
    if rc != 0:
        result["error"] = f"Collection failed (rc={rc})"
        result["output_tail"] = out[-500:]
        return result
    print(f"         Collection done")

    # Step 4: AWAC training
    print(f"  [4/4] AWAC training...")
    awac_env = {
        **base_env,
        "RL_ENTRY_COST": "0.015", "RL_ENTRY_ESCALATION": "0.015",
        "RL_SIDE_IMBALANCE": "0.03",
        "AWAC_LAMBDA": "1.0", "AWAC_VALUE_EPOCHS": "10",
        "AWAC_EPOCHS": str(awac_epochs), "AWAC_VAL_INTERVAL": "5",
        "AWAC_TIME_BUDGET": str(time_budget * 3),
    }
    rc, out = _run_cmd(
        [sys.executable, "-m", "v2.train_awac",
         "--data", data_path, "--encoder", encoder_path,
         "--bc-checkpoint", bc_path, "--trajectory-dir", traj_dir,
         "--output", awac_path, "--fold", "0"],
        env_extra=awac_env, timeout=time_budget * 3 + 120,
    )
    awac_metrics = _extract_metrics_json(out)
    result["steps"]["awac"] = {"rc": rc, "metrics": awac_metrics}
    if rc != 0 or not os.path.exists(awac_path):
        result["error"] = f"AWAC failed (rc={rc})"
        result["output_tail"] = out[-500:]
        return result
    print(f"         AWAC done: {awac_metrics}")

    # Step 5: Evaluate
    print(f"  [eval] Evaluating on fold 0...")
    eval_env = {**base_env}
    eval_result = _evaluate_fold0(awac_path, data_path, encoder_path, eval_env)
    result["eval"] = eval_result
    result["elapsed"] = time.time() - t0
    print(f"         Eval: {eval_result}")

    return result


def write_report(results: list[dict], baseline: dict, output_path: str):
    """Write structured markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Autoresearch Report — {now}\n",
        f"## Baseline: AWAC-RL K=1 (fold 0)\n",
        f"PF={baseline['pf']:.3f}  Net={baseline['net_pnl']:.3f}  "
        f"C%={baseline['call_pct']:.0f}  Flips={baseline['flips']:.2f}  "
        f"CallPF={baseline['call_pf']:.3f}  PutPF={baseline['put_pf']:.3f}\n",
        f"\n## Candidates\n",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"\n### {i}. {r['name']}\n")
        lines.append(f"- **Hypothesis:** {r['hypothesis']}\n")
        lines.append(f"- **Env:** `{r['env_vars']}`\n")

        if "error" in r:
            lines.append(f"- **Result:** FAILED — {r['error']}\n")
            if "output_tail" in r:
                lines.append(f"- **Output tail:** `{r['output_tail'][-200:]}`\n")
            lines.append(f"- **Verdict:** REJECT\n")
            continue

        ev = r["eval"]
        if "error" in ev:
            lines.append(f"- **Result:** EVAL FAILED — {ev['error']}\n")
            lines.append(f"- **Verdict:** REJECT\n")
            continue

        pf_delta = ev["pf"] - baseline["pf"]
        lines.append(f"- PF={ev['pf']:.3f}  Net={ev['net_pnl']:.3f}  "
                      f"C%={ev['call_pct']:.0f}  Flips={ev['flips']:.2f}\n")
        lines.append(f"- CallPF={ev['call_pf']:.3f}  PutPF={ev['put_pf']:.3f}\n")
        lines.append(f"- vs baseline: PF {pf_delta:+.3f}\n")

        gates = _apply_gates(ev, baseline)
        gate_str = "  ".join(f"{k}={v}" for k, v in gates.items())
        lines.append(f"- **Gates:** {gate_str}\n")

        all_pass = all(v == "PASS" for v in gates.values())
        verdict = "PROMOTE" if all_pass else "ARCHIVE"
        lines.append(f"- **Verdict:** {verdict}\n")
        lines.append(f"- **Elapsed:** {r.get('elapsed', 0):.0f}s\n")

    # Summary
    lines.append(f"\n## Summary\n")
    promoted = [r for r in results if "error" not in r and "error" not in r.get("eval", {})]
    if promoted:
        best = max(promoted, key=lambda r: r["eval"]["pf"])
        lines.append(f"Best candidate: **{best['name']}** (PF={best['eval']['pf']:.3f})\n")
    else:
        lines.append(f"No candidates produced valid results.\n")

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"\nReport written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Overnight autoresearch runner")
    parser.add_argument("--config", required=True, help="Path to candidate config JSON")
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--encoder", default="v2/models/model_trained_encoder.pt")
    parser.add_argument("--work-dir", default="v2/autoresearch_work")
    parser.add_argument("--report", default=None,
                        help="Report output path (default: v2/autoresearch_report_YYYYMMDD.md)")
    parser.add_argument("--bc-epochs", type=int, default=20)
    parser.add_argument("--rl-epochs", type=int, default=50)
    parser.add_argument("--awac-epochs", type=int, default=100)
    parser.add_argument("--time-budget", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    args = parser.parse_args()

    with open(args.config) as f:
        candidates = json.load(f)

    print(f"Autoresearch: {len(candidates)} candidates")
    for c in candidates:
        print(f"  - {c['name']}: {c['hypothesis']}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the above candidates. Exiting.")
        return

    os.makedirs(args.work_dir, exist_ok=True)

    # Baseline: evaluate current best AWAC-RL K=1 on fold 0
    print("\nEvaluating baseline (AWAC-RL K=1, fold 0)...")
    baseline_env = {
        "ENV_LATE_ENTRY_BAR": "89", "ENV_LATE_EXIT_BAR": "95",
        "ENV_DECAY_COEFF": "0.001",
    }
    baseline = _evaluate_fold0(
        "v2/models/seq_agent_awac_rl_fold0.pt",
        args.data, args.encoder, baseline_env,
    )
    print(f"Baseline: {baseline}")

    # Run candidates
    results = []
    for c in candidates:
        r = run_candidate(
            name=c["name"],
            env_vars=c.get("env", {}),
            hypothesis=c["hypothesis"],
            data_path=args.data,
            encoder_path=args.encoder,
            work_dir=args.work_dir,
            bc_epochs=args.bc_epochs,
            rl_epochs=args.rl_epochs,
            awac_epochs=args.awac_epochs,
            time_budget=args.time_budget,
        )
        results.append(r)

    # Write report
    report_path = args.report or f"v2/autoresearch_report_{datetime.now().strftime('%Y%m%d')}.md"
    write_report(results, baseline, report_path)

    # Save raw results as JSON for programmatic access
    raw_path = report_path.replace(".md", ".json")
    with open(raw_path, "w") as f:
        json.dump({"baseline": baseline, "candidates": results}, f, indent=2, default=str)
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()
