"""GPU Training Sweep: run multiple training configurations and evaluate.

Usage on GPU:
    python -m v2.ops.gpu_sweep [--data v2/data.pt] [--hours 2]

Runs a grid of hyperparameter configurations, trains each, evaluates
on validation data via replay, and reports results sorted by score.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Sweep configurations
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    name: str
    epochs: int = 50
    batch_size: int = 2048
    lr: float = 3e-4
    d_model: int = 64
    depth: int = 3
    dropout: float = 0.1
    gate_pos_weight: float = 0.3
    lookback: int = 60

    def env(self) -> dict[str, str]:
        return {
            "TRAIN_EPOCHS": str(self.epochs),
            "TRAIN_BATCH_SIZE": str(self.batch_size),
            "TRAIN_LR": str(self.lr),
            "TRAIN_D_MODEL": str(self.d_model),
            "TRAIN_DEPTH": str(self.depth),
            "TRAIN_DROPOUT": str(self.dropout),
            "WEIGHT_GATE_POS": str(self.gate_pos_weight),
            "TRAIN_LOOKBACK": str(self.lookback),
            "TIME_BUDGET": "600",  # 10 min per experiment max
        }


# The sweep grid: each config is one experiment
SWEEP_CONFIGS = [
    # Phase 1: Gate selectivity sweep (most important)
    SweepConfig(name="gate_0.1", gate_pos_weight=0.1, epochs=50, lr=3e-4),
    SweepConfig(name="gate_0.2", gate_pos_weight=0.2, epochs=50, lr=3e-4),
    SweepConfig(name="gate_0.3", gate_pos_weight=0.3, epochs=50, lr=3e-4),
    SweepConfig(name="gate_0.5", gate_pos_weight=0.5, epochs=50, lr=3e-4),

    # Phase 2: Learning rate sweep (with best gate_pos_weight from phase 1)
    SweepConfig(name="lr_1e-4", gate_pos_weight=0.2, epochs=50, lr=1e-4),
    SweepConfig(name="lr_5e-4", gate_pos_weight=0.2, epochs=50, lr=5e-4),
    SweepConfig(name="lr_1e-3", gate_pos_weight=0.2, epochs=50, lr=1e-3),

    # Phase 3: Architecture sweep
    SweepConfig(name="d128_d4", gate_pos_weight=0.2, d_model=128, depth=4, epochs=50, lr=3e-4),
    SweepConfig(name="d128_d3", gate_pos_weight=0.2, d_model=128, depth=3, epochs=50, lr=3e-4),
    SweepConfig(name="look120", gate_pos_weight=0.2, lookback=120, epochs=50, lr=3e-4),

    # Phase 4: Dropout sweep
    SweepConfig(name="drop_0.05", gate_pos_weight=0.2, dropout=0.05, epochs=50, lr=3e-4),
    SweepConfig(name="drop_0.2", gate_pos_weight=0.2, dropout=0.2, epochs=50, lr=3e-4),
]


def run_experiment(config: SweepConfig, data_path: str, output_dir: str) -> dict:
    """Run a single training experiment and return results."""
    model_path = os.path.join(output_dir, f"{config.name}_model.pt")
    log_path = os.path.join(output_dir, f"{config.name}.log")

    env = dict(os.environ)
    env.update(config.env())

    print(f"\n{'='*60}")
    print(f"  Experiment: {config.name}")
    print(f"  lr={config.lr} d={config.d_model} depth={config.depth} "
          f"gate_pw={config.gate_pos_weight} drop={config.dropout}")
    print(f"{'='*60}")

    t0 = time.time()

    # Train
    cmd = [sys.executable, "-m", "v2.train"]
    env["TRAIN_MODEL_PATH"] = model_path  # We'll need to add this to train.py

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=700,
        )
        log_f.write(result.stdout)
        if result.stderr:
            log_f.write("\n--- STDERR ---\n")
            log_f.write(result.stderr)

    train_time = time.time() - t0

    # Parse metrics from output
    metrics = {"name": config.name, "train_time": round(train_time, 1)}
    for line in result.stdout.splitlines():
        if line.startswith("METRICS_JSON:"):
            try:
                m = json.loads(line[len("METRICS_JSON:"):])
                metrics.update(m)
            except json.JSONDecodeError:
                pass
        # Capture last epoch line
        if "Epoch" in line and "val_loss" in line:
            metrics["last_epoch_line"] = line.strip()

    if result.returncode != 0:
        metrics["error"] = f"exit code {result.returncode}"
        print(f"  FAILED: {result.stderr[:200] if result.stderr else 'unknown error'}")
        return metrics

    # Evaluate with replay at multiple gate thresholds
    if os.path.exists(model_path):
        for gate in [0.4, 0.5, 0.6, 0.7]:
            try:
                eval_result = subprocess.run(
                    [sys.executable, "-c", f"""
import torch
from v2.replay import load_model, replay_validation
data = torch.load('{data_path}', map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)
model = load_model('{model_path}', device='cuda' if torch.cuda.is_available() else 'cpu')
m, trades = replay_validation(model, data, max_days=60, min_gate_prob={gate}, device='cuda' if torch.cuda.is_available() else 'cpu')
import json
print(json.dumps({{
    'gate': {gate},
    'pf': round(m.profit_factor, 3),
    'wr': round(m.win_rate, 3),
    'tpd': round(m.trades_per_day, 2),
    'score': round(m.score, 4),
    'trades': m.total_trades,
    'calls': m.call_count,
    'puts': m.put_count,
    'dd': round(m.max_drawdown, 3),
    'sl': m.stop_loss_count,
    'tp': m.take_profit_count,
}}))
"""],
                    capture_output=True, text=True, timeout=300,
                )
                if eval_result.returncode == 0:
                    for line in eval_result.stdout.strip().splitlines():
                        try:
                            eval_data = json.loads(line)
                            key = f"gate_{gate}"
                            metrics[key] = eval_data
                            if eval_data.get("score", 0) > metrics.get("best_score", -999):
                                metrics["best_score"] = eval_data["score"]
                                metrics["best_gate"] = gate
                                metrics["best_pf"] = eval_data["pf"]
                                metrics["best_tpd"] = eval_data["tpd"]
                                metrics["best_wr"] = eval_data["wr"]
                        except json.JSONDecodeError:
                            pass
            except subprocess.TimeoutExpired:
                metrics[f"gate_{gate}"] = {"error": "timeout"}

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="v2/data.pt")
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--output", default="v2/sweep_results")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    deadline = time.time() + args.hours * 3600

    print(f"GPU Sweep: {len(SWEEP_CONFIGS)} experiments, {args.hours}h budget")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")

    all_results = []

    for i, config in enumerate(SWEEP_CONFIGS):
        if time.time() > deadline:
            print(f"\nTime budget reached after {i} experiments")
            break

        result = run_experiment(config, args.data, args.output)
        all_results.append(result)

        # Print summary so far
        print(f"\n  val_loss={result.get('val_loss', '?'):.4f}" if isinstance(result.get('val_loss'), float) else "")
        if "best_score" in result:
            print(f"  best: gate={result.get('best_gate')} "
                  f"PF={result.get('best_pf')} TPD={result.get('best_tpd')} "
                  f"WR={result.get('best_wr')} score={result.get('best_score')}")

    # Final report
    print(f"\n\n{'='*70}")
    print(f"  SWEEP RESULTS ({len(all_results)} experiments)")
    print(f"{'='*70}")

    # Sort by best score
    scored = [r for r in all_results if "best_score" in r]
    scored.sort(key=lambda r: r.get("best_score", -999), reverse=True)

    print(f"\n{'Name':<15} {'Score':>8} {'PF':>8} {'WR':>6} {'TPD':>6} {'Gate':>5} {'Time':>6}")
    print("-" * 60)
    for r in scored:
        print(f"{r['name']:<15} {r.get('best_score', 0):>8.4f} "
              f"{r.get('best_pf', 0):>8.3f} {r.get('best_wr', 0):>5.1%} "
              f"{r.get('best_tpd', 0):>6.2f} {r.get('best_gate', '-'):>5} "
              f"{r.get('train_time', 0):>5.0f}s")

    # Save results
    results_path = os.path.join(args.output, "sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Identify best
    if scored and scored[0].get("best_score", 0) > 0:
        best = scored[0]
        print(f"\n*** BEST: {best['name']} — Score={best['best_score']:.4f} "
              f"PF={best['best_pf']:.3f} TPD={best['best_tpd']:.2f}")
    else:
        print("\n*** No configuration achieved positive score yet.")
        print("    Likely need: more selective gate, better direction balance, or structural changes.")


if __name__ == "__main__":
    main()
