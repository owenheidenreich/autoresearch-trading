"""Policy parameter sweep via local replay.

Tests combinations of exit policy parameters against the current
promoted model without retraining. Zero GPU cost.

Usage:
    python3 -m v2.analysis.policy_sweep
"""
from __future__ import annotations

import itertools
import time

import torch

from v2.core.policy import DecisionPolicy
from v2.replay import replay_validation, load_model_from_path, print_metrics


# Trailing tier configurations: (label, extra_tiers_tuple)
TIER_CONFIGS: list[tuple[str, tuple[tuple[float, float], ...]]] = [
    ("none",          ()),
    ("+30%→+10%",     ((0.30, 0.10),)),
    ("+30%→+15%",     ((0.30, 0.15),)),
    ("+35%→+10%",     ((0.35, 0.10),)),
    ("+25%→+8%",      ((0.25, 0.08),)),
]


def main():
    print("Loading model and data...")
    model = load_model_from_path("v2/models/model.pt")
    data = torch.load("v2/data.pt", map_location="cpu", weights_only=False)

    # --- Parameter grid (single-param sweeps) ---
    grid = {
        "stop_pct":              [0.20, 0.25, 0.30],
        "target_pct":            [0.30, 0.40, 0.50],
        "max_hold_bars":         [30, 45, 120],
        "cooldown_bars":         [1, 3, 5],
        "breakeven_trigger_pct": [0.10, 0.15, 0.20],
    }

    # Current baseline
    baseline = DecisionPolicy()
    print(f"\nBaseline policy: stop={baseline.stop_pct}, target={baseline.target_pct}, "
          f"hold={baseline.max_hold_bars}, cool={baseline.cooldown_bars}, "
          f"bkeven={baseline.breakeven_trigger_pct}")

    print(f"\nRunning baseline replay...")
    t0 = time.time()
    base_metrics, _, _ = replay_validation(model, data, mask_key="promote_mask", policy=baseline)
    print(f"  Baseline: score={base_metrics.score:.4f} PF={base_metrics.profit_factor:.3f} "
          f"DD={base_metrics.max_account_drawdown:.1%} Sortino={base_metrics.daily_sortino:.2f} "
          f"trades={base_metrics.total_trades} WR={base_metrics.win_rate:.1%} "
          f"+day={base_metrics.positive_day_rate:.1%} "
          f"net=${base_metrics.net_pnl_dollars:+,.0f}")
    print(f"  ({time.time()-t0:.1f}s)")

    # --- Single-parameter sweeps first (cheaper, identify which params matter) ---
    print(f"\n{'='*80}")
    print(f"  SINGLE-PARAMETER SWEEPS")
    print(f"{'='*80}")

    best_per_param: dict[str, tuple[float, dict]] = {}

    for param_name, values in grid.items():
        print(f"\n--- Sweeping {param_name} ---")
        results = []
        for val in values:
            policy_kwargs = baseline.to_dict()
            policy_kwargs[param_name] = val
            policy = DecisionPolicy.from_dict(policy_kwargs)

            t0 = time.time()
            metrics, trades, _ = replay_validation(model, data, mask_key="promote_mask", policy=policy)
            elapsed = time.time() - t0

            # Compute a composite metric for ranking
            # Use score if no gate failure, else penalize
            effective_score = metrics.score if metrics.gate_failure is None else -0.2

            results.append({
                "value": val,
                "score": metrics.score,
                "pf": metrics.profit_factor,
                "dd": metrics.max_account_drawdown,
                "sortino": metrics.daily_sortino,
                "trades": metrics.total_trades,
                "wr": metrics.win_rate,
                "pdr": metrics.positive_day_rate,
                "pnl": metrics.net_pnl_dollars,
                "gate": metrics.gate_failure,
                "effective_score": effective_score,
            })

            marker = " <<<" if val != getattr(baseline, param_name) and effective_score > base_metrics.score else ""
            gate_str = f" GATE:{metrics.gate_failure}" if metrics.gate_failure else ""
            print(f"  {param_name}={val}: score={metrics.score:.4f} PF={metrics.profit_factor:.3f} "
                  f"DD={metrics.max_account_drawdown:.1%} Sortino={metrics.daily_sortino:.2f} "
                  f"trades={metrics.total_trades} WR={metrics.win_rate:.1%} "
                  f"net=${metrics.net_pnl_dollars:+,.0f}{gate_str}{marker}")

        # Track best value per parameter
        best = max(results, key=lambda r: r["effective_score"])
        best_per_param[param_name] = (best["value"], best)

    # --- Trailing tier sweep ---
    print(f"\n{'='*80}")
    print(f"  TRAILING TIER SWEEP")
    print(f"{'='*80}")

    tier_results = []
    for label, extra_tiers in TIER_CONFIGS:
        policy_kwargs = baseline.to_dict()
        policy_kwargs["extra_trailing_tiers"] = extra_tiers
        policy = DecisionPolicy.from_dict(policy_kwargs)

        t0 = time.time()
        metrics, trades, _ = replay_validation(model, data, mask_key="promote_mask", policy=policy)
        elapsed = time.time() - t0

        effective_score = metrics.score if metrics.gate_failure is None else -0.2

        tier_results.append({
            "label": label,
            "extra_tiers": extra_tiers,
            "score": metrics.score,
            "pf": metrics.profit_factor,
            "dd": metrics.max_account_drawdown,
            "sortino": metrics.daily_sortino,
            "trades": metrics.total_trades,
            "wr": metrics.win_rate,
            "pdr": metrics.positive_day_rate,
            "pnl": metrics.net_pnl_dollars,
            "gate": metrics.gate_failure,
            "effective_score": effective_score,
        })

        marker = " <<<" if extra_tiers and effective_score > base_metrics.score else ""
        gate_str = f" GATE:{metrics.gate_failure}" if metrics.gate_failure else ""
        print(f"  tier={label}: score={metrics.score:.4f} PF={metrics.profit_factor:.3f} "
              f"DD={metrics.max_account_drawdown:.1%} Sortino={metrics.daily_sortino:.2f} "
              f"trades={metrics.total_trades} WR={metrics.win_rate:.1%} "
              f"+day={metrics.positive_day_rate:.1%} "
              f"net=${metrics.net_pnl_dollars:+,.0f}{gate_str}{marker} ({elapsed:.1f}s)")

    best_tier = max(tier_results, key=lambda r: r["effective_score"])

    # --- Trailing tier x stop_pct cross-sweep ---
    print(f"\n{'='*80}")
    print(f"  TRAILING TIER x STOP_PCT CROSS-SWEEP")
    print(f"{'='*80}")

    stop_values = [0.20, 0.25, 0.30]
    cross_results = []

    for label, extra_tiers in TIER_CONFIGS:
        for stop in stop_values:
            policy_kwargs = baseline.to_dict()
            policy_kwargs["extra_trailing_tiers"] = extra_tiers
            policy_kwargs["stop_pct"] = stop
            policy = DecisionPolicy.from_dict(policy_kwargs)

            t0 = time.time()
            metrics, trades, _ = replay_validation(model, data, mask_key="promote_mask", policy=policy)
            elapsed = time.time() - t0

            effective_score = metrics.score if metrics.gate_failure is None else -0.2

            cross_results.append({
                "tier_label": label,
                "extra_tiers": extra_tiers,
                "stop_pct": stop,
                "score": metrics.score,
                "pf": metrics.profit_factor,
                "dd": metrics.max_account_drawdown,
                "sortino": metrics.daily_sortino,
                "trades": metrics.total_trades,
                "wr": metrics.win_rate,
                "pdr": metrics.positive_day_rate,
                "pnl": metrics.net_pnl_dollars,
                "gate": metrics.gate_failure,
                "effective_score": effective_score,
            })

            is_baseline = (not extra_tiers and stop == baseline.stop_pct)
            marker = " <<<" if not is_baseline and effective_score > base_metrics.score else ""
            gate_str = f" GATE:{metrics.gate_failure}" if metrics.gate_failure else ""
            print(f"  tier={label:12s} stop={stop}: score={metrics.score:.4f} PF={metrics.profit_factor:.3f} "
                  f"DD={metrics.max_account_drawdown:.1%} trades={metrics.total_trades} "
                  f"WR={metrics.win_rate:.1%} +day={metrics.positive_day_rate:.1%} "
                  f"net=${metrics.net_pnl_dollars:+,.0f}{gate_str}{marker}")

    # --- Report best single-param improvements ---
    print(f"\n{'='*80}")
    print(f"  BEST SINGLE-PARAMETER VALUES")
    print(f"{'='*80}")
    print(f"  Baseline score: {base_metrics.score:.4f}")
    combo_kwargs = baseline.to_dict()
    for param_name, (best_val, best_result) in best_per_param.items():
        current = getattr(baseline, param_name)
        improved = "IMPROVED" if best_result["effective_score"] > base_metrics.score else "no change"
        print(f"  {param_name}: {current} -> {best_val} (score {best_result['effective_score']:.4f}) [{improved}]")
        if best_result["effective_score"] > base_metrics.score:
            combo_kwargs[param_name] = best_val

    # --- Report best trailing tier ---
    print(f"\n  Best tier config: {best_tier['label']} (score {best_tier['effective_score']:.4f})")

    # --- Report best cross-sweep result ---
    best_cross = max(cross_results, key=lambda r: r["effective_score"])
    print(f"\n  Best cross-sweep: tier={best_cross['tier_label']} stop={best_cross['stop_pct']} "
          f"(score {best_cross['effective_score']:.4f} PF={best_cross['pf']:.3f} "
          f"DD={best_cross['dd']:.1%} trades={best_cross['trades']} "
          f"net=${best_cross['pnl']:+,.0f})")

    # --- Test the combined best (single-param winners + best tier) ---
    print(f"\n{'='*80}")
    print(f"  COMBINED BEST PARAMETERS")
    print(f"{'='*80}")
    if best_tier["extra_tiers"]:
        combo_kwargs["extra_trailing_tiers"] = best_tier["extra_tiers"]
    combo_policy = DecisionPolicy.from_dict(combo_kwargs)
    print(f"  Policy: stop={combo_policy.stop_pct}, target={combo_policy.target_pct}, "
          f"hold={combo_policy.max_hold_bars}, cool={combo_policy.cooldown_bars}, "
          f"bkeven={combo_policy.breakeven_trigger_pct}, "
          f"extra_tiers={combo_policy.extra_trailing_tiers}")

    t0 = time.time()
    combo_metrics, _, _ = replay_validation(model, data, mask_key="promote_mask", policy=combo_policy)
    gate_str = f" GATE:{combo_metrics.gate_failure}" if combo_metrics.gate_failure else ""
    print(f"  Result: score={combo_metrics.score:.4f} PF={combo_metrics.profit_factor:.3f} "
          f"DD={combo_metrics.max_account_drawdown:.1%} Sortino={combo_metrics.daily_sortino:.2f} "
          f"trades={combo_metrics.total_trades} WR={combo_metrics.win_rate:.1%} "
          f"+day={combo_metrics.positive_day_rate:.1%} "
          f"net=${combo_metrics.net_pnl_dollars:+,.0f}{gate_str}")
    print(f"  vs baseline: score {base_metrics.score:.4f} -> {combo_metrics.score:.4f} "
          f"({combo_metrics.score - base_metrics.score:+.4f})")

    print(f"\n  Fingerprints:")
    print(f"    Baseline: {baseline.fingerprint()}")
    print(f"    Combo:    {combo_policy.fingerprint()}")
    print(f"    Combo policy JSON:\n{combo_policy.to_json()}")


if __name__ == "__main__":
    main()
