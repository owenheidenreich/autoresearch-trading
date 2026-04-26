"""Runtime oracle-vs-no-oracle gate.

⚠️  RETRACTED 2026-04-25: original feature set included `time_stop_margin_raw`
and `side_margin_raw` which are ORACLE LABELS computed from realized
future PnL (`ts_call - ts_put` and `best_forward_pnl_call - best_forward_pnl_put`
in v3/layer2/common.py:339, :330). The +24% PF lift on forward walk was
classifier reward-hacking. With those features stripped:

    Always oracle (baseline):     PF 1.890
    Clean-feature gate (thr 0.50): PF 1.736 (worse than baseline)
    Beat-baseline rate (20 seeds): 0%

The gate is preserved for educational reference but should NOT be deployed.
The feature list FEATURES below has been updated to remove the leaks; if
re-trained on clean features, the lift does not materialize.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_MODEL_PATH = "v3/artifacts/oracle_gate/gate_classifier.pkl"
DEFAULT_THRESHOLD = 0.45  # leans toward conservative oracle use
FEATURES = [
    # CLEAN entry-time features (no future-PnL info).
    # Removed 2026-04-25: `side_margin_raw` (= oracle_call - oracle_put using
    # best_forward_pnl_*) and `time_stop_margin_raw` (= realized ts_call - ts_put)
    # — these are ORACLE LABELS from common.py:330,339 and constituted reward
    # hacking when used as classifier features.
    "sigma_pos",
    "iv_percentile",
    "vix",
    "atm_iv",
    "vwap_slope",
    "volume_ratio",
    "first15_range_pct",
    "decision_margin",
    "pred_win_prob",
    "pred_clean_entry_prob",
    "pred_stopout_risk",
    "late_window_40_120_flag",
    "bars_since_break_above_first15",
    "bars_since_break_below_first15",
]


@dataclass
class OracleGate:
    classifier: object
    threshold: float
    feature_names: list[str]
    n_train: int
    train_metadata: dict

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "OracleGate":
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def predict_prob_oracle_better(self, features: dict | pd.Series | np.ndarray) -> float:
        """Single-bar inference. `features` can be a dict, Series, or np.array."""
        if isinstance(features, dict):
            x = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        elif isinstance(features, pd.Series):
            x = features[self.feature_names].fillna(0).values.reshape(1, -1)
        elif isinstance(features, np.ndarray):
            x = features.reshape(1, -1) if features.ndim == 1 else features
        else:
            raise TypeError(f"unexpected features type {type(features)}")
        return float(self.classifier.predict_proba(x)[0, 1])

    def use_oracle_exit(self, features) -> bool:
        """Returns True if oracle's predicted exit should be used; False to let
        the trade run (use time-stop / no-oracle exit)."""
        return self.predict_prob_oracle_better(features) > self.threshold


def train_gate(
    seeds=(42, 43, 44, 45, 46),
    output_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
    n_estimators: int = 80,
    max_depth: int = 3,
    random_state: int = 0,
) -> OracleGate:
    """Train the gate classifier on the OOS chosen_trades from each seed."""
    from sklearn.ensemble import GradientBoostingClassifier

    rows = []
    for s in seeds:
        df = pd.read_pickle(
            f"v3/artifacts/layer2_unified_policy_spx_combined_3seed_001_seed{s}/seed_{s}/chosen_trades.pkl"
        )
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["chosen_objective_pnl"]) & np.isfinite(df["chosen_time_stop_pnl"])
        df = df[finite].reset_index(drop=True)
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)

    X = all_df[FEATURES].fillna(0).values
    y = (all_df["chosen_objective_pnl"] > all_df["chosen_time_stop_pnl"]).astype(int).values

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    clf.fit(X, y)

    metadata = {
        "champion": "spx_combined_3seed_001",
        "seeds": list(seeds),
        "n_train_rows": int(len(all_df)),
        "pct_oracle_better": float(y.mean()),
        "feature_importance": dict(zip(FEATURES, [float(x) for x in clf.feature_importances_])),
        "validated_on": "v3/artifacts/forward_walk/spx_combined_3seed_001_with_oracle.json (53 trades, 42 days, post 2026-02-24)",
        "fw_pf_baseline_always_oracle": 1.890,
        "fw_pf_gate_at_threshold_0_55": 2.316,
        "fw_pf_perfect_ceiling": 3.203,
        "stability_20_seeds_pct_beat_baseline": 100,
    }
    gate = OracleGate(
        classifier=clf,
        threshold=threshold,
        feature_names=list(FEATURES),
        n_train=int(len(all_df)),
        train_metadata=metadata,
    )
    gate.save(output_path)
    print(f"Trained gate saved to {output_path}")
    print(f"  threshold: {threshold}")
    print(f"  n_train: {len(all_df)}")
    print(f"  pct oracle better in training: {y.mean()*100:.1f}%")
    print(f"  top features:")
    sorted_features = sorted(
        zip(FEATURES, clf.feature_importances_), key=lambda r: r[1], reverse=True
    )
    for f, w in sorted_features[:5]:
        print(f"    {f:35s}: {w:.4f}")
    return gate


def evaluate_gate_on_forward_walk(
    gate: OracleGate,
    seeds=(42, 43, 44, 45, 46),
) -> dict:
    """Evaluate the trained gate on the held-out forward-walk trades."""
    fw_rows = []
    for s in seeds:
        df = pd.read_pickle(f"v3/artifacts/forward_walk/forward_walk_chosen_seed{s}.pkl")
        if df.empty:
            continue
        df = df.copy()
        df["seed"] = s
        finite = np.isfinite(df["fwd_pnl_hybrid_with_oracle"]) & np.isfinite(df["fwd_pnl_time_stop"])
        df = df[finite].reset_index(drop=True)
        fw_rows.append(df)
    fw_df = pd.concat(fw_rows, ignore_index=True)

    X = fw_df[gate.feature_names].fillna(0).values
    oracle_pnl = fw_df["fwd_pnl_hybrid_with_oracle"].astype(float).values
    no_oracle_pnl = fw_df["fwd_pnl_time_stop"].astype(float).values
    probs = gate.classifier.predict_proba(X)[:, 1]
    use_oracle = probs > gate.threshold
    pnl = np.where(use_oracle, oracle_pnl, no_oracle_pnl)

    pos = pnl[pnl > 0].sum()
    neg = pnl[pnl < 0].sum()
    pf_value = float(pos / abs(neg)) if neg < 0 else float("inf") if pos > 0 else 0.0

    pos0 = oracle_pnl[oracle_pnl > 0].sum()
    neg0 = oracle_pnl[oracle_pnl < 0].sum()
    baseline_pf = float(pos0 / abs(neg0)) if neg0 < 0 else 0.0

    return {
        "n": int(len(fw_df)),
        "threshold": float(gate.threshold),
        "use_oracle_count": int(use_oracle.sum()),
        "use_no_oracle_count": int((~use_oracle).sum()),
        "gate_pf": pf_value,
        "gate_sum": float(pnl.sum()),
        "baseline_pf_always_oracle": baseline_pf,
        "baseline_sum": float(oracle_pnl.sum()),
        "lift_pf_pct": float((pf_value - baseline_pf) / baseline_pf * 100),
        "lift_sum_dollars": float(pnl.sum() - oracle_pnl.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "evaluate"], default="train", nargs="?")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--output", default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    if args.action == "train":
        gate = train_gate(output_path=args.output, threshold=args.threshold)
        result = evaluate_gate_on_forward_walk(gate)
        print()
        print(f"=== Forward-walk evaluation (threshold {result['threshold']}) ===")
        print(f"  n trades: {result['n']}")
        print(f"  use_oracle: {result['use_oracle_count']} | use_no_oracle: {result['use_no_oracle_count']}")
        print(f"  baseline (always oracle): PF {result['baseline_pf_always_oracle']:.3f}, sum ${result['baseline_sum']:.0f}")
        print(f"  gate:                     PF {result['gate_pf']:.3f}, sum ${result['gate_sum']:.0f}")
        print(f"  lift:                     {result['lift_pf_pct']:+.1f}% PF, ${result['lift_sum_dollars']:+.0f}")
    else:
        gate = OracleGate.load(args.output)
        result = evaluate_gate_on_forward_walk(gate)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
