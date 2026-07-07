"""Protocol101 stage-2 exit experiment: learned per-minute HOLD/EXIT.

Approved 2026-07-07 (PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md). A
position, once open, is managed minute-by-minute: the model decides HOLD or
EXIT-NOW from the position's evolving state, replacing stage-1's fixed
stop/target/time rule. This realizes the trading truth that 0DTE positions are
almost never held to expiry (theta decay); loss truncation is the exit's job.

Isolation ladder (owner-approved): (b) train/score exits on FIXED simple OTM
entries first, so any P&L gain is attributable to the exit alone; then (a) fold
the exit onto the real stage-1 entries.

Exit label (v3 Layer-3, proven; project_v3_layer3_works): at each held minute s
the target is "is the current bid the remaining peak" -> exit now. The policy
exits at the first minute its predicted exit-probability clears a threshold,
else at the forced flat. Everything is causal: minute-s features use only the
path through minute s. Data flows through the governed loader; holdout locked.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.model.protocol101_serial_simulator import (
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    PINNED_FORCED_FLAT_BEFORE_ET,
    PINNED_LABEL_POLICIES,
    databento_symbol_from_contract_id,
    raw_cbbo_frame,
    stable_hash,
)
from v4.scripts.run_protocol101_stage1_experiment import (
    ALL_MONTH_TAGS,
    FEE_PER_TRADE,
    fold_boundaries,
    governed_sessions,
)

import pickle

SCHEMA_VERSION = "Protocol101Stage2ExitExperimentV1"
GATES_DOC = "v4/docs/PROTOCOL101_STAGE2_OBJECTIVE_AND_GATES_PROPOSAL.md"
DEFAULT_RAW_ROOT = Path("data/raw")
NY = "America/New_York"
FIXED_ENTRY_OFFSETS = {"C": 20.0, "P": -20.0}   # ~20-pt OTM call and put
ENTRY_MINUTE_STRIDE = 3
SELECTION_SEEDS = (42, 43, 44)
CONFIRMATION_SEED = 99
EXIT_THRESHOLD = 0.50
S2_MAX_DRAWDOWN_PCT = 0.25
S4_NULL_DRAWS = 200
S4_MIN_Z = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--isolation", choices=("fixed_entries", "real_entries"), default="fixed_entries")
    p.add_argument("--exit-threshold", type=float, default=EXIT_THRESHOLD)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--out-root", type=Path, default=Path("v4/audit/autoresearch"))
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    return p.parse_args()


def _forced_flat_utc(session: str) -> pd.Timestamp:
    hh, mm = (int(x) for x in PINNED_FORCED_FLAT_BEFORE_ET.split(":"))
    local = pd.Timestamp(f"{session} {hh:02d}:{mm:02d}:00").tz_localize(NY)
    return local.tz_convert("UTC")


def _symbol_bid_paths(raw_root: Path, session: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per raw symbol: (quote_time int64 ns, bid) sorted, absent bid -> 0.0
    (PINNED_NO_BID_CONVENTION: no bid = executable zero on the exit path)."""
    frame = raw_cbbo_frame(raw_root, session)
    if frame.empty:
        return {}
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol, grp in frame.groupby("symbol", sort=False):
        grp = grp.sort_values("quote_time")
        ts = pd.to_datetime(grp["quote_time"], utc=True).astype("int64").to_numpy()
        bid = pd.to_numeric(grp["bid"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[str(symbol)] = (ts, bid)
    return out


@dataclass
class Position:
    session: str
    entry_time: str
    contract_id: str
    right: str
    offset: float
    entry_ask: float
    bid_path: np.ndarray          # forward bids, one per minute after entry
    minute_grid_utc: list[str]    # decision-usable timestamp per bid


def _fixed_entry_positions(session: str, path: Path, raw_root: Path) -> list[Position]:
    rows = pickle.load(path.open("rb"))
    paths = _symbol_bid_paths(raw_root, session)
    if not paths:
        return []
    forced_flat_ns = int(_forced_flat_utc(session).value)
    positions: list[Position] = []
    for row_idx in range(0, len(rows), ENTRY_MINUTE_STRIDE):
        row = rows[row_idx]
        decision = pd.Timestamp(row.get("decision_time"))
        mask = np.asarray(row.get("candidate_mask"), dtype=bool)
        if mask.ndim != 2 or not mask.any():
            continue
        offsets = np.asarray(row.get("strike_offsets"), dtype=float)
        ids = np.asarray(row.get("contract_ids"), dtype=object)
        metadata = row.get("contract_quote_metadata") or {}
        for right, target_off in FIXED_ENTRY_OFFSETS.items():
            r_idx = 0 if right == "C" else 1
            # nearest tradable strike to the target OTM offset
            eligible = [k for k in range(mask.shape[0]) if mask[k, r_idx]]
            if not eligible:
                continue
            k = min(eligible, key=lambda j: abs(float(offsets[j]) - target_off))
            cid = str(ids[k, r_idx])
            meta = metadata.get(cid) or {}
            ask = meta.get("ask")
            if cid in ("", "None") or ask is None or not np.isfinite(float(ask)):
                continue
            symbol = databento_symbol_from_contract_id(cid)
            series = paths.get(str(symbol))
            if series is None:
                continue
            ts, bid = series
            start = int(np.searchsorted(ts, decision.value, side="right"))
            end = int(np.searchsorted(ts, forced_flat_ns, side="right"))
            if end - start < 2:
                continue
            positions.append(
                Position(
                    session=session,
                    entry_time=decision.isoformat(),
                    contract_id=cid,
                    right=right,
                    offset=float(offsets[k]),
                    entry_ask=float(ask),
                    bid_path=bid[start:end],
                    minute_grid_utc=[pd.Timestamp(t).isoformat() for t in ts[start:end]],
                )
            )
    return positions


def _exit_examples(pos: Position) -> dict[str, np.ndarray]:
    """Per-held-minute features + peak-timing exit label for one position."""
    b = pos.bid_path
    n = len(b)
    A = pos.entry_ask
    feats, labels = [], []
    running_max = np.maximum.accumulate(b)
    remaining_max = np.maximum.accumulate(b[::-1])[::-1]  # max(b[s..end])
    for s in range(n):
        unreal = b[s] / A - 1.0 if A > 0 else 0.0
        vel1 = b[s] - b[s - 1] if s >= 1 else 0.0
        vel3 = b[s] - b[s - 3] if s >= 3 else 0.0
        dd_from_peak = b[s] / running_max[s] - 1.0 if running_max[s] > 0 else 0.0
        max_ret_so_far = running_max[s] / A - 1.0 if A > 0 else 0.0
        feats.append([
            unreal, float(s), float(n - s), pos.offset, 1.0 if pos.right == "C" else 0.0,
            vel1, vel3, dd_from_peak, max_ret_so_far, A,
        ])
        labels.append(1.0 if b[s] >= remaining_max[s] - 1e-9 else 0.0)
    return {"X": np.asarray(feats, dtype=float), "y": np.asarray(labels, dtype=float)}


def _apply_exit_policy(pos: Position, model, threshold: float) -> tuple[float, int]:
    ex = _exit_examples(pos)
    probs = model.predict_proba(np.nan_to_num(ex["X"], nan=0.0))[:, 1]
    fire = np.flatnonzero(probs > threshold)
    s = int(fire[0]) if len(fire) else len(pos.bid_path) - 1
    exit_bid = float(pos.bid_path[s])
    return (exit_bid - pos.entry_ask) * 100.0, s


def _apply_fixed_policy(pos: Position, policy: dict[str, Any]) -> tuple[float, int]:
    A = pos.entry_ask
    stop = A * (1.0 - float(policy["stop_loss_pct"]))
    target = A * (1.0 + float(policy["take_profit_pct"]))
    hold_cap = min(int(policy["max_hold_minutes"]), len(pos.bid_path) - 1)
    for s in range(len(pos.bid_path)):
        if s > hold_cap:
            s = hold_cap
            break
        b = pos.bid_path[s]
        if b <= stop or b >= target:
            break
    else:
        s = len(pos.bid_path) - 1
    s = min(s, hold_cap)
    return (pos.bid_path[s] - A) * 100.0, s


def _replay(positions: list[Position], exits: list[tuple[float, int]], fee: float, split: str) -> dict[str, Any]:
    candidates = []
    for pos, (pnl, s) in zip(positions, exits):
        entry = pd.Timestamp(pos.entry_time)
        candidates.append(
            SerialCandidate(
                split=split, session=pos.session, decision_time=entry.to_pydatetime(),
                contract_id=pos.contract_id, right=pos.right, offset=pos.offset,
                entry_ask=pos.entry_ask, score=0.0, raw_label_pnl=pnl - fee,
                cooldown_minutes=float(s + 1), max_hold_minutes=float(s + 1),
            )
        )
    trades, state = simulate_serial_candidates(candidates, config=SerialSimulatorConfig())
    equity = np.asarray(state.equity_by_account.get(split, [10_000.0]))
    peaks = np.maximum.accumulate(equity)
    dd = float((peaks - equity).max()) if len(equity) else 0.0
    dd_pct = float(np.maximum((peaks - equity) / np.maximum(peaks, 1e-9), 0.0).max()) if len(equity) else 0.0
    sess_days = len({p.session for p in positions})
    losers = [t.raw_label_pnl for t in trades if t.raw_label_pnl < 0]
    return {
        "trades": len(trades), "net_pnl": float(sum(t.raw_label_pnl for t in trades)),
        "final_cash": float(equity[-1]) if len(equity) else 10_000.0,
        "max_drawdown": dd, "max_drawdown_pct": dd_pct,
        "trades_per_day": float(len(trades) / sess_days) if sess_days else 0.0,
        "mean_loss_on_losers": float(np.mean(losers)) if losers else 0.0,
        "went_bankrupt": bool(len(equity) and equity.min() < 0),
    }


def _train_exit_model(positions: list[Position], seed: int, max_iter: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X = np.concatenate([_exit_examples(p)["X"] for p in positions])
    y = np.concatenate([_exit_examples(p)["y"] for p in positions])
    model = HistGradientBoostingClassifier(max_iter=max_iter, random_state=seed)
    model.fit(np.nan_to_num(X, nan=0.0), y.astype(int))
    return model


def _best_fixed_baseline(positions: list[Position], fee: float, split: str) -> dict[str, Any]:
    best = None
    for policy in PINNED_LABEL_POLICIES:
        exits = [_apply_fixed_policy(p, policy) for p in positions]
        sim = _replay(positions, exits, fee, f"{split}_fx{policy['policy_idx']}")
        sim["policy_idx"] = int(policy["policy_idx"])
        if best is None or sim["net_pnl"] > best["net_pnl"]:
            best = sim
    return best


def _random_exit_null(positions: list[Position], learned_exit_minutes: list[int], fee: float, rng, split: str) -> list[float]:
    """Matched-rate random-exit null: exit each position at a random minute drawn
    from the learned exit-minute distribution."""
    pool = np.asarray(learned_exit_minutes)
    nulls = []
    for _ in range(S4_NULL_DRAWS):
        exits = []
        for p in positions:
            s = int(min(rng.choice(pool), len(p.bid_path) - 1))
            exits.append(((p.bid_path[s] - p.entry_ask) * 100.0, s))
        nulls.append(_replay(positions, exits, fee, split)["net_pnl"])
    return nulls


def main() -> int:
    args = parse_args()
    out_dir = args.out_root / f"protocol101_stage2_{args.experiment_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": SCHEMA_VERSION, "experiment_id": args.experiment_id,
        "hypothesis": args.hypothesis, "gates_doc": GATES_DOC, "isolation": args.isolation,
        "exit_threshold": args.exit_threshold, "fixed_entry_offsets": FIXED_ENTRY_OFFSETS,
        "entry_minute_stride": ENTRY_MINUTE_STRIDE, "selection_seeds": SELECTION_SEEDS,
        "fee_per_trade": FEE_PER_TRADE, "model": f"HistGBClassifier(max_iter={args.max_iter})",
        "exit_label": "peak_timing_bid_is_remaining_max", "month_tags": ALL_MONTH_TAGS,
    }
    config["config_hash"] = stable_hash(config)
    (out_dir / "preregistration.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"preregistered": str(out_dir / "preregistration.json"), "config_hash": config["config_hash"]}))

    if args.isolation != "fixed_entries":
        raise SystemExit("real_entries isolation is stage-2b; not implemented in this slice")

    sessions = governed_sessions("train")
    by_session = {s: _fixed_entry_positions(s, p, args.raw_root) for s, p in sessions}
    all_names = [s for s, _ in sessions]
    folds = fold_boundaries(all_names)

    seed_results: dict[int, Any] = {}
    for seed in SELECTION_SEEDS:
        fold_rows = []
        for fold in folds:
            train_pos = [pp for s in fold["train_sessions"] for pp in by_session.get(s, [])]
            test_pos = [pp for s in fold["test_sessions"] for pp in by_session.get(s, [])]
            if len(train_pos) < 50 or len(test_pos) < 10:
                continue
            model = _train_exit_model(train_pos, seed, args.max_iter)
            learned = [_apply_exit_policy(p, model, args.exit_threshold) for p in test_pos]
            learned_sim = _replay(test_pos, learned, FEE_PER_TRADE, f"s{seed}f{fold['fold']}_learned")
            baseline = _best_fixed_baseline(test_pos, FEE_PER_TRADE, f"s{seed}f{fold['fold']}")
            rng = np.random.default_rng(seed * 100 + fold["fold"])
            nulls = _random_exit_null(test_pos, [s for _, s in learned], FEE_PER_TRADE, rng, f"s{seed}f{fold['fold']}_null")
            mu, sigma = float(np.mean(nulls)), float(np.std(nulls, ddof=1)) or 1e-9
            fold_rows.append({
                "fold": fold["fold"], "test_positions": len(test_pos),
                "learned": learned_sim, "best_fixed_baseline": baseline,
                "improvement_over_baseline": learned_sim["net_pnl"] - baseline["net_pnl"],
                "vs_random_exit_z": float((learned_sim["net_pnl"] - mu) / sigma),
                "random_exit_null_mean": mu,
            })
        if not fold_rows:
            continue
        seed_results[seed] = {
            "folds": fold_rows,
            "pooled_learned_pnl": float(sum(f["learned"]["net_pnl"] for f in fold_rows)),
            "pooled_baseline_pnl": float(sum(f["best_fixed_baseline"]["net_pnl"] for f in fold_rows)),
            "folds_beating_baseline": int(sum(f["improvement_over_baseline"] > 0 for f in fold_rows)),
            "worst_fold_dd_pct": float(max(f["learned"]["max_drawdown_pct"] for f in fold_rows)),
            "any_bankruptcy": bool(any(f["learned"]["went_bankrupt"] for f in fold_rows)),
            "mean_vs_random_z": float(np.mean([f["vs_random_exit_z"] for f in fold_rows])),
            "mean_loss_on_losers_learned": float(np.mean([f["learned"]["mean_loss_on_losers"] for f in fold_rows])),
            "mean_loss_on_losers_baseline": float(np.mean([f["best_fixed_baseline"]["mean_loss_on_losers"] for f in fold_rows])),
            "trades_per_day": float(np.mean([f["learned"]["trades_per_day"] for f in fold_rows])),
        }

    per_seed = list(seed_results.values())
    worst = min(per_seed, key=lambda r: r["pooled_learned_pnl"]) if per_seed else {}
    gates = {
        "S1_beats_hold_baseline": bool(per_seed and all(
            r["pooled_learned_pnl"] > r["pooled_baseline_pnl"] and r["folds_beating_baseline"] >= 4 for r in per_seed[:1])),
        "S2_drawdown": bool(per_seed and all(r["worst_fold_dd_pct"] <= S2_MAX_DRAWDOWN_PCT for r in per_seed)),
        "S3_no_ruin": bool(per_seed and not any(r["any_bankruptcy"] for r in per_seed)),
        "S4_beats_random_exit": bool(per_seed and per_seed[0]["mean_vs_random_z"] >= S4_MIN_Z),
        "S5_seed_robustness": bool(worst and worst["pooled_learned_pnl"] > worst["pooled_baseline_pnl"]
                                   and worst["worst_fold_dd_pct"] <= S2_MAX_DRAWDOWN_PCT and not worst["any_bankruptcy"]),
        "S6_frequency": bool(per_seed and all(0.3 <= r["trades_per_day"] <= 6.0 for r in per_seed)),
        "S7_loss_truncation": bool(per_seed and all(
            r["mean_loss_on_losers_learned"] > r["mean_loss_on_losers_baseline"] for r in per_seed)),
        "S9_confirmation": None,
    }
    payload = {**config, "labels_used_for_strategy_selection": True, "model_tier": True,
               "promotion": False, "seed_results": seed_results, "gates": gates}
    payload["result_hash"] = stable_hash(payload)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    lines = [f"# Stage-2 Exit Experiment `{args.experiment_id}`", "",
             f"- Hypothesis: {args.hypothesis}", f"- Isolation: {args.isolation} | config hash `{config['config_hash']}`",
             f"- Positions: {sum(len(v) for v in by_session.values())} across {len(all_names)} sessions", "", "## Per-seed", ""]
    for seed, r in seed_results.items():
        lines.append(
            f"- seed {seed}: learned ${r['pooled_learned_pnl']:.0f} vs best-fixed ${r['pooled_baseline_pnl']:.0f} "
            f"({r['folds_beating_baseline']}/5 folds beat it), vs-random z {r['mean_vs_random_z']:.2f}, "
            f"max DD {r['worst_fold_dd_pct']*100:.0f}%, bankrupt={r['any_bankruptcy']}, "
            f"loser mean ${r['mean_loss_on_losers_learned']:.0f} (fixed ${r['mean_loss_on_losers_baseline']:.0f}), "
            f"{r['trades_per_day']:.2f} tr/day")
    lines += ["", "## Gates", ""] + [f"- {k}: **{v}**" for k, v in gates.items()]
    lines += ["", "## Guardrails", "", "- Stage-2 model tier; no promotion, no paper-submit; holdout untouched."]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "report": str(out_dir / "report.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
