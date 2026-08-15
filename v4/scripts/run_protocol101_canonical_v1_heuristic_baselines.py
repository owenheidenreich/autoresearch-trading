"""Protocol101 canonical v1 fixed-heuristic baseline packet.

This is an offline-only probe packet. It runs preregistered, non-learned
canonical v1 feature heuristics through the same strict serial simulator used by
the stage-1 experiments, then compares them with the already recalibrated random
null/canary packet. It does not train models, select production thresholds,
touch broker/runtime state, or claim paper readiness.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    PINNED_LABEL_POLICIES,
    stable_hash,
)
from v4.scripts.run_protocol101_stage1_experiment import (
    EMBARGO_SESSIONS,
    FEE_PER_TRADE,
    FOLD_COUNT,
    MAX_CANDIDATES_PER_ROW,
    ROW_STRIDE,
    CandidateTable,
    replay,
    subset,
)
from v4.scripts.protocol101_training_scope import load_training_scope


SCHEMA_VERSION = "Protocol101CanonicalV1HeuristicBaselinesV1"
DEFAULT_OUT_DIR = Path("v4/audit/autoresearch/protocol101_canonical_v1_heuristic_baselines_training_scope")
DEFAULT_NULL_SUMMARY = Path("v4/audit/autoresearch/protocol101_canonical_v1_null_canary_training_scope/summary.json")
CONTRACT = "protocol101-live-v2-microstructure-masked"
CANONICAL_SELECTION_CONTRACT = "protocol101-canonical-minute-game-v1.4"
FEE_OVERLAY_DOLLARS = 3.00
STRADDLE_THRESHOLD_Q = 0.80
SKEW_THRESHOLD_Q = 0.80
MOMENTUM_THRESHOLD_Q = 0.80
DELTA_GEOMETRY_THRESHOLD_Q = 0.80
NEAR_ATM_ABS_OFFSET = 25.0
G2_MIN_Z = 3.0
G4_MAX_DRAWDOWN_PCT = 0.25
G7_TRADES_PER_DAY = (0.3, 6.0)

HEURISTICS = (
    "straddle_mid_expansion",
    "put_call_ratio_skew",
    "near_atm_band_momentum",
    "internal_delta_geometry",
)
THRESHOLD_QUANTILES = {
    "straddle_mid_expansion": STRADDLE_THRESHOLD_Q,
    "put_call_ratio_skew": SKEW_THRESHOLD_Q,
    "near_atm_band_momentum": MOMENTUM_THRESHOLD_Q,
    "internal_delta_geometry": DELTA_GEOMETRY_THRESHOLD_Q,
}


@dataclass(frozen=True)
class HeuristicTable:
    table: CandidateTable
    scores: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--null-summary", type=Path, default=DEFAULT_NULL_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def finite_quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def positive_quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return finite_quantile(values, q)
    return float(np.quantile(finite, q))


def extract_heuristic_table(sessions: list[tuple[str, Path]]) -> HeuristicTable:
    feats: list[np.ndarray] = []
    pnl: list[float] = []
    session_idx: list[int] = []
    stratum: list[int] = []
    session_name: list[str] = []
    decision_time: list[str] = []
    contract_id: list[str] = []
    right: list[str] = []
    offset: list[float] = []
    entry_ask: list[float] = []
    policy_col: list[int] = []
    score_lists: dict[str, list[float]] = {name: [] for name in HEURISTICS}

    for s_idx, (name, path) in enumerate(sessions):
        rows = pickle.load(path.open("rb"))
        for row_idx in range(0, len(rows), ROW_STRIDE):
            row = rows[row_idx]
            mask = np.asarray(row.get("candidate_mask"), dtype=bool)
            if mask.ndim != 2 or not mask.any():
                continue
            ladder = np.asarray(row.get("option_ladder"), dtype=float)
            labels = np.asarray(row.get("labels_net_pnl"), dtype=float)
            offsets = np.asarray(row.get("strike_offsets"), dtype=float)
            window = np.asarray(row.get("market_window"), dtype=float)
            market = window[-1]
            feature_names = tuple(row.get("feature_names") or ())
            market_names = tuple(row.get("market_feature_names") or ())
            mid_idx = feature_names.index("mid")
            ask_idx = feature_names.index("ask")
            delta_idx = feature_names.index("delta")
            spx_idx = market_names.index("spx_close")
            metadata = row.get("contract_quote_metadata") or {}
            ids = np.asarray(row.get("contract_ids"), dtype=object)
            decision = pd.Timestamp(row.get("decision_time"))
            minute_of_day = decision.hour * 60 + decision.minute
            candidates = np.argwhere(mask)
            if len(candidates) > MAX_CANDIDATES_PER_ROW:
                keep = np.linspace(0, len(candidates) - 1, MAX_CANDIDATES_PER_ROW).astype(int)
                candidates = candidates[keep]

            mid = ladder[:, :, mid_idx]
            call_mid = mid[:, 0]
            put_mid = mid[:, 1]
            spx = float(market[spx_idx]) if np.isfinite(float(market[spx_idx])) else np.nan
            straddle_by_strike = (call_mid + put_mid) / max(spx, 1e-9) * 10_000.0
            with np.errstate(divide="ignore", invalid="ignore"):
                skew_by_strike = np.abs(np.log(np.maximum(put_mid, 1e-9) / np.maximum(call_mid, 1e-9)))

            previous_mid = None
            if row_idx >= 5:
                previous_row = rows[row_idx - 5]
                previous_ladder = np.asarray(previous_row.get("option_ladder"), dtype=float)
                previous_mid = previous_ladder[:, :, mid_idx]

            for k_idx, r_idx in candidates:
                cid = str(ids[k_idx, r_idx])
                ask = (metadata.get(cid) or {}).get("ask")
                if ask is None or not np.isfinite(float(ask)):
                    ask = float(ladder[k_idx, r_idx, ask_idx])
                if not np.isfinite(float(ask)):
                    continue

                current_mid = float(mid[k_idx, r_idx])
                if previous_mid is None or abs(float(offsets[k_idx])) > NEAR_ATM_ABS_OFFSET:
                    momentum_score = np.nan
                else:
                    prior_mid = float(previous_mid[k_idx, r_idx])
                    if current_mid > 0.0 and prior_mid > 0.0:
                        momentum_score = abs(float(np.log(current_mid / prior_mid)))
                    else:
                        momentum_score = np.nan
                delta = float(ladder[k_idx, r_idx, delta_idx])
                delta_geometry = -abs(abs(delta) - 0.35) - abs(float(offsets[k_idx])) / 100.0
                base_scores = {
                    "straddle_mid_expansion": float(straddle_by_strike[k_idx]),
                    "put_call_ratio_skew": float(skew_by_strike[k_idx]),
                    "near_atm_band_momentum": float(momentum_score),
                    "internal_delta_geometry": float(delta_geometry),
                }

                for p_idx in range(labels.shape[2]):
                    value = float(labels[k_idx, r_idx, p_idx])
                    if not np.isfinite(value):
                        continue
                    feats.append(np.asarray([0.0], dtype=float))
                    pnl.append(value)
                    session_idx.append(s_idx)
                    near = int(abs(float(offsets[k_idx])) <= 20.0)
                    stratum.append(s_idx * 1000 + p_idx * 10 + near)
                    session_name.append(name)
                    decision_time.append(decision.isoformat())
                    contract_id.append(cid)
                    right.append("C" if int(r_idx) == 0 else "P")
                    offset.append(float(offsets[k_idx]))
                    entry_ask.append(float(ask))
                    policy_col.append(int(p_idx))
                    for heuristic in HEURISTICS:
                        score_lists[heuristic].append(base_scores[heuristic])

    table = CandidateTable(
        X=np.asarray(feats, dtype=float),
        pnl=np.asarray(pnl, dtype=float),
        session_idx=np.asarray(session_idx, dtype=int),
        stratum=np.asarray(stratum, dtype=int),
        session_name=session_name,
        decision_time=decision_time,
        contract_id=contract_id,
        right=right,
        offset=np.asarray(offset, dtype=float),
        entry_ask=np.asarray(entry_ask, dtype=float),
        policy_idx=np.asarray(policy_col, dtype=int),
    )
    return HeuristicTable(
        table=table,
        scores={name: np.asarray(values, dtype=float) for name, values in score_lists.items()},
    )


def select_by_score(table: CandidateTable, scores: np.ndarray, threshold: float | None) -> np.ndarray:
    if threshold is None:
        return np.asarray([], dtype=int)
    chosen: dict[tuple[str, str], int] = {}
    for i, (session, decision) in enumerate(zip(table.session_name, table.decision_time)):
        score = float(scores[i])
        if not np.isfinite(score) or score < threshold:
            continue
        key = (session, decision)
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = i
            continue
        if score > scores[existing]:
            chosen[key] = i
        elif score == scores[existing]:
            # Deterministic, conservative tie-break: nearer ATM, then lower strike
            # offset, then calls before puts.
            current_key = (abs(table.offset[i]), table.offset[i], table.right[i])
            existing_key = (abs(table.offset[existing]), table.offset[existing], table.right[existing])
            if current_key < existing_key:
                chosen[key] = i
    return np.asarray(sorted(chosen.values()), dtype=int)


def pooled_null(policy_idx: int, null_summary: dict[str, Any]) -> dict[str, float]:
    result = null_summary["policy_results"][str(policy_idx)]["pooled_random_net_pnl"]
    return {
        "mean": float(result["mean"]),
        "std": float(result["std"]) if float(result["std"]) > 0 else 1e-9,
        "p95": float(result["p95"]),
    }


def main() -> int:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise SystemExit(f"{args.out_dir} exists; pass --force to overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    null_summary = json.loads(args.null_summary.read_text())
    training_scope = load_training_scope()
    preregistration = {
        "schema_version": f"{SCHEMA_VERSION}Preregistration",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract": CONTRACT,
        "canonical_selection_contract": CANONICAL_SELECTION_CONTRACT,
        "scope": "offline fixed-heuristic baseline probes only",
        "heuristics": {
            "straddle_mid_expansion": {
                "definition": "(call_mid + put_mid) / spx_close * 10000 at the candidate strike",
                "threshold": f"train-fold finite-score q{STRADDLE_THRESHOLD_Q:.2f}",
            },
            "put_call_ratio_skew": {
                "definition": "abs(log(put_mid / call_mid)) at the candidate strike",
                "threshold": f"train-fold finite-score q{SKEW_THRESHOLD_Q:.2f}",
            },
            "near_atm_band_momentum": {
                "definition": "abs(log(current_mid / mid_5m_ago)) for |offset| <= 25",
                "threshold": f"train-fold positive finite-score q{MOMENTUM_THRESHOLD_Q:.2f}",
            },
            "internal_delta_geometry": {
                "definition": "-abs(abs(delta)-0.35) - abs(offset)/100 using the processed delta field",
                "threshold": f"train-fold finite-score q{DELTA_GEOMETRY_THRESHOLD_Q:.2f}",
            },
        },
        "folds": FOLD_COUNT,
        "embargo_sessions": EMBARGO_SESSIONS,
        "row_stride": ROW_STRIDE,
        "max_candidates_per_row": MAX_CANDIDATES_PER_ROW,
        "design_path": str(training_scope.design_path),
        "manifest_path": str(training_scope.manifest_path),
        "acceptance_registry_path": str(training_scope.acceptance_registry_path),
        "acceptance_registry_hash": training_scope.acceptance_registry_hash,
        "fold_governance_hash": training_scope.fold_governance_hash,
        "fee_overlay_dollars": FEE_OVERLAY_DOLLARS,
        "gates": {
            "G1": "fee-adjusted net PnL > 0 on >=4/5 folds and pooled > 0",
            "G2": f"pooled z >= {G2_MIN_Z} against canonical v1 null canary",
            "G4": f"max strict-serial drawdown <= {G4_MAX_DRAWDOWN_PCT:.0%} of peak equity",
            "G7": f"trades/day within {G7_TRADES_PER_DAY}",
        },
        "side_effects": {
            "model_training_executed": False,
            "threshold_optimization_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_download": False,
            "promotion_or_default_changed": False,
            "runtime_flags_edited": False,
            "launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    preregistration["preregistration_hash"] = stable_hash(preregistration)
    write_json(args.out_dir / "preregistration.json", preregistration)

    sessions = training_scope.sessions
    heuristic_table = extract_heuristic_table(sessions)
    table = heuristic_table.table
    folds = training_scope.folds
    session_arr = np.asarray(table.session_name)
    policy_results: dict[str, Any] = {}
    rows_for_csv: list[dict[str, Any]] = []

    for policy_idx in range(len(PINNED_LABEL_POLICIES)):
        policy_mask = table.policy_idx == policy_idx
        policy_results[str(policy_idx)] = {}
        null = pooled_null(policy_idx, null_summary)
        for heuristic, scores_all in heuristic_table.scores.items():
            fold_results = []
            pooled_pnl = 0.0
            pooled_trades = 0
            pooled_selected = 0
            max_drawdown_pct = 0.0
            for fold in folds:
                train_mask = policy_mask & np.isin(session_arr, fold["train_sessions"])
                test_mask = policy_mask & np.isin(session_arr, fold["test_sessions"])
                train_scores = scores_all[train_mask]
                if heuristic == "near_atm_band_momentum":
                    threshold = positive_quantile(train_scores, THRESHOLD_QUANTILES[heuristic])
                else:
                    threshold = finite_quantile(train_scores, THRESHOLD_QUANTILES[heuristic])
                test_table = subset(table, test_mask)
                test_scores = scores_all[test_mask]
                selected = select_by_score(test_table, test_scores, threshold)
                sim = replay(test_table, selected, FEE_PER_TRADE, f"{heuristic}_policy{policy_idx}_fold{fold['fold']}")
                pooled_pnl += float(sim["net_pnl"])
                pooled_trades += int(sim["trades_executed"])
                pooled_selected += int(sim["candidates_selected"])
                max_drawdown_pct = max(max_drawdown_pct, float(sim["max_drawdown_pct"]))
                fold_row = {
                    "fold": int(fold["fold"]),
                    "threshold": threshold,
                    "test_sessions": len(fold["test_sessions"]),
                    **sim,
                }
                fold_results.append(fold_row)
                rows_for_csv.append(
                    {
                        "policy_idx": policy_idx,
                        "heuristic": heuristic,
                        "fold": int(fold["fold"]),
                        "threshold": threshold,
                        "net_pnl": sim["net_pnl"],
                        "max_drawdown_pct": sim["max_drawdown_pct"],
                        "trades_executed": sim["trades_executed"],
                        "trades_per_day": sim["trades_per_day"],
                    }
                )
            z = (pooled_pnl - null["mean"]) / null["std"]
            profitable_folds = sum(1 for row in fold_results if float(row["net_pnl"]) > 0.0)
            mean_trades_per_day = float(np.mean([row["trades_per_day"] for row in fold_results]))
            gates = {
                "G1_profitability": bool(profitable_folds >= 4 and pooled_pnl > 0.0),
                "G2_beats_no_skill": bool(z >= G2_MIN_Z),
                "G4_drawdown": bool(max_drawdown_pct <= G4_MAX_DRAWDOWN_PCT),
                "G7_frequency": bool(G7_TRADES_PER_DAY[0] <= mean_trades_per_day <= G7_TRADES_PER_DAY[1]),
            }
            policy_results[str(policy_idx)][heuristic] = {
                "folds": fold_results,
                "pooled_net_pnl": float(pooled_pnl),
                "pooled_selected_candidates": int(pooled_selected),
                "pooled_trades_executed": int(pooled_trades),
                "profitable_folds": int(profitable_folds),
                "max_drawdown_pct": float(max_drawdown_pct),
                "mean_trades_per_day": mean_trades_per_day,
                "null_pooled_mean": null["mean"],
                "null_pooled_std": null["std"],
                "null_pooled_p95": null["p95"],
                "pooled_null_z": float(z),
                "gates": gates,
                "all_reported_gates_pass": bool(all(gates.values())),
            }

    csv_path = args.out_dir / "heuristic_fold_results.csv"
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)

    winners: list[dict[str, Any]] = []
    for policy_idx, heuristics in policy_results.items():
        for heuristic, result in heuristics.items():
            winners.append(
                {
                    "policy_idx": int(policy_idx),
                    "heuristic": heuristic,
                    "pooled_net_pnl": result["pooled_net_pnl"],
                    "pooled_null_z": result["pooled_null_z"],
                    "max_drawdown_pct": result["max_drawdown_pct"],
                    "mean_trades_per_day": result["mean_trades_per_day"],
                    "all_reported_gates_pass": result["all_reported_gates_pass"],
                }
            )
    winners = sorted(winners, key=lambda row: (row["all_reported_gates_pass"], row["pooled_net_pnl"]), reverse=True)
    side_effects = preregistration["side_effects"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete",
        "highest_allowed_claim": "canonical v1 fixed-heuristic baseline packet complete",
        "contract": CONTRACT,
        "canonical_selection_contract": CANONICAL_SELECTION_CONTRACT,
        "sessions": len(sessions),
        "candidate_rows": int(len(table.pnl)),
        "null_summary": str(args.null_summary),
        "design_path": str(training_scope.design_path),
        "manifest_path": str(training_scope.manifest_path),
        "acceptance_registry_path": str(training_scope.acceptance_registry_path),
        "acceptance_registry_hash": training_scope.acceptance_registry_hash,
        "fold_governance_hash": training_scope.fold_governance_hash,
        "preregistration_hash": preregistration["preregistration_hash"],
        "policy_results": policy_results,
        "ranked_heuristics": winners,
        "side_effects": side_effects,
    }
    write_json(args.out_dir / "summary.json", summary)

    lines = [
        "# Protocol101 Canonical V1 Fixed-Heuristic Baselines",
        "",
        f"- Status: `{summary['status']}`",
        f"- Sessions: `{len(sessions)}`",
        f"- Candidate rows: `{len(table.pnl)}`",
        f"- Null canary: `{args.null_summary}`",
        f"- Highest allowed claim: {summary['highest_allowed_claim']}",
        "",
        "## Top Results",
        "",
        "| Rank | Policy | Heuristic | PnL | Null z | Max DD % | Trades/day | Gates |",
        "|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(winners[:12], start=1):
        lines.append(
            "| {rank} | {policy_idx} | {heuristic} | ${pooled_net_pnl:,.0f} | {pooled_null_z:.2f} | {dd:.2%} | {tpd:.2f} | {gates} |".format(
                rank=rank,
                policy_idx=row["policy_idx"],
                heuristic=row["heuristic"],
                pooled_net_pnl=row["pooled_net_pnl"],
                pooled_null_z=row["pooled_null_z"],
                dd=row["max_drawdown_pct"],
                tpd=row["mean_trades_per_day"],
                gates="pass" if row["all_reported_gates_pass"] else "fail",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This packet is a fixed-rule baseline, not a trained candidate. It is useful as a G3 reference and as a probe for whether simple canonical features carry obvious edge before model search.",
            "",
            "The script writes no model artifacts and records all broker, paper, paid-data, launchd, runtime, promotion, and real-money side-effect flags as false.",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "complete", "out_dir": str(args.out_dir), "top_result": winners[0] if winners else None}, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
