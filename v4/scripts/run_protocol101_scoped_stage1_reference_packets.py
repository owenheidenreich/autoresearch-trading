"""Legacy-v4 compatibility CLI plus fresh-v5 Stage-1 reference wrappers.

The historical CLI is retained only to reproduce its old audit namespaces and
simulator-v4 evidence. It is not current evidence. Fresh callers must use the
v5 wrappers in this module, which write only into separately supplied
``protocol101_full_trader_stage1_reference_v5_*`` namespaces.
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

from v4.model.protocol101_canonical_stage1_contract import (
    CONTRACT_ID,
    boundary_stable_mask,
)
from v4.model.protocol101_serial_simulator import (
    SerialCandidate,
    SerialSimulatorConfig,
    simulate_serial_candidates,
)
from v4.model.protocol101_stage1_reference_multiplicity import (
    MATCHED_RANDOM_DRAWS,
    MATCHED_RANDOM_SEED,
    MatchedRandomSchedule,
    ReferenceOpportunity,
    generate_matched_random_schedule,
    run_fixed_heuristic_reference,
    run_matched_random_reference,
)
from v4.scripts.protocol101_training_scope import load_training_scope
from v4.scripts.run_protocol101_owned_raw_acceptance_verifier import (
    PINNED_LABEL_POLICIES,
    stable_hash,
)


BASE_AUDIT = Path("v4/audit/autoresearch")
NULL_OUT = BASE_AUDIT / "protocol101_scoped_canonical_stage1_null_canary"
HEURISTIC_OUT = BASE_AUDIT / "protocol101_scoped_canonical_stage1_heuristic_baseline"
INTERSECTION_SUMMARY = (
    BASE_AUDIT / "protocol101_canonical_v1_intersection_guard_audit/summary.json"
)
FEE = 3.0
FEE_SENSITIVITY = (2.60, 4.00)
RANDOM_DRAWS = 200
RANDOM_SEED = 101
DAILY_LOSS_FRACTION = 0.05
LEGACY_COMPATIBILITY_MODE = "legacy-v4-compatibility"
FRESH_REFERENCE_NAMESPACE_PREFIX = (
    "protocol101_full_trader_stage1_reference_v5_"
)


@dataclass(frozen=True)
class Candidate:
    contract_id: str
    right: str
    offset: float
    entry_ask: float
    pnl_by_policy: tuple[float, ...]


@dataclass(frozen=True)
class Minute:
    session: str
    decision_time: pd.Timestamp
    candidates: tuple[Candidate, ...]
    vwap_side: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(LEGACY_COMPATIBILITY_MODE,),
        default=LEGACY_COMPATIBILITY_MODE,
        help=(
            "Historical simulator-v4 compatibility only; fresh v5 machinery "
            "uses run_protocol101_full_trader_stage1_reference_"
            "multiplicity_validation."
        ),
    )
    parser.add_argument("--null-out", type=Path, default=NULL_OUT)
    parser.add_argument("--heuristic-out", type=Path, default=HEURISTIC_OUT)
    parser.add_argument("--random-draws", type=int, default=RANDOM_DRAWS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_fresh_v5_matched_random_reference(
    opportunities: list[ReferenceOpportunity],
    *,
    policy_index: int,
    draws: int = MATCHED_RANDOM_DRAWS,
    seed: int = MATCHED_RANDOM_SEED,
) -> tuple[MatchedRandomSchedule, dict[str, Any]]:
    """Build one fresh matched-random packet through v5 only."""

    schedule = generate_matched_random_schedule(
        opportunities,
        policy_index=policy_index,
        draws=draws,
        seed=seed,
    )
    return schedule, run_matched_random_reference(opportunities, schedule)


def build_fresh_v5_fixed_heuristic_reference(
    opportunities: list[ReferenceOpportunity],
) -> dict[str, Any]:
    """Build the fresh fixed-P5 heuristic packet through v5 only."""

    return run_fixed_heuristic_reference(opportunities)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def policy_holds() -> dict[int, float]:
    return {
        int(item["policy_idx"]): float(item["max_hold_minutes"])
        for item in PINNED_LABEL_POLICIES
    }


def build_minutes(paths: list[Path], margins: dict[str, float]) -> list[Minute]:
    minutes: list[Minute] = []
    for path in paths:
        session = path.stem
        rows = pickle.load(path.open("rb"))
        for row in rows:
            mask = boundary_stable_mask(row, margins)
            labels = np.asarray(row.get("labels_net_pnl"), dtype=float)
            ids = np.asarray(row.get("contract_ids"), dtype=object)
            offsets = np.asarray(row.get("strike_offsets"), dtype=float)
            rights = tuple(str(value) for value in (row.get("rights") or ("C", "P")))
            metadata = row.get("contract_quote_metadata") or {}
            candidates: list[Candidate] = []
            for strike_idx, right_idx in np.argwhere(mask):
                values = tuple(float(value) for value in labels[strike_idx, right_idx])
                if not all(math.isfinite(value) for value in values):
                    continue
                cid = str(ids[strike_idx, right_idx])
                ask = (metadata.get(cid) or {}).get("ask")
                if ask is None or not math.isfinite(float(ask)) or float(ask) <= 0.0:
                    continue
                candidates.append(
                    Candidate(
                        contract_id=cid,
                        right=rights[int(right_idx)],
                        offset=float(offsets[int(strike_idx)]),
                        entry_ask=float(ask),
                        pnl_by_policy=values,
                    )
                )
            if not candidates:
                continue
            market = np.asarray(row.get("market_window"), dtype=float)[-1]
            market_names = tuple(row.get("market_feature_names") or ())
            spx = float(market[market_names.index("spx_close")])
            vwap = float(market[market_names.index("spx_vwap")])
            minutes.append(
                Minute(
                    session=session,
                    decision_time=pd.Timestamp(row.get("decision_time")),
                    candidates=tuple(candidates),
                    vwap_side="C" if spx >= vwap else "P",
                )
            )
    return minutes


def serial_candidates(
    minutes: list[Minute],
    *,
    policy_idx: int,
    fee: float,
    choices: np.ndarray | None = None,
    heuristic: bool = False,
) -> list[SerialCandidate]:
    hold = policy_holds()[policy_idx]
    out: list[SerialCandidate] = []
    for minute_idx, minute in enumerate(minutes):
        if heuristic:
            side = [item for item in minute.candidates if item.right == minute.vwap_side]
            if not side:
                continue
            chosen = min(
                side,
                key=lambda item: (
                    abs(item.offset),
                    item.offset,
                    0 if item.right == "C" else 1,
                    item.contract_id,
                ),
            )
        else:
            assert choices is not None
            chosen = minute.candidates[int(choices[minute_idx]) % len(minute.candidates)]
        out.append(
            SerialCandidate(
                split="validation",
                session=minute.session,
                decision_time=minute.decision_time.to_pydatetime(),
                contract_id=chosen.contract_id,
                right=chosen.right,
                offset=chosen.offset,
                entry_ask=chosen.entry_ask,
                score=0.0,
                raw_label_pnl=float(chosen.pnl_by_policy[policy_idx]) - float(fee),
                cooldown_minutes=hold,
                max_hold_minutes=hold,
                strategy="random_null" if not heuristic else "vwap_side_nearest_atm",
            )
        )
    return out


def replay(
    candidates: list[SerialCandidate],
    *,
    fee: float,
) -> dict[str, Any]:
    trades, state = simulate_serial_candidates(
        candidates,
        config=SerialSimulatorConfig(
            starting_cash=10_000.0,
            max_daily_loss_fraction_of_session_start_equity=DAILY_LOSS_FRACTION,
            affordability_reserve_per_trade=float(fee),
        ),
    )
    pnl = np.asarray([trade.raw_label_pnl for trade in trades], dtype=float)
    equity = np.asarray(state.equity_by_account.get("validation", [10_000.0]), dtype=float)
    peak = np.maximum.accumulate(equity)
    return {
        "fee": fee,
        "trades": len(trades),
        "net_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "ending_equity": float(equity[-1]),
        "max_drawdown": float(np.max(peak - equity)) if len(equity) else 0.0,
        "min_equity": float(np.min(equity)) if len(equity) else 10_000.0,
        "daily_loss_stop_blocks": int(state.skipped["daily_loss_stop"]),
        "simulator_semantics": state.semantics,
    }


def preregistration(scope, random_draws: int) -> dict[str, Any]:
    payload = {
        "schema_version": "Protocol101ScopedStage1ReferencePreregistrationV1",
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "contract_id": CONTRACT_ID,
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "validation_sessions": fold["validation_sessions"],
            }
            for fold in scope.folds
        ],
        "random_null": {
            "draws": int(random_draws),
            "seed": RANDOM_SEED,
            "selection": "uniform one boundary-stable candidate per decision minute",
        },
        "heuristic": {
            "name": "vwap_side_nearest_atm_plus_best_single_shape",
            "side": "call when SPX >= completed-minute VWAP, otherwise put",
            "slot": "nearest ATM boundary-stable candidate; deterministic tie-break",
            "shape_selection": "maximum pooled fee-adjusted PnL across the frozen seven shapes",
        },
        "fees": {"primary": FEE, "sensitivity": list(FEE_SENSITIVITY)},
        "daily_loss_fraction_of_session_start_equity": DAILY_LOSS_FRACTION,
        "labels": "labels_net_pnl pessimistic ask-in/bid-out minus fee overlay",
        "results_inspected_before_preregistration": False,
    }
    payload["preregistration_hash"] = stable_hash(payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.mode != LEGACY_COMPATIBILITY_MODE:
        raise SystemExit("only explicit legacy-v4 compatibility is available")
    for out_dir in (args.null_out, args.heuristic_out):
        if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
            raise SystemExit(f"{out_dir} exists; pass --force to overwrite")
        out_dir.mkdir(parents=True, exist_ok=True)

    scope = load_training_scope()
    prereg = preregistration(scope, args.random_draws)
    write_json(args.null_out / "preregistration.json", prereg)
    write_json(args.heuristic_out / "preregistration.json", prereg)
    margins = load_json(INTERSECTION_SUMMARY)["vendor_only_training_guard_policy"][
        "use_boundary_stable_margins"
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    null_policy_results: dict[str, Any] = {}
    heuristic_policy_results: dict[str, Any] = {}
    for policy_idx in sorted(policy_holds()):
        fold_minutes = []
        for fold in scope.folds:
            paths = [
                dict(scope.sessions)[session]
                for session in fold["validation_sessions"]
            ]
            fold_minutes.append(build_minutes(paths, margins))

        heuristic_folds = []
        for fold_idx, minutes in enumerate(fold_minutes):
            result = replay(
                serial_candidates(
                    minutes, policy_idx=policy_idx, fee=FEE, heuristic=True
                ),
                fee=FEE,
            )
            heuristic_folds.append({"fold": fold_idx, **result})
        heuristic_policy_results[str(policy_idx)] = {
            "folds": heuristic_folds,
            "pooled_net_pnl": float(sum(item["net_pnl"] for item in heuristic_folds)),
            "profitable_folds": int(sum(item["net_pnl"] > 0.0 for item in heuristic_folds)),
            "pooled_trades": int(sum(item["trades"] for item in heuristic_folds)),
            "minimum_fold_equity": float(min(item["min_equity"] for item in heuristic_folds)),
        }

        draw_totals: list[float] = []
        for _draw in range(int(args.random_draws)):
            pooled = 0.0
            for minutes in fold_minutes:
                choices = np.asarray(
                    [rng.integers(0, len(minute.candidates)) for minute in minutes],
                    dtype=int,
                )
                result = replay(
                    serial_candidates(
                        minutes,
                        policy_idx=policy_idx,
                        fee=FEE,
                        choices=choices,
                    ),
                    fee=FEE,
                )
                pooled += float(result["net_pnl"])
            draw_totals.append(pooled)
        values = np.asarray(draw_totals, dtype=float)
        null_policy_results[str(policy_idx)] = {
            "pooled_random_net_pnl": {
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
        }
        print(
            json.dumps(
                {
                    "policy_idx": policy_idx,
                    "null_draws": len(values),
                    "heuristic_pnl": heuristic_policy_results[str(policy_idx)][
                        "pooled_net_pnl"
                    ],
                }
            ),
            flush=True,
        )

    ranked = sorted(
        (
            {"policy_idx": int(policy_idx), **result}
            for policy_idx, result in heuristic_policy_results.items()
        ),
        key=lambda item: (-item["pooled_net_pnl"], item["policy_idx"]),
    )
    common = {
        "contract_id": CONTRACT_ID,
        "fold_governance_hash": scope.fold_governance_hash,
        "acceptance_registry_hash": scope.acceptance_registry_hash,
        "fold_eligible_sessions": len(scope.sessions),
        "protected_holdout_read": False,
        "side_effects": {
            "model_training_executed": False,
            "threshold_selection_executed": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "real_money_path_changed": False,
        },
    }
    null_summary = {
        "schema_version": "Protocol101ScopedStage1NullCanaryV1",
        "status": "pass",
        "evidence_status": "historical_legacy_v4_compatibility_only",
        "legacy_compatibility_mode": LEGACY_COMPATIBILITY_MODE,
        "preregistration_hash": prereg["preregistration_hash"],
        "policy_results": null_policy_results,
        **common,
    }
    heuristic_summary = {
        "schema_version": "Protocol101ScopedStage1HeuristicBaselineV1",
        "status": "pass",
        "evidence_status": "historical_legacy_v4_compatibility_only",
        "legacy_compatibility_mode": LEGACY_COMPATIBILITY_MODE,
        "preregistration_hash": prereg["preregistration_hash"],
        "heuristic": "vwap_side_nearest_atm_plus_best_single_shape",
        "policy_results": heuristic_policy_results,
        "ranked_policies": ranked,
        "g3_fixed_baseline": ranked[0],
        **common,
    }
    write_json(args.null_out / "summary.json", null_summary)
    write_json(args.heuristic_out / "summary.json", heuristic_summary)
    (args.null_out / "report.md").write_text(
        "# Scoped Stage-1 Random Null - Legacy v4 Compatibility\n\n"
        "- This packet is historical and is not current Stage-1 evidence.\n"
        f"- Draws per policy: `{args.random_draws}`\n"
        f"- Contract: `{CONTRACT_ID}`\n"
        "- No model training or threshold selection occurred.\n"
    )
    best = ranked[0]
    (args.heuristic_out / "report.md").write_text(
        "# Scoped Stage-1 Fixed Heuristic - Legacy v4 Compatibility\n\n"
        "- This packet is historical and is not current Stage-1 evidence.\n"
        f"- Contract: `{CONTRACT_ID}`\n"
        f"- Fixed G3 baseline policy: `{best['policy_idx']}`\n"
        f"- Pooled fee-adjusted PnL: `${best['pooled_net_pnl']:,.2f}`\n"
        "- No model training or threshold selection occurred.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
