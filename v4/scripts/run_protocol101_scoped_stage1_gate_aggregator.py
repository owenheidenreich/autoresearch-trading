"""Aggregate a complete fresh Protocol101 Stage-1 campaign under FT1C."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from v4.model.protocol101_serial_simulator_v5 import (
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.model.protocol101_stage1_gate_contract import (
    AGGREGATION_SCHEMA,
    FOLDS,
    HYPOTHESES,
    POLICIES,
    PRIMARY_FEE,
    REFERENCE_ACCEPTANCE_ROUTE,
    ROWS,
    SEEDS,
    D6_ROUTE,
    evaluate_row_gates,
    stable_hash,
    ten_bin_ece,
    validate_campaign_packet,
    verify_frozen_acceptance_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-packet", type=Path, required=True)
    parser.add_argument(
        "--execution-provenance-authority",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--execution-provenance-authority-sha256",
        required=True,
    )
    parser.add_argument("--control-authority", type=Path, required=True)
    parser.add_argument("--control-authority-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def deserialize_candidate(payload: Mapping[str, Any]) -> SerialCandidateV5:
    fields = {
        name: value
        for name, value in payload.items()
        if name in SerialCandidateV5.__dataclass_fields__
    }
    return SerialCandidateV5(**fields)


def _candidate_at_fee(
    candidate: SerialCandidateV5,
    fee: float,
) -> SerialCandidateV5:
    pnl = (
        (candidate.label_executable_exit_bid - candidate.entry_ask) * 100.0
        - fee
    )
    return replace(candidate, raw_label_pnl_after_campaign_fee=float(pnl))


def _drawdown_and_minimum(
    equity_events: Iterable[Mapping[str, Any]],
    *,
    starting_cash: float = 10_000.0,
) -> tuple[float, float]:
    equities = [starting_cash]
    equities.extend(float(item["equity"]) for item in equity_events)
    peak = equities[0]
    maximum = 0.0
    for equity in equities:
        peak = max(peak, equity)
        maximum = max(maximum, peak - equity)
    return float(maximum), float(min(equities))


def _replay(
    candidates: Sequence[SerialCandidateV5],
    *,
    fee: float = PRIMARY_FEE,
) -> dict[str, Any]:
    adjusted = [_candidate_at_fee(candidate, fee) for candidate in candidates]
    trades, state = simulate_serial_candidates_v5(
        adjusted,
        config=SerialSimulatorV5Config(
            campaign_round_trip_fee_dollars=fee,
            affordability_reserve_per_trade=fee,
        ),
    )
    events = next(iter(state.equity_events_by_account.values()), ())
    max_drawdown, minimum_equity = _drawdown_and_minimum(events)
    return {
        "fee_adjusted_net_pnl": float(
            sum(trade.raw_label_pnl_after_campaign_fee for trade in trades)
        ),
        "max_drawdown": max_drawdown,
        "minimum_equity": minimum_equity,
        "trade_count": len(trades),
        "trade_identity_hash": state.trade_identity_hash,
        "candidate_stream_hash": state.candidate_stream_hash,
        "candidate_payload_hash": state.candidate_payload_hash,
        "simulator_version": state.semantics["simulator_version"],
        "skipped": dict(state.skipped),
        "skipped_events": [
            {
                "candidate_identity": list(event.candidate_identity),
                "reason_code": event.reason_code,
                "decision_time_ns": event.decision_time_ns,
                "pending_contract_id_or_null": (
                    event.pending_contract_id_or_null
                ),
                "pending_source_quote_time_ns_or_null": (
                    event.pending_source_quote_time_ns_or_null
                ),
                "pending_realized_exit_time_ns_or_null": (
                    event.pending_realized_exit_time_ns_or_null
                ),
            }
            for event in state.skipped_events
        ],
        "trades": [
            {
                "session": trade.session,
                "decision_time_ns": trade.decision_time_ns,
                "contract_id": trade.contract_id,
                "label_source_exit_quote_time_ns": (
                    trade.label_source_exit_quote_time_ns
                ),
                "label_realized_exit_time_ns": (
                    trade.label_realized_exit_time_ns
                ),
                "raw_label_pnl_after_campaign_fee": (
                    trade.raw_label_pnl_after_campaign_fee
                ),
            }
            for trade in trades
        ],
    }


def _control_status(packet: Mapping[str, Any]) -> dict[str, Any]:
    controls = packet["controls"]
    d1 = controls["D1"]
    d5 = controls["D5"]
    d6 = controls["D6"]
    maxt = controls["maxT"]
    d1_pass = bool(
        d1["independently_accepted"]
        and d1["passes"]
        and d1["joint_G1_G2_pass_count"] <= 1
        and d1["median_pnl"] <= 0.0
        and d1["median_z"] < 1.0
    )
    d5_pass = bool(
        d5["independently_accepted"]
        and d5["identity_complete"]
        and d5["acceptance_route"] == REFERENCE_ACCEPTANCE_ROUTE
    )
    d6_pass = bool(
        d6["independently_accepted"]
        and d6["route"] == D6_ROUTE
        and d6["acceptance_route"] == REFERENCE_ACCEPTANCE_ROUTE
    )
    maxt_valid = bool(maxt["valid"])
    return {
        "D1": d1_pass,
        "D5": d5_pass,
        "D6": d6_pass,
        "maxT_valid": maxt_valid,
        "all_global_controls_pass": bool(
            d1_pass and d5_pass and d6_pass and maxt_valid
        ),
    }


def _invalid_result(
    blockers: Sequence[str],
    *,
    packet_hash: str | None,
    execution_authority_sha256: str | None = None,
    control_authority_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATION_SCHEMA,
        "valid": False,
        "campaign_packet_sha256": packet_hash,
        "execution_provenance_authority_sha256": (
            execution_authority_sha256
        ),
        "control_authority_sha256": control_authority_sha256,
        "blockers": sorted(set(blockers)),
        "global_controls": {
            "D1": False,
            "D5": False,
            "D6": False,
            "maxT_valid": False,
            "all_global_controls_pass": False,
        },
        "rows": [
            {
                "row_id": row_id,
                "hard_gate_eligible": False,
                "multiplicity_adjusted_real_entry_signal": False,
                "status": "campaign_invalid",
            }
            for row_id in ROWS
        ],
        "G8_role": "report_only",
        "G9_executed": False,
        "aggregation_sha256": None,
    }


def _validate_primary_diagnostics(
    unit: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> list[str]:
    identity = str(unit["unit_id"])
    diagnostics = unit["diagnostics"]
    pnl = float(replay["fee_adjusted_net_pnl"])
    blockers: list[str] = []
    comparisons = {
        "fee_3.00": diagnostics["fee_sensitivity"]["3.00"],
        "pessimistic_fill": diagnostics["fill_edge_band"][
            "pessimistic_executable"
        ],
        "noise_1.0x": diagnostics["noise_diagnostics"]["1.0x"],
    }
    for label, value in comparisons.items():
        if not math.isclose(float(value), pnl, rel_tol=0.0, abs_tol=1e-9):
            blockers.append(f"primary_diagnostic_mismatch:{identity}:{label}")
    return blockers


def _seed_result(
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    units: Sequence[Mapping[str, Any]],
    matched_null_z: float,
) -> tuple[dict[str, Any], list[str]]:
    fold_results: list[dict[str, Any]] = []
    all_candidates: list[SerialCandidateV5] = []
    all_observations: list[Mapping[str, Any]] = []
    blockers: list[str] = []
    fee_totals = {"2.60": 0.0, "3.00": 0.0, "4.00": 0.0}
    for unit in units:
        candidates = [
            deserialize_candidate(candidate) for candidate in unit["candidates"]
        ]
        replay = _replay(candidates)
        blockers.extend(_validate_primary_diagnostics(unit, replay))
        all_candidates.extend(candidates)
        all_observations.extend(unit["calibration_observations"])
        session_count = len(unit["session_ids"])
        fold_result = {
            "fold": int(unit["fold"]),
            "era": str(unit["era"]),
            "fee_adjusted_net_pnl": replay["fee_adjusted_net_pnl"],
            "max_drawdown": replay["max_drawdown"],
            "minimum_equity": replay["minimum_equity"],
            "trade_count": replay["trade_count"],
            "session_count": session_count,
            "trades_per_session": replay["trade_count"] / session_count,
            "three_per_day_rail_report_only": (
                replay["trade_count"] / session_count <= 3.0
            ),
            "trade_identity_hash": replay["trade_identity_hash"],
            "candidate_stream_hash": replay["candidate_stream_hash"],
            "candidate_payload_hash": replay["candidate_payload_hash"],
            "skipped": replay["skipped"],
            "skipped_events": replay["skipped_events"],
            "source_and_realized_clocks": [
                {
                    "trade_id": candidate["trade_id"],
                    "source_exit_quote_time_ns": candidate[
                        "label_source_exit_quote_time_ns"
                    ],
                    "realized_exit_time_ns": candidate[
                        "label_realized_exit_time_ns"
                    ],
                }
                for candidate in unit["candidates"]
            ],
            "diagnostics": unit["diagnostics"],
        }
        fold_results.append(fold_result)
        for fee in (2.6, 3.0, 4.0):
            fee_totals[f"{fee:.2f}"] += _replay(candidates, fee=fee)[
                "fee_adjusted_net_pnl"
            ]
    pooled = _replay(all_candidates)
    ece = ten_bin_ece(all_observations)
    return (
        {
            "seed": seed,
            "hypothesis": hypothesis,
            "policy": policy,
            "fold_results": fold_results,
            "continuous_oof_replay": pooled,
            "matched_null_z": float(matched_null_z),
            "ece": ece,
            "ece_observations": len(all_observations),
            "calibration_observations": list(all_observations),
            "fee_sensitivity_pooled_net_pnl": fee_totals,
        },
        blockers,
    )


def aggregate_campaign(
    packet: Mapping[str, Any],
    *,
    execution_authority: Mapping[str, Any] | None = None,
    execution_authority_sha256: str | None = None,
    control_authority: Mapping[str, Any] | None = None,
    control_authority_sha256: str | None = None,
) -> dict[str, Any]:
    structural_blockers = validate_campaign_packet(
        packet,
        execution_authority=execution_authority,
        execution_authority_sha256=execution_authority_sha256,
        control_authority=control_authority,
        control_authority_sha256=control_authority_sha256,
    )
    packet_hash = packet.get("packet_sha256")
    if structural_blockers:
        return _invalid_result(
            structural_blockers,
            packet_hash=packet_hash,
            execution_authority_sha256=execution_authority_sha256,
            control_authority_sha256=control_authority_sha256,
        )
    controls = _control_status(packet)
    unit_map = {
        (
            unit["hypothesis"],
            unit["policy"],
            int(unit["seed"]),
            int(unit["fold"]),
        ): unit
        for unit in packet["units"]
    }
    reference_map = {
        row["row_id"]: row for row in packet["references"]["rows"]
    }
    maxT_map = {
        row["row_id"]: row for row in packet["controls"]["maxT"]["rows"]
    }
    rows: list[dict[str, Any]] = []
    replay_blockers: list[str] = []
    try:
        for hypothesis in HYPOTHESES:
            for policy in POLICIES:
                row_id = f"{hypothesis}/{policy}"
                reference = reference_map[row_id]
                seed_results: list[dict[str, Any]] = []
                for seed in SEEDS:
                    units = [
                        unit_map[(hypothesis, policy, seed, fold)]
                        for fold in FOLDS
                    ]
                    seed_result, blockers = _seed_result(
                        hypothesis=hypothesis,
                        policy=policy,
                        seed=seed,
                        units=units,
                        matched_null_z=reference["matched_null_z_by_seed"][
                            str(seed)
                        ],
                    )
                    seed_results.append(seed_result)
                    replay_blockers.extend(blockers)
                gate_result = evaluate_row_gates(
                    seed_results,
                    heuristic_pooled_pnl=reference["heuristic_pooled_pnl"],
                    maxT_pass=bool(maxT_map[row_id]["hard_pass"]),
                    global_controls_pass=controls[
                        "all_global_controls_pass"
                    ],
                )
                rows.append(
                    {
                        "row_id": row_id,
                        "hypothesis": hypothesis,
                        "policy": policy,
                        "policy_index": int(policy[1:]),
                        "seed_results": seed_results,
                        "heuristic_pooled_pnl": reference[
                            "heuristic_pooled_pnl"
                        ],
                        "maxT": dict(maxT_map[row_id]),
                        **gate_result,
                    }
                )
    except Exception as exc:
        replay_blockers.append(
            f"v5_replay_or_reconstruction_failed:{type(exc).__name__}:{exc}"
        )
    if replay_blockers:
        return _invalid_result(
            replay_blockers,
            packet_hash=packet_hash,
            execution_authority_sha256=execution_authority_sha256,
            control_authority_sha256=control_authority_sha256,
        )
    result = {
        "schema_version": AGGREGATION_SCHEMA,
        "valid": bool(controls["all_global_controls_pass"]),
        "campaign_packet_sha256": packet_hash,
        "execution_provenance_authority_sha256": (
            execution_authority_sha256
        ),
        "control_authority_sha256": control_authority_sha256,
        "blockers": (
            []
            if controls["all_global_controls_pass"]
            else [
                name
                for name in ("D1", "D5", "D6", "maxT_valid")
                if not controls[name]
            ]
        ),
        "global_controls": controls,
        "rows": rows,
        "row_count": len(rows),
        "unit_count": len(packet["units"]),
        "G8_role": "report_only",
        "G9_executed": False,
        "real_campaign_economics_executed": False,
        "aggregation_sha256": None,
    }
    result["aggregation_sha256"] = stable_hash(
        {**result, "aggregation_sha256": None}
    )
    return result


# Backward-compatible public name for callers that import the old module.
aggregate = aggregate_campaign


def main() -> int:
    args = parse_args()
    workspace = Path(__file__).resolve().parents[2]
    frozen = verify_frozen_acceptance_inputs(workspace)
    if not frozen["valid"]:
        result = _invalid_result(
            frozen["blockers"],
            packet_hash=None,
        )
    else:
        result = aggregate_campaign(
            load_json(args.campaign_packet),
            execution_authority=load_json(
                args.execution_provenance_authority
            ),
            execution_authority_sha256=(
                args.execution_provenance_authority_sha256
            ),
            control_authority=load_json(args.control_authority),
            control_authority_sha256=args.control_authority_sha256,
        )
    write_json(args.out_dir / "aggregation.json", result)
    write_json(
        args.out_dir / "summary.json",
        {
            "schema_version": "Protocol101FT1CGateAggregationSummaryV1",
            "valid": result["valid"],
            "row_count": len(result["rows"]),
            "eligible_row_count": sum(
                bool(row.get("hard_gate_eligible")) for row in result["rows"]
            ),
            "G8_role": "report_only",
            "G9_executed": False,
            "blockers": result["blockers"],
        },
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
