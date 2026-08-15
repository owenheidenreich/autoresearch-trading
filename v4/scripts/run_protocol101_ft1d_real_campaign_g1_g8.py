"""Assemble and evaluate the owner-authorized Protocol101 FT1D campaign.

This producer stops at G1-G8. It never audits, ranks, selects, runs G9, or
touches HOLD/EXIT, protected evidence, broker, paper, or runtime state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from v4.model import protocol101_stage1_gate_contract as gate
from v4.model import protocol101_stage1_reference_multiplicity as references
from v4.model import protocol101_scoped_stage1_hgb as hgb_core
from v4.model import protocol101_fresh_model_artifact as model_artifact
from v4.model import protocol101_fresh_unit_summary as unit_summary
from v4.model.protocol101_canonical_stage1_contract import (
    HYPOTHESES,
)
from v4.model.protocol101_divergence_noise import DivergenceNoiseModel
from v4.model.protocol101_repair_artifacts import verify_replay_packet
from v4.model.protocol101_regimen_repair import (
    assert_processed_row_identities,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    RepairedCanonicalDecision,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.model.protocol101_scoped_stage1_hgb import HGBUnitConfig
from v4.model.protocol101_stage1_controller_journal import (
    append_checkpoint,
    validate_journal,
)
from v4.scripts.run_protocol101_fair_contract_training_runner import (
    DEFAULT_ERA_MANIFEST,
)
from v4.scripts import run_protocol101_scoped_stage1_gate_aggregator as aggregator
from v4.scripts import run_protocol101_scoped_stage1_hgb_runner as runner
from v4.scripts.protocol101_training_scope import load_training_scope


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "v4/audit/autoresearch"
FITTED_ROOT = (
    AUDIT_ROOT / "protocol101_full_trader_stage1_entry_fresh_attempt001"
)
EXECUTION_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_execution_attempt001"
)
OUTPUT_ROOT = (
    AUDIT_ROOT
    / "protocol101_full_trader_stage1_entry_campaign_g1_g8_attempt001"
)
JOURNAL_PATH = EXECUTION_ROOT / "controller_journal.jsonl"
MATERIALIZATION_RECEIPT = (
    EXECUTION_ROOT / "two_clock_materialization_receipt.json"
)
UNIT_PACKET_PATH = OUTPUT_ROOT / "real_unit_campaign_packet.json"
EXECUTION_AUTHORITY_PATH = OUTPUT_ROOT / "execution_provenance_authority.json"
EXECUTION_VALIDATION_PATH = (
    OUTPUT_ROOT / "execution_provenance_validation.json"
)
REFERENCES_DETAIL_PATH = OUTPUT_ROOT / "real_v5_references_detail.json"
REFERENCES_CONTROL_PATH = OUTPUT_ROOT / "real_v5_references_d1_d5_d6.json"
REFERENCES_VALIDATION_PATH = OUTPUT_ROOT / "real_v5_references_validation.json"
REFERENCE_ROWS_ROOT = OUTPUT_ROOT / "references" / "rows"
FIXED_REFERENCE_PATH = OUTPUT_ROOT / "references" / "fixed_heuristic.json"
D1_ROOT = OUTPUT_ROOT / "D1"
D1_DETAIL_PATH = D1_ROOT / "D1_detail.json"
D5_DETAIL_PATH = OUTPUT_ROOT / "D5" / "D5_detail.json"
D6_DETAIL_PATH = OUTPUT_ROOT / "D6" / "D6_detail.json"
CONTROL_AUTHORITY_PATH = OUTPUT_ROOT / "control_authority.json"
CONTROL_AUTHORITY_VALIDATION_PATH = (
    OUTPUT_ROOT / "control_authority_validation.json"
)
MAXT_ROOT = OUTPUT_ROOT / "maxT"
MAXT_GRID_PATH = MAXT_ROOT / "maxT_grid.json"
MAXT_SCHEDULE_PATH = MAXT_ROOT / "frozen_schedule.npy"
MAXT_SCHEDULE_MANIFEST_PATH = MAXT_ROOT / "frozen_schedule_manifest.json"
MAXT_MAX_NULL_PATH = MAXT_ROOT / "max_null.npy"
MAXT_DETAIL_PATH = MAXT_ROOT / "maxT_detail.json"
MAXT_CONTROL_PATH = MAXT_ROOT / "maxT_control.json"
MAXT_VALIDATION_PATH = MAXT_ROOT / "maxT_validation.json"
FINAL_CAMPAIGN_PACKET_PATH = OUTPUT_ROOT / "real_campaign_packet.json"
AGGREGATION_PATH = OUTPUT_ROOT / "g1_g8_aggregation.json"
AGGREGATION_VALIDATION_PATH = OUTPUT_ROOT / "g1_g8_validation.json"
TERMINAL_VALIDATION_PATH = OUTPUT_ROOT / "terminal_validation.json"
ALL_WAIT_CERTIFICATION_PATH = (
    OUTPUT_ROOT / "all_wait_unit_source_certification.json"
)
PROGRESS_PATH = EXECUTION_ROOT / "progress.json"
NY = ZoneInfo("America/New_York")
HYPOTHESIS_ORDER = ("H0", "H1", "H2", "H3")
POLICY_ORDER = tuple(f"P{index}" for index in range(7))
SEED_ORDER = (42, 43, 44)
FOLD_ORDER = (1, 2, 3, 4, 5)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_immutable(path: Path, payload: Any) -> None:
    if path.exists():
        if not path.is_file() or load_json(path) != payload:
            raise RuntimeError(f"immutable_json_conflict:{path}")
        return
    write_json_atomic(path, payload)


def write_bytes_immutable(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"immutable_bytes_conflict:{path}")
        return
    write_bytes_atomic(path, payload)


def update_progress(
    *,
    status: str,
    current_node: str,
    blocker_classification: str | None = None,
    blocker: str | None = None,
) -> None:
    payload = load_json(PROGRESS_PATH) if PROGRESS_PATH.is_file() else {}
    payload.update(
        {
            "status": status,
            "current_node": current_node,
            "blocker_classification": blocker_classification,
            "blocker": blocker,
        }
    )
    write_json_atomic(PROGRESS_PATH, payload)


def _candidate(payload: Mapping[str, Any]) -> SerialCandidateV5:
    return SerialCandidateV5(
        **{
            name: value
            for name, value in payload.items()
            if name in SerialCandidateV5.__dataclass_fields__
        }
    )


def _replay(
    candidates: Iterable[Mapping[str, Any]],
    *,
    fee: float = 3.0,
) -> tuple[list[Any], Any]:
    adjusted = []
    for payload in candidates:
        item = _candidate(payload)
        pnl = (
            (item.label_executable_exit_bid - item.entry_ask) * 100.0
            - fee
        )
        adjusted.append(
            SerialCandidateV5(
                **{
                    **{
                        name: getattr(item, name)
                        for name in SerialCandidateV5.__dataclass_fields__
                    },
                    "raw_label_pnl_after_campaign_fee": float(pnl),
                }
            )
        )
    return simulate_serial_candidates_v5(
        adjusted,
        config=SerialSimulatorV5Config(
            starting_cash=10_000.0,
            campaign_round_trip_fee_dollars=fee,
            affordability_reserve_per_trade=fee,
            max_daily_loss_fraction_of_session_start_equity=0.05,
            no_new_entries_after_et="15:30",
            forced_flat_before_et="15:55",
        ),
    )


def _trade_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(value["session"]),
        int(value["decision_time_ns"]),
        str(value["contract_id"]),
        int(value["policy_index"]),
    )


def _time_bucket(decision_time_ns: int) -> str:
    value = datetime.fromtimestamp(
        int(decision_time_ns) / 1_000_000_000,
        tz=NY,
    )
    minutes = value.hour * 60 + value.minute
    if minutes < 10 * 60 + 30:
        return "opening"
    if minutes < 14 * 60:
        return "midday"
    return "late"


def _concentration(values: Mapping[str, float]) -> float:
    absolute = [abs(float(value)) for value in values.values()]
    total = sum(absolute)
    return max(absolute, default=0.0) / total if total > 0.0 else 0.0


def _diagnostics(
    result: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    trades: list[Any],
    state: Any,
    *,
    policy: str,
) -> dict[str, Any]:
    validation = result["validation"]
    primary_pnl = float(sum(
        item.raw_label_pnl_after_campaign_fee for item in trades
    ))
    fee_sensitivity = {
        key: float(value["net_pnl"])
        for key, value in validation["fee_sensitivity"].items()
    }
    fill = validation["fill_edge_band"]
    mid_pnl = float(fill["mid_path_net_pnl_same_executed_trades"])
    noise = {
        "0.0x": float(validation["noise_diagnostics"]["0.0x"]["metrics"]["net_pnl"]),
        "0.5x": float(validation["noise_diagnostics"]["0.5x"]["metrics"]["net_pnl"]),
        "1.0x": primary_pnl,
        "2.0x": float(validation["noise_diagnostics"]["2.0x"]["metrics"]["net_pnl"]),
    }
    side_time = {
        "call_trades": 0,
        "put_trades": 0,
        "opening": 0,
        "midday": 0,
        "late": 0,
    }
    day_pnl: dict[str, float] = defaultdict(float)
    month_pnl: dict[str, float] = defaultdict(float)
    outcome = {
        "large_loss": 0,
        "small_loss": 0,
        "small_win": 0,
        "large_win": 0,
    }
    for trade in trades:
        side_time["call_trades" if trade.right == "C" else "put_trades"] += 1
        side_time[_time_bucket(trade.decision_time_ns)] += 1
        pnl = float(trade.raw_label_pnl_after_campaign_fee)
        day_pnl[trade.session] += pnl
        month_pnl[trade.session[:7]] += pnl
        if pnl < -100.0:
            outcome["large_loss"] += 1
        elif pnl < 0.0:
            outcome["small_loss"] += 1
        elif pnl < 100.0:
            outcome["small_win"] += 1
        else:
            outcome["large_win"] += 1
    trade_pnl = {
        f"{item.session}:{item.decision_time_ns}:{item.contract_id}": float(
            item.raw_label_pnl_after_campaign_fee
        )
        for item in trades
    }
    executed = {_trade_key(item.__dict__) for item in trades}
    skipped_candidates = [
        item for item in candidates if _trade_key(item) not in executed
    ]
    candidate_positive = sum(
        max(0.0, float(item["raw_label_pnl_after_campaign_fee"]))
        for item in candidates
    )
    executed_positive = sum(
        max(0.0, float(item.raw_label_pnl_after_campaign_fee))
        for item in trades
    )
    events = [
        event
        for account_events in state.equity_events_by_account.values()
        for event in account_events
    ]
    peak = 10_000.0
    underwater_start: int | None = None
    maximum_underwater_ns = 0
    underwater_events = 0
    for event in events:
        equity = float(event["equity"])
        event_time = int(event.get("event_time_ns") or 0)
        if equity >= peak:
            peak = equity
            if underwater_start is not None:
                maximum_underwater_ns = max(
                    maximum_underwater_ns,
                    event_time - underwater_start,
                )
                underwater_start = None
        elif underwater_start is None:
            underwater_start = event_time
            underwater_events += 1
    if underwater_start is not None and events:
        maximum_underwater_ns = max(
            maximum_underwater_ns,
            int(events[-1].get("event_time_ns") or 0) - underwater_start,
        )
    shape = {item: 0 for item in POLICY_ORDER}
    shape[policy] = len(trades)
    return {
        "fee_sensitivity": fee_sensitivity,
        "fill_edge_band": {
            "pessimistic_executable": primary_pnl,
            "mid_diagnostic": mid_pnl,
            "favorable_diagnostic": mid_pnl,
        },
        "noise_diagnostics": noise,
        "side_time_exposure": side_time,
        "concentration": {
            "top_trade_fraction": _concentration(trade_pnl),
            "top_day_fraction": _concentration(day_pnl),
            "top_month_fraction": _concentration(month_pnl),
        },
        "churn": {
            "candidate_count": len(candidates),
            "executed_count": len(trades),
            "overlap_skips": int(state.skipped.get("overlap", 0)),
        },
        "skipped_opportunity": {
            "count": len(skipped_candidates),
            "fee_adjusted_pnl": float(sum(
                item["raw_label_pnl_after_campaign_fee"]
                for item in skipped_candidates
            )),
        },
        "worst_day": min(day_pnl.values(), default=0.0),
        "underwater_duration": {
            "events": underwater_events,
            "maximum_minutes": maximum_underwater_ns / 60_000_000_000,
        },
        "outcome_buckets": outcome,
        "harvest_ratio": (
            executed_positive / candidate_positive
            if candidate_positive > 0.0
            else 0.0
        ),
        "daily_breaker_events": int(state.skipped.get("daily_loss_stop", 0)),
        "shape_usage": shape,
    }


def _era_map() -> dict[str, str]:
    manifest = load_json(ROOT / DEFAULT_ERA_MANIFEST)
    return {
        str(item["session"]): str(item["era"])
        for item in manifest["sessions"]
    }


def _majority_era(sessions: list[str], eras: Mapping[str, str]) -> str:
    counts = Counter(eras[session] for session in sessions)
    return min(counts, key=lambda name: (-counts[name], name))


def _expected_summary_path(
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
) -> Path:
    return (
        FITTED_ROOT
        / hypothesis
        / "units"
        / hypothesis
        / f"policy{int(policy[1:])}"
        / f"seed{seed}"
        / f"expanding_fold_{fold:02d}"
        / "summary.json"
    )


def _unit_provenance(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    summary_path: Path,
) -> dict[str, str]:
    result = summary["unit"]
    source = summary["provenance"]
    return {
        "campaign_hash": str(summary["campaign_contract_sha256"]),
        "fold_hash": str(manifest["fold_governance_hash"]),
        "registry_hash": str(manifest["acceptance_registry_hash"]),
        "feature_hash": stable_hash(result["feature_names"]),
        "simulator_hash": str(manifest["simulator_source_hash"]),
        "schema_hash": stable_hash(
            {
                "replay_schema": manifest["schema_version"],
                "two_clock_schema": manifest["two_clock_schema_version"],
            }
        ),
        "model_hash": str(summary["model_artifact"]["sha256"]),
        "threshold_hash": str(manifest["threshold_hash"]),
        "epsilon_hash": str(manifest["epsilon_hash"]),
        "source_hash": stable_hash(
            {
                "summary_sha256": sha256_path(summary_path),
                "replay_manifest_sha256": sha256_path(
                    Path(summary["replay_packet"]["path"]) / "manifest.json"
                ),
                "two_clock_materialization_receipt_sha256": source[
                    "two_clock_materialization_receipt_sha256"
                ],
            }
        ),
        "code_hash": stable_hash(
            {
                key: source[key]
                for key in (
                    "runner_core_source_sha256",
                    "runner_source_sha256",
                    "scientific_runner_source_sha256",
                    "two_clock_materializer_source_sha256",
                )
            }
        ),
    }


def _gate_candidate(
    payload: Mapping[str, Any],
    *,
    hypothesis: str,
    policy: str,
    seed: int,
    fold: int,
    replay_manifest_sha256: str,
) -> dict[str, Any]:
    result = dict(payload)
    source_split = str(result["split"])
    source_fold = str(result["fold"])
    trade_identity = {
        "hypothesis": hypothesis,
        "policy": policy,
        "seed": seed,
        "fold": fold,
        "session": result["session"],
        "decision_time_ns": result["decision_time_ns"],
        "contract_id": result["contract_id"],
    }
    result["trade_id"] = "FT1D-" + stable_hash(trade_identity)
    result["split"] = f"SYNTH-{hypothesis}-{policy}-S{seed}"
    result["fold"] = f"F{fold}"
    result["metadata"] = {
        **dict(result.get("metadata") or {}),
        "campaign_economic_value": True,
        "real_immutable_replay": True,
        "compatibility_split_token_only": True,
        "source_split": source_split,
        "source_fold": source_fold,
        "replay_manifest_sha256": replay_manifest_sha256,
    }
    return result


def build_real_unit_packet() -> tuple[dict[str, Any], dict[str, Any]]:
    materialization = load_json(MATERIALIZATION_RECEIPT)
    if (
        materialization.get("status")
        != "complete_verified_additive_two_clock_materialization"
        or materialization.get("session_count") != 271
    ):
        raise RuntimeError("two_clock_materialization_receipt_invalid")
    era_by_session = _era_map()
    units: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    model_hashes: set[str] = set()
    for hypothesis in HYPOTHESIS_ORDER:
        hypothesis_summary = load_json(FITTED_ROOT / hypothesis / "summary.json")
        summary_refs = {
            str(item["path"]): str(item["sha256"])
            for item in hypothesis_summary["unit_artifacts"]
        }
        for policy in POLICY_ORDER:
            for seed in SEED_ORDER:
                for fold in FOLD_ORDER:
                    summary_path = _expected_summary_path(
                        hypothesis,
                        policy,
                        seed,
                        fold,
                    )
                    relative = str(summary_path.relative_to(ROOT))
                    reference_sha256 = summary_refs.get(
                        str(summary_path),
                        summary_refs.get(relative),
                    )
                    summary, accepted_reference_sha256s = (
                        _load_verified_fitted_summary_reference(summary_path)
                    )
                    if reference_sha256 not in accepted_reference_sha256s:
                        raise RuntimeError(
                            f"hypothesis_summary_hash_mismatch:{relative}"
                        )
                    is_compact = (
                        summary.get("schema_version")
                        == unit_summary.COMPACT_SUMMARY_SCHEMA
                    )
                    result = summary["unit"]
                    expected_fold_id = f"expanding_fold_{fold:02d}"
                    if (
                        result["hypothesis"] != hypothesis
                        or int(result["policy_index"]) != int(policy[1:])
                        or int(result["seed"]) != seed
                        or result["fold"] != expected_fold_id
                        or summary["fold_id"] != expected_fold_id
                        or result["feature_names"] != list(HYPOTHESES[hypothesis])
                        or result["simulator_version"]
                        != PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
                    ):
                        raise RuntimeError(f"unit_contract_mismatch:{relative}")
                    model_path = Path(summary["model_artifact"]["path"])
                    model_hash = sha256_path(model_path)
                    if model_hash != summary["model_artifact"]["sha256"]:
                        raise RuntimeError(f"model_hash_mismatch:{relative}")
                    fitted_model = pickle.loads(model_path.read_bytes())
                    binding = getattr(
                        fitted_model,
                        model_artifact.MODEL_ARTIFACT_BINDING_ATTRIBUTE,
                        None,
                    )
                    if (
                        not isinstance(binding, Mapping)
                        or binding.get("schema_version")
                        != model_artifact.MODEL_ARTIFACT_BINDING_SCHEMA
                        or binding.get("campaign_namespace")
                        != gate.CAMPAIGN_NAMESPACE
                        or binding.get("unit_identity")
                        != model_path.parent.relative_to(FITTED_ROOT).as_posix()
                        or binding.get("prediction_semantics_changed") is not False
                        or binding.get("scientific_model_state_changed") is not False
                        or binding.get("binding_sources", {}).get(
                            "serializer_source_sha256"
                        )
                        != sha256_path(Path(model_artifact.__file__))
                    ):
                        raise RuntimeError(
                            f"fresh_model_artifact_binding_mismatch:{relative}"
                        )
                    if model_hash in model_hashes:
                        raise RuntimeError(f"fresh_model_hash_duplicate:{relative}")
                    model_hashes.add(model_hash)
                    replay_dir = Path(summary["replay_packet"]["path"])
                    manifest = verify_replay_packet(replay_dir)
                    replay_manifest_sha256 = sha256_path(
                        replay_dir / "manifest.json"
                    )
                    if (
                        replay_manifest_sha256
                        != summary["replay_packet"]["manifest_sha256"]
                    ):
                        raise RuntimeError(
                            f"replay_manifest_hash_mismatch:{relative}"
                        )
                    immutable_candidates = load_jsonl(
                        replay_dir / "candidate_intents.jsonl"
                    )
                    if (
                        not is_compact
                        and immutable_candidates
                        != result["validation"]["entry_intents"]
                    ):
                        raise RuntimeError(
                            f"candidate_stream_summary_mismatch:{relative}"
                        )
                    candidates = [
                        _gate_candidate(
                            item,
                            hypothesis=hypothesis,
                            policy=policy,
                            seed=seed,
                            fold=fold,
                            replay_manifest_sha256=replay_manifest_sha256,
                        )
                        for item in immutable_candidates
                    ]
                    replay_source_candidates = _replay_source_candidates(
                        result,
                        immutable_candidates,
                        is_compact=is_compact,
                    )
                    replay_candidates = [
                        {
                            **item,
                            "split": replay_source_candidates[index]["split"],
                            "fold": replay_source_candidates[index]["fold"],
                        }
                        for index, item in enumerate(candidates)
                    ]
                    trades, state = _replay(replay_candidates)
                    diagnostics = _diagnostics(
                        result,
                        replay_candidates,
                        trades,
                        state,
                        policy=policy,
                    )
                    compact_observations = result["validation"].get(
                        "calibration_observations_compact"
                    )
                    calibration_observations = (
                        [
                            {
                                "confidence": float(item["confidence"]),
                                "won": float(item["won"]),
                            }
                            for item in compact_observations
                        ]
                        if compact_observations is not None
                        else [
                            {
                                "confidence": float(
                                    item["calibrated_confidence"]
                                ),
                                "won": float(
                                    float(item["selected_label_before_fee"])
                                    - 3.0
                                    > 0.0
                                ),
                            }
                            for item in result["validation"]["diagnostics"]
                        ]
                    )
                    validation_sessions = list(summary["validation_sessions"])
                    unit = {
                        "unit_id": gate.unit_id(
                            hypothesis,
                            policy,
                            seed,
                            fold,
                        ),
                        "hypothesis": hypothesis,
                        "policy": policy,
                        "seed": seed,
                        "fold": fold,
                        "era": _majority_era(
                            validation_sessions,
                            era_by_session,
                        ),
                        "campaign_namespace": gate.CAMPAIGN_NAMESPACE,
                        "simulator_version": (
                            PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
                        ),
                        "G9": False,
                        "provenance": _unit_provenance(
                            summary,
                            manifest,
                            summary_path=summary_path,
                        ),
                        "session_ids": validation_sessions,
                        "candidates": candidates,
                        "calibration_observations": calibration_observations,
                        "diagnostics": diagnostics,
                    }
                    units.append(gate.seal_unit(unit))
                    source_bindings.append(
                        {
                            "unit_id": unit["unit_id"],
                            "summary_path": relative,
                            "summary_sha256": sha256_path(summary_path),
                            "model_path": str(model_path),
                            "model_sha256": model_hash,
                            "replay_packet_path": str(replay_dir),
                            "replay_manifest_sha256": replay_manifest_sha256,
                        }
                    )
    packet = {
        "schema_version": "Protocol101FT1DRealUnitCampaignPacketV1",
        "campaign_namespace": gate.CAMPAIGN_NAMESPACE,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "unit_count": len(units),
        "units": units,
        "source_bindings": source_bindings,
        "forbidden_evidence": {
            "seed_45_present": False,
            "G9_run": False,
            "protected_holdout_present": False,
            "sealed_evidence_present": False,
            "recorder_evidence_present": False,
        },
        "materialization_receipt_sha256": sha256_path(
            MATERIALIZATION_RECEIPT
        ),
    }
    packet["packet_sha256"] = stable_hash(packet)
    authority = gate.build_execution_provenance_authority(packet)
    return packet, authority


def validate_real_unit_packet(
    packet: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    units = packet.get("units")
    expected_axes = gate.expected_unit_axes()
    observed_axes = tuple(
        (
            item.get("hypothesis"),
            item.get("policy"),
            item.get("seed"),
            item.get("fold"),
        )
        for item in units
    ) if isinstance(units, list) else ()
    if observed_axes != expected_axes:
        blockers.append("real_unit_axis_grid_mismatch")
    if packet.get("packet_sha256") != stable_hash(
        {key: value for key, value in packet.items() if key != "packet_sha256"}
    ):
        blockers.append("real_unit_packet_hash_mismatch")
    expected_authority = gate.build_execution_provenance_authority(packet)
    if authority != expected_authority:
        blockers.append("execution_authority_binding_mismatch")
    if stable_hash(authority) != stable_hash(expected_authority):
        blockers.append("execution_authority_hash_mismatch")
    if isinstance(units, list) and len(units) == 420:
        for unit, axis in zip(units, expected_axes):
            blockers.extend(
                gate._validate_unit(
                    _rehydrate_unit_gate_field_order(unit),
                    axis,
                )
            )
    else:
        blockers.append("real_unit_count_mismatch")
    blockers, certified_abstentions = _filter_certified_abstention_blockers(
        blockers,
        packet,
    )
    return {
        "schema_version": "Protocol101FT1DRealUnitCampaignValidationV2",
        "status": "PASS" if not blockers else "FAIL",
        "routing_decision": "real_420_unit_campaign_packet_validated",
        "unit_count": len(units) if isinstance(units, list) else 0,
        "certified_abstention_unit_count": len(certified_abstentions),
        "certified_abstention_unit_ids": sorted(certified_abstentions),
        "all_wait_certification_sha256": (
            sha256_path(ALL_WAIT_CERTIFICATION_PATH)
            if ALL_WAIT_CERTIFICATION_PATH.is_file()
            else None
        ),
        "execution_authority_sha256": stable_hash(authority),
        "blockers": sorted(set(blockers)),
    }


def _rehydrate_unit_gate_field_order(
    unit: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(unit)
    provenance = unit.get("provenance")
    diagnostics = unit.get("diagnostics")
    if isinstance(provenance, Mapping):
        result["provenance"] = {
            key: provenance[key]
            for key in gate.PROVENANCE_HASH_FIELDS
            if key in provenance
        }
    if isinstance(diagnostics, Mapping):
        result["diagnostics"] = {
            key: diagnostics[key]
            for key in gate.DIAGNOSTIC_KEYS
            if key in diagnostics
        }
    return result


def _certified_abstention_ids(
    packet: Mapping[str, Any],
    certification: Mapping[str, Any],
) -> tuple[set[str], list[str]]:
    blockers: list[str] = []
    without_hash = dict(certification)
    receipt_sha256 = without_hash.pop("receipt_sha256", None)
    if receipt_sha256 != stable_hash(without_hash):
        return set(), ["all_wait_certification_self_hash_mismatch"]
    if certification.get("status") != "PASS":
        return set(), ["all_wait_certification_not_pass"]
    units = packet.get("units")
    if not isinstance(units, list):
        return set(), ["all_wait_packet_units_missing"]
    packet_units = {
        str(unit.get("unit_id")): unit
        for unit in units
        if isinstance(unit, Mapping)
    }
    empty_unit_ids = {
        unit_id
        for unit_id, unit in packet_units.items()
        if isinstance(unit.get("candidates"), list)
        and len(unit["candidates"]) == 0
    }
    rows = certification.get("units")
    if not isinstance(rows, list):
        return set(), ["all_wait_certification_units_missing"]
    certified: set[str] = set()
    empty_file_sha256 = hashlib.sha256(b"").hexdigest()
    empty_list_sha256 = gate.stable_hash([])
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            blockers.append(f"all_wait_certification_row_invalid:{index}")
            continue
        unit_id = str(row.get("unit_id"))
        unit = packet_units.get(unit_id)
        stream_bytes = row.get("replay_stream_bytes")
        stream_hashes = row.get("replay_stream_sha256")
        list_hashes = row.get("empty_list_hashes")
        primary = row.get("packet_primary_economics")
        stream_names = {
            "candidate_intents.jsonl",
            "trades.jsonl",
            "skipped_events.jsonl",
        }
        list_hash_names = {
            "candidate_stream_hash",
            "candidate_payload_hash",
            "trade_identity_hash",
        }
        valid = (
            row.get("status") == "PASS"
            and not row.get("blockers")
            and unit is not None
            and unit_id in empty_unit_ids
            and row.get("packet_unit_sha256") == unit.get("unit_sha256")
            and isinstance(row.get("decision_count"), int)
            and int(row["decision_count"]) > 0
            and isinstance(row.get("candidate_frame_count"), int)
            and int(row["candidate_frame_count"]) > 0
            and row.get("diagnostic_count") == row.get("decision_count")
            and row.get("action_enter_count") == 0
            and row.get("entry_intent_count") == 0
            and row.get("trade_count") == 0
            and row.get("skipped_event_count") == 0
            and row.get("packet_candidate_count") == 0
            and isinstance(stream_bytes, Mapping)
            and set(stream_bytes) == stream_names
            and set(stream_bytes.values()) == {0}
            and isinstance(stream_hashes, Mapping)
            and set(stream_hashes) == stream_names
            and set(stream_hashes.values()) == {empty_file_sha256}
            and isinstance(list_hashes, Mapping)
            and set(list_hashes) == list_hash_names
            and set(list_hashes.values()) == {empty_list_sha256}
            and isinstance(primary, list)
            and primary == [0.0, 0.0, 0.0]
        )
        if not valid:
            blockers.append(
                f"all_wait_certification_row_invalid:{unit_id}"
            )
            continue
        if unit_id in certified:
            blockers.append(
                f"all_wait_certification_unit_duplicate:{unit_id}"
            )
            continue
        certified.add(unit_id)
    missing = sorted(empty_unit_ids - certified)
    extra = sorted(certified - empty_unit_ids)
    blockers.extend(
        f"uncertified_empty_candidate_unit:{unit_id}"
        for unit_id in missing
    )
    blockers.extend(
        f"certified_unit_not_empty:{unit_id}"
        for unit_id in extra
    )
    return certified, blockers


def _filter_certified_abstention_blockers(
    blockers: Iterable[str],
    packet: Mapping[str, Any],
    certification: Mapping[str, Any] | None = None,
) -> tuple[list[str], set[str]]:
    if certification is None:
        if not ALL_WAIT_CERTIFICATION_PATH.is_file():
            return (
                list(blockers) + ["all_wait_certification_missing"],
                set(),
            )
        certification = load_json(ALL_WAIT_CERTIFICATION_PATH)
    certified, certification_blockers = _certified_abstention_ids(
        packet,
        certification,
    )
    filtered = [
        blocker
        for blocker in blockers
        if not (
            blocker.startswith("unit_candidates_missing:")
            and blocker.split(":", 1)[1] in certified
        )
    ]
    filtered.extend(certification_blockers)
    return sorted(set(filtered)), certified


def _validate_campaign_packet_with_abstentions(
    packet: Mapping[str, Any],
    **kwargs: Any,
) -> list[str]:
    gate_packet = _rehydrate_campaign_packet_gate_field_order(packet)
    control_authority = kwargs.get("control_authority")
    if isinstance(control_authority, Mapping):
        kwargs["control_authority"] = (
            _rehydrate_control_authority_gate_field_order(
                control_authority
            )
        )
    blockers = gate.validate_campaign_packet(gate_packet, **kwargs)
    filtered, _certified = _filter_certified_abstention_blockers(
        blockers,
        gate_packet,
    )
    return filtered


def _ordered_mapping(
    value: Any,
    expected_keys: Iterable[str],
) -> Any:
    if not isinstance(value, Mapping):
        return value
    result = {
        key: value[key]
        for key in expected_keys
        if key in value
    }
    result.update(
        {
            key: item
            for key, item in value.items()
            if key not in result
        }
    )
    return result


def _rehydrate_campaign_packet_gate_field_order(
    packet: Mapping[str, Any],
) -> dict[str, Any]:
    result = _ordered_mapping(packet, gate.CAMPAIGN_PACKET_FIELDS)
    if not isinstance(result, dict):
        return dict(packet)
    result["forbidden_evidence"] = _ordered_mapping(
        packet.get("forbidden_evidence"),
        gate.FORBIDDEN_EVIDENCE_FLAGS,
    )
    result["controls"] = _ordered_mapping(
        packet.get("controls"),
        ("D1", "D5", "D6", "maxT"),
    )
    references_payload = packet.get("references")
    if isinstance(references_payload, Mapping):
        references_result = dict(references_payload)
        rows = references_payload.get("rows")
        if isinstance(rows, list):
            references_result["rows"] = [
                {
                    **dict(row),
                    "matched_null_z_by_seed": _ordered_mapping(
                        row.get("matched_null_z_by_seed"),
                        (str(seed) for seed in SEED_ORDER),
                    ),
                }
                if isinstance(row, Mapping)
                else row
                for row in rows
            ]
        result["references"] = references_result
    units = packet.get("units")
    if isinstance(units, list):
        result["units"] = [
            _rehydrate_unit_gate_field_order(unit)
            if isinstance(unit, Mapping)
            else unit
            for unit in units
        ]
    return result


def _rehydrate_control_authority_gate_field_order(
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(authority)
    result["bindings"] = _ordered_mapping(
        authority.get("bindings"),
        ("references", "D1", "D5", "D6", "maxT"),
    )
    return result


def _load_verified_fitted_summary_reference(
    path: Path,
) -> tuple[dict[str, Any], set[str]]:
    summary = load_json(path)
    if summary.get("schema_version") == unit_summary.COMPACT_SUMMARY_SCHEMA:
        summary = unit_summary.validate_compact_summary(path)
        return (
            summary,
            {
                sha256_path(path),
                str(summary["full_summary_archive"]["uncompressed_sha256"]),
            },
        )
    without_hash = dict(summary)
    summary_hash = without_hash.pop("summary_hash", None)
    if summary_hash != runner.stable_hash(without_hash):
        raise RuntimeError(f"unit_summary_self_hash_mismatch:{path}")
    return summary, {sha256_path(path)}


def _replay_source_candidates(
    result: Mapping[str, Any],
    immutable_candidates: list[dict[str, Any]],
    *,
    is_compact: bool,
) -> list[dict[str, Any]]:
    if is_compact:
        return immutable_candidates
    return list(result["validation"]["entry_intents"])


def run_units_stage() -> dict[str, Any]:
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "EXECUTION_PROVENANCE_AUTHORITY":
        raise RuntimeError(
            f"journal_not_at_execution_authority:{journal['next_node']}"
        )
    packet, authority = build_real_unit_packet()
    validation = validate_real_unit_packet(packet, authority)
    write_json_immutable(UNIT_PACKET_PATH, packet)
    write_json_immutable(EXECUTION_AUTHORITY_PATH, authority)
    write_json_immutable(EXECUTION_VALIDATION_PATH, validation)
    if validation["status"] != "PASS":
        raise RuntimeError("real_unit_campaign_validation_failed")
    append_checkpoint(
        JOURNAL_PATH,
        workspace_root=ROOT,
        node="EXECUTION_PROVENANCE_AUTHORITY",
        artifact_path=EXECUTION_AUTHORITY_PATH,
        validator_route=validation["routing_decision"],
        validator_receipt_path=EXECUTION_VALIDATION_PATH,
    )
    return validation


def _materialized_scope() -> Any:
    scope = load_training_scope()
    receipt = load_json(
        EXECUTION_ROOT / "two_clock_materialization_receipt.json"
    )
    without_hash = dict(receipt)
    receipt_hash = without_hash.pop("receipt_sha256", None)
    if receipt_hash != stable_hash(without_hash):
        raise RuntimeError("materialization_receipt_self_hash_mismatch")
    rows = list(receipt.get("sessions") or [])
    expected = [session for session, _path in scope.sessions]
    if [item.get("session") for item in rows] != expected:
        raise RuntimeError("materialization_session_grid_mismatch")
    paths = {
        str(item["session"]): ROOT / str(item["output_path"])
        for item in rows
    }
    return replace(
        scope,
        sessions=[(session, paths[session]) for session in expected],
    )


def _packet_units(
    packet: Mapping[str, Any],
    *,
    hypothesis: str,
    policy: str,
    seed: int,
) -> list[dict[str, Any]]:
    selected = [
        item
        for item in packet["units"]
        if item["hypothesis"] == hypothesis
        and item["policy"] == policy
        and int(item["seed"]) == seed
    ]
    if [int(item["fold"]) for item in selected] != list(FOLD_ORDER):
        raise RuntimeError(
            f"packet_fold_grid_mismatch:{hypothesis}:{policy}:{seed}"
        )
    return selected


def _continuous_candidates(
    units: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        dict(candidate)
        for unit in units
        for candidate in unit["candidates"]
    ]


def _continuous_metrics(
    units: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = _continuous_candidates(units)
    trades, state = _replay(candidates)
    return references.replay_metrics(trades, state)


def _policy_decision_maps(
    *,
    scope: Any,
    policy_index: int,
) -> dict[int, dict[tuple[str, int], Any]]:
    path_map = {session: path for session, path in scope.sessions}
    margins = runner.guard_margins()
    result: dict[int, dict[tuple[str, int], Any]] = {}
    for fold, fold_payload in zip(FOLD_ORDER, scope.folds):
        sessions = list(fold_payload["validation_sessions"])
        decisions = runner.load_repaired_decisions(
            [(session, path_map[session]) for session in sessions],
            hypothesis="H0",
            policy_index=policy_index,
            guard_margins=margins,
            split=f"FT1D-REFERENCE-F{fold}",
        )
        mapping = {
            (
                item.base.session,
                int(item.base.decision_time.value),
            ): item
            for item in decisions
        }
        if len(mapping) != len(decisions):
            raise RuntimeError(f"reference_decision_identity_duplicate:P{policy_index}:F{fold}")
        result[fold] = mapping
    return result


def _matched_opportunities(
    units: list[Mapping[str, Any]],
    *,
    decision_maps: Mapping[int, Mapping[tuple[str, int], Any]],
    campaign_id: str,
) -> list[references.ReferenceOpportunity]:
    opportunities: list[references.ReferenceOpportunity] = []
    for unit in units:
        fold = int(unit["fold"])
        mapping = decision_maps[fold]
        ordered = sorted(
            unit["candidates"],
            key=lambda item: (
                str(item["session"]),
                int(item["decision_time_ns"]),
                str(item["contract_id"]),
            ),
        )
        seen_decisions: set[tuple[str, int]] = set()
        for ordinal, candidate in enumerate(ordered):
            key = (
                str(candidate["session"]),
                int(candidate["decision_time_ns"]),
            )
            if key in seen_decisions:
                raise RuntimeError(
                    f"matched_reference_duplicate_entry_decision:{campaign_id}:{key}"
                )
            seen_decisions.add(key)
            repaired = mapping.get(key)
            if repaired is None:
                raise RuntimeError(
                    f"matched_reference_decision_missing:{campaign_id}:{key}"
                )
            gap = float(repaired.base.features[0, 0])
            opportunities.append(
                references.ReferenceOpportunity(
                    campaign_id=campaign_id,
                    fold=f"F{fold}",
                    split="validation",
                    policy_index=int(candidate["policy_index"]),
                    repaired=repaired,
                    vwap_side="C" if gap >= 0.0 else "P",
                    decision_ordinal=ordinal,
                )
            )
    return opportunities


def _stream_random_reference(
    opportunities: list[references.ReferenceOpportunity],
    *,
    policy_index: int,
) -> dict[str, Any]:
    schedule = references.generate_matched_random_schedule(
        opportunities,
        policy_index=policy_index,
    )
    draw_rows: list[dict[str, Any]] = []
    pnl_values: list[float] = []
    for draw_index in range(schedule.draws):
        candidates = references.matched_random_draw_candidates(
            opportunities,
            schedule,
            draw_index=draw_index,
        )
        trades, state = references.replay_reference_v5(candidates)
        metrics = references.replay_metrics(trades, state)
        pnl_values.append(float(metrics["net_pnl"]))
        draw_rows.append(
            {
                "draw_index": draw_index,
                "net_pnl": float(metrics["net_pnl"]),
                "trades": int(metrics["trades"]),
                "candidate_stream_hash": metrics["candidate_stream_hash"],
                "candidate_payload_hash": metrics["candidate_payload_hash"],
                "trade_identity_hash": metrics["trade_identity_hash"],
            }
        )
    mean = statistics.fmean(pnl_values)
    std = statistics.stdev(pnl_values) if len(pnl_values) > 1 else 0.0
    return {
        "schema_version": references.REFERENCE_SCHEMA,
        "kind": "matched_random_selection_v5_real_streaming",
        "policy_index": policy_index,
        "draws": schedule.draws,
        "seed": schedule.seed,
        "prng": schedule.prng,
        "opportunity_count": len(opportunities),
        "schedule_hash": schedule.schedule_hash,
        "draws_summary": draw_rows,
        "pooled_net_pnl_distribution": {
            "count": len(pnl_values),
            "mean": float(mean),
            "sample_std": float(std),
            "minimum": float(min(pnl_values)),
            "maximum": float(max(pnl_values)),
        },
        "identity_complete": True,
    }


def _all_policy5_opportunities(
    decision_maps: Mapping[int, Mapping[tuple[str, int], Any]],
) -> list[references.ReferenceOpportunity]:
    result: list[references.ReferenceOpportunity] = []
    for fold in FOLD_ORDER:
        mapping = decision_maps[fold]
        for ordinal, ((_session, _clock), repaired) in enumerate(
            sorted(mapping.items())
        ):
            gap = float(repaired.base.features[0, 0])
            result.append(
                references.ReferenceOpportunity(
                    campaign_id="FT1D-FIXED-HEURISTIC-P5",
                    fold=f"F{fold}",
                    split="validation",
                    policy_index=5,
                    repaired=repaired,
                    vwap_side="C" if gap >= 0.0 else "P",
                    decision_ordinal=ordinal,
                )
            )
    return result


def build_real_references(
    packet: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scope = _materialized_scope()
    detail_rows: list[dict[str, Any]] = []
    z_by_row: dict[str, dict[str, float]] = {
        row: {} for row in gate.ROWS
    }
    heuristic_metrics: dict[str, Any] | None = None
    for policy_index, policy in enumerate(POLICY_ORDER):
        decision_maps = _policy_decision_maps(
            scope=scope,
            policy_index=policy_index,
        )
        if policy_index == references.FIXED_HEURISTIC_POLICY:
            heuristic = references.run_fixed_heuristic_reference(
                _all_policy5_opportunities(decision_maps)
            )
            heuristic_metrics = dict(heuristic["continuous_pooled_metrics"])
        for hypothesis in HYPOTHESIS_ORDER:
            row_id = f"{hypothesis}/{policy}"
            for seed in SEED_ORDER:
                units = _packet_units(
                    packet,
                    hypothesis=hypothesis,
                    policy=policy,
                    seed=seed,
                )
                observed = _continuous_metrics(units)
                opportunities = _matched_opportunities(
                    units,
                    decision_maps=decision_maps,
                    campaign_id=f"FT1D-{row_id}-S{seed}",
                )
                random_result = _stream_random_reference(
                    opportunities,
                    policy_index=policy_index,
                )
                distribution = random_result[
                    "pooled_net_pnl_distribution"
                ]
                std = float(distribution["sample_std"])
                if not math.isfinite(std) or std <= 0.0:
                    raise RuntimeError(
                        f"matched_random_zero_variance:{row_id}:S{seed}"
                    )
                z = (
                    float(observed["net_pnl"])
                    - float(distribution["mean"])
                ) / std
                if not math.isfinite(z):
                    raise RuntimeError(
                        f"matched_random_nonfinite_z:{row_id}:S{seed}"
                    )
                z_by_row[row_id][str(seed)] = float(z)
                detail_rows.append(
                    {
                        "row_id": row_id,
                        "seed": seed,
                        "observed_continuous_oof_metrics": observed,
                        "matched_random": random_result,
                        "matched_null_z": float(z),
                    }
                )
    if heuristic_metrics is None:
        raise RuntimeError("fixed_heuristic_reference_missing")
    detail = {
        "schema_version": "Protocol101FT1DRealV5ReferencesDetailV1",
        "status": "real_v5_references_complete",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "row_seed_count": len(detail_rows),
        "matched_random_draws_per_row_seed": references.MATCHED_RANDOM_DRAWS,
        "fixed_heuristic_policy": references.FIXED_HEURISTIC_POLICY,
        "fixed_heuristic_continuous_metrics": heuristic_metrics,
        "rows": detail_rows,
        "side_effects": {
            "protected_holdout_read": False,
            "seed_45_or_G9_executed": False,
            "broker_endpoint_called": False,
            "paid_data_downloaded": False,
        },
    }
    detail["detail_sha256"] = stable_hash(detail)
    write_json_immutable(REFERENCES_DETAIL_PATH, detail)
    gate_references = {
        "schema_version": "Protocol101FT1CFreshIndependentReferencesV1",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(REFERENCES_DETAIL_PATH),
        "rows": [
            {
                "row_id": row_id,
                "heuristic_pooled_pnl": float(
                    heuristic_metrics["net_pnl"]
                ),
                "matched_null_z_by_seed": z_by_row[row_id],
            }
            for row_id in gate.ROWS
        ],
    }
    return detail, gate_references


def _reference_row_path(row_id: str) -> Path:
    return REFERENCE_ROWS_ROOT / f"{row_id.replace('/', '_')}.json"


def _verify_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
    identity: str,
) -> None:
    without_hash = dict(payload)
    observed = without_hash.pop(field, None)
    if observed != stable_hash(without_hash):
        raise RuntimeError(f"self_hash_mismatch:{identity}:{field}")


def run_reference_row(row_id: str) -> dict[str, Any]:
    if row_id not in gate.ROWS:
        raise RuntimeError(f"reference_row_forbidden:{row_id}")
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "REAL_V5_REFERENCES_D1_D5_D6":
        raise RuntimeError(
            f"journal_not_at_real_references:{journal['next_node']}"
        )
    output_path = _reference_row_path(row_id)
    if output_path.is_file():
        existing = load_json(output_path)
        _verify_self_hash(
            existing,
            field="row_sha256",
            identity=f"reference:{row_id}",
        )
        if (
            existing.get("row_id") != row_id
            or existing.get("seed_count") != 3
            or existing.get("acceptance_route")
            != gate.REFERENCE_ACCEPTANCE_ROUTE
        ):
            raise RuntimeError(f"reference_row_resume_mismatch:{row_id}")
        return existing

    packet = load_json(UNIT_PACKET_PATH)
    hypothesis, policy = row_id.split("/")
    policy_index = int(policy[1:])
    units_by_seed = {
        seed: _packet_units(
            packet,
            hypothesis=hypothesis,
            policy=policy,
            seed=seed,
        )
        for seed in SEED_ORDER
    }
    del packet
    scope = _materialized_scope()
    decision_maps = _policy_decision_maps(
        scope=scope,
        policy_index=policy_index,
    )
    seed_rows: list[dict[str, Any]] = []
    for seed in SEED_ORDER:
        units = units_by_seed[seed]
        observed = _continuous_metrics(units)
        opportunities = _matched_opportunities(
            units,
            decision_maps=decision_maps,
            campaign_id=f"FT1D-{row_id}-S{seed}",
        )
        random_result = _stream_random_reference(
            opportunities,
            policy_index=policy_index,
        )
        distribution = random_result["pooled_net_pnl_distribution"]
        std = float(distribution["sample_std"])
        if not math.isfinite(std) or std <= 0.0:
            raise RuntimeError(
                f"matched_random_zero_variance:{row_id}:S{seed}"
            )
        z = (
            float(observed["net_pnl"]) - float(distribution["mean"])
        ) / std
        if not math.isfinite(z):
            raise RuntimeError(
                f"matched_random_nonfinite_z:{row_id}:S{seed}"
            )
        seed_rows.append(
            {
                "seed": seed,
                "observed_continuous_oof_metrics": observed,
                "matched_random": random_result,
                "matched_null_z": float(z),
            }
        )
    result = {
        "schema_version": "Protocol101FT1DRealV5ReferenceRowV1",
        "row_id": row_id,
        "policy_index": policy_index,
        "seed_count": len(seed_rows),
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "matched_random_draws_per_seed": references.MATCHED_RANDOM_DRAWS,
        "seeds": seed_rows,
        "row_sha256": None,
    }
    result["row_sha256"] = stable_hash(
        {key: value for key, value in result.items() if key != "row_sha256"}
    )
    write_json_immutable(output_path, result)
    return result


def run_fixed_reference() -> dict[str, Any]:
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "REAL_V5_REFERENCES_D1_D5_D6":
        raise RuntimeError(
            f"journal_not_at_real_references:{journal['next_node']}"
        )
    if FIXED_REFERENCE_PATH.is_file():
        existing = load_json(FIXED_REFERENCE_PATH)
        _verify_self_hash(
            existing,
            field="reference_sha256",
            identity="fixed_reference",
        )
        return existing
    scope = _materialized_scope()
    decision_maps = _policy_decision_maps(
        scope=scope,
        policy_index=references.FIXED_HEURISTIC_POLICY,
    )
    opportunities = _all_policy5_opportunities(decision_maps)
    reference = references.run_fixed_heuristic_reference(opportunities)
    result = {
        "schema_version": "Protocol101FT1DRealV5FixedHeuristicV1",
        "kind": "fixed_vwap_side_nearest_atm_best_single_shape_P5",
        "policy_index": references.FIXED_HEURISTIC_POLICY,
        "opportunity_count": len(opportunities),
        "candidate_count": len(reference["candidates"]),
        "trade_count": len(reference["trades"]),
        "per_fold_metrics": reference["per_fold_metrics"],
        "continuous_pooled_metrics": reference[
            "continuous_pooled_metrics"
        ],
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "reference_sha256": None,
    }
    result["reference_sha256"] = stable_hash(
        {
            key: value
            for key, value in result.items()
            if key != "reference_sha256"
        }
    )
    write_json_immutable(FIXED_REFERENCE_PATH, result)
    return result


def combine_real_references() -> tuple[dict[str, Any], dict[str, Any]]:
    fixed = run_fixed_reference()
    bindings: list[dict[str, Any]] = []
    z_by_row: dict[str, dict[str, float]] = {}
    for row_id in gate.ROWS:
        path = _reference_row_path(row_id)
        row = load_json(path)
        _verify_self_hash(
            row,
            field="row_sha256",
            identity=f"reference:{row_id}",
        )
        if (
            row.get("row_id") != row_id
            or [int(item["seed"]) for item in row.get("seeds", [])]
            != list(SEED_ORDER)
        ):
            raise RuntimeError(f"reference_row_grid_mismatch:{row_id}")
        z_by_row[row_id] = {
            str(item["seed"]): float(item["matched_null_z"])
            for item in row["seeds"]
        }
        bindings.append(
            {
                "row_id": row_id,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_path(path),
                "row_sha256": row["row_sha256"],
            }
        )
    detail = {
        "schema_version": "Protocol101FT1DRealV5ReferencesDetailV1",
        "status": "real_v5_references_complete",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "row_count": len(bindings),
        "row_seed_count": len(bindings) * len(SEED_ORDER),
        "matched_random_draws_per_row_seed": references.MATCHED_RANDOM_DRAWS,
        "fixed_heuristic": {
            "path": str(FIXED_REFERENCE_PATH.relative_to(ROOT)),
            "sha256": sha256_path(FIXED_REFERENCE_PATH),
            "continuous_pooled_metrics": fixed[
                "continuous_pooled_metrics"
            ],
        },
        "row_bindings": bindings,
        "side_effects": {
            "protected_holdout_read": False,
            "seed_45_or_G9_executed": False,
            "broker_endpoint_called": False,
            "paid_data_downloaded": False,
        },
        "detail_sha256": None,
    }
    detail["detail_sha256"] = stable_hash(
        {key: value for key, value in detail.items() if key != "detail_sha256"}
    )
    write_json_immutable(REFERENCES_DETAIL_PATH, detail)
    gate_references = {
        "schema_version": "Protocol101FT1CFreshIndependentReferencesV1",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(REFERENCES_DETAIL_PATH),
        "rows": [
            {
                "row_id": row_id,
                "heuristic_pooled_pnl": float(
                    fixed["continuous_pooled_metrics"]["net_pnl"]
                ),
                "matched_null_z_by_seed": z_by_row[row_id],
            }
            for row_id in gate.ROWS
        ],
    }
    return detail, gate_references


def _d1_fold_root(seed: int, fold: int) -> Path:
    return D1_ROOT / f"seed{seed}" / f"fold{fold}"


def _d1_seed_path(seed: int) -> Path:
    return D1_ROOT / f"seed{seed}" / "seed_summary.json"


def _d1_schedule_decision(
    *,
    session: str,
    row: Mapping[str, Any],
    row_index: int,
) -> RepairedCanonicalDecision:
    decision_time = pd.Timestamp(row["decision_time"])
    if decision_time.tzinfo is None:
        decision_time = decision_time.tz_localize("UTC")
    feature_count = len(HYPOTHESES["H2"])
    base = CanonicalDecision(
        session=session,
        decision_time=decision_time.tz_convert("UTC"),
        features=np.zeros((1, feature_count), dtype=np.float64),
        labels=np.asarray([float(row_index)], dtype=np.float64),
        mid_labels=np.asarray([float(row_index)], dtype=np.float64),
        entry_asks=np.ones(1, dtype=np.float64),
        offsets=np.zeros(1, dtype=np.float64),
        rights=np.asarray(["C"], dtype=object),
        contract_ids=np.asarray(
            [f"D1-SCHEDULE:{session}:{row_index}"],
            dtype=object,
        ),
        strike_indices=np.zeros(1, dtype=np.int64),
        right_indices=np.zeros(1, dtype=np.int64),
    )
    zeros_i64 = np.zeros(1, dtype=np.int64)
    zeros_float = np.zeros(1, dtype=np.float64)
    return RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=zeros_i64.copy(),
        source_exit_quote_time_ns=zeros_i64.copy(),
        exit_quote_age_ms=zeros_float.copy(),
        exit_reason_codes=np.zeros(1, dtype=np.uint8),
        executable_exit_bids=zeros_float.copy(),
        policy_deadline_ns=zeros_i64.copy(),
        invalid_reason_codes=np.zeros(1, dtype=np.uint8),
        canonical_strike_slots=zeros_i64.copy(),
        source_quote_time_ns=zeros_i64.copy(),
        source_context_time_ns=zeros_i64.copy(),
    )


def _d1_apply_moved_targets(
    destination: RepairedCanonicalDecision,
    net_labels: np.ndarray,
    mid_labels: np.ndarray,
    *,
    validation_unpermuted: bool,
) -> RepairedCanonicalDecision | None:
    net_ladder = np.asarray(net_labels, dtype=np.float64)
    mid_ladder = np.asarray(mid_labels, dtype=np.float64)
    if net_ladder.shape != (21, 2) or mid_ladder.shape != (21, 2):
        raise RuntimeError("D1_raw_P5_target_ladder_shape_mismatch")
    indexes = (
        destination.base.strike_indices,
        destination.base.right_indices,
    )
    selected_net = net_ladder[indexes]
    selected_mid = mid_ladder[indexes]
    valid = np.isfinite(selected_net) & np.isfinite(selected_mid)
    if validation_unpermuted and (
        not bool(valid.all())
        or not np.array_equal(
            selected_net,
            destination.base.labels,
        )
        or not np.array_equal(
            selected_mid,
            destination.base.mid_labels,
        )
    ):
        raise RuntimeError("D1_validation_target_identity_mismatch")
    if not bool(valid.any()):
        return None

    def selected(values: np.ndarray) -> np.ndarray:
        return np.asarray(values)[valid]

    base = replace(
        destination.base,
        features=selected(destination.base.features),
        labels=selected_net[valid],
        mid_labels=selected_mid[valid],
        entry_asks=selected(destination.base.entry_asks),
        offsets=selected(destination.base.offsets),
        rights=selected(destination.base.rights),
        contract_ids=selected(destination.base.contract_ids),
        strike_indices=selected(destination.base.strike_indices),
        right_indices=selected(destination.base.right_indices),
    )
    return replace(
        destination,
        base=base,
        realized_exit_time_ns=selected(
            destination.realized_exit_time_ns
        ),
        source_exit_quote_time_ns=selected(
            destination.source_exit_quote_time_ns
        ),
        exit_quote_age_ms=selected(destination.exit_quote_age_ms),
        exit_reason_codes=selected(destination.exit_reason_codes),
        executable_exit_bids=selected(destination.executable_exit_bids),
        policy_deadline_ns=selected(destination.policy_deadline_ns),
        invalid_reason_codes=selected(destination.invalid_reason_codes),
        canonical_strike_slots=selected(
            destination.canonical_strike_slots
        ),
        source_quote_time_ns=selected(destination.source_quote_time_ns),
        source_context_time_ns=selected(
            destination.source_context_time_ns
        ),
    )


def _d1_hashable_targets(values: np.ndarray) -> list[float | None]:
    return [
        float(value) if math.isfinite(float(value)) else None
        for value in np.asarray(values, dtype=np.float64).reshape(-1)
    ]


def _d1_non_candidate_intents(
    decisions: list[RepairedCanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    threshold: float,
    epsilon: float,
    config: HGBUnitConfig,
    fold: str,
) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    for repaired, scores in zip(decisions, scores_by_decision):
        decision = repaired.base
        (
            selected,
            top_score,
            _second_score,
            _margin,
            _confident,
        ) = hgb_core.selected_index_for_scores(
            decision,
            scores,
            epsilon=epsilon,
            k_slot=config.k_slot,
        )
        if not (
            math.isfinite(top_score)
            and top_score
            > float(threshold) + float(config.k_action) * float(epsilon)
        ):
            continue
        intents.append(
            {
                "split": "D1_NON_CANDIDATE_CALIBRATION",
                "fold": fold,
                "session": decision.session,
                "decision_time_ns": int(decision.decision_time.value),
                "contract_id": str(decision.contract_ids[selected]),
                "right": str(decision.rights[selected]),
                "canonical_strike_slot": int(
                    repaired.canonical_strike_slots[selected]
                ),
                "policy_index": int(config.policy_index),
                "entry_ask": float(decision.entry_asks[selected]),
                "score": float(top_score),
                "target_pnl_after_campaign_fee": float(
                    decision.labels[selected] - float(config.fee)
                ),
                "label_realized_exit_time_ns": int(
                    repaired.realized_exit_time_ns[selected]
                ),
                "label_source_exit_quote_time_ns": int(
                    repaired.source_exit_quote_time_ns[selected]
                ),
                "label_exit_quote_age_ms": float(
                    repaired.exit_quote_age_ms[selected]
                ),
                "label_exit_reason_code": int(
                    repaired.exit_reason_codes[selected]
                ),
                "label_policy_deadline_ns": int(
                    repaired.policy_deadline_ns[selected]
                ),
                "label_invalid_reason_code": int(
                    repaired.invalid_reason_codes[selected]
                ),
                "non_candidate": True,
                "replay_permitted": False,
                "executable_quote_pnl_claim": False,
            }
        )
    return intents


def _d1_non_candidate_serial_target_evaluation(
    intents: list[dict[str, Any]],
    *,
    config: HGBUnitConfig,
) -> dict[str, Any]:
    ordered = sorted(
        intents,
        key=lambda item: (
            str(item["session"]),
            int(item["decision_time_ns"]),
            str(item["contract_id"]),
            int(item["policy_index"]),
        ),
    )
    identities = [
        (
            item["split"],
            item["fold"],
            item["session"],
            item["decision_time_ns"],
            item["contract_id"],
            item["policy_index"],
        )
        for item in ordered
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError("D1_non_candidate_intent_identity_duplicate")
    for item in ordered:
        decision = int(item["decision_time_ns"])
        source = int(item["label_source_exit_quote_time_ns"])
        realized = int(item["label_realized_exit_time_ns"])
        deadline = int(item["label_policy_deadline_ns"])
        expected_age = (realized - source) / 1_000_000.0
        if (
            not decision < source <= realized <= deadline
            or int(item["label_invalid_reason_code"]) != 0
            or not math.isclose(
                float(item["label_exit_quote_age_ms"]),
                expected_age,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isfinite(
                float(item["target_pnl_after_campaign_fee"])
            )
            or not math.isfinite(float(item["entry_ask"]))
            or float(item["entry_ask"]) <= 0.0
            or item.get("non_candidate") is not True
            or item.get("replay_permitted") is not False
            or item.get("executable_quote_pnl_claim") is not False
        ):
            raise RuntimeError(
                "D1_non_candidate_intent_contract_invalid:"
                f"{item['session']}:{item['decision_time_ns']}"
            )
        clock_sessions = {
            datetime.fromtimestamp(
                value / 1_000_000_000,
                tz=NY,
            ).date().isoformat()
            for value in (decision, source, realized, deadline)
        }
        if clock_sessions != {str(item["session"])}:
            raise RuntimeError(
                "D1_non_candidate_intent_cross_session"
            )

    starting_cash = 10_000.0
    cash = starting_cash
    pending: dict[str, Any] | None = None
    active_session: str | None = None
    session_start: dict[str, float] = {}
    realized_session: dict[str, float] = defaultdict(float)
    equity = [starting_cash]
    entered = 0
    skipped = {
        "after_entry_cutoff": 0,
        "overlap": 0,
        "daily_loss_stop": 0,
        "unaffordable": 0,
    }

    def realize_pending() -> None:
        nonlocal cash, pending
        if pending is None:
            return
        pnl = float(pending["target_pnl_after_campaign_fee"])
        cash += pnl
        realized_session[str(pending["session"])] += pnl
        equity.append(float(cash))
        pending = None

    for item in ordered:
        decision_time_ns = int(item["decision_time_ns"])
        if (
            pending is not None
            and int(pending["label_realized_exit_time_ns"])
            <= decision_time_ns
        ):
            realize_pending()
        session = str(item["session"])
        if active_session is not None and active_session != session:
            if pending is not None:
                raise RuntimeError(
                    "D1_non_candidate_pending_crossed_session"
                )
        active_session = session
        session_start.setdefault(session, float(cash))
        local = datetime.fromtimestamp(
            decision_time_ns / 1_000_000_000,
            tz=NY,
        )
        if (local.hour, local.minute) > (15, 30):
            skipped["after_entry_cutoff"] += 1
            continue
        if pending is not None:
            skipped["overlap"] += 1
            continue
        daily_limit = (
            float(config.daily_loss_fraction)
            * float(session_start[session])
        )
        if (
            daily_limit > 0.0
            and realized_session[session] <= -daily_limit
        ):
            skipped["daily_loss_stop"] += 1
            continue
        required_cash = (
            float(item["entry_ask"]) * 100.0 + float(config.fee)
        )
        if required_cash > cash + 1e-9:
            skipped["unaffordable"] += 1
            continue
        pending = item
        entered += 1
    realize_pending()
    equity_array = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity_array)
    drawdown = peak - equity_array
    return {
        "schema_version": (
            "Protocol101FT1DD1NonCandidateSerialTargetEvaluationV1"
        ),
        "non_candidate": True,
        "replay_permitted": False,
        "simulator_v5_called": False,
        "executable_quote_pnl_claim": False,
        "serial_rules": {
            "one_account": True,
            "one_contract": True,
            "starting_cash": starting_cash,
            "contract_multiplier_for_affordability_only": 100.0,
            "campaign_fee_for_target_and_affordability": float(
                config.fee
            ),
            "daily_stop_fraction": float(
                config.daily_loss_fraction
            ),
            "no_new_entries_after_et": "15:30",
            "occupancy_clock": "destination_realized_exit_time_ns",
        },
        "selected_target_count": len(ordered),
        "serial_entered_target_count": entered,
        "target_net_pnl_after_campaign_fee": float(
            cash - starting_cash
        ),
        "minimum_target_equity": float(equity_array.min()),
        "maximum_target_drawdown": float(drawdown.max()),
        "skipped": skipped,
        "intent_schedule_sha256": stable_hash(ordered),
    }


def _choose_D1_non_candidate_threshold(
    decisions: list[RepairedCanonicalDecision],
    scores_by_decision: list[np.ndarray],
    *,
    epsilon: float,
    config: HGBUnitConfig,
    fold: str = "",
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in hgb_core.threshold_candidates(scores_by_decision):
        intents = _d1_non_candidate_intents(
            decisions,
            scores_by_decision,
            threshold=threshold,
            epsilon=epsilon,
            config=config,
            fold=fold,
        )
        evaluation = _d1_non_candidate_serial_target_evaluation(
            intents,
            config=config,
        )
        rows.append(
            {
                "threshold": float(threshold),
                **evaluation,
            }
        )
    best = max(
        rows,
        key=lambda row: (
            float(row["target_net_pnl_after_campaign_fee"]),
            -int(row["serial_entered_target_count"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"]), rows


def _d1_role_decisions(
    *,
    path_map: Mapping[str, Path],
    sessions: list[str],
    role: str,
    seed: int | None,
    fold: int,
    guard_margins: Mapping[str, float],
) -> tuple[list[Any], list[dict[str, Any]]]:
    decisions: list[Any] = []
    receipts: list[dict[str, Any]] = []
    for session in sessions:
        source_path = path_map[session]
        with source_path.open("rb") as handle:
            rows = pickle.load(handle)
        if not isinstance(rows, list):
            raise TypeError(f"{source_path} expected list rows")
        assert_processed_row_identities(
            rows,
            split=f"FT1D-D1-F{fold}:{role}:{session}",
            session=session,
            boundary="D1 full-row clock before target override",
        )
        if len(rows) != references.D1_NORMAL_ROWS:
            raise RuntimeError(
                f"D1_source_row_count_mismatch:{session}:{len(rows)}"
            )
        loaded = runner.load_repaired_decisions(
            [(session, source_path)],
            hypothesis="H2",
            policy_index=references.FIXED_HEURISTIC_POLICY,
            guard_margins=dict(guard_margins),
            split=f"FT1D-D1-F{fold}:{role}:{session}",
        )
        schedule_decisions = [
            _d1_schedule_decision(
                session=session,
                row=row,
                row_index=row_index,
            )
            for row_index, row in enumerate(rows)
        ]
        included, schedule_overrides, receipt = (
            references.build_d1_target_overrides(
                schedule_decisions,
                role=role,
                seed=seed if role in {"fit", "calibration"} else None,
                policy_index=references.FIXED_HEURISTIC_POLICY,
            )
        )
        source_rows = [
            int(item.source_row_index)
            for item in schedule_overrides
        ]
        actual_overrides = [
            references.D1TargetOverride(
                session=session,
                destination_decision_time_ns=int(
                    destination.base.decision_time.value
                ),
                destination_row_index=destination_index,
                source_row_index=source_index,
                net_labels=np.asarray(
                    rows[source_index]["labels_net_pnl"],
                    dtype=np.float64,
                )[:, :, references.FIXED_HEURISTIC_POLICY],
                mid_labels=np.asarray(
                    rows[source_index]["labels_mid_pnl"],
                    dtype=np.float64,
                )[:, :, references.FIXED_HEURISTIC_POLICY],
                role=role,
                seed=seed if role in {"fit", "calibration"} else None,
            )
            for destination_index, (destination, source_index) in enumerate(
                zip(included, source_rows)
            )
        ]
        net_arrays, mid_arrays = references.d1_target_arrays(
            included,
            actual_overrides,
        )
        loaded_by_time = {
            int(item.base.decision_time.value): item
            for item in loaded
        }
        if len(loaded_by_time) != len(loaded):
            raise RuntimeError(
                f"D1_loaded_decision_identity_duplicate:{session}"
            )
        moved: list[RepairedCanonicalDecision] = []
        zero_candidate_rows: list[int] = []
        source_nonfinite_target_count = 0
        target_bindings: list[dict[str, Any]] = []
        for destination_index, (
            destination,
            override,
            net_labels,
            mid_labels,
        ) in enumerate(
            zip(included, actual_overrides, net_arrays, mid_arrays)
        ):
            destination_time = int(destination.base.decision_time.value)
            source_nonfinite_target_count += int(
                (~np.isfinite(net_labels)).sum()
                + (~np.isfinite(mid_labels)).sum()
            )
            target_bindings.append(
                {
                    "destination_decision_time_ns": destination_time,
                    "destination_row_index": destination_index,
                    "source_row_index": int(override.source_row_index),
                    "net_labels": _d1_hashable_targets(net_labels),
                    "mid_labels": _d1_hashable_targets(mid_labels),
                }
            )
            runner_destination = loaded_by_time.get(destination_time)
            if runner_destination is None:
                zero_candidate_rows.append(destination_index)
                continue
            selected = _d1_apply_moved_targets(
                runner_destination,
                net_labels,
                mid_labels,
                validation_unpermuted=role == "validation",
            )
            if selected is None:
                zero_candidate_rows.append(destination_index)
                continue
            moved.append(selected)
        included_times = {
            int(item.base.decision_time.value)
            for item in included
        }
        unexpected_loaded = sorted(
            time_value
            for time_value in loaded_by_time
            if time_value in included_times
            and time_value
            not in {
                int(item.base.decision_time.value)
                for item in moved
            }
            and time_value
            not in {
                int(included[index].base.decision_time.value)
                for index in zero_candidate_rows
            }
        )
        if unexpected_loaded:
            raise RuntimeError(
                f"D1_runner_destination_unaccounted:{session}:"
                f"{len(unexpected_loaded)}"
            )
        decisions.extend(moved)
        receipt = {
            **receipt,
            "adapter_schema_version": (
                "Protocol101FT1DRealD1FullRowTargetAdapterV1"
            ),
            "schedule_fixture_only": True,
            "normal_source_row_count": len(rows),
            "loaded_candidate_decision_count": len(loaded),
            "included_clock_row_count": len(included),
            "runner_candidate_decision_count": len(moved),
            "zero_or_invalid_moved_target_row_count": len(
                zero_candidate_rows
            ),
            "zero_or_invalid_moved_target_row_indices": (
                zero_candidate_rows
            ),
            "source_nonfinite_target_count": (
                source_nonfinite_target_count
            ),
            "actual_target_override_sha256": stable_hash(
                target_bindings
            ),
            "actual_target_override_type": (
                "Protocol101D1TargetOverride"
            ),
            "canonical_slot_mapping": (
                "destination_strike_index_and_right_index"
            ),
            "destination_features_identities_asks_and_exit_metadata_moved": (
                False
            ),
        }
        receipts.append(
            {
                "session": session,
                "role": role,
                "source_path": str(source_path.relative_to(ROOT)),
                "source_sha256": sha256_path(source_path),
                **receipt,
            }
        )
    return decisions, receipts


def _load_verified_d1_fold(seed: int, fold: int) -> dict[str, Any] | None:
    summary_path = _d1_fold_root(seed, fold) / "summary.json"
    if not summary_path.is_file():
        return None
    summary = load_json(summary_path)
    _verify_self_hash(
        summary,
        field="summary_sha256",
        identity=f"D1:S{seed}:F{fold}",
    )
    if (
        summary.get("hypothesis") != "H2"
        or summary.get("policy") != "P5"
        or summary.get("seed") != seed
        or summary.get("fold") != fold
        or summary.get("materialization_receipt_sha256")
        != sha256_path(MATERIALIZATION_RECEIPT)
    ):
        raise RuntimeError(f"D1_resume_axis_or_source_mismatch:S{seed}:F{fold}")
    model_path = ROOT / str(summary["model_artifact"]["path"])
    if sha256_path(model_path) != summary["model_artifact"]["sha256"]:
        raise RuntimeError(f"D1_resume_model_hash_mismatch:S{seed}:F{fold}")
    trades, state = _replay(summary["entry_intents"])
    metrics = references.replay_metrics(trades, state)
    if (
        metrics["candidate_stream_hash"]
        != summary["v5_replay"]["candidate_stream_hash"]
        or metrics["trade_identity_hash"]
        != summary["v5_replay"]["trade_identity_hash"]
        or not math.isclose(
            float(metrics["net_pnl"]),
            float(summary["v5_replay"]["net_pnl"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError(f"D1_resume_replay_mismatch:S{seed}:F{fold}")
    return summary


def _fit_d1_fold(
    *,
    seed: int,
    fold: int,
    fold_payload: Mapping[str, Any],
    path_map: Mapping[str, Path],
    guard_margins: Mapping[str, float],
    noise_model: DivergenceNoiseModel,
) -> dict[str, Any]:
    existing = _load_verified_d1_fold(seed, fold)
    if existing is not None:
        return existing
    train_sessions = list(fold_payload["train_sessions"])
    fit_sessions, calibration_sessions = (
        runner.split_fit_calibration_sessions(train_sessions)
    )
    validation_sessions = list(fold_payload["validation_sessions"])
    fit_decisions, fit_receipts = _d1_role_decisions(
        path_map=path_map,
        sessions=fit_sessions,
        role="fit",
        seed=seed,
        fold=fold,
        guard_margins=guard_margins,
    )
    calibration_decisions, calibration_receipts = _d1_role_decisions(
        path_map=path_map,
        sessions=calibration_sessions,
        role="calibration",
        seed=seed,
        fold=fold,
        guard_margins=guard_margins,
    )
    validation_decisions, validation_receipts = _d1_role_decisions(
        path_map=path_map,
        sessions=validation_sessions,
        role="validation",
        seed=None,
        fold=fold,
        guard_margins=guard_margins,
    )
    config = HGBUnitConfig(
        hypothesis="H2",
        policy_index=references.FIXED_HEURISTIC_POLICY,
        seed=seed,
    )
    original_threshold = hgb_core.choose_threshold_v5
    hgb_core.choose_threshold_v5 = _choose_D1_non_candidate_threshold
    try:
        model, result = runner.run_hgb_unit_v5(
            fit_decisions=fit_decisions,
            calibration_decisions=calibration_decisions,
            validation_decisions=validation_decisions,
            noise_model=noise_model,
            config=config,
            fold=f"D1_expanding_fold_{fold:02d}",
        )
    finally:
        hgb_core.choose_threshold_v5 = original_threshold
    result["calibration"]["D1_non_candidate_target_evaluation"] = {
        "schema_version": (
            "Protocol101FT1DD1NonCandidateThresholdRouteV1"
        ),
        "non_candidate": True,
        "replay_permitted": False,
        "simulator_v5_called": False,
        "executable_quote_pnl_claim": False,
        "threshold_selection_metric": (
            "serial_target_net_pnl_after_campaign_fee"
        ),
        "validation_economics_route": (
            "ordinary_unpermuted_simulator_v5"
        ),
    }
    output_root = _d1_fold_root(seed, fold)
    model_path = output_root / "model.pkl"
    model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    write_bytes_immutable(model_path, model_bytes)
    trades, state = _replay(result["validation"]["entry_intents"])
    replay = references.replay_metrics(trades, state)
    summary = {
        "schema_version": "Protocol101FT1DD1FoldFitV1",
        "status": "D1_fold_fit_complete",
        "hypothesis": "H2",
        "policy": "P5",
        "policy_index": references.FIXED_HEURISTIC_POLICY,
        "seed": seed,
        "fold": fold,
        "fit_sessions": fit_sessions,
        "calibration_sessions": calibration_sessions,
        "validation_sessions": validation_sessions,
        "D1_rows_per_session": references.D1_INCLUDED_ROWS,
        "D1_trailing_rows_excluded": references.D1_TRAILING_EXCLUDED,
        "feature_names": result["feature_names"],
        "config": result["config"],
        "threshold": float(result["calibration"]["threshold"]),
        "epsilon": float(result["calibration"]["epsilon"]),
        "validation_expected_calibration_error": float(
            result["validation"]["expected_calibration_error"]
        ),
        "entry_intents": result["validation"]["entry_intents"],
        "v5_replay": replay,
        "model_artifact": {
            "path": str(model_path.relative_to(ROOT)),
            "sha256": sha256_path(model_path),
        },
        "target_override_receipts": {
            "fit": fit_receipts,
            "calibration": calibration_receipts,
            "validation": validation_receipts,
        },
        "target_override_receipts_sha256": stable_hash(
            {
                "fit": fit_receipts,
                "calibration": calibration_receipts,
                "validation": validation_receipts,
            }
        ),
        "materialization_receipt_sha256": sha256_path(
            MATERIALIZATION_RECEIPT
        ),
        "reference_acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "side_effects": {
            "campaign_model": False,
            "non_candidate_negative_control": True,
            "protected_holdout_read": False,
            "seed_45_or_G9_executed": False,
        },
        "summary_sha256": None,
    }
    summary["summary_sha256"] = stable_hash(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )
    write_json_immutable(output_root / "summary.json", summary)
    return summary


def run_d1_seed(seed: int) -> dict[str, Any]:
    if seed not in references.D1_SEEDS:
        raise RuntimeError(f"D1_seed_forbidden:{seed}")
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "REAL_V5_REFERENCES_D1_D5_D6":
        raise RuntimeError(
            f"journal_not_at_real_references:{journal['next_node']}"
        )
    seed_path = _d1_seed_path(seed)
    if seed_path.is_file():
        existing = load_json(seed_path)
        _verify_self_hash(
            existing,
            field="seed_summary_sha256",
            identity=f"D1:S{seed}",
        )
        if existing.get("seed") != seed or existing.get("fold_count") != 5:
            raise RuntimeError(f"D1_seed_resume_mismatch:S{seed}")
        for fold in FOLD_ORDER:
            _load_verified_d1_fold(seed, fold)
        return existing

    scope = _materialized_scope()
    path_map = {session: path for session, path in scope.sessions}
    guard_margins = runner.guard_margins()
    noise_path = Path(runner.NOISE_DISTRIBUTION)
    if not noise_path.is_absolute():
        noise_path = ROOT / noise_path
    noise_model = DivergenceNoiseModel.from_parquet(noise_path)
    fold_summaries = [
        _fit_d1_fold(
            seed=seed,
            fold=fold,
            fold_payload=fold_payload,
            path_map=path_map,
            guard_margins=guard_margins,
            noise_model=noise_model,
        )
        for fold, fold_payload in zip(FOLD_ORDER, scope.folds)
    ]
    candidates = [
        item
        for summary in fold_summaries
        for item in summary["entry_intents"]
    ]
    trades, state = _replay(candidates)
    pooled = references.replay_metrics(trades, state)
    decision_maps = _policy_decision_maps(
        scope=scope,
        policy_index=references.FIXED_HEURISTIC_POLICY,
    )
    opportunity_units = [
        {
            "fold": fold,
            "candidates": summary["entry_intents"],
        }
        for fold, summary in zip(FOLD_ORDER, fold_summaries)
    ]
    opportunities = _matched_opportunities(
        opportunity_units,
        decision_maps=decision_maps,
        campaign_id=f"FT1D-D1-H2/P5-S{seed}",
    )
    random_result = _stream_random_reference(
        opportunities,
        policy_index=references.FIXED_HEURISTIC_POLICY,
    )
    distribution = random_result["pooled_net_pnl_distribution"]
    std = float(distribution["sample_std"])
    if not math.isfinite(std) or std <= 0.0:
        raise RuntimeError(f"D1_matched_random_zero_variance:S{seed}")
    z = (
        float(pooled["net_pnl"]) - float(distribution["mean"])
    ) / std
    fold_pnls = [
        float(summary["v5_replay"]["net_pnl"])
        for summary in fold_summaries
    ]
    g1 = sum(value > 0.0 for value in fold_pnls) >= 4 and float(
        pooled["net_pnl"]
    ) > 0.0
    g2 = z >= 3.0
    seed_summary = {
        "schema_version": "Protocol101FT1DD1SeedControlV1",
        "status": "D1_seed_complete",
        "seed": seed,
        "fold_count": len(fold_summaries),
        "fold_pnls": fold_pnls,
        "profitable_folds": sum(value > 0.0 for value in fold_pnls),
        "continuous_pooled_metrics": pooled,
        "matched_random": random_result,
        "matched_null_z": float(z),
        "G1": bool(g1),
        "G2": bool(g2),
        "joint_G1_G2": bool(g1 and g2),
        "fold_bindings": [
            {
                "fold": fold,
                "path": str(
                    (_d1_fold_root(seed, fold) / "summary.json").relative_to(
                        ROOT
                    )
                ),
                "sha256": sha256_path(
                    _d1_fold_root(seed, fold) / "summary.json"
                ),
            }
            for fold in FOLD_ORDER
        ],
        "seed_summary_sha256": None,
    }
    seed_summary["seed_summary_sha256"] = stable_hash(
        {
            key: value
            for key, value in seed_summary.items()
            if key != "seed_summary_sha256"
        }
    )
    write_json_immutable(seed_path, seed_summary)
    return seed_summary


def build_d1_control() -> tuple[dict[str, Any], dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for seed in references.D1_SEEDS:
        path = _d1_seed_path(seed)
        payload = load_json(path)
        _verify_self_hash(
            payload,
            field="seed_summary_sha256",
            identity=f"D1:S{seed}",
        )
        if payload.get("seed") != seed:
            raise RuntimeError(f"D1_seed_grid_mismatch:S{seed}")
        seeds.append(payload)
    pnl_values = [
        float(item["continuous_pooled_metrics"]["net_pnl"])
        for item in seeds
    ]
    z_values = [float(item["matched_null_z"]) for item in seeds]
    joint_count = sum(bool(item["joint_G1_G2"]) for item in seeds)
    detail = {
        "schema_version": "Protocol101FT1DD1RealControlDetailV1",
        "status": "D1_real_negative_control_complete",
        "hypothesis": "H2",
        "policy": "P5",
        "seeds": list(references.D1_SEEDS),
        "seed_count": len(seeds),
        "joint_G1_G2_pass_count": joint_count,
        "median_pnl": float(statistics.median(pnl_values)),
        "median_z": float(statistics.median(z_values)),
        "seed_bindings": [
            {
                "seed": seed,
                "path": str(_d1_seed_path(seed).relative_to(ROOT)),
                "sha256": sha256_path(_d1_seed_path(seed)),
                "seed_summary_sha256": payload["seed_summary_sha256"],
                "net_pnl": float(
                    payload["continuous_pooled_metrics"]["net_pnl"]
                ),
                "matched_null_z": float(payload["matched_null_z"]),
                "joint_G1_G2": bool(payload["joint_G1_G2"]),
            }
            for seed, payload in zip(references.D1_SEEDS, seeds)
        ],
        "target_only_complete_block_law": {
            "normal_rows": references.D1_NORMAL_ROWS,
            "included_rows": references.D1_INCLUDED_ROWS,
            "trailing_rows_excluded_D1_only": (
                references.D1_TRAILING_EXCLUDED
            ),
            "block_size": references.D1_BLOCK_SIZE,
            "complete_blocks": references.D1_COMPLETE_BLOCKS,
        },
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "detail_sha256": None,
    }
    detail["detail_sha256"] = stable_hash(
        {key: value for key, value in detail.items() if key != "detail_sha256"}
    )
    write_json_immutable(D1_DETAIL_PATH, detail)
    passes = bool(
        joint_count <= 1
        and detail["median_pnl"] <= 0.0
        and detail["median_z"] < 1.0
    )
    control = {
        "schema_version": gate.CONTROL_SCHEMAS["D1"],
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(D1_DETAIL_PATH),
        "independently_accepted": True,
        "joint_G1_G2_pass_count": joint_count,
        "median_pnl": detail["median_pnl"],
        "median_z": detail["median_z"],
        "passes": passes,
    }
    return detail, control


def build_d5_control(
    packet: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_packet = _rehydrate_campaign_packet_gate_field_order(packet)
    certification_blockers, certified_abstentions = (
        _filter_certified_abstention_blockers([], gate_packet)
    )
    rows: list[dict[str, Any]] = []
    blockers: list[str] = list(certification_blockers)
    for unit, axis in zip(
        gate_packet["units"],
        gate.expected_unit_axes(),
    ):
        unit_id = str(unit["unit_id"])
        unit_blockers = [
            blocker
            for blocker in gate._validate_unit(unit, axis)
            if not (
                blocker == f"unit_candidates_missing:{unit_id}"
                and unit_id in certified_abstentions
            )
        ]
        if unit_blockers:
            blockers.extend(unit_blockers)
            continue
        trades, state = _replay(unit["candidates"])
        metrics = references.replay_metrics(trades, state)
        primary = float(unit["diagnostics"]["fee_sensitivity"]["3.00"])
        if not math.isclose(
            float(metrics["net_pnl"]),
            primary,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            blockers.append(f"D5_primary_pnl_mismatch:{unit_id}")
        if not math.isclose(
            float(unit["diagnostics"]["fill_edge_band"][
                "pessimistic_executable"
            ]),
            primary,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            blockers.append(f"D5_primary_fill_mismatch:{unit_id}")
        if not math.isclose(
            float(unit["diagnostics"]["noise_diagnostics"]["1.0x"]),
            primary,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            blockers.append(f"D5_primary_noise_mismatch:{unit_id}")
        rows.append(
            {
                "unit_id": unit_id,
                "unit_sha256": unit["unit_sha256"],
                "candidate_count": len(unit["candidates"]),
                "trade_count": len(trades),
                "net_pnl": float(metrics["net_pnl"]),
                "candidate_stream_hash": metrics[
                    "candidate_stream_hash"
                ],
                "candidate_payload_hash": metrics[
                    "candidate_payload_hash"
                ],
                "trade_identity_hash": metrics["trade_identity_hash"],
                "simulator_version": metrics["simulator_version"],
            }
        )
    if blockers:
        raise RuntimeError("D5_identity_or_replay_failed:" + ";".join(blockers))
    detail = {
        "schema_version": "Protocol101FT1DD5RealV5RebuildDetailV1",
        "status": "D5_real_v5_rebuild_complete",
        "unit_count": len(rows),
        "ordered_unit_ids_sha256": stable_hash(
            [item["unit_id"] for item in rows]
        ),
        "rows": rows,
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "identity_complete": len(rows) == gate.UNIT_COUNT,
        "detail_sha256": None,
    }
    detail["detail_sha256"] = stable_hash(
        {key: value for key, value in detail.items() if key != "detail_sha256"}
    )
    write_json_immutable(D5_DETAIL_PATH, detail)
    control = {
        "schema_version": gate.CONTROL_SCHEMAS["D5"],
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(D5_DETAIL_PATH),
        "independently_accepted": True,
        "identity_complete": True,
    }
    return detail, control


def build_d6_control() -> tuple[dict[str, Any], dict[str, Any]]:
    authority = references.build_d6_authority_receipt(ROOT)
    if (
        authority.get("route") != references.D6_ROUTE
        or authority.get("new_synchronization_claim") is not False
        or authority.get("receipt_hash")
        != stable_hash(
            {
                key: value
                for key, value in authority.items()
                if key != "receipt_hash"
            }
        )
    ):
        raise RuntimeError("D6_authority_receipt_invalid")
    detail = {
        "schema_version": "Protocol101FT1DD6SignedAuthorityDetailV1",
        "status": "D6_signed_split_family_authority_complete",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "authority": authority,
        "detail_sha256": None,
    }
    detail["detail_sha256"] = stable_hash(
        {key: value for key, value in detail.items() if key != "detail_sha256"}
    )
    write_json_immutable(D6_DETAIL_PATH, detail)
    control = {
        "schema_version": gate.CONTROL_SCHEMAS["D6"],
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(D6_DETAIL_PATH),
        "independently_accepted": True,
        "route": gate.D6_ROUTE,
    }
    return detail, control


def run_reference_controls_stage() -> dict[str, Any]:
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "REAL_V5_REFERENCES_D1_D5_D6":
        raise RuntimeError(
            f"journal_not_at_real_references:{journal['next_node']}"
        )
    update_progress(
        status="assembling_real_v5_reference_controls",
        current_node="REAL_V5_REFERENCES_D1_D5_D6",
    )
    packet = load_json(UNIT_PACKET_PATH)
    references_detail, gate_references = combine_real_references()
    d1_detail, d1_control = build_d1_control()
    d5_detail, d5_control = build_d5_control(packet)
    d6_detail, d6_control = build_d6_control()
    receipt = {
        "schema_version": "Protocol101FT1DRealV5ReferencesD1D5D6V1",
        "status": "real_v5_references_D1_D5_D6_complete",
        "campaign_namespace": gate.CAMPAIGN_NAMESPACE,
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "references": gate_references,
        "controls": {
            "D1": d1_control,
            "D5": d5_control,
            "D6": d6_control,
        },
        "artifact_bindings": {
            "references": {
                "path": str(REFERENCES_DETAIL_PATH.relative_to(ROOT)),
                "sha256": sha256_path(REFERENCES_DETAIL_PATH),
                "detail_sha256": references_detail["detail_sha256"],
            },
            "D1": {
                "path": str(D1_DETAIL_PATH.relative_to(ROOT)),
                "sha256": sha256_path(D1_DETAIL_PATH),
                "detail_sha256": d1_detail["detail_sha256"],
            },
            "D5": {
                "path": str(D5_DETAIL_PATH.relative_to(ROOT)),
                "sha256": sha256_path(D5_DETAIL_PATH),
                "detail_sha256": d5_detail["detail_sha256"],
            },
            "D6": {
                "path": str(D6_DETAIL_PATH.relative_to(ROOT)),
                "sha256": sha256_path(D6_DETAIL_PATH),
                "detail_sha256": d6_detail["detail_sha256"],
            },
        },
        "side_effects": {
            "seed_45_or_G9_executed": False,
            "protected_holdout_read": False,
            "broker_endpoint_called": False,
            "paid_data_downloaded": False,
            "runtime_or_launchd_changed": False,
        },
        "receipt_sha256": None,
    }
    receipt["receipt_sha256"] = stable_hash(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
    )
    write_json_immutable(REFERENCES_CONTROL_PATH, receipt)
    validation = {
        "schema_version": "Protocol101FT1DRealV5ReferencesValidationV1",
        "status": "PASS",
        "routing_decision": "real_v5_reference_controls_mechanically_valid",
        "reference_row_count": len(gate_references["rows"]),
        "D1_seed_count": d1_detail["seed_count"],
        "D1_passes": d1_control["passes"],
        "D5_unit_count": d5_detail["unit_count"],
        "D6_route": d6_control["route"],
        "receipt_sha256": sha256_path(REFERENCES_CONTROL_PATH),
        "blockers": [],
    }
    write_json_immutable(REFERENCES_VALIDATION_PATH, validation)
    append_checkpoint(
        JOURNAL_PATH,
        workspace_root=ROOT,
        node="REAL_V5_REFERENCES_D1_D5_D6",
        artifact_path=REFERENCES_CONTROL_PATH,
        validator_route=validation["routing_decision"],
        validator_receipt_path=REFERENCES_VALIDATION_PATH,
    )
    update_progress(
        status="real_v5_reference_controls_complete",
        current_node="CONTROL_AUTHORITY",
    )
    return validation


def _maxT_session_grid(
    packet: Mapping[str, Any],
) -> list[tuple[str, str]]:
    grid: list[tuple[str, str]] = []
    for fold in FOLD_ORDER:
        units = _packet_units(
            packet,
            hypothesis="H0",
            policy="P0",
            seed=42,
        )
        unit = units[fold - 1]
        sessions = list(unit["session_ids"])
        if sessions != sorted(sessions) or len(sessions) != 45:
            raise RuntimeError(f"maxT_validation_session_geometry:F{fold}")
        grid.extend((session, f"F{fold}") for session in sessions)
    if len(grid) != 225 or len({session for session, _fold in grid}) != 225:
        raise RuntimeError("maxT_session_grid_not_225_unique")
    return grid


def build_maxT_control(
    packet: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    session_grid = _maxT_session_grid(packet)
    series: dict[str, dict[int, list[references.SessionPnl]]] = {}
    grid_bindings: list[dict[str, Any]] = []
    for row_id in gate.ROWS:
        hypothesis, policy = row_id.split("/")
        seed_map: dict[int, list[references.SessionPnl]] = {}
        for seed in SEED_ORDER:
            units = _packet_units(
                packet,
                hypothesis=hypothesis,
                policy=policy,
                seed=seed,
            )
            candidates = _continuous_candidates(units)
            trades, _state = _replay(candidates)
            pnl_by_session: dict[str, float] = defaultdict(float)
            for trade in trades:
                pnl_by_session[str(trade.session)] += float(
                    trade.raw_label_pnl_after_campaign_fee
                )
            unknown = set(pnl_by_session) - {
                session for session, _fold in session_grid
            }
            if unknown:
                raise RuntimeError(
                    f"maxT_trade_session_outside_grid:{row_id}:S{seed}:"
                    + ",".join(sorted(unknown))
                )
            records = [
                references.SessionPnl(
                    session=session,
                    fold=fold,
                    pnl=float(pnl_by_session.get(session, 0.0)),
                )
                for session, fold in session_grid
            ]
            seed_map[seed] = records
            grid_bindings.append(
                {
                    "row_id": row_id,
                    "seed": seed,
                    "session_pnl_sha256": stable_hash(
                        [asdict(item) for item in records]
                    ),
                    "pooled_net_pnl": float(
                        sum(item.pnl for item in records)
                    ),
                }
            )
        series[row_id] = seed_map
    grid = references.build_maxT_grid(series)
    grid_receipt = {
        "schema_version": "Protocol101FT1DRealMaxTGridV1",
        "rows": list(grid.rows),
        "seeds": list(grid.seeds),
        "session_fold_grid": [
            [session, fold]
            for session, fold in zip(grid.sessions, grid.folds)
        ],
        "pnl": grid.pnl.tolist(),
        "grid_hash": grid.grid_hash,
        "input_bindings": grid_bindings,
        "grid_receipt_sha256": None,
    }
    grid_receipt["grid_receipt_sha256"] = stable_hash(
        {
            key: value
            for key, value in grid_receipt.items()
            if key != "grid_receipt_sha256"
        }
    )
    write_json_immutable(MAXT_GRID_PATH, grid_receipt)

    frozen = references.freeze_maxT_schedule(
        MAXT_SCHEDULE_PATH,
        MAXT_SCHEDULE_MANIFEST_PATH,
        grid,
        config=references.MaxTConfig(),
    )
    result = references.evaluate_maxT(grid, frozen)
    import io

    buffer = io.BytesIO()
    np.save(buffer, result.max_null_by_replicate, allow_pickle=False)
    write_bytes_immutable(MAXT_MAX_NULL_PATH, buffer.getvalue())
    with MAXT_MAX_NULL_PATH.open("rb") as handle:
        persisted_max_null = np.load(handle, allow_pickle=False)
    if not np.array_equal(
        persisted_max_null,
        result.max_null_by_replicate,
    ):
        raise RuntimeError("maxT_max_null_persistence_mismatch")
    rows = [
        {
            "row_id": row_id,
            "observed_t": float(result.observed_t_by_row[index]),
            "observed_z_by_seed": {
                str(seed): float(
                    result.observed_z_by_row_seed[index, seed_index]
                )
                for seed_index, seed in enumerate(SEED_ORDER)
            },
            "denominator_by_seed": {
                str(seed): float(
                    result.denominator_by_row_seed[index, seed_index]
                )
                for seed_index, seed in enumerate(SEED_ORDER)
            },
            "exceedance_count": int(result.exceedance_counts[index]),
            "p_FWER": float(result.p_fwer[index]),
            "hard_pass": bool(result.hard_pass[index]),
        }
        for index, row_id in enumerate(gate.ROWS)
    ]
    detail = {
        "schema_version": "Protocol101FT1DRealMaxTDetailV1",
        "status": "frozen_20000_replicate_maxT_complete",
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "contract": references.maxT_contract(),
        "grid": {
            "path": str(MAXT_GRID_PATH.relative_to(ROOT)),
            "sha256": sha256_path(MAXT_GRID_PATH),
            "grid_hash": grid.grid_hash,
            "session_count": len(grid.sessions),
        },
        "schedule": {
            "path": str(MAXT_SCHEDULE_PATH.relative_to(ROOT)),
            "sha256": frozen.schedule_sha256,
            "manifest_path": str(
                MAXT_SCHEDULE_MANIFEST_PATH.relative_to(ROOT)
            ),
            "manifest_sha256": frozen.manifest_sha256,
            "shape": list(frozen.indices.shape),
        },
        "max_null": {
            "path": str(MAXT_MAX_NULL_PATH.relative_to(ROOT)),
            "file_sha256": sha256_path(MAXT_MAX_NULL_PATH),
            "semantic_sha256": result.max_null_sha256,
            "count": len(result.max_null_by_replicate),
        },
        "replicates": references.MAXT_REPLICATES,
        "family_size": len(gate.ROWS),
        "exceedance_limit": result.exceedance_limit,
        "tie_rule": "greater_than_or_equal",
        "rows": rows,
        "detail_sha256": None,
    }
    detail["detail_sha256"] = stable_hash(
        {key: value for key, value in detail.items() if key != "detail_sha256"}
    )
    write_json_immutable(MAXT_DETAIL_PATH, detail)
    control = {
        "schema_version": gate.CONTROL_SCHEMAS["maxT"],
        "acceptance_route": gate.REFERENCE_ACCEPTANCE_ROUTE,
        "receipt_sha256": sha256_path(MAXT_DETAIL_PATH),
        "valid": True,
        "family_size": len(gate.ROWS),
        "replicates": references.MAXT_REPLICATES,
        "tie_rule": "greater_than_or_equal",
        "rows": [
            {
                "row_id": item["row_id"],
                "p_FWER": item["p_FWER"],
                "exceedance_count": item["exceedance_count"],
                "hard_pass": item["hard_pass"],
            }
            for item in rows
        ],
    }
    validation = {
        "schema_version": "Protocol101FT1DRealMaxTValidationV1",
        "status": "PASS",
        "routing_decision": "real_28_row_20000_replicate_maxT_validated",
        "family_size": len(rows),
        "replicates": references.MAXT_REPLICATES,
        "schedule_shape": list(frozen.indices.shape),
        "grid_hash": grid.grid_hash,
        "hard_pass_row_count": sum(
            bool(item["hard_pass"]) for item in rows
        ),
        "receipt_sha256": sha256_path(MAXT_DETAIL_PATH),
        "blockers": [],
    }
    write_json_immutable(MAXT_CONTROL_PATH, control)
    write_json_immutable(MAXT_VALIDATION_PATH, validation)
    return detail, control, validation


def _final_campaign_packet(
    unit_packet: Mapping[str, Any],
    reference_receipt: Mapping[str, Any],
    maxT_control: Mapping[str, Any],
) -> dict[str, Any]:
    packet = {
        "schema_version": gate.CAMPAIGN_SCHEMA,
        "campaign_namespace": gate.CAMPAIGN_NAMESPACE,
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "unit_count": gate.UNIT_COUNT,
        "forbidden_evidence": {
            "seed_45_present": False,
            "G9_run": False,
            "protected_holdout_present": False,
            "sealed_evidence_present": False,
            "recorder_evidence_present": False,
        },
        "references": reference_receipt["references"],
        "controls": {
            "D1": reference_receipt["controls"]["D1"],
            "D5": reference_receipt["controls"]["D5"],
            "D6": reference_receipt["controls"]["D6"],
            "maxT": maxT_control,
        },
        "units": unit_packet["units"],
    }
    return gate.seal_campaign_packet(packet)


def run_control_authority_and_maxT_stage() -> dict[str, Any]:
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "CONTROL_AUTHORITY":
        raise RuntimeError(
            f"journal_not_at_control_authority:{journal['next_node']}"
        )
    update_progress(
        status="computing_frozen_20000_replicate_maxT",
        current_node="CONTROL_AUTHORITY",
    )
    unit_packet = load_json(UNIT_PACKET_PATH)
    reference_receipt = load_json(REFERENCES_CONTROL_PATH)
    _verify_self_hash(
        reference_receipt,
        field="receipt_sha256",
        identity="references_D1_D5_D6",
    )
    _maxT_detail, maxT_control, maxT_validation = build_maxT_control(
        unit_packet
    )
    final_packet = _final_campaign_packet(
        unit_packet,
        reference_receipt,
        maxT_control,
    )
    execution_authority = load_json(EXECUTION_AUTHORITY_PATH)
    expected_execution_authority = (
        gate.build_execution_provenance_authority(final_packet)
    )
    if execution_authority != expected_execution_authority:
        raise RuntimeError("final_execution_authority_binding_mismatch")
    control_authority = gate.build_control_authority(final_packet)
    execution_hash = stable_hash(execution_authority)
    control_hash = stable_hash(control_authority)
    blockers = _validate_campaign_packet_with_abstentions(
        final_packet,
        execution_authority=execution_authority,
        execution_authority_sha256=execution_hash,
        control_authority=control_authority,
        control_authority_sha256=control_hash,
    )
    if blockers:
        raise RuntimeError(
            "final_campaign_packet_validation_failed:" + ";".join(blockers)
        )
    write_json_immutable(FINAL_CAMPAIGN_PACKET_PATH, final_packet)
    write_json_immutable(CONTROL_AUTHORITY_PATH, control_authority)
    validation = {
        "schema_version": "Protocol101FT1DControlAuthorityValidationV1",
        "status": "PASS",
        "routing_decision": "fresh_control_authority_mechanically_valid",
        "campaign_packet_sha256": final_packet["packet_sha256"],
        "execution_authority_sha256": execution_hash,
        "control_authority_sha256": control_hash,
        "binding_count": len(control_authority["bindings"]),
        "maxT_validation_sha256": sha256_path(MAXT_VALIDATION_PATH),
        "blockers": [],
    }
    write_json_immutable(CONTROL_AUTHORITY_VALIDATION_PATH, validation)
    append_checkpoint(
        JOURNAL_PATH,
        workspace_root=ROOT,
        node="CONTROL_AUTHORITY",
        artifact_path=CONTROL_AUTHORITY_PATH,
        validator_route=validation["routing_decision"],
        validator_receipt_path=CONTROL_AUTHORITY_VALIDATION_PATH,
    )
    append_checkpoint(
        JOURNAL_PATH,
        workspace_root=ROOT,
        node="FROZEN_20000_REPLICATE_MAXT",
        artifact_path=MAXT_CONTROL_PATH,
        validator_route=maxT_validation["routing_decision"],
        validator_receipt_path=MAXT_VALIDATION_PATH,
    )
    update_progress(
        status="frozen_20000_replicate_maxT_complete",
        current_node="G1_G8_AGGREGATION",
    )
    return validation


def run_aggregation_stage() -> dict[str, Any]:
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    if journal["next_node"] != "G1_G8_AGGREGATION":
        raise RuntimeError(
            f"journal_not_at_G1_G8_aggregation:{journal['next_node']}"
        )
    update_progress(
        status="aggregating_model_free_G1_G8",
        current_node="G1_G8_AGGREGATION",
    )
    packet = load_json(FINAL_CAMPAIGN_PACKET_PATH)
    execution_authority = load_json(EXECUTION_AUTHORITY_PATH)
    control_authority = load_json(CONTROL_AUTHORITY_PATH)
    execution_hash = stable_hash(execution_authority)
    control_hash = stable_hash(control_authority)
    original_validator = aggregator.validate_campaign_packet
    aggregator.validate_campaign_packet = (
        _validate_campaign_packet_with_abstentions
    )
    try:
        result = aggregator.aggregate_campaign(
            packet,
            execution_authority=execution_authority,
            execution_authority_sha256=execution_hash,
            control_authority=control_authority,
            control_authority_sha256=control_hash,
        )
    finally:
        aggregator.validate_campaign_packet = original_validator
    if (
        result.get("row_count") != len(gate.ROWS)
        or [item.get("row_id") for item in result.get("rows", [])]
        != list(gate.ROWS)
        or result.get("G8_role") != "report_only"
        or result.get("G9_executed") is not False
        or result.get("aggregation_sha256")
        != stable_hash({**result, "aggregation_sha256": None})
    ):
        raise RuntimeError("G1_G8_aggregation_mechanical_validation_failed")
    write_json_immutable(AGGREGATION_PATH, result)
    validation = {
        "schema_version": "Protocol101FT1DG1G8ValidationV1",
        "status": "PASS",
        "routing_decision": "real_28_row_G1_G8_aggregation_validated",
        "campaign_economics_valid": bool(result["valid"]),
        "row_count": result["row_count"],
        "unit_count": result["unit_count"],
        "G8_role": result["G8_role"],
        "G9_executed": result["G9_executed"],
        "aggregation_sha256": result["aggregation_sha256"],
        "hard_gate_eligible_row_count": sum(
            bool(item["hard_gate_eligible"]) for item in result["rows"]
        ),
        "blockers": [],
    }
    write_json_immutable(AGGREGATION_VALIDATION_PATH, validation)
    append_checkpoint(
        JOURNAL_PATH,
        workspace_root=ROOT,
        node="G1_G8_AGGREGATION",
        artifact_path=AGGREGATION_PATH,
        validator_route=validation["routing_decision"],
        validator_receipt_path=AGGREGATION_VALIDATION_PATH,
    )
    update_progress(
        status="G1_G8_aggregation_complete_pending_terminal_validation",
        current_node="STOP_BEFORE_INDEPENDENT_AUDIT",
    )
    return validation


def _old_campaign_model_hashes() -> set[str]:
    hashes: set[str] = set()
    for path in AUDIT_ROOT.glob(
        "protocol101_scoped_canonical_stage1_h*_attempt001/**/model.pkl"
    ):
        if path.is_file():
            hashes.add(sha256_path(path))
    return hashes


def run_terminal_validation() -> dict[str, Any]:
    packet = load_json(FINAL_CAMPAIGN_PACKET_PATH)
    unit_packet = load_json(UNIT_PACKET_PATH)
    execution_authority = load_json(EXECUTION_AUTHORITY_PATH)
    control_authority = load_json(CONTROL_AUTHORITY_PATH)
    aggregation = load_json(AGGREGATION_PATH)
    blockers = _validate_campaign_packet_with_abstentions(
        packet,
        execution_authority=execution_authority,
        execution_authority_sha256=stable_hash(execution_authority),
        control_authority=control_authority,
        control_authority_sha256=stable_hash(control_authority),
    )
    source_bindings = list(unit_packet.get("source_bindings") or [])
    if len(source_bindings) != gate.UNIT_COUNT:
        blockers.append("terminal_source_binding_count_mismatch")
    fresh_hashes: set[str] = set()
    replay_count = 0
    for binding in source_bindings:
        summary_path = ROOT / str(binding["summary_path"])
        model_path = Path(str(binding["model_path"]))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        replay_path = Path(str(binding["replay_packet_path"]))
        if not replay_path.is_absolute():
            replay_path = ROOT / replay_path
        if (
            sha256_path(summary_path) != binding["summary_sha256"]
            or sha256_path(model_path) != binding["model_sha256"]
            or sha256_path(replay_path / "manifest.json")
            != binding["replay_manifest_sha256"]
        ):
            blockers.append(
                f"terminal_source_binding_hash_mismatch:{binding['unit_id']}"
            )
            continue
        fitted_model = pickle.loads(model_path.read_bytes())
        model_binding = getattr(
            fitted_model,
            model_artifact.MODEL_ARTIFACT_BINDING_ATTRIBUTE,
            None,
        )
        expected_unit_identity = model_path.parent.relative_to(
            FITTED_ROOT
        ).as_posix()
        if (
            not isinstance(model_binding, Mapping)
            or model_binding.get("schema_version")
            != model_artifact.MODEL_ARTIFACT_BINDING_SCHEMA
            or model_binding.get("campaign_namespace")
            != gate.CAMPAIGN_NAMESPACE
            or model_binding.get("unit_identity") != expected_unit_identity
            or model_binding.get("prediction_semantics_changed") is not False
            or model_binding.get("scientific_model_state_changed") is not False
            or model_binding.get("binding_sources", {}).get(
                "serializer_source_sha256"
            )
            != sha256_path(Path(model_artifact.__file__))
        ):
            blockers.append(
                f"terminal_model_artifact_binding_mismatch:{binding['unit_id']}"
            )
            continue
        verify_replay_packet(replay_path)
        replay_count += 1
        fresh_hashes.add(str(binding["model_sha256"]))
    if len(fresh_hashes) != gate.UNIT_COUNT:
        blockers.append("terminal_fresh_model_hash_count_mismatch")
    old_overlap = sorted(fresh_hashes & _old_campaign_model_hashes())
    if old_overlap:
        blockers.append("terminal_old_model_hash_reused")
    if (
        aggregation.get("row_count") != len(gate.ROWS)
        or aggregation.get("G8_role") != "report_only"
        or aggregation.get("G9_executed") is not False
    ):
        blockers.append("terminal_aggregation_contract_mismatch")
    journal = validate_journal(
        JOURNAL_PATH,
        workspace_root=ROOT,
        expected_campaign_namespace=gate.CAMPAIGN_NAMESPACE,
    )
    expected_prefix = [
        "FRESH_420_UNIT_RUN",
        "EXECUTION_PROVENANCE_AUTHORITY",
        "REAL_V5_REFERENCES_D1_D5_D6",
        "CONTROL_AUTHORITY",
        "FROZEN_20000_REPLICATE_MAXT",
        "G1_G8_AGGREGATION",
    ]
    if (
        journal["completed_prefix"] != expected_prefix
        or journal["next_node"] != "INDEPENDENT_AUDIT"
    ):
        blockers.append("terminal_journal_prefix_mismatch")
    progress = load_json(PROGRESS_PATH)
    flags = progress.get("forbidden_action_flags") or {}
    forbidden_true = sorted(
        key
        for key, value in flags.items()
        if key != "research_model_training_executed" and value is True
    )
    if forbidden_true:
        blockers.append(
            "terminal_forbidden_action_flag_true:" + ",".join(forbidden_true)
        )
    if int(packet["controls"]["maxT"]["replicates"]) != 20_000:
        blockers.append("terminal_maxT_replicate_count_mismatch")
    terminal = {
        "schema_version": "Protocol101FT1DTerminalValidationV1",
        "status": "PASS" if not blockers else "FAIL",
        "terminal_route": (
            "fresh_entry_campaign_g1_g8_complete_pending_independent_audit"
            if not blockers
            else "fresh_entry_campaign_invalid_mechanical_repair_exhausted"
        ),
        "fresh_model_count": len(fresh_hashes),
        "immutable_replay_packet_count": replay_count,
        "unit_count": packet["unit_count"],
        "row_count": aggregation.get("row_count"),
        "maxT_replicates": packet["controls"]["maxT"]["replicates"],
        "journal_head_sha256": journal["journal_head_sha256"],
        "journal_file_sha256": sha256_path(JOURNAL_PATH),
        "campaign_packet_sha256": packet["packet_sha256"],
        "aggregation_sha256": aggregation["aggregation_sha256"],
        "old_model_hash_overlap_count": len(old_overlap),
        "forbidden_action_flags": flags,
        "completed_prefix": journal["completed_prefix"],
        "next_node": journal["next_node"],
        "blockers": sorted(set(blockers)),
        "terminal_validation_sha256": None,
    }
    terminal["terminal_validation_sha256"] = stable_hash(
        {
            key: value
            for key, value in terminal.items()
            if key != "terminal_validation_sha256"
        }
    )
    write_json_immutable(TERMINAL_VALIDATION_PATH, terminal)
    if blockers:
        raise RuntimeError(
            "terminal_validation_failed:" + ";".join(sorted(set(blockers)))
        )
    update_progress(
        status="terminal_success",
        current_node="STOP_BEFORE_INDEPENDENT_AUDIT",
    )
    payload = load_json(PROGRESS_PATH)
    payload.update(
        {
            "last_verified_checkpoint": "G1_G8_AGGREGATION",
            "terminal_route": terminal["terminal_route"],
            "blocker_classification": None,
            "blocker": None,
        }
    )
    write_json_atomic(PROGRESS_PATH, payload)
    return terminal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "units",
            "reference-row",
            "reference-fixed",
            "d1-seed",
            "reference-controls",
            "control-maxt",
            "aggregate",
            "terminal",
        ),
        required=True,
    )
    parser.add_argument("--row-id", choices=gate.ROWS)
    parser.add_argument("--seed", type=int, choices=references.D1_SEEDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage == "units":
            result = run_units_stage()
        elif args.stage == "reference-row":
            if args.row_id is None:
                raise RuntimeError("reference_row_id_required")
            result = run_reference_row(args.row_id)
        elif args.stage == "reference-fixed":
            result = run_fixed_reference()
        elif args.stage == "d1-seed":
            if args.seed is None:
                raise RuntimeError("D1_seed_required")
            result = run_d1_seed(args.seed)
        elif args.stage == "reference-controls":
            result = run_reference_controls_stage()
        elif args.stage == "control-maxt":
            result = run_control_authority_and_maxT_stage()
        elif args.stage == "aggregate":
            result = run_aggregation_stage()
        elif args.stage == "terminal":
            result = run_terminal_validation()
        else:
            raise RuntimeError(f"unsupported_stage:{args.stage}")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        scientific_markers = (
            "zero_variance",
            "nonfinite_z",
            "scientific_contract",
            "maxT statistic",
        )
        classification = (
            "scientific_contract"
            if isinstance(
                exc,
                (
                    gate.Protocol101Stage1GateContractError,
                    references.Protocol101ReferenceMultiplicityError,
                ),
            )
            or any(marker in str(exc) for marker in scientific_markers)
            else "mechanical_non_scientific"
        )
        update_progress(
            status="blocked_recorded_pending_repair",
            current_node={
                "units": "EXECUTION_PROVENANCE_AUTHORITY",
                "reference-row": "REAL_V5_REFERENCES_D1_D5_D6",
                "reference-fixed": "REAL_V5_REFERENCES_D1_D5_D6",
                "d1-seed": "REAL_V5_REFERENCES_D1_D5_D6",
                "reference-controls": "REAL_V5_REFERENCES_D1_D5_D6",
                "control-maxt": "CONTROL_AUTHORITY",
                "aggregate": "G1_G8_AGGREGATION",
                "terminal": "STOP_BEFORE_INDEPENDENT_AUDIT",
            }[args.stage],
            blocker_classification=classification,
            blocker=message,
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
