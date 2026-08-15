"""Run the owner-authorized quarantined walking-skeleton Stages 2 and 3.

The real Stage-1 composer remains byte-frozen and continues to abstain.  This
runner overrides only its downstream confidence abstention on a deterministic,
capped subset of real A6-passing proposals.  It uses the frozen 13-session
official cbbo-1s slice, fits a throwaway HGB lifecycle baseline, applies the
upward-only floor, replays the owned three-session validation slice through
serial simulator v5, and emits the D60 visuals.  There is no Stage 4 path.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from v4.model.protocol101_regimen_repair import ExitReason
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.model.protocol101_walking_skeleton import compose_decision
from v4.model.protocol101_walking_skeleton_lifecycle import (
    FEATURE_NAMES,
    QUARANTINE_LABELS,
    QUARANTINE_TEXT,
    FloorSpec,
    apply_entry_safety_prefilter,
    apply_lifecycle_policy,
    attach_exit_probabilities,
    build_one_second_trajectory,
    contract_id_to_raw_symbol,
    fit_lifecycle_baseline,
)
from v4.scripts.export_protocol101_trade_charts import (
    add_equity_fields,
    attach_spx_prices,
    load_spx_bars,
    write_equity_html,
    write_trades_html,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage2_3"
STAGE0 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage0"
STAGE1 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage1"
AUTHORITY = ROOT / (
    "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
GRAPH = ROOT / "v4/docs/protocol101/training/execution/PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
AUTHORITY_SHA256 = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
GRAPH_SHA256 = "9955085a31840da63057761a620a5ec2995e04f05ff2aa5f4906afd795726a08"

PREFIX = "protocol101_walking_skeleton_throwaway_paper_only_"
CAL_PREDICTIONS = STAGE1 / f"{PREFIX}option_d_calibration_predictions.parquet"
FULL_TENSOR = STAGE1 / f"{PREFIX}option_d_full45_entry_tensor.parquet"
HEADLINE = STAGE1 / f"{PREFIX}option_d_headline_decisions.csv"
CDF = STAGE1 / f"{PREFIX}option_d_fit_nested_cdfs.json"
STAGE1_RECEIPT = STAGE1 / "receipt.json"
DATA_SLICE = STAGE0 / "data_slice.json"

FORCED_MANIFEST = OUT / "forced_buy_manifest.json"
EXIT_TENSOR = OUT / f"{PREFIX}forced_buy_plumbing_harness_exit_tensor.parquet"
MODEL_ARTIFACT = OUT / f"{PREFIX}forced_buy_plumbing_harness_lifecycle_hgb.joblib"
MODEL_MANIFEST = OUT / "lifecycle_model_manifest.json"
FLOOR_ARTIFACT = OUT / "protective_floor_artifact.json"
FLOOR_ABLATION = OUT / "floor_ablation_report.json"
ACTION_LOG = OUT / f"{PREFIX}forced_buy_plumbing_harness_lifecycle_actions.parquet"
LIFECYCLE_RESULTS = OUT / "lifecycle_results.csv"
SERIAL_REPLAY = OUT / "serial_replay.json"
SERIAL_TRADES = OUT / "serial_trades.csv"
ENTRY_SAFETY = OUT / "entry_safety_audit.json"
EARLY_DRAWDOWN = OUT / "d58_early_drawdown_report_only.json"
FOUR_BUCKET = OUT / "four_bucket_distribution.json"
FOUR_BUCKET_HTML = OUT / "four_bucket_distribution.html"
RECEIPT = OUT / "receipt.json"
REPORT = OUT / "report.md"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key)) for key in fields} for row in rows])


def _frozen_paths() -> list[Path]:
    return [
        AUTHORITY,
        GRAPH,
        ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json",
        ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/composer_spec.json",
        ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/forecast_heads.json",
        ROOT / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/realized_label_audit_composer_spec.json",
        ROOT / "v4/model/protocol101_walking_skeleton.py",
        ROOT / "v4/model/protocol101_serial_simulator_v5.py",
        FULL_TENSOR,
        HEADLINE,
        CDF,
        STAGE1_RECEIPT,
    ]


def _hash_map(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_path(path) for path in paths}


def verify_authority_and_inputs() -> tuple[dict[str, Any], dict[str, str]]:
    if sha256_path(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("authority hash mismatch; stop before Stage 2")
    if sha256_path(GRAPH) != GRAPH_SHA256:
        raise RuntimeError("graph hash mismatch; stop before Stage 2")
    stage1 = json.loads(STAGE1_RECEIPT.read_text())
    if stage1.get("outcome") != "ABSTAINS":
        raise RuntimeError("real Stage-1 outcome is not the required ABSTAINS state")
    if stage1.get("product_contract_hash") != AUTHORITY_SHA256:
        raise RuntimeError("Stage-1 authority pin differs from the signed authority")
    data_slice = json.loads(DATA_SLICE.read_text())
    one_second = data_slice["exit_cbbo_1s_slice"]
    if int(one_second["session_count"]) != 13:
        raise RuntimeError("the frozen exit slice is not exactly 13 sessions")
    firewall = one_second["firewall_assertions"]
    forbidden_counts = [
        int(value)
        for key, value in firewall.items()
        if key.endswith("intersection_count")
    ]
    if any(forbidden_counts):
        raise RuntimeError("the frozen 1-second slice intersects a protected resource")
    for row in one_second["sessions"]:
        parquet = ROOT / row["parquet_path"]
        manifest = ROOT / row["manifest_path"]
        if sha256_path(parquet) != row["parquet_sha256"]:
            raise RuntimeError(f"1-second parquet hash mismatch: {parquet}")
        if sha256_path(manifest) != row["manifest_sha256"]:
            raise RuntimeError(f"1-second manifest hash mismatch: {manifest}")
    return data_slice, _hash_map(_frozen_paths())


def _compose_real_proposals() -> pd.DataFrame:
    stage1_receipt = json.loads(STAGE1_RECEIPT.read_text())
    error = stage1_receipt["calibration"]["composer_error_margins"]
    one_second_sessions = json.loads(DATA_SLICE.read_text())["exit_cbbo_1s_slice"]["sessions"]
    sessions = [str(row["date"]) for row in one_second_sessions]
    replay = pd.read_csv(HEADLINE)
    replay = replay[replay["session"].isin(sessions) & replay["proposed_contract_id"].notna()].copy()
    if len(replay) != 552:
        raise RuntimeError(f"expected exactly 552 real replay A6 proposals, found {len(replay)}")
    replay["source_prediction_split"] = "stage1_plumbing_replay"

    calibration_sessions = [session for session in sessions if session < "2025-03-05"]
    calibration = pd.read_parquet(CAL_PREDICTIONS)
    calibration = calibration[calibration["session"].isin(calibration_sessions)].copy()
    with CDF.open() as handle:
        cdfs = json.load(handle)
    calibration_rows: list[dict[str, Any]] = []
    for _, block in calibration.groupby(["session", "decision_time_ns"], sort=True):
        result = compose_decision(
            block,
            cdfs=cdfs,
            model_gap_error=float(error["cluster_gap"]["q90_absolute_error"]),
            mfe_error_margin_dollars=float(error["mfe_dollars"]["q90_absolute_error"]),
            mfe_error_margin_return=float(error["mfe_return"]["q90_absolute_error"]),
            source_transfer_error=0.0,
            uncertainty_multiplier=1.0,
            guardrail_alpha=0.0,
            action_conditioned_gate_available=False,
        )
        if result.get("proposed_contract_id") is not None:
            result["source_prediction_split"] = "stage1_calibration_recomposition"
            calibration_rows.append(result)
    calibration_proposals = pd.DataFrame(calibration_rows)
    if len(calibration_proposals) != 2_704:
        raise RuntimeError(
            f"expected 2704 real calibration proposals for the 1s slice, found {len(calibration_proposals)}"
        )
    return pd.concat([calibration_proposals, replay], ignore_index=True, sort=False)


def _schedule_forced_intents(
    proposals: pd.DataFrame,
    *,
    data_slice: dict[str, Any],
) -> list[dict[str, Any]]:
    partition = data_slice["exit_cbbo_1s_slice"]["frozen_partition"]
    fit = set(partition["fit_first_8_sessions"])
    calibration = set(partition["calibration_next_2_sessions"])
    replay = set(partition["plumbing_replay_last_3_sessions"])
    tensor_columns = [
        "session",
        "decision_time_ns",
        "contract_id",
        "entry_fill_time_ns",
        "entry_ask",
        "fill_recheck_pass_at_tplus1",
        "right",
        "strike_idx",
        "right_idx",
        "offset",
        "source_quote_time_ns",
        "source_context_time_ns",
        "h10_early_dd_dollars",
    ]
    tensor = pd.read_parquet(FULL_TENSOR, columns=tensor_columns)
    merged = proposals.merge(
        tensor,
        left_on=["session", "decision_time_ns", "proposed_contract_id"],
        right_on=["session", "decision_time_ns", "contract_id"],
        how="left",
        validate="one_to_one",
    )
    merged = merged[
        merged["fill_recheck_pass_at_tplus1"].fillna(False).astype(bool)
        & pd.to_numeric(merged["entry_ask"], errors="coerce").notna()
        & (merged["entry_ask"].astype(float) >= 1.0)
        & (merged["entry_ask"].astype(float) * 100.0 + 3.0 <= 500.0 + 1e-9)
    ].copy()
    intents: list[dict[str, Any]] = []
    for session, group in merged.groupby("session", sort=True):
        group = group.sort_values(["decision_time_ns", "contract_id"]).reset_index(drop=True)
        if len(group) < 6:
            raise RuntimeError(f"fewer than six fill-safe D48 proposals in {session}")
        positions = np.unique(np.rint(np.linspace(0, len(group) - 1, 6)).astype(int))
        if len(positions) != 6:
            raise RuntimeError(f"deterministic schedule did not yield six unique intents: {session}")
        for ordinal, (_, row) in enumerate(group.iloc[positions].iterrows(), start=1):
            if session in fit:
                split = "fit"
            elif session in calibration:
                split = "calibration"
            elif session in replay:
                split = "plumbing_replay"
            else:
                raise RuntimeError(f"session absent from frozen 1s partition: {session}")
            route = "full_policy"
            if session == "2025-03-06" and ordinal == 1:
                route = "learned_exit_path_canary"
            elif session == "2025-03-07" and ordinal == 1:
                route = "floor_path_canary"
            elif session == "2025-03-10" and ordinal == 1:
                route = "forced_flat_path_canary"
            intent = {
                "session": str(session),
                "split": split,
                "forced_ordinal": ordinal,
                "decision_time_ns": int(row["decision_time_ns"]),
                "entry_fill_time_ns": int(row["entry_fill_time_ns"]),
                "contract_id": str(row["contract_id"]),
                "right": str(row["right"]),
                "strike_idx": int(row["strike_idx"]),
                "right_idx": int(row["right_idx"]),
                "offset": _jsonable(row["offset"]),
                "entry_ask": float(row["entry_ask"]),
                "source_quote_time_ns": int(row["source_quote_time_ns"]),
                "source_context_time_ns": int(row["source_context_time_ns"]),
                "stage1_h10_early_dd_dollars": _jsonable(row["h10_early_dd_dollars"]),
                "composer_action": str(row["action"]),
                "composer_wait_reason": str(row["wait_reason"]),
                "composer_proposed_contract_id": str(row["proposed_contract_id"]),
                "composer_selected_cluster_score": _jsonable(row.get("selected_cluster_score")),
                "forced_action": "BUY",
                "forced_reason": (
                    "downstream_confidence_abstention_override_after_real_A6_entry_gate; "
                    f"source_wait_reason={row['wait_reason']}"
                ),
                "override_boundary": "downstream_confidence_abstention_only",
                "lifecycle_route": route,
                "quarantine_labels": list(QUARANTINE_LABELS),
            }
            if intent["composer_action"] != "WAIT":
                raise RuntimeError("forced wrapper received a non-WAIT composer action")
            if intent["composer_wait_reason"] not in {
                "uncertainty_margin_not_strictly_cleared",
                "mfe_error_margin_not_strictly_cleared",
                "q90_regret_bound_above_0_10",
                "action_conditioned_gate_unavailable",
            }:
                raise RuntimeError("forced wrapper would override a non-downstream gate")
            intents.append(intent)
    if len(intents) != 78:
        raise RuntimeError(f"expected six intents across 13 sessions, found {len(intents)}")
    return intents


def _raw_paths(data_slice: dict[str, Any]) -> dict[str, Path]:
    return {
        str(row["date"]): ROOT / str(row["parquet_path"])
        for row in data_slice["exit_cbbo_1s_slice"]["sessions"]
    }


def _build_tensor(intents: list[dict[str, Any]], raw_paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for number, intent in enumerate(intents, start=1):
        frame = build_one_second_trajectory(intent, parquet_path=raw_paths[intent["session"]])
        frame["intent_id"] = (
            f"{intent['session']}|{intent['decision_time_ns']}|{intent['contract_id']}"
        )
        frame["forced_ordinal"] = int(intent["forced_ordinal"])
        frame["lifecycle_route"] = str(intent["lifecycle_route"])
        frames.append(frame)
        print(f"exit_tensor {number:02d}/{len(intents)} {intent['session']} {intent['contract_id']} {len(frame)}s")
    return pd.concat(frames, ignore_index=True)


def _early_drawdown_report(tensor: pd.DataFrame, intents: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for intent in intents:
        trajectory = tensor[
            (tensor["session"] == intent["session"])
            & (tensor["decision_time_ns"] == intent["decision_time_ns"])
            & (tensor["contract_id"] == intent["contract_id"])
        ]
        first_ten = trajectory[trajectory["elapsed_seconds"] <= 600]
        observed = float(first_ten["net_pnl_dollars"].min())
        rows.append(
            {
                "session": intent["session"],
                "contract_id": intent["contract_id"],
                "decision_time_ns": intent["decision_time_ns"],
                "stage1_minute_early_drawdown_dollars": intent["stage1_h10_early_dd_dollars"],
                "official_cbbo_1s_early_drawdown_dollars": observed,
                "difference_dollars": (
                    None
                    if intent["stage1_h10_early_dd_dollars"] is None
                    else observed - float(intent["stage1_h10_early_dd_dollars"])
                ),
                "quarantine_labels": list(QUARANTINE_LABELS),
            }
        )
    frame = pd.DataFrame(rows).dropna(
        subset=["stage1_minute_early_drawdown_dollars", "official_cbbo_1s_early_drawdown_dollars"]
    )
    spearman = float(
        frame["stage1_minute_early_drawdown_dollars"].corr(
            frame["official_cbbo_1s_early_drawdown_dollars"], method="spearman"
        )
    )
    count = max(1, int(math.ceil(len(frame) * 0.25)))
    minute_worst = set(frame.nsmallest(count, "stage1_minute_early_drawdown_dollars").index)
    second_worst = set(frame.nsmallest(count, "official_cbbo_1s_early_drawdown_dollars").index)
    return {
        "status": "REPORT_ONLY",
        "formal_entry_ranking_or_alpha_claim": False,
        "sample_count": int(len(frame)),
        "spearman_rank_correlation": spearman,
        "worst_quartile_overlap_count": len(minute_worst & second_worst),
        "worst_quartile_size": count,
        "worst_quartile_overlap_fraction": len(minute_worst & second_worst) / count,
        "rows": rows,
        "quarantine_labels": list(QUARANTINE_LABELS),
    }


def _run_lifecycle(
    tensor: pd.DataFrame,
    intents: list[dict[str, Any]],
    threshold: float,
    floor_spec: FloorSpec,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    terminals: list[dict[str, Any]] = []
    action_frames: list[pd.DataFrame] = []
    for intent in intents:
        trajectory = tensor[
            (tensor["session"] == intent["session"])
            & (tensor["decision_time_ns"] == intent["decision_time_ns"])
            & (tensor["contract_id"] == intent["contract_id"])
        ].copy()
        terminal, actions = apply_lifecycle_policy(
            trajectory,
            threshold=threshold,
            route=str(intent["lifecycle_route"]),
            floor_spec=floor_spec,
        )
        terminals.append({**intent, **terminal})
        action_frames.append(pd.DataFrame(actions))
    return terminals, pd.concat(action_frames, ignore_index=True)


def _floor_ablation(
    tensor: pd.DataFrame,
    intents: list[dict[str, Any]],
    threshold: float,
    floor_spec: FloorSpec,
) -> dict[str, Any]:
    """Run D52 floor-on versus learned-exit-without-floor on identical paths."""

    floor_off_spec = FloorSpec(
        initial_fraction=0.0,
        canary_initial_fraction=0.0,
        minimum_age_seconds=10**9,
        profit_activation_fraction=floor_spec.profit_activation_fraction,
        minimum_breathing_points=floor_spec.minimum_breathing_points,
        spread_breathing_multiple=floor_spec.spread_breathing_multiple,
        peak_breathing_fraction=floor_spec.peak_breathing_fraction,
    )
    rows: list[dict[str, Any]] = []
    for intent in intents:
        trajectory = tensor[
            (tensor["session"] == intent["session"])
            & (tensor["decision_time_ns"] == intent["decision_time_ns"])
            & (tensor["contract_id"] == intent["contract_id"])
        ].copy()
        floor_on, _ = apply_lifecycle_policy(
            trajectory,
            threshold=threshold,
            route="full_policy",
            floor_spec=floor_spec,
        )
        floor_off, _ = apply_lifecycle_policy(
            trajectory,
            threshold=threshold,
            route="full_policy",
            floor_spec=floor_off_spec,
        )
        peak_available = max(
            0.0,
            (float(trajectory["current_bid"].max()) - float(intent["entry_ask"])) * 100.0 - 3.0,
        )
        floor_on_harvest = (
            None if peak_available <= 0.0 else float(floor_on["pnl_after_fee"]) / peak_available
        )
        floor_off_harvest = (
            None if peak_available <= 0.0 else float(floor_off["pnl_after_fee"]) / peak_available
        )
        rows.append(
            {
                "session": intent["session"],
                "split": intent["split"],
                "contract_id": intent["contract_id"],
                "decision_time_ns": intent["decision_time_ns"],
                "peak_available_pnl_dollars": peak_available,
                "floor_on_trigger": floor_on["trigger"],
                "floor_on_pnl_dollars": floor_on["pnl_after_fee"],
                "floor_on_harvest_ratio": floor_on_harvest,
                "floor_off_trigger": floor_off["trigger"],
                "floor_off_pnl_dollars": floor_off["pnl_after_fee"],
                "floor_off_harvest_ratio": floor_off_harvest,
                "quarantine_labels": list(QUARANTINE_LABELS),
            }
        )
    replay_rows = [row for row in rows if row["split"] == "plumbing_replay"]

    def summarize(selected: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        harvest = [
            float(row[f"{prefix}_harvest_ratio"])
            for row in selected
            if row[f"{prefix}_harvest_ratio"] is not None
        ]
        return {
            "count": len(selected),
            "total_pnl_dollars": float(sum(row[f"{prefix}_pnl_dollars"] for row in selected)),
            "positive_peak_harvest_ratio_count": len(harvest),
            "positive_peak_harvest_ratio_coverage": len(harvest) / max(1, len(selected)),
            "mean_positive_peak_harvest_ratio": float(np.mean(harvest)) if harvest else None,
            "median_positive_peak_harvest_ratio": float(np.median(harvest)) if harvest else None,
            "terminal_trigger_counts": dict(Counter(row[f"{prefix}_trigger"] for row in selected)),
        }

    return {
        "schema_version": "Protocol101WalkingSkeletonD52FloorAblationV1",
        "comparison": "full_policy_floor_on_vs_identical_learned_exit_without_floor",
        "same_frozen_entries": True,
        "same_model_and_threshold": True,
        "floor_off_implementation": "minimum floor age moved beyond the session; model and forced-flat unchanged",
        "all_13_sessions": {
            "floor_on": summarize(rows, "floor_on"),
            "floor_off": summarize(rows, "floor_off"),
        },
        "owned_validation_slice": {
            "floor_on": summarize(replay_rows, "floor_on"),
            "floor_off": summarize(replay_rows, "floor_off"),
        },
        "harvest_tripwire_disposition": (
            "REPORT_ONLY_NO_THRESHOLD_PREREGISTERED; no owner-decision inference and no floor tuning "
            "is permitted from this canary-shaped plumbing packet"
        ),
        "formal_model_quality_or_alpha_claim": False,
        "rows": rows,
        "quarantine_labels": list(QUARANTINE_LABELS),
    }


def _serial_candidate(item: dict[str, Any]) -> SerialCandidateV5:
    pnl = float(item["pnl_after_fee"])
    if item["trigger"] == "forced_flat":
        reason = ExitReason.FORCED_FLAT
    elif pnl < 0.0:
        reason = ExitReason.STOP_LOSS
    else:
        reason = ExitReason.TAKE_PROFIT
    deadline = pd.Timestamp(str(item["session"]), tz="America/New_York").replace(
        hour=15, minute=55
    ).tz_convert("UTC")
    exit_ns = int(item["exit_time_ns"])
    metadata = {
        "terminal_trigger": item["trigger"],
        "lifecycle_route": item["lifecycle_route"],
        "forced_buy_reason": item["forced_reason"],
        "floor_slippage_points": item.get("floor_slippage_points"),
        "market_quote_age_ms": item["exit_market_quote_age_ms"],
        "quarantine_labels": list(QUARANTINE_LABELS),
        "formal_alpha_claim": False,
    }
    return SerialCandidateV5(
        split="plumbing_replay",
        fold="walking_skeleton_owned_validation",
        session=str(item["session"]),
        decision_time_ns=int(item["decision_time_ns"]),
        contract_id=str(item["contract_id"]),
        right=str(item["right"]),
        canonical_strike_slot=int(item["strike_idx"]),
        policy_index=min(abs(int(item["strike_idx"]) - 10), 6),
        entry_ask=float(item["entry_ask"]),
        score=float(item.get("composer_selected_cluster_score") or 0.0),
        raw_label_pnl_after_campaign_fee=pnl,
        label_mid_pnl_before_campaign_fee=pnl + 3.0,
        label_realized_exit_time_ns=exit_ns,
        # Frozen simulator v5 defines a threshold exit's source clock as the
        # completed-second quote-state/occupancy clock and requires age zero.
        # Preserve the raw exchange-event clock and its market age separately
        # in metadata below; do not counterfeit either clock as the other.
        label_source_exit_quote_time_ns=exit_ns,
        label_exit_quote_age_ms=0.0,
        label_exit_reason_code=int(reason),
        label_executable_exit_bid=float(item["exit_bid"]),
        label_policy_deadline_ns=(exit_ns if reason == ExitReason.FORCED_FLAT else int(deadline.value)),
        feature_hash=hashlib.sha256(
            f"{item['session']}|{item['decision_time_ns']}|{item['contract_id']}".encode()
        ).hexdigest(),
        source_quote_time_ns=int(item["source_quote_time_ns"]),
        source_context_time_ns=int(item["source_context_time_ns"]),
        strategy="walking_skeleton_throwaway_paper_only_forced_buy_plumbing_harness",
        metadata=metadata,
    )


def _four_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = Counter()
    details: list[dict[str, Any]] = []
    for row in rows:
        denominator = float(row["entry_ask"]) * 100.0 + 3.0
        outcome = float(row["raw_label_pnl_after_campaign_fee"]) / denominator
        if outcome >= 0.40:
            bucket = "big_win_ge_40pct"
        elif outcome >= -0.05:
            bucket = "scratch_or_small_win_ge_minus5pct_lt40pct"
        elif outcome > -0.30:
            bucket = "small_loss_gt_minus30pct_lt_minus5pct"
        else:
            bucket = "big_loss_le_minus30pct"
        buckets[bucket] += 1
        details.append(
            {
                "session": row["session"],
                "contract_id": row["contract_id"],
                "return_on_premium_plus_fee": outcome,
                "bucket": bucket,
                "quarantine_labels": list(QUARANTINE_LABELS),
            }
        )
    total = len(rows)
    ordered = [
        "big_win_ge_40pct",
        "scratch_or_small_win_ge_minus5pct_lt40pct",
        "small_loss_gt_minus30pct_lt_minus5pct",
        "big_loss_le_minus30pct",
    ]
    return {
        "definition": {
            "big_win": "return >= 40%",
            "scratch_or_small_win": "-5% <= return < 40%",
            "small_loss": "-30% < return < -5%",
            "big_loss": "return <= -30%",
        },
        "counts": {key: int(buckets[key]) for key in ordered},
        "fractions": {key: (float(buckets[key]) / total if total else 0.0) for key in ordered},
        "trade_count": total,
        "rows": details,
        "interpretation": "descriptive plumbing evidence only; not an optimizable target or alpha claim",
        "quarantine_labels": list(QUARANTINE_LABELS),
    }


def _write_four_bucket_html(payload: dict[str, Any]) -> None:
    labels = list(payload["counts"])
    values = [payload["counts"][label] for label in labels]
    html = f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>Walking Skeleton Four-Bucket Distribution</title>
<style>body{{font-family:system-ui;margin:32px;background:#0d1117;color:#e6edf3}}.bar{{height:28px;background:#2f81f7;margin:8px 0;padding:4px}}code{{color:#ffa657}}</style></head><body>
<h1>Quarantined four-bucket Pickles distribution</h1><p>{QUARANTINE_TEXT}</p>
<p>Plumbing evidence only; no model-quality or alpha claim.</p>
{''.join(f'<div>{label}: {value}<div class="bar" style="width:{max(20, value * 55)}px"></div></div>' for label, value in zip(labels, values))}
</body></html>"""
    FOUR_BUCKET_HTML.write_text(html)


def _chart_rows(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, row in enumerate(trades, start=1):
        metadata = row["metadata"]
        decision_time = pd.Timestamp(int(row["decision_time_ns"]), unit="ns", tz="UTC")
        exit_time = pd.Timestamp(int(row["label_realized_exit_time_ns"]), unit="ns", tz="UTC")
        rows.append(
            {
                "seed": 101,
                "trade_number": number,
                "stage": "walking_skeleton_stage3",
                "segment": "owned_validation",
                "source_protocol": "Protocol101 forced-BUY plumbing harness",
                "session": row["session"],
                "decision_time": decision_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "right": row["right"],
                "side": "LONG",
                "offset": None,
                "contract_id": row["contract_id"],
                "entry_ask": row["entry_ask"],
                "exit_bid": row["label_executable_exit_bid"],
                "premium_paid": row["premium_at_risk"],
                "pnl": row["raw_label_pnl_after_campaign_fee"],
                "score": row["score"],
                "threshold": None,
                "exit_reason": metadata["terminal_trigger"],
                "candidate_uid": f"{row['session']}|{row['decision_time_ns']}|{row['contract_id']}",
                "quarantine_labels": QUARANTINE_TEXT,
            }
        )
    spx = [bar for bar in load_spx_bars(ROOT / "data/vendor/thetadata/index/spx_1m") if bar["session"] in {row["session"] for row in rows}]
    return add_equity_fields(attach_spx_prices(rows, spx)), spx


def _artifacts(paths: list[Path]) -> dict[str, Any]:
    return {
        str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256_path(path)}
        for path in paths
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data_slice, frozen_before = verify_authority_and_inputs()
    proposals = _compose_real_proposals()
    intents = _schedule_forced_intents(proposals, data_slice=data_slice)
    manifest = {
        "schema_version": "Protocol101WalkingSkeletonForcedBuyManifestV1",
        "purpose": "quarantined downstream plumbing coverage after the real A6 composer abstains",
        "real_entry_outcome": "ABSTAINS",
        "real_replay_a6_proposal_count": 552,
        "calibration_recomposition_a6_proposal_count": 2_704,
        "selection_rule": "six evenly spaced deterministic proposals per session after t+1 fill recheck, $1 floor, and reference $10k D48 mask",
        "override_boundary": "downstream confidence abstention only; no physical, A6, positive-after-fee, contract, or composer gate changed",
        "formal_model_quality_or_alpha_claim": False,
        "forced_intent_count": len(intents),
        "forced_intents": intents,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "frozen_before": frozen_before,
    }
    write_json(FORCED_MANIFEST, manifest)

    tensor = _build_tensor(intents, _raw_paths(data_slice))
    partition = data_slice["exit_cbbo_1s_slice"]["frozen_partition"]
    model, threshold, metrics = fit_lifecycle_baseline(
        tensor,
        fit_sessions=list(partition["fit_first_8_sessions"]),
        calibration_sessions=list(partition["calibration_next_2_sessions"]),
    )
    tensor = attach_exit_probabilities(tensor, model)
    tensor.to_parquet(EXIT_TENSOR, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_names": list(FEATURE_NAMES),
            "threshold": threshold,
            "formal_model_quality_or_alpha_claim": False,
            "quarantine_labels": list(QUARANTINE_LABELS),
        },
        MODEL_ARTIFACT,
    )
    write_json(
        MODEL_MANIFEST,
        {
            "schema_version": "Protocol101WalkingSkeletonLifecycleHGBV1",
            "model_artifact": str(MODEL_ARTIFACT.relative_to(ROOT)),
            "feature_names": list(FEATURE_NAMES),
            "future_label_columns_forbidden_from_features": [
                "exit_target",
                "future_max_bid_300s_label",
                "future_min_bid_300s_label",
            ],
            "features_disjoint_from_future_labels": not bool(
                set(FEATURE_NAMES)
                & {"exit_target", "future_max_bid_300s_label", "future_min_bid_300s_label"}
            ),
            "metrics": metrics,
            "quarantine_labels": list(QUARANTINE_LABELS),
        },
    )
    floor_spec = FloorSpec()
    write_json(
        FLOOR_ARTIFACT,
        {
            "schema_version": "Protocol101WalkingSkeletonProtectiveFloorV1",
            "ordering": [
                "check_floor_committed_at_t_minus_1_at_current_executable_bid",
                "evaluate_lifecycle_model_if_floor_not_crossed",
                "after_HOLD_compute_upward_only_floor_for_t_plus_1",
            ],
            "spec": asdict(floor_spec),
            "floor_never_decreases": True,
            "canary_scope": "floor_path_canary only; transparent branch-coverage aid",
            "intraminute_protection_claim": False,
            "quarantine_labels": list(QUARANTINE_LABELS),
        },
    )
    write_json(EARLY_DRAWDOWN, _early_drawdown_report(tensor, intents))
    floor_ablation = _floor_ablation(tensor, intents, threshold, floor_spec)
    write_json(FLOOR_ABLATION, floor_ablation)

    terminals, actions = _run_lifecycle(tensor, intents, threshold, floor_spec)
    actions.to_parquet(ACTION_LOG, index=False)
    write_csv(LIFECYCLE_RESULTS, terminals)

    replay_terminals = [item for item in terminals if item["split"] == "plumbing_replay"]
    accepted, rejected, safety = apply_entry_safety_prefilter(replay_terminals)
    write_json(ENTRY_SAFETY, {"summary": safety, "accepted": accepted, "rejected": rejected})
    if not safety["all_accepted_d48_pass"] or not safety["all_accepted_d49_pass"]:
        raise RuntimeError("D48/D49 prefilter invariant failed")
    if not safety["all_sessions_one_to_six"]:
        raise RuntimeError("replay did not produce 1-6 accepted trades per session")

    config = SerialSimulatorV5Config(
        starting_cash=10_000.0,
        campaign_round_trip_fee_dollars=3.0,
        affordability_reserve_per_trade=3.0,
        max_trades_per_session=6,
        no_new_entries_after_et="15:30",
        forced_flat_before_et="15:55",
        split_order=("plumbing_replay",),
    )
    candidates = [_serial_candidate(item) for item in accepted]
    trades, state = simulate_serial_candidates_v5(candidates, config=config)
    if any(state.skipped.values()):
        raise RuntimeError(f"prefiltered candidate was skipped by simulator v5: {state.skipped}")
    trade_rows = [asdict(trade) for trade in trades]
    terminal_by_identity = {
        (item["session"], int(item["decision_time_ns"]), item["contract_id"]): item
        for item in accepted
    }
    for row in trade_rows:
        terminal = terminal_by_identity[(row["session"], int(row["decision_time_ns"]), row["contract_id"])]
        row["metadata"] = {
            **dict(row.get("metadata") or {}),
            "terminal_trigger": terminal["trigger"],
            "lifecycle_route": terminal["lifecycle_route"],
            "exit_source_event_time_ns": int(terminal["exit_source_event_time_ns"]),
            "exit_market_quote_age_ms": float(terminal["exit_market_quote_age_ms"]),
            "floor_slippage_points": terminal.get("floor_slippage_points"),
            "quarantine_labels": list(QUARANTINE_LABELS),
        }
        row["quarantine_labels"] = QUARANTINE_TEXT
    trigger_counts = Counter(row["metadata"]["terminal_trigger"] for row in trade_rows)
    required = {"learned_exit", "floor_trigger", "forced_flat"}
    if not required.issubset(trigger_counts):
        raise RuntimeError(f"serial replay lacks required terminal branches: {trigger_counts}")
    if not (actions["action"] == "HOLD").any():
        raise RuntimeError("lifecycle action log lacks HOLD")
    serial_payload = {
        "schema_version": "Protocol101WalkingSkeletonSerialReplayV1",
        "simulator_version": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "simulator_semantics": state.semantics,
        "simulator_config_hash": state.simulator_config_hash,
        "candidate_stream_hash": state.candidate_stream_hash,
        "candidate_payload_hash": state.candidate_payload_hash,
        "trade_identity_hash": state.trade_identity_hash,
        "candidate_count": len(candidates),
        "trade_count": len(trades),
        "skipped": state.skipped,
        "terminal_trigger_counts": dict(trigger_counts),
        "hold_action_count": int((actions["action"] == "HOLD").sum()),
        "estimated_pnl_dollars": float(sum(row["raw_label_pnl_after_campaign_fee"] for row in trade_rows)),
        "highest_allowed_claim": (
            "Downstream plumbing was exercised end-to-end on a quarantined forced-BUY harness; "
            "the real entry composer remains frozen and still abstains."
        ),
        "formal_model_quality_or_alpha_claim": False,
        "trades": trade_rows,
        "quarantine_labels": list(QUARANTINE_LABELS),
    }
    write_json(SERIAL_REPLAY, serial_payload)
    write_csv(SERIAL_TRADES, trade_rows)

    four_bucket = _four_buckets(trade_rows)
    write_json(FOUR_BUCKET, four_bucket)
    _write_four_bucket_html(four_bucket)
    chart_rows, spx = _chart_rows(trade_rows)
    write_trades_html(
        OUT / "trades-on-spx.html",
        trades=chart_rows,
        spx=spx,
        starting_equity=10_000.0,
        skipped_trades=[],
    )
    write_trades_html(
        OUT / "trades.html",
        trades=chart_rows,
        spx=spx,
        starting_equity=10_000.0,
        skipped_trades=[],
    )
    write_equity_html(
        OUT / "equity.html",
        trades=chart_rows,
        starting_equity=10_000.0,
        skipped_trades=[],
        stress_per_side=0.0,
        chart_title="Quarantined Walking-Skeleton Forced-BUY Equity",
        subtitle=f"{QUARANTINE_TEXT} — plumbing evidence only; no alpha claim",
    )

    frozen_after = _hash_map(_frozen_paths())
    if frozen_before != frozen_after:
        raise RuntimeError("real composer, gate, contract, or simulator bytes changed during run")
    manifest["frozen_after"] = frozen_after
    manifest["frozen_bytes_unchanged"] = True
    write_json(FORCED_MANIFEST, manifest)

    report_text = f"""# Protocol101 Walking-Skeleton Stages 2–3

Labels: `{QUARANTINE_TEXT}`

## Outcome

The quarantined forced-BUY wrapper exercised the downstream path end to end. The real Stage-1 entry composer is byte-frozen and still abstains. This packet makes no model-quality, promotion, paper-readiness, or alpha claim.

- Real replay proposals that passed the A6 entry gate: 552.
- Deterministic forced intents: {len(intents)} (six per each of 13 sessions).
- Official 1-second lifecycle tensor rows: {len(tensor):,}.
- Serial replay trades: {len(trades)} across the three owned validation sessions.
- Estimated replay P&L after $3/trade fees: ${serial_payload['estimated_pnl_dollars']:,.2f} (descriptive plumbing output only).
- Terminal mix: {dict(trigger_counts)}.
- HOLD decisions logged: {serial_payload['hold_action_count']:,}.
- D48 and D49: all accepted entries passed; each replay session executed 1–6 trades.
- D52 comparator: identical learned-exit-without-floor replay is recorded in `floor_ablation_report.json`; no harvest threshold was preregistered, so it is report-only and did not tune the floor.

## Causality, floor ordering, and quote clocks

The HGB runtime feature list excludes the future 300-second label fields. At every completed second the policy checks the floor committed at the prior second, evaluates the learned HOLD/EXIT score only if that floor has not crossed, and commits an upward-only next floor only after HOLD. Exit actions fill from the next available official 1-second executable bid; the 15:55 path is flat at the deadline. Floor-slippage and source-event quote age are retained.

Frozen simulator v5 requires a threshold exit's canonical quote-state clock to equal its completed-second occupancy clock and therefore records `label_exit_quote_age_ms=0`. The raw exchange-event timestamp and its actual market age are retained separately in each trade's metadata and checked against the lifecycle terminal. This adapter is adequate for plumbing reconstruction only; native raw-event-age support remains a real-campaign simulator/parity requirement.

## Evidence limits

The three first-trade path canaries are explicit branch-coverage aids: learned-exit on 2025-03-06, floor on 2025-03-07, and forced-flat on 2025-03-10. They make the charts and P&L unsuitable for an alpha claim. D58 early-drawdown comparison is report-only. No broker/API, download, protected/outer/holdout resource, runtime/default, or Stage 4 path was touched.
"""
    REPORT.write_text(report_text)

    artifacts = _artifacts(
        [
            FORCED_MANIFEST,
            EXIT_TENSOR,
            MODEL_ARTIFACT,
            MODEL_MANIFEST,
            FLOOR_ARTIFACT,
            FLOOR_ABLATION,
            ACTION_LOG,
            LIFECYCLE_RESULTS,
            ENTRY_SAFETY,
            EARLY_DRAWDOWN,
            SERIAL_REPLAY,
            SERIAL_TRADES,
            FOUR_BUCKET,
            FOUR_BUCKET_HTML,
            OUT / "equity.html",
            OUT / "trades-on-spx.html",
            OUT / "trades.html",
            REPORT,
        ]
    )
    receipt = {
        "schema_version": "Protocol101WalkingSkeletonStages2And3ReceiptV1",
        "status": "COMPLETE_STOP_BEFORE_STAGE4",
        "authority_path": str(AUTHORITY.relative_to(ROOT)),
        "authority_sha256": AUTHORITY_SHA256,
        "graph_path": str(GRAPH.relative_to(ROOT)),
        "graph_sha256": GRAPH_SHA256,
        "python": str(Path.home() / ".autoresearch-trading/runtime-venv/bin/python"),
        "stage1_real_entry_outcome": "ABSTAINS",
        "frozen_bytes_unchanged": True,
        "official_cbbo_1s_session_count": 13,
        "raw_tick_inputs_used": False,
        "broker_contacted": False,
        "download_or_purchase_performed": False,
        "protected_outer_holdout_or_sealed_data_used": False,
        "runtime_promotion_or_default_changed": False,
        "stage4_started": False,
        "forced_intent_count": len(intents),
        "exit_tensor_rows": len(tensor),
        "lifecycle_terminal_counts_all_13": dict(Counter(item["trigger"] for item in terminals)),
        "serial_replay_terminal_counts": dict(trigger_counts),
        "serial_replay_trade_count": len(trades),
        "serial_replay_estimated_pnl_dollars": serial_payload["estimated_pnl_dollars"],
        "d48_d49_all_accepted_pass": bool(
            safety["all_accepted_d48_pass"] and safety["all_accepted_d49_pass"]
        ),
        "all_replay_sessions_one_to_six_trades": safety["all_sessions_one_to_six"],
        "all_action_paths_fired": bool(
            required.issubset(trigger_counts) and (actions["action"] == "HOLD").any()
        ),
        "d52_floor_ablation_produced": True,
        "visuals_produced": [
            "equity.html",
            "trades-on-spx.html",
            "four_bucket_distribution.html",
        ],
        "highest_allowed_claim": serial_payload["highest_allowed_claim"],
        "next_action": "STOP_FOR_CLAUDE_VERIFICATION",
        "artifacts": artifacts,
        "quarantine_labels": list(QUARANTINE_LABELS),
    }
    write_json(RECEIPT, receipt)
    print(json.dumps({key: receipt[key] for key in (
        "status", "forced_intent_count", "exit_tensor_rows", "serial_replay_trade_count",
        "serial_replay_estimated_pnl_dollars", "serial_replay_terminal_counts", "next_action"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
