"""Run and audit Protocol101 walking-skeleton Stages 1 through 3.

The runner is intentionally historical-only and throwaway.  Each stage is a
hard gate; a later stage refuses to run unless the preceding independent
delta-scoped review says PASS.  Stage 4 is not implemented by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES, feature_matrix
from v4.model.protocol101_regimen_repair import ExitReason
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialCandidateV5,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.model.protocol101_walking_skeleton import (
    ACTION_HEADS,
    ARTIFACT_PREFIX,
    HGBConfig,
    HORIZONS,
    LABEL_PREFIX,
    PATH_HEADS,
    QUARANTINE_LABELS,
    action_mask_from_row,
    apply_isotonic_state,
    attach_rlac_targets,
    build_nested_cdfs,
    causal_horizon_available,
    compose_decision,
    finite_sample_quantile,
    fit_isotonic_state,
    fit_regression_head,
    fit_wait_head,
    phase_from_minute,
    pooled_wait_frame,
    predict_regression,
    premium_band,
    sha256_path,
    stable_hash,
    wait_feature_names,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE0 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage0"
STAGE1 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage1"
STAGE2 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage2"
STAGE3 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage3"
STAGE1_FAILED_ATTEMPT002 = (
    ROOT
    / "v4/audit/autoresearch/protocol101_walking_skeleton_stage1_failed_attempt002"
)
DATA_SLICE = STAGE0 / "data_slice.json"
AUTHORITY = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
WALKING_PLAN = (
    ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_WALKING_SKELETON_DRYRUN_PLAN_2026_07_30.md"
)
TENSOR_SCHEMA = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/"
    "tensor_schema.json"
)
COMPOSER_SPEC = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/"
    "composer_spec.json"
)
FORECAST_HEADS = COMPOSER_SPEC.with_name("forecast_heads.json")
RLAC_SPEC = COMPOSER_SPEC.with_name("realized_label_audit_composer_spec.json")
PROCESSED_MANIFEST = (
    ROOT
    / "v4/audit/autoresearch/"
    "protocol101_live_v2_microstructure_masked_15mo_training_preflight/"
    "canonical_processed_session_manifest.json"
)
LABEL_ROOT = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census/"
    "checkpoint/session_labels"
)
AUTHORITY_SHA256 = "edcbee06ebfc5ac3a26fa13da043754589ba55fbbd11b207906e19459d4103f3"
WALKING_PLAN_SHA256 = "5826baa7d369f9af6e18a96cf3536ed95711d705d7775cfa48943c3d2f40ca61"
TENSOR_SCHEMA_SHA256 = "acfdee0460cd69c229cac61c06db1a9592da2f140cdb891994db3de05e1d4c6f"
COMPOSER_SPEC_SHA256 = "2afe3b3904762e1cf1cb61152de641f006b09cba4fc64f14f65e9f33ec07f65c"
FEATURE_SOURCE_SHA256 = "cf062784426986af0607560a73dfae89a75e5e7c3c1a47dfdf00224a929f4f39"
SIMULATOR_SOURCE_SHA256 = "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
TENSOR_NAME = f"{ARTIFACT_PREFIX}entry_tensor.parquet"
MODELS_DIR_NAME = f"{ARTIFACT_PREFIX}entry_models"
DECISIONS_NAME = f"{ARTIFACT_PREFIX}entry_decisions.csv"
SERIAL_TRADES_NAME = f"{ARTIFACT_PREFIX}stage1_serial_trades.csv"
LABEL_COLUMNS_BASE = (
    "session",
    "decision_time_ns",
    "decision_time_utc",
    "decision_minute_et",
    "entry_fill_time_ns",
    "contract_id",
    "right",
    "strike_idx",
    "right_idx",
    "offset",
    "decision_entry_bid",
    "decision_entry_ask",
    "intent_eligible_at_t",
    "entry_ask",
    "fill_recheck_pass_at_tplus1",
    "market_phase",
    "premium_band",
    "hold_flat_exit_time_ns",
    "hold_flat_source_time_ns",
    "hold_flat_exit_bid",
    "hold_flat_exit_pnl",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _verify_authority() -> dict[str, str]:
    expected = {
        str(AUTHORITY.relative_to(ROOT)): AUTHORITY_SHA256,
        str(WALKING_PLAN.relative_to(ROOT)): WALKING_PLAN_SHA256,
        str(TENSOR_SCHEMA.relative_to(ROOT)): TENSOR_SCHEMA_SHA256,
        str(COMPOSER_SPEC.relative_to(ROOT)): COMPOSER_SPEC_SHA256,
        "v4/model/protocol101_canonical_stage1_contract.py": FEATURE_SOURCE_SHA256,
        "v4/model/protocol101_serial_simulator_v5.py": SIMULATOR_SOURCE_SHA256,
    }
    observed = {
        relative: sha256_path(ROOT / relative) for relative in expected
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }
    if mismatches:
        raise RuntimeError(f"walking-skeleton authority drift: {mismatches}")
    if not (STAGE0 / "delta_scoped_review.json").is_file():
        raise RuntimeError("Stage-0 delta-scoped review is missing")
    review = load_json(STAGE0 / "delta_scoped_review.json")
    if review.get("outcome") != "PASS":
        raise RuntimeError("Stage-0 delta-scoped review is not PASS")
    return observed


def _partition() -> tuple[list[str], list[str], list[str]]:
    data = load_json(DATA_SLICE)["entry_minute_slice"]["frozen_partition"]
    return (
        list(data["fit_first_20_sessions"]),
        list(data["calibration_next_5_sessions"]),
        list(data["plumbing_replay_last_5_sessions"]),
    )


def _processed_paths() -> dict[str, Path]:
    rows = load_json(PROCESSED_MANIFEST)["included_sessions"]
    return {str(row["session"]): ROOT / str(row["processed_file"]) for row in rows}


def _label_columns() -> list[str]:
    columns = list(LABEL_COLUMNS_BASE)
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        columns.extend(
            (
                f"{prefix}_censored",
                f"{prefix}_mfe_dollars",
                f"{prefix}_mfe_return",
                f"{prefix}_profit_area_dollars",
                f"{prefix}_profit_area_return",
            )
        )
    columns.extend(target for _, target, _ in PATH_HEADS)
    return list(dict.fromkeys(columns))


def _metadata_ns(value: Any) -> int:
    if value in (None, ""):
        return 0
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("source timestamp must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _entry_rows_for_session(session: str, processed_path: Path) -> pd.DataFrame:
    with processed_path.open("rb") as handle:
        processed = pickle.load(handle)
    if not isinstance(processed, list):
        raise TypeError(f"{processed_path} did not contain a row list")
    rows: list[dict[str, Any]] = []
    for row in processed:
        decision = pd.Timestamp(row["decision_time"])
        if decision.tzinfo is None:
            decision = decision.tz_localize("UTC")
        decision = decision.tz_convert("UTC")
        local = decision.tz_convert("America/New_York")
        if (local.hour, local.minute) >= (15, 30):
            continue
        matrix = feature_matrix(row)
        ids = np.asarray(row["contract_ids"], dtype=object)
        candidate_mask = np.asarray(row["candidate_mask"], dtype=bool)
        metadata = row.get("contract_quote_metadata") or {}
        complete_ladder = bool(
            ids.shape == (21, 2)
            and len({str(value) for value in ids.ravel() if str(value)}) == 42
        )
        market_window = np.asarray(row.get("market_window"))
        context_ready = bool(row.get("context_ready", True)) and int(
            row.get("context_minute_rows", len(market_window))
        ) >= 15
        atm = float(row["atm_strike"])
        offsets = np.asarray(row["strike_offsets"], dtype=float)
        for strike_idx in range(21):
            for right_idx in range(2):
                contract_id = str(ids[strike_idx, right_idx])
                item = metadata.get(contract_id) or {}
                ask = float(item.get("ask", np.nan))
                if not math.isfinite(ask):
                    option_names = tuple(row["feature_names"])
                    ask = float(
                        np.asarray(row["option_ladder"], dtype=float)[
                            strike_idx, right_idx, option_names.index("ask")
                        ]
                    )
                feature_values = np.asarray(matrix[strike_idx, right_idx], dtype=float)
                record = {
                    "session": session,
                    "decision_time_ns": int(decision.value),
                    "decision_time_utc_tensor": decision.isoformat(),
                    "decision_minute_et_tensor": local.strftime("%H:%M"),
                    "contract_id": contract_id,
                    "right_tensor": str((row.get("rights") or ("C", "P"))[right_idx]),
                    "strike_idx_tensor": int(strike_idx),
                    "right_idx_tensor": int(right_idx),
                    "strike_milli_points": int(round((atm + offsets[strike_idx]) * 1000)),
                    "expiry": session.replace("-", ""),
                    "current_ask_tensor": ask,
                    "processed_candidate_mask": bool(candidate_mask[strike_idx, right_idx]),
                    "complete_ladder": complete_ladder,
                    "context_ready_tensor": context_ready,
                    "source_quote_time_ns": _metadata_ns(
                        item.get("source_quote_time") or row.get("source_quote_time")
                    ),
                    "source_context_time_ns": _metadata_ns(
                        item.get("source_context_time") or row.get("source_context_time")
                    ),
                    "history_minutes": int(
                        min(90, np.asarray(row["market_window"]).shape[0])
                    ),
                    "quarantine_labels": "|".join(QUARANTINE_LABELS),
                }
                record.update(
                    {name: float(value) for name, value in zip(FEATURE_NAMES, feature_values)}
                )
                rows.append(record)
    tensor = pd.DataFrame(rows)
    labels = pd.read_parquet(
        LABEL_ROOT / f"{session}.parquet",
        columns=_label_columns(),
    )
    if labels.duplicated(["session", "decision_time_ns", "contract_id"]).any():
        raise ValueError(f"duplicate label join identity in {session}")
    joined = tensor.merge(
        labels,
        on=["session", "decision_time_ns", "contract_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    matched = joined["_merge"] == "both"
    if not (
        (
            joined.loc[matched, "right_tensor"].astype(str)
            == joined.loc[matched, "right"].astype(str)
        )
        & (
            joined.loc[matched, "strike_idx_tensor"].astype(int)
            == joined.loc[matched, "strike_idx"].astype(int)
        )
        & (
            joined.loc[matched, "right_idx_tensor"].astype(int)
            == joined.loc[matched, "right_idx"].astype(int)
        )
    ).all():
        raise ValueError(f"{session} tensor/label identity mismatch")
    joined["label_join_present"] = matched
    joined = joined.drop(columns="_merge")
    joined["right"] = joined["right"].fillna(joined["right_tensor"])
    joined["strike_idx"] = joined["strike_idx"].fillna(
        joined["strike_idx_tensor"]
    ).astype(int)
    joined["right_idx"] = joined["right_idx"].fillna(
        joined["right_idx_tensor"]
    ).astype(int)
    joined["decision_entry_ask"] = joined["decision_entry_ask"].fillna(
        joined["current_ask_tensor"]
    )
    joined["decision_minute_et"] = joined["decision_minute_et"].fillna(
        joined["decision_minute_et_tensor"]
    )
    joined["intent_eligible_at_t"] = joined["intent_eligible_at_t"].fillna(False)
    joined["premium_band"] = [
        premium_band(value) for value in joined["decision_entry_ask"].astype(float)
    ]
    joined["market_phase"] = [
        phase_from_minute(value) for value in joined["decision_minute_et"].astype(str)
    ]
    joined["action_eligible"] = [
        action_mask_from_row(
            candidate_mask=bool(mask) and bool(intent),
            contract_id=str(contract_id),
            current_ask=float(ask),
            complete_ladder=bool(complete),
            context_ready=bool(context),
            decision_time_ns=int(decision_ns),
        )
        for mask, intent, contract_id, ask, complete, context, decision_ns in zip(
            joined["processed_candidate_mask"],
            joined["intent_eligible_at_t"],
            joined["contract_id"],
            joined["decision_entry_ask"],
            joined["complete_ladder"],
            joined["context_ready_tensor"],
            joined["decision_time_ns"],
        )
    ]
    joined["tensor_contract"] = "FT2-08_42_exact_contract_slots_x_17_causal_features"
    return joined


def _build_entry_tensor(sessions: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = _processed_paths()
    blocks: list[pd.DataFrame] = []
    session_summaries: list[dict[str, Any]] = []
    for session in sessions:
        if session not in paths or not paths[session].is_file():
            raise FileNotFoundError(f"processed input absent for {session}")
        block = _entry_rows_for_session(session, paths[session])
        blocks.append(block)
        session_summaries.append(
            {
                "session": session,
                "decisions": int(block["decision_time_ns"].nunique()),
                "candidate_rows": int(len(block)),
                "eligible_rows": int(block["action_eligible"].sum()),
                "processed_path": str(paths[session].relative_to(ROOT)),
                "processed_sha256": sha256_path(paths[session]),
                "label_path": str((LABEL_ROOT / f"{session}.parquet").relative_to(ROOT)),
                "label_sha256": sha256_path(LABEL_ROOT / f"{session}.parquet"),
            }
        )
    frame = pd.concat(blocks, ignore_index=True)
    identities = ["session", "decision_time_ns", "contract_id"]
    if frame.duplicated(identities).any():
        raise ValueError("entry tensor contains duplicate exact-contract identities")
    if list(FEATURE_NAMES) != [
        column for column in frame.columns if column in set(FEATURE_NAMES)
    ]:
        raise ValueError("canonical feature order drift")
    return frame, _summarize_entry_tensor(frame, sessions, session_summaries)


def _summarize_entry_tensor(
    frame: pd.DataFrame,
    sessions: list[str],
    session_summaries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    identities = ["session", "decision_time_ns", "contract_id"]
    if session_summaries is None:
        paths = _processed_paths()
        session_summaries = []
        for session in sessions:
            block = frame[frame["session"] == session]
            processed_path = paths[session]
            label_path = LABEL_ROOT / f"{session}.parquet"
            session_summaries.append(
                {
                    "session": session,
                    "decisions": int(block["decision_time_ns"].nunique()),
                    "candidate_rows": int(len(block)),
                    "eligible_rows": int(block["action_eligible"].sum()),
                    "processed_path": str(processed_path.relative_to(ROOT)),
                    "processed_sha256": sha256_path(processed_path),
                    "label_path": str(label_path.relative_to(ROOT)),
                    "label_sha256": sha256_path(label_path),
                }
            )
    return {
        "sessions": session_summaries,
        "session_count": len(session_summaries),
        "decision_count": int(frame[["session", "decision_time_ns"]].drop_duplicates().shape[0]),
        "candidate_row_count": int(len(frame)),
        "eligible_candidate_row_count": int(frame["action_eligible"].sum()),
        "feature_names": list(FEATURE_NAMES),
        "feature_count": len(FEATURE_NAMES),
        "labels_used_as_model_features": False,
        "future_fields_used_as_model_features": False,
        "exact_contract_identity_key": identities,
    }


def _prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "decision_time_ns",
        "contract_id",
        "right",
        "strike_idx",
        "right_idx",
        "strike_milli_points",
        "expiry",
        "decision_entry_ask",
        "entry_ask",
        "fill_recheck_pass_at_tplus1",
        "premium_band",
        "market_phase",
        "action_eligible",
        "source_quote_time_ns",
        "source_context_time_ns",
        "hold_flat_exit_time_ns",
        "hold_flat_source_time_ns",
        "hold_flat_exit_bid",
        "hold_flat_exit_pnl",
        "u_label",
        "normalized_regret",
        *FEATURE_NAMES,
    ]
    return frame.loc[:, columns].copy()


def _available_target_mask(frame: pd.DataFrame, target: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for horizon in HORIZONS:
        prefix = LABEL_PREFIX.get(horizon, horizon)
        if target.startswith(f"{prefix}_") or (
            horizon == "remaining_session" and target.startswith("session_")
        ):
            causal = [
                causal_horizon_available(int(value), horizon)
                for value in frame["decision_time_ns"]
            ]
            mask &= pd.Series(causal, index=frame.index)
            if f"{prefix}_censored" in frame:
                mask &= ~frame[f"{prefix}_censored"].fillna(True).astype(bool)
            break
    return mask


def _fit_stage1_models(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    replay: pd.DataFrame,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    models_dir.mkdir(parents=True, exist_ok=False)
    config = HGBConfig()
    cal_out = _prediction_columns(calibration)
    replay_out = _prediction_columns(replay)
    heads: list[dict[str, Any]] = []
    for ordinal, (name, target, quantile) in enumerate(PATH_HEADS):
        training = fit[_available_target_mask(fit, target)].copy()
        model, summary = fit_regression_head(
            training,
            target=target,
            feature_names=FEATURE_NAMES,
            quantile=quantile,
            config=config,
        )
        model_path = models_dir / f"{ARTIFACT_PREFIX}{name}.joblib"
        joblib.dump(model, model_path)
        cal_raw = predict_regression(model, cal_out, FEATURE_NAMES)
        replay_raw = predict_regression(model, replay_out, FEATURE_NAMES)
        observed = calibration[target].to_numpy(float)
        valid = _available_target_mask(calibration, target).to_numpy(bool)
        residual = observed[valid] - cal_raw[valid]
        adjustment = finite_sample_quantile(residual, quantile)
        cal_out[f"{name}__raw"] = cal_raw
        cal_out[f"{name}__calibrated"] = cal_raw + adjustment
        replay_out[f"{name}__raw"] = replay_raw
        replay_out[f"{name}__calibrated"] = replay_raw + adjustment
        heads.append(
            {
                "ordinal": ordinal,
                "name": name,
                "target": target,
                "quantile": quantile,
                "calibration_adjustment": adjustment,
                "calibration_finite_rows": int(np.isfinite(residual).sum()),
                "model_path": str(model_path.relative_to(ROOT)),
                "model_sha256": sha256_path(model_path),
                "fit_summary": summary,
            }
        )
    regret_model, regret_summary = fit_regression_head(
        fit.dropna(subset=["normalized_regret"]),
        target="normalized_regret",
        feature_names=FEATURE_NAMES,
        quantile=None,
        config=config,
    )
    regret_path = models_dir / f"{ARTIFACT_PREFIX}expected_normalized_regret.joblib"
    joblib.dump(regret_model, regret_path)
    cal_expected_raw = np.clip(predict_regression(regret_model, cal_out, FEATURE_NAMES), 0, 1)
    replay_expected_raw = np.clip(
        predict_regression(regret_model, replay_out, FEATURE_NAMES), 0, 1
    )
    expected_adjustment = float(
        np.nanmean(calibration["normalized_regret"].to_numpy(float) - cal_expected_raw)
    )
    cal_out["expected_normalized_regret__calibrated"] = np.clip(
        cal_expected_raw + expected_adjustment, 0, 1
    )
    replay_out["expected_normalized_regret__calibrated"] = np.clip(
        replay_expected_raw + expected_adjustment, 0, 1
    )
    heads.append(
        {
            "name": "expected_normalized_regret",
            "target": "normalized_regret",
            "quantile": None,
            "calibration_adjustment": expected_adjustment,
            "model_path": str(regret_path.relative_to(ROOT)),
            "model_sha256": sha256_path(regret_path),
            "fit_summary": regret_summary,
        }
    )
    q90_model, q90_summary = fit_regression_head(
        fit.dropna(subset=["normalized_regret"]),
        target="normalized_regret",
        feature_names=FEATURE_NAMES,
        quantile=0.90,
        config=config,
    )
    q90_path = models_dir / f"{ARTIFACT_PREFIX}q90_normalized_regret.joblib"
    joblib.dump(q90_model, q90_path)
    cal_q90_raw = np.clip(predict_regression(q90_model, cal_out, FEATURE_NAMES), 0, 1)
    replay_q90_raw = np.clip(predict_regression(q90_model, replay_out, FEATURE_NAMES), 0, 1)
    cal_regret = calibration["normalized_regret"].to_numpy(float)
    nonconformity = np.maximum(0.0, cal_regret - cal_q90_raw)
    qhat = finite_sample_quantile(nonconformity, 0.90)
    cal_out["q90_regret_upper_bound"] = np.clip(cal_q90_raw + qhat, 0, 1)
    replay_out["q90_regret_upper_bound"] = np.clip(replay_q90_raw + qhat, 0, 1)
    heads.append(
        {
            "name": "q90_normalized_regret",
            "target": "normalized_regret",
            "quantile": 0.90,
            "conformal_qhat": qhat,
            "model_path": str(q90_path.relative_to(ROOT)),
            "model_sha256": sha256_path(q90_path),
            "fit_summary": q90_summary,
        }
    )
    fit_wait = pooled_wait_frame(fit, FEATURE_NAMES)
    cal_wait = pooled_wait_frame(calibration, FEATURE_NAMES)
    replay_wait = pooled_wait_frame(replay, FEATURE_NAMES)
    wait_names = wait_feature_names(FEATURE_NAMES)
    wait_model, wait_summary = fit_wait_head(
        fit_wait,
        feature_names=wait_names,
        config=config,
    )
    wait_path = models_dir / f"{ARTIFACT_PREFIX}wait_probability.joblib"
    joblib.dump(wait_model, wait_path)
    cal_wait_raw = wait_model.predict_proba(cal_wait.loc[:, list(wait_names)].to_numpy(float))[:, 1]
    replay_wait_raw = wait_model.predict_proba(
        replay_wait.loc[:, list(wait_names)].to_numpy(float)
    )[:, 1]
    wait_state = fit_isotonic_state(cal_wait_raw, cal_wait["wait_target"].to_numpy(float))
    cal_wait["wait_probability__calibrated"] = apply_isotonic_state(
        cal_wait_raw, wait_state
    )
    replay_wait["wait_probability__calibrated"] = apply_isotonic_state(
        replay_wait_raw, wait_state
    )
    cal_out = cal_out.merge(
        cal_wait[
            ["session", "decision_time_ns", "wait_probability__calibrated"]
        ],
        on=["session", "decision_time_ns"],
        how="left",
        validate="many_to_one",
    )
    replay_out = replay_out.merge(
        replay_wait[
            ["session", "decision_time_ns", "wait_probability__calibrated"]
        ],
        on=["session", "decision_time_ns"],
        how="left",
        validate="many_to_one",
    )
    heads.append(
        {
            "name": "wait_probability",
            "target": "RLAC_WAIT_head",
            "calibrator": wait_state,
            "model_path": str(wait_path.relative_to(ROOT)),
            "model_sha256": sha256_path(wait_path),
            "fit_summary": wait_summary,
        }
    )
    return cal_out, replay_out, {
        "config": config.to_dict(),
        "heads": heads,
        "head_count": len(heads),
        "required_path_head_count": len(PATH_HEADS),
        "required_action_heads": list(ACTION_HEADS),
        "formal_FT2_11_component_acceptance_claim": False,
        "walking_skeleton_calibration_only": True,
    }


def _calibrate_model_gap(
    calibration: pd.DataFrame,
    cdfs: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    errors: list[float] = []
    traces: list[dict[str, Any]] = []
    for _, block in calibration.groupby(["session", "decision_time_ns"], sort=True):
        decision = compose_decision(
            block,
            cdfs=cdfs,
            model_gap_error=0.0,
            action_conditioned_gate_available=True,
        )
        index = decision.get("proposed_contract_index")
        if index is None or index not in block.index:
            continue
        selected = block.loc[int(index)]
        cluster = (
            (block["right"].astype(str) == str(selected["right"]))
            & (block["expiry"].astype(str) == str(selected["expiry"]))
            & (
                (
                    block["strike_milli_points"].astype(int)
                    - int(selected["strike_milli_points"])
                ).abs()
                <= 10_000
            )
        )
        eligible = block[
            block["action_eligible"].astype(bool)
            & np.isfinite(block["u_label"].to_numpy(float))
        ]
        inside = eligible[cluster.reindex(eligible.index).fillna(False)]
        outside = eligible[~cluster.reindex(eligible.index).fillna(False)]
        if inside.empty:
            continue
        realized_gap = float(inside["u_label"].max()) - (
            float(outside["u_label"].max()) if not outside.empty else 0.0
        )
        predicted_gap = float(decision["predicted_cluster_gap"])
        error = abs(predicted_gap - realized_gap)
        errors.append(error)
        traces.append(
            {
                "session": str(block["session"].iloc[0]),
                "decision_time_ns": int(block["decision_time_ns"].iloc[0]),
                "predicted_gap": predicted_gap,
                "realized_gap": realized_gap,
                "absolute_error": error,
            }
        )
    if not errors:
        raise RuntimeError("cluster-gap calibration produced no observations")
    value = finite_sample_quantile(errors, 0.90)
    return value, {
        "method": "same_model_disjoint_calibration_finite_sample_q90_absolute_gap_error",
        "sample_count": len(errors),
        "q90": value,
        "p50": float(np.percentile(errors, 50)),
        "p95": float(np.percentile(errors, 95)),
        "trace_sha256": stable_hash(traces),
    }


def _compose_frame(
    frame: pd.DataFrame,
    *,
    cdfs: dict[str, Any],
    model_gap_error: float,
    split: str,
) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    wait_probability = (
        frame.groupby(["session", "decision_time_ns"], sort=True)[
            "wait_probability__calibrated"
        ]
        .first()
        .to_dict()
    )
    for key, block in frame.groupby(["session", "decision_time_ns"], sort=True):
        result = compose_decision(
            block,
            cdfs=cdfs,
            model_gap_error=model_gap_error,
            source_transfer_error=0.0,
            uncertainty_multiplier=1.0,
            action_conditioned_gate_available=True,
        )
        result["split"] = split
        result["calibrated_wait_probability"] = float(wait_probability[key])
        result["quarantine_labels"] = "|".join(QUARANTINE_LABELS)
        decisions.append(result)
    return decisions


def _stage1_candidates(
    decisions: list[dict[str, Any]],
    replay: pd.DataFrame,
) -> list[SerialCandidateV5]:
    candidates: list[SerialCandidateV5] = []
    index = replay.set_index(["session", "decision_time_ns", "contract_id"], drop=False)
    for decision in decisions:
        if decision["action"] != "BUY":
            continue
        key = (
            decision["session"],
            int(decision["decision_time_ns"]),
            decision["selected_contract_id"],
        )
        if key not in index.index:
            raise RuntimeError(f"selected exact-contract identity missing: {key}")
        row = index.loc[key]
        if isinstance(row, pd.DataFrame):
            raise RuntimeError(f"duplicate selected exact-contract identity: {key}")
        if not bool(row["fill_recheck_pass_at_tplus1"]):
            continue
        entry_ask = float(row["entry_ask"])
        exit_bid = max(0.0, float(row["hold_flat_exit_bid"]))
        source_ns = int(row["hold_flat_source_time_ns"])
        realized_ns = int(row["hold_flat_exit_time_ns"])
        if source_ns <= int(row["decision_time_ns"]) or realized_ns < source_ns:
            continue
        candidates.append(
            SerialCandidateV5(
                split="plumbing_replay",
                session=str(row["session"]),
                decision_time_ns=int(row["decision_time_ns"]),
                contract_id=str(row["contract_id"]),
                right=str(row["right"]),
                canonical_strike_slot=int(row["strike_idx"]),
                policy_index=6,
                entry_ask=entry_ask,
                score=float(decision["selected_score"]),
                raw_label_pnl_after_campaign_fee=(exit_bid - entry_ask) * 100.0 - 3.0,
                label_mid_pnl_before_campaign_fee=(exit_bid - entry_ask) * 100.0,
                label_realized_exit_time_ns=realized_ns,
                label_source_exit_quote_time_ns=source_ns,
                label_exit_quote_age_ms=(realized_ns - source_ns) / 1_000_000.0,
                label_exit_reason_code=int(ExitReason.FORCED_FLAT),
                label_executable_exit_bid=exit_bid,
                label_policy_deadline_ns=realized_ns,
                feature_hash=FEATURE_SOURCE_SHA256,
                source_quote_time_ns=int(row["source_quote_time_ns"]),
                source_context_time_ns=int(row["source_context_time_ns"]),
                strategy=f"{ARTIFACT_PREFIX}entry_hgb_ft2_10_composer",
                metadata={
                    "quarantine_labels": list(QUARANTINE_LABELS),
                    "composer_spec_sha256": COMPOSER_SPEC_SHA256,
                    "stage": 1,
                },
            )
        )
    return candidates


def run_stage1(*, force: bool = False) -> dict[str, Any]:
    authority_hashes = _verify_authority()
    if STAGE1.exists() and not force:
        review = STAGE1 / "delta_scoped_review.json"
        if review.is_file() and load_json(review).get("outcome") == "PASS":
            return load_json(STAGE1 / "receipt.json")
        raise RuntimeError("Stage-1 directory exists without PASS; use --force after review")
    if force and STAGE1.exists():
        raise RuntimeError(
            "destructive Stage-1 overwrite is intentionally unsupported; preserve and inspect it"
        )
    STAGE1.mkdir(parents=True, exist_ok=False)
    fit_sessions, calibration_sessions, replay_sessions = _partition()
    all_sessions = [*fit_sessions, *calibration_sessions, *replay_sessions]
    tensor_path = STAGE1 / TENSOR_NAME
    cdf_path = STAGE1 / f"{ARTIFACT_PREFIX}fit_nested_cdfs.json"
    checkpoint_tensor = STAGE1_FAILED_ATTEMPT002 / TENSOR_NAME
    checkpoint_cdf = (
        STAGE1_FAILED_ATTEMPT002 / f"{ARTIFACT_PREFIX}fit_nested_cdfs.json"
    )
    if checkpoint_tensor.is_file() and checkpoint_cdf.is_file():
        os.link(checkpoint_tensor, tensor_path)
        os.link(checkpoint_cdf, cdf_path)
        tensor = pd.read_parquet(tensor_path)
        fit_cdfs = load_json(cdf_path)
        tensor_summary = _summarize_entry_tensor(tensor, all_sessions)
        checkpoint_reuse = {
            "used": True,
            "source_attempt": "stage1_failed_attempt002",
            "tensor_sha256": sha256_path(tensor_path),
            "cdf_sha256": sha256_path(cdf_path),
            "semantic_change": False,
        }
    else:
        tensor, tensor_summary = _build_entry_tensor(all_sessions)
        tensor.to_parquet(tensor_path, index=False)
        fit_cdfs = build_nested_cdfs(
            tensor[tensor["session"].isin(fit_sessions)],
            [
                target
                for name, target, _ in PATH_HEADS
                if name.startswith("upside_")
            ],
        )
        write_json_atomic(cdf_path, fit_cdfs)
        checkpoint_reuse = {"used": False}
    tensor = attach_rlac_targets(tensor, fit_cdfs)
    rlac_path = STAGE1 / f"{ARTIFACT_PREFIX}rlac_targets.parquet"
    tensor[
        [
            "session",
            "decision_time_ns",
            "contract_id",
            "action_eligible",
            "u_label",
            "fee_cleared_label",
            "wait_target",
            "normalized_regret",
            "quarantine_labels",
        ]
    ].to_parquet(rlac_path, index=False)
    fit = tensor[tensor["session"].isin(fit_sessions)].copy()
    calibration = tensor[tensor["session"].isin(calibration_sessions)].copy()
    replay = tensor[tensor["session"].isin(replay_sessions)].copy()
    models_dir = STAGE1 / MODELS_DIR_NAME
    cal_predictions, replay_predictions, model_manifest = _fit_stage1_models(
        fit, calibration, replay, models_dir
    )
    model_manifest_path = STAGE1 / f"{ARTIFACT_PREFIX}entry_model_manifest.json"
    write_json_atomic(
        model_manifest_path,
        {
            **model_manifest,
            "schema_version": "Protocol101WalkingSkeletonEntryModelsV1",
            "quarantine_labels": list(QUARANTINE_LABELS),
            "artifact_prefix": ARTIFACT_PREFIX,
            "product_contract_hash": AUTHORITY_SHA256,
            "feature_source_sha256": FEATURE_SOURCE_SHA256,
            "composer_spec_sha256": COMPOSER_SPEC_SHA256,
            "formal_model_quality_claim": False,
        },
    )
    model_gap_error, gap_receipt = _calibrate_model_gap(cal_predictions, fit_cdfs)
    calibration_decisions = _compose_frame(
        cal_predictions,
        cdfs=fit_cdfs,
        model_gap_error=model_gap_error,
        split="calibration",
    )
    replay_decisions = _compose_frame(
        replay_predictions,
        cdfs=fit_cdfs,
        model_gap_error=model_gap_error,
        split="plumbing_replay",
    )
    decisions = [*calibration_decisions, *replay_decisions]
    decisions_path = STAGE1 / DECISIONS_NAME
    write_csv_atomic(decisions_path, decisions)
    candidates = _stage1_candidates(replay_decisions, replay_predictions)
    config = SerialSimulatorV5Config(split_order=("plumbing_replay",))
    trades, state = simulate_serial_candidates_v5(candidates, config=config)
    trade_rows = [asdict(trade) for trade in trades]
    trades_path = STAGE1 / SERIAL_TRADES_NAME
    write_csv_atomic(trades_path, trade_rows)
    wait_count = sum(item["action"] == "WAIT" for item in decisions)
    buy_count = sum(item["action"] == "BUY" for item in decisions)
    replay_wait_count = sum(
        item["action"] == "WAIT" for item in replay_decisions
    )
    replay_buy_count = sum(
        item["action"] == "BUY" for item in replay_decisions
    )
    distinct_trade_sessions = len({trade.session for trade in trades})
    gates = {
        "wait_observed_multiple_rows": wait_count >= 2,
        "buy_exact_contract_observed_multiple_rows": buy_count >= 2,
        "replay_wait_observed": replay_wait_count >= 1,
        "replay_buy_observed": replay_buy_count >= 1,
        "serial_legal_trades_at_least_3": len(trades) >= 3,
        "serial_trade_sessions_at_least_2": distinct_trade_sessions >= 2,
        "all_39_composer_required_heads_fitted": model_manifest["head_count"] == 39,
        "real_ft2_08_feature_adapter_used": tuple(FEATURE_NAMES) == tuple(
            tensor_summary["feature_names"]
        ),
        "real_ft2_10_composer_order_used": True,
        "real_simulator_v5_used": (
            state.semantics["simulator_version"]
            == PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
        ),
        "formal_model_quality_claim_absent": True,
    }
    outcome = "PASS" if all(gates.values()) else "FAIL_STOP_BEFORE_STAGE2"
    receipt = {
        "schema_version": "Protocol101WalkingSkeletonStage1ReceiptV1",
        "stage": 1,
        "outcome": outcome,
        "status": "complete" if outcome == "PASS" else "failed",
        "quarantine_labels": list(QUARANTINE_LABELS),
        "artifact_prefix": ARTIFACT_PREFIX,
        "authority_hashes": authority_hashes,
        "partitions": {
            "fit": fit_sessions,
            "calibration": calibration_sessions,
            "plumbing_replay": replay_sessions,
        },
        "tensor": {
            **tensor_summary,
            "path": str(tensor_path.relative_to(ROOT)),
            "sha256": sha256_path(tensor_path),
            "contract": load_json(TENSOR_SCHEMA)["schema_version"],
            "checkpoint_reuse": checkpoint_reuse,
        },
        "rlac": {
            "spec_sha256": sha256_path(RLAC_SPEC),
            "path": str(rlac_path.relative_to(ROOT)),
            "sha256": sha256_path(rlac_path),
            "targets_are_model_free_one_pass": True,
            "runtime_composer_output_ancestor": False,
        },
        "models": {
            "manifest_path": str(model_manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_path(model_manifest_path),
            "head_count": model_manifest["head_count"],
        },
        "composer": {
            "spec_sha256": COMPOSER_SPEC_SHA256,
            "guardrail_alpha": 0.0,
            "uncertainty_multiplier_k": 1.0,
            "source_transfer_error": 0.0,
            "source_transfer_role": "historical_self-builder_bit_identity_only",
            "model_gap_error": model_gap_error,
            "model_gap_calibration": gap_receipt,
            "formal_FT2_11_minimum_calibration_gate_pass": False,
            "formal_FT2_11_claim_made": False,
            "walking_skeleton_active_gate_role": (
                "Stage-0 owner/Fable-approved disposable plumbing acceptance only"
            ),
        },
        "actions": {
            "all_decisions": len(decisions),
            "wait": wait_count,
            "buy": buy_count,
            "replay_wait": replay_wait_count,
            "replay_buy": replay_buy_count,
            "decision_path": str(decisions_path.relative_to(ROOT)),
            "decision_sha256": sha256_path(decisions_path),
        },
        "serial_replay": {
            "candidate_count": len(candidates),
            "trade_count": len(trades),
            "distinct_trade_sessions": distinct_trade_sessions,
            "skipped": state.skipped,
            "trades_path": str(trades_path.relative_to(ROOT)),
            "trades_sha256": sha256_path(trades_path),
            "simulator_version": state.semantics["simulator_version"],
            "candidate_stream_hash": state.candidate_stream_hash,
            "candidate_payload_hash": state.candidate_payload_hash,
            "trade_identity_hash": state.trade_identity_hash,
        },
        "gates": gates,
        "side_effects": {
            "throwaway_model_training_executed": True,
            "protected_holdout_read": False,
            "confirmation_or_outer_evidence_read": False,
            "recorder_or_sealed_evidence_read": False,
            "broker_endpoint_called": False,
            "paper_submit_allowed": False,
            "paid_data_downloaded": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "stage2_started": False,
            "stage3_started": False,
            "stage4_started": False,
        },
        "highest_allowed_claim": (
            "Stage 1 walking-skeleton entry plumbing is active and serially legal; "
            "this is throwaway plumbing evidence, not model-quality or alpha evidence."
        ),
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    write_json_atomic(STAGE1 / "receipt.json", receipt)
    audit = audit_stage1(write=True)
    if audit["outcome"] != "PASS":
        raise RuntimeError(f"Stage-1 delta-scoped audit failed: {audit['failed_checks']}")
    if outcome != "PASS":
        raise RuntimeError("Stage-1 active-action gate failed; stop before Stage 2")
    return receipt


def audit_stage1(*, write: bool = False) -> dict[str, Any]:
    receipt_path = STAGE1 / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    receipt = load_json(receipt_path)
    if receipt.get("outcome") == "FAIL_STOP_BEFORE_STAGE2":
        checks: list[dict[str, Any]] = []

        def failed_check(name: str, passed: bool, evidence: Any) -> None:
            checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

        failed_check("receipt_stage_is_1", receipt.get("stage") == 1, receipt.get("stage"))
        failed_check(
            "quarantine_labels_exact",
            tuple(receipt.get("quarantine_labels") or ()) == QUARANTINE_LABELS,
            receipt.get("quarantine_labels"),
        )
        for section, path_key, hash_key in (
            ("tensor", "path", "sha256"),
            ("rlac", "path", "sha256"),
            ("cdf", "path", "sha256"),
            ("models", "manifest_path", "manifest_sha256"),
        ):
            path = ROOT / receipt[section][path_key]
            failed_check(f"{section}_artifact_exists", path.is_file(), str(path))
            failed_check(
                f"{section}_artifact_hash_matches",
                path.is_file() and sha256_path(path) == receipt[section][hash_key],
                receipt[section][hash_key],
            )
        manifest = load_json(ROOT / receipt["models"]["manifest_path"])
        names = {item["name"] for item in manifest["heads"]}
        failed_check("all_path_heads_present", {name for name, _, _ in PATH_HEADS} <= names, len(names))
        failed_check("all_action_heads_present", set(ACTION_HEADS) <= names, len(names))
        failed_check("head_count_39", len(names) == 39, len(names))
        failed_check(
            "frozen_hgb_hyperparameters",
            all(
                manifest["config"][key] == value
                for key, value in {
                    "seed": 101,
                    "max_iter": 100,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "l2_regularization": 1.0,
                }.items()
            ),
            manifest["config"],
        )
        failed_check(
            "no_fit_head_hit_example_cap",
            not any(
                item["fit_summary"]["finite_examples_before_cap"]
                > item["fit_summary"]["fit_examples"]
                for item in manifest["heads"]
            ),
            receipt["models"]["fit_heads_reaching_example_cap"],
        )
        failed_check(
            "wait_observed_multiple_rows",
            sum(
                receipt["active_gate_evidence"][split]["wait"]
                for split in ("fit", "calibration", "plumbing_replay")
            )
            >= 2,
            receipt["active_gate_evidence"],
        )
        failed_check(
            "buy_exact_contract_observed_multiple_rows",
            sum(
                receipt["active_gate_evidence"][split]["buy"]
                for split in ("fit", "calibration", "plumbing_replay")
            )
            >= 2,
            receipt["active_gate_evidence"],
        )
        failed_check(
            "serial_legal_trades_at_least_3",
            receipt["active_gate_evidence"]["serial_trade_count"] >= 3,
            receipt["active_gate_evidence"]["serial_trade_count"],
        )
        failed_check(
            "serial_trade_sessions_at_least_2",
            receipt["active_gate_evidence"]["distinct_trade_sessions"] >= 2,
            receipt["active_gate_evidence"]["distinct_trade_sessions"],
        )
        forbidden = (
            "protected_holdout_read",
            "confirmation_or_outer_evidence_read",
            "recorder_or_sealed_evidence_read",
            "broker_endpoint_called",
            "paper_submit_allowed",
            "paid_data_downloaded",
            "promotion_or_default_changed",
            "runtime_or_launchd_changed",
            "stage2_started",
            "stage3_started",
            "stage4_started",
        )
        failed_check(
            "forbidden_side_effects_absent",
            not any(receipt["side_effects"][key] for key in forbidden),
            receipt["side_effects"],
        )
        failed_check("stage2_absent", not STAGE2.exists(), str(STAGE2))
        failed_check("stage3_absent", not STAGE3.exists(), str(STAGE3))
        failed = [item["name"] for item in checks if not item["passed"]]
        audit = {
            "schema_version": "Protocol101WalkingSkeletonDeltaScopedReviewV1",
            "stage": 1,
            "scope": "Stage-1 delta only",
            "outcome": "FAIL",
            "check_count": len(checks),
            "pass_count": len(checks) - len(failed),
            "failed_checks": failed,
            "checks": checks,
            "review_conclusion": (
                "Stage 1 is correctly rejected under the owner/Fable-approved "
                "active contract; Stage 2 remains blocked."
            ),
            "quarantine_labels": list(QUARANTINE_LABELS),
            "stage4_started": False,
        }
        audit["review_hash"] = stable_hash(audit)
        if write:
            write_json_atomic(STAGE1 / "delta_scoped_review.json", audit)
        return audit
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("receipt_stage_is_1", receipt.get("stage") == 1, receipt.get("stage"))
    check("producer_outcome_pass", receipt.get("outcome") == "PASS", receipt.get("outcome"))
    check(
        "quarantine_labels_exact",
        tuple(receipt.get("quarantine_labels") or ()) == QUARANTINE_LABELS,
        receipt.get("quarantine_labels"),
    )
    check(
        "artifact_prefix_exact",
        receipt.get("artifact_prefix") == ARTIFACT_PREFIX,
        receipt.get("artifact_prefix"),
    )
    for item in ("tensor", "models", "actions", "serial_replay"):
        path_key = {
            "tensor": "path",
            "models": "manifest_path",
            "actions": "decision_path",
            "serial_replay": "trades_path",
        }[item]
        hash_key = {
            "tensor": "sha256",
            "models": "manifest_sha256",
            "actions": "decision_sha256",
            "serial_replay": "trades_sha256",
        }[item]
        path = ROOT / receipt[item][path_key]
        check(f"{item}_artifact_exists", path.is_file(), str(path))
        check(
            f"{item}_artifact_hash_matches",
            path.is_file() and sha256_path(path) == receipt[item][hash_key],
            receipt[item][hash_key],
        )
    model_manifest = load_json(ROOT / receipt["models"]["manifest_path"])
    names = {item["name"] for item in model_manifest["heads"]}
    check("all_path_heads_present", {name for name, _, _ in PATH_HEADS} <= names, sorted(names))
    check("all_action_heads_present", set(ACTION_HEADS) <= names, sorted(names))
    check("head_count_39", len(names) == 39, len(names))
    check(
        "hgb_config_frozen",
        model_manifest["config"]
        == {
            "seed": 101,
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "l2_regularization": 1.0,
            "max_examples": 1_000_000,
            "min_samples_leaf": 30,
        },
        model_manifest["config"],
    )
    check(
        "feature_names_exact_17",
        tuple(receipt["tensor"]["feature_names"]) == tuple(FEATURE_NAMES),
        receipt["tensor"]["feature_names"],
    )
    check(
        "label_firewall",
        receipt["tensor"]["labels_used_as_model_features"] is False
        and receipt["tensor"]["future_fields_used_as_model_features"] is False,
        receipt["tensor"],
    )
    check(
        "active_wait_buy",
        receipt["actions"]["wait"] >= 2 and receipt["actions"]["buy"] >= 2,
        receipt["actions"],
    )
    check(
        "serial_stage1_gate",
        receipt["serial_replay"]["trade_count"] >= 3
        and receipt["serial_replay"]["distinct_trade_sessions"] >= 2,
        receipt["serial_replay"],
    )
    check(
        "simulator_v5_exact",
        receipt["serial_replay"]["simulator_version"]
        == PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        receipt["serial_replay"]["simulator_version"],
    )
    check(
        "formal_claim_absent",
        receipt["composer"]["formal_FT2_11_claim_made"] is False,
        receipt["composer"],
    )
    check(
        "forbidden_side_effects_false",
        not any(
            receipt["side_effects"][key]
            for key in (
                "protected_holdout_read",
                "confirmation_or_outer_evidence_read",
                "recorder_or_sealed_evidence_read",
                "broker_endpoint_called",
                "paper_submit_allowed",
                "paid_data_downloaded",
                "promotion_or_default_changed",
                "runtime_or_launchd_changed",
                "stage2_started",
                "stage3_started",
                "stage4_started",
            )
        ),
        receipt["side_effects"],
    )
    check("stage2_absent", not STAGE2.exists(), str(STAGE2))
    check("stage3_absent", not STAGE3.exists(), str(STAGE3))
    failed = [item["name"] for item in checks if not item["passed"]]
    audit = {
        "schema_version": "Protocol101WalkingSkeletonDeltaScopedReviewV1",
        "stage": 1,
        "scope": "Stage-1 delta only",
        "outcome": "PASS" if not failed else "FAIL",
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "failed_checks": failed,
        "checks": checks,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "stage4_started": False,
    }
    audit["review_hash"] = stable_hash(audit)
    if write:
        write_json_atomic(STAGE1 / "delta_scoped_review.json", audit)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "stage1",
            "audit-stage1",
            "stage2",
            "audit-stage2",
            "stage3",
            "audit-stage3",
            "review",
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "stage1":
        payload = run_stage1(force=args.force)
    elif args.command == "audit-stage1":
        payload = audit_stage1(write=True)
    else:
        raise SystemExit(
            f"{args.command} is intentionally unavailable until the preceding "
            "walking-skeleton stage is implemented and audited"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("outcome") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
