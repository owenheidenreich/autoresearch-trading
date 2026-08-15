"""Evaluate the frozen compact Q(WAIT)-versus-Q(ENTER) policy.

The evaluator reads midpoint outcomes first.  If and only if the real policy's
gross midpoint P&L is positive per scored session, it opens executable bid
economics and the two frozen controls.  There is no operating point: the first
current action whose predicted value exceeds feasible WAIT is the action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256
from v5.ops.build_quoted_dataset import FEES_PER_ROUND_TRIP_USD
from v5.research.causal_day_action_advantage import HORIZON_MINUTES, LABEL_NAME
from v5.research.causal_day_action_value_attainability import (
    action_value_selector_attainability,
)
from v5.research.causal_day_action_value_selection import (
    KEYS,
    select_first_positive_advantage,
    select_outcome_blind_matched_control,
)
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_compact_interaction import ARCHITECTURE_NAME
from v5.research.causal_day_declaration_reseal import (
    assert_mechanical_reseal,
    load_pinned_source_pair,
)
from v5.research.causal_day_policy_gate import (
    REOPENED_CORPUS,
    REOPENED_RESEARCH_LAW_SHA256,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)
from v5.research.validation.replay_gate import block_bootstrap_lower_bound


DECLARATION_SCHEMA = "v5.causal-day-action-value-evaluation-declaration.v3"
FIT_SCHEMA = "v5.causal-day-action-value-fit.v1"
FEATURE_AUDIT_SCHEMA = "v5.causal-day-fit-feature-audit.v1"
RECEIPT_SCHEMA = "v5.causal-day-action-value-economics.v1"
DECLARED_FAMILY_SIZE = 649
BOOTSTRAP_SEED = 39_401_202_608_14
STARTING_EQUITY_USD = 10_000.0
MAX_PREMIUM_LOSS_SHARE = 0.05


def _verified(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"schema mismatch: {path}")
    return value


def _prediction_info(fit: dict[str, Any], *, null: bool, kind: str) -> dict[str, Any]:
    matches = [
        value
        for value in fit["predictions"]
        if bool(value["null"]) is null and str(value["kind"]) == kind
    ]
    if len(matches) != 1:
        raise RuntimeError(f"fit receipt lacks one {kind} prediction artifact for null={null}")
    info = matches[0]
    path = Path(info["path"])
    if not path.is_file() or file_sha256(path) != info["sha256"]:
        raise RuntimeError(f"prediction artifact hash mismatch: {path}")
    return info


def _select_from_fit(fit: dict[str, Any], *, null: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_info = _prediction_info(fit, null=null, kind="candidate")
    minute_info = _prediction_info(fit, null=null, kind="minute")
    candidates = pd.read_parquet(candidate_info["path"])
    minutes = pd.read_parquet(minute_info["path"])
    return select_first_positive_advantage(candidates, minutes), minutes


def attach_causal_fields(selected: pd.DataFrame, causal_population: pd.DataFrame) -> pd.DataFrame:
    """One-to-one attachment of causal matching fields, without outcomes."""

    causal_columns = [
        *KEYS,
        "entry_regime",
        "right",
        "self_delta",
        "entry_ask_usd",
        "entry_mid_usd",
    ]
    missing = sorted(set(causal_columns) - set(causal_population.columns))
    if missing:
        raise RuntimeError(f"causal action population missing: {missing}")
    causal = causal_population[causal_columns].copy()
    if causal.duplicated(list(KEYS)).any():
        raise RuntimeError("causal action key is not unique")
    value = selected.merge(causal, on=list(KEYS), how="left", validate="one_to_one")
    if len(value) != len(selected) or value[causal_columns[3:]].isna().any().any():
        raise RuntimeError("selected prediction lacks a complete causal action row")
    return value


def attach_outcome(
    selected: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    column: str,
) -> pd.DataFrame:
    """Attach exactly one frozen payoff column after action selection."""

    required = {*KEYS, column}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise RuntimeError(f"outcome table missing: {missing}")
    scoped = outcomes[[*KEYS, column]].copy()
    if scoped.duplicated(list(KEYS)).any():
        raise RuntimeError("outcome key is not unique")
    value = selected.merge(scoped, on=list(KEYS), how="left", validate="one_to_one")
    numeric = pd.to_numeric(value[column], errors="coerce")
    if len(value) != len(selected) or not np.isfinite(numeric.to_numpy(float)).all():
        raise RuntimeError(f"selected action lacks finite {column}")
    value[column] = numeric
    return value


def session_values(
    trades: pd.DataFrame,
    sessions: list[str],
    *,
    column: str,
) -> np.ndarray:
    if trades.empty:
        return np.zeros(len(sessions), dtype=float)
    if trades["session"].astype(str).duplicated().any():
        raise RuntimeError("one-trade evaluation contains duplicate session actions")
    totals = trades.assign(session=trades["session"].astype(str)).set_index("session")[column]
    values = np.asarray([float(totals.get(session, 0.0)) for session in sessions], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("session economic vector is non-finite")
    return values


def fold_evidence(
    values: np.ndarray,
    sessions: list[str],
    fold_by_session: dict[str, int],
) -> tuple[int, dict[str, float]]:
    frame = pd.DataFrame(
        {
            "session": sessions,
            "value": np.asarray(values, dtype=float),
            "fold": [fold_by_session[session] for session in sessions],
        }
    )
    means = {
        str(int(fold)): float(group["value"].mean())
        for fold, group in frame.groupby("fold", sort=True)
    }
    if set(means) != {"1", "2", "3", "4", "5"}:
        raise RuntimeError("economic population does not cover all five score folds")
    return sum(value > 0.0 for value in means.values()), means


def summarize_vector(values: np.ndarray, trades: pd.DataFrame, *, column: str) -> dict[str, Any]:
    trade_values = pd.to_numeric(trades[column], errors="raise") if len(trades) else pd.Series(dtype=float)
    return {
        "scored_sessions": int(len(values)),
        "trades": int(len(trades)),
        "abstentions": int(len(values) - len(trades)),
        "mean_usd_per_session": float(np.mean(values)),
        "mean_usd_per_trade": float(trade_values.mean()) if len(trade_values) else None,
        "median_usd_per_trade": float(trade_values.median()) if len(trade_values) else None,
        "win_rate_per_trade": float(trade_values.gt(0.0).mean()) if len(trade_values) else None,
        "worst_trade_usd": float(trade_values.min()) if len(trade_values) else None,
    }


def corrected_lower_bound(values: np.ndarray, *, seed: int) -> float:
    confidence = 1.0 - 0.05 / DECLARED_FAMILY_SIZE
    return block_bootstrap_lower_bound(values, confidence=confidence, seed=seed)


def _artifact(path: Path, rows: int) -> dict[str, Any]:
    return {"path": str(path), "rows": int(rows), "sha256": file_sha256(path)}


def run(
    *,
    evaluation_declaration_path: Path,
    fit_receipt_path: Path,
    out_root: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    if out_root.exists() or evidence_dir.exists():
        raise RuntimeError("refusing to overwrite action-value economics or evidence")
    declaration = _verified(evaluation_declaration_path, DECLARATION_SCHEMA)
    for relative, digest in declaration["implementation_hashes"].items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != digest:
            raise RuntimeError(f"evaluation implementation hash mismatch: {relative}")
    fit_declaration_path = Path(declaration["fit_declaration"]["path"])
    if file_sha256(fit_declaration_path) != declaration["fit_declaration"]["sha256"]:
        raise RuntimeError("fit declaration changed after evaluation declaration")
    fit_declaration = _verified(
        fit_declaration_path, "v5.causal-day-action-value-fit-declaration.v3"
    )
    source_fit, source_evaluation = load_pinned_source_pair()
    reseal_hash = assert_mechanical_reseal(
        source_fit,
        source_evaluation,
        fit_declaration,
        declaration,
    )
    if reseal_hash != REOPENED_RESEARCH_LAW_SHA256:
        raise RuntimeError("evaluation declaration is outside the signed research law")
    parameter_counts = computed_parameter_counts()
    parameters = parameter_counts[ARCHITECTURE_NAME]
    attainability = action_value_selector_attainability()
    if declaration["architecture"]["computed_parameters"] != parameters:
        raise RuntimeError("evaluation canonical parameter count drift")
    if declaration["architecture"]["canonical_parameter_counts"] != parameter_counts:
        raise RuntimeError("evaluation canonical architecture family drift")
    if declaration["selector_attainability"] != attainability.to_dict():
        raise RuntimeError("evaluation selector attainability proof drift")

    # The same owner-controlled boundary applies to evaluation.  No fit receipt,
    # prediction or outcome is opened before the exact architecture/label passes.
    assert_fit_permitted(
        ARCHITECTURE_NAME,
        selector_attainability=attainability,
        sessions=243,
        trainable_parameters=parameters,
        reopening=load_reopening(),
        label=LABEL_NAME,
        horizon=HORIZON_MINUTES,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )

    fit = _verified(fit_receipt_path, FIT_SCHEMA)
    if fit["declaration"]["sha256"] != file_sha256(fit_declaration_path):
        raise RuntimeError("fit receipt is not bound to the frozen fit declaration")
    if fit.get("economics_read") is not False or fit.get("operating_point_tuned") is not False:
        raise RuntimeError("fit crossed the pre-economics firewall")
    audit_path = Path(declaration["feature_audit"]["path"])
    if file_sha256(audit_path) != declaration["feature_audit"]["sha256"]:
        raise RuntimeError("feature audit changed after declaration")
    audit = _verified(audit_path, FEATURE_AUDIT_SCHEMA)
    assertions = audit.get("assertions", {})
    if audit.get("status") != "PASS_BEFORE_ECONOMICS" or not all(assertions.values()):
        raise RuntimeError("feature audit is not fully green before economics")
    action_path = Path(declaration["action_values"]["candidate_path"])
    if file_sha256(action_path) != declaration["action_values"]["candidate_sha256"]:
        raise RuntimeError("action-value candidate table changed after declaration")

    real_selected, real_minutes = _select_from_fit(fit, null=False)
    sessions = sorted(real_minutes["session"].astype(str).unique())
    if len(sessions) != 150 or not all(session < "2026-08-06" for session in sessions):
        raise RuntimeError("economic score index or reserved-session boundary drifted")
    fold_pairs = real_minutes[["session", "fold"]].drop_duplicates()
    if fold_pairs["session"].astype(str).duplicated().any():
        raise RuntimeError("one score session maps to multiple folds")
    fold_by_session = {
        str(session): int(fold) for session, fold in fold_pairs.itertuples(index=False)
    }

    causal_population = pd.read_parquet(
        action_path,
        columns=[
            *KEYS,
            "entry_regime",
            "right",
            "self_delta",
            "entry_ask_usd",
            "entry_mid_usd",
        ],
    )
    real_causal = attach_causal_fields(real_selected, causal_population)
    mid_outcomes = pd.read_parquet(action_path, columns=[*KEYS, "q_enter_mid_120m_usd"])
    real_mid = attach_outcome(real_causal, mid_outcomes, column="q_enter_mid_120m_usd")
    real_mid["gross_mid_usd"] = real_mid["q_enter_mid_120m_usd"] + FEES_PER_ROUND_TRIP_USD
    mid_values = session_values(real_mid, sessions, column="gross_mid_usd")
    mid_summary = summarize_vector(mid_values, real_mid, column="gross_mid_usd")
    mid_positive = bool(len(real_mid) and mid_summary["mean_usd_per_session"] > 0.0)

    artifact_frames: dict[str, pd.DataFrame] = {"real_selected_mid_only": real_mid}
    controls: dict[str, Any] | None = None
    executable: dict[str, Any] | None = None
    extra_kills: dict[str, Any] = {
        "executable_net_positive_corrected": None,
        "risk_compatible_with_10000_account": None,
    }

    if mid_positive:
        shuffled_selected, shuffled_minutes = _select_from_fit(fit, null=True)
        if sorted(shuffled_minutes["session"].astype(str).unique()) != sessions:
            raise RuntimeError("real and shuffled score indices differ")
        shuffled_causal = attach_causal_fields(shuffled_selected, causal_population)
        matched_causal = select_outcome_blind_matched_control(real_causal, causal_population)
        matched_causal = matched_causal.merge(
            real_causal[["session", "fold"]], on="session", how="left", validate="one_to_one"
        )
        bid_outcomes = pd.read_parquet(action_path, columns=[*KEYS, "q_enter_bid_120m_usd"])
        real_bid = attach_outcome(real_causal, bid_outcomes, column="q_enter_bid_120m_usd")
        matched_bid = attach_outcome(
            matched_causal, bid_outcomes, column="q_enter_bid_120m_usd"
        )
        shuffled_bid = attach_outcome(
            shuffled_causal, bid_outcomes, column="q_enter_bid_120m_usd"
        )
        for frame in (real_bid, matched_bid, shuffled_bid):
            frame["net_bid_usd"] = frame["q_enter_bid_120m_usd"]
        real_values = session_values(real_bid, sessions, column="net_bid_usd")
        matched_values = session_values(matched_bid, sessions, column="net_bid_usd")
        shuffled_values = session_values(shuffled_bid, sessions, column="net_bid_usd")
        delta_matched = real_values - matched_values
        delta_shuffled = real_values - shuffled_values
        absolute_lcb = corrected_lower_bound(real_values, seed=BOOTSTRAP_SEED)
        matched_lcb = corrected_lower_bound(delta_matched, seed=BOOTSTRAP_SEED + 1)
        shuffled_lcb = corrected_lower_bound(delta_shuffled, seed=BOOTSTRAP_SEED + 2)
        real_folds, real_fold_means = fold_evidence(real_values, sessions, fold_by_session)
        matched_folds, matched_fold_means = fold_evidence(
            delta_matched, sessions, fold_by_session
        )
        shuffled_folds, shuffled_fold_means = fold_evidence(
            delta_shuffled, sessions, fold_by_session
        )
        chronology = bool(real_folds >= 4 and matched_folds >= 4 and shuffled_folds >= 4)
        beats_matched = bool(float(delta_matched.mean()) > 0.0 and matched_lcb > 0.0)
        beats_shuffled = bool(float(delta_shuffled.mean()) > 0.0 and shuffled_lcb > 0.0)
        executable_positive = bool(float(real_values.mean()) > 0.0 and absolute_lcb > 0.0)
        max_ticket_loss = (
            float(real_bid["entry_ask_usd"].max()) + FEES_PER_ROUND_TRIP_USD
            if len(real_bid)
            else 0.0
        )
        worst = float(real_bid["net_bid_usd"].min()) if len(real_bid) else 0.0
        risk_pass = bool(
            max_ticket_loss <= STARTING_EQUITY_USD * MAX_PREMIUM_LOSS_SHARE
            and worst >= -STARTING_EQUITY_USD * MAX_PREMIUM_LOSS_SHARE
        )
        executable = {
            "real": summarize_vector(real_values, real_bid, column="net_bid_usd"),
            "matched": summarize_vector(matched_values, matched_bid, column="net_bid_usd"),
            "shuffled": summarize_vector(shuffled_values, shuffled_bid, column="net_bid_usd"),
            "real_minus_matched_mean_usd_per_session": float(delta_matched.mean()),
            "real_minus_shuffled_mean_usd_per_session": float(delta_shuffled.mean()),
            "corrected_moving_block_lower_bounds": {
                "family_size": DECLARED_FAMILY_SIZE,
                "real_usd_per_session": absolute_lcb,
                "real_minus_matched_usd_per_session": matched_lcb,
                "real_minus_shuffled_usd_per_session": shuffled_lcb,
                "seed": BOOTSTRAP_SEED,
            },
            "chronological_folds": {
                "real_positive": real_folds,
                "real_minus_matched_positive": matched_folds,
                "real_minus_shuffled_positive": shuffled_folds,
                "real_means": real_fold_means,
                "real_minus_matched_means": matched_fold_means,
                "real_minus_shuffled_means": shuffled_fold_means,
            },
            "risk": {
                "starting_equity_usd": STARTING_EQUITY_USD,
                "maximum_allowed_loss_usd": STARTING_EQUITY_USD * MAX_PREMIUM_LOSS_SHARE,
                "largest_selected_ticket_plus_fee_usd": max_ticket_loss,
                "worst_realised_trade_usd": worst,
                "passed": risk_pass,
            },
        }
        controls = {
            "matched_trade_count_preserved": len(matched_bid) == len(real_bid),
            "matched_max_minute_distance": (
                int(matched_bid["match_minute_distance"].max()) if len(matched_bid) else None
            ),
            "beats_matched_executable_corrected": beats_matched,
            "beats_shuffled_executable_corrected": beats_shuffled,
        }
        extra_kills = {
            "executable_net_positive_corrected": executable_positive,
            "risk_compatible_with_10000_account": risk_pass,
        }
        for name, frame in (
            ("real_selected_executable", real_bid),
            ("matched_selected_executable", matched_bid),
            ("shuffled_selected_executable", shuffled_bid),
        ):
            artifact_frames[name] = frame
    else:
        beats_matched = False
        beats_shuffled = False
        chronology = False

    kill_results = {
        "mid_to_mid_gross_positive": mid_positive,
        "beats_composition_matched_control": beats_matched if mid_positive else None,
        "beats_shuffled_label_null": beats_shuffled if mid_positive else None,
        "per_feature_timestamp_audit": bool(
            assertions.get("every_feature_has_maximum_timestamp_t")
            and assertions.get("no_future_feature")
            and assertions.get("actual_future_mutation_proof")
        ),
        "no_post_entry_slot_filter": bool(assertions.get("no_post_entry_candidate_filter")),
        "chronological_out_of_sample": chronology if mid_positive else None,
        "no_reserved_sessions": all(session < "2026-08-06" for session in sessions),
    }
    complete_pass = bool(
        all(value is True for value in kill_results.values())
        and all(value is True for value in extra_kills.values())
    )
    if not mid_positive:
        status = "NEGATIVE_STOP_MID_TO_MID_GROSS_NOT_POSITIVE"
    elif complete_pass:
        status = "TRADING_POLICY_BREAKTHROUGH_CANDIDATE_NOT_PROMOTED"
    else:
        status = "NEGATIVE_EXECUTABLE_CONTROL_CHRONOLOGY_OR_RISK_FAILED"

    # No output path exists until all outcome joins, controls, inference and
    # risk checks finish successfully.  A structural error therefore remains
    # retryable without deleting or overwriting partial evidence.
    out_root.mkdir(parents=True, exist_ok=False)
    evidence_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, Any] = {}
    for name, frame in artifact_frames.items():
        path = out_root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[name] = _artifact(path, len(frame))

    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "created_on": "2026-08-14",
        "status": status,
        "evaluation_declaration": {
            "path": str(evaluation_declaration_path),
            "sha256": file_sha256(evaluation_declaration_path),
        },
        "fit_receipt": {"path": str(fit_receipt_path), "sha256": file_sha256(fit_receipt_path)},
        "population": {
            "scored_sessions": len(sessions),
            "first_session": sessions[0],
            "last_session": sessions[-1],
        },
        "primary_midpoint_firewall": mid_summary,
        "executable_economics": executable,
        "controls": controls,
        "kill_condition_results": kill_results,
        "additional_completion_requirements": extra_kills,
        "fit_gate": {
            "status": "PERMITTED",
            "architecture": ARCHITECTURE_NAME,
            "parameters": parameters,
            "parameter_count_source": "computed_parameter_counts()['compact_interaction_entry']",
            "label": LABEL_NAME,
            "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
        },
        "selector_attainability": attainability.to_dict(),
        "mechanical_reseal": {
            "status": "PASS_ZERO_RESEARCH_LAW_DIFFERENCES",
            "research_law_sha256": reseal_hash,
        },
        "integrity": {
            "one_trade_per_session": True,
            "wait_has_structural_zero_floor": True,
            "external_threshold": False,
            "current_day_rank": False,
            "bid_economics_read": mid_positive,
            "reserved_sessions_used": False,
            "promotion_or_order": False,
        },
        "artifacts": artifacts,
        "implementation_hashes": declaration["implementation_hashes"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    receipt_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (out_root / "receipt.json").write_text(receipt_text)
    (evidence_dir / "receipt.json").write_text(receipt_text)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-declaration", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        evaluation_declaration_path=args.evaluation_declaration,
        fit_receipt_path=args.fit_receipt,
        out_root=args.out_root,
        evidence_dir=args.evidence_dir,
    )
    print(json.dumps({"status": payload["status"], "mid": payload["primary_midpoint_firewall"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
