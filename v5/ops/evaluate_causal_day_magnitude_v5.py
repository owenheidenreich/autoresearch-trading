"""Evaluate the corrected Job-39 selector with kill condition 1 first."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v5.ops.build_causal_day_dataset import canonical_json, file_sha256, minute_number
from v5.research.causal_day_architectures import computed_parameter_counts
from v5.research.causal_day_policy_gate import (
    CONSERVATIVE_EFFECTIVE_OBSERVATIONS,
    MEASURED_EFFECTIVE_OBSERVATIONS,
    REOPENED_CORPUS,
    REOPENED_LABEL,
    REQUIRED_KILL_CONDITIONS,
    assert_fit_permitted,
    load_reopening,
)
from v5.research.causal_day_selection import select_rank_clock_trades
from v5.research.knobs import frozen_value
from v5.research.validation.replay_gate import block_bootstrap_lower_bound


PRIMARY_ARCHITECTURE = "neural_four_head"
PRIMARY_HORIZON = 120
PRIMARY_TRADE_CAP = 2
PRIMARY_RISK_MODE = "ticket_only"
FEES_PER_ROUND_TRIP_USD = 3.08
LEGACY_DECLARED_FAMILY_SIZE = 648
MATCH_SEED = 39_202_608_15
BOOTSTRAP_SEED = 39_202_608_16


def _verified(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = value.get("receipt_sha256")
    unsigned = dict(value)
    unsigned.pop("receipt_sha256", None)
    if hashlib.sha256(canonical_json(unsigned)).hexdigest() != expected:
        raise RuntimeError(f"receipt self-hash mismatch: {path}")
    if value.get("schema_version") != schema:
        raise RuntimeError(f"unexpected receipt schema at {path}")
    return value


def _prediction_member(fit: dict, *, null: bool) -> dict:
    matches = [
        value
        for value in fit["predictions"]
        if value["architecture"] == PRIMARY_ARCHITECTURE
        and value["horizon_minutes"] == PRIMARY_HORIZON
        and value["shuffled_label_null"] is null
    ]
    if len(matches) != 1:
        raise RuntimeError("fit receipt does not identify one primary prediction member")
    value = matches[0]
    if file_sha256(Path(value["path"])) != value["sha256"]:
        raise RuntimeError("primary prediction artifact hash mismatch")
    return value


def _cutoffs(calibration: dict, *, null: bool) -> dict[int, float]:
    name = "shuffled" if null else "real"
    rows = calibration["selector"]["calibrations"][name]
    values = {int(row["fold"]): float(row["cutoff_points"]) for row in rows}
    if set(values) != set(range(1, 6)):
        raise RuntimeError(f"incomplete {name} cutoff calibration")
    return values


def _primary_candidate_columns() -> list[str]:
    return [
        "session",
        "entry_minute",
        "contract_id",
        "right",
        "entry_regime",
        "self_delta",
        "entry_ask_usd",
        "entry_mid_usd",
        "spread_usd",
        "moneyness_itm_points",
        "clock_exit_minute_120m",
        "clock_exit_mid_value_120m",
        "net_mid_120m_usd",
    ]


def join_mid_only(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Join only the columns needed for selection and spread-free kill #1."""

    keys = ["session", "entry_minute", "contract_id"]
    if predictions.duplicated(keys).any() or candidates.duplicated(keys).any():
        raise RuntimeError("prediction/candidate key is not one-to-one")
    required = set(_primary_candidate_columns())
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"primary candidate table is incomplete: {missing}")
    merged = predictions.merge(
        candidates[_primary_candidate_columns()],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    candidate_values = [value for value in _primary_candidate_columns() if value not in keys]
    if len(merged) != len(predictions) or merged[candidate_values].isna().any().any():
        raise RuntimeError("a scored contract has no complete mid-only candidate outcome")
    if not merged["session"].astype(str).lt("2026-08-06").all():
        raise RuntimeError("reserved session entered the economic population")
    direct_gross = (
        pd.to_numeric(merged["clock_exit_mid_value_120m"], errors="raise") * 100.0
        - pd.to_numeric(merged["entry_mid_usd"], errors="raise")
    )
    stored_gross = pd.to_numeric(merged["net_mid_120m_usd"], errors="raise") + FEES_PER_ROUND_TRIP_USD
    if not np.allclose(direct_gross, stored_gross, atol=1e-8, rtol=0.0):
        raise RuntimeError("mid-gross accounting does not reproduce from source prices")
    return merged


def summarize_trades(trades: pd.DataFrame) -> dict[str, Any]:
    trade_count = len(trades)
    fold_means = {}
    for fold in range(1, 6):
        values = trades.loc[trades["fold"].astype(int).eq(fold), "gross_mid_usd"]
        fold_means[str(fold)] = float(values.mean()) if len(values) else None
    return {
        "trades": trade_count,
        "days_traded": int(trades["session"].nunique()) if trade_count else 0,
        "mean_gross_mid_to_mid_usd_per_trade": (
            float(trades["gross_mid_usd"].mean()) if trade_count else None
        ),
        "median_gross_mid_to_mid_usd_per_trade": (
            float(trades["gross_mid_usd"].median()) if trade_count else None
        ),
        "fold_mean_gross_mid_to_mid_usd_per_trade": fold_means,
    }


def _stratum_columns(frame: pd.DataFrame, calibration: dict) -> pd.DataFrame:
    value = frame.copy()
    edges = calibration["matched_control_quintiles"]
    delta_edges = np.asarray(edges["delta_internal_edges"], dtype=float)
    premium_edges = np.asarray(edges["premium_internal_edges_usd"], dtype=float)
    value["_delta_quintile"] = np.searchsorted(
        delta_edges,
        pd.to_numeric(value["self_delta"], errors="raise").abs().to_numpy(float),
        side="right",
    )
    value["_premium_quintile"] = np.searchsorted(
        premium_edges,
        pd.to_numeric(value["entry_ask_usd"], errors="raise").to_numpy(float),
        side="right",
    )
    value["_stratum"] = list(
        zip(
            value["entry_regime"].astype(str),
            value["right"].astype(str),
            value["_delta_quintile"].astype(int),
            value["_premium_quintile"].astype(int),
            strict=True,
        )
    )
    return value


def _stable_order(frame: pd.DataFrame, *, session: str, slot: int) -> pd.DataFrame:
    value = frame.copy()
    value["_random_order"] = [
        hashlib.sha256(
            f"{MATCH_SEED}|{session}|{slot}|{minute}|{contract}".encode()
        ).hexdigest()
        for minute, contract in zip(value["entry_minute"], value["contract_id"], strict=True)
    ]
    return value.sort_values("_random_order", kind="mergesort")


def _nonoverlapping(left: pd.Series, right: pd.Series) -> bool:
    left_entry = minute_number(str(left["entry_minute"]))
    left_exit = minute_number(str(left["clock_exit_minute_120m"]))
    right_entry = minute_number(str(right["entry_minute"]))
    right_exit = minute_number(str(right["clock_exit_minute_120m"]))
    return bool(left_exit < right_entry or right_exit < left_entry)


def select_composition_matched_control(
    model_trades: pd.DataFrame,
    causal_population: pd.DataFrame,
    *,
    calibration: dict,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Match regime x side x delta quintile x premium quintile without outcomes."""

    required_columns = {
        "session",
        "entry_minute",
        "contract_id",
        "right",
        "entry_regime",
        "self_delta",
        "entry_ask_usd",
        "clock_exit_minute_120m",
    }
    if not required_columns <= set(causal_population):
        raise RuntimeError("matched-control causal population is incomplete")
    targets = _stratum_columns(model_trades, calibration)
    population = _stratum_columns(causal_population, calibration)
    selected: list[pd.Series] = []
    failures: list[dict[str, Any]] = []
    model_keys = set(
        zip(
            model_trades["session"].astype(str),
            model_trades["entry_minute"].astype(str),
            model_trades["contract_id"].astype(str),
            strict=True,
        )
    )
    population = population[
        [
            (str(row.session), str(row.entry_minute), str(row.contract_id)) not in model_keys
            for row in population.itertuples(index=False)
        ]
    ]

    for session, session_targets in targets.groupby("session", sort=True):
        session_population = population[population["session"].astype(str).eq(str(session))]
        wanted = session_targets.sort_values("entry_minute", kind="mergesort")["_stratum"].tolist()
        choices = [
            _stable_order(
                session_population[
                    session_population["_stratum"].map(lambda value: value == stratum)
                ],
                session=str(session),
                slot=slot,
            )
            for slot, stratum in enumerate(wanted)
        ]
        if any(frame.empty for frame in choices):
            failures.append({"session": str(session), "reason": "empty exact stratum"})
            continue
        assignment: list[pd.Series] | None = None
        if len(choices) == 1:
            assignment = [choices[0].iloc[0]]
        elif len(choices) == 2:
            for _, first in choices[0].iterrows():
                for _, second in choices[1].iterrows():
                    if (
                        str(first["entry_minute"]) == str(second["entry_minute"])
                        and str(first["contract_id"]) == str(second["contract_id"])
                    ):
                        continue
                    if _nonoverlapping(first, second):
                        assignment = [first, second]
                        break
                if assignment is not None:
                    break
        else:
            raise RuntimeError("matched control received more trades than the frozen cap")
        if assignment is None:
            failures.append({"session": str(session), "reason": "no nonoverlapping exact match"})
            continue
        for slot, row in enumerate(assignment):
            row = row.copy()
            row["matched_slot"] = slot
            selected.append(row)
    return pd.DataFrame(selected), failures


def _attach_control_economics(
    selected: pd.DataFrame,
    economics: pd.DataFrame,
    fold_by_session: dict[str, int],
) -> pd.DataFrame:
    keys = ["session", "entry_minute", "contract_id"]
    if selected.empty:
        return pd.DataFrame(
            columns=[*keys, "net_mid_120m_usd", "gross_mid_usd", "fold"]
        )
    value = selected[keys].merge(economics, on=keys, how="left", validate="one_to_one")
    if value["net_mid_120m_usd"].isna().any():
        raise RuntimeError("matched control has missing mid outcome")
    value["gross_mid_usd"] = (
        pd.to_numeric(value["net_mid_120m_usd"], errors="raise") + FEES_PER_ROUND_TRIP_USD
    )
    value["fold"] = value["session"].astype(str).map(fold_by_session)
    if value["fold"].isna().any():
        raise RuntimeError("matched control session has no score fold")
    return value


def _session_totals(
    trades: pd.DataFrame,
    sessions: list[str],
) -> np.ndarray:
    totals = trades.groupby("session")["gross_mid_usd"].sum()
    return np.asarray([float(totals.get(session, 0.0)) for session in sessions], dtype=float)


def _fold_positive_count(
    values: np.ndarray,
    sessions: list[str],
    fold_by_session: dict[str, int],
) -> tuple[int, dict[str, float]]:
    frame = pd.DataFrame(
        {
            "session": sessions,
            "value": values,
            "fold": [fold_by_session[value] for value in sessions],
        }
    )
    means = {
        str(fold): float(rows["value"].mean())
        for fold, rows in frame.groupby("fold", sort=True)
    }
    return sum(value > 0.0 for value in means.values()), means


def evaluate_controls(
    *,
    real_trades: pd.DataFrame,
    shuffled_scored: pd.DataFrame,
    shuffled_cutoffs: dict[int, float],
    candidates_path: Path,
    calibration: dict,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    shuffled_trades = select_rank_clock_trades(
        shuffled_scored,
        horizon=PRIMARY_HORIZON,
        fold_cutoffs=shuffled_cutoffs,
        trade_cap=PRIMARY_TRADE_CAP,
    )
    causal_columns = [
        "session",
        "entry_minute",
        "contract_id",
        "right",
        "entry_regime",
        "self_delta",
        "entry_ask_usd",
        "clock_exit_minute_120m",
    ]
    causal_population = pd.read_parquet(candidates_path, columns=causal_columns)
    matched_keys, match_failures = select_composition_matched_control(
        real_trades,
        causal_population,
        calibration=calibration,
    )
    economics = pd.read_parquet(
        candidates_path,
        columns=["session", "entry_minute", "contract_id", "net_mid_120m_usd"],
    )
    fold_by_session = {
        str(session): int(fold)
        for session, fold in real_trades[["session", "fold"]].drop_duplicates().itertuples(index=False)
    }
    matched_trades = _attach_control_economics(matched_keys, economics, fold_by_session)
    real_summary = summarize_trades(real_trades)
    matched_summary = summarize_trades(matched_trades)
    shuffled_summary = summarize_trades(shuffled_trades)
    complete_match = not match_failures and len(matched_trades) == len(real_trades)
    beats_matched = bool(
        complete_match
        and real_summary["mean_gross_mid_to_mid_usd_per_trade"]
        > matched_summary["mean_gross_mid_to_mid_usd_per_trade"]
    )
    beats_shuffled = bool(
        len(shuffled_trades)
        and real_summary["mean_gross_mid_to_mid_usd_per_trade"]
        > shuffled_summary["mean_gross_mid_to_mid_usd_per_trade"]
    )

    sessions = sorted(shuffled_scored["session"].astype(str).unique())
    fold_map = {
        str(session): int(fold)
        for session, fold in shuffled_scored[["session", "fold"]].drop_duplicates().itertuples(index=False)
    }
    real_total = _session_totals(real_trades, sessions)
    matched_total = _session_totals(matched_trades, sessions)
    shuffled_total = _session_totals(shuffled_trades, sessions)
    family_confidence = 1.0 - 0.05 / LEGACY_DECLARED_FAMILY_SIZE
    absolute_lcb = block_bootstrap_lower_bound(
        real_total,
        confidence=family_confidence,
        seed=BOOTSTRAP_SEED,
    )
    matched_lcb = block_bootstrap_lower_bound(
        real_total - matched_total,
        confidence=family_confidence,
        seed=BOOTSTRAP_SEED + 1,
    )
    shuffled_lcb = block_bootstrap_lower_bound(
        real_total - shuffled_total,
        confidence=family_confidence,
        seed=BOOTSTRAP_SEED + 2,
    )
    absolute_positive, absolute_folds = _fold_positive_count(
        real_total, sessions, fold_map
    )
    matched_positive, matched_folds = _fold_positive_count(
        real_total - matched_total, sessions, fold_map
    )
    shuffled_positive, shuffled_folds = _fold_positive_count(
        real_total - shuffled_total, sessions, fold_map
    )
    chronology_pass = bool(
        absolute_lcb > 0.0
        and matched_lcb > 0.0
        and shuffled_lcb > 0.0
        and absolute_positive >= 4
        and matched_positive >= 4
        and shuffled_positive >= 4
    )
    return (
        {
            "beats_composition_matched_control": beats_matched,
            "beats_shuffled_label_null": beats_shuffled,
            "chronological_out_of_sample": chronology_pass,
            "matched_control_complete": complete_match,
            "matched_control_failures": match_failures,
            "real": real_summary,
            "matched": matched_summary,
            "shuffled": shuffled_summary,
            "corrected_session_block_bootstrap": {
                "declared_family_size": LEGACY_DECLARED_FAMILY_SIZE,
                "per_member_one_sided_confidence": family_confidence,
                "absolute_real_lower_bound_usd_per_session": absolute_lcb,
                "real_minus_matched_lower_bound_usd_per_session": matched_lcb,
                "real_minus_shuffled_lower_bound_usd_per_session": shuffled_lcb,
                "seed": BOOTSTRAP_SEED,
            },
            "four_of_five_fold_rule": {
                "real_positive_folds": absolute_positive,
                "real_minus_matched_positive_folds": matched_positive,
                "real_minus_shuffled_positive_folds": shuffled_positive,
                "real_fold_means": absolute_folds,
                "real_minus_matched_fold_means": matched_folds,
                "real_minus_shuffled_fold_means": shuffled_folds,
            },
        },
        matched_trades,
        shuffled_trades,
    )


def run(
    *,
    declaration_path: Path,
    fit_receipt_path: Path,
    feature_audit_path: Path,
    calibration_path: Path,
    candidates_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    if out_dir.exists():
        raise RuntimeError(f"refusing to overwrite primary economics: {out_dir}")
    declaration = _verified(
        declaration_path, "v5.causal-day-trader-fit-declaration.v5"
    )
    fit = _verified(fit_receipt_path, "v5.causal-day-magnitude-fit.v1")
    audit = _verified(
        feature_audit_path, "v5.causal-day-fit-feature-audit.v1"
    )
    calibration = _verified(
        calibration_path, "v5.causal-day-rank-selector-calibration.v1"
    )
    if audit.get("status") != "PASS_BEFORE_ECONOMICS" or not all(
        audit.get("assertions", {}).values()
    ):
        raise RuntimeError("feature timestamp audit is not fully green")
    if calibration.get("status") != "PASS_BEFORE_ECONOMICS" or calibration.get("economics_read") is not False:
        raise RuntimeError("selector calibration crossed the economic firewall")
    if calibration["declaration"]["sha256"] != file_sha256(declaration_path):
        raise RuntimeError("calibration is not bound to V5")
    if calibration["candidate_table"]["sha256"] != file_sha256(candidates_path):
        raise RuntimeError("candidate table changed after rank calibration")

    counts = computed_parameter_counts()
    assert_fit_permitted(
        PRIMARY_ARCHITECTURE,
        sessions=int(declaration["source_population"]["sessions"]),
        trainable_parameters=counts[PRIMARY_ARCHITECTURE],
        reopening=load_reopening(),
        label=REOPENED_LABEL,
        horizon=PRIMARY_HORIZON,
        corpus=REOPENED_CORPUS,
        declared_kill_conditions=REQUIRED_KILL_CONDITIONS,
    )

    real_info = _prediction_member(fit, null=False)
    real_predictions = pd.read_parquet(real_info["path"])
    primary_candidates = pd.read_parquet(
        candidates_path, columns=_primary_candidate_columns()
    )
    real_scored = join_mid_only(real_predictions, primary_candidates)
    real_trades = select_rank_clock_trades(
        real_scored,
        horizon=PRIMARY_HORIZON,
        fold_cutoffs=_cutoffs(calibration, null=False),
        trade_cap=PRIMARY_TRADE_CAP,
    )
    primary = summarize_trades(real_trades)
    primary.update(
        {
            "architecture": PRIMARY_ARCHITECTURE,
            "horizon_minutes": PRIMARY_HORIZON,
            "selector": "causal_training_prefix_rank",
            "trade_cap": PRIMARY_TRADE_CAP,
            "risk_mode": PRIMARY_RISK_MODE,
            "metric": "mean gross mid-to-mid dollars per trade before fees",
            "spread_removed": True,
            "fees_removed": True,
            "pass_rule": "strictly greater than zero with at least one trade",
        }
    )
    primary["passed"] = bool(
        len(real_trades)
        and primary["mean_gross_mid_to_mid_usd_per_trade"] is not None
        and primary["mean_gross_mid_to_mid_usd_per_trade"] > 0.0
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    real_path = out_dir / "primary_selected_trades.parquet"
    real_trades.to_parquet(real_path, index=False)
    controls = None
    matched_path = None
    shuffled_path = None
    if primary["passed"]:
        shuffled_info = _prediction_member(fit, null=True)
        shuffled_predictions = pd.read_parquet(shuffled_info["path"])
        shuffled_scored = join_mid_only(shuffled_predictions, primary_candidates)
        controls, matched_trades, shuffled_trades = evaluate_controls(
            real_trades=real_trades,
            shuffled_scored=shuffled_scored,
            shuffled_cutoffs=_cutoffs(calibration, null=True),
            candidates_path=candidates_path,
            calibration=calibration,
        )
        matched_path = out_dir / "composition_matched_trades.parquet"
        shuffled_path = out_dir / "shuffled_null_trades.parquet"
        matched_trades.to_parquet(matched_path, index=False)
        shuffled_trades.to_parquet(shuffled_path, index=False)

    assertions = audit["assertions"]
    kill_results = {
        "mid_to_mid_gross_positive": primary["passed"],
        "beats_composition_matched_control": (
            controls["beats_composition_matched_control"] if controls else None
        ),
        "beats_shuffled_label_null": (
            controls["beats_shuffled_label_null"] if controls else None
        ),
        "per_feature_timestamp_audit": bool(
            assertions["every_feature_has_maximum_timestamp_t"]
            and assertions["no_future_feature"]
            and assertions["actual_future_mutation_proof"]
        ),
        "no_post_entry_slot_filter": bool(assertions["no_post_entry_candidate_filter"]),
        "chronological_out_of_sample": (
            controls["chronological_out_of_sample"] if controls else None
        ),
        "no_reserved_sessions": bool(assertions["reserved_sessions_absent"]),
    }
    if not primary["passed"]:
        status = "NEGATIVE_STOP_MID_TO_MID_GROSS_NOT_POSITIVE"
    elif all(value is True for value in kill_results.values()):
        status = "PRIMARY_SEVEN_KILLS_PASS_BID_AND_EXIT_STAGES_NOT_RUN"
    else:
        status = "NEGATIVE_CONTROL_OR_CHRONOLOGY_KILL_FAILED"

    per_parameter = int(frozen_value("minimum_sessions_per_neural_parameter"))
    receipt: dict[str, Any] = {
        "schema_version": "v5.causal-day-magnitude-corrected-primary-economics.v1",
        "created_on": "2026-08-14",
        "status": status,
        "declaration": {
            "path": str(declaration_path),
            "sha256": file_sha256(declaration_path),
        },
        "fit_receipt": {
            "path": str(fit_receipt_path),
            "sha256": file_sha256(fit_receipt_path),
        },
        "feature_audit": {
            "path": str(feature_audit_path),
            "sha256": file_sha256(feature_audit_path),
        },
        "selector_calibration": {
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
        },
        "fit_gate": {
            "status": "PERMITTED",
            "parameters": counts[PRIMARY_ARCHITECTURE],
            "kill_conditions": list(REQUIRED_KILL_CONDITIONS),
        },
        "primary_kill_condition": primary,
        "kill_condition_results": kill_results,
        "conditional_controls": controls,
        "evidence_budget": {
            "generous_effective_observations": MEASURED_EFFECTIVE_OBSERVATIONS,
            "generous_parameter_budget": MEASURED_EFFECTIVE_OBSERVATIONS // per_parameter,
            "conservative_effective_observations": CONSERVATIVE_EFFECTIVE_OBSERVATIONS,
            "conservative_parameter_budget": CONSERVATIVE_EFFECTIVE_OBSERVATIONS // per_parameter,
            "conservative_reported_range": [29, 50],
            "primary_architecture_parameters": counts[PRIMARY_ARCHITECTURE],
            "conservative_route_admits_primary": False,
        },
        "artifacts": {
            "primary_selected_trades": {
                "path": str(real_path),
                "sha256": file_sha256(real_path),
                "rows": len(real_trades),
            },
            "composition_matched_trades": (
                {
                    "path": str(matched_path),
                    "sha256": file_sha256(matched_path),
                }
                if matched_path is not None
                else None
            ),
            "shuffled_null_trades": (
                {
                    "path": str(shuffled_path),
                    "sha256": file_sha256(shuffled_path),
                }
                if shuffled_path is not None
                else None
            ),
        },
        "bid_economics_read": False,
        "exit_policy_fit": False,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    receipt["receipt_sha256"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(receipt_path)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--fit-receipt", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    run(
        declaration_path=args.declaration,
        fit_receipt_path=args.fit_receipt,
        feature_audit_path=args.feature_audit,
        calibration_path=args.calibration,
        candidates_path=args.candidates,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
