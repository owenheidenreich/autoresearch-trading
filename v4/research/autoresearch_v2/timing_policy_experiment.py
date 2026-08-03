"""Compiled lower-variance causal entry-timing policy experiment.

This family tests one causal signal per fixed time block.  A fitted controller
chooses ENTER_NOW or WAIT on the same contract; if the contract is no longer
eligible at the delayed decision, WAIT deterministically becomes NO_TRADE.
Every arm uses the same frozen 25-minute exit policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .builtin import SHORT25_EXIT
from .cache import PredictionCache, prediction_cache_key, stable_hash
from .causality import assert_mutate_future_invariant
from .compiler import CompiledHypothesis, load_and_compile
from .dataset import (
    FEATURE_INVENTORY,
    POLICY_INDEX,
    ROOT,
    SIGNED17,
    load_development_frame,
    rolling_folds,
    target_column,
)
from .registry import DuplicateSemanticHypothesis, read_registry, register
from .replay import replay_absolute, serial_summary, valid_for_policy
from .runner import clean, engine_source_hash, write_json
from .schema import SCHEMA_VERSION, TERMINAL_STATUSES
from .statistics import paired_summary, session_blocked_max_t


EPOCH_ID = "development_2025-08-01_2026-06-09_corrected_v3_two_clock"
REFERENCE_THRESHOLD = 0.0
PRACTICAL_EFFECT = 10.0
MAXIMUM_MDE = 25.0
PERMUTATIONS = 4999
MICRO_FEATURES = ("size_imbalance", "depth_total", "opt_spread", "iv", "theta")
TIME_FEATURES = ("minute_of_session", "minutes_to_1555")
MODEL_FEATURES = tuple(SIGNED17) + MICRO_FEATURES + TIME_FEATURES
VARIANTS: tuple[tuple[str, int, float | None], ...] = (
    ("learned_wait5_gt0", 5, 0.0),
    ("learned_wait5_gt10", 5, 10.0),
    ("always_wait5", 5, None),
    ("learned_wait15_gt0", 15, 0.0),
    ("learned_wait15_gt10", 15, 10.0),
    ("always_wait15", 15, None),
)
MODEL_PARAMETERS = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "max_depth": 3,
    "min_samples_leaf": 80,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "random_state": 211,
}


def _feature_specs() -> list[dict[str, str]]:
    items = [{"name": name, **FEATURE_INVENTORY[name]} for name in SIGNED17]
    items.extend(
        {
            "name": name,
            "family": "live_safe_microstructure",
            "available_at": "decision_time",
            "live_twin": "Protocol101 live option ladder BBO/size and causal derived Greeks",
        }
        for name in MICRO_FEATURES
    )
    items.extend(
        {
            "name": name,
            "family": "causal_clock",
            "available_at": "decision_time",
            "live_twin": "New_York decision clock",
        }
        for name in TIME_FEATURES
    )
    return items


def hypothesis_payload(variant: str, delay: int, gate: float | None) -> dict[str, Any]:
    behavior = "always waits" if gate is None else f"waits when predicted delay advantage exceeds ${gate:g}"
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": f"entry_block_first_{variant}_short25_v1",
        "claim": (
            f"A causal first-signal-per-fixed-block controller that {behavior} for {delay} minutes "
            "improves strict one-account dollars over immediate entry on the same signal and contract."
        ),
        "component": "timing",
        "feature_families": {
            "add": ["signed17", "live_safe_microstructure", "causal_clock"],
            "remove": [],
        },
        "features": _feature_specs(),
        "target": {
            "kind": f"same_contract_wait_{delay}m_incremental_pnl",
            "fields": [SHORT25_EXIT],
            "weights": [1.0],
            "frozen_reference_exit": SHORT25_EXIT,
            "required_materialized_fields": [
                "policy_net_pnl",
                "policy_mid_pnl",
                "realized_exit_time",
                "source_exit_quote_time",
                "exit_quote_age",
                "exit_reason",
                "executable_exit_bid",
                "policy_deadline",
            ],
        },
        "arms": [variant, "enter_now"],
        "primary_contrast": variant,
        "threshold": {"kind": "fixed", "value": REFERENCE_THRESHOLD, "fit_role": "none"},
        "paired_baseline": {
            "name": "enter_now",
            "exposure_matching": ["signal_identity", "contract_id", "side", "moneyness"],
        },
        "model": {
            "family": "hist_gradient_boosting",
            "hyperparameters": {key: value for key, value in MODEL_PARAMETERS.items() if key != "random_state"},
            "seeds": [211],
            "max_fits": 10,
        },
        "development_epoch": {
            "epoch_id": EPOCH_ID,
            "session_manifest": (
                "v4/audit/autoresearch/"
                "protocol101_pathd_entry_exit_model_research_corrected_v3_2_2026_08_01/"
                "session_assignments.json"
            ),
            "allowed_role": "development",
            "confirmation_rule": "fresh_epoch_once_then_roll_to_development",
        },
        "power": {
            "alpha": 0.05,
            "power": 0.80,
            "practical_effect_dollars_per_session": PRACTICAL_EFFECT,
            "maximum_mde_dollars_per_session": MAXIMUM_MDE,
        },
        "compute_budget": {"max_permutations": PERMUTATIONS, "max_minutes": 90},
        "terminal_statuses": list(TERMINAL_STATUSES),
    }


def initialize(hypotheses: Path, preregistration: Path) -> dict[str, Any]:
    if preregistration.exists() or hypotheses.exists():
        raise FileExistsError("timing family preregistration already exists")
    hypotheses.mkdir(parents=True)
    compiled = []
    for variant, delay, gate in VARIANTS:
        payload = hypothesis_payload(variant, delay, gate)
        path = hypotheses / f"{payload['hypothesis_id']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        item = load_and_compile(path)
        compiled.append(
            {
                "hypothesis_id": item.spec.hypothesis_id,
                "semantic_hash": item.semantic_hash,
                "path": str(path),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "autoresearch_v2.timing_family_preregistration.v1",
        "status": "FROZEN_BEFORE_RESULTS",
        "epoch_id": EPOCH_ID,
        "opportunity_definition": {
            "blocks_et": ["10:01-10:59", "11:00-11:59", "12:00-12:59", "13:00-13:59", "14:00-14:59", "15:00-15:20"],
            "rule": "first decision in each block whose top OOF candidate prediction is greater than zero",
            "candidate_tie_break": "descending OOF reference prediction then candidate_uid",
            "unavailable_after_wait": "NO_TRADE with zero arm PnL",
        },
        "frozen_exit": SHORT25_EXIT,
        "features": list(MODEL_FEATURES),
        "model": MODEL_PARAMETERS,
        "variants": [
            {"name": variant, "delay_minutes": delay, "wait_advantage_gate": gate}
            for variant, delay, gate in VARIANTS
        ],
        "family_correction": "session_blocked_sign_flip_maxT_over_all_six_variants",
        "routing": {
            "cheap_screen": "mean policy delta per causal signal within session",
            "final_gate": "raw strict one-account total dollar delta per session",
            "requirements": [
                "family_maxT_p<=0.05",
                "mean_delta>=10",
                "MDE80<=25",
                "strict_serial_one_sided_p<=0.05",
                "strict_serial_delta_positive_in_at_least_4_of_5_folds",
                "absolute_arm_PnL_positive_in_at_least_4_of_5_folds",
            ],
            "winner": "highest worst-fold strict-serial delta then mean delta then semantic hash",
        },
        "hypotheses": compiled,
        "holdout_access_count": 0,
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    write_json(preregistration, payload)
    return payload


def _verify_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    semantic = dict(payload)
    observed = semantic.pop("preregistration_sha256", None)
    if observed != stable_hash(semantic) or payload.get("status") != "FROZEN_BEFORE_RESULTS":
        raise RuntimeError("timing family preregistration drifted")
    return payload


def _derived(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    local = pd.to_datetime(result["decision_time_ns"], unit="ns", utc=True).dt.tz_convert(
        "America/New_York"
    )
    result["minute_of_session"] = local.dt.hour * 60 + local.dt.minute - 570
    result["minutes_to_1555"] = 955 - (local.dt.hour * 60 + local.dt.minute)
    return result


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**MODEL_PARAMETERS)


def _future_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = (
        "pnl_",
        "mid_pnl_",
        "exit_ns_",
        "source_exit_ns_",
        "quote_age_ms_",
        "reason_",
        "exit_bid_",
        "deadline_ns_",
        "invalid_",
        "_future_",
    )
    return [column for column in frame if column.startswith(prefixes)]


def _fit_cached(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: np.ndarray,
    target_identity: Mapping[str, Any],
    fold: Mapping[str, Any],
    foundation_hash: str,
    cache: PredictionCache,
    features: Sequence[str] = MODEL_FEATURES,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = tuple(features)
    ordered = test.sort_values("candidate_uid").copy()
    row_hash = stable_hash(ordered["candidate_uid"].tolist())
    key = prediction_cache_key(
        features=features,
        target=target_identity,
        model={"family": "hist_gradient_boosting", "hyperparameters": MODEL_PARAMETERS},
        fold=dict(fold),
        seed=211,
        foundation_hash=foundation_hash,
    )
    cached = cache.load(key, row_hash=row_hash)
    hit = cached is not None
    finite = np.isfinite(np.asarray(target, dtype=float))
    if cached is None:
        model = _model()
        model.fit(train.loc[finite, features].to_numpy(float), np.asarray(target, dtype=float)[finite])
        predictions = model.predict(ordered.loc[:, features].to_numpy(float))
    else:
        predictions, model = cached
    invariance = assert_mutate_future_invariant(
        ordered,
        feature_columns=features,
        future_columns=_future_columns(ordered),
        scorer=lambda matrix, fitted=model: fitted.predict(matrix.to_numpy(float)),
    )
    if not hit:
        cache.store(
            key,
            predictions,
            row_hash=row_hash,
            model=model,
            mutate_future_invariance_status=str(invariance["status"]),
        )
    return np.asarray(predictions, dtype=float), {
        "cache_key": key,
        "cache_hit": hit,
        "train_rows": int(finite.sum()),
        "test_rows": len(ordered),
        "mutate_future_invariance": invariance,
        "ordered_candidate_uids": ordered["candidate_uid"].tolist(),
    }


def _wait_targets(frame: pd.DataFrame, delay: int) -> np.ndarray:
    target = target_column(SHORT25_EXIT)
    later = frame[["session", "decision_time_ns", "contract_id", target]].copy()
    later["decision_time_ns"] -= int(delay * 60 * 1_000_000_000)
    later = later.rename(columns={target: "_future_wait_pnl"})
    merged = frame[["candidate_uid", "session", "decision_time_ns", "contract_id", target]].merge(
        later, on=["session", "decision_time_ns", "contract_id"], how="left", validate="one_to_one"
    )
    if merged["candidate_uid"].tolist() != frame["candidate_uid"].tolist():
        raise RuntimeError("wait target merge changed candidate ordering")
    return merged["_future_wait_pnl"].to_numpy(float) - merged[target].to_numpy(float)


def fit_oof(
    frame: pd.DataFrame,
    *,
    folds: Sequence[Mapping[str, Any]],
    foundation_hash: str,
    cache: PredictionCache,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = target_column(SHORT25_EXIT)
    outputs = []
    receipts = []
    for fold in folds:
        train = frame[frame["session"].isin(fold["model_fit"])].copy()
        test = frame[frame["session"].isin(fold["outer_test"])].copy()
        test = test.sort_values("candidate_uid").reset_index(drop=True)
        ref_predictions, ref_receipt = _fit_cached(
            train,
            test,
            target=train[target].to_numpy(float),
            target_identity={"kind": "single_policy_pnl", "policy": SHORT25_EXIT},
            fold=fold,
            foundation_hash=foundation_hash,
            cache=cache,
        )
        test["_ref_pred"] = ref_predictions
        delay_receipts = {}
        for delay in (5, 15):
            train_delta = _wait_targets(train, delay)
            predictions, receipt = _fit_cached(
                train,
                test,
                target=train_delta,
                target_identity={
                    "kind": "same_contract_wait_incremental_pnl",
                    "policy": SHORT25_EXIT,
                    "delay_minutes": delay,
                    "unavailable_behavior": "NO_TRADE",
                },
                fold=fold,
                foundation_hash=foundation_hash,
                cache=cache,
            )
            test[f"_delta_pred_{delay}"] = predictions
            delay_receipts[str(delay)] = {key: value for key, value in receipt.items() if key != "ordered_candidate_uids"}
        test["fold"] = int(fold["fold"])
        outputs.append(test)
        receipts.append(
            {
                "fold": int(fold["fold"]),
                "train_sessions": len(fold["model_fit"]),
                "test_sessions": len(fold["outer_test"]),
                "reference": {key: value for key, value in ref_receipt.items() if key != "ordered_candidate_uids"},
                "delay": delay_receipts,
            }
        )
    return pd.concat(outputs, ignore_index=True), {
        "features": list(MODEL_FEATURES),
        "folds": receipts,
        "fit_count": sum(
            int(not row["reference"]["cache_hit"])
            + sum(int(not item["cache_hit"]) for item in row["delay"].values())
            for row in receipts
        ),
        "cache_hit_count": sum(
            int(row["reference"]["cache_hit"])
            + sum(int(item["cache_hit"]) for item in row["delay"].values())
            for row in receipts
        ),
    }


def _block(frame: pd.DataFrame) -> np.ndarray:
    minute = frame["minute_of_session"].to_numpy(int)
    return np.select(
        [
            (minute >= 31) & (minute < 90),
            (minute >= 90) & (minute < 150),
            (minute >= 150) & (minute < 210),
            (minute >= 210) & (minute < 270),
            (minute >= 270) & (minute < 330),
            (minute >= 330) & (minute <= 350),
        ],
        [0, 1, 2, 3, 4, 5],
        default=-1,
    )


def opportunities(oof: pd.DataFrame) -> pd.DataFrame:
    eligible = oof[(oof["_ref_pred"] > REFERENCE_THRESHOLD)].copy()
    eligible["_block"] = _block(eligible)
    eligible = eligible[eligible["_block"] >= 0]
    eligible = eligible.sort_values(
        ["session", "decision_time_ns", "_ref_pred", "candidate_uid"],
        ascending=[True, True, False, True],
    )
    per_decision = eligible.groupby(["session", "decision_time_ns"], sort=True).head(1)
    return (
        per_decision.sort_values(["session", "_block", "decision_time_ns", "candidate_uid"])
        .groupby(["session", "_block"], sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def _session_mean(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {session: float(np.mean(items)) for session, items in sorted(values.items())}


def build_policy(
    oof: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    delay: int,
    gate: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    lookup = {
        (str(row["session"]), int(row["decision_time_ns"]), str(row["contract_id"])): row
        for row in oof.to_dict("records")
    }
    target = target_column(SHORT25_EXIT)
    by_session: dict[str, list[float]] = {}
    arm_rows = []
    baseline_rows = []
    counts = {"signals": 0, "wait_decisions": 0, "enter_now_decisions": 0, "wait_unavailable_no_trade": 0}
    for now in signals.to_dict("records"):
        if not valid_for_policy(now, SHORT25_EXIT):
            continue
        counts["signals"] += 1
        baseline = dict(now)
        baseline["_pred"] = float(now["_ref_pred"])
        baseline_rows.append(baseline)
        should_wait = gate is None or float(now[f"_delta_pred_{delay}"]) > float(gate)
        arm: dict[str, Any] | None = now
        if should_wait:
            counts["wait_decisions"] += 1
            arm = lookup.get(
                (
                    str(now["session"]),
                    int(now["decision_time_ns"]) + int(delay * 60 * 1_000_000_000),
                    str(now["contract_id"]),
                )
            )
            if arm is None or not valid_for_policy(arm, SHORT25_EXIT):
                arm = None
                counts["wait_unavailable_no_trade"] += 1
        else:
            counts["enter_now_decisions"] += 1
        if arm is None:
            arm_pnl = 0.0
        else:
            arm = dict(arm)
            arm["_pred"] = float(now["_ref_pred"] + now[f"_delta_pred_{delay}"])
            arm_rows.append(arm)
            arm_pnl = float(arm[target])
        by_session.setdefault(str(now["session"]), []).append(arm_pnl - float(now[target]))
    return _session_mean(by_session), arm_rows, baseline_rows, counts


def _trade_sessions(trades: Sequence[Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for trade in trades:
        result[str(trade.session)] = result.get(str(trade.session), 0.0) + float(
            trade.raw_label_pnl_after_campaign_fee
        )
    return result


def strict_serial(
    arm_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    sessions: Sequence[str],
    session_to_fold: Mapping[str, int],
    variant: str,
) -> dict[str, Any]:
    arm_trades, arm_state = replay_absolute(
        arm_rows, policy=SHORT25_EXIT, strategy=f"autoresearch_v2:{variant}"
    )
    baseline_trades, baseline_state = replay_absolute(
        baseline_rows, policy=SHORT25_EXIT, strategy="autoresearch_v2:enter_now"
    )
    arm = _trade_sessions(arm_trades)
    baseline = _trade_sessions(baseline_trades)
    deltas = {session: arm.get(session, 0.0) - baseline.get(session, 0.0) for session in sessions}
    fold_delta = {
        str(fold): float(sum(value for session, value in deltas.items() if session_to_fold[session] == fold))
        for fold in range(1, 6)
    }
    fold_arm = {
        str(fold): float(sum(value for session, value in arm.items() if session_to_fold[session] == fold))
        for fold in range(1, 6)
    }
    return {
        "statistics": paired_summary(deltas),
        "session_deltas": deltas,
        "fold_delta_pnl": fold_delta,
        "fold_arm_pnl": fold_arm,
        "positive_delta_folds": sum(value > 0.0 for value in fold_delta.values()),
        "positive_arm_folds": sum(value > 0.0 for value in fold_arm.values()),
        "arm": serial_summary(arm_trades, arm_state),
        "baseline": serial_summary(baseline_trades, baseline_state),
    }


def _status(screen: Mapping[str, Any], serial: Mapping[str, Any]) -> str:
    if float(screen.get("mde80") or float("inf")) > MAXIMUM_MDE:
        return "UNDERPOWERED"
    if float(screen.get("mean") or 0.0) < PRACTICAL_EFFECT or float(
        screen.get("maxT_p_one_sided") or 1.0
    ) > 0.05:
        return "NO_INCREMENTAL_EDGE"
    stats = serial["statistics"]
    if float(stats.get("mde80") or float("inf")) > MAXIMUM_MDE:
        return "UNDERPOWERED"
    if not (
        float(stats.get("mean") or 0.0) >= PRACTICAL_EFFECT
        and float(stats.get("p_one_sided") or 1.0) <= 0.05
        and int(serial["positive_delta_folds"]) >= 4
        and int(serial["positive_arm_folds"]) >= 4
        and float(serial["arm"]["total_pnl"]) > 0.0
    ):
        return "NO_INCREMENTAL_EDGE"
    return "PROVISIONAL_EDGE"


def run(
    *,
    foundation: Path,
    hypotheses: Path,
    preregistration: Path,
    output: Path,
    cache_root: Path,
    registry_path: Path,
) -> dict[str, Any]:
    prereg = _verify_preregistration(preregistration)
    compiled: dict[str, CompiledHypothesis] = {}
    for path in sorted(hypotheses.glob("*.json")):
        item = load_and_compile(path)
        compiled[item.spec.hypothesis_id] = item
    expected_ids = {
        f"entry_block_first_{variant}_short25_v1" for variant, _, _ in VARIANTS
    }
    if set(compiled) != expected_ids:
        raise RuntimeError("timing hypothesis family membership drifted")
    prior = {row["semantic_hash"] for row in read_registry(registry_path)}
    duplicates = prior & {item.semantic_hash for item in compiled.values()}
    if duplicates:
        raise DuplicateSemanticHypothesis("timing hypothesis already exists in registry")
    frame, foundation_payload = load_development_frame(foundation)
    if foundation_payload.get("generation") != "corrected-v3.2-distinct-two-clock":
        raise RuntimeError("timing experiment requires the distinct corrected-v3 foundation")
    frame = _derived(frame)
    folds = rolling_folds(frame["session"].unique())
    cache = PredictionCache(cache_root)
    oof, fit_receipt = fit_oof(
        frame,
        folds=folds,
        foundation_hash=foundation_payload["foundation_sha256"],
        cache=cache,
    )
    signals = opportunities(oof)
    sessions = sorted(str(value) for value in oof["session"].unique())
    session_to_fold = {
        str(session): int(row["fold"])
        for _, row in oof[["session", "fold"]].drop_duplicates().iterrows()
        for session in [row["session"]]
    }
    contrasts = {}
    policies = {}
    for variant, delay, gate in VARIANTS:
        contrasts[variant], arm_rows, baseline_rows, counts = build_policy(
            oof, signals, delay=delay, gate=gate
        )
        policies[variant] = {
            "arm_rows": arm_rows,
            "baseline_rows": baseline_rows,
            "counts": counts,
        }
    corrected = session_blocked_max_t(contrasts, permutations=PERMUTATIONS, seed=211)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "session_mean_deltas.json", contrasts)
    results = {}
    survivors = []
    for variant, delay, gate in VARIANTS:
        hypothesis_id = f"entry_block_first_{variant}_short25_v1"
        item = compiled[hypothesis_id]
        serial = strict_serial(
            policies[variant]["arm_rows"],
            policies[variant]["baseline_rows"],
            sessions=sessions,
            session_to_fold=session_to_fold,
            variant=variant,
        )
        status = _status(corrected[variant], serial)
        result = {
            "schema_version": "autoresearch_v2.timing_policy_result.v1",
            "hypothesis_id": hypothesis_id,
            "semantic_hash": item.semantic_hash,
            "engine_source_hash": engine_source_hash(),
            "foundation_sha256": foundation_payload["foundation_sha256"],
            "preregistration_sha256": prereg["preregistration_sha256"],
            "status": status,
            "evidence_grade": "development_only_non_promotable",
            "holdout_access_count": 0,
            "protected_holdout_opened": False,
            "variant": {"delay_minutes": delay, "wait_advantage_gate": gate},
            "opportunities": policies[variant]["counts"],
            "screen": corrected[variant],
            "family_correction": {
                "method": "session_blocked_sign_flip_maxT",
                "permutations": PERMUTATIONS,
                "family": sorted(contrasts),
            },
            "strict_serial": serial,
        }
        result_path = output / "results" / f"{hypothesis_id}.json"
        write_json(result_path, result)
        register(
            registry_path,
            item,
            status=status,
            result_path=str(result_path),
            engine_source_hash=engine_source_hash(),
        )
        results[hypothesis_id] = result
        if status == "PROVISIONAL_EDGE":
            survivors.append(result)
    survivors.sort(
        key=lambda item: (
            -min(item["strict_serial"]["fold_delta_pnl"].values()),
            -float(item["strict_serial"]["statistics"]["mean"]),
            item["semantic_hash"],
        )
    )
    winner = survivors[0]["hypothesis_id"] if survivors else None
    suite = {
        "schema_version": "autoresearch_v2.timing_policy_suite.v1",
        "status": "PROVISIONAL_EDGE" if winner else "NO_SURVIVOR",
        "winner": winner,
        "terminal_status_counts": {
            status: sum(item["status"] == status for item in results.values())
            for status in TERMINAL_STATUSES
        },
        "foundation_sha256": foundation_payload["foundation_sha256"],
        "foundation_sessions": foundation_payload["session_count"],
        "oof_sessions": len(sessions),
        "signal_count": len(signals),
        "fit_receipt": fit_receipt,
        "holdout_access_count": 0,
        "protected_holdout_opened": False,
        "results": {key: value["status"] for key, value in results.items()},
    }
    write_json(output / "suite_result.json", suite)
    (output / "report.md").write_text(
        "# Corrected-v3 block-first entry timing family\n\n"
        f"- Suite status: `{suite['status']}`\n"
        f"- Winner: `{winner}`\n"
        f"- Foundation sessions: `{foundation_payload['session_count']}`\n"
        f"- OOF sessions: `{len(sessions)}`\n"
        f"- Causal signals: `{len(signals)}`\n"
        "- Holdout opens: `0`\n"
    )
    return suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--hypotheses", type=Path, required=True)
    init.add_argument("--preregistration", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--foundation", type=Path, required=True)
    execute.add_argument("--hypotheses", type=Path, required=True)
    execute.add_argument("--preregistration", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--cache", type=Path, required=True)
    execute.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        result = initialize(args.hypotheses, args.preregistration)
    else:
        result = run(
            foundation=args.foundation,
            hypotheses=args.hypotheses,
            preregistration=args.preregistration,
            output=args.output,
            cache_root=args.cache,
            registry_path=args.registry,
        )
    print(json.dumps(clean(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
