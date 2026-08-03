"""Attribute corrected-v3 block-first entry economics to genuine model edge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .builtin import SHORT25_EXIT
from .cache import PredictionCache, stable_hash
from .compiler import CompiledHypothesis, load_and_compile
from .dataset import FEATURE_INVENTORY, SIGNED17, load_development_frame, rolling_folds, target_column
from .registry import DuplicateSemanticHypothesis, read_registry, register
from .replay import paired_lockstep_component_replay, replay_absolute, serial_summary, valid_for_policy
from .runner import clean, engine_source_hash, write_json
from .schema import SCHEMA_VERSION, TERMINAL_STATUSES
from .statistics import paired_summary, session_blocked_max_t
from .timing_policy_experiment import (
    EPOCH_ID,
    MAXIMUM_MDE,
    MICRO_FEATURES,
    MODEL_FEATURES,
    MODEL_PARAMETERS,
    PERMUTATIONS,
    PRACTICAL_EFFECT,
    TIME_FEATURES,
    _block,
    _derived,
    _fit_cached,
    fit_oof as fit_full_oof,
    opportunities,
    strict_serial,
)


CANDIDATES = ("signed18", "full25")
REQUIRED_CONTROLS = ("deterministic", "shuffled", "nearest_atm")


def _features(names: Sequence[str]) -> list[dict[str, str]]:
    result = []
    for name in names:
        if name in FEATURE_INVENTORY:
            result.append({"name": name, **FEATURE_INVENTORY[name]})
        elif name in MICRO_FEATURES:
            result.append(
                {
                    "name": name,
                    "family": "live_safe_microstructure",
                    "available_at": "decision_time",
                    "live_twin": "Protocol101 live option ladder BBO/size and causal derived Greeks",
                }
            )
        elif name in TIME_FEATURES:
            result.append(
                {
                    "name": name,
                    "family": "causal_clock",
                    "available_at": "decision_time",
                    "live_twin": "New_York decision clock",
                }
            )
        else:
            raise KeyError(name)
    return result


def hypothesis_payload(candidate: str) -> dict[str, Any]:
    names = tuple(SIGNED17) if candidate == "signed18" else MODEL_FEATURES
    families = ["signed17"] if candidate == "signed18" else [
        "signed17",
        "live_safe_microstructure",
        "causal_clock",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": f"entry_block_first_{candidate}_short25_attribution_v1",
        "claim": (
            f"The {candidate} block-first entry policy has incremental strict-serial edge over "
            "deterministic schedule, shuffled-target, and time-matched nearest-ATM controls."
        ),
        "component": "entry_policy",
        "feature_families": {"add": families, "remove": []},
        "features": _features(names),
        "target": {
            "kind": "single_policy_pnl",
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
        "arms": [candidate, "deterministic", "shuffled", "nearest_atm", "exact_random"],
        "primary_contrast": candidate,
        "threshold": {"kind": "fixed", "value": 0.0, "fit_role": "none"},
        "paired_baseline": {
            "name": "required_control_family",
            "exposure_matching": ["session", "fixed_time_block", "frozen_exit", "account_rules"],
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
    if hypotheses.exists() or preregistration.exists():
        raise FileExistsError("entry attribution preregistration already exists")
    hypotheses.mkdir(parents=True)
    compiled = []
    for candidate in CANDIDATES:
        payload = hypothesis_payload(candidate)
        path = hypotheses / f"{payload['hypothesis_id']}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        item = load_and_compile(path)
        compiled.append(
            {"hypothesis_id": item.spec.hypothesis_id, "semantic_hash": item.semantic_hash, "path": str(path)}
        )
    payload: dict[str, Any] = {
        "schema_version": "autoresearch_v2.entry_attribution_preregistration.v1",
        "status": "FROZEN_BEFORE_RESULTS",
        "development_observation": (
            "The immediate-entry reference arm in the prior preregistered timing family had "
            "positive absolute strict-serial PnL in five folds; this family tests attribution."
        ),
        "epoch_id": EPOCH_ID,
        "frozen_exit": SHORT25_EXIT,
        "candidate_policies": list(CANDIDATES),
        "controls": {
            "deterministic": "first decision per fixed block; momentum15-aligned side; nearest ATM",
            "shuffled": "same model and policy with target values deterministically permuted within training session",
            "nearest_atm": "same model signal time and side; nearest ATM within $1.50 premium caliper",
            "exact_random": "same model signal time and side; fixed SHA-256 choice within $1.50 premium caliper",
        },
        "family": [
            f"{candidate}_vs_{control}"
            for candidate in CANDIDATES
            for control in (*REQUIRED_CONTROLS, "exact_random")
        ] + ["full25_vs_signed18"],
        "qualification": {
            "each_required_control": [
                "family_maxT_p<=0.05",
                "screen_mean_delta>=10",
                "screen_MDE80<=25",
                "strict_serial_mean_delta>=10",
                "strict_serial_one_sided_p<=0.05",
                "strict_serial_MDE80<=25",
                "strict_serial_delta_positive_in_at_least_4_of_5_folds",
            ],
            "absolute_candidate": "positive strict-serial PnL in at least 4 of 5 folds and total PnL>0",
            "winner": "prefer signed18 if both qualify; otherwise qualifying candidate",
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
        raise RuntimeError("entry attribution preregistration drifted")
    return payload


def _shuffled_targets(train: pd.DataFrame, *, fold: int) -> np.ndarray:
    target = target_column(SHORT25_EXIT)
    values = train[target].to_numpy(float).copy()
    sessions = train["session"].astype(str).to_numpy()
    for session in sorted(set(sessions)):
        indices = np.flatnonzero(sessions == session)
        finite = indices[np.isfinite(values[indices])]
        seed = int(hashlib.sha256(f"{fold}|{session}|strong_shuffle".encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        values[finite] = values[rng.permutation(finite)]
    return values


def fit_controls(
    frame: pd.DataFrame,
    *,
    folds: Sequence[Mapping[str, Any]],
    foundation_hash: str,
    cache: PredictionCache,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outputs = []
    receipts = []
    target = target_column(SHORT25_EXIT)
    for fold in folds:
        train = frame[frame["session"].isin(fold["model_fit"])].copy()
        test = frame[frame["session"].isin(fold["outer_test"])].copy()
        test = test.sort_values("candidate_uid").reset_index(drop=True)
        signed, signed_receipt = _fit_cached(
            train,
            test,
            target=train[target].to_numpy(float),
            target_identity={"kind": "single_policy_pnl", "policy": SHORT25_EXIT, "feature_set": "signed18"},
            fold=fold,
            foundation_hash=foundation_hash,
            cache=cache,
            features=SIGNED17,
        )
        shuffled, shuffled_receipt = _fit_cached(
            train,
            test,
            target=_shuffled_targets(train, fold=int(fold["fold"])),
            target_identity={
                "kind": "within_training_session_strong_shuffle",
                "policy": SHORT25_EXIT,
                "feature_set": "full25",
                "seed": 211,
            },
            fold=fold,
            foundation_hash=foundation_hash,
            cache=cache,
            features=MODEL_FEATURES,
        )
        test["_signed_pred"] = signed
        test["_shuffled_pred"] = shuffled
        test["fold"] = int(fold["fold"])
        outputs.append(test)
        receipts.append(
            {
                "fold": int(fold["fold"]),
                "signed18": {key: value for key, value in signed_receipt.items() if key != "ordered_candidate_uids"},
                "shuffled": {key: value for key, value in shuffled_receipt.items() if key != "ordered_candidate_uids"},
            }
        )
    return pd.concat(outputs, ignore_index=True), {"folds": receipts}


def _opportunities_for(oof: pd.DataFrame, column: str) -> pd.DataFrame:
    copy = oof.copy()
    copy["_ref_pred"] = copy[column]
    result = opportunities(copy)
    result["_pred"] = result[column]
    return result


def deterministic_policy(oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    frame["_block"] = _block(frame)
    frame = frame[frame["_block"] >= 0]
    first_times = (
        frame.groupby(["session", "_block"], sort=True)["decision_time_ns"].min().rename("_first")
    )
    frame = frame.join(first_times, on=["session", "_block"])
    frame = frame[frame["decision_time_ns"] == frame["_first"]].copy()
    desired = np.where(frame["momentum_15m_bps"].to_numpy(float) >= 0.0, "C", "P")
    frame = frame[frame["right"].to_numpy(str) == desired]
    frame = frame.sort_values(
        ["session", "_block", "abs_moneyness", "opt_spread", "candidate_uid"]
    )
    result = frame.groupby(["session", "_block"], sort=True).head(1).copy()
    result["_pred"] = 0.0
    return result.reset_index(drop=True)


def matched_control_pairs(
    oof: pd.DataFrame, signals: pd.DataFrame, *, random_control: bool
) -> tuple[list[tuple[str, dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    groups = {
        (str(session), int(decision)): group
        for (session, decision), group in oof.groupby(["session", "decision_time_ns"], sort=False)
    }
    pairs = []
    controls = []
    for arm in signals.to_dict("records"):
        pool = groups[(str(arm["session"]), int(arm["decision_time_ns"]))]
        pool = pool[
            (pool["right"] == arm["right"])
            & ((pool["entry_ask"] - float(arm["entry_ask"])).abs() <= 1.5)
        ].copy()
        if pool.empty:
            continue
        if random_control:
            pool["_control_key"] = [
                hashlib.sha256(
                    f"{arm['session']}|{arm['decision_time_ns']}|{value}|exact_random".encode()
                ).hexdigest()
                for value in pool["candidate_uid"]
            ]
            control = pool.sort_values(["_control_key", "candidate_uid"]).iloc[0].to_dict()
        else:
            control = pool.sort_values(["abs_moneyness", "candidate_uid"]).iloc[0].to_dict()
        control["_pred"] = 0.0
        arm["_pred"] = float(arm["_pred"])
        signal = f"{arm['session']}|{arm['decision_time_ns']}|{arm.get('_block', '')}"
        pairs.append((signal, arm, control))
        controls.append(control)
    return pairs, controls


def _valid_session_pnl(rows: Sequence[Mapping[str, Any]], sessions: Sequence[str]) -> dict[str, float]:
    target = target_column(SHORT25_EXIT)
    result = {session: 0.0 for session in sessions}
    for row in rows:
        if valid_for_policy(dict(row), SHORT25_EXIT):
            result[str(row["session"])] += float(row[target])
    return result


def _policy_delta(
    arm: Sequence[Mapping[str, Any]], baseline: Sequence[Mapping[str, Any]], sessions: Sequence[str]
) -> dict[str, float]:
    a = _valid_session_pnl(arm, sessions)
    b = _valid_session_pnl(baseline, sessions)
    return {session: a[session] - b[session] for session in sessions}


def _paired_delta(
    pairs: Sequence[tuple[str, dict[str, Any], dict[str, Any]]], sessions: Sequence[str]
) -> dict[str, float]:
    target = target_column(SHORT25_EXIT)
    result: dict[str, list[float]] = {session: [] for session in sessions}
    for _, arm, baseline in pairs:
        if valid_for_policy(arm, SHORT25_EXIT) and valid_for_policy(baseline, SHORT25_EXIT):
            result[str(arm["session"])].append(float(arm[target]) - float(baseline[target]))
    return {session: float(np.sum(values)) if values else 0.0 for session, values in result.items()}


def _paired_serial(
    pairs: Sequence[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    sessions: Sequence[str],
    session_to_fold: Mapping[str, int],
    arm_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    replay = paired_lockstep_component_replay(
        pairs, policy=SHORT25_EXIT, arm_name=arm_name, baseline_name=baseline_name
    )
    deltas = {session: float(replay["session_deltas"].get(session, 0.0)) for session in sessions}
    fold_delta = {
        str(fold): float(sum(value for session, value in deltas.items() if session_to_fold[session] == fold))
        for fold in range(1, 6)
    }
    return {
        "statistics": paired_summary(deltas),
        "session_deltas": deltas,
        "fold_delta_pnl": fold_delta,
        "positive_delta_folds": sum(value > 0.0 for value in fold_delta.values()),
        "paired": {key: value for key, value in replay.items() if key != "session_deltas"},
    }


def _contrast_status(screen: Mapping[str, Any], serial: Mapping[str, Any]) -> str:
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
    compiled: dict[str, CompiledHypothesis] = {
        item.spec.hypothesis_id: item
        for item in (load_and_compile(path) for path in sorted(hypotheses.glob("*.json")))
    }
    expected = {f"entry_block_first_{name}_short25_attribution_v1" for name in CANDIDATES}
    if set(compiled) != expected:
        raise RuntimeError("entry attribution hypothesis membership drifted")
    prior = {row["semantic_hash"] for row in read_registry(registry_path)}
    if prior & {item.semantic_hash for item in compiled.values()}:
        raise DuplicateSemanticHypothesis("entry attribution semantic duplicate")
    frame, foundation_payload = load_development_frame(foundation)
    if foundation_payload.get("generation") != "corrected-v3.2-distinct-two-clock":
        raise RuntimeError("entry attribution requires corrected-v3 foundation")
    frame = _derived(frame)
    folds = rolling_folds(frame["session"].unique())
    cache = PredictionCache(cache_root)
    full_oof, full_fit = fit_full_oof(
        frame, folds=folds, foundation_hash=foundation_payload["foundation_sha256"], cache=cache
    )
    controls_oof, control_fit = fit_controls(
        frame, folds=folds, foundation_hash=foundation_payload["foundation_sha256"], cache=cache
    )
    prediction_columns = ["candidate_uid", "_signed_pred", "_shuffled_pred"]
    oof = full_oof.merge(
        controls_oof[prediction_columns], on="candidate_uid", how="left", validate="one_to_one"
    )
    sessions = sorted(str(value) for value in oof["session"].unique())
    session_to_fold = {
        str(row["session"]): int(row["fold"])
        for _, row in oof[["session", "fold"]].drop_duplicates().iterrows()
    }
    policy_rows = {
        "full25": _opportunities_for(oof, "_ref_pred").to_dict("records"),
        "signed18": _opportunities_for(oof, "_signed_pred").to_dict("records"),
        "shuffled": _opportunities_for(oof, "_shuffled_pred").to_dict("records"),
        "deterministic": deterministic_policy(oof).to_dict("records"),
    }
    pair_sets = {}
    control_rows = {}
    for candidate in CANDIDATES:
        signal_frame = pd.DataFrame(policy_rows[candidate])
        for name, random_control in (("nearest_atm", False), ("exact_random", True)):
            pairs, rows = matched_control_pairs(oof, signal_frame, random_control=random_control)
            pair_sets[f"{candidate}_vs_{name}"] = pairs
            control_rows[f"{candidate}_{name}"] = rows
    contrasts: dict[str, dict[str, float]] = {}
    for candidate in CANDIDATES:
        contrasts[f"{candidate}_vs_deterministic"] = _policy_delta(
            policy_rows[candidate], policy_rows["deterministic"], sessions
        )
        contrasts[f"{candidate}_vs_shuffled"] = _policy_delta(
            policy_rows[candidate], policy_rows["shuffled"], sessions
        )
        for name in ("nearest_atm", "exact_random"):
            contrasts[f"{candidate}_vs_{name}"] = _paired_delta(
                pair_sets[f"{candidate}_vs_{name}"], sessions
            )
    contrasts["full25_vs_signed18"] = _policy_delta(
        policy_rows["full25"], policy_rows["signed18"], sessions
    )
    corrected = session_blocked_max_t(contrasts, permutations=PERMUTATIONS, seed=307)
    serial = {}
    for candidate in CANDIDATES:
        for control in ("deterministic", "shuffled"):
            serial[f"{candidate}_vs_{control}"] = strict_serial(
                policy_rows[candidate],
                policy_rows[control],
                sessions=sessions,
                session_to_fold=session_to_fold,
                variant=f"{candidate}_vs_{control}",
            )
        for control in ("nearest_atm", "exact_random"):
            serial[f"{candidate}_vs_{control}"] = _paired_serial(
                pair_sets[f"{candidate}_vs_{control}"],
                sessions=sessions,
                session_to_fold=session_to_fold,
                arm_name=candidate,
                baseline_name=control,
            )
    serial["full25_vs_signed18"] = strict_serial(
        policy_rows["full25"],
        policy_rows["signed18"],
        sessions=sessions,
        session_to_fold=session_to_fold,
        variant="full25_vs_signed18",
    )
    contrast_status = {
        name: _contrast_status(corrected[name], serial[name]) for name in corrected
    }
    candidate_status = {}
    for candidate in CANDIDATES:
        required = [contrast_status[f"{candidate}_vs_{name}"] for name in REQUIRED_CONTROLS]
        absolute = serial[f"{candidate}_vs_deterministic"]
        candidate_status[candidate] = (
            "PROVISIONAL_EDGE"
            if all(status == "PROVISIONAL_EDGE" for status in required)
            and int(absolute["positive_arm_folds"]) >= 4
            and float(absolute["arm"]["total_pnl"]) > 0.0
            else (
                "UNDERPOWERED" if any(status == "UNDERPOWERED" for status in required)
                else "NO_INCREMENTAL_EDGE"
            )
        )
    winner = (
        "signed18"
        if candidate_status["signed18"] == "PROVISIONAL_EDGE"
        else "full25" if candidate_status["full25"] == "PROVISIONAL_EDGE" else None
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "session_deltas.json", contrasts)
    write_json(output / "serial_results.json", serial)
    source_hash = engine_source_hash()
    for candidate in CANDIDATES:
        hypothesis_id = f"entry_block_first_{candidate}_short25_attribution_v1"
        result = {
            "schema_version": "autoresearch_v2.entry_attribution_result.v1",
            "hypothesis_id": hypothesis_id,
            "semantic_hash": compiled[hypothesis_id].semantic_hash,
            "engine_source_hash": source_hash,
            "foundation_sha256": foundation_payload["foundation_sha256"],
            "preregistration_sha256": prereg["preregistration_sha256"],
            "status": candidate_status[candidate],
            "evidence_grade": "development_only_non_promotable",
            "holdout_access_count": 0,
            "protected_holdout_opened": False,
            "candidate": candidate,
            "required_controls": {
                name: {
                    "status": contrast_status[f"{candidate}_vs_{name}"],
                    "screen": corrected[f"{candidate}_vs_{name}"],
                    "strict_serial": serial[f"{candidate}_vs_{name}"],
                }
                for name in REQUIRED_CONTROLS
            },
            "diagnostic_exact_random": {
                "status": contrast_status[f"{candidate}_vs_exact_random"],
                "screen": corrected[f"{candidate}_vs_exact_random"],
                "strict_serial": serial[f"{candidate}_vs_exact_random"],
            },
        }
        result_path = output / "results" / f"{hypothesis_id}.json"
        write_json(result_path, result)
        register(
            registry_path,
            compiled[hypothesis_id],
            status=candidate_status[candidate],
            result_path=str(result_path),
            engine_source_hash=source_hash,
        )
    suite = {
        "schema_version": "autoresearch_v2.entry_attribution_suite.v1",
        "status": "PROVISIONAL_EDGE" if winner else "NO_SURVIVOR",
        "winner": winner,
        "candidate_status": candidate_status,
        "contrast_status": contrast_status,
        "screens": corrected,
        "foundation_sha256": foundation_payload["foundation_sha256"],
        "foundation_sessions": foundation_payload["session_count"],
        "oof_sessions": len(sessions),
        "policy_signal_counts": {key: len(value) for key, value in policy_rows.items()},
        "fit_receipts": {"full": full_fit, "controls": control_fit},
        "holdout_access_count": 0,
        "protected_holdout_opened": False,
    }
    write_json(output / "suite_result.json", suite)
    (output / "report.md").write_text(
        "# Corrected-v3 block-first entry attribution\n\n"
        f"- Status: `{suite['status']}`\n"
        f"- Winner: `{winner}`\n"
        f"- Candidate statuses: `{candidate_status}`\n"
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
