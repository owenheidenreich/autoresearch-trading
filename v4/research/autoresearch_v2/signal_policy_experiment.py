"""Test the reduced learned-signal + deterministic-nearest-ATM entry policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .builtin import SHORT25_EXIT
from .cache import PredictionCache, stable_hash
from .compiler import CompiledHypothesis, load_and_compile
from .dataset import FEATURE_INVENTORY, SIGNED17, load_development_frame, rolling_folds
from .entry_attribution_experiment import (
    deterministic_policy,
    fit_controls,
    _opportunities_for,
    _policy_delta,
)
from .registry import DuplicateSemanticHypothesis, read_registry, register
from .runner import clean, engine_source_hash, write_json
from .schema import SCHEMA_VERSION, TERMINAL_STATUSES
from .statistics import session_blocked_max_t
from .timing_policy_experiment import (
    EPOCH_ID,
    MICRO_FEATURES,
    MODEL_FEATURES,
    MODEL_PARAMETERS,
    PERMUTATIONS,
    TIME_FEATURES,
    _derived,
    fit_oof as fit_full_oof,
    strict_serial,
)


PRACTICAL_EFFECT = 100.0
MAXIMUM_MDE = 250.0
CANDIDATES = (
    "signed18_model_side_nearest",
    "signed18_momentum_side_nearest",
    "full25_momentum_side_nearest",
)
REQUIRED_CONTROLS = ("deterministic", "shuffled_momentum_side_nearest")


def _feature_specs(names: Sequence[str]) -> list[dict[str, str]]:
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
    full = candidate.startswith("full25")
    names = MODEL_FEATURES if full else tuple(SIGNED17)
    families = ["signed17", "live_safe_microstructure", "causal_clock"] if full else ["signed17"]
    return {
        "schema_version": SCHEMA_VERSION,
        "hypothesis_id": f"entry_signal_policy_{candidate}_short25_v1",
        "claim": (
            f"The {candidate} causal block-first signal with deterministic nearest-ATM contract "
            "selection improves strict one-account dollars over deterministic and shuffled timing controls."
        ),
        "component": "entry_policy",
        "feature_families": {"add": families, "remove": ["model_contract_ranking"]},
        "features": _feature_specs(names),
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
        "arms": [candidate, *REQUIRED_CONTROLS],
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
        "compute_budget": {"max_permutations": PERMUTATIONS, "max_minutes": 60},
        "terminal_statuses": list(TERMINAL_STATUSES),
    }


def initialize(hypotheses: Path, preregistration: Path) -> dict[str, Any]:
    if hypotheses.exists() or preregistration.exists():
        raise FileExistsError("signal policy preregistration already exists")
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
        "schema_version": "autoresearch_v2.signal_policy_preregistration.v1",
        "status": "FROZEN_BEFORE_RESULTS",
        "development_observation": (
            "Prior corrected attribution found timing/side policy value but no contract-ranking value; "
            "this family removes learned contract ranking."
        ),
        "epoch_id": EPOCH_ID,
        "frozen_exit": SHORT25_EXIT,
        "candidate_policies": list(CANDIDATES),
        "contract_rule": "nearest absolute moneyness at signal time on the selected side",
        "side_rules": {
            "model_side": "side of the highest OOF candidate at the first qualifying block decision",
            "momentum_side": "call iff completed-minute momentum_15m_bps>=0 else put",
        },
        "required_controls": {
            "deterministic": "first decision per fixed block, momentum side, nearest ATM",
            "shuffled_momentum_side_nearest": (
                "first qualifying within-training-session shuffled-target signal per block, momentum side, nearest ATM"
            ),
        },
        "family": [
            f"{candidate}_vs_{control}" for candidate in CANDIDATES for control in REQUIRED_CONTROLS
        ] + [
            "signed18_model_side_nearest_vs_signed18_momentum_side_nearest",
            "full25_momentum_side_nearest_vs_signed18_momentum_side_nearest",
        ],
        "qualification": {
            "practical_effect_dollars_per_session": PRACTICAL_EFFECT,
            "maximum_mde_dollars_per_session": MAXIMUM_MDE,
            "required_for_each_control": [
                "family_maxT_p<=0.05",
                "screen_mean_delta>=100",
                "screen_MDE80<=250",
                "strict_serial_mean_delta>=100",
                "strict_serial_one_sided_p<=0.05",
                "strict_serial_MDE80<=250",
                "strict_serial_delta_positive_in_at_least_4_of_5_folds",
            ],
            "absolute_candidate": "positive strict-serial PnL in at least 4 of 5 folds and total PnL>0",
            "winner_order": list(CANDIDATES),
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
        raise RuntimeError("signal policy preregistration drifted")
    return payload


def nearest_policy(
    oof: pd.DataFrame, signals: pd.DataFrame, *, side_rule: str
) -> pd.DataFrame:
    groups = {
        (str(session), int(decision)): group
        for (session, decision), group in oof.groupby(["session", "decision_time_ns"], sort=False)
    }
    selected = []
    for signal in signals.to_dict("records"):
        group = groups[(str(signal["session"]), int(signal["decision_time_ns"]))]
        if side_rule == "model":
            side = str(signal["right"])
        elif side_rule == "momentum":
            side = "C" if float(signal["momentum_15m_bps"]) >= 0.0 else "P"
        else:
            raise ValueError(side_rule)
        pool = group[group["right"] == side].sort_values(["abs_moneyness", "candidate_uid"])
        if pool.empty:
            continue
        row = pool.iloc[0].to_dict()
        row["_pred"] = float(signal["_pred"])
        row["_block"] = int(signal["_block"])
        selected.append(row)
    return pd.DataFrame(selected)


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
    expected = {f"entry_signal_policy_{candidate}_short25_v1" for candidate in CANDIDATES}
    if set(compiled) != expected:
        raise RuntimeError("signal policy family membership drifted")
    prior = {row["semantic_hash"] for row in read_registry(registry_path)}
    if prior & {item.semantic_hash for item in compiled.values()}:
        raise DuplicateSemanticHypothesis("signal policy semantic duplicate")
    frame, foundation_payload = load_development_frame(foundation)
    if foundation_payload.get("generation") != "corrected-v3.2-distinct-two-clock":
        raise RuntimeError("signal policy requires corrected-v3 foundation")
    frame = _derived(frame)
    folds = rolling_folds(frame["session"].unique())
    cache = PredictionCache(cache_root)
    full_oof, full_fit = fit_full_oof(
        frame, folds=folds, foundation_hash=foundation_payload["foundation_sha256"], cache=cache
    )
    controls_oof, control_fit = fit_controls(
        frame, folds=folds, foundation_hash=foundation_payload["foundation_sha256"], cache=cache
    )
    oof = full_oof.merge(
        controls_oof[["candidate_uid", "_signed_pred", "_shuffled_pred"]],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    sessions = sorted(str(value) for value in oof["session"].unique())
    session_to_fold = {
        str(row["session"]): int(row["fold"])
        for _, row in oof[["session", "fold"]].drop_duplicates().iterrows()
    }
    signed_signals = _opportunities_for(oof, "_signed_pred")
    full_signals = _opportunities_for(oof, "_ref_pred")
    shuffled_signals = _opportunities_for(oof, "_shuffled_pred")
    policies = {
        "signed18_model_side_nearest": nearest_policy(oof, signed_signals, side_rule="model").to_dict("records"),
        "signed18_momentum_side_nearest": nearest_policy(oof, signed_signals, side_rule="momentum").to_dict("records"),
        "full25_momentum_side_nearest": nearest_policy(oof, full_signals, side_rule="momentum").to_dict("records"),
        "shuffled_momentum_side_nearest": nearest_policy(oof, shuffled_signals, side_rule="momentum").to_dict("records"),
        "deterministic": deterministic_policy(oof).to_dict("records"),
    }
    contrasts = {
        f"{candidate}_vs_{control}": _policy_delta(policies[candidate], policies[control], sessions)
        for candidate in CANDIDATES
        for control in REQUIRED_CONTROLS
    }
    contrasts["signed18_model_side_nearest_vs_signed18_momentum_side_nearest"] = _policy_delta(
        policies["signed18_model_side_nearest"], policies["signed18_momentum_side_nearest"], sessions
    )
    contrasts["full25_momentum_side_nearest_vs_signed18_momentum_side_nearest"] = _policy_delta(
        policies["full25_momentum_side_nearest"], policies["signed18_momentum_side_nearest"], sessions
    )
    corrected = session_blocked_max_t(contrasts, permutations=PERMUTATIONS, seed=401)
    serial = {}
    for name in contrasts:
        arm_name, baseline_name = name.split("_vs_", 1)
        serial[name] = strict_serial(
            policies[arm_name],
            policies[baseline_name],
            sessions=sessions,
            session_to_fold=session_to_fold,
            variant=name,
        )
    contrast_status = {name: _status(corrected[name], serial[name]) for name in contrasts}
    candidate_status = {}
    for candidate in CANDIDATES:
        required = [contrast_status[f"{candidate}_vs_{control}"] for control in REQUIRED_CONTROLS]
        absolute = serial[f"{candidate}_vs_deterministic"]
        candidate_status[candidate] = (
            "PROVISIONAL_EDGE"
            if all(status == "PROVISIONAL_EDGE" for status in required)
            and int(absolute["positive_arm_folds"]) >= 4
            and float(absolute["arm"]["total_pnl"]) > 0.0
            else "UNDERPOWERED" if any(status == "UNDERPOWERED" for status in required)
            else "NO_INCREMENTAL_EDGE"
        )
    winner = next(
        (candidate for candidate in CANDIDATES if candidate_status[candidate] == "PROVISIONAL_EDGE"),
        None,
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "session_deltas.json", contrasts)
    write_json(output / "serial_results.json", serial)
    source_hash = engine_source_hash()
    for candidate in CANDIDATES:
        hypothesis_id = f"entry_signal_policy_{candidate}_short25_v1"
        result = {
            "schema_version": "autoresearch_v2.signal_policy_result.v1",
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
                control: {
                    "status": contrast_status[f"{candidate}_vs_{control}"],
                    "screen": corrected[f"{candidate}_vs_{control}"],
                    "strict_serial": serial[f"{candidate}_vs_{control}"],
                }
                for control in REQUIRED_CONTROLS
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
        "schema_version": "autoresearch_v2.signal_policy_suite.v1",
        "status": "PROVISIONAL_EDGE" if winner else "NO_SURVIVOR",
        "winner": winner,
        "candidate_status": candidate_status,
        "contrast_status": contrast_status,
        "screens": corrected,
        "foundation_sha256": foundation_payload["foundation_sha256"],
        "foundation_sessions": foundation_payload["session_count"],
        "oof_sessions": len(sessions),
        "policy_signal_counts": {name: len(rows) for name, rows in policies.items()},
        "fit_receipts": {"full": full_fit, "controls": control_fit},
        "holdout_access_count": 0,
        "protected_holdout_opened": False,
    }
    write_json(output / "suite_result.json", suite)
    (output / "report.md").write_text(
        "# Reduced learned-signal entry policy\n\n"
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
