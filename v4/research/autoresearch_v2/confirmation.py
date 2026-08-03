"""One-shot confirmation for the frozen autoresearch_v2 entry-policy winner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.dataset.spxw_0dte_neural import NeuralDatasetConfig
from v4.model.protocol101_regimen_repair import TWO_CLOCK_PROCESSED_ROW_SCHEMA
from v4.scripts.materialize_protocol101_ft1d_two_clock_rows import _attach_two_clock_labels

from .builtin import SHORT25_EXIT
from .cache import stable_hash
from .corrected_v3_foundation import (
    CORRECTED_ROOT,
    SESSION_MANIFEST,
    _atomic_json,
    _atomic_pickle,
    _validate_additive_parity,
)
from .dataset import ROOT, SIGNED17, _flatten_session, load_development_frame, sha256_path, target_column
from .entry_attribution_experiment import deterministic_policy, _opportunities_for
from .replay import replay_absolute, serial_summary
from .runner import clean, engine_source_hash, write_json
from .signal_policy_experiment import nearest_policy
from .statistics import paired_summary
from .timing_policy_experiment import MODEL_PARAMETERS, _derived


WINNER = "signed18_model_side_nearest"
DEVELOPMENT_FOUNDATION = ROOT / (
    "v4/research/autoresearch_v2/foundations/"
    "development_2025-08-01_2026-06-09_corrected_v3_two_clock.json"
)
DEVELOPMENT_PACKET = ROOT / (
    "v4/audit/autoresearch/"
    "autoresearch_v2_corrected_v3_signal_policy_2026_08_02_attempt001"
)
PRACTICAL_POINT_BAR = 100.0
ALPHA = 0.05
MIN_POSITIVE_SESSIONS = 15
MIN_TRADES = 100
PERMUTATIONS = 99_999


def _confirmation_sessions() -> tuple[str, ...]:
    assignments = json.loads(SESSION_MANIFEST.read_text())
    sessions = tuple(str(value) for value in assignments["protected_holdout_primary_non_degraded_29"])
    if len(sessions) != 29 or len(set(sessions)) != 29:
        raise RuntimeError("protected confirmation session set drifted")
    if set(sessions) & set(assignments["pre_holdout_session_indices_1_215"]):
        raise RuntimeError("confirmation sessions overlap development")
    if set(sessions) & set(assignments.get("burned_smoke_dates", ())):
        raise RuntimeError("confirmation sessions overlap burned smoke dates")
    return sessions


def initialize(preregistration: Path) -> dict[str, Any]:
    if preregistration.exists():
        raise FileExistsError("confirmation preregistration already exists")
    suite = json.loads((DEVELOPMENT_PACKET / "suite_result.json").read_text())
    if suite.get("winner") != WINNER or suite.get("status") != "PROVISIONAL_EDGE":
        raise RuntimeError("no frozen provisional winner")
    serial = json.loads((DEVELOPMENT_PACKET / "serial_results.json").read_text())[
        f"{WINNER}_vs_deterministic"
    ]
    stats = serial["statistics"]
    session_count = len(_confirmation_sessions())
    standardized = float(stats["mean"]) * math.sqrt(session_count) / float(stats["std"])
    estimated_power = float(norm.cdf(standardized - norm.ppf(1.0 - ALPHA)))
    if estimated_power < 0.80:
        raise RuntimeError("protected confirmation is underpowered; holdout remains sealed")
    winner_result = json.loads(
        (
            DEVELOPMENT_PACKET
            / "results"
            / "entry_signal_policy_signed18_model_side_nearest_short25_v1.json"
        ).read_text()
    )
    payload: dict[str, Any] = {
        "schema_version": "autoresearch_v2.confirmation_preregistration.v1",
        "status": "FROZEN_BEFORE_HOLDOUT_OPEN",
        "winner": WINNER,
        "winner_semantic_hash": winner_result["semantic_hash"],
        "development_foundation_sha256": suite["foundation_sha256"],
        "development_packet": str(DEVELOPMENT_PACKET.relative_to(ROOT)),
        "development_engine_source_hash": winner_result["engine_source_hash"],
        "candidate_contract": {
            "features": list(SIGNED17),
            "model": MODEL_PARAMETERS,
            "target": SHORT25_EXIT,
            "threshold": 0.0,
            "opportunity": "first qualifying OOF-equivalent score per six fixed ET blocks",
            "side": "side of highest-scoring candidate at signal decision",
            "contract": "nearest absolute moneyness on selected side",
            "exit": SHORT25_EXIT,
            "account": "protocol101_serial_simulator_v5 defaults",
        },
        "confirmation_sessions_sha256": stable_hash(list(_confirmation_sessions())),
        "confirmation_session_count": session_count,
        "holdout_open_count_before": 0,
        "holdout_open_count_allowed": 1,
        "power_check": {
            "alpha": ALPHA,
            "alternative_mean_from_development": stats["mean"],
            "standard_deviation_from_development": stats["std"],
            "estimated_power_against_zero": estimated_power,
            "required_power": 0.80,
        },
        "single_primary_test": {
            "contrast": f"{WINNER}_vs_deterministic",
            "permutation_method": "session_sign_flip_monte_carlo",
            "permutations": PERMUTATIONS,
            "seed": 509,
            "alpha": ALPHA,
            "pass_if_all": [
                "one_sided_permutation_p<=0.05",
                "paired_mean_dollars_per_session>=100",
                "positive_session_count>=15_of_29",
                "candidate_total_pnl>0",
                "candidate_trade_count>=100",
            ],
        },
        "terminal_rule": {
            "pass": "CONFIRMED_EDGE",
            "fail": "NO_INCREMENTAL_EDGE",
            "no_rescue_or_secondary_winner": True,
        },
    }
    payload["preregistration_sha256"] = stable_hash(payload)
    write_json(preregistration, payload)
    return payload


def _verify_preregistration(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    semantic = dict(payload)
    observed = semantic.pop("preregistration_sha256", None)
    if observed != stable_hash(semantic) or payload.get("status") != "FROZEN_BEFORE_HOLDOUT_OPEN":
        raise RuntimeError("confirmation preregistration drifted")
    if payload["confirmation_sessions_sha256"] != stable_hash(list(_confirmation_sessions())):
        raise RuntimeError("confirmation session identity drifted")
    return payload


def prepare(*, preregistration: Path, output: Path) -> dict[str, Any]:
    prereg = _verify_preregistration(preregistration)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "pre_holdout_model_freeze.json"
    model_path = output / "model" / "entry_model.pkl"
    if freeze_path.exists():
        frozen = json.loads(freeze_path.read_text())
        if sha256_path(model_path) != frozen["model_sha256"]:
            raise RuntimeError("pre-holdout model artifact drifted")
        return frozen
    frame, foundation = load_development_frame(DEVELOPMENT_FOUNDATION)
    if foundation["foundation_sha256"] != prereg["development_foundation_sha256"]:
        raise RuntimeError("development foundation differs from confirmation preregistration")
    target = target_column(SHORT25_EXIT)
    finite = np.isfinite(frame[target].to_numpy(float))
    model = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    model.fit(
        frame.loc[finite, list(SIGNED17)].to_numpy(float),
        frame.loc[finite, target].to_numpy(float),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, model_path)
    frozen: dict[str, Any] = {
        "schema_version": "autoresearch_v2.pre_holdout_model_freeze.v1",
        "status": "MODEL_FROZEN_BEFORE_HOLDOUT_OPEN",
        "winner": WINNER,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "foundation_sha256": foundation["foundation_sha256"],
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_path(model_path),
        "features": list(SIGNED17),
        "target": target,
        "training_rows": int(finite.sum()),
        "training_sessions": int(frame["session"].nunique()),
        "model_parameters": MODEL_PARAMETERS,
        "sklearn_version": sklearn_version,
        "engine_source_hash": engine_source_hash(),
        "holdout_access_count": 0,
    }
    frozen["freeze_sha256"] = stable_hash(frozen)
    write_json(freeze_path, frozen)
    return frozen


def _paths(output: Path, session: str) -> dict[str, Path]:
    aligned = CORRECTED_ROOT / "aligned"
    return {
        "legacy": aligned / "processed/minute_entry" / f"{session}.pkl",
        "normalized": aligned / "normalized" / f"databento_spxw_0dte_{session}_official_context.parquet",
        "output": output / "holdout_two_clock" / "processed" / f"{session}.pkl",
        "receipt": output / "holdout_two_clock" / "receipts" / f"{session}.json",
    }


def _materialize_confirmation_session(output: Path, session: str) -> dict[str, Any]:
    if session not in _confirmation_sessions():
        raise RuntimeError("attempted non-preregistered confirmation session")
    paths = _paths(output, session)
    if paths["output"].exists() and paths["receipt"].exists():
        receipt = json.loads(paths["receipt"].read_text())
        semantic = dict(receipt)
        observed = semantic.pop("receipt_sha256", None)
        if observed == stable_hash(semantic) and sha256_path(paths["output"]) == receipt["output_sha256"]:
            return receipt
        raise RuntimeError(f"confirmation materialization drifted:{session}")
    if paths["output"].exists() or paths["receipt"].exists():
        raise RuntimeError(f"partial confirmation materialization:{session}")
    for name in ("legacy", "normalized"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"confirmation source missing:{session}:{name}")
    with paths["legacy"].open("rb") as handle:
        legacy = pickle.load(handle)
    normalized = pq.read_table(paths["normalized"])
    config = NeuralDatasetConfig(
        feature_contract="protocol101-live-v2-microstructure-masked",
        compute_policy_labels=True,
        processed_row_schema_version=TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    )
    materialized = _attach_two_clock_labels(legacy, normalized, config=config)
    _validate_additive_parity(legacy, materialized, session=session)
    _atomic_pickle(paths["output"], materialized)
    receipt: dict[str, Any] = {
        "schema_version": "autoresearch_v2.confirmation_two_clock_receipt.v1",
        "status": "verified_confirmation_two_clock_materialization",
        "session": session,
        "role": "protected_confirmation",
        "holdout_access_count": 1,
        "source_hashes": {
            "legacy": sha256_path(paths["legacy"]),
            "normalized": sha256_path(paths["normalized"]),
            "session_manifest": sha256_path(SESSION_MANIFEST),
        },
        "output_path": str(paths["output"].relative_to(ROOT)),
        "output_sha256": sha256_path(paths["output"]),
        "row_count": len(materialized),
        "legacy_field_parity": True,
        "processed_row_schema_version": TWO_CLOCK_PROCESSED_ROW_SCHEMA,
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    _atomic_json(paths["receipt"], receipt)
    return receipt


def _session_pnl(trades: list[Any], sessions: tuple[str, ...]) -> dict[str, float]:
    result = {session: 0.0 for session in sessions}
    for trade in trades:
        result[str(trade.session)] += float(trade.raw_label_pnl_after_campaign_fee)
    return result


def _sign_flip_p(values: Mapping[str, float], *, permutations: int, seed: int) -> float:
    sample = np.asarray([float(value) for _, value in sorted(values.items())], dtype=float)
    observed = float(np.mean(sample))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        exceed += float(np.mean(sample * rng.choice((-1.0, 1.0), size=len(sample)))) >= observed
    return float((exceed + 1) / (permutations + 1))


def confirm(*, preregistration: Path, output: Path) -> dict[str, Any]:
    prereg = _verify_preregistration(preregistration)
    output = output.resolve()
    frozen = prepare(preregistration=preregistration, output=output)
    result_path = output / "confirmation_result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    access_path = output / "holdout_access_receipt.json"
    if access_path.exists():
        access = json.loads(access_path.read_text())
        if access.get("preregistration_sha256") != prereg["preregistration_sha256"]:
            raise RuntimeError("holdout access receipt belongs to another experiment")
    else:
        access = {
            "schema_version": "autoresearch_v2.holdout_access_receipt.v1",
            "status": "ACCESS_STARTED",
            "holdout_open_count": 1,
            "preregistration_sha256": prereg["preregistration_sha256"],
            "model_sha256": frozen["model_sha256"],
            "sessions_sha256": prereg["confirmation_sessions_sha256"],
        }
        access["receipt_sha256"] = stable_hash(access)
        _atomic_json(access_path, access)
    sessions = _confirmation_sessions()
    receipts = [_materialize_confirmation_session(output, session) for session in sessions]
    frames = [
        _flatten_session(session, _paths(output, session)["output"])
        for session in sessions
    ]
    frame = _derived(pd.concat(frames, ignore_index=True))
    with (ROOT / frozen["model_path"]).open("rb") as handle:
        model = pickle.load(handle)
    frame["_signed_pred"] = model.predict(frame.loc[:, list(SIGNED17)].to_numpy(float))
    signals = _opportunities_for(frame, "_signed_pred")
    candidate = nearest_policy(frame, signals, side_rule="model")
    deterministic = deterministic_policy(frame)
    candidate_rows = candidate.to_dict("records")
    deterministic_rows = deterministic.to_dict("records")
    arm_trades, arm_state = replay_absolute(
        candidate_rows, policy=SHORT25_EXIT, strategy="autoresearch_v2:confirmed_entry_model"
    )
    baseline_trades, baseline_state = replay_absolute(
        deterministic_rows, policy=SHORT25_EXIT, strategy="autoresearch_v2:confirmation_deterministic"
    )
    arm = _session_pnl(arm_trades, sessions)
    baseline = _session_pnl(baseline_trades, sessions)
    deltas = {session: arm[session] - baseline[session] for session in sessions}
    stats = paired_summary(deltas)
    permutation_p = _sign_flip_p(deltas, permutations=PERMUTATIONS, seed=509)
    positive_sessions = sum(value > 0.0 for value in deltas.values())
    passed = (
        permutation_p <= ALPHA
        and float(stats["mean"]) >= PRACTICAL_POINT_BAR
        and positive_sessions >= MIN_POSITIVE_SESSIONS
        and sum(arm.values()) > 0.0
        and len(arm_trades) >= MIN_TRADES
    )
    result: dict[str, Any] = {
        "schema_version": "autoresearch_v2.confirmation_result.v1",
        "status": "CONFIRMED_EDGE" if passed else "NO_INCREMENTAL_EDGE",
        "winner": WINNER,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "model_sha256": frozen["model_sha256"],
        "engine_source_hash": engine_source_hash(),
        "holdout_open_count": 1,
        "confirmation_session_count": len(sessions),
        "statistics": stats,
        "one_sided_sign_flip_p": permutation_p,
        "positive_session_count": positive_sessions,
        "candidate": serial_summary(arm_trades, arm_state),
        "deterministic_control": serial_summary(baseline_trades, baseline_state),
        "session_deltas": deltas,
        "signal_count": len(signals),
        "materialization_receipt_hashes": {
            str(item["session"]): str(item["receipt_sha256"]) for item in receipts
        },
        "pass_checks": {
            "permutation_p_le_0_05": permutation_p <= ALPHA,
            "mean_ge_100": float(stats["mean"]) >= PRACTICAL_POINT_BAR,
            "positive_sessions_ge_15": positive_sessions >= MIN_POSITIVE_SESSIONS,
            "candidate_total_pnl_positive": sum(arm.values()) > 0.0,
            "candidate_trades_ge_100": len(arm_trades) >= MIN_TRADES,
        },
    }
    write_json(result_path, result)
    complete = dict(access)
    complete.pop("receipt_sha256", None)
    complete["status"] = "ACCESS_COMPLETE"
    complete["result_status"] = result["status"]
    complete["result_sha256"] = sha256_path(result_path)
    complete["receipt_sha256"] = stable_hash(complete)
    _atomic_json(access_path, complete)
    (output / "report.md").write_text(
        "# One-shot protected confirmation\n\n"
        f"- Status: `{result['status']}`\n"
        f"- Mean delta/session: `${stats['mean']:.2f}`\n"
        f"- One-sided sign-flip p: `{permutation_p:.6f}`\n"
        f"- Positive sessions: `{positive_sessions}/{len(sessions)}`\n"
        f"- Candidate total PnL: `${sum(arm.values()):.2f}`\n"
        "- Holdout open count: `1`\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--preregistration", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--preregistration", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("confirm")
    run.add_argument("--preregistration", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        result = initialize(args.preregistration)
    elif args.command == "prepare":
        result = prepare(preregistration=args.preregistration, output=args.output)
    else:
        result = confirm(preregistration=args.preregistration, output=args.output)
    print(json.dumps(clean(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
