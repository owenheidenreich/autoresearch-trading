"""Expand the exact policy-neutral M0/M1 path target across development."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_policy_neutral_selector import (
    M0_FEATURES,
    RAW_TARGET_COLUMNS,
    candidate_order,
    deterministic_group_sample,
    group_balanced_weights,
    select_position,
)
from v4.scripts.run_protocol101_policy_neutral_contract_selector import (
    CAMPAIGN_ID,
    SESSION_ROOT as PRIOR_SESSION_ROOT,
    _scope_maps,
    _session_frame,
)

from .builtin import REFERENCE_EXIT
from .cache import PredictionCache, prediction_cache_key, stable_hash
from .causality import assert_mutate_future_invariant
from .compiler import load_and_compile
from .dataset import ROOT, load_development_frame, rolling_folds, sha256_path, verify_foundation
from .registry import DuplicateSemanticHypothesis, read_registry, register
from .replay import paired_lockstep_component_replay
from .runner import clean, engine_source_hash, write_json
from .statistics import paired_summary, session_blocked_max_t


HYPOTHESIS_ID = "entry_policy_neutral_m0_m1_full_epoch_v1"
MODEL_CONFIG = {
    "loss": "squared_error",
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_depth": 3,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 101,
}
MAX_TRAINING_CANDIDATES = 350_000
UTILITY_EFFECT_BAR = 0.01
SERIAL_EFFECT_BAR_DOLLARS = 10.0
SERIAL_MAX_MDE_DOLLARS = 25.0


def _normalized_sources(foundation: Mapping[str, Any]) -> dict[str, Path]:
    _, _, normalized = _scope_maps()
    sources: dict[str, Path] = {}
    for row in foundation["sessions"]:
        session = str(row["session"])
        receipt = json.loads((ROOT / row["receipt_path"]).read_text())
        path = Path(normalized[session])
        observed = sha256_path(path)
        expected = receipt["source_hashes"]["normalized_official_context"]
        if observed != expected:
            raise RuntimeError(f"normalized path foundation drifted: {session}")
        sources[session] = path
    return sources


def materialize_risk_sets(
    *, foundation_path: Path, cache_root: Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    foundation = verify_foundation(foundation_path)
    normalized = _normalized_sources(foundation)
    source_rows: list[dict[str, Any]] = []
    paths: dict[str, Path] = {}
    cache_root.mkdir(parents=True, exist_ok=True)
    for index, foundation_row in enumerate(foundation["sessions"], start=1):
        session = str(foundation_row["session"])
        prior_path = PRIOR_SESSION_ROOT / f"{session}.parquet"
        prior_receipt_path = PRIOR_SESSION_ROOT / f"{session}.json"
        cache_path = cache_root / f"{session}.parquet"
        cache_receipt_path = cache_root / f"{session}.json"
        reused_prior = False
        if prior_path.is_file() and prior_receipt_path.is_file():
            receipt = json.loads(prior_receipt_path.read_text())
            if receipt.get("parquet_sha256") == sha256_path(prior_path):
                path = prior_path
                reused_prior = True
            else:
                raise RuntimeError(f"prior risk-set receipt drifted: {session}")
        elif cache_path.is_file() and cache_receipt_path.is_file():
            receipt = json.loads(cache_receipt_path.read_text())
            if (
                receipt.get("parquet_sha256") == sha256_path(cache_path)
                and receipt.get("two_clock_sha256") == foundation_row["sha256"]
                and receipt.get("normalized_sha256") == sha256_path(normalized[session])
            ):
                path = cache_path
            else:
                raise RuntimeError(f"cached risk-set receipt drifted: {session}")
        else:
            frame, receipt = _session_frame(
                session=session,
                processed_path=ROOT / foundation_row["path"],
                normalized_path=normalized[session],
                fold_id="autoresearch_v2_development",
                split="development",
                campaign_id=CAMPAIGN_ID,
                require_fixed_p5_opportunity=False,
            )
            temporary = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
            frame.to_parquet(temporary, index=False, compression="zstd")
            os.replace(temporary, cache_path)
            receipt = {
                **receipt,
                "two_clock_sha256": foundation_row["sha256"],
                "normalized_sha256": sha256_path(normalized[session]),
                "parquet_sha256": sha256_path(cache_path),
            }
            write_json(cache_receipt_path, receipt)
            path = cache_path
        paths[session] = path
        source_rows.append(
            {
                "session": session,
                "risk_set_path": str(path),
                "risk_set_sha256": sha256_path(path),
                "normalized_path": str(normalized[session]),
                "normalized_sha256": sha256_path(normalized[session]),
                "two_clock_sha256": foundation_row["sha256"],
                "reused_prior_campaign_risk_set": reused_prior,
            }
        )
        if index % 10 == 0 or index == len(foundation["sessions"]):
            print(
                json.dumps(
                    {
                        "phase": "path_target_materialization",
                        "completed_sessions": index,
                        "total_sessions": len(foundation["sessions"]),
                    }
                ),
                flush=True,
            )
    manifest = {
        "schema_version": "autoresearch_v2.path_target_foundation.v1",
        "role": "development",
        "holdout_access_count": 0,
        "two_clock_foundation_sha256": foundation["foundation_sha256"],
        "session_count": len(source_rows),
        "sources": source_rows,
    }
    manifest["path_target_foundation_sha256"] = stable_hash(manifest)
    return paths, manifest


def _load_risk_frame(paths: Mapping[str, Path]) -> pd.DataFrame:
    columns = [
        "session",
        "decision_time_ns",
        "contract_id",
        "right",
        "offset",
        "strike_index",
        "entry_ask",
        "primary_utility",
        *FEATURE_NAMES,
        *RAW_TARGET_COLUMNS,
    ]
    frames = [pd.read_parquet(path, columns=columns) for _, path in sorted(paths.items())]
    frame = pd.concat(frames, ignore_index=True)
    frame["candidate_uid"] = (
        frame["session"].astype(str)
        + "|"
        + frame["decision_time_ns"].astype(str)
        + "|"
        + frame["contract_id"].astype(str)
    )
    if frame.duplicated(["session", "decision_time_ns", "contract_id"]).any():
        raise RuntimeError("path risk-set candidate identity duplicated")
    return frame


def _fit_oof_models(
    frame: pd.DataFrame,
    *,
    folds: Sequence[Mapping[str, Any]],
    foundation_hash: str,
    cache_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache = PredictionCache(cache_root)
    outputs = []
    receipts = []
    model_features = {"M0": tuple(M0_FEATURES), "M1": tuple(FEATURE_NAMES)}
    future_columns = [*RAW_TARGET_COLUMNS, "primary_utility"]
    for fold in folds:
        train = frame[frame["session"].isin(fold["model_fit"])].copy()
        test = frame[frame["session"].isin(fold["outer_test"])].copy()
        train = train[np.isfinite(train["primary_utility"].to_numpy(float))]
        test = test[np.isfinite(test["primary_utility"].to_numpy(float))]
        test = test.sort_values("candidate_uid").reset_index(drop=True)
        row_hash = stable_hash(test["candidate_uid"].tolist())
        fold_receipts = []
        for model_name, features in model_features.items():
            sampled = deterministic_group_sample(
                train,
                maximum_candidates=MAX_TRAINING_CANDIDATES,
                fold_id=f"autoresearch_v2_{model_name}_{fold['fold']}",
            ).reset_index(drop=True)
            key = prediction_cache_key(
                features=features,
                target={"kind": "policy_neutral_15m_path_utility", "field": "primary_utility"},
                model={"name": model_name, "config": MODEL_CONFIG},
                fold=dict(fold),
                seed=101,
                foundation_hash=foundation_hash,
            )
            cached = cache.load(key, row_hash=row_hash)
            cache_hit = cached is not None
            if cached is None:
                model = HistGradientBoostingRegressor(**MODEL_CONFIG)
                model.fit(
                    sampled.loc[:, features].to_numpy(float),
                    sampled["primary_utility"].to_numpy(float),
                    sample_weight=group_balanced_weights(sampled),
                )
                prediction = model.predict(test.loc[:, features].to_numpy(float))
            else:
                prediction, model = cached
            invariance = assert_mutate_future_invariant(
                test,
                feature_columns=features,
                future_columns=future_columns,
                scorer=lambda values, fitted=model: fitted.predict(values.to_numpy(float)),
            )
            if not cache_hit:
                cache.store(
                    key,
                    prediction,
                    row_hash=row_hash,
                    model=model,
                    mutate_future_invariance_status=str(invariance["status"]),
                )
            test[f"_{model_name}_pred"] = np.asarray(prediction, dtype=float)
            fold_receipts.append(
                {
                    "model": model_name,
                    "fold": int(fold["fold"]),
                    "train_sessions": len(fold["model_fit"]),
                    "test_sessions": len(fold["outer_test"]),
                    "source_train_rows": len(train),
                    "sampled_train_rows": len(sampled),
                    "test_rows": len(test),
                    "cache_key": key,
                    "cache_hit": cache_hit,
                    "mutate_future_invariance": invariance,
                }
            )
        test["fold"] = int(fold["fold"])
        outputs.append(test)
        receipts.extend(fold_receipts)
    return pd.concat(outputs, ignore_index=True), {
        "models": receipts,
        "fit_count": sum(not row["cache_hit"] for row in receipts),
        "cache_hit_count": sum(row["cache_hit"] for row in receipts),
    }


def _local_random_position(session: str, decision: int, count: int) -> int:
    seed = int.from_bytes(
        hashlib.sha256(f"autoresearch_v2|exact_random|{session}|{decision}".encode()).digest()[:8],
        "big",
    )
    return int(np.random.Generator(np.random.PCG64DXSM(seed)).integers(0, count))


def _select(oof: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    selected = []
    deltas: dict[str, dict[str, list[float]]] = {
        name: {}
        for name in ("M1_vs_M0", "M1_vs_nearest_atm", "M1_vs_exact_random", "M0_vs_nearest_atm", "M0_vs_exact_random")
    }
    for (session, decision), group in oof.groupby(["session", "decision_time_ns"], sort=True):
        group = group.reset_index(drop=True)
        positions = {
            "M0": select_position(group, group["_M0_pred"].to_numpy(float)),
            "M1": select_position(group, group["_M1_pred"].to_numpy(float)),
            "nearest_atm": int(candidate_order(group)[0]),
            "exact_random": _local_random_position(str(session), int(decision), len(group)),
        }
        records = {name: group.iloc[position] for name, position in positions.items()}
        selected.append(
            {
                "session": str(session),
                "decision_time_ns": int(decision),
                **{
                    f"{name}_{field}": records[name][field]
                    for name in records
                    for field in ("contract_id", "right", "offset", "entry_ask", "primary_utility")
                },
                "M0_score": float(records["M0"]["_M0_pred"]),
                "M1_score": float(records["M1"]["_M1_pred"]),
            }
        )
        utilities = {name: float(row["primary_utility"]) for name, row in records.items()}
        values = {
            "M1_vs_M0": utilities["M1"] - utilities["M0"],
            "M1_vs_nearest_atm": utilities["M1"] - utilities["nearest_atm"],
            "M1_vs_exact_random": utilities["M1"] - utilities["exact_random"],
            "M0_vs_nearest_atm": utilities["M0"] - utilities["nearest_atm"],
            "M0_vs_exact_random": utilities["M0"] - utilities["exact_random"],
        }
        for name, value in values.items():
            deltas[name].setdefault(str(session), []).append(value)
    session_deltas = {
        name: {session: float(np.mean(values)) for session, values in by_session.items()}
        for name, by_session in deltas.items()
    }
    return pd.DataFrame(selected), session_deltas


def _status(summary: Mapping[str, Any]) -> str:
    if summary.get("ci_high") is not None and float(summary["ci_high"]) < UTILITY_EFFECT_BAR:
        return "NO_INCREMENTAL_EDGE"
    if float(summary.get("mde80") or float("inf")) > UTILITY_EFFECT_BAR:
        return "UNDERPOWERED"
    if (
        float(summary.get("mean") or 0.0) < UTILITY_EFFECT_BAR
        or float(summary.get("maxT_p_one_sided") or 1.0) > 0.05
    ):
        return "NO_INCREMENTAL_EDGE"
    return "PROVISIONAL_EDGE"


def run(
    *,
    foundation: Path,
    hypothesis_path: Path,
    output: Path,
    risk_cache: Path,
    prediction_cache: Path,
    registry_path: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    compiled = load_and_compile(hypothesis_path)
    if compiled.spec.hypothesis_id != HYPOTHESIS_ID:
        raise RuntimeError("wrong typed hypothesis supplied to path experiment")
    if any(row["semantic_hash"] == compiled.semantic_hash for row in read_registry(registry_path)):
        raise DuplicateSemanticHypothesis("path hypothesis already exists in global registry")
    paths, path_foundation = materialize_risk_sets(
        foundation_path=foundation, cache_root=risk_cache
    )
    write_json(output / "path_target_foundation.json", path_foundation)
    frame = _load_risk_frame(paths)
    folds = rolling_folds(frame["session"].unique())
    oof, fit_receipt = _fit_oof_models(
        frame,
        folds=folds,
        foundation_hash=path_foundation["path_target_foundation_sha256"],
        cache_root=prediction_cache,
    )
    selected, contrasts = _select(oof)
    corrected = session_blocked_max_t(contrasts, permutations=4999, seed=101)
    status = _status(corrected["M1_vs_M0"])
    serial: dict[str, Any] = {
        "routed": False,
        "reason": "M1_vs_M0 did not survive the exact path-utility screen",
    }
    if status == "PROVISIONAL_EDGE":
        development, _ = load_development_frame(foundation)
        lookup = {
            (str(row["session"]), int(row["decision_time_ns"]), str(row["contract_id"])): row
            for row in development.to_dict("records")
        }
        pairs = []
        for row in selected.to_dict("records"):
            arm = lookup.get((row["session"], row["decision_time_ns"], row["M1_contract_id"]))
            baseline = lookup.get((row["session"], row["decision_time_ns"], row["M0_contract_id"]))
            if arm is None or baseline is None:
                continue
            if arm["right"] != baseline["right"] or abs(arm["entry_ask"] - baseline["entry_ask"]) > 1.5:
                continue
            arm["_pred"] = float(row["M1_score"])
            baseline["_pred"] = float(row["M0_score"])
            pairs.append((f"{row['session']}|{row['decision_time_ns']}", arm, baseline))
        replay = paired_lockstep_component_replay(
            pairs,
            policy=REFERENCE_EXIT,
            arm_name="M1",
            baseline_name="M0",
        )
        serial_stats = paired_summary(replay["session_deltas"])
        serial = {
            "routed": True,
            "statistics": serial_stats,
            "paired": replay,
        }
        if float(serial_stats.get("mde80") or float("inf")) > SERIAL_MAX_MDE_DOLLARS:
            status = "UNDERPOWERED"
        elif not (
            float(serial_stats.get("mean") or 0.0) >= SERIAL_EFFECT_BAR_DOLLARS
            and float(serial_stats.get("p_one_sided") or 1.0) <= 0.05
        ):
            status = "NO_INCREMENTAL_EDGE"
    source_hash = engine_source_hash()
    result = {
        "schema_version": "autoresearch_v2.path_experiment_result.v1",
        "hypothesis_id": HYPOTHESIS_ID,
        "semantic_hash": compiled.semantic_hash,
        "engine_source_hash": source_hash,
        "status": status,
        "evidence_grade": "development_only_non_promotable",
        "holdout_access_count": 0,
        "protected_holdout_opened": False,
        "path_target": {
            "raw_components": list(RAW_TARGET_COLUMNS),
            "primary": "decision-local equal-weight percentile utility",
            "unit": "utility_points_per_opportunity_averaged_within_session",
            "practical_effect": UTILITY_EFFECT_BAR,
        },
        "foundation": path_foundation,
        "fit_receipt": fit_receipt,
        "family_correction": {
            "method": "session_blocked_sign_flip_maxT",
            "permutations": 4999,
            "family": sorted(contrasts),
        },
        "screen": corrected,
        "serial_replay": serial,
    }
    write_json(output / "compiled_hypothesis.json", compiled.payload())
    write_json(output / "session_deltas.json", contrasts)
    write_json(output / "selected_opportunities.json", selected.to_dict("records"))
    write_json(output / "result.json", result)
    register(
        registry_path,
        compiled,
        status=status,
        result_path=str(output / "result.json"),
        engine_source_hash=source_hash,
    )
    (output / "report.md").write_text(
        "# Exact policy-neutral M0/M1 full-epoch result\n\n"
        f"- Status: `{status}`\n"
        f"- Sessions: `{path_foundation['session_count']}`\n"
        f"- M1 vs M0 mean utility: `{corrected['M1_vs_M0']['mean']}`\n"
        f"- M1 vs M0 maxT p: `{corrected['M1_vs_M0']['maxT_p_one_sided']}`\n"
        f"- M1 vs M0 MDE80: `{corrected['M1_vs_M0']['mde80']}`\n"
        "- Holdout opens: `0`\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation", type=Path, required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-cache", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        foundation=args.foundation,
        hypothesis_path=args.hypothesis,
        output=args.output,
        risk_cache=args.risk_cache,
        prediction_cache=args.prediction_cache,
        registry_path=args.registry,
    )
    print(json.dumps(clean(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
