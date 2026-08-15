"""Run the owner-authorized Option-D walking-skeleton Stage-1 rerun.

This is a historical-only, capped, quarantined dry-run of the real FT2-10
entry plumbing.  TRADES and ABSTAINS are both complete outcomes.  The module
cannot start Stage 2, contact a broker, download data, or change a runtime or
promotion artifact.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from v4.model.protocol101_canonical_stage1_contract import FEATURE_NAMES
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    SerialSimulatorV5Config,
    simulate_serial_candidates_v5,
)
from v4.model.protocol101_walking_skeleton import (
    ACTION_HEADS,
    ARTIFACT_PREFIX,
    EXPECTED_GATE_HEADS,
    HGBConfig,
    HORIZONS,
    LABEL_PREFIX,
    PATH_HEADS,
    QUARANTINE_LABELS,
    apply_isotonic_state,
    attach_rlac_targets,
    build_nested_cdfs,
    causal_horizon_available,
    compose_decision,
    finite_sample_quantile,
    fit_isotonic_state,
    fit_regression_head,
    fit_wait_head,
    pooled_wait_frame,
    predict_regression,
    sha256_path,
    stable_hash,
    wait_feature_names,
)
from v4.scripts.run_protocol101_walking_skeleton_stages1_3 import (
    ROOT,
    _available_target_mask,
    _entry_rows_for_session,
    _prediction_columns,
    _processed_paths,
    _stage1_candidates,
    _summarize_entry_tensor,
    load_json,
    write_json_atomic,
)


STAGE0 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage0"
STAGE1 = ROOT / "v4/audit/autoresearch/protocol101_walking_skeleton_stage1"
PREREG = STAGE0 / "option_d_preregistration.json"
FIREWALL = STAGE0 / "option_d_firewall_proof.json"
STAGE0_RECEIPT = STAGE0 / "receipt.json"
CALIBRATION_SPEC = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_10_entry_science_contract/calibration_spec.json"
)
COMPOSER_SPEC = CALIBRATION_SPEC.with_name("composer_spec.json")
FORECAST_HEADS = CALIBRATION_SPEC.with_name("forecast_heads.json")
RLAC_SPEC = CALIBRATION_SPEC.with_name("realized_label_audit_composer_spec.json")
TENSOR_SCHEMA = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract/tensor_schema.json"
)
CONSISTENCY_CHECKER_OUTPUT = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_scoped_final_round_5_fixes_attempt001/"
    "consistency_checker_output.json"
)
AUTHORITY = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
AUTHORITY_SHA256 = "82d9573e120d6395825aa8a5f2d66fdac9bf32d825190737876b204dd112e2f2"
PRIOR_TENSOR_SHA256 = "16d93862bccc1ea83e052faff06d5792ad89205a571eb08b559af0dcc454193e"
FULL45_TENSOR_SHA256 = "31147c6b8dc9167b80911fe7ee0f6508fe620266ec29b79ff588a0cf076f411f"
PRIOR = (
    STAGE1 / "superseded/quick_gbt_all_wait_2026_07_31"
    / f"{ARTIFACT_PREFIX}entry_tensor.parquet"
)
PRE_A6_FULL45_TENSOR = (
    STAGE1 / "superseded/pre_a6_q10_gate_2026_07_31"
    / f"{ARTIFACT_PREFIX}option_d_full45_entry_tensor.parquet"
)
TENSOR = STAGE1 / f"{ARTIFACT_PREFIX}option_d_full45_entry_tensor.parquet"
CDF = STAGE1 / f"{ARTIFACT_PREFIX}option_d_fit_nested_cdfs.json"
RLAC = STAGE1 / f"{ARTIFACT_PREFIX}option_d_rlac_targets.parquet"
MODELS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_entry_models"
CHECKPOINTS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_checkpoints"
MANIFEST = STAGE1 / f"{ARTIFACT_PREFIX}option_d_entry_model_manifest.json"
CAL_PREDICTIONS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_calibration_predictions.parquet"
REPLAY_PREDICTIONS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_replay_predictions_label_stripped.parquet"
VARIANT_DECISIONS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_all_composer_variants.csv"
HEADLINE_DECISIONS = STAGE1 / f"{ARTIFACT_PREFIX}option_d_headline_decisions.csv"
TRADES = STAGE1 / f"{ARTIFACT_PREFIX}option_d_stage1_serial_trades.csv"
DIAGNOSTIC_JSON = STAGE1 / "diagnostic_packet.json"
DIAGNOSTIC_MD = STAGE1 / "diagnostic_packet.md"
RECEIPT = STAGE1 / "receipt.json"
REVIEW = STAGE1 / "delta_scoped_review.json"
JOURNAL = STAGE1 / "execution_journal.json"
IMPLEMENTATION_REPORT = (
    ROOT
    / "v4/audit/autoresearch/protocol101_ft2_10_entry_gate_convexity_amendment/"
    "implementation_report.md"
)

IDENTITY = ("session", "decision_time_ns", "contract_id")
INFERENCE_FORBIDDEN_EXACT = {
    "entry_ask",
    "fill_recheck_pass_at_tplus1",
    "hold_flat_exit_time_ns",
    "hold_flat_source_time_ns",
    "hold_flat_exit_bid",
    "hold_flat_exit_pnl",
    "u_label",
    "normalized_regret",
    "wait_target",
    "fee_cleared_label",
}
MODEL_CONFIG_ID = "option_d_hgb_c1"
SEEDS = (101, 102, 103)
WALL_CLOCK_CAP_SECONDS = 2_700.0
PHASE_MIN_SESSIONS = 8
PHASE_MIN_ROWS = 2_000
HEADLINE_ALPHA = 0.0
HEADLINE_K = 1.0
ALPHAS = (0.0, 0.1, 0.2, 0.25, 0.3)
KS = (1.0, 1.25, 1.5, 2.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = fields or sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _log(step: str, **details: Any) -> None:
    print(f"[option-d] {step}", flush=True)
    prior = load_json(JOURNAL) if JOURNAL.is_file() else {"events": []}
    prior["events"].append(
        {
            "step": step,
            "monotonic_elapsed_seconds": round(time.monotonic() - RUN_STARTED, 3),
            **details,
        }
    )
    _write_json(JOURNAL, prior)


def _verify_frozen_authority() -> dict[str, Any]:
    prereg = load_json(PREREG)
    firewall = load_json(FIREWALL)
    stage0 = load_json(STAGE0_RECEIPT)
    observed = {
        "authority": sha256_path(AUTHORITY),
        "calibration_spec": sha256_path(CALIBRATION_SPEC),
        "composer_spec": sha256_path(COMPOSER_SPEC),
        "forecast_heads": sha256_path(FORECAST_HEADS),
        "rlac_spec": sha256_path(RLAC_SPEC),
        "tensor_schema": sha256_path(TENSOR_SCHEMA),
        "consistency_checker_output": sha256_path(CONSISTENCY_CHECKER_OUTPUT),
    }
    expected = prereg["source_hashes"]
    mismatches = {
        key: {"expected": expected[key], "observed": value}
        for key, value in observed.items()
        if expected.get(key) != value
    }
    if observed["authority"] != AUTHORITY_SHA256:
        mismatches["authority_constant"] = {
            "expected": AUTHORITY_SHA256,
            "observed": observed["authority"],
        }
    if mismatches:
        raise RuntimeError(f"frozen Option-D authority drift: {mismatches}")
    if prereg.get("status") != "FROZEN_A6_BEFORE_FIT":
        raise RuntimeError("Option-D preregistration was not frozen before fit")
    if stage0.get("outcome") != "PASS_A6_FROZEN_STAGE1_RERUN_AUTHORIZED":
        raise RuntimeError("Stage-0 Option-D amendment receipt is not PASS")
    if firewall.get("status") != "PASS":
        raise RuntimeError("Option-D firewall proof is not PASS")
    config = prereg["model"]["configs"]
    if len(config) != 1 or config[0]["config_id"] != MODEL_CONFIG_ID:
        raise RuntimeError("Option-D preregistered configuration drift")
    return {"observed_hashes": observed, "preregistration": prereg, "firewall": firewall}


def _partition(prereg: dict[str, Any]) -> dict[str, list[str]]:
    source = prereg["partition"]
    return {
        "fit": list(source["fit_first_20_sessions"]),
        "embargo": list(source["embargo_next_1_session"]),
        "calibration": list(source["calibration_next_20_sessions"]),
        "replay": list(source["plumbing_replay_last_4_sessions"]),
    }


def _tensor_summary(frame: pd.DataFrame, sessions: list[str]) -> dict[str, Any]:
    return _summarize_entry_tensor(frame, sessions)


def _build_or_load_tensor(partition: dict[str, list[str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_sessions = [
        *partition["fit"],
        *partition["embargo"],
        *partition["calibration"],
        *partition["replay"],
    ]
    if TENSOR.is_file():
        frame = pd.read_parquet(TENSOR)
        observed = frame["session"].drop_duplicates().astype(str).tolist()
        if observed != all_sessions:
            raise RuntimeError("checkpointed full45 tensor session order drift")
        return frame, _tensor_summary(frame, all_sessions)
    if PRE_A6_FULL45_TENSOR.is_file():
        observed_hash = sha256_path(PRE_A6_FULL45_TENSOR)
        if observed_hash != FULL45_TENSOR_SHA256:
            raise RuntimeError(
                f"pre-A6 full45 tensor hash drift: {observed_hash}"
            )
        shutil.copy2(PRE_A6_FULL45_TENSOR, TENSOR)
        frame = pd.read_parquet(TENSOR)
        observed = frame["session"].drop_duplicates().astype(str).tolist()
        if observed != all_sessions:
            raise RuntimeError("pre-A6 full45 tensor session order drift")
        return frame, _tensor_summary(frame, all_sessions)
    if sha256_path(PRIOR) != PRIOR_TENSOR_SHA256:
        raise RuntimeError("superseded first30 tensor hash drift")
    prior = pd.read_parquet(PRIOR)
    prior_sessions = prior["session"].drop_duplicates().astype(str).tolist()
    if prior_sessions != all_sessions[:30]:
        raise RuntimeError("superseded tensor is not the exact frozen first30 prefix")
    paths = _processed_paths()
    blocks = [prior]
    for ordinal, session in enumerate(all_sessions[30:], start=31):
        _log("tensor_session", ordinal=ordinal, session=session)
        blocks.append(_entry_rows_for_session(session, paths[session]))
    frame = pd.concat(blocks, ignore_index=True)
    if frame.duplicated(list(IDENTITY)).any():
        raise RuntimeError("full45 tensor has duplicate exact-contract identity")
    if frame["session"].drop_duplicates().astype(str).tolist() != all_sessions:
        raise RuntimeError("full45 tensor chronological identity drift")
    frame.to_parquet(TENSOR, index=False)
    return frame, _tensor_summary(frame, all_sessions)


def _target_columns() -> list[str]:
    return list(dict.fromkeys(target for _, target, _ in PATH_HEADS))


def _build_or_load_cdfs(fit: pd.DataFrame) -> dict[str, Any]:
    if CDF.is_file():
        return load_json(CDF)
    columns = ["session", "decision_time_ns", "premium_band", "market_phase", *_target_columns()]
    source = fit.loc[:, columns].copy()
    for target in _target_columns():
        source.loc[~_available_target_mask(fit, target).to_numpy(bool), target] = np.nan
    _log("nested_cdf_build_started", target_count=len(_target_columns()))
    cdfs = build_nested_cdfs(source, _target_columns())
    _write_json(CDF, cdfs)
    del source
    gc.collect()
    return cdfs


def _attach_or_load_rlac(
    frames: dict[str, pd.DataFrame], cdfs: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    roles = ("fit", "calibration", "replay")
    if RLAC.is_file():
        targets = pd.read_parquet(RLAC)
        result: dict[str, pd.DataFrame] = {}
        for role in roles:
            result[role] = frames[role].merge(
                targets[targets["role"] == role].drop(columns="role"),
                on=list(IDENTITY),
                how="left",
                validate="one_to_one",
            )
        return result
    result = {}
    target_rows = []
    for role in roles:
        _log("rlac_target_build", role=role)
        result[role] = attach_rlac_targets(frames[role], cdfs)
        block = result[role][
            [*IDENTITY, "u_label", "fee_cleared_label", "wait_target", "normalized_regret"]
        ].copy()
        block["role"] = role
        target_rows.append(block)
    pd.concat(target_rows, ignore_index=True).to_parquet(RLAC, index=False)
    return result


def _config(seed: int) -> HGBConfig:
    return HGBConfig(
        seed=seed,
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        l2_regularization=1.0,
        max_examples=1_000_000,
        min_samples_leaf=30,
    )


def _checkpoint_elapsed() -> float:
    if not CHECKPOINTS.is_dir():
        return 0.0
    total = 0.0
    for path in CHECKPOINTS.glob("*.meta.json"):
        total += float(load_json(path).get("model_fit_wallclock_seconds", 0.0))
    return total


class Budget:
    def __init__(self) -> None:
        self.prior = _checkpoint_elapsed()
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return self.prior + (time.monotonic() - self.started)

    def check(self, label: str) -> None:
        if self.elapsed >= WALL_CLOCK_CAP_SECONDS:
            raise RuntimeError(
                f"Option-D model-fit wall-clock cap reached before {label}: {self.elapsed:.1f}s"
            )


def _phase_qhat_states(
    calibration: pd.DataFrame,
    target: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    observed = pd.to_numeric(calibration[target], errors="coerce").to_numpy(float)
    valid = _available_target_mask(calibration, target).to_numpy(bool, copy=True)
    valid &= np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper)
    scores = np.maximum(lower - observed, observed - upper)
    if not valid.any():
        raise RuntimeError(f"no disjoint calibration rows for {target}")
    global_qhat = finite_sample_quantile(scores[valid], 0.80)
    states: dict[str, Any] = {
        "__GLOBAL__": {
            "qhat": global_qhat,
            "valid_rows": int(valid.sum()),
            "distinct_sessions": int(calibration.loc[valid, "session"].nunique()),
            "source": "global_same_head",
        }
    }
    phases = calibration["market_phase"].astype(str).to_numpy()
    sessions = calibration["session"].astype(str).to_numpy()
    for phase in sorted(set(phases)):
        mask = valid & (phases == phase)
        distinct = len(set(sessions[mask]))
        if int(mask.sum()) >= PHASE_MIN_ROWS and distinct >= PHASE_MIN_SESSIONS:
            states[phase] = {
                "qhat": finite_sample_quantile(scores[mask], 0.80),
                "valid_rows": int(mask.sum()),
                "distinct_sessions": int(distinct),
                "source": "phase_specific",
            }
        else:
            states[phase] = {
                "qhat": global_qhat,
                "valid_rows": int(mask.sum()),
                "distinct_sessions": int(distinct),
                "source": "global_same_head_fallback",
            }
    qhats = np.asarray([float(states.get(phase, states["__GLOBAL__"])["qhat"]) for phase in phases])
    calibrated_lower = lower - qhats
    calibrated_upper = upper + qhats
    covered = valid & (observed >= calibrated_lower) & (observed <= calibrated_upper)
    diagnostics = {
        "valid_rows": int(valid.sum()),
        "raw_monotonicity_violations": int(np.sum(lower > upper)),
        "calibrated_interval_crossings": int(np.sum(calibrated_lower > calibrated_upper)),
        "empirical_coverage": float(covered[valid].mean()),
        "global_qhat": float(global_qhat),
        "phase_specific_count": sum(state["source"] == "phase_specific" for key, state in states.items() if key != "__GLOBAL__"),
        "phase_fallback_count": sum("fallback" in state["source"] for key, state in states.items() if key != "__GLOBAL__"),
    }
    return states, calibrated_lower, calibrated_upper, diagnostics


def _apply_phase_states(
    frame: pd.DataFrame,
    states: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    qhats = np.asarray(
        [
            float(states.get(str(phase), states["__GLOBAL__"])["qhat"])
            for phase in frame["market_phase"]
        ],
        dtype=float,
    )
    return lower - qhats, upper + qhats


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _fit_path_heads(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    replay: pd.DataFrame,
    budget: Budget,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    MODELS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    fit_out = _prediction_columns(fit).reset_index(drop=True)
    cal_out = _prediction_columns(calibration).reset_index(drop=True)
    replay_out = _prediction_columns(replay).reset_index(drop=True)
    heads: list[dict[str, Any]] = []
    for ordinal, (name, target, primary_quantile) in enumerate(PATH_HEADS, start=1):
        checkpoint = CHECKPOINTS / f"path_{ordinal:02d}_{_safe_name(name)}.npz"
        meta_path = CHECKPOINTS / f"path_{ordinal:02d}_{_safe_name(name)}.meta.json"
        if checkpoint.is_file() and meta_path.is_file():
            arrays = np.load(checkpoint)
            meta = load_json(meta_path)
        else:
            head_started = time.monotonic()
            training = fit.loc[_available_target_mask(fit, target)].copy()
            per_seed: list[dict[str, Any]] = []
            fit_primary: list[np.ndarray] = []
            cal_primary: list[np.ndarray] = []
            replay_primary: list[np.ndarray] = []
            cal_composer: list[np.ndarray] = []
            replay_composer: list[np.ndarray] = []
            for seed in SEEDS:
                budget.check(f"{name}/seed{seed}/q10")
                config = _config(seed)
                lower_model, lower_summary = fit_regression_head(
                    training,
                    target=target,
                    feature_names=FEATURE_NAMES,
                    quantile=0.10,
                    config=config,
                )
                fit_lower = predict_regression(lower_model, fit, FEATURE_NAMES)
                train_lower = predict_regression(lower_model, training, FEATURE_NAMES)
                increment_target = f"__{name}_nonnegative_q90_log_increment"
                increment_training = training.copy()
                increment_training[increment_target] = np.log1p(
                    np.maximum(
                        pd.to_numeric(training[target], errors="coerce").to_numpy(float)
                        - train_lower,
                        0.0,
                    )
                )
                budget.check(f"{name}/seed{seed}/q90_increment")
                upper_model, upper_summary = fit_regression_head(
                    increment_training,
                    target=increment_target,
                    feature_names=FEATURE_NAMES,
                    quantile=0.90,
                    config=config,
                )
                cal_lower = predict_regression(lower_model, calibration, FEATURE_NAMES)
                replay_lower = predict_regression(lower_model, replay, FEATURE_NAMES)
                fit_upper = fit_lower + np.expm1(
                    np.maximum(predict_regression(upper_model, fit, FEATURE_NAMES), 0.0)
                )
                cal_upper = cal_lower + np.expm1(
                    np.maximum(predict_regression(upper_model, calibration, FEATURE_NAMES), 0.0)
                )
                replay_upper = replay_lower + np.expm1(
                    np.maximum(predict_regression(upper_model, replay, FEATURE_NAMES), 0.0)
                )
                states, cal_lower_c, cal_upper_c, diagnostics = _phase_qhat_states(
                    calibration, target, cal_lower, cal_upper
                )
                replay_lower_c, replay_upper_c = _apply_phase_states(
                    replay, states, replay_lower, replay_upper
                )
                lower_path = MODELS / f"{ARTIFACT_PREFIX}{name}_seed{seed}_q10.joblib"
                upper_path = MODELS / f"{ARTIFACT_PREFIX}{name}_seed{seed}_q90_increment.joblib"
                joblib.dump(lower_model, lower_path)
                joblib.dump(upper_model, upper_path)
                primary_is_lower = float(primary_quantile) == 0.10
                fit_primary.append(fit_lower if primary_is_lower else fit_upper)
                cal_primary.append(cal_lower if primary_is_lower else cal_upper)
                replay_primary.append(replay_lower if primary_is_lower else replay_upper)
                cal_composer.append(cal_lower_c if primary_is_lower else cal_upper_c)
                replay_composer.append(replay_lower_c if primary_is_lower else replay_upper_c)
                per_seed.append(
                    {
                        "seed": seed,
                        "lower_model": str(lower_path.relative_to(ROOT)),
                        "lower_model_sha256": sha256_path(lower_path),
                        "upper_increment_model": str(upper_path.relative_to(ROOT)),
                        "upper_increment_model_sha256": sha256_path(upper_path),
                        "q10_fit_summary": lower_summary,
                        "q90_increment_fit_summary": upper_summary,
                        "calibration_states": states,
                        "calibration_diagnostics": diagnostics,
                    }
                )
                del lower_model, upper_model, increment_training
                gc.collect()
            arrays_payload = {
                "fit_raw": np.median(np.vstack(fit_primary), axis=0),
                "cal_raw": np.median(np.vstack(cal_primary), axis=0),
                "replay_raw": np.median(np.vstack(replay_primary), axis=0),
                "cal_calibrated": np.median(np.vstack(cal_composer), axis=0),
                "replay_calibrated": np.median(np.vstack(replay_composer), axis=0),
            }
            temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{os.getpid()}.npz")
            np.savez_compressed(temporary, **arrays_payload)
            os.replace(temporary, checkpoint)
            meta = {
                "ordinal": ordinal,
                "name": name,
                "target": target,
                "primary_quantile": primary_quantile,
                "signed_interval_parameterization": "q10_direct_plus_nonnegative_log1p_increment_to_q90",
                "ensemble": "median_of_three_seed_specific_calibrated_primary_endpoints",
                "seed_models": per_seed,
                "model_fit_wallclock_seconds": time.monotonic() - head_started,
                "checkpoint_sha256": sha256_path(checkpoint),
            }
            _write_json(meta_path, meta)
            arrays = arrays_payload
        fit_out[f"{name}__raw"] = np.asarray(arrays["fit_raw"], dtype=float)
        cal_out[f"{name}__raw"] = np.asarray(arrays["cal_raw"], dtype=float)
        replay_out[f"{name}__raw"] = np.asarray(arrays["replay_raw"], dtype=float)
        cal_out[f"{name}__calibrated"] = np.asarray(arrays["cal_calibrated"], dtype=float)
        replay_out[f"{name}__calibrated"] = np.asarray(arrays["replay_calibrated"], dtype=float)
        heads.append(meta)
        _log(
            "path_head_complete",
            ordinal=ordinal,
            total=len(PATH_HEADS),
            name=name,
            model_budget_elapsed_seconds=round(budget.elapsed, 3),
        )
    return fit_out, cal_out, replay_out, heads


def _expected_gate_bootstrap_seed(head_name: str) -> int:
    digest = hashlib.sha256(
        f"{AUTHORITY_SHA256}|{head_name}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _session_clustered_mean_lower_correction(
    *,
    sessions: pd.Series,
    observed: np.ndarray,
    predicted_mean: np.ndarray,
    valid: np.ndarray,
    head_name: str,
    replicates: int = 2_000,
) -> tuple[float, dict[str, Any]]:
    residual = np.asarray(observed, dtype=float) - np.asarray(predicted_mean, dtype=float)
    finite = np.asarray(valid, dtype=bool) & np.isfinite(residual)
    source = pd.DataFrame(
        {
            "session": sessions.astype(str).to_numpy(),
            "residual": residual,
        }
    ).loc[finite]
    session_means = (
        source.groupby("session", sort=True)["residual"].mean().astype(float)
    )
    if len(session_means) < 20:
        raise RuntimeError(
            f"{head_name} expected-upside calibration has only "
            f"{len(session_means)} distinct sessions"
        )
    seed = _expected_gate_bootstrap_seed(head_name)
    values = session_means.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(replicates), len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    correction = finite_sample_quantile(bootstrap_means, 0.10)
    return correction, {
        "method": "deterministic_session_cluster_bootstrap_lower_confidence_correction",
        "confidence_level": 0.90,
        "lower_tail_probability": 0.10,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_seed_material": f"{AUTHORITY_SHA256}|{head_name}",
        "finite_calibration_rows": int(finite.sum()),
        "distinct_calibration_sessions": int(len(session_means)),
        "session_weighting": "equal_after_within_session_mean_residual",
        "residual": "realized_fee_adjusted_target_minus_seed_ensemble_predicted_conditional_mean",
        "session_mean_residual_summary": _summary(session_means.tolist()),
        "bootstrap_mean_residual_summary": _summary(bootstrap_means.tolist()),
        "signed_lower_confidence_correction": float(correction),
        "individual_outcome_quantile_used": False,
        "post_calibration_refit": False,
    }


def _fit_expected_gate_heads(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    replay: pd.DataFrame,
    fit_out: pd.DataFrame,
    cal_out: pd.DataFrame,
    replay_out: pd.DataFrame,
    budget: Budget,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fit the A6 conditional-mean gate companions without changing q10 ranking."""

    MODELS.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    # Path-head insertion is intentionally checkpointed one column at a time;
    # compact once before adding the bounded A6 support block.
    fit_out = fit_out.copy()
    cal_out = cal_out.copy()
    replay_out = replay_out.copy()
    heads: list[dict[str, Any]] = []
    for ordinal, (name, target) in enumerate(EXPECTED_GATE_HEADS, start=1):
        checkpoint = CHECKPOINTS / f"expected_{ordinal:02d}_{_safe_name(name)}.npz"
        meta_path = CHECKPOINTS / f"expected_{ordinal:02d}_{_safe_name(name)}.meta.json"
        if checkpoint.is_file() and meta_path.is_file():
            arrays = np.load(checkpoint)
            meta = load_json(meta_path)
        else:
            head_started = time.monotonic()
            training = fit.loc[_available_target_mask(fit, target)].copy()
            fit_predictions: list[np.ndarray] = []
            calibration_predictions: list[np.ndarray] = []
            replay_predictions: list[np.ndarray] = []
            per_seed: list[dict[str, Any]] = []
            for seed in SEEDS:
                budget.check(f"{name}/seed{seed}/conditional_mean")
                model, fit_summary = fit_regression_head(
                    training,
                    target=target,
                    feature_names=FEATURE_NAMES,
                    quantile=None,
                    config=_config(seed),
                )
                fit_predictions.append(predict_regression(model, fit, FEATURE_NAMES))
                calibration_predictions.append(
                    predict_regression(model, calibration, FEATURE_NAMES)
                )
                replay_predictions.append(
                    predict_regression(model, replay, FEATURE_NAMES)
                )
                model_path = (
                    MODELS
                    / f"{ARTIFACT_PREFIX}{name}_seed{seed}_conditional_mean.joblib"
                )
                joblib.dump(model, model_path)
                per_seed.append(
                    {
                        "seed": seed,
                        "model": str(model_path.relative_to(ROOT)),
                        "model_sha256": sha256_path(model_path),
                        "fit_summary": fit_summary,
                    }
                )
                del model
                gc.collect()
            fit_raw = np.median(np.vstack(fit_predictions), axis=0)
            cal_raw = np.median(np.vstack(calibration_predictions), axis=0)
            replay_raw = np.median(np.vstack(replay_predictions), axis=0)
            observed = pd.to_numeric(calibration[target], errors="coerce").to_numpy(float)
            valid = _available_target_mask(calibration, target).to_numpy(bool, copy=True)
            valid &= np.isfinite(observed) & np.isfinite(cal_raw)
            correction, calibration_state = _session_clustered_mean_lower_correction(
                sessions=calibration["session"],
                observed=observed,
                predicted_mean=cal_raw,
                valid=valid,
                head_name=name,
            )
            arrays_payload = {
                "fit_raw": fit_raw,
                "cal_raw": cal_raw,
                "replay_raw": replay_raw,
                "cal_calibrated_lower": cal_raw + correction,
                "replay_calibrated_lower": replay_raw + correction,
            }
            temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{os.getpid()}.npz")
            np.savez_compressed(temporary, **arrays_payload)
            os.replace(temporary, checkpoint)
            meta = {
                "ordinal": ordinal,
                "name": name,
                "target": target,
                "role": "positive_after_fee_expected_upside_gate_support_only",
                "loss": "squared_error",
                "ensemble": "median_of_three_seed_specific_conditional_mean_forecasts",
                "seed_models": per_seed,
                "calibration_state": calibration_state,
                "model_fit_wallclock_seconds": time.monotonic() - head_started,
                "checkpoint_sha256": sha256_path(checkpoint),
            }
            _write_json(meta_path, meta)
            arrays = arrays_payload
        fit_out[f"{name}__raw"] = np.asarray(arrays["fit_raw"], dtype=float)
        cal_out[f"{name}__raw"] = np.asarray(arrays["cal_raw"], dtype=float)
        replay_out[f"{name}__raw"] = np.asarray(arrays["replay_raw"], dtype=float)
        cal_out[f"{name}__calibrated_lower"] = np.asarray(
            arrays["cal_calibrated_lower"], dtype=float
        )
        replay_out[f"{name}__calibrated_lower"] = np.asarray(
            arrays["replay_calibrated_lower"], dtype=float
        )
        heads.append(meta)
        _log(
            "expected_gate_head_complete",
            ordinal=ordinal,
            total=len(EXPECTED_GATE_HEADS),
            name=name,
            model_budget_elapsed_seconds=round(budget.elapsed, 3),
        )
    return fit_out, cal_out, replay_out, heads


def _tail_target(observed: np.ndarray, mean: np.ndarray) -> np.ndarray:
    denominator = np.maximum(1.0 - np.asarray(mean, dtype=float), 1e-12)
    return np.clip((np.asarray(observed, dtype=float) - mean) / denominator, 0.0, 1.0)


def _fit_action_heads(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    replay: pd.DataFrame,
    fit_out: pd.DataFrame,
    cal_out: pd.DataFrame,
    replay_out: pd.DataFrame,
    budget: Budget,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    checkpoint = CHECKPOINTS / "action_heads.npz"
    meta_path = CHECKPOINTS / "action_heads.meta.json"
    path_features = tuple(f"{name}__raw" for name, _, _ in PATH_HEADS)
    contract_features = (*FEATURE_NAMES, *path_features)
    if checkpoint.is_file() and meta_path.is_file():
        arrays = np.load(checkpoint)
        meta = load_json(meta_path)
    else:
        action_started = time.monotonic()
        regret_fit = fit_out.copy()
        regret_fit["normalized_regret"] = fit["normalized_regret"].to_numpy(float)
        regret_cal = cal_out.copy()
        regret_cal["normalized_regret"] = calibration["normalized_regret"].to_numpy(float)
        fit_valid = regret_fit[
            regret_fit["action_eligible"].astype(bool)
            & np.isfinite(regret_fit["normalized_regret"].to_numpy(float))
        ].copy()
        cal_valid_mask = (
            regret_cal["action_eligible"].astype(bool).to_numpy()
            & np.isfinite(regret_cal["normalized_regret"].to_numpy(float))
        )
        expected_cal: list[np.ndarray] = []
        expected_replay: list[np.ndarray] = []
        q90_cal: list[np.ndarray] = []
        q90_replay: list[np.ndarray] = []
        regret_seeds: list[dict[str, Any]] = []
        for seed in SEEDS:
            config = _config(seed)
            budget.check(f"expected_regret/seed{seed}")
            mean_model, mean_summary = fit_regression_head(
                fit_valid,
                target="normalized_regret",
                feature_names=contract_features,
                quantile=None,
                config=config,
            )
            fit_mean = np.clip(predict_regression(mean_model, fit_valid, contract_features), 0.0, 1.0)
            tail_name = "__normalized_regret_nonnegative_tail"
            tail_frame = fit_valid.copy()
            tail_frame[tail_name] = _tail_target(
                fit_valid["normalized_regret"].to_numpy(float), fit_mean
            )
            budget.check(f"q90_regret/seed{seed}")
            tail_model, tail_summary = fit_regression_head(
                tail_frame,
                target=tail_name,
                feature_names=contract_features,
                quantile=0.90,
                config=config,
            )
            cal_mean = np.clip(predict_regression(mean_model, regret_cal, contract_features), 0.0, 1.0)
            replay_mean = np.clip(predict_regression(mean_model, replay_out, contract_features), 0.0, 1.0)
            cal_tail = np.clip(predict_regression(tail_model, regret_cal, contract_features), 0.0, 1.0)
            replay_tail = np.clip(predict_regression(tail_model, replay_out, contract_features), 0.0, 1.0)
            cal_raw_q90 = cal_mean + (1.0 - cal_mean) * cal_tail
            replay_raw_q90 = replay_mean + (1.0 - replay_mean) * replay_tail
            observed = regret_cal["normalized_regret"].to_numpy(float)
            scores = np.maximum(0.0, observed[cal_valid_mask] - cal_raw_q90[cal_valid_mask])
            qhat = finite_sample_quantile(scores, 0.90)
            expected_cal.append(cal_mean)
            expected_replay.append(replay_mean)
            q90_cal.append(np.clip(cal_raw_q90 + qhat, 0.0, 1.0))
            q90_replay.append(np.clip(replay_raw_q90 + qhat, 0.0, 1.0))
            mean_path = MODELS / f"{ARTIFACT_PREFIX}expected_normalized_regret_seed{seed}.joblib"
            tail_path = MODELS / f"{ARTIFACT_PREFIX}q90_normalized_regret_seed{seed}_tail.joblib"
            joblib.dump(mean_model, mean_path)
            joblib.dump(tail_model, tail_path)
            regret_seeds.append(
                {
                    "seed": seed,
                    "mean_model": str(mean_path.relative_to(ROOT)),
                    "mean_model_sha256": sha256_path(mean_path),
                    "tail_model": str(tail_path.relative_to(ROOT)),
                    "tail_model_sha256": sha256_path(tail_path),
                    "mean_fit_summary": mean_summary,
                    "tail_fit_summary": tail_summary,
                    "one_sided_conformal_qhat": qhat,
                }
            )
            del mean_model, tail_model, tail_frame
            gc.collect()

        wait_fit_source = fit_out.copy()
        wait_fit_source["wait_target"] = fit["wait_target"].to_numpy(float)
        wait_cal_source = cal_out.copy()
        wait_cal_source["wait_target"] = calibration["wait_target"].to_numpy(float)
        wait_fit = pooled_wait_frame(wait_fit_source, (*FEATURE_NAMES, *path_features))
        wait_cal = pooled_wait_frame(wait_cal_source, (*FEATURE_NAMES, *path_features))
        wait_replay = pooled_wait_frame(replay_out, (*FEATURE_NAMES, *path_features))
        wait_names = wait_feature_names((*FEATURE_NAMES, *path_features))
        wait_cal_predictions: list[np.ndarray] = []
        wait_replay_predictions: list[np.ndarray] = []
        wait_seeds: list[dict[str, Any]] = []
        for seed in SEEDS:
            budget.check(f"wait_probability/seed{seed}")
            wait_model, wait_summary = fit_wait_head(
                wait_fit,
                feature_names=wait_names,
                config=_config(seed),
            )
            cal_raw = wait_model.predict_proba(wait_cal.loc[:, list(wait_names)].to_numpy(float))[:, 1]
            replay_raw = wait_model.predict_proba(wait_replay.loc[:, list(wait_names)].to_numpy(float))[:, 1]
            valid = np.isfinite(wait_cal["wait_target"].to_numpy(float))
            state = fit_isotonic_state(cal_raw[valid], wait_cal.loc[valid, "wait_target"].to_numpy(float))
            wait_cal_predictions.append(apply_isotonic_state(cal_raw, state))
            wait_replay_predictions.append(apply_isotonic_state(replay_raw, state))
            path = MODELS / f"{ARTIFACT_PREFIX}wait_probability_seed{seed}.joblib"
            joblib.dump(wait_model, path)
            wait_seeds.append(
                {
                    "seed": seed,
                    "model": str(path.relative_to(ROOT)),
                    "model_sha256": sha256_path(path),
                    "fit_summary": wait_summary,
                    "isotonic_calibrator": state,
                }
            )
        arrays_payload = {
            "cal_expected": np.median(np.vstack(expected_cal), axis=0),
            "replay_expected": np.median(np.vstack(expected_replay), axis=0),
            "cal_q90": np.median(np.vstack(q90_cal), axis=0),
            "replay_q90": np.median(np.vstack(q90_replay), axis=0),
            "cal_wait": np.median(np.vstack(wait_cal_predictions), axis=0),
            "replay_wait": np.median(np.vstack(wait_replay_predictions), axis=0),
            "cal_wait_session": wait_cal["session"].astype(str).to_numpy(),
            "cal_wait_decision": wait_cal["decision_time_ns"].to_numpy(np.int64),
            "replay_wait_session": wait_replay["session"].astype(str).to_numpy(),
            "replay_wait_decision": wait_replay["decision_time_ns"].to_numpy(np.int64),
        }
        temporary = checkpoint.with_name(f".{checkpoint.name}.tmp-{os.getpid()}.npz")
        np.savez_compressed(temporary, **arrays_payload)
        os.replace(temporary, checkpoint)
        meta = {
            "primary_heads": list(ACTION_HEADS),
            "regret_input_feature_count": len(contract_features),
            "wait_input_feature_count": len(wait_names),
            "wait_inputs_include_raw_path_forecasts": True,
            "q90_parameterization": "mean_plus_one_minus_mean_times_clipped_nonnegative_tail",
            "regret_seed_models": regret_seeds,
            "wait_seed_models": wait_seeds,
            "model_fit_wallclock_seconds": time.monotonic() - action_started,
            "checkpoint_sha256": sha256_path(checkpoint),
        }
        _write_json(meta_path, meta)
        arrays = arrays_payload

    cal_out["expected_normalized_regret__calibrated"] = np.asarray(arrays["cal_expected"], dtype=float)
    replay_out["expected_normalized_regret__calibrated"] = np.asarray(arrays["replay_expected"], dtype=float)
    cal_out["q90_regret_upper_bound"] = np.asarray(arrays["cal_q90"], dtype=float)
    replay_out["q90_regret_upper_bound"] = np.asarray(arrays["replay_q90"], dtype=float)
    cal_map = {
        (str(session), int(decision)): float(value)
        for session, decision, value in zip(
            arrays["cal_wait_session"], arrays["cal_wait_decision"], arrays["cal_wait"]
        )
    }
    replay_map = {
        (str(session), int(decision)): float(value)
        for session, decision, value in zip(
            arrays["replay_wait_session"], arrays["replay_wait_decision"], arrays["replay_wait"]
        )
    }
    cal_out["wait_probability__calibrated"] = [
        cal_map[(str(session), int(decision))]
        for session, decision in zip(cal_out["session"], cal_out["decision_time_ns"])
    ]
    replay_out["wait_probability__calibrated"] = [
        replay_map[(str(session), int(decision))]
        for session, decision in zip(replay_out["session"], replay_out["decision_time_ns"])
    ]
    _log("action_heads_complete", model_budget_elapsed_seconds=round(budget.elapsed, 3))
    return cal_out, replay_out, meta


def _reliability_bins(predicted: np.ndarray, observed: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(predicted, kind="mergesort")
    x, y = predicted[order], observed[order]
    groups: list[dict[str, Any]] = []
    for value in np.unique(x):
        mask = x == value
        groups.append({"count": int(mask.sum()), "predicted_sum": float(x[mask].sum()), "observed_sum": float(y[mask].sum())})
    while len(groups) > 1 and any(group["count"] < 10 for group in groups):
        index = next(i for i, group in enumerate(groups) if group["count"] < 10)
        neighbor = index + 1 if index + 1 < len(groups) else index - 1
        lo, hi = sorted((index, neighbor))
        merged = {
            key: groups[lo][key] + groups[hi][key]
            for key in ("count", "predicted_sum", "observed_sum")
        }
        groups[lo : hi + 1] = [merged]
    return [
        {
            "count": group["count"],
            "mean_prediction": group["predicted_sum"] / group["count"],
            "observed_rate": group["observed_sum"] / group["count"],
            "gap": abs(group["predicted_sum"] - group["observed_sum"]) / group["count"],
        }
        for group in groups
    ]


def _action_gate(
    calibration: pd.DataFrame, cal_out: pd.DataFrame
) -> dict[str, Any]:
    wait_truth = pooled_wait_frame(calibration, FEATURE_NAMES)
    wait_pred = (
        cal_out[["session", "decision_time_ns", "wait_probability__calibrated"]]
        .drop_duplicates(["session", "decision_time_ns"])
    )
    wait = wait_truth.merge(wait_pred, on=["session", "decision_time_ns"], validate="one_to_one")
    wait = wait[np.isfinite(wait["wait_target"].to_numpy(float))].copy()
    y = wait["wait_target"].to_numpy(float)
    p = wait["wait_probability__calibrated"].to_numpy(float)
    bins = _reliability_bins(p, y)
    ece = float(sum(row["count"] * row["gap"] for row in bins) / max(1, len(y)))
    max_gap = float(max((row["gap"] for row in bins), default=float("nan")))
    brier = float(np.mean((p - y) ** 2))
    constant_brier = float(np.mean((float(np.mean(y)) - y) ** 2))
    brier_skill = 0.0 if constant_brier == 0.0 else 1.0 - brier / constant_brier

    eligible = cal_out["action_eligible"].astype(bool).to_numpy()
    observed_regret = calibration["normalized_regret"].to_numpy(float)
    valid = eligible & np.isfinite(observed_regret)
    expected = cal_out["expected_normalized_regret__calibrated"].to_numpy(float)
    upper = cal_out["q90_regret_upper_bound"].to_numpy(float)
    mean_error = abs(float(np.mean(expected[valid])) - float(np.mean(observed_regret[valid])))
    coverage = float(np.mean(observed_regret[valid] <= upper[valid]))
    minimums = {
        "distinct_sessions_at_least_20": int(wait["session"].nunique()) >= 20,
        "flat_decisions_at_least_100": len(wait) >= 100,
        "realized_enter_outcomes_at_least_50": int(np.sum(y == 0.0)) >= 50,
        "realized_wait_outcomes_at_least_50": int(np.sum(y == 1.0)) >= 50,
        "rlac_contract_targets_at_least_50": int(valid.sum()) >= 50,
    }
    metrics = {
        "wait_ece_at_most_0_10": ece <= 0.10,
        "wait_max_bin_gap_at_most_0_20": max_gap <= 0.20,
        "wait_brier_skill_at_least_0": brier_skill >= 0.0,
        "absolute_mean_expected_regret_error_at_most_0_10": mean_error <= 0.10,
        "q90_regret_coverage_between_0_85_and_0_95": 0.85 <= coverage <= 0.95,
    }
    return {
        "available": all(minimums.values()) and all(metrics.values()),
        "minimum_evidence_checks": minimums,
        "metric_checks": metrics,
        "wait": {
            "distinct_sessions": int(wait["session"].nunique()),
            "flat_decisions": int(len(wait)),
            "realized_enter_outcomes": int(np.sum(y == 0.0)),
            "realized_wait_outcomes": int(np.sum(y == 1.0)),
            "ece": ece,
            "maximum_bin_gap": max_gap,
            "brier": brier,
            "constant_rate_brier": constant_brier,
            "brier_skill": brier_skill,
            "bins": bins,
        },
        "regret": {
            "rlac_contract_targets": int(valid.sum()),
            "absolute_mean_calibration_error": mean_error,
            "empirical_q90_upper_bound_coverage": coverage,
            "q90_bound_threshold": 0.10,
        },
    }


def _realized_mfe_means(row: pd.Series) -> tuple[float, float] | None:
    dollars: list[float] = []
    returns: list[float] = []
    for horizon in HORIZONS:
        if not causal_horizon_available(int(row["decision_time_ns"]), horizon):
            continue
        prefix = LABEL_PREFIX.get(horizon, horizon)
        if bool(row.get(f"{prefix}_censored", True)):
            continue
        d = float(row.get(f"{prefix}_mfe_dollars", np.nan))
        r = float(row.get(f"{prefix}_mfe_return", np.nan))
        if math.isfinite(d) and math.isfinite(r):
            dollars.append(d)
            returns.append(r)
    if not dollars:
        return None
    return float(np.mean(dollars)), float(np.mean(returns))


def _calibrate_composer_errors(
    calibration: pd.DataFrame,
    cal_out: pd.DataFrame,
    cdfs: dict[str, Any],
) -> dict[str, Any]:
    gap_errors: list[float] = []
    mfe_dollar_errors: list[float] = []
    mfe_return_errors: list[float] = []
    proposed = 0
    for _, block in cal_out.groupby(["session", "decision_time_ns"], sort=True):
        decision = compose_decision(
            block,
            cdfs=cdfs,
            model_gap_error=0.0,
            mfe_error_margin_dollars=0.0,
            mfe_error_margin_return=0.0,
            source_transfer_error=0.0,
            uncertainty_multiplier=1.0,
            guardrail_alpha=0.0,
            action_conditioned_gate_available=True,
        )
        index = decision.get("proposed_contract_index")
        if index is None:
            continue
        proposed += 1
        selected = block.loc[int(index)]
        cluster = (
            (block["right"].astype(str) == str(selected["right"]))
            & (block["expiry"].astype(str) == str(selected["expiry"]))
            & ((block["strike_milli_points"].astype(int) - int(selected["strike_milli_points"])).abs() <= 10_000)
        )
        valid = block["action_eligible"].astype(bool) & np.isfinite(block["u_label"].to_numpy(float))
        inside = block[valid & cluster]
        outside = block[valid & ~cluster]
        if not inside.empty:
            realized_gap = float(inside["u_label"].max()) - (float(outside["u_label"].max()) if not outside.empty else 0.0)
            gap_errors.append(abs(float(decision["predicted_cluster_gap"]) - realized_gap))
        realized = _realized_mfe_means(calibration.loc[int(index)])
        if realized is not None:
            mfe_dollar_errors.append(abs(float(decision["selected_mean_mfe_dollars"]) - realized[0]))
            mfe_return_errors.append(abs(float(decision["selected_mean_mfe_return"]) - realized[1]))
    sufficient = bool(gap_errors and mfe_dollar_errors and mfe_return_errors)
    return {
        "sufficient": sufficient,
        "proposed_calibration_decisions": proposed,
        "cluster_gap": {
            "sample_count": len(gap_errors),
            "q90_absolute_error": finite_sample_quantile(gap_errors, 0.90) if gap_errors else 0.0,
        },
        "mfe_dollars": {
            "sample_count": len(mfe_dollar_errors),
            "q90_absolute_error": finite_sample_quantile(mfe_dollar_errors, 0.90) if mfe_dollar_errors else 0.0,
        },
        "mfe_return": {
            "sample_count": len(mfe_return_errors),
            "q90_absolute_error": finite_sample_quantile(mfe_return_errors, 0.90) if mfe_return_errors else 0.0,
        },
        "method": "same_final_model_disjoint_calibration_finite_sample_q90_absolute_error",
    }


def _compose_variants(
    replay: pd.DataFrame,
    cdfs: dict[str, Any],
    errors: dict[str, Any],
    action_gate_available: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    all_rows: list[dict[str, Any]] = []
    headline: list[dict[str, Any]] = []
    pre_gate: list[dict[str, Any]] = []
    model_gap = float(errors["cluster_gap"]["q90_absolute_error"])
    mfe_d = float(errors["mfe_dollars"]["q90_absolute_error"])
    mfe_r = float(errors["mfe_return"]["q90_absolute_error"])
    effective_gate = action_gate_available and bool(errors["sufficient"])
    for key, block in replay.groupby(["session", "decision_time_ns"], sort=True):
        for alpha in ALPHAS:
            for k in KS:
                result = compose_decision(
                    block,
                    cdfs=cdfs,
                    model_gap_error=model_gap,
                    mfe_error_margin_dollars=mfe_d,
                    mfe_error_margin_return=mfe_r,
                    source_transfer_error=0.0,
                    uncertainty_multiplier=k,
                    guardrail_alpha=alpha,
                    action_conditioned_gate_available=effective_gate,
                )
                result.update(
                    {
                        "variant_id": f"alpha_{alpha:g}_k_{k:g}",
                        "guardrail_alpha": alpha,
                        "uncertainty_multiplier_k": k,
                        "headline_control": alpha == HEADLINE_ALPHA and k == HEADLINE_K,
                        "split": "plumbing_replay",
                        "quarantine_labels": "|".join(QUARANTINE_LABELS),
                        "calibrated_wait_probability": float(block["wait_probability__calibrated"].iloc[0]),
                    }
                )
                all_rows.append(result)
                if alpha == HEADLINE_ALPHA and k == HEADLINE_K:
                    headline.append(result)
                    diagnostic = compose_decision(
                        block,
                        cdfs=cdfs,
                        model_gap_error=model_gap,
                        mfe_error_margin_dollars=mfe_d,
                        mfe_error_margin_return=mfe_r,
                        source_transfer_error=0.0,
                        uncertainty_multiplier=k,
                        guardrail_alpha=alpha,
                        action_conditioned_gate_available=True,
                    )
                    diagnostic["split"] = "plumbing_replay_pre_action_gate_diagnostic"
                    pre_gate.append(diagnostic)
    return all_rows, headline, pre_gate


def _summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "share_strictly_positive": float(np.mean(finite > 0.0)),
    }


def _near_misses(replay: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    axes = ("mfe_dollars", "mfe_return", "profit_area_dollars", "profit_area_return")
    kinds = ("expected_raw", "expected_lower", "q10_raw", "q10_calibrated")
    distributions = {kind: {axis: [] for axis in axes} for kind in kinds}
    counts = {kind: Counter() for kind in kinds}
    passing_decisions = {kind: set() for kind in kinds}
    for index, row in replay[replay["action_eligible"].astype(bool)].iterrows():
        available = [h for h in HORIZONS if causal_horizon_available(int(row["decision_time_ns"]), h)]
        if not available:
            continue
        decision_key = (str(row["session"]), int(row["decision_time_ns"]))
        record: dict[str, Any] = {
            "session": str(row["session"]),
            "decision_time_ns": int(row["decision_time_ns"]),
            "decision_time_utc": pd.Timestamp(
                int(row["decision_time_ns"]), unit="ns", tz="UTC"
            ).isoformat(),
            "decision_minute_et": pd.Timestamp(
                int(row["decision_time_ns"]), unit="ns", tz="UTC"
            ).tz_convert("America/New_York").strftime("%H:%M"),
            "contract_id": str(row["contract_id"]),
            "right": str(row["right"]),
            "strike_milli_points": int(row["strike_milli_points"]),
            "decision_entry_ask": float(row["decision_entry_ask"]),
        }
        sources = {
            "expected_raw": (
                "expected_upside_{horizon}_mean_{axis}__raw"
            ),
            "expected_lower": (
                "expected_upside_{horizon}_mean_{axis}__calibrated_lower"
            ),
            "q10_raw": "upside_{horizon}_q10_{axis}__raw",
            "q10_calibrated": "upside_{horizon}_q10_{axis}__calibrated",
        }
        for kind, template in sources.items():
            values: dict[str, float] = {}
            for axis in axes:
                series = [
                    float(row[template.format(horizon=horizon, axis=axis)])
                    for horizon in available
                ]
                values[axis] = float(np.mean(series))
                distributions[kind][axis].append(values[axis])
                record[f"{kind}_{axis}"] = values[axis]
            positive = sum(value > 0.0 for value in values.values())
            counts[kind][positive] += 1
            record[f"{kind}_positive_axis_count"] = positive
            if positive == 4:
                passing_decisions[kind].add(decision_key)
        rows.append(record)
    scale = {}
    for axis in axes:
        absolute = np.abs(np.asarray(distributions["expected_lower"][axis], dtype=float))
        nonzero = absolute[absolute > 1e-12]
        scale[axis] = float(np.median(nonzero)) if len(nonzero) else 1.0
    for row in rows:
        row["expected_lower_normalized_min_margin"] = min(
            row[f"expected_lower_{axis}"] / scale[axis] for axis in axes
        )
    rows.sort(
        key=lambda item: (
            -item["expected_lower_positive_axis_count"],
            -item["expected_lower_normalized_min_margin"],
            item["session"],
            item["decision_time_ns"],
            item["contract_id"],
        )
    )
    return {
        "eligible_contract_rows": len(rows),
        "distributions": {
            kind: {axis: _summary(values) for axis, values in by_axis.items()}
            for kind, by_axis in distributions.items()
        },
        "positive_axis_count_histogram": {
            kind: {str(value): int(counter.get(value, 0)) for value in range(5)}
            for kind, counter in counts.items()
        },
        "expected_raw_four_axis_pass_rows": int(counts["expected_raw"].get(4, 0)),
        "expected_lower_four_axis_pass_rows": int(counts["expected_lower"].get(4, 0)),
        "expected_raw_four_axis_pass_decisions": len(passing_decisions["expected_raw"]),
        "expected_lower_four_axis_pass_decisions": len(passing_decisions["expected_lower"]),
        "q10_raw_four_axis_pass_rows": int(counts["q10_raw"].get(4, 0)),
        "q10_calibrated_four_axis_pass_rows": int(counts["q10_calibrated"].get(4, 0)),
        "top_near_misses": rows[:25],
    }


def _trade_profile(trades: list[Any]) -> dict[str, Any]:
    buckets = Counter()
    premium_bands = Counter()
    for trade in trades:
        premium_bands[
            "le_1" if trade.entry_ask <= 1.0 else "1_3" if trade.entry_ask <= 3.0 else "3_8" if trade.entry_ask <= 8.0 else "8_20" if trade.entry_ask <= 20.0 else "20p"
        ] += 1
        return_on_premium = float(trade.raw_label_pnl_after_campaign_fee) / max(1e-12, float(trade.entry_ask) * 100.0)
        if return_on_premium >= 0.40:
            buckets["big_win_ge_40pct"] += 1
        elif return_on_premium >= -0.05:
            buckets["scratch_or_small_win_ge_minus5_lt40pct"] += 1
        elif return_on_premium > -0.30:
            buckets["small_loss_gt_minus30_lt_minus5pct"] += 1
        else:
            buckets["big_loss_le_minus30pct"] += 1
    return {
        "trade_count": len(trades),
        "distinct_sessions": len({trade.session for trade in trades}),
        "premium_bands": dict(sorted(premium_bands.items())),
        "pickles_four_buckets": dict(sorted(buckets.items())),
        "serial_legality": "simulator_v5_one_account_hold_flat_ask_entry_bid_exit_fee_aware",
    }


def _downstream_binding_analysis(
    headline: list[dict[str, Any]] | pd.DataFrame,
    action_gate: dict[str, Any],
) -> dict[str, Any]:
    frame = headline.copy() if isinstance(headline, pd.DataFrame) else pd.DataFrame(headline)
    if frame.empty or "proposed_contract_id" not in frame:
        return {"proposed_decisions_after_expected_upside_gate": 0}
    proposed = frame[frame["proposed_contract_id"].notna()].copy()
    if proposed.empty:
        return {"proposed_decisions_after_expected_upside_gate": 0}
    uncertainty = proposed["predicted_cluster_gap"].astype(float) > proposed[
        "wait_margin"
    ].astype(float)
    mfe = (
        proposed["selected_mean_mfe_dollars"].astype(float)
        > proposed["mfe_error_margin_dollars"].astype(float)
    ) & (
        proposed["selected_mean_mfe_return"].astype(float)
        > proposed["mfe_error_margin_return"].astype(float)
    )
    regret = proposed["selected_q90_regret_upper_bound"].astype(float) <= 0.10
    pre_action = uncertainty & mfe & regret
    action_available = bool(action_gate["available"])
    individual_counts = {
        "uncertainty_cluster_gap": int(uncertainty.sum()),
        "unchanged_q10_mfe_error_margin": int(mfe.sum()),
        "q90_regret_upper_bound": int(regret.sum()),
        "mandatory_action_conditioned_gate": int(len(proposed))
        if action_available
        else 0,
    }
    second_order_verdict = (
        "CONFIRMED_ALL_FROZEN_NUMERIC_DOWNSTREAM_GATES_INDIVIDUALLY_REJECT_"
        "EVERY_EXPECTED_UPSIDE_PROPOSER_AND_ACTION_GATE_IS_UNAVAILABLE"
        if not any(
            individual_counts[key]
            for key in (
                "uncertainty_cluster_gap",
                "unchanged_q10_mfe_error_margin",
                "q90_regret_upper_bound",
            )
        )
        and not action_available
        else "MIXED_DOWNSTREAM_BINDING"
    )
    return {
        "proposed_decisions_after_expected_upside_gate": int(len(proposed)),
        "individual_strict_pass_counts": individual_counts,
        "all_three_numeric_downstream_gates_pass": int(pre_action.sum()),
        "all_downstream_gates_including_action_conditioned_pass": int(
            pre_action.sum()
        )
        if action_available
        else 0,
        "action_conditioned_gate_available": action_available,
        "second_order_verdict": second_order_verdict,
        "thresholds": {
            "uncertainty_wait_margin": float(proposed["wait_margin"].iloc[0]),
            "q10_mfe_error_margin_dollars": float(
                proposed["mfe_error_margin_dollars"].iloc[0]
            ),
            "q10_mfe_error_margin_return": float(
                proposed["mfe_error_margin_return"].iloc[0]
            ),
            "q90_regret_upper_bound_maximum": 0.10,
        },
        "distributions": {
            "predicted_cluster_gap": _summary(
                proposed["predicted_cluster_gap"].astype(float).tolist()
            ),
            "selected_unchanged_q10_mean_mfe_dollars": _summary(
                proposed["selected_mean_mfe_dollars"].astype(float).tolist()
            ),
            "selected_unchanged_q10_mean_mfe_return": _summary(
                proposed["selected_mean_mfe_return"].astype(float).tolist()
            ),
            "selected_q90_regret_upper_bound": _summary(
                proposed["selected_q90_regret_upper_bound"].astype(float).tolist()
            ),
        },
        "interpretation": (
            "Individual pass counts are report-only counterfactual diagnostics. "
            "The frozen composer still applies gates sequentially in its governed order."
        ),
    }


def _diagnostic(
    outcome: str,
    replay: pd.DataFrame,
    headline: list[dict[str, Any]],
    pre_gate: list[dict[str, Any]],
    action_gate: dict[str, Any],
    errors: dict[str, Any],
    path_heads: list[dict[str, Any]],
    trades: list[Any],
) -> dict[str, Any]:
    near = _near_misses(replay)
    final_reasons = Counter(row.get("wait_reason") or "BUY" for row in headline)
    pre_reasons = Counter(row.get("wait_reason") or "BUY" for row in pre_gate)
    raw_pass = near["expected_raw_four_axis_pass_rows"]
    cal_pass = near["expected_lower_four_axis_pass_rows"]
    pre_buy = int(pre_reasons.get("BUY", 0))
    if outcome == "TRADES":
        cause = "headline composer emitted serial-legal trades"
    elif raw_pass > 0 and cal_pass == 0:
        cause = "expected_mean_lower_confidence_calibration_dominant_abstention"
    elif raw_pass == 0:
        cause = "conditional_mean_signal_or_capped_model_capacity_limit_before_calibration"
    elif cal_pass > 0 and pre_buy == 0:
        cause = "downstream_uncertainty_mfe_error_or_regret_constraint"
    elif pre_buy > 0 and not action_gate["available"]:
        cause = "mandatory_action_conditioned_calibration_gate"
    else:
        cause = "mixed_downstream_constraints"
    qhats = [
        float(seed["calibration_diagnostics"]["global_qhat"])
        for head in path_heads
        for seed in head["seed_models"]
    ]
    coverages = [
        float(seed["calibration_diagnostics"]["empirical_coverage"])
        for head in path_heads
        for seed in head["seed_models"]
    ]
    return {
        "schema_version": "Protocol101WalkingSkeletonOptionDA6Stage1DiagnosticV1",
        "outcome": outcome,
        "tentative_cause": cause,
        "data_quantity_interpretation": (
            "The fit and calibration roles each meet the exact 20-session FT2-10 minima. "
            "This dry-run does not establish formal data insufficiency and cannot rule out benefit from broader development data."
        ),
        "near_miss_analysis": near,
        "headline_rejection_counts": dict(sorted(final_reasons.items())),
        "pre_action_gate_rejection_counts": dict(sorted(pre_reasons.items())),
        "action_conditioned_gate": action_gate,
        "composer_error_calibration": errors,
        "downstream_gate_binding_analysis": _downstream_binding_analysis(
            headline, action_gate
        ),
        "path_conformal_global_qhat_distribution": _summary(qhats),
        "path_conformal_empirical_coverage_distribution": _summary(coverages),
        "trade_profile": _trade_profile(trades),
        "scientific_claim_allowed": False,
        "stop": "STOP_FOR_CLAUDE_VERIFICATION",
    }


def _diagnostic_markdown(payload: dict[str, Any]) -> str:
    near = payload["near_miss_analysis"]
    downstream = payload["downstream_gate_binding_analysis"]
    return f"""# Option-D Stage-1 Diagnostic

Outcome: **{payload['outcome']}**  
Tentative cause: **{payload['tentative_cause']}**

This is a quarantined walking-skeleton plumbing result, not model-quality or alpha evidence.

## Expected-upside gate evidence

- Physically eligible replay contract rows: {near['eligible_contract_rows']}
- Raw conditional-mean four-axis passes: {near['expected_raw_four_axis_pass_rows']}
- Session-clustered lower-confidence four-axis passes: {near['expected_lower_four_axis_pass_rows']}
- Raw q10 four-axis passes (ranking diagnostic only): {near['q10_raw_four_axis_pass_rows']}
- Calibrated q10 four-axis passes (ranking diagnostic only): {near['q10_calibrated_four_axis_pass_rows']}
- Headline reasons: `{json.dumps(payload['headline_rejection_counts'], sort_keys=True)}`
- Pre-action-gate reasons: `{json.dumps(payload['pre_action_gate_rejection_counts'], sort_keys=True)}`

## Downstream binding

- Decisions with an expected-upside passer: {downstream['proposed_decisions_after_expected_upside_gate']}
- Individual strict pass counts: `{json.dumps(downstream.get('individual_strict_pass_counts', {}), sort_keys=True)}`
- All three numeric downstream gates pass: {downstream.get('all_three_numeric_downstream_gates_pass', 0)}
- All downstream gates including the action-conditioned gate pass: {downstream.get('all_downstream_gates_including_action_conditioned_pass', 0)}
- Second-order verdict: `{downstream.get('second_order_verdict', 'UNKNOWN')}`

## Action-conditioned gate

- Available: {payload['action_conditioned_gate']['available']}
- WAIT outcomes: {payload['action_conditioned_gate']['wait']['realized_wait_outcomes']}
- ENTER outcomes: {payload['action_conditioned_gate']['wait']['realized_enter_outcomes']}
- q90 regret coverage: {payload['action_conditioned_gate']['regret']['empirical_q90_upper_bound_coverage']:.6f}

## Interpretation

{payload['data_quantity_interpretation']}

Stop: `STOP_FOR_CLAUDE_VERIFICATION`.
"""


def _label_stripped(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        column
        for column in frame.columns
        if column not in INFERENCE_FORBIDDEN_EXACT
    ]
    return frame.loc[:, keep].copy()


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_path(path), "bytes": path.stat().st_size}


def _audit(receipt: dict[str, Any], write: bool = True) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    check("outcome_is_valid", receipt["outcome"] in {"TRADES", "ABSTAINS"}, receipt["outcome"])
    check("authority_hash_exact", receipt["product_contract_hash"] == AUTHORITY_SHA256, receipt["product_contract_hash"])
    check("quarantine_labels_exact", tuple(receipt["quarantine_labels"]) == QUARANTINE_LABELS, receipt["quarantine_labels"])
    check("artifact_prefix_exact", receipt["artifact_prefix"] == ARTIFACT_PREFIX, receipt["artifact_prefix"])
    check("full45_session_count", receipt["tensor"]["session_count"] == 45, receipt["tensor"]["session_count"])
    check("partition_is_20_1_20_4", [len(receipt["partitions"][key]) for key in ("fit", "embargo", "calibration", "replay")] == [20, 1, 20, 4], receipt["partitions"])
    check("embargo_excluded_from_model_roles", receipt["embargo_excluded_from_fit_calibration_and_replay"], receipt["partitions"]["embargo"])
    check("one_config_within_cap", receipt["models"]["preregistered_config_count"] == 1 <= receipt["models"]["config_cap"], receipt["models"])
    check("three_frozen_seeds", receipt["models"]["seeds"] == list(SEEDS), receipt["models"]["seeds"])
    check("all_39_primary_heads", receipt["models"]["primary_head_count"] == 39, receipt["models"]["primary_head_count"])
    check("all_36_signed_pairs", receipt["models"]["signed_path_pair_count"] == 36, receipt["models"]["signed_path_pair_count"])
    check(
        "all_28_expected_gate_support_heads",
        receipt["models"]["expected_gate_support_head_count"] == 28,
        receipt["models"]["expected_gate_support_head_count"],
    )
    check("hgb_only_no_neural_gpu", receipt["models"]["family"] == "sklearn HistGradientBoosting tabular only" and not receipt["models"]["neural_used"] and not receipt["models"]["gpu_used"], receipt["models"])
    check("model_wallclock_within_cap", receipt["models"]["fit_wallclock_seconds"] <= WALL_CLOCK_CAP_SECONDS, receipt["models"]["fit_wallclock_seconds"])
    check("real_split_conformal_recorded", receipt["calibration"]["method"] == "same-final-model disjoint split conformalized quantile regression", receipt["calibration"]["method"])
    check("composer_grid_all_20", receipt["composer"]["variant_count"] == 20, receipt["composer"]["variant_count"])
    check("headline_control_exact", receipt["composer"]["headline_control"] == {"guardrail_alpha": 0.0, "k": 1.0}, receipt["composer"]["headline_control"])
    check("no_fabricated_buy", receipt["composer"]["fabricated_or_injected_buy"] is False, receipt["composer"]["fabricated_or_injected_buy"])
    check("real_simulator_v5", receipt["serial_replay"]["simulator_version"] == PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION, receipt["serial_replay"]["simulator_version"])
    check("trades_outcome_consistent", (receipt["outcome"] == "TRADES") == (receipt["serial_replay"]["trade_count"] > 0), receipt["serial_replay"]["trade_count"])
    check("label_stripped_replay", receipt["inference_artifacts"]["replay_predictions_label_stripped"], receipt["inference_artifacts"])
    check("no_protected_or_broker_or_download", all(not receipt["side_effects"][key] for key in ("protected_resource_read", "broker_contacted", "paid_download")), receipt["side_effects"])
    check("stage2_not_started", receipt["side_effects"]["stage2_started"] is False, receipt["side_effects"])
    check("stop_exact", receipt["stop"] == "STOP_FOR_CLAUDE_VERIFICATION", receipt["stop"])
    receipt_without_hash = dict(receipt)
    observed_receipt_hash = receipt_without_hash.pop("receipt_hash", None)
    check(
        "receipt_hash_recomputes",
        stable_hash(receipt_without_hash) == observed_receipt_hash,
        observed_receipt_hash,
    )
    manifest = load_json(ROOT / receipt["models"]["manifest"])
    path_seed_models = [
        seed_state
        for head in manifest["path_heads"]
        for seed_state in head["seed_models"]
    ]
    expected_seed_models = [
        seed_state
        for head in manifest["expected_gate_heads"]
        for seed_state in head["seed_models"]
    ]
    model_records: list[tuple[str, str]] = []
    for seed_state in path_seed_models:
        model_records.extend(
            (
                (seed_state["lower_model"], seed_state["lower_model_sha256"]),
                (
                    seed_state["upper_increment_model"],
                    seed_state["upper_increment_model_sha256"],
                ),
            )
        )
    for seed_state in expected_seed_models:
        model_records.append((seed_state["model"], seed_state["model_sha256"]))
    for seed_state in manifest["action_heads"]["regret_seed_models"]:
        model_records.extend(
            (
                (seed_state["mean_model"], seed_state["mean_model_sha256"]),
                (seed_state["tail_model"], seed_state["tail_model_sha256"]),
            )
        )
    for seed_state in manifest["action_heads"]["wait_seed_models"]:
        model_records.append((seed_state["model"], seed_state["model_sha256"]))
    check("component_model_inventory_309", len(model_records) == 309, len(model_records))
    bad_models = [
        relative
        for relative, expected in model_records
        if not (ROOT / relative).is_file() or sha256_path(ROOT / relative) != expected
    ]
    check("all_component_model_hashes_match", not bad_models, bad_models[:10])
    check(
        "signed_raw_intervals_never_cross",
        sum(
            seed_state["calibration_diagnostics"]["raw_monotonicity_violations"]
            for seed_state in path_seed_models
        )
        == 0,
        len(path_seed_models),
    )
    check(
        "calibrated_intervals_never_cross",
        sum(
            seed_state["calibration_diagnostics"]["calibrated_interval_crossings"]
            for seed_state in path_seed_models
        )
        == 0,
        len(path_seed_models),
    )
    expected_config_without_seed = {
        "l2_regularization": 1.0,
        "learning_rate": 0.05,
        "max_depth": 4,
        "max_examples": 1_000_000,
        "max_iter": 200,
        "min_samples_leaf": 30,
    }
    configs_exact = True
    fit_sessions_exact = True
    for seed_state in path_seed_models:
        for summary_key in ("q10_fit_summary", "q90_increment_fit_summary"):
            config = dict(seed_state[summary_key]["config"])
            seed = config.pop("seed")
            configs_exact &= seed in SEEDS and config == expected_config_without_seed
            fit_sessions_exact &= seed_state[summary_key]["fit_sessions"] == receipt["partitions"]["fit"]
    expected_configs_exact = True
    expected_fit_sessions_exact = True
    for seed_state in expected_seed_models:
        config = dict(seed_state["fit_summary"]["config"])
        seed = config.pop("seed")
        expected_configs_exact &= (
            seed in SEEDS and config == expected_config_without_seed
        )
        expected_fit_sessions_exact &= (
            seed_state["fit_summary"]["fit_sessions"]
            == receipt["partitions"]["fit"]
        )
    check("all_path_configs_exactly_preregistered", configs_exact, expected_config_without_seed)
    check("all_path_fit_populations_exclude_embargo", fit_sessions_exact, receipt["partitions"]["fit"])
    check(
        "all_expected_gate_configs_exactly_preregistered",
        expected_configs_exact,
        expected_config_without_seed,
    )
    check(
        "all_expected_gate_fit_populations_exclude_embargo",
        expected_fit_sessions_exact,
        receipt["partitions"]["fit"],
    )
    expected_calibration_states = [
        head["calibration_state"] for head in manifest["expected_gate_heads"]
    ]
    check(
        "expected_gate_uses_session_clustered_mean_lower_confidence",
        len(expected_calibration_states) == 28
        and all(
            state["method"]
            == "deterministic_session_cluster_bootstrap_lower_confidence_correction"
            and state["distinct_calibration_sessions"] >= 20
            and state["bootstrap_replicates"] == 2_000
            and state["individual_outcome_quantile_used"] is False
            for state in expected_calibration_states
        ),
        len(expected_calibration_states),
    )
    tensor_identity = pd.read_parquet(ROOT / receipt["artifacts"]["tensor"]["path"], columns=list(IDENTITY))
    check(
        "tensor_exact_identity_unique",
        not tensor_identity.duplicated(list(IDENTITY)).any(),
        len(tensor_identity),
    )
    check(
        "tensor_session_order_matches_frozen_full45",
        tensor_identity["session"].drop_duplicates().astype(str).tolist()
        == [
            *receipt["partitions"]["fit"],
            *receipt["partitions"]["embargo"],
            *receipt["partitions"]["calibration"],
            *receipt["partitions"]["replay"],
        ],
        tensor_identity["session"].nunique(),
    )
    rlac_roles = pd.read_parquet(
        ROOT / receipt["artifacts"]["rlac_targets"]["path"],
        columns=["session", "role"],
    )
    check(
        "rlac_roles_exclude_embargo",
        not set(receipt["partitions"]["embargo"])
        & set(rlac_roles["session"].astype(str)),
        sorted(rlac_roles["role"].unique().tolist()),
    )
    inference = pd.read_parquet(ROOT / receipt["artifacts"]["replay_predictions"]["path"])
    present_forbidden = sorted(set(inference.columns) & INFERENCE_FORBIDDEN_EXACT)
    check("inference_parquet_has_no_future_or_label_fields", not present_forbidden, present_forbidden)
    check(
        "inference_parquet_has_only_replay_sessions",
        inference["session"].drop_duplicates().astype(str).tolist()
        == receipt["partitions"]["replay"],
        inference["session"].drop_duplicates().astype(str).tolist(),
    )
    variants = pd.read_csv(ROOT / receipt["artifacts"]["variant_decisions"]["path"])
    variant_counts = variants.groupby(["session", "decision_time_ns"]).size()
    observed_grid = {
        (float(alpha), float(k))
        for alpha, k in variants[["guardrail_alpha", "uncertainty_multiplier_k"]].drop_duplicates().itertuples(index=False, name=None)
    }
    check("every_replay_decision_has_all_20_variants", bool((variant_counts == 20).all()), variant_counts.value_counts().to_dict())
    check("composer_grid_exact", observed_grid == {(alpha, k) for alpha in ALPHAS for k in KS}, sorted(observed_grid))
    diagnostic = load_json(ROOT / receipt["diagnostic"]["path"])
    check(
        "amended_expected_upside_gate_is_empirically_satisfiable",
        diagnostic["near_miss_analysis"]["expected_lower_four_axis_pass_rows"] > 0,
        diagnostic["near_miss_analysis"]["expected_lower_four_axis_pass_rows"],
    )
    check(
        "q10_remains_ranking_diagnostic_not_positive_gate",
        diagnostic["near_miss_analysis"]["q10_calibrated_four_axis_pass_rows"] == 0,
        diagnostic["near_miss_analysis"]["q10_calibrated_four_axis_pass_rows"],
    )
    downstream = diagnostic.get("downstream_gate_binding_analysis", {})
    check(
        "downstream_binding_is_fully_reported",
        downstream.get("proposed_decisions_after_expected_upside_gate")
        == diagnostic["near_miss_analysis"][
            "expected_lower_four_axis_pass_decisions"
        ]
        and set(downstream.get("individual_strict_pass_counts", {}))
        == {
            "uncertainty_cluster_gap",
            "unchanged_q10_mfe_error_margin",
            "q90_regret_upper_bound",
            "mandatory_action_conditioned_gate",
        },
        downstream,
    )
    for name, artifact in receipt["artifacts"].items():
        path = ROOT / artifact["path"]
        check(f"artifact_{name}_exists", path.is_file(), artifact["path"])
        check(f"artifact_{name}_hash_matches", path.is_file() and sha256_path(path) == artifact["sha256"], artifact["sha256"])
    review = {
        "schema_version": "Protocol101WalkingSkeletonOptionDA6Stage1DeltaReviewV1",
        "scope": "A6 entry-gate amendment plus Stage-1 faithful capped rerun only",
        "outcome": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "checks": checks,
        "failed_checks": [item["name"] for item in checks if not item["passed"]],
        "stage2_reviewed_or_started": False,
        "stop": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    if write:
        _write_json(REVIEW, review)
    return review


def repair_existing_inference_artifact_and_reaudit() -> dict[str, Any]:
    """Repair the post-run inference package without refitting or recomposing."""

    if not RECEIPT.is_file():
        raise FileNotFoundError(RECEIPT)
    frame = pd.read_parquet(REPLAY_PREDICTIONS)
    stripped = _label_stripped(frame)
    stripped.to_parquet(REPLAY_PREDICTIONS, index=False)
    receipt = load_json(RECEIPT)
    present = sorted(set(stripped.columns) & INFERENCE_FORBIDDEN_EXACT)
    receipt["inference_artifacts"] = {
        "replay_predictions_label_stripped": not present,
        "forbidden_columns_present": present,
        "forbidden_field_contract": sorted(INFERENCE_FORBIDDEN_EXACT),
    }
    receipt["artifacts"]["replay_predictions"] = _artifact(REPLAY_PREDICTIONS)
    receipt["postrun_packaging_repairs"] = [
        {
            "scope": "replay inference parquet only",
            "reason": "remove t+1 fill and hold-flat outcome fields left by the internal simulator frame",
            "model_refit": False,
            "recalibration": False,
            "composer_rerun": False,
            "economic_outcome_changed": False,
        }
    ]
    receipt.pop("receipt_hash", None)
    receipt["receipt_hash"] = stable_hash(receipt)
    _write_json(RECEIPT, receipt)
    review = _audit(receipt, write=True)
    if review["outcome"] != "PASS":
        raise RuntimeError(f"post-run delta review failed: {review['failed_checks']}")
    return receipt


def refresh_existing_near_miss_diagnostic_and_reaudit() -> dict[str, Any]:
    """Refresh report-only diagnostics; never touch model decisions."""

    receipt = load_json(RECEIPT)
    diagnostic = load_json(DIAGNOSTIC_JSON)
    replay = pd.read_parquet(REPLAY_PREDICTIONS)
    headline = pd.read_csv(HEADLINE_DECISIONS)
    diagnostic["near_miss_analysis"] = _near_misses(replay)
    diagnostic["downstream_gate_binding_analysis"] = _downstream_binding_analysis(
        headline, diagnostic["action_conditioned_gate"]
    )
    _write_json(DIAGNOSTIC_JSON, diagnostic)
    DIAGNOSTIC_MD.write_text(_diagnostic_markdown(diagnostic))
    receipt["diagnostic"]["sha256"] = sha256_path(DIAGNOSTIC_JSON)
    receipt["artifacts"]["diagnostic_json"] = _artifact(DIAGNOSTIC_JSON)
    receipt["artifacts"]["diagnostic_markdown"] = _artifact(DIAGNOSTIC_MD)
    receipt["artifacts"]["implementation_report"] = _artifact(
        IMPLEMENTATION_REPORT
    )
    packaging_record = {
        "scope": "report-only amended-gate and downstream-binding diagnostics",
        "reason": "record individual downstream pass counts after the A6 expected-upside gate without changing governed sequential decisions",
        "model_refit": False,
        "recalibration": False,
        "composer_rerun": False,
        "economic_outcome_changed": False,
    }
    repairs = receipt.setdefault("postrun_packaging_repairs", [])
    if packaging_record not in repairs:
        repairs.append(packaging_record)
    receipt.pop("receipt_hash", None)
    receipt["receipt_hash"] = stable_hash(receipt)
    _write_json(RECEIPT, receipt)
    review = _audit(receipt, write=True)
    if review["outcome"] != "PASS":
        raise RuntimeError(f"diagnostic refresh delta review failed: {review['failed_checks']}")
    return receipt


def run() -> dict[str, Any]:
    STAGE1.mkdir(parents=True, exist_ok=True)
    if RECEIPT.is_file() and REVIEW.is_file() and load_json(REVIEW).get("outcome") == "PASS":
        return load_json(RECEIPT)
    frozen = _verify_frozen_authority()
    prereg = frozen["preregistration"]
    partition = _partition(prereg)
    _log("authority_and_stage0_verified")

    tensor, tensor_summary = _build_or_load_tensor(partition)
    _log("full45_tensor_ready", rows=len(tensor), sha256=sha256_path(TENSOR))
    frames = {
        role: tensor[tensor["session"].isin(sessions)].copy().reset_index(drop=True)
        for role, sessions in partition.items()
    }
    cdfs = _build_or_load_cdfs(frames["fit"])
    _log("fit_nested_cdfs_ready", state_count=len(cdfs), sha256=sha256_path(CDF))
    modeled = _attach_or_load_rlac(frames, cdfs)
    _log("rlac_targets_ready", sha256=sha256_path(RLAC))
    del tensor, frames
    gc.collect()

    budget = Budget()
    fit_out, cal_out, replay_out, path_heads = _fit_path_heads(
        modeled["fit"], modeled["calibration"], modeled["replay"], budget
    )
    fit_out, cal_out, replay_out, expected_gate_heads = _fit_expected_gate_heads(
        modeled["fit"],
        modeled["calibration"],
        modeled["replay"],
        fit_out,
        cal_out,
        replay_out,
        budget,
    )
    cal_out, replay_out, action_heads = _fit_action_heads(
        modeled["fit"],
        modeled["calibration"],
        modeled["replay"],
        fit_out,
        cal_out,
        replay_out,
        budget,
    )
    model_fit_seconds = budget.elapsed
    manifest = {
        "schema_version": "Protocol101WalkingSkeletonOptionDA6EntryModelManifestV1",
        "product_contract_hash": AUTHORITY_SHA256,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "artifact_prefix": ARTIFACT_PREFIX,
        "config_id": MODEL_CONFIG_ID,
        "preregistered_config_count": 1,
        "config_cap": 5,
        "seeds": list(SEEDS),
        "family": "sklearn HistGradientBoosting tabular only",
        "neural_used": False,
        "gpu_used": False,
        "primary_head_count": 39,
        "signed_path_pair_count": len(path_heads),
        "expected_gate_support_head_count": len(expected_gate_heads),
        "expected_gate_support_component_model_count": len(expected_gate_heads)
        * len(SEEDS),
        "component_model_count": (
            len(path_heads) * 2 * len(SEEDS)
            + len(expected_gate_heads) * len(SEEDS)
            + 3 * len(SEEDS)
        ),
        "path_heads": path_heads,
        "expected_gate_heads": expected_gate_heads,
        "action_heads": action_heads,
        "fit_wallclock_seconds": model_fit_seconds,
        "wallclock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "formal_model_quality_claim": False,
    }
    _write_json(MANIFEST, manifest)
    _log("model_manifest_frozen", fit_wallclock_seconds=round(model_fit_seconds, 3))

    action_gate = _action_gate(modeled["calibration"], cal_out)
    errors = _calibrate_composer_errors(modeled["calibration"], cal_out, cdfs)
    all_decisions, headline, pre_gate = _compose_variants(
        replay_out, cdfs, errors, action_gate["available"]
    )
    _write_csv(VARIANT_DECISIONS, all_decisions)
    _write_csv(HEADLINE_DECISIONS, headline)
    _log(
        "composer_complete",
        headline_buy=sum(row["action"] == "BUY" for row in headline),
        headline_wait=sum(row["action"] == "WAIT" for row in headline),
    )

    candidates = _stage1_candidates(headline, replay_out)
    trades, state = simulate_serial_candidates_v5(
        candidates,
        config=SerialSimulatorV5Config(split_order=("plumbing_replay",)),
    )
    trade_rows = [asdict(trade) for trade in trades]
    trade_fields = list(trade_rows[0]) if trade_rows else [
        "split", "session", "decision_time_ns", "contract_id", "entry_ask", "raw_label_pnl_after_campaign_fee"
    ]
    _write_csv(TRADES, trade_rows, trade_fields)
    outcome = "TRADES" if trades else "ABSTAINS"

    cal_out.to_parquet(CAL_PREDICTIONS, index=False)
    stripped = _label_stripped(replay_out)
    stripped.to_parquet(REPLAY_PREDICTIONS, index=False)
    forbidden_inference_columns = sorted(set(stripped.columns) & INFERENCE_FORBIDDEN_EXACT)
    diagnostic = _diagnostic(
        outcome, replay_out, headline, pre_gate, action_gate, errors, path_heads, trades
    )
    _write_json(DIAGNOSTIC_JSON, diagnostic)
    DIAGNOSTIC_MD.write_text(_diagnostic_markdown(diagnostic))

    artifacts = {
        "tensor": _artifact(TENSOR),
        "nested_cdfs": _artifact(CDF),
        "rlac_targets": _artifact(RLAC),
        "model_manifest": _artifact(MANIFEST),
        "calibration_predictions": _artifact(CAL_PREDICTIONS),
        "replay_predictions": _artifact(REPLAY_PREDICTIONS),
        "variant_decisions": _artifact(VARIANT_DECISIONS),
        "headline_decisions": _artifact(HEADLINE_DECISIONS),
        "serial_trades": _artifact(TRADES),
        "diagnostic_json": _artifact(DIAGNOSTIC_JSON),
        "diagnostic_markdown": _artifact(DIAGNOSTIC_MD),
    }
    receipt = {
        "schema_version": "Protocol101WalkingSkeletonOptionDA6Stage1ReceiptV1",
        "stage": 1,
        "outcome": outcome,
        "status": "complete",
        "product_contract_hash": AUTHORITY_SHA256,
        "quarantine_labels": list(QUARANTINE_LABELS),
        "artifact_prefix": ARTIFACT_PREFIX,
        "partitions": partition,
        "embargo_excluded_from_fit_calibration_and_replay": True,
        "tensor": {**tensor_summary, "sha256": sha256_path(TENSOR)},
        "models": {
            "manifest": str(MANIFEST.relative_to(ROOT)),
            "preregistered_config_count": 1,
            "config_cap": 5,
            "config_id": MODEL_CONFIG_ID,
            "seeds": list(SEEDS),
            "primary_head_count": 39,
            "signed_path_pair_count": 36,
            "expected_gate_support_head_count": len(expected_gate_heads),
            "expected_gate_support_component_model_count": len(expected_gate_heads)
            * len(SEEDS),
            "component_model_count": manifest["component_model_count"],
            "family": manifest["family"],
            "neural_used": False,
            "gpu_used": False,
            "fit_wallclock_seconds": model_fit_seconds,
            "wallclock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        },
        "calibration": {
            "method": "same-final-model disjoint split conformalized quantile regression",
            "continuous_pair": "[predicted_q10-qhat,predicted_q90+qhat]",
            "nonconformity": "max(predicted_q10-y,y-predicted_q90)",
            "coverage": 0.80,
            "phase_specific_minimum": {"distinct_sessions": 8, "valid_rows": 2000},
            "post_calibration_refit": False,
            "expected_upside_gate": {
                "method": "deterministic session-clustered one-sided 90 percent lower confidence correction for conditional mean",
                "support_head_count": len(expected_gate_heads),
                "bootstrap_replicates_per_head": 2_000,
                "minimum_distinct_calibration_sessions": 20,
                "individual_outcome_quantile_used": False,
                "all_registered_horizons_including_h3_h5": True,
                "strict_fee_adjusted_threshold": 0.0,
                "q10_ranking_unchanged": True,
            },
            "action_conditioned_gate": action_gate,
            "composer_error_margins": errors,
        },
        "composer": {
            "implementation": "real FT2-10 order and frozen values",
            "variant_count": len(ALPHAS) * len(KS),
            "guardrail_anchor_grid": list(ALPHAS),
            "uncertainty_multiplier_grid": list(KS),
            "headline_control": {"guardrail_alpha": 0.0, "k": 1.0},
            "headline_decisions": len(headline),
            "headline_buy": sum(row["action"] == "BUY" for row in headline),
            "headline_wait": sum(row["action"] == "WAIT" for row in headline),
            "fabricated_or_injected_buy": False,
            "threshold_changed": False,
            "bar_changed": False,
            "anchor_changed": False,
        },
        "serial_replay": {
            "candidate_count": len(candidates),
            "trade_count": len(trades),
            "distinct_trade_sessions": len({trade.session for trade in trades}),
            "simulator_version": state.semantics["simulator_version"],
            "candidate_stream_hash": state.candidate_stream_hash,
            "candidate_payload_hash": state.candidate_payload_hash,
            "trade_identity_hash": state.trade_identity_hash,
            "skipped": state.skipped,
            "skipped_events": [asdict(item) for item in state.skipped_events],
            "profile": _trade_profile(trades),
            "stage2_unblocked": bool(trades),
            "stage2_started": False,
        },
        "diagnostic": {
            "tentative_cause": diagnostic["tentative_cause"],
            "path": str(DIAGNOSTIC_JSON.relative_to(ROOT)),
            "sha256": sha256_path(DIAGNOSTIC_JSON),
        },
        "inference_artifacts": {
            "replay_predictions_label_stripped": not forbidden_inference_columns,
            "forbidden_columns_present": forbidden_inference_columns,
        },
        "artifacts": artifacts,
        "side_effects": {
            "throwaway_model_training_executed": True,
            "protected_resource_read": False,
            "broker_contacted": False,
            "paper_submit": False,
            "paid_download": False,
            "promotion_or_default_changed": False,
            "runtime_or_launchd_changed": False,
            "stage2_started": False,
            "stage3_started": False,
            "stage4_started": False,
        },
        "highest_allowed_claim": (
            f"The faithful capped Option-D Stage-1 entry dry-run {outcome.lower()}; "
            "this is quarantined plumbing evidence only, not a scientific, model-quality, or alpha claim."
        ),
        "next_gate": "Claude verification required; no Stage 2 work started",
        "stop": "STOP_FOR_CLAUDE_VERIFICATION",
    }
    receipt["receipt_hash"] = stable_hash(receipt)
    _write_json(RECEIPT, receipt)
    review = _audit(receipt, write=True)
    if review["outcome"] != "PASS":
        raise RuntimeError(f"Option-D delta-scoped review failed: {review['failed_checks']}")
    _log("stage1_complete", outcome=outcome, receipt_hash=receipt["receipt_hash"])
    return receipt


RUN_STARTED = time.monotonic()


if __name__ == "__main__":
    result = run()
    print(json.dumps({"outcome": result["outcome"], "receipt_hash": result["receipt_hash"], "stop": result["stop"]}, indent=2))
