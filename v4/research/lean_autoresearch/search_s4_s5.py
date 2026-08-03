"""Fresh S4/S5 lean entry search with paired attribution and a one-shot holdout.

Run order::

    python -m v4.research.lean_autoresearch.search_s4_s5 --preregister
    python -m v4.research.lean_autoresearch.search_s4_s5 --full
    python -m v4.research.lean_autoresearch.search_s4_s5 --holdout

The 72 S4/S5 candidates and their 72 paired S0/S3 controls use the same fixed
144-configuration budget and the same five disqualifier categories as the
spent S0-S3 experiment. The protected holdout cannot be decoded by the search
path. A model artifact may be written only after a one-shot holdout pass.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from v4.research.lean_autoresearch import harness as H

OUT = Path("v4/audit/autoresearch/lean_entry_autoresearch_s4_s5_2026_08_02")
PREREG = OUT / "preregistration.json"
RESULTS = OUT / "search_results.json"
VERDICT = OUT / "verdict.json"
WINNER_SEAL = OUT / "winner_seal.json"
HOLDOUT_RECEIPT = OUT / "holdout_access_receipt.json"
HOLDOUT_RESULT = OUT / "holdout_result.json"
MODEL_DIR = OUT / "confirmed_entry_model"
MODEL_ARTIFACT = MODEL_DIR / "model.joblib"
MODEL_MANIFEST = MODEL_DIR / "manifest.json"
REPORT = OUT / "report.md"

PREVIOUS_PREREG = Path(
    "v4/audit/autoresearch/lean_entry_autoresearch_2026_08_02/preregistration.json"
)
HARNESS_SOURCE = Path("v4/research/lean_autoresearch/harness.py")
SEARCH_SOURCE = Path(__file__)
SESSION_ASSIGNMENTS = Path(H.SESSION_ASSIGNMENTS)

BUDGET = 144
CANDIDATE_TRIALS = 72
PAIRED_CONTROL_TRIALS = 72
ALPHA = 0.05
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 5105
SELECTION_METRIC = "worst_fold_econ_then_incremental_econ"
FEATURE_SETS = ("S4", "S5")
PAIRED_BASELINE = {"S4": "S0", "S5": "S3"}
SELF_HASH_FIELDS = {
    "lean_autoresearch.s4_s5.preregistration.v1": "preregistration_sha256",
    "lean_autoresearch.s4_s5.search_results.v1": "search_results_sha256",
    "lean_autoresearch.s4_s5.verdict.v1": "verdict_sha256",
    "lean_autoresearch.s4_s5.winner_seal.v1": "winner_seal_sha256",
    "lean_autoresearch.holdout_access_receipt.v2": "receipt_sha256",
    "lean_autoresearch.s4_s5.holdout_result.v1": "holdout_result_sha256",
    "lean_autoresearch.s4_s5.model_manifest.v1": "manifest_sha256",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_self_hash(doc: dict, field: str) -> dict:
    if field in doc:
        raise ValueError(f"self-hash field already present: {field}")
    out = dict(doc)
    out[field] = _sha256(doc)
    return out


def _write_json_exclusive(path: Path, doc: dict, *, self_hash_field: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _with_self_hash(doc, self_hash_field)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
            handle.write("\n")
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    return payload


def _read_verified_json(path: Path, *, schema: str) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != schema:
        raise RuntimeError(f"unexpected schema in {path}")
    field = SELF_HASH_FIELDS[schema]
    semantic = dict(doc)
    observed = semantic.pop(field, None)
    expected = _sha256(semantic)
    if observed != expected:
        raise RuntimeError(f"self-hash mismatch in {path}")
    return doc


def build_candidate_grid() -> list[dict]:
    return [
        {
            "feature_set": feature_set,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "threshold": threshold,
            "exit_policy": exit_policy,
        }
        for feature_set, max_depth, min_samples_leaf, threshold, exit_policy in itertools.product(
            FEATURE_SETS,
            (3, 4),
            (30, 50),
            ("gt0", "gt_fee_reserve", "gt_session_q60"),
            H.EXIT_POLICIES,
        )
    ]


def paired_control_config(candidate: dict) -> dict:
    out = dict(candidate)
    out["feature_set"] = PAIRED_BASELINE[candidate["feature_set"]]
    return out


def build_control_grid() -> list[dict]:
    return [paired_control_config(candidate) for candidate in build_candidate_grid()]


def _development_sessions() -> list[str]:
    excluded = H.HOLDOUT | frozenset(H._SA.get("opra_degraded_diagnostic_only", []))
    return [
        session
        for session in H._SA["pre_holdout_session_indices_1_215"]
        if session not in excluded
    ]


def development_corpus_fingerprint() -> dict:
    """Hash exact development feature sources; protected files are never opened here."""

    files = []
    for session in _development_sessions():
        sources = {
            "processed_minute_entry": Path(H.PROC) / f"{session}.pkl",
            "official_spx_1m": Path(H.OFFICIAL_INDEX) / "spx_1m" / f"{session}.parquet",
            "official_vix_1m": Path(H.OFFICIAL_INDEX) / "vix_1m" / f"{session}.parquet",
        }
        for source, path in sources.items():
            if not path.is_file():
                raise RuntimeError(f"missing development corpus file: {path}")
            files.append(
                {
                    "session": session,
                    "source": source,
                    "size_bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {
        "corpus_root": str(Path(H.CORP).resolve()),
        "development_session_count": len(_development_sessions()),
        "source_file_count": len(files),
        "files_sha256": _sha256(files),
    }


def _source_hashes() -> dict[str, str]:
    return {
        "harness.py": _file_sha256(HARNESS_SOURCE),
        "search_s4_s5.py": _file_sha256(SEARCH_SOURCE),
        "session_assignments.json": _file_sha256(SESSION_ASSIGNMENTS),
        "spent_s0_s3_preregistration.json": _file_sha256(PREVIOUS_PREREG),
    }


def preregister() -> dict:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"fresh preregistration output is not empty: {OUT}")
    candidates = build_candidate_grid()
    controls = build_control_grid()
    if len(candidates) != CANDIDATE_TRIALS or len(controls) != PAIRED_CONTROL_TRIALS:
        raise AssertionError("S4/S5 grid no longer matches its fixed budget")
    if len({_sha256(item) for item in candidates + controls}) != BUDGET:
        raise AssertionError("candidate/control grid contains duplicates")

    corpus = development_corpus_fingerprint()
    doc = {
        "schema_version": "lean_autoresearch.s4_s5.preregistration.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Fresh causal volatility/regime/time entry hypothesis; NULL is first-class; "
            "research-only and not promotable"
        ),
        "supersedes_spent_grid": str(PREVIOUS_PREREG),
        "hard_rule": (
            "No entry-model artifact unless one candidate passes all development guards, "
            "is selected before holdout access, and passes the one-shot holdout bar"
        ),
        "budget": {
            "total_evaluated_configurations": BUDGET,
            "new_candidate_configurations": CANDIDATE_TRIALS,
            "paired_attribution_control_configurations": PAIRED_CONTROL_TRIALS,
            "multiplicity_denominator": BUDGET,
        },
        "feature_hypothesis": {
            "S4": H.FEATURE_SETS["S4"],
            "S5": H.FEATURE_SETS["S5"],
            "new_family": H.VOLATILITY_TIME,
            "paired_baseline": PAIRED_BASELINE,
            "role": "model inputs only; never gates",
            "causal_source": (
                "Official ThetaData SPX/VIX 1m history rebuilt under v3.2 BAR_OPEN_TIMESTAMP "
                "+60-second availability; both current source clocks must be <= decision_time"
            ),
            "formulas": {
                "vix_level": "current completed-minute vix_close",
                "vix_change_5m": "vix[t] - vix[t-5]",
                "vix_change_15m": "vix[t] - vix[t-15]",
                "vix_realized_spread": (
                    "vix[t] - 100*sqrt(mean(last 15 one-minute SPX log-return^2)*252*390)"
                ),
                "time": "New_York minutes since 09:30 and minutes remaining to 15:55",
                "spx_realized_vol": (
                    "5m/15m annualized RMS log-return volatility using completed minutes"
                ),
                "range_expansion": (
                    "(cumulative session_range[t]-session_range[t-5 or t-15])/SPX[t]*1e4"
                ),
            },
            "live_twin": (
                "v4.live.protocol101_live_entry LiveIndexState.market_window supplies the same "
                "completed-minute SPX/VIX values and derives cumulative session_range with a "
                "completed-minute cutoff; exact historical selection is event_time+60<=decision"
            ),
        },
        "search_space": {
            "feature_sets": list(FEATURE_SETS),
            "max_depth": [3, 4],
            "min_samples_leaf": [30, 50],
            "threshold": ["gt0", "gt_fee_reserve", "gt_session_q60"],
            "exit_policy": list(H.EXIT_POLICIES),
        },
        "guards": {
            "G1_controls_and_feature_attribution": (
                "candidate mean OOF rho > shuffled +0.01; candidate mean econ > shuffled and "
                "sign-reversed; candidate mean rho > paired old-family baseline +0.01; candidate "
                "mean econ > paired baseline; incremental rho/econ nonnegative in fold5; and "
                "incremental econ >0 in >=4/5 folds"
            ),
            "G2_forward_stability": (
                "fold5 candidate rho>=0 and econ>=0; candidate rho>0 in >=4/5 and econ>0 in >=4/5"
            ),
            "G3_minimum_power": "min over folds n_enter>=40 and n_sessions>=20",
            "G4_economic_bar": "mean candidate econ>0 and candidate econ>0 in >=4/5 folds",
            "G5_leakage_tripwire": "max absolute fold OOF rho<=0.5",
        },
        "selection": {
            "eligible": "all five guards pass",
            "metric": SELECTION_METRIC,
            "ordering": (
                "descending worst-fold econ, descending mean paired incremental econ, "
                "descending mean OOF rho, then ascending canonical config SHA-256"
            ),
        },
        "holdout_protocol": {
            "sealed_during_search": True,
            "sessions": "protected_holdout_primary_non_degraded_29",
            "open_count_allowed": 1,
            "winner_frozen_before_access": True,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "pass_bar": (
                f"OOF-style holdout rho>0, session-clustered econ>0, n_enter>=40, "
                f"n_sessions>=20, and conservative one-sided bootstrap LCB at "
                f"alpha={ALPHA}/{BUDGET}>0"
            ),
            "model_artifact_rule": "write only after pass; otherwise no model artifact",
        },
        "candidate_grid_sha256": _sha256(candidates),
        "paired_control_grid_sha256": _sha256(controls),
        "source_hashes": _source_hashes(),
        "development_corpus": corpus,
    }
    frozen = _write_json_exclusive(
        PREREG, doc, self_hash_field="preregistration_sha256"
    )
    print(f"froze S4/S5 preregistration -> {PREREG}")
    print(f"  prereg_sha256={frozen['preregistration_sha256']}")
    print(
        f"  budget={BUDGET} ({CANDIDATE_TRIALS} new + "
        f"{PAIRED_CONTROL_TRIALS} paired controls), holdout opens=0"
    )
    return frozen


def validate_frozen_foundation() -> dict:
    """Fail before decoding any fold if code or development bytes drifted."""

    prereg = _read_verified_json(
        PREREG, schema="lean_autoresearch.s4_s5.preregistration.v1"
    )
    if prereg["candidate_grid_sha256"] != _sha256(build_candidate_grid()):
        raise RuntimeError("FOUNDATION DRIFT: candidate grid mismatch")
    if prereg["paired_control_grid_sha256"] != _sha256(build_control_grid()):
        raise RuntimeError("FOUNDATION DRIFT: paired-control grid mismatch")
    if prereg["source_hashes"] != _source_hashes():
        raise RuntimeError("FOUNDATION DRIFT: source hash mismatch")
    observed_corpus = development_corpus_fingerprint()
    if prereg["development_corpus"] != observed_corpus:
        raise RuntimeError("FOUNDATION DRIFT: development corpus mismatch")
    print(
        "foundation-stability gate PASS: source and development corpus reproduce "
        "byte-identically; protected holdout was not opened"
    )
    return prereg


def apply_guards(candidate: dict, paired: dict) -> dict:
    pf = candidate["per_fold"]
    bf = paired["per_fold"]
    rho = np.asarray([fold["oof_rho"] for fold in pf], dtype=float)
    econ = np.asarray([fold["econ"] for fold in pf], dtype=float)
    shuffled_rho = np.asarray([fold["shuffled_rho"] for fold in pf], dtype=float)
    shuffled_econ = np.asarray([fold["shuffled_econ"] for fold in pf], dtype=float)
    signrev_econ = np.asarray([fold["signrev_econ"] for fold in pf], dtype=float)
    baseline_rho = np.asarray([fold["oof_rho"] for fold in bf], dtype=float)
    baseline_econ = np.asarray([fold["econ"] for fold in bf], dtype=float)
    n_enter = np.asarray([fold["n_enter"] for fold in pf], dtype=int)
    n_sessions = np.asarray([fold["n_sessions"] for fold in pf], dtype=int)
    delta_rho = rho - baseline_rho
    delta_econ = econ - baseline_econ

    g1_negative = bool(
        rho.mean() > shuffled_rho.mean() + 0.01
        and np.nanmean(econ) > np.nanmean(shuffled_econ)
        and np.nanmean(econ) > np.nanmean(signrev_econ)
    )
    g1_attribution = bool(
        rho.mean() > baseline_rho.mean() + 0.01
        and np.nanmean(econ) > np.nanmean(baseline_econ)
        and delta_rho[-1] >= 0
        and delta_econ[-1] >= 0
        and int((np.nan_to_num(delta_econ) > 0).sum()) >= 4
    )
    guards = {
        "G1_controls_and_feature_attribution": g1_negative and g1_attribution,
        "G2_forward_stability": bool(
            rho[-1] >= 0
            and econ[-1] >= 0
            and int((rho > 0).sum()) >= 4
            and int((np.nan_to_num(econ) > 0).sum()) >= 4
        ),
        "G3_minimum_power": bool(n_enter.min() >= 40 and n_sessions.min() >= 20),
        "G4_economic_bar": bool(
            np.nanmean(econ) > 0 and int((np.nan_to_num(econ) > 0).sum()) >= 4
        ),
        "G5_leakage_tripwire": bool(np.abs(rho).max() <= 0.5),
    }
    reasons = [name for name, passed in guards.items() if not passed]
    return {
        "passed": not reasons,
        "guards": guards,
        "reasons": reasons,
        "g1_negative_controls_passed": g1_negative,
        "g1_paired_attribution_passed": g1_attribution,
        "mean_oof_rho": float(rho.mean()),
        "fold5_oof_rho": float(rho[-1]),
        "mean_econ": float(np.nanmean(econ)),
        "worst_fold_econ": float(np.nanmin(econ)),
        "mean_paired_baseline_rho": float(baseline_rho.mean()),
        "mean_paired_baseline_econ": float(np.nanmean(baseline_econ)),
        "mean_incremental_rho": float(delta_rho.mean()),
        "mean_incremental_econ": float(np.nanmean(delta_econ)),
        "fold5_incremental_rho": float(delta_rho[-1]),
        "fold5_incremental_econ": float(delta_econ[-1]),
        "positive_incremental_econ_folds": int((np.nan_to_num(delta_econ) > 0).sum()),
        "min_n_enter": int(n_enter.min()),
        "min_n_sessions": int(n_sessions.min()),
        "mean_shuffled_rho": float(shuffled_rho.mean()),
        "mean_shuffled_econ": float(np.nanmean(shuffled_econ)),
        "mean_signrev_econ": float(np.nanmean(signrev_econ)),
    }


def run_search(configs: list[dict]) -> list[dict]:
    out = []
    started = time.time()
    for index, config in enumerate(configs, 1):
        baseline_config = paired_control_config(config)
        baseline = H.evaluate_config(baseline_config, include_shuffled_control=False)
        candidate = H.evaluate_config(config, include_shuffled_control=True)
        guard = apply_guards(candidate, baseline)
        row = {
            "index": index,
            "config": config,
            **guard,
            "per_fold": candidate["per_fold"],
            "paired_control": {
                "config": baseline_config,
                "per_fold": baseline["per_fold"],
            },
        }
        out.append(row)
        status = "PASS" if guard["passed"] else "x"
        print(
            f"[{index:>2}/{len(configs)}] {config['feature_set']} "
            f"d{config['max_depth']} l{config['min_samples_leaf']} "
            f"{config['threshold']:>15} {config['exit_policy'][11:26]:>15} | "
            f"rho {guard['mean_oof_rho']:+.3f} dR {guard['mean_incremental_rho']:+.3f} "
            f"econ {guard['mean_econ']:+.1f} dE {guard['mean_incremental_econ']:+.1f} | "
            f"{status} {','.join(guard['reasons'])} ({time.time() - started:.0f}s)",
            flush=True,
        )
    return out


def _winner_order(row: dict) -> tuple:
    return (
        -row["worst_fold_econ"],
        -row["mean_incremental_econ"],
        -row["mean_oof_rho"],
        _sha256(row["config"]),
    )


def summarize_and_write(results: list[dict], prereg: dict) -> dict:
    if RESULTS.exists() or VERDICT.exists() or WINNER_SEAL.exists():
        raise RuntimeError("search is immutable and cannot be rerun in this output directory")
    results_doc = {
        "schema_version": "lean_autoresearch.s4_s5.search_results.v1",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "candidate_trials": len(results),
        "paired_control_trials": len(results),
        "holdout_opened": False,
        "results": results,
    }
    frozen_results = _write_json_exclusive(
        RESULTS, results_doc, self_hash_field="search_results_sha256"
    )

    survivors = sorted((row for row in results if row["passed"]), key=_winner_order)
    winner = None
    if survivors:
        selected = survivors[0]
        winner = {
            "config": selected["config"],
            "paired_control_config": selected["paired_control"]["config"],
            "mean_oof_rho": selected["mean_oof_rho"],
            "fold5_oof_rho": selected["fold5_oof_rho"],
            "mean_econ": selected["mean_econ"],
            "worst_fold_econ": selected["worst_fold_econ"],
            "mean_incremental_rho": selected["mean_incremental_rho"],
            "mean_incremental_econ": selected["mean_incremental_econ"],
            "fold5_incremental_econ": selected["fold5_incremental_econ"],
        }
    verdict_doc = {
        "schema_version": "lean_autoresearch.s4_s5.verdict.v1",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "search_results_sha256": frozen_results["search_results_sha256"],
        "candidate_trials": len(results),
        "paired_control_trials": len(results),
        "n_survivors": len(survivors),
        "selection_metric": SELECTION_METRIC,
        "holdout_opened": False,
        "result": "NULL_NO_NEW_ENTRY_EDGE" if not survivors else "SURVIVOR_PENDING_HOLDOUT",
        "winner": winner,
        "model_artifact_created": False,
        "next_direction_if_null": "exit/lifecycle",
        "claim_boundary": "research-only; not promotable; no live/broker/runtime change",
    }
    frozen_verdict = _write_json_exclusive(
        VERDICT, verdict_doc, self_hash_field="verdict_sha256"
    )
    if winner:
        seal_doc = {
            "schema_version": "lean_autoresearch.s4_s5.winner_seal.v1",
            "preregistration_sha256": prereg["preregistration_sha256"],
            "search_results_sha256": frozen_results["search_results_sha256"],
            "verdict_sha256": frozen_verdict["verdict_sha256"],
            "selection_metric": SELECTION_METRIC,
            "winner": winner,
            "holdout_opened": False,
        }
        seal = _write_json_exclusive(
            WINNER_SEAL, seal_doc, self_hash_field="winner_seal_sha256"
        )
        print(f"winner sealed before holdout: {seal['winner_seal_sha256']}")

    lines = [
        "# Lean S4/S5 entry autoresearch result",
        "",
        f"- Candidate configurations: {len(results)}",
        f"- Paired old-family controls: {len(results)}",
        f"- Survivors: {len(survivors)}",
        f"- Verdict: `{frozen_verdict['result']}`",
        "- Holdout opens: 0 at search completion",
        "- Model artifact: not created at search completion",
        "",
    ]
    if winner:
        lines.extend(
            [
                "A single winner was frozen before any protected access. The next permitted "
                "operation is the one-shot holdout command.",
                "",
                f"Winner: `{json.dumps(winner['config'], sort_keys=True)}`",
            ]
        )
    else:
        lines.extend(
            [
                "No S4/S5 candidate both survived the five guards and demonstrated a "
                "forward-stable incremental effect over its paired old-family control.",
                "The protected holdout remains sealed, no entry model may be built, and the "
                "pre-registered pivot is exit/lifecycle research.",
            ]
        )
    descriptor = os.open(REPORT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print("\n===== S4/S5 SEARCH VERDICT =====")
    print(
        f"candidates={len(results)} paired_controls={len(results)} "
        f"survivors={len(survivors)} result={frozen_verdict['result']}"
    )
    if winner:
        print("NEXT: --holdout will create the single access receipt before decoding.")
    else:
        print("Holdout remains SEALED. No entry-model build is permitted; pivot exit/lifecycle.")
    return frozen_verdict


def run_full_search() -> dict:
    if RESULTS.exists() or VERDICT.exists():
        raise RuntimeError("full search already completed; outputs are immutable")
    prereg = validate_frozen_foundation()
    results = run_search(build_candidate_grid())
    return summarize_and_write(results, prereg)


def _bootstrap_lcb(session_diffs: np.ndarray) -> float:
    values = np.asarray(session_diffs, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, ALPHA / BUDGET, method="lower"))


def _load_holdout_authority() -> tuple[dict, dict, dict, dict]:
    prereg = validate_frozen_foundation()
    results = _read_verified_json(
        RESULTS, schema="lean_autoresearch.s4_s5.search_results.v1"
    )
    verdict = _read_verified_json(VERDICT, schema="lean_autoresearch.s4_s5.verdict.v1")
    seal = _read_verified_json(WINNER_SEAL, schema="lean_autoresearch.s4_s5.winner_seal.v1")
    if verdict["result"] != "SURVIVOR_PENDING_HOLDOUT" or not verdict["winner"]:
        raise RuntimeError("no pre-committed survivor is authorized for holdout")
    if (
        results["preregistration_sha256"] != prereg["preregistration_sha256"]
        or verdict["search_results_sha256"] != results["search_results_sha256"]
        or seal["verdict_sha256"] != verdict["verdict_sha256"]
        or seal["winner"] != verdict["winner"]
    ):
        raise RuntimeError("holdout authority chain is inconsistent")
    return prereg, results, verdict, seal


def run_one_shot_holdout() -> dict:
    if HOLDOUT_RECEIPT.exists() or HOLDOUT_RESULT.exists():
        raise RuntimeError("one-shot holdout already consumed or attempted; refusing a second open")
    prereg, _, verdict, seal = _load_holdout_authority()
    winner = verdict["winner"]
    config = winner["config"]

    # Prepare and fit the immutable winner using development data before recording access.
    development = H.full_preholdout_frame()
    features = H.FEATURE_SETS[config["feature_set"]]
    target = config["exit_policy"]
    model = H.make_hgb(
        max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"]
    )
    model.fit(development[features].to_numpy(float), development[target].to_numpy(float))

    sessions = list(H._SA["protected_holdout_primary_non_degraded_29"])
    receipt_doc = {
        "schema_version": "lean_autoresearch.holdout_access_receipt.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "open_count": 1,
        "sessions": sessions,
        "sessions_sha256": hashlib.sha256(("\n".join(sessions) + "\n").encode()).hexdigest(),
        "preregistration_sha256": prereg["preregistration_sha256"],
        "winner_seal_sha256": seal["winner_seal_sha256"],
        "winner_config_sha256": _sha256(config),
        "purpose": "single confirmatory S4/S5 entry-edge evaluation",
    }
    receipt = _write_json_exclusive(
        HOLDOUT_RECEIPT, receipt_doc, self_hash_field="receipt_sha256"
    )
    print("HOLDOUT ACCESS RECEIPT WRITTEN: the single open is now consumed", flush=True)

    try:
        holdout = H.load_holdout_frame(access_receipt=receipt)
        predictions = model.predict(holdout[features].to_numpy(float))
        y = holdout[target].to_numpy(float)
        rho_raw = spearmanr(predictions, y).correlation
        rho = float(rho_raw) if rho_raw == rho_raw else 0.0
        econ = H.session_clustered_econ(
            holdout, predictions, target, config["threshold"]
        )
        lcb = _bootstrap_lcb(np.asarray(econ["session_diffs"], dtype=float))
        passed = bool(
            rho > 0
            and econ["econ"] > 0
            and econ["n_enter"] >= 40
            and econ["n_sessions"] >= 20
            and lcb > 0
        )
        result_doc = {
            "schema_version": "lean_autoresearch.s4_s5.holdout_result.v1",
            "status": "PASS_MODEL_BUILD_AUTHORIZED" if passed else "FAIL_NO_MODEL_BUILD",
            "receipt_sha256": receipt["receipt_sha256"],
            "winner_seal_sha256": seal["winner_seal_sha256"],
            "config": config,
            "oof_style_rho": rho,
            "session_clustered_econ": econ["econ"],
            "n_enter": econ["n_enter"],
            "n_sessions": econ["n_sessions"],
            "session_diffs": econ["session_diffs"],
            "bootstrap_lcb": lcb,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "alpha": ALPHA / BUDGET,
            "passed": passed,
            "model_artifact_authorized": passed,
        }
    except BaseException as exc:
        failure_doc = {
            "schema_version": "lean_autoresearch.s4_s5.holdout_result.v1",
            "status": "ACCESS_FAILED_AFTER_RECEIPT_NO_MODEL_BUILD",
            "receipt_sha256": receipt["receipt_sha256"],
            "winner_seal_sha256": seal["winner_seal_sha256"],
            "config": config,
            "passed": False,
            "model_artifact_authorized": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _write_json_exclusive(
            HOLDOUT_RESULT, failure_doc, self_hash_field="holdout_result_sha256"
        )
        raise

    frozen_result = _write_json_exclusive(
        HOLDOUT_RESULT, result_doc, self_hash_field="holdout_result_sha256"
    )
    print(
        f"holdout {frozen_result['status']}: rho={rho:+.4f} econ={econ['econ']:+.2f} "
        f"LCB={lcb:+.2f} sessions={econ['n_sessions']} enters={econ['n_enter']}"
    )
    if passed:
        _write_model_artifact(model, prereg, seal, frozen_result)
    else:
        print("No model artifact created. Pre-registered pivot: exit/lifecycle.")
    return frozen_result


def _write_model_artifact(model, prereg: dict, seal: dict, holdout_result: dict) -> dict:
    if not holdout_result.get("passed") or not holdout_result.get("model_artifact_authorized"):
        raise RuntimeError("holdout did not authorize a model artifact")
    if MODEL_ARTIFACT.exists() or MODEL_MANIFEST.exists():
        raise RuntimeError("confirmed model artifact already exists")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_DIR / "model.joblib.tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, MODEL_ARTIFACT)
    config = holdout_result["config"]
    manifest_doc = {
        "schema_version": "lean_autoresearch.s4_s5.model_manifest.v1",
        "claim_boundary": "research-only confirmed entry model; not runtime/promotable",
        "config": config,
        "features": H.FEATURE_SETS[config["feature_set"]],
        "target": config["exit_policy"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "winner_seal_sha256": seal["winner_seal_sha256"],
        "holdout_result_sha256": holdout_result["holdout_result_sha256"],
        "artifact": MODEL_ARTIFACT.name,
        "artifact_sha256": _file_sha256(MODEL_ARTIFACT),
        "live_broker_runtime_changes": False,
    }
    manifest = _write_json_exclusive(
        MODEL_MANIFEST, manifest_doc, self_hash_field="manifest_sha256"
    )
    print(f"confirmed research model created -> {MODEL_ARTIFACT}")
    return manifest


def build_confirmed_model() -> dict:
    """Recover artifact creation after a post-result serialization interruption."""

    prereg, _, _, seal = _load_holdout_authority()
    result = _read_verified_json(
        HOLDOUT_RESULT, schema="lean_autoresearch.s4_s5.holdout_result.v1"
    )
    if not result.get("passed"):
        raise RuntimeError("holdout did not pass; model build remains forbidden")
    config = result["config"]
    development = H.full_preholdout_frame()
    model = H.make_hgb(
        max_depth=config["max_depth"], min_samples_leaf=config["min_samples_leaf"]
    )
    features = H.FEATURE_SETS[config["feature_set"]]
    target = config["exit_policy"]
    model.fit(development[features].to_numpy(float), development[target].to_numpy(float))
    return _write_model_artifact(model, prereg, seal, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--preregister", action="store_true")
    operations.add_argument("--verify-foundation", action="store_true")
    operations.add_argument("--full", action="store_true")
    operations.add_argument("--holdout", action="store_true")
    operations.add_argument("--build-confirmed", action="store_true")
    args = parser.parse_args()
    if args.preregister:
        preregister()
    elif args.verify_foundation:
        validate_frozen_foundation()
    elif args.full:
        run_full_search()
    elif args.holdout:
        run_one_shot_holdout()
    elif args.build_confirmed:
        build_confirmed_model()


if __name__ == "__main__":
    main()
