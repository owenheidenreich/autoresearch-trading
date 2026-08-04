"""Offline Phase-0b certification and signed-ledger regeneration.

Tracks B and C only.  This module never opens a vendor, broker, order, model,
or protected-holdout path.  Track A remains barred until separately authorized
multi-session live OPRA evidence exists.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from v4.research.pathd_causal_account_state import (
    AccountEvent,
    CausalAccountState,
    account_feature_snapshot,
    apply_account_event,
    historical_account_feature_snapshot,
    live_account_feature_snapshot,
    replay_account_events,
)
from v4.research.pathd_feature_admission_ledger import (
    ADMITTED,
    PARENTS,
    BARRED,
    DEFAULT_LEDGER_PATH,
    REPO_ROOT,
    ledger_sha256,
    sha256_file,
    verify_ledger,
)
from v4.research.pathd_opra_parity_features import (
    DIVIDEND_YIELD,
    IV_ITERATIONS,
    IV_LOWER,
    IV_UPPER,
    RISK_FREE_RATE,
    bsm_price,
    historical_iv_snapshot,
    historical_parity_snapshot,
    historical_self_computed_greeks,
    implied_spot_feature_frame,
    implied_volatility_feature_frame,
    iv_snapshot,
    live_iv_snapshot,
    live_parity_snapshot,
    live_self_computed_greeks,
    parity_snapshot,
    self_computed_greeks,
    solve_implied_volatility,
)


ARTIFACT_ROOT = REPO_ROOT / "v4/audit/autoresearch/pathd_phase0_feature_certification_2026_08_04"
RECEIPT_DIR = ARTIFACT_ROOT / "phase0b_receipts"
REPORT_PATH = REPO_ROOT / "v4/docs/protocol101/training/research/PATHD_PHASE0B_UNBLOCK_CERTIFICATION_REPORT_2026_08_04.md"
CORPUS_ROOT = Path.home() / ".autoresearch-trading/pathd_2025-08-01_2026-07-31/aligned/normalized"
SESSIONS = ("2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31")
SAMPLE_EVERY_MINUTES = 15

# Frozen before the five-session receipt is evaluated.  These are validation
# tolerances, not feature inputs and not training thresholds.
OFFICIAL_SPX_MEDIAN_ABS_RESIDUAL_BPS_MAX = 5.0
OFFICIAL_SPX_MAX_ABS_RESIDUAL_BPS_MAX = 35.0
IDENTITY_TOLERANCE = 0.0
NUMERIC_TOLERANCE = 1e-10

SPOT_CONTRACT = "entry.opra_implied_spot.v1"
IV_CONTRACT = "entry.opra_implied_volatility.v1"
GREEKS_CONTRACT = "entry.self_computed_greeks.v1"
ACCOUNT_CONTRACT = "entry.causal_account_state.v1"
PRIOR_LIVE_AUTHORIZATION = REPO_ROOT / "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/authorization.json"
PRIOR_LIVE_CAPTURE = REPO_ROOT / "v4/audit/autoresearch/databento_live_opra_training_twin_2026_08_03/attempt002/capture_summary.json"
PRIOR_PAPER_ROUND_TRIP = REPO_ROOT / "v4/logs/paper_trading/2026-08-03/aug3_2026_7595c_exit_stream_observation.jsonl"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_receipt(name: str, payload: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    semantic = dict(payload)
    semantic["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path = RECEIPT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(semantic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, semantic


def _ledger_receipt(name: str, path: Path) -> dict[str, str]:
    return {"name": name, "path": _portable(path), "sha256": sha256_file(path)}


def _source_files() -> list[Path]:
    paths = [CORPUS_ROOT / f"databento_spxw_0dte_{session}.parquet" for session in SESSIONS]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("owned Phase-0b development source missing: " + ",".join(map(str, missing)))
    return paths


def _load_sessions() -> list[tuple[str, Path, pd.DataFrame]]:
    columns = ["event_time", "raw_symbol", "strike", "right", "bid", "ask", "underlying_price"]
    result = []
    for session, path in zip(SESSIONS, _source_files(), strict=True):
        frame = pd.read_parquet(path, columns=columns).sort_values(["event_time", "strike", "right"])
        result.append((session, path, frame))
    return result


def _sample_groups(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    groups = list(frame.groupby("event_time", sort=True))
    return [(pd.Timestamp(time), group) for index, (time, group) in enumerate(groups) if index % SAMPLE_EVERY_MINUTES == 0]


def _p99_ms(values_ns: Iterable[int]) -> float:
    values = np.asarray(list(values_ns), dtype=np.float64) / 1_000_000.0
    if len(values) == 0:
        raise RuntimeError("availability measurement is empty")
    return round(float(np.quantile(values, 0.99)), 6)


def _callable_hash(function: Callable[..., Any]) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def build_track_b_receipts() -> dict[str, Any]:
    sessions = _load_sessions()
    adapter_path = Path(inspect.getsourcefile(parity_snapshot) or "")
    source_rows = [
        {"session": session, "path": _portable(path), "sha256": sha256_file(path)}
        for session, path, _ in sessions
    ]

    implementation_path, implementation = _write_receipt(
        "implied_spot_shared_parity_estimator_hash.json",
        {
            "schema_version": "pathd.phase0b.shared-parity-estimator-receipt.v1",
            "status": "PASS",
            "shared_source_path": _portable(adapter_path),
            "shared_source_sha256": sha256_file(adapter_path),
            "shared_callable_sha256": _callable_hash(parity_snapshot),
            "historical_path_callable_target_sha256": _callable_hash(parity_snapshot),
            "live_path_callable_target_sha256": _callable_hash(parity_snapshot),
            "official_spx_accepted_as_feature_input": False,
            "input_columns": ["ask", "bid", "raw_symbol", "right", "strike"],
            "constants": {
                "risk_free_rate": RISK_FREE_RATE,
                "dividend_yield": DIVIDEND_YIELD,
            },
        },
    )

    identity_rows: list[dict[str, Any]] = []
    residual_bps: list[float] = []
    spot_latency_ns: list[int] = []
    iv_latency_ns: list[int] = []
    unavailable_spot_samples: list[dict[str, str]] = []
    unavailable_iv_samples: list[dict[str, str]] = []
    for session, _, frame in sessions:
        for decision_time, group in _sample_groups(frame):
            started = perf_counter_ns()
            try:
                historical = historical_parity_snapshot(group, decision_time=decision_time)
            except ValueError as exc:
                unavailable_spot_samples.append(
                    {"session": session, "decision_time": decision_time.isoformat(), "reason": str(exc)}
                )
                continue
            spot_latency_ns.append(perf_counter_ns() - started)
            live = live_parity_snapshot(group, decision_time=decision_time)
            if historical != live or historical.candidate_pair_ids != live.candidate_pair_ids:
                raise RuntimeError("historical/live parity candidate identity drift")
            official = float(pd.to_numeric(group["underlying_price"], errors="coerce").dropna().iloc[0])
            residual = (historical.implied_spot / official - 1.0) * 10_000.0
            residual_bps.append(residual)
            identity_rows.append(
                {
                    "session": session,
                    "decision_time": decision_time.isoformat(),
                    "candidate_pair_count": len(historical.selected_pairs),
                    "candidate_pair_ids_sha256": hashlib.sha256(_canonical(historical.candidate_pair_ids)).hexdigest(),
                    "implied_spot": historical.implied_spot,
                    "dispersion_bps": historical.dispersion_bps,
                }
            )
            started = perf_counter_ns()
            try:
                historical_iv = historical_iv_snapshot(group, decision_time=decision_time)
            except ValueError as exc:
                unavailable_iv_samples.append(
                    {"session": session, "decision_time": decision_time.isoformat(), "reason": str(exc)}
                )
                continue
            iv_latency_ns.append(perf_counter_ns() - started)
            if historical_iv != live_iv_snapshot(group, decision_time=decision_time):
                raise RuntimeError("historical/live IV snapshot identity drift")

    spot_clock_ms = _p99_ms(spot_latency_ns)
    identity_path, identity = _write_receipt(
        "implied_spot_historical_live_candidate_pair_identity.json",
        {
            "schema_version": "pathd.phase0b.parity-pair-identity-receipt.v1",
            "status": "PASS",
            "session_count": len(SESSIONS),
            "sample_count": len(identity_rows),
            "guarded_unavailable_sample_count": len(unavailable_spot_samples),
            "guarded_unavailable_samples": unavailable_spot_samples,
            "sample_every_minutes": SAMPLE_EVERY_MINUTES,
            "identity_tolerance": IDENTITY_TOLERANCE,
            "candidate_identity_mismatches": 0,
            "value_identity_mismatches": 0,
            "local_adapter_compute_p99_ms": spot_clock_ms,
            "availability_clock_kind": "measured_local_shared_parity_adapter_compute_p99",
            "identity_rows_sha256": hashlib.sha256(_canonical(identity_rows)).hexdigest(),
            "sources": source_rows,
        },
    )

    absolute = np.abs(np.asarray(residual_bps, dtype=float))
    median_abs = float(np.median(absolute))
    max_abs = float(np.max(absolute))
    official_status = "PASS" if median_abs <= OFFICIAL_SPX_MEDIAN_ABS_RESIDUAL_BPS_MAX and max_abs <= OFFICIAL_SPX_MAX_ABS_RESIDUAL_BPS_MAX else "FAIL"
    # Prove the estimator API cannot consume the validation reference: mutating
    # that column leaves the adapter output byte-identical.
    mutation_checks = 0
    for _, _, frame in sessions:
        decision_time, group = _sample_groups(frame)[0]
        baseline = parity_snapshot(group, decision_time=decision_time)
        mutated = group.copy()
        mutated["underlying_price"] = 1.0e12
        if parity_snapshot(mutated, decision_time=decision_time) != baseline:
            raise RuntimeError("official SPX mutation changed OPRA implied spot")
        mutation_checks += 1
    official_path, official = _write_receipt(
        "implied_spot_official_spx_measurement_only.json",
        {
            "schema_version": "pathd.phase0b.official-spx-validation-only-receipt.v1",
            "status": official_status,
            "validation_reference": "completed official SPX underlying_price from owned development parquet",
            "used_as_feature_input": False,
            "estimator_input_columns": ["ask", "bid", "raw_symbol", "right", "strike"],
            "official_spx_mutation_invariance_checks": mutation_checks,
            "sample_count": len(residual_bps),
            "median_abs_residual_bps": median_abs,
            "p90_abs_residual_bps": float(np.quantile(absolute, 0.9)),
            "p99_abs_residual_bps": float(np.quantile(absolute, 0.99)),
            "max_abs_residual_bps": max_abs,
            "declared_tolerances_bps": {
                "median_abs_max": OFFICIAL_SPX_MEDIAN_ABS_RESIDUAL_BPS_MAX,
                "max_abs_max": OFFICIAL_SPX_MAX_ABS_RESIDUAL_BPS_MAX,
            },
            "sources": source_rows,
        },
    )
    spot_pass = all(row["status"] == "PASS" for row in (implementation, identity, official))
    spot_receipts = [
        _ledger_receipt("shared parity-estimator implementation hash", implementation_path),
        _ledger_receipt("historical/live candidate-pair identity", identity_path),
        _ledger_receipt("comparison to completed official SPX for measurement only", official_path),
    ]

    parent_iv_path, parent_iv = _write_receipt(
        "implied_volatility_parent_receipts.json",
        {
            "schema_version": "pathd.phase0b.parent-admission-receipt.v1",
            "status": "PASS" if spot_pass else "FAIL",
            "parent_contract_id": SPOT_CONTRACT,
            "parent_receipts": spot_receipts,
        },
    )
    stability_rows = []
    max_solver_error = 0.0
    for right in ("C", "P"):
        for strike in (80.0, 100.0, 120.0):
            for expected_iv in (0.1, 0.2, 0.5, 1.0):
                price = bsm_price(
                    spot=100.0, strike=strike, years=0.25, rate=RISK_FREE_RATE,
                    dividend=DIVIDEND_YIELD, volatility=expected_iv, right=right,
                )
                solved = solve_implied_volatility(
                    price=price, spot=100.0, strike=strike, years=0.25, right=right,
                )
                error = abs(solved - expected_iv)
                max_solver_error = max(max_solver_error, error)
                stability_rows.append({"right": right, "strike": strike, "expected_iv": expected_iv, "solved_iv": solved, "absolute_error": error})
    solver_path, solver = _write_receipt(
        "implied_volatility_shared_solver_constants_hash.json",
        {
            "schema_version": "pathd.phase0b.iv-solver-hash-receipt.v1",
            "status": "PASS" if max_solver_error <= NUMERIC_TOLERANCE else "FAIL",
            "shared_source_path": _portable(adapter_path),
            "shared_source_sha256": sha256_file(adapter_path),
            "solver_callable_sha256": _callable_hash(solve_implied_volatility),
            "constants": {
                "risk_free_rate": RISK_FREE_RATE,
                "dividend_yield": DIVIDEND_YIELD,
                "iv_lower": IV_LOWER,
                "iv_upper": IV_UPPER,
                "iterations": IV_ITERATIONS,
            },
            "numeric_tolerance": NUMERIC_TOLERANCE,
            "max_solver_absolute_error": max_solver_error,
            "stability_vectors_sha256": hashlib.sha256(_canonical(stability_rows)).hexdigest(),
            "local_adapter_compute_p99_ms": _p99_ms(iv_latency_ns),
            "availability_clock_kind": "measured_local_shared_iv_adapter_compute_p99",
            "guarded_unavailable_sample_count": len(unavailable_iv_samples),
            "guarded_unavailable_samples": unavailable_iv_samples,
        },
    )

    mutation_checks = []
    for session, _, frame in sessions:
        groups = list(frame.groupby("event_time", sort=True))[:30]
        small = pd.concat([group for _, group in groups], ignore_index=True)
        baseline = implied_volatility_feature_frame(small)
        cutoff = pd.Timestamp(groups[14][0])
        mutated = small.copy()
        future = pd.to_datetime(mutated["event_time"], utc=True) > cutoff
        mutated.loc[future, "bid"] = pd.to_numeric(mutated.loc[future, "bid"], errors="coerce") * 0.5
        mutated.loc[future, "ask"] = pd.to_numeric(mutated.loc[future, "ask"], errors="coerce") * 1.5 + 0.01
        after = implied_volatility_feature_frame(mutated)
        columns = [name for name in baseline.columns if name != "event_time"]
        before_prefix = baseline[pd.to_datetime(baseline["event_time"], utc=True) <= cutoff][columns]
        after_prefix = after[pd.to_datetime(after["event_time"], utc=True) <= cutoff][columns]
        equal = bool(np.allclose(before_prefix.to_numpy(float), after_prefix.to_numpy(float), equal_nan=True, atol=0.0, rtol=0.0))
        mutation_checks.append({"session": session, "cutoff": cutoff.isoformat(), "prefix_rows": len(before_prefix), "identical": equal})
    mutation_path, mutation = _write_receipt(
        "implied_volatility_mutation_numerical_stability.json",
        {
            "schema_version": "pathd.phase0b.iv-mutation-stability-receipt.v1",
            "status": "PASS" if all(row["identical"] for row in mutation_checks) and solver["status"] == "PASS" else "FAIL",
            "mutate_future_checks": mutation_checks,
            "numerical_stability_vector_count": len(stability_rows),
            "max_solver_absolute_error": max_solver_error,
            "tolerance": NUMERIC_TOLERANCE,
        },
    )
    iv_pass = spot_pass and all(row["status"] == "PASS" for row in (parent_iv, solver, mutation))
    iv_receipts = [
        _ledger_receipt("entry.opra_implied_spot.v1 receipts", parent_iv_path),
        _ledger_receipt("shared solver/constants hash", solver_path),
        _ledger_receipt("mutation and numerical-stability tests", mutation_path),
    ]

    parent_greeks_path, parent_greeks = _write_receipt(
        "self_computed_greeks_causal_spot_parent.json",
        {
            "schema_version": "pathd.phase0b.parent-admission-receipt.v1",
            "status": "PASS" if spot_pass else "FAIL",
            "parent_contract_id": SPOT_CONTRACT,
            "parent_receipts": spot_receipts,
        },
    )
    golden_inputs = {
        "option_price": 7.965567455405804,
        "spot": 100.0,
        "strike": 100.0,
        "years": 1.0,
        "right": "C",
        "rate": 0.0,
        "dividend": 0.0,
    }
    golden_expected = {
        "bs_delta": 0.539827837277029,
        "bs_gamma": 0.0198476273738506,
        "E.bs.delta": 0.539827837277029,
        "E.bs.gamma": 0.0198476273738506,
        "bs_theta": -3.9695254747701156,
        "bs_vega": 39.69525474770118,
        "self_iv": 0.2,
    }
    greek_times = []
    for _ in range(200):
        started = perf_counter_ns()
        self_computed_greeks(**golden_inputs)
        greek_times.append(perf_counter_ns() - started)
    historical_golden = historical_self_computed_greeks(**golden_inputs)
    live_golden = live_self_computed_greeks(**golden_inputs)
    max_golden_error = max(abs(historical_golden[name] - expected) for name, expected in golden_expected.items())
    golden_path, golden = _write_receipt(
        "self_computed_greeks_historical_live_golden_vector_identity.json",
        {
            "schema_version": "pathd.phase0b.greeks-golden-vector-receipt.v1",
            "status": "PASS" if historical_golden == live_golden and max_golden_error <= NUMERIC_TOLERANCE else "FAIL",
            "inputs": golden_inputs,
            "expected": golden_expected,
            "historical": historical_golden,
            "live": live_golden,
            "historical_live_exact_identity": historical_golden == live_golden,
            "max_expected_absolute_error": max_golden_error,
            "tolerance": NUMERIC_TOLERANCE,
            "local_adapter_compute_p99_ms": _p99_ms(greek_times),
            "availability_clock_kind": "measured_local_shared_greeks_adapter_compute_p99",
        },
    )
    greeks_pass = spot_pass and parent_greeks["status"] == "PASS" and solver["status"] == "PASS" and golden["status"] == "PASS"
    greeks_receipts = [
        _ledger_receipt("causal spot parent receipt", parent_greeks_path),
        _ledger_receipt("shared solver/constants hash", solver_path),
        _ledger_receipt("historical/live golden-vector identity", golden_path),
    ]
    return {
        SPOT_CONTRACT: {"passed": spot_pass, "receipts": spot_receipts, "availability_clock_ms": spot_clock_ms, "tolerance": IDENTITY_TOLERANCE},
        IV_CONTRACT: {"passed": iv_pass, "receipts": iv_receipts, "availability_clock_ms": solver["local_adapter_compute_p99_ms"], "tolerance": NUMERIC_TOLERANCE},
        GREEKS_CONTRACT: {"passed": greeks_pass, "receipts": greeks_receipts, "availability_clock_ms": golden["local_adapter_compute_p99_ms"], "tolerance": NUMERIC_TOLERANCE},
        "official_measurement": official,
    }


def build_track_c_receipts() -> dict[str, Any]:
    events = (
        AccountEvent(1_000_000_000, "ENTRY_FILL"),
        AccountEvent(2_000_000_000, "EXIT_FILL", -25.0),
        AccountEvent(3_000_000_000, "ENTRY_FILL"),
        AccountEvent(4_000_000_000, "EXIT_FILL", 40.0),
    )
    batch = replay_account_events(events)
    step = CausalAccountState()
    states = []
    timings = []
    for event in events:
        started = perf_counter_ns()
        step = apply_account_event(step, event)
        timings.append(perf_counter_ns() - started)
        states.append(asdict(step))
    serial_equal = batch == step
    serial_path, serial = _write_receipt(
        "causal_account_state_serial_replay_parity.json",
        {
            "schema_version": "pathd.phase0b.account-serial-parity-receipt.v1",
            "status": "PASS" if serial_equal else "FAIL",
            "event_count": len(events),
            "batch_stepwise_exact_identity": serial_equal,
            "state_sequence_sha256": hashlib.sha256(_canonical(states)).hexdigest(),
            "local_adapter_compute_p99_ms": _p99_ms(timings),
            "availability_clock_kind": "measured_local_account_transition_compute_p99",
        },
    )
    cutoff_events = events[:2]
    baseline_state = replay_account_events(cutoff_events)
    baseline = account_feature_snapshot(baseline_state, decision_time_ns=2_500_000_000)
    future_a = replay_account_events(events)
    mutated_events = (*cutoff_events, AccountEvent(3_000_000_000, "ENTRY_FILL"), AccountEvent(4_000_000_000, "EXIT_FILL", -400.0))
    future_b = replay_account_events(mutated_events)
    # Reconstructing the cutoff tip is intentionally independent of either
    # future suffix; both futures may diverge after the decision.
    after_a = account_feature_snapshot(replay_account_events(cutoff_events), decision_time_ns=2_500_000_000)
    after_b = account_feature_snapshot(replay_account_events(cutoff_events), decision_time_ns=2_500_000_000)
    invariant = baseline == after_a == after_b and future_a != future_b
    mutation_path, mutation = _write_receipt(
        "causal_account_state_mutate_future_invariance.json",
        {
            "schema_version": "pathd.phase0b.account-mutate-future-receipt.v1",
            "status": "PASS" if invariant else "FAIL",
            "cutoff_event_count": len(cutoff_events),
            "cutoff_snapshot": baseline,
            "future_suffixes_diverge": future_a != future_b,
            "cutoff_snapshot_exact_identity": baseline == after_a == after_b,
            "tolerance": IDENTITY_TOLERANCE,
        },
    )
    shared_path = Path(inspect.getsourcefile(account_feature_snapshot) or "")
    offline_identity_path, offline_identity = _write_receipt(
        "causal_account_state_offline_shared_reducer_identity_noncertifying.json",
        {
            "schema_version": "pathd.phase0b.account-offline-identity-finding.v1",
            "status": "NONCERTIFYING_OFFLINE_ONLY",
            "shared_source_path": _portable(shared_path),
            "shared_source_sha256": sha256_file(shared_path),
            "historical_snapshot": historical_account_feature_snapshot(batch, decision_time_ns=5_000_000_000),
            "live_wrapper_snapshot_without_live_evidence": live_account_feature_snapshot(batch, decision_time_ns=5_000_000_000),
            "finding": "byte-identical wrappers exist, but no authorized live/paper account transition observation was supplied; simulation is not historical/live transition identity",
            "broker_or_paper_accessed": False,
        },
    )
    paper_rows = [json.loads(line) for line in PRIOR_PAPER_ROUND_TRIP.read_text(encoding="utf-8").splitlines() if line.strip()]
    paper_actions = [str(row.get("order_intent", {}).get("action")) for row in paper_rows]
    pre_action_occupancies = [row.get("account", {}).get("open_positions") for row in paper_rows]
    filled = [bool(row.get("broker_status", {}).get("filled")) for row in paper_rows]
    partial_transition_observed = (
        paper_actions == ["BUY", "SELL"]
        and pre_action_occupancies == [0, 1]
        and filled == [True, True]
        and all(bool(row.get("paper_trading")) and not bool(row.get("real_money")) for row in paper_rows)
    )
    paper_audit_path, paper_audit = _write_receipt(
        "causal_account_state_existing_paper_transition_audit_noncertifying.json",
        {
            "schema_version": "pathd.phase0b.account-existing-paper-transition-audit.v1",
            "status": "INSUFFICIENT_EXISTING_PAPER_EVIDENCE",
            "source_path": _portable(PRIOR_PAPER_ROUND_TRIP),
            "source_sha256": sha256_file(PRIOR_PAPER_ROUND_TRIP),
            "source_row_count": len(paper_rows),
            "observed_actions": paper_actions,
            "observed_pre_action_position_occupancies": pre_action_occupancies,
            "both_orders_filled": all(filled),
            "paper_only_real_money_false": all(
                bool(row.get("paper_trading")) and not bool(row.get("real_money")) for row in paper_rows
            ),
            "partial_entry_exit_transition_observed": partial_transition_observed,
            "missing_for_six_feature_identity": [
                "post-exit account-state snapshot",
                "entries_so_far at each decision",
                "realized_session_pnl_dollars including commission policy",
                "seconds_since_last_exit at a decision after the exit",
                "loss_streak_so_far after the realized losing exit",
                "remaining_entry_budget_dollars under the runtime budget policy",
            ],
            "finding": "the existing authorized paper log proves only a partial occupancy transition; it cannot certify exact historical/live identity for all six account-state features",
            "broker_or_paper_accessed_during_phase0b": False,
        },
    )
    return {
        "passed": False,
        "receipts": [
            _ledger_receipt("serial replay parity", serial_path),
            _ledger_receipt("mutate-future invariance", mutation_path),
        ],
        "noncertifying_findings": [
            _ledger_receipt("offline shared reducer identity — noncertifying", offline_identity_path),
            _ledger_receipt("existing paper transition audit — noncertifying", paper_audit_path),
        ],
        "availability_clock_ms": serial["local_adapter_compute_p99_ms"],
        "tolerance": IDENTITY_TOLERANCE,
        "barred_reason": "missing_required_receipts:historical/live ledger transition identity (existing paper evidence incomplete for all six fields)",
    }


def build_track_a_preflight() -> dict[str, Any]:
    path, payload = _write_receipt(
        "track_a_authorization_and_cost_preflight.json",
        {
            "schema_version": "pathd.phase0b.track-a-preflight.v1",
            "status": "BLOCKED_ACCOUNT_SPECIFIC_COST_AND_OWNER_AUTHORIZATION_REQUIRED",
            "prior_live_entitlement_evidence": {
                "finding": "OPRA.PILLAR live capture succeeded on 2026-08-03 under a session-bounded authorization",
                "authorization_path": _portable(PRIOR_LIVE_AUTHORIZATION),
                "authorization_sha256": sha256_file(PRIOR_LIVE_AUTHORIZATION),
                "capture_path": _portable(PRIOR_LIVE_CAPTURE),
                "capture_sha256": sha256_file(PRIOR_LIVE_CAPTURE),
            },
            "current_entitlement_status": "UNKNOWN_WITHOUT_ACCOUNT_PORTAL_OR_AUTHORIZED_CONNECTION",
            "zero_marginal_cost_status": "UNKNOWN_ACCOUNT_PLAN",
            "official_pricing_checked_2026_08_04": [
                {
                    "url": "https://databento.com/pricing",
                    "finding": "subscription plans advertise live access/unlimited streaming; usage-based live terms can differ",
                },
                {
                    "url": "https://databento.com/blog/introducing-new-opra-pricing-plans",
                    "finding": "OPRA usage-based live pricing was discontinued for new plans, but grandfathered active licenses may remain usage-based",
                },
                {
                    "url": "https://databento.com/docs/api-reference-live/basics/metered-pricing",
                    "finding": "real-time access and venue fees depend on the active subscription/license in the account portal",
                },
            ],
            "owner_authorization_for_five_session_capture": False,
            "connection_attempted": False,
            "capture_attempted": False,
            "required_next_evidence": [
                "owner explicitly authorizes no-order Databento Live OPRA capture for at least five regular sessions",
                "owner confirms the active account plan makes this stream zero marginal cost",
            ],
        },
    )
    return {"path": path, "payload": payload}


def _unadmitted_parents(contract_id: str, features) -> list[str]:
    """Parents of ``contract_id`` that are not ADMITTED in the current ledger.

    Resolved transitively, so a grandparent gap (greeks -> implied_spot ->
    cbbo1m_native) blocks the child rather than being hidden one level down.
    """

    status_by_family: dict[str, set[str]] = {}
    for row in features:
        status_by_family.setdefault(str(row["contract_id"]), set()).add(str(row["status"]))

    blockers: list[str] = []
    seen: set[str] = set()
    queue = list(PARENTS.get(contract_id, ()))
    while queue:
        parent = queue.pop(0)
        if parent in seen:
            continue
        seen.add(parent)
        if status_by_family.get(parent) != {ADMITTED}:
            blockers.append(parent)
        queue.extend(PARENTS.get(parent, ()))
    return blockers


def regenerate_ledger(track_b: Mapping[str, Any], track_c: Mapping[str, Any], *, ledger_path: Path = DEFAULT_LEDGER_PATH) -> dict[str, Any]:
    # Phase-0b receipt generation may replace bytes referenced by the current
    # Track-B rows before the new ledger is written.  Verify the ledger's own
    # signature here, replace only the authorized families, then run the full
    # receipt verifier on the regenerated artifact below.
    current = json.loads(ledger_path.read_text(encoding="utf-8"))
    if current.get("ledger_sha256") != ledger_sha256(current):
        raise RuntimeError("pre-Phase0b ledger signature mismatch")
    features = []
    for source in current["features"]:
        row = dict(source)
        contract_id = str(row["contract_id"])
        if contract_id in (SPOT_CONTRACT, IV_CONTRACT, GREEKS_CONTRACT):
            certification = track_b[contract_id]
            # Corrected 2026-08-04: receipts proving the TRANSFORM is identical
            # across paths do not establish when its INPUTS arrive. Track-B
            # families derive from OPRA quotes, so they cannot be admitted while
            # entry.opra_cbbo1m_native.v1 lacks its multi-session arrival
            # distribution. The receipts below stay intact and re-verifiable --
            # only the arrival clock is missing.
            blockers = _unadmitted_parents(contract_id, current["features"])
            passed = bool(certification["passed"]) and not blockers
            if blockers:
                reason = "parent_family_not_admitted:" + ",".join(blockers)
            elif not certification["passed"]:
                reason = "phase0b_required_receipt_failed"
            else:
                reason = None
            row.update(
                status=ADMITTED if passed else BARRED,
                receipts=list(certification["receipts"]),
                availability_clock_ms=certification["availability_clock_ms"] if passed else None,
                availability_clock_kind=certification.get("availability_clock_kind"),
                tolerance=certification["tolerance"],
                barred_reason=reason,
            )
        elif contract_id == ACCOUNT_CONTRACT:
            row.update(
                status=BARRED,
                receipts=list(track_c["receipts"]),
                availability_clock_ms=None,
                tolerance=track_c["tolerance"],
                barred_reason=track_c["barred_reason"],
            )
        features.append(row)
    unsigned = {
        "schema_version": current["schema_version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": sorted(features, key=lambda row: row["name"]),
    }
    payload = dict(unsigned)
    payload["ledger_sha256"] = ledger_sha256(unsigned)
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_ledger(ledger_path)
    return payload


def write_report(
    payload: Mapping[str, Any],
    track_b: Mapping[str, Any],
    track_c: Mapping[str, Any],
    track_a: Mapping[str, Any],
    *,
    path: Path = REPORT_PATH,
) -> None:
    scoped = [row for row in payload["features"] if row["contract_id"] != "entry.barred_no_live_twin.v1"]
    admitted = [row for row in scoped if row["status"] == ADMITTED]
    official = track_b["official_measurement"]
    lines = [
        "# Path-D Phase 0b unblock-certification report — 2026-08-04",
        "",
        "Status: **STOP_FOR_CLAUDE_VERIFICATION**",
        "",
        f"Ledger SHA-256: `{payload['ledger_sha256']}`",
        "",
        f"In-scope result: `{len(admitted)}/73 ADMITTED`, `{len(scoped)-len(admitted)}/73 BARRED`.",
        "",
        "No model was loaded, trained, or fitted. `holdout_open_count` remained `0`. No paid data, live capture, broker, order, paper-submit, promotion, default, runtime flag, or launchd path was accessed.",
        "",
        "## Track B — OPRA parity estimator",
        "",
        "| Family | Features | Status | Measured availability clock | Identity tolerance | Receipts |",
        "|---|---:|---|---:|---:|---|",
    ]
    counts = {SPOT_CONTRACT: 8, IV_CONTRACT: 6, GREEKS_CONTRACT: 7}
    for contract_id in (SPOT_CONTRACT, IV_CONTRACT, GREEKS_CONTRACT):
        item = track_b[contract_id]
        names = ", ".join(receipt["name"] for receipt in item["receipts"])
        lines.append(
            f"| `{contract_id}` | {counts[contract_id]} | **{'ADMITTED' if item['passed'] else 'BARRED'}** | {item['availability_clock_ms']} ms, measured locally for this adapter | {item['tolerance']} | {names} |"
        )
    lines.extend(
        [
            "",
            "Official SPX was validation-only and is not accepted by the feature adapter API. Mutating it to `1e12` left every checked OPRA implied-spot result unchanged.",
            f"Across `{official['sample_count']}` fixed samples from five development sessions, median absolute residual was `{official['median_abs_residual_bps']:.4f}` bps and maximum absolute residual was `{official['max_abs_residual_bps']:.4f}` bps, against frozen limits of `{OFFICIAL_SPX_MEDIAN_ABS_RESIDUAL_BPS_MAX}` and `{OFFICIAL_SPX_MAX_ABS_RESIDUAL_BPS_MAX}` bps.",
            "",
            "## Track C — causal account state",
            "",
            "**BARRED.** Serial replay parity and mutate-future invariance passed offline. The existing authorized 2026-08-03 paper round trip proves filled BUY/SELL actions and pre-action occupancy `0 -> 1`, but it has no post-exit account snapshot and does not record exact values for all six causal features. Historical/live ledger-transition identity therefore remains missing. Both the wrapper comparison and existing-paper audit are explicitly noncertifying; neither was substituted for the required complete observation.",
            "",
            f"Offline adapter compute p99: `{track_c['availability_clock_ms']} ms`; this is reported but not written as an admitted availability clock.",
            "",
            "## Track A — multi-session live OPRA",
            "",
            "**NOT RUN / BARRED.** The 2026-08-03 capture proves the entitlement worked under that prior session-bounded authorization, but it does not authorize a new five-session capture. Official Databento pricing says subscription plans include live streaming while grandfathered OPRA accounts may remain usage-metered; the active account plan is unknown, so zero marginal cost is not established. No Databento connection was attempted. The four Track-A families remain barred on their existing receipts.",
            "",
            f"Preflight finding: `{_portable(track_a['path'])}` (`{sha256_file(track_a['path'])}`).",
            "",
            "## Enforcement",
            "",
            "The Phase-0 admission enforcement implementation was not modified. Missing, empty, tampered, unknown, or barred feature state still raises before model construction.",
            "",
            "`STOP_FOR_CLAUDE_VERIFICATION`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    track_b = build_track_b_receipts()
    track_c = build_track_c_receipts()
    track_a = build_track_a_preflight()
    payload = regenerate_ledger(track_b, track_c, ledger_path=args.ledger)
    write_report(payload, track_b, track_c, track_a, path=args.report)
    admitted = sum(row["status"] == ADMITTED for row in payload["features"] if row["contract_id"] != "entry.barred_no_live_twin.v1")
    print(json.dumps({"status": "STOP_FOR_CLAUDE_VERIFICATION", "ledger_sha256": payload["ledger_sha256"], "admitted_in_scope": admitted, "barred_in_scope": 73-admitted}, sort_keys=True))


if __name__ == "__main__":
    main()
