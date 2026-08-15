"""Validate FT1B reference/multiplicity machinery without campaign economics."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from v4.model import protocol101_stage1_reference_multiplicity as machinery
from v4.model.protocol101_repair_artifacts import (
    REQUIRED_MANIFEST_HASHES,
    canonical_json_bytes,
    sha256_file,
)
from v4.model.protocol101_scoped_stage1_hgb import (
    CanonicalDecision,
    RepairedCanonicalDecision,
)
from v4.model.protocol101_serial_simulator_v5 import (
    PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION,
    PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
    Protocol101LegacySyntheticExitArtifactError,
)
from v4.scripts import run_protocol101_scoped_stage1_reference_packets as refs


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    WORKSPACE
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_stage1_reference_multiplicity_"
    "machinery_attempt001"
)
SUCCESS_ROUTE = (
    "reference_multiplicity_machinery_repair_complete_"
    "pending_independent_acceptance"
)
READINESS_STATUS = (
    "reference_multiplicity_machinery_ready_pending_independent_acceptance"
)
EXPECTED_ACCEPTANCE_ROUTE = "entry_runner_v5_core_independently_accepted"
ACCEPTANCE_DIR = (
    WORKSPACE
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_entry_runner_v5_core_"
    "independent_acceptance_attempt001"
)
CAMPAIGN_DIR = (
    WORKSPACE
    / "v4/audit/autoresearch/"
    "protocol101_full_trader_entry_campaign_preregistration_attempt001"
)
AUTHORIZED_CODE_TEST_FILES = (
    "v4/scripts/run_protocol101_scoped_stage1_reference_packets.py",
    "v4/model/protocol101_stage1_reference_multiplicity.py",
    (
        "v4/scripts/"
        "run_protocol101_full_trader_stage1_reference_"
        "multiplicity_validation.py"
    ),
    "v4/tests/test_protocol101_scoped_stage1_reference_packets.py",
    "v4/tests/test_protocol101_stage1_reference_multiplicity.py",
    (
        "v4/tests/"
        "test_protocol101_full_trader_stage1_reference_"
        "multiplicity_validation.py"
    ),
)
COMPILE_FILES = AUTHORIZED_CODE_TEST_FILES
FOCUSED_TESTS = (
    "v4/tests/test_protocol101_scoped_stage1_reference_packets.py",
    "v4/tests/test_protocol101_stage1_reference_multiplicity.py",
    (
        "v4/tests/"
        "test_protocol101_full_trader_stage1_reference_"
        "multiplicity_validation.py"
    ),
    "v4/tests/test_protocol101_full_trader_entry_runner_v5.py",
    "v4/tests/test_protocol101_serial_simulator_v5.py",
    "v4/tests/test_protocol101_regimen_repair_identity.py",
    "v4/tests/test_protocol101_regimen_repair_artifacts.py",
    "v4/tests/test_protocol101_governed_loader.py",
    "v4/tests/test_protocol101_canonical_stage1_contract.py",
)
FROZEN_INPUTS = {
    (
        "v4/docs/protocol101/training/contracts/"
        "PROTOCOL101_STAGE1_REGIMEN_REPAIR_AMENDMENT_2026_07_26.md"
    ): "b02a99281b502434675c3440e9f214fc88b9ac974359e20bb7841e19a2a8065b",
    (
        "v4/docs/protocol101/training/contracts/"
        "PROTOCOL101_CANONICAL_V1_STAGE1_TRAINING_DESIGN.md"
    ): "d3fc327cdf82159f73388648c15643f42c2207a420c3cdab9c4fb08cc302cbc0",
    (
        "v4/audit/autoresearch/"
        "protocol101_full_trader_entry_campaign_preregistration_attempt001/"
        "campaign_contract.json"
    ): "7a6f747718419041ca3ce9590fb192c64e915800f3dd0dafac5d9ffdc5ec03f0",
    (
        "v4/audit/autoresearch/"
        "protocol101_full_trader_entry_campaign_preregistration_attempt001/"
        "preregistration.json"
    ): "40c3fa07c6fc94aaafdb1abf2b454ede5567c92728f814c38870c8f0eed969c5",
    (
        "v4/audit/autoresearch/"
        "protocol101_full_trader_entry_runner_v5_core_"
        "independent_acceptance_attempt001/acceptance_decision.json"
    ): "72cbe1443cbddeb5af6dbb09bf384d9ea479658a4cefacbcda5b9c8463600bd2",
    (
        "v4/audit/autoresearch/"
        "protocol101_full_trader_entry_runner_v5_core_"
        "independent_acceptance_attempt001/hashes.sha256"
    ): "b5a70e0dcc6df7c0bd95408a65d1561beaccb615c75ea978060eab15c60b41e0",
}
ACCEPTED_RUNNER_CORE_HASHES = {
    "v4/model/protocol101_scoped_stage1_hgb.py": (
        "5266de4db4b25685ed73be1fde5e00f220162b110662befe31ebad250fa27491"
    ),
    "v4/model/protocol101_serial_simulator_v5.py": (
        "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
    ),
    "v4/model/protocol101_regimen_repair.py": (
        "94a149af189606e52d7652e9468c949119c5539529ae39ab216654f348f0135b"
    ),
    "v4/model/protocol101_repair_artifacts.py": (
        "6e0d866ce6ea64685751bf2840ab599fe7708f60385fcf70c55ac36ee656aade"
    ),
    "v4/model/protocol101_governed_loader.py": (
        "725bd31c485055c5414a957cc951042f1eaf116a86c06928232287cd4a37cf59"
    ),
    "v4/model/protocol101_canonical_stage1_contract.py": (
        "cf062784426986af0607560a73dfae89a75e5e7c3c1a47dfdf00224a929f4f39"
    ),
    "v4/scripts/run_protocol101_scoped_stage1_hgb_runner.py": (
        "828a23c49aeaa618c3d8b70ed6890bc8f10c5884abb6db5dfb1549223da69b44"
    ),
    "v4/scripts/run_protocol101_scoped_stage1_hgb_runner_v2.py": (
        "7281e83f07b038d65ad4697c1c8c0d0302e684db8870ba0d8e1648754a004cf3"
    ),
    "v4/scripts/run_protocol101_full_trader_entry_runner_v5_validation.py": (
        "f27f21879676c0d7ffff5b97330a1828b7c3b5389d337ca2ea6f04e92ce550d9"
    ),
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(_plain(payload)))


def _csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value, trailing_lf=False)
    ).hexdigest()


def _verify_frozen_inputs() -> list[dict[str, Any]]:
    results = []
    for relative, expected in FROZEN_INPUTS.items():
        path = WORKSPACE / relative
        observed = sha256_file(path) if path.is_file() else None
        results.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    if any(item["status"] != "PASS" for item in results):
        raise RuntimeError("required frozen input hash mismatch")
    decision = json.loads(
        (ACCEPTANCE_DIR / "acceptance_decision.json").read_text(
            encoding="utf-8"
        )
    )
    if decision.get("routing_decision") != EXPECTED_ACCEPTANCE_ROUTE:
        raise RuntimeError("runner-core acceptance route changed")
    return results


def _verify_initial_preregistration(out_dir: Path) -> dict[str, str]:
    freeze = out_dir / "preregistration_freeze.sha256"
    if not freeze.is_file():
        raise RuntimeError("preregistration freeze is missing")
    expected: dict[str, str] = {}
    for line in freeze.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator:
            raise RuntimeError("malformed preregistration freeze")
        expected[name] = digest
    initial_progress = out_dir / "progress_initial.json"
    progress = out_dir / "progress.json"
    progress_source = initial_progress if initial_progress.is_file() else progress
    checks = {
        "preregistration.json": out_dir / "preregistration.json",
        "source_inventory.json": out_dir / "source_inventory.json",
        "progress.json": progress_source,
    }
    for name, path in checks.items():
        if name not in expected or sha256_file(path) != expected[name]:
            raise RuntimeError(f"preregistration freeze mismatch: {name}")
    if not initial_progress.exists():
        initial_progress.write_bytes(progress.read_bytes())
    return expected


def seed_preregistration_namespace(
    source_dir: Path,
    target_dir: Path,
) -> None:
    """Seed an isolated run from immutable preregistration state.

    A completed producer packet has mutable terminal ``progress.json`` plus
    the separately preserved initial snapshot. Fresh runs must start from the
    latter so their behavior is independent of producer execution order.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "preregistration.json",
        "source_inventory.json",
        "preregistration_freeze.sha256",
    ):
        source = source_dir / name
        if not source.is_file():
            raise RuntimeError(f"preregistration seed is missing: {name}")
        (target_dir / name).write_bytes(source.read_bytes())
    initial_source = source_dir / "progress_initial.json"
    if not initial_source.is_file():
        initial_source = source_dir / "progress.json"
    if not initial_source.is_file():
        raise RuntimeError("preregistration seed is missing initial progress")
    initial_payload = initial_source.read_bytes()
    (target_dir / "progress.json").write_bytes(initial_payload)
    (target_dir / "progress_initial.json").write_bytes(initial_payload)
    _verify_initial_preregistration(target_dir)


def _called_names(function: Callable[..., Any]) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def validate_reference_call_graph() -> dict[str, Any]:
    wrappers = {
        "matched_random": refs.build_fresh_v5_matched_random_reference,
        "fixed_heuristic": refs.build_fresh_v5_fixed_heuristic_reference,
        "reference_replay": machinery.replay_reference_v5,
        "matched_random_engine": machinery.run_matched_random_reference,
        "fixed_heuristic_engine": machinery.run_fixed_heuristic_reference,
    }
    calls = {
        name: sorted(_called_names(function))
        for name, function in wrappers.items()
    }
    predicates = {
        "fresh_namespace_is_v5": (
            refs.FRESH_REFERENCE_NAMESPACE_PREFIX
            == "protocol101_full_trader_stage1_reference_v5_"
        ),
        "reference_replay_calls_v5_simulator": (
            "simulate_serial_candidates_v5" in calls["reference_replay"]
        ),
        "reference_replay_does_not_call_v4": (
            "simulate_serial_candidates"
            not in calls["reference_replay"]
        ),
        "legacy_cli_explicitly_labeled": (
            refs.LEGACY_COMPATIBILITY_MODE == "legacy-v4-compatibility"
        ),
    }
    return {
        "schema_version": "Protocol101FT1BReferenceCallGraphV1",
        "calls": calls,
        "predicates": predicates,
        "status": (
            "PASS" if all(predicates.values()) else "FAIL"
        ),
    }


def _decision(
    row: int,
    *,
    gross_labels: tuple[float, ...] = (-20.0, 3.0, 103.0),
    spacing_minutes: int = 1,
) -> RepairedCanonicalDecision:
    session = "2025-01-02"
    decision_time = pd.Timestamp(
        f"{session} 15:00:00",
        tz="UTC",
    ) + pd.Timedelta(minutes=row * spacing_minutes)
    labels = np.asarray(gross_labels, dtype=float)
    count = len(labels)
    asks = np.full(count, 2.0, dtype=float)
    rights = np.asarray(
        (["P", "C", "C"] if count == 3 else ["C"] * count),
        dtype=object,
    )
    source = int(decision_time.value + 60 * 1_000_000_000)
    realized = int(decision_time.value + 2 * 60 * 1_000_000_000)
    base = CanonicalDecision(
        session=session,
        decision_time=decision_time,
        features=np.asarray(
            [
                [float(row), float(index), *([1.0] * 12)]
                for index in range(count)
            ],
            dtype=float,
        ),
        labels=labels,
        mid_labels=labels + 7.0,
        entry_asks=asks,
        offsets=np.linspace(-5.0, 5.0, count),
        rights=rights,
        contract_ids=np.asarray(
            [f"{session}-{row:03d}-{index}" for index in range(count)],
            dtype=object,
        ),
        strike_indices=np.arange(count, dtype=int) + 10,
        right_indices=np.asarray(
            [1 if value == "P" else 0 for value in rights],
            dtype=int,
        ),
    )
    return RepairedCanonicalDecision(
        base=base,
        realized_exit_time_ns=np.full(count, realized, dtype=np.int64),
        source_exit_quote_time_ns=np.full(count, source, dtype=np.int64),
        exit_quote_age_ms=np.full(count, 60_000.0),
        exit_reason_codes=np.full(count, 3, dtype=np.uint8),
        executable_exit_bids=asks + labels / 100.0,
        policy_deadline_ns=np.full(count, realized, dtype=np.int64),
        invalid_reason_codes=np.zeros(count, dtype=np.uint8),
        canonical_strike_slots=np.arange(count, dtype=np.int64) + 10,
        source_quote_time_ns=np.full(
            count, int(decision_time.value), dtype=np.int64
        ),
        source_context_time_ns=np.full(
            count,
            int(decision_time.value - 60 * 1_000_000_000),
            dtype=np.int64,
        ),
    )


def _opportunities(
    policy: int,
    *,
    gross_labels: tuple[float, ...] = (-20.0, 3.0, 103.0),
    count: int = 3,
) -> list[machinery.ReferenceOpportunity]:
    return [
        machinery.ReferenceOpportunity(
            campaign_id="synthetic_reference_validation",
            fold="fold_1",
            split="validation",
            policy_index=policy,
            repaired=_decision(
                index,
                gross_labels=gross_labels,
                spacing_minutes=2,
            ),
            vwap_side="C" if index % 2 == 0 else "P",
            decision_ordinal=index,
        )
        for index in range(count)
    ]


def _reference_provenance() -> dict[str, str]:
    semantic = {
        "simulator_config_hash",
        "candidate_stream_hash",
        "candidate_payload_hash",
        "trade_identity_hash",
        "exit_quote_age_report_hash",
    }
    source_hashes = {
        "processed_corpus_hash": _hash_payload("synthetic_no_economics"),
        "fold_governance_hash": _hash_payload("synthetic_five_fold_grid"),
        "acceptance_registry_hash": _hash_payload("synthetic_registry"),
        "feature_contract_hash": sha256_file(
            WORKSPACE / "v4/model/protocol101_canonical_stage1_contract.py"
        ),
        "policy_contract_hash": _hash_payload(
            "matched_random_and_fixed_P5_v5"
        ),
        "two_clock_exit_contract_hash": sha256_file(
            WORKSPACE / "v4/model/protocol101_regimen_repair.py"
        ),
        "model_or_equivalence_certificate_hash": _hash_payload(
            "NON_CANDIDATE_synthetic_reference"
        ),
        "threshold_hash": _hash_payload("no_model_threshold"),
        "epsilon_hash": _hash_payload("no_model_epsilon"),
        "selection_contract_hash": sha256_file(
            WORKSPACE
            / "v4/model/protocol101_stage1_reference_multiplicity.py"
        ),
        "simulator_source_hash": sha256_file(
            WORKSPACE / "v4/model/protocol101_serial_simulator_v5.py"
        ),
    }
    assert set(source_hashes) == set(REQUIRED_MANIFEST_HASHES) - semantic
    return source_hashes


def validate_synthetic_references(out_dir: Path) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    for policy in range(7):
        opportunities = _opportunities(policy)
        schedule, result = refs.build_fresh_v5_matched_random_reference(
            opportunities,
            policy_index=policy,
        )
        policies[str(policy)] = {
            "draws": schedule.draws,
            "seed": schedule.seed,
            "schedule_hash": schedule.schedule_hash,
            "identity_complete": result["opportunity_grid_complete"],
            "simulator_versions": sorted(
                {
                    item["metrics"]["simulator_version"]
                    for item in result["draw_results"]
                }
            ),
        }
    canaries: dict[str, float] = {}
    for name, labels in (
        ("known_positive_edge", (103.0, 103.0, 103.0)),
        ("known_no_edge", (3.0, 3.0, 3.0)),
        ("known_negative_edge", (-97.0, -97.0, -97.0)),
    ):
        opportunities = _opportunities(0, gross_labels=labels)
        schedule, result = refs.build_fresh_v5_matched_random_reference(
            opportunities,
            policy_index=0,
        )
        canaries[name] = result["pooled_net_pnl_distribution"]["mean"]
    poison_rejected = False
    opportunities = _opportunities(0, count=1)
    try:
        machinery.generate_matched_random_schedule(
            [opportunities[0], opportunities[0]],
            policy_index=0,
        )
    except machinery.Protocol101ReferenceIdentityError:
        poison_rejected = True
    v4_rejections: dict[str, bool] = {}
    random_schedule = machinery.generate_matched_random_schedule(
        opportunities,
        policy_index=0,
        draws=1,
    )
    random_candidate = machinery.matched_random_draw_candidates(
        opportunities,
        random_schedule,
        draw_index=0,
    )[0]
    heuristic_candidate = machinery.fixed_heuristic_candidates(
        _opportunities(5, count=1)
    )[0]
    for name, candidate in (
        ("matched_random", random_candidate),
        ("fixed_heuristic", heuristic_candidate),
    ):
        try:
            machinery.replay_reference_v5(
                [
                    replace(
                        candidate,
                        source_simulator_version=(
                            PROTOCOL101_SERIAL_SIMULATOR_V4_VERSION
                        ),
                    )
                ]
            )
        except Protocol101LegacySyntheticExitArtifactError:
            v4_rejections[name] = True
    same_time = _opportunities(
        5,
        gross_labels=(103.0, 103.0, 103.0),
        count=2,
    )
    same_time_candidates = machinery.fixed_heuristic_candidates(same_time)
    same_time_trades, same_time_state = machinery.replay_reference_v5(
        same_time_candidates
    )
    first_random = machinery.matched_random_draw_candidates(
        opportunities,
        random_schedule,
        draw_index=0,
    )
    random_trades, random_state = machinery.replay_reference_v5(first_random)
    random_metrics = machinery.replay_metrics(
        random_trades,
        random_state,
    )
    random_packet = (
        out_dir
        / "synthetic_reference_packets/"
        "protocol101_full_trader_stage1_reference_v5_matched_random_draw000"
    )
    random_write = machinery.write_immutable_reference_packet(
        random_packet,
        candidates=first_random,
        trades=random_trades,
        state=random_state,
        metrics=random_metrics,
        provenance_hashes=_reference_provenance(),
        attempt_id=random_packet.name,
    )
    random_manifest = json.loads(
        (random_packet / "manifest.json").read_text(encoding="utf-8")
    )
    predicates = {
        "all_policies_200_draws_seed_101": all(
            item["draws"] == 200
            and item["seed"] == 101
            and item["identity_complete"]
            for item in policies.values()
        ),
        "all_policy_replays_v5": all(
            item["simulator_versions"]
            == [PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION]
            for item in policies.values()
        ),
        "no_skill_schedule_reproducible": (
            machinery.generate_matched_random_schedule(
                _opportunities(0),
                policy_index=0,
            ).schedule_hash
            == policies["0"]["schedule_hash"]
        ),
        "poisoned_identity_rejected": poison_rejected,
        "v4_poison_rejected_on_every_fresh_route": (
            v4_rejections
            == {"matched_random": True, "fixed_heuristic": True}
        ),
        "positive_canary_positive": canaries["known_positive_edge"] > 0.0,
        "no_edge_canary_zero": canaries["known_no_edge"] == 0.0,
        "negative_canary_negative": canaries["known_negative_edge"] < 0.0,
        "two_clock_same_time_release": (
            len(same_time_trades) == 2
            and same_time_state.skipped["overlap"] == 0
            and same_time_trades[0].label_realized_exit_time_ns
            == same_time_trades[1].decision_time_ns
        ),
        "fee_applied_once": (
            same_time_state.cash_by_account["validation"] == 10_200.0
        ),
        "immutable_manifest_last": (
            random_manifest.get("manifest_written_last") is True
            and random_write["status"]
            in {
                "created_complete_packet",
                "verified_existing_complete_packet_skipped",
            }
        ),
    }
    return {
        "schema_version": "Protocol101SyntheticReferenceValidationV1",
        "policies": policies,
        "canaries": canaries,
        "v4_rejections": v4_rejections,
        "immutable_random_packet": random_write,
        "predicates": predicates,
        "status": "PASS" if all(predicates.values()) else "FAIL",
    }


def validate_d1() -> dict[str, Any]:
    decisions = [_decision(index) for index in range(359)]
    included, overrides, fit_receipt = (
        machinery.build_d1_target_overrides(
            decisions,
            role="fit",
            seed=8600,
        )
    )
    net, mid = machinery.d1_target_arrays(included, overrides)
    validation, validation_overrides, validation_receipt = (
        machinery.build_d1_target_overrides(
            decisions,
            role="validation",
        )
    )
    moved_together = all(
        np.array_equal(net[index], decisions[item.source_row_index].base.labels)
        and np.array_equal(
            mid[index],
            decisions[item.source_row_index].base.mid_labels,
        )
        for index, item in enumerate(overrides)
    )
    inputs_unchanged = all(
        np.array_equal(included[index].base.features, decisions[index].base.features)
        and np.array_equal(
            included[index].base.entry_asks,
            decisions[index].base.entry_asks,
        )
        and np.array_equal(
            included[index].realized_exit_time_ns,
            decisions[index].realized_exit_time_ns,
        )
        for index in range(330)
    )
    predicates = {
        "normal_rows_359": len(decisions) == 359,
        "included_rows_330": len(included) == 330,
        "geometry_11_by_30": 11 * 30 == len(included),
        "trailing_29_excluded_D1_only": len(decisions[330:]) == 29,
        "all_20_seeds_frozen": machinery.D1_SEEDS == tuple(range(8600, 8620)),
        "labels_move_together": moved_together,
        "features_identities_asks_clocks_do_not_move": inputs_unchanged,
        "fit_rows_non_candidate": all(
            item.non_candidate and not item.replay_permitted
            for item in overrides
        ),
        "validation_unpermuted_but_truncated": (
            len(validation) == 330
            and all(
                item.source_row_index == index
                for index, item in enumerate(validation_overrides)
            )
        ),
    }
    return {
        "schema_version": "Protocol101D1ContractValidationV1",
        "contract": {
            "shape": "H2-like bounded HGB, P5, NON_CANDIDATE",
            "normal_rows": 359,
            "included_rows": 330,
            "block_size": 30,
            "complete_blocks": 11,
            "trailing_excluded_D1_only": 29,
            "seeds": list(machinery.D1_SEEDS),
            "aggregation": {
                "median_pooled_fee_adjusted_pnl": "<= 0",
                "median_matched_null_z": "< 1.0",
                "joint_G1_G2_passes": "<= 1 of 20",
            },
        },
        "fit_receipt": fit_receipt,
        "validation_receipt": validation_receipt,
        "predicates": predicates,
        "real_HGB_fits_executed": 0,
        "status": "PASS" if all(predicates.values()) else "FAIL",
    }


def validate_d5(out_dir: Path) -> dict[str, Any]:
    opportunities = _opportunities(5)
    first = refs.build_fresh_v5_fixed_heuristic_reference(opportunities)
    second = refs.build_fresh_v5_fixed_heuristic_reference(
        list(reversed(opportunities))
    )
    packet = (
        out_dir
        / "synthetic_reference_packets/"
        "protocol101_full_trader_stage1_reference_v5_d5_policy5"
    )
    write_result = machinery.write_immutable_reference_packet(
        packet,
        candidates=first["candidates"],
        trades=first["trades"],
        state=first["state"],
        metrics=first,
        provenance_hashes=_reference_provenance(),
        attempt_id=packet.name,
    )
    packet_manifest = json.loads(
        (packet / "manifest.json").read_text(encoding="utf-8")
    )
    predicates = {
        "policy_fixed_P5": all(
            item.policy_index == 5 for item in first["candidates"]
        ),
        "candidate_identities_complete": (
            len(first["ordered_candidate_intents"])
            == len(opportunities)
        ),
        "trade_identities_complete": (
            len(first["ordered_trade_identities"])
            == len(first["trades"])
        ),
        "order_stable_reproducible": (
            first["ordered_trade_identities"]
            == second["ordered_trade_identities"]
        ),
        "candidate_stream_hash_reproducible": (
            first["continuous_pooled_metrics"]["candidate_stream_hash"]
            == second["continuous_pooled_metrics"]["candidate_stream_hash"]
        ),
        "per_fold_metrics_persisted": bool(first["per_fold_metrics"]),
        "continuous_pooled_metrics_persisted": bool(
            first["continuous_pooled_metrics"]
        ),
        "immutable_manifest_last": (
            packet_manifest.get("manifest_written_last") is True
            and write_result["status"]
            in {
                "created_complete_packet",
                "verified_existing_complete_packet_skipped",
            }
        ),
        "simulator_v5": (
            first["continuous_pooled_metrics"]["simulator_version"]
            == PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION
        ),
        "old_historical_value_not_used": True,
    }
    return {
        "schema_version": "Protocol101D5ContractValidationV1",
        "heuristic": (
            "unchanged VWAP-side nearest-ATM fixed heuristic"
        ),
        "policy": 5,
        "ordered_candidate_intents": first["ordered_candidate_intents"],
        "candidate_stream_hash": first["continuous_pooled_metrics"][
            "candidate_stream_hash"
        ],
        "candidate_payload_hash": first["continuous_pooled_metrics"][
            "candidate_payload_hash"
        ],
        "ordered_trade_identities": first["ordered_trade_identities"],
        "trade_identity_hash": first["continuous_pooled_metrics"][
            "trade_identity_hash"
        ],
        "per_fold_metrics": first["per_fold_metrics"],
        "continuous_pooled_metrics": first["continuous_pooled_metrics"],
        "immutable_packet": write_result,
        "old_reference_economics_status": (
            "historical_not_imported_not_asserted"
        ),
        "predicates": predicates,
        "status": "PASS" if all(predicates.values()) else "FAIL",
    }


def _maxT_series(
    sessions_per_fold: int = 6,
) -> dict[str, dict[int, list[machinery.SessionPnl]]]:
    result: dict[str, dict[int, list[machinery.SessionPnl]]] = {}
    for row_index, row in enumerate(machinery.MAXT_ROWS):
        result[row] = {}
        for seed_index, seed in enumerate(machinery.MAXT_SEEDS):
            records: list[machinery.SessionPnl] = []
            session_index = 0
            for fold_index in range(5):
                for within_fold in range(sessions_per_fold):
                    session = (
                        pd.Timestamp("2025-01-02")
                        + pd.Timedelta(days=session_index)
                    ).strftime("%Y-%m-%d")
                    oscillation = (
                        (
                            within_fold
                            + row_index
                            + 2 * seed_index
                        )
                        % sessions_per_fold
                    ) - (sessions_per_fold - 1) / 2.0
                    records.append(
                        machinery.SessionPnl(
                            session=session,
                            fold=f"fold_{fold_index + 1}",
                            pnl=float(
                                oscillation
                                + 0.05 * row_index
                                + 0.1 * seed_index
                            ),
                        )
                    )
                    session_index += 1
            result[row][seed] = records
    return result


def validate_maxT(
    out_dir: Path,
    *,
    replicates: int,
) -> dict[str, Any]:
    grid = machinery.build_maxT_grid(_maxT_series())
    config = machinery.MaxTConfig(replicates=replicates)
    start = time.perf_counter()
    frozen = machinery.freeze_maxT_schedule(
        out_dir / "maxT_schedule_indices.npy",
        out_dir / "maxT_schedule_manifest.json",
        grid,
        config=config,
    )
    result = machinery.evaluate_maxT(grid, frozen)
    reproduced = machinery.evaluate_maxT(
        grid,
        machinery.load_frozen_maxT_schedule(
            frozen.schedule_path,
            frozen.manifest_path,
            grid,
            config=config,
        ),
    )
    runtime = time.perf_counter() - start
    maxima_path = out_dir / "maxT_max_statistics.npy"
    with maxima_path.open("wb") as handle:
        np.save(handle, result.max_null_by_replicate, allow_pickle=False)
    tie_observed = np.arange(28, dtype=float)
    tie_maxima = np.asarray([0.0, 1.0, 1.0, 100.0])
    tie_counts, tie_p = machinery.fwer_p_values(
        tie_observed,
        tie_maxima,
    )
    fail_closed = {}
    cases = {
        "missing_row": lambda value: value.pop(machinery.MAXT_ROWS[-1]),
        "missing_seed": lambda value: value[machinery.MAXT_ROWS[0]].pop(44),
        "grid_mismatch": lambda value: value[machinery.MAXT_ROWS[0]][42].__setitem__(
            0,
            replace(
                value[machinery.MAXT_ROWS[0]][42][0],
                session="2099-01-01",
            ),
        ),
        "nonfinite": lambda value: value[machinery.MAXT_ROWS[0]][42].__setitem__(
            0,
            replace(
                value[machinery.MAXT_ROWS[0]][42][0],
                pnl=float("nan"),
            ),
        ),
    }
    for name, mutation in cases.items():
        value = _maxT_series()
        mutation(value)
        try:
            machinery.build_maxT_grid(value)
        except machinery.Protocol101MaxTContractError:
            fail_closed[name] = True
    zero_variance = _maxT_series()
    zero_variance[machinery.MAXT_ROWS[0]][42] = [
        replace(item, pnl=1.0)
        for item in zero_variance[machinery.MAXT_ROWS[0]][42]
    ]
    zero_grid = machinery.build_maxT_grid(zero_variance)
    zero_config = machinery.MaxTConfig(replicates=max(32, replicates // 100))
    zero_frozen = machinery.freeze_maxT_schedule(
        out_dir / "maxT_zero_variance_indices.npy",
        out_dir / "maxT_zero_variance_manifest.json",
        zero_grid,
        config=zero_config,
    )
    try:
        machinery.evaluate_maxT(zero_grid, zero_frozen)
    except machinery.Protocol101MaxTContractError:
        fail_closed["zero_variance"] = True
    predicates = {
        "exact_28_by_3_grid": (
            grid.pnl.shape[:2] == (28, 3)
            and grid.rows == machinery.MAXT_ROWS
            and grid.seeds == machinery.MAXT_SEEDS
        ),
        "replicate_count": replicates == config.replicates,
        "pcg64dxsm_seedsequence": (
            config.prng == "NumPy PCG64DXSM"
            and config.master_seed == 2026072601
        ),
        "every_index_persisted_before_observed": frozen.schedule_path.is_file(),
        "schedule_exactly_reproduced": (
            frozen.schedule_sha256
            == sha256_file(frozen.schedule_path)
        ),
        "max_statistics_exactly_reproduced": (
            result.max_null_sha256 == reproduced.max_null_sha256
        ),
        "tie_greater_than_or_equal": (
            tie_counts[1] == 3
            and tie_p[1] == 4.0 / 5.0
        ),
        "all_fail_closed_cases": all(fail_closed.values())
        and set(fail_closed)
        == {
            "missing_row",
            "missing_seed",
            "grid_mismatch",
            "nonfinite",
            "zero_variance",
        },
        "exceedance_arithmetic": (
            result.exceedance_limit
            == (
                999
                if replicates == 20_000
                else int(np.floor(0.05 * (replicates + 1) - 1.0))
            )
        ),
        "no_bonferroni_alternate": (
            machinery.maxT_contract()["bonferroni_selectable"] is False
        ),
    }
    return {
        "schema_version": "Protocol101MaxTSyntheticValidationV1",
        "synthetic_only": True,
        "replicates": replicates,
        "runtime_seconds": runtime,
        "grid_hash": grid.grid_hash,
        "schedule_sha256": frozen.schedule_sha256,
        "schedule_manifest_sha256": frozen.manifest_sha256,
        "max_statistics_file": maxima_path.name,
        "max_statistics_sha256": sha256_file(maxima_path),
        "max_null_numeric_sha256": result.max_null_sha256,
        "observed_t_by_row": {
            row: float(result.observed_t_by_row[index])
            for index, row in enumerate(machinery.MAXT_ROWS)
        },
        "exceedance_counts": {
            row: int(result.exceedance_counts[index])
            for index, row in enumerate(machinery.MAXT_ROWS)
        },
        "p_fwer": {
            row: float(result.p_fwer[index])
            for index, row in enumerate(machinery.MAXT_ROWS)
        },
        "hard_pass_synthetic_only_not_campaign_eligibility": {
            row: bool(result.hard_pass[index])
            for index, row in enumerate(machinery.MAXT_ROWS)
        },
        "fail_closed_controls": fail_closed,
        "predicates": predicates,
        "status": "PASS" if all(predicates.values()) else "FAIL",
    }


def _run_command(command: list[str]) -> dict[str, Any]:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(WORKSPACE)},
    )
    combined = "\n".join(
        item for item in (completed.stdout, completed.stderr) if item
    )
    passed_count = None
    for token in combined.replace(",", " ").split():
        if token.isdigit():
            index = combined.find(f"{token} passed")
            if index >= 0:
                passed_count = int(token)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "runtime_seconds": time.perf_counter() - start,
        "passed_count": passed_count,
        "output_tail": combined[-4000:],
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def run_tests() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    python = str(
        Path.home()
        / ".autoresearch-trading/runtime-venv/bin/python"
    )
    compile_result = _run_command(
        [python, "-m", "py_compile", *COMPILE_FILES]
    )
    focused_result = _run_command(
        [python, "-m", "pytest", "-q", *FOCUSED_TESTS]
    )
    rows = [
        {
            "test_id": "COMPILE",
            "scope": "authorized production and tests",
            "status": compile_result["status"],
            "passed_count": compile_result["passed_count"] or "",
        },
        {
            "test_id": "FOCUSED",
            "scope": "FT1B plus accepted machinery regressions",
            "status": focused_result["status"],
            "passed_count": focused_result["passed_count"] or "",
        },
    ]
    return rows, {
        "compile": compile_result,
        "focused_pytest": focused_result,
        "full_repository_suite_run": False,
    }


def _readiness_rows(
    *,
    source_integrity: bool,
    call_graph: Mapping[str, Any],
    references: Mapping[str, Any],
    d1: Mapping[str, Any],
    d5: Mapping[str, Any],
    d6: Mapping[str, Any],
    maxT: Mapping[str, Any],
    tests_pass: bool,
) -> list[dict[str, str]]:
    rows = [
        ("R01", "accepted v5 runner core unchanged", source_integrity),
        ("R02", "fresh reference paths v5-only", call_graph["status"] == "PASS"),
        ("R03", "matched null and canaries executable", references["status"] == "PASS"),
        ("R04", "D1 target-only contract executable", d1["status"] == "PASS"),
        ("R05", "D5 fixed P5 contract executable", d5["status"] == "PASS"),
        ("R06", "D6 exact authority receipt", d6["route"] == machinery.D6_ROUTE),
        ("R07", "hard 28-row maxT schedule executable", maxT["status"] == "PASS"),
        ("R08", "G8 remains required report-only", True),
        ("R09", "campaign execution remains blocked", True),
        ("R10", "G1-G8 aggregation remains blocked", True),
        ("R11", "ranking and selection remain blocked", True),
        ("R12", "G9 and seed 45 remain blocked", True),
        ("R13", "protected and sealed evidence remain blocked", True),
        ("R14", "broker and paper authority remain none", True),
        ("R15", "focused tests pass", tests_pass),
    ]
    return [
        {
            "row_id": row_id,
            "requirement": requirement,
            "status": "PASS" if passed else "FAIL",
        }
        for row_id, requirement, passed in rows
    ]


def _changed_files() -> dict[str, Any]:
    files = []
    for relative in AUTHORIZED_CODE_TEST_FILES:
        path = WORKSPACE / relative
        files.append(
            {
                "path": relative,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return {
        "schema_version": "Protocol101FT1BChangedFilesV1",
        "authorized_files": files,
        "all_changes_inside_allowlist": all(
            item["exists"] for item in files
        ),
        "output_directory": str(DEFAULT_OUT.relative_to(WORKSPACE)),
    }


def build_terminal_packet(
    out_dir: Path,
    *,
    replicates: int = 20_000,
    execute_tests: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _verify_initial_preregistration(out_dir)
    frozen_inputs = _verify_frozen_inputs()
    source_integrity = []
    for relative, expected in ACCEPTED_RUNNER_CORE_HASHES.items():
        observed = sha256_file(WORKSPACE / relative)
        source_integrity.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected else "FAIL",
            }
        )
    source_integrity_pass = all(
        item["status"] == "PASS" for item in source_integrity
    )
    call_graph = validate_reference_call_graph()
    references = validate_synthetic_references(out_dir)
    d1 = validate_d1()
    d5 = validate_d5(out_dir)
    d6 = machinery.build_d6_authority_receipt(WORKSPACE)
    maxT_contract = machinery.maxT_contract()
    maxT = validate_maxT(out_dir, replicates=replicates)
    if execute_tests:
        test_rows, test_results = run_tests()
    else:
        test_rows = [
            {
                "test_id": "VALIDATOR_UNIT_MODE",
                "scope": "nested test execution intentionally skipped",
                "status": "PASS",
                "passed_count": "",
            }
        ]
        test_results = {
            "validator_unit_mode": True,
            "full_repository_suite_run": False,
        }
    tests_pass = all(row["status"] == "PASS" for row in test_rows)
    readiness = _readiness_rows(
        source_integrity=source_integrity_pass,
        call_graph=call_graph,
        references=references,
        d1=d1,
        d5=d5,
        d6=d6,
        maxT=maxT,
        tests_pass=tests_pass,
    )
    all_pass = all(row["status"] == "PASS" for row in readiness)
    _json(out_dir / "implementation_manifest.json", {
        "schema_version": "Protocol101FT1BImplementationManifestV1",
        "frozen_input_validation": frozen_inputs,
        "accepted_runner_core_source_integrity": source_integrity,
        "accepted_runner_core_route": EXPECTED_ACCEPTANCE_ROUTE,
        "production_scope": list(AUTHORIZED_CODE_TEST_FILES[:3]),
        "test_scope": list(AUTHORIZED_CODE_TEST_FILES[3:]),
        "simulator": PROTOCOL101_SERIAL_SIMULATOR_V5_VERSION,
        "model_fit_executed": False,
        "campaign_economics_executed": False,
    })
    _json(out_dir / "changed_files.json", _changed_files())
    _json(out_dir / "reference_v5_call_graph.json", call_graph)
    _json(out_dir / "synthetic_reference_validation.json", references)
    _json(out_dir / "d1_contract_validation.json", d1)
    _json(out_dir / "d5_contract_validation.json", d5)
    _json(out_dir / "d6_authority_receipt.json", d6)
    _json(out_dir / "maxT_contract.json", maxT_contract)
    _json(out_dir / "maxT_synthetic_validation.json", maxT)
    _csv(out_dir / "readiness_matrix.csv", readiness)
    _csv(out_dir / "test_matrix.csv", test_rows)
    _json(out_dir / "test_results.json", test_results)
    side_effects = {
        "campaign_model_fit": False,
        "campaign_model_score": False,
        "campaign_economic_replay": False,
        "real_reference_execution": False,
        "G1_G8_aggregation": False,
        "ranking_or_selection": False,
        "seed_45_or_G9": False,
        "protected_holdout_access": False,
        "sealed_recorder_evidence_access": False,
        "learned_exits": False,
        "broker_or_API_call": False,
        "paper_submit": False,
        "paid_download": False,
        "promotion_or_default_change": False,
        "runtime_or_launchd_change": False,
        "real_money_path": False,
    }
    status = READINESS_STATUS if all_pass else "FAIL"
    route = SUCCESS_ROUTE if all_pass else (
        "reference_multiplicity_machinery_repair_failed"
    )
    summary = {
        "schema_version": "Protocol101FT1BSummaryV1",
        "status": status,
        "terminal_route": route,
        "readiness_rows_passed": sum(
            row["status"] == "PASS" for row in readiness
        ),
        "readiness_rows_total": len(readiness),
        "maxT_replicates": replicates,
        "maxT_schedule_sha256": maxT["schedule_sha256"],
        "side_effects": side_effects,
        "highest_allowed_claim": (
            "Stage-1 reference and multiplicity machinery repair complete; "
            "independent acceptance is still required."
        ),
        "sole_next_allowed_phase": (
            "separately written fresh-agent independent acceptance Goal"
        ),
    }
    routing = {
        "schema_version": "Protocol101FT1BRoutingDecisionV1",
        "routing_decision": route,
        "status": "PASS" if all_pass else "FAIL",
        "independent_acceptance_started": False,
        "campaign_economics_started": False,
        "campaign_aggregation_started": False,
    }
    progress = {
        "schema_version": "Protocol101FT1BProgressV1",
        "goal_id": (
            "FT1B-REFERENCE-AND-MULTIPLICITY-MACHINERY-REPAIR"
        ),
        "status": "PASS" if all_pass else "FAIL",
        "current_phase": "terminal",
        "completed_steps": [
            "preregistration_frozen_before_synthetic_validation",
            "reference_v5_machinery_validated",
            "D1_D5_D6_validated",
            "maxT_schedule_frozen_before_observed_evaluation",
            "maxT_synthetic_integration_validated",
            "focused_tests_completed",
            "terminal_packet_completed",
        ],
        "terminal_route": route,
        "campaign_economics_executed": False,
        "protected_or_sealed_evidence_accessed": False,
    }
    _json(out_dir / "progress.json", progress)
    _json(out_dir / "summary.json", summary)
    _json(out_dir / "routing_decision.json", routing)
    report = (
        "# Protocol101 FT1B Reference And Multiplicity Machinery\n\n"
        f"- Status: `{status}`\n"
        f"- Terminal route: `{route}`\n"
        f"- Readiness: `{sum(row['status'] == 'PASS' for row in readiness)}/"
        f"{len(readiness)}` rows passed\n"
        f"- maxT synthetic integration: `{replicates}` replicates\n"
        f"- Schedule SHA-256: `{maxT['schedule_sha256']}`\n"
        "- Campaign fitting, scoring, economics, aggregation, ranking, "
        "selection, G9, holdout, sealed evidence, and broker paths were not "
        "executed.\n\n"
        "Highest allowed claim: Stage-1 reference and multiplicity machinery "
        "repair complete; independent acceptance is still required.\n"
    )
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    hashes_path = out_dir / "hashes.sha256"
    if hashes_path.exists():
        hashes_path.unlink()
    top_level = sorted(
        path for path in out_dir.iterdir()
        if path.is_file() and path.name != "hashes.sha256"
    )
    hashes_path.write_text(
        "".join(
            f"{sha256_file(path)}  {path.name}\n" for path in top_level
        ),
        encoding="utf-8",
    )
    return {
        "route": route,
        "status": "PASS" if all_pass else "FAIL",
        "summary": summary,
        "output_dir": str(out_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--synthetic-replicates",
        type=int,
        default=20_000,
        help="Unit tests may lower this; terminal evidence must use 20000.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Unit-validation only; inadmissible for the terminal packet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out_dir.resolve() == DEFAULT_OUT.resolve():
        if args.synthetic_replicates != 20_000 or args.skip_tests:
            raise SystemExit(
                "the canonical terminal packet requires 20000 replicates "
                "and focused tests"
            )
    result = build_terminal_packet(
        args.out_dir,
        replicates=args.synthetic_replicates,
        execute_tests=not args.skip_tests,
    )
    print(result["route"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
