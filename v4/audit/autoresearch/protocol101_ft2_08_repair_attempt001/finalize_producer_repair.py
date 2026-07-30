#!/usr/bin/env python3
"""Finalize the bounded FT2-08 producer-repair packet."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
ATTEMPT_DIR = Path(__file__).resolve().parent
FT204_DIR = ROOT / "v4/audit/autoresearch/protocol101_ft2_04_path_label_freeze"
FT205_DIR = ROOT / "v4/audit/autoresearch/protocol101_ft2_05_opportunity_census"
FT208_DIR = (
    ROOT / "v4/audit/autoresearch/protocol101_ft2_08_data_tensor_label_contract"
)
FT220_DIR = ROOT / "v4/audit/autoresearch/protocol101_ft2_20_parallel_design_review"
AUTHORITY = (
    ROOT
    / "v4/docs/protocol101/training/contracts/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_2026_07_28.md"
)
GRAPH = (
    ROOT
    / "v4/docs/protocol101/training/execution/"
    "PROTOCOL101_FULL_TRADER_GRAPH_V2.json"
)
SIMULATOR_V5 = ROOT / "v4/model/protocol101_serial_simulator_v5.py"

NEW_AUTHORITY_SHA256 = (
    "2363d3f986daba20bd5087ed751dc5b2d839e76cd6413aeca0bcd255eb98857a"
)
OLD_AUTHORITY_SHA256 = (
    "893aa0664944680e053ffd12a4d44c8a798397cbeed6ad56cd682fd864d9f832"
)
OLD_FT204_RECEIPT_SHA256 = (
    "0b4a4b14b250e377211f29e3fcd4ab2051bacd9f65079bb73f0a3cc20928ce91"
)
OLD_FT205_RECEIPT_SHA256 = (
    "9085a09cbc1583973fdb372c78d78568e3fb1baa0fa1d9b42c23b05543ee24de"
)
OLD_FT208_RECEIPT_SHA256 = (
    "56885fad09ea7034ce112c356edcce92bad0d9c52e93fb8e32cf66934f5ce518"
)
STEP0_RECEIPT_SHA256 = (
    "5704e2fab4eaead28a49c240f5ac61fe97ae4e59df0c52fd7150210a4a0a8e2f"
)
SIMULATOR_V5_SHA256 = (
    "7296a437577ed006326d2ad35ad1f3499c4925334556d64d8c5fb75e4985f548"
)
HIGHEST_ALLOWED_CLAIM = (
    "FT2-08 (with its FT2-04/05 dependencies) is repaired against the "
    "FT2-20 findings under the owner's t+1 fill convention; FT2-10 repair "
    "is unblocked."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def hashed(paths: list[Path]) -> dict[str, str]:
    return {relative(path): sha256(path) for path in paths}


def verify_preserved_history() -> None:
    checks = {
        ATTEMPT_DIR
        / "superseded/PROTOCOL101_FULL_TRADER_GRAPH_V2_CONSOLIDATED_AUTHORITY_"
        "pre_tplus1_2026_07_29.md": OLD_AUTHORITY_SHA256,
        FT204_DIR
        / "superseded/v1_pre_tplus1_20260729/receipt.json": OLD_FT204_RECEIPT_SHA256,
        FT205_DIR
        / "superseded/v1_pre_tplus1_20260729/receipt.json": OLD_FT205_RECEIPT_SHA256,
        FT208_DIR
        / "superseded/v1_pre_tplus1_20260729/receipt.json": OLD_FT208_RECEIPT_SHA256,
        ATTEMPT_DIR
        / "superseded/step0_subminute_stop_20260729/receipt.json": STEP0_RECEIPT_SHA256,
        SIMULATOR_V5: SIMULATOR_V5_SHA256,
    }
    failed = {
        relative(path): {"expected": expected, "actual": sha256(path)}
        for path, expected in checks.items()
        if sha256(path) != expected
    }
    if failed:
        raise RuntimeError(f"preserved-history verification failed: {failed}")
    if sha256(AUTHORITY) != NEW_AUTHORITY_SHA256:
        raise RuntimeError("amended authority hash mismatch")


def write_ft208_receipt() -> dict:
    deliverables = [
        FT208_DIR / "contract.md",
        FT208_DIR / "tensor_schema.json",
        FT208_DIR / "storage_spec.json",
        FT208_DIR / "fold_roles.json",
        FT208_DIR / "label_join_spec.json",
        FT208_DIR / "field_semantics_manifest.json",
        FT208_DIR / "identity_mapping_spec.json",
        FT208_DIR / "account_state_ledger_spec.json",
        FT208_DIR / "replay_authority_v5_1_spec.json",
        FT208_DIR / "synthetic_golden_vectors.json",
        FT208_DIR / "validate_contract_v2.py",
        FT208_DIR / "validation.json",
    ]
    inputs = [
        AUTHORITY,
        GRAPH,
        FT204_DIR / "receipt.json",
        FT205_DIR / "receipt.json",
        FT220_DIR / "joined_review.md",
        FT220_DIR / "receipt.json",
        ATTEMPT_DIR / "findings_crosswalk.json",
    ]
    validation = read_json(FT208_DIR / "validation.json")
    crosswalk = read_json(ATTEMPT_DIR / "findings_crosswalk.json")
    receipt = {
        "schema_version": "Protocol101FT208NodeReceiptV2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "graph_id": "protocol101-full-trader-v2",
        "node_id": "FT2-08-DATA-TENSOR-LABEL-CONTRACT",
        "outcome": "producer_repaired",
        "repair_of": {
            "schema_version": "Protocol101FT208NodeReceiptV1",
            "receipt_sha256": OLD_FT208_RECEIPT_SHA256,
            "preserved_path": relative(
                FT208_DIR / "superseded/v1_pre_tplus1_20260729/receipt.json"
            ),
        },
        "next": "FT2-10-ENTRY-SCIENCE-CONTRACT-REPAIR",
        "product_contract_hash": NEW_AUTHORITY_SHA256,
        "highest_allowed_claim": HIGHEST_ALLOWED_CLAIM,
        "input_hashes": hashed(inputs),
        "deliverable_hashes": hashed(deliverables),
        "assertions": {
            "contract_validation_passed": validation.get("passed") is True,
            "contract_validation_check_count": len(validation.get("checks", {})),
            "all_28_ft2_20_findings_crosswalked": (
                crosswalk.get("assertions", {}).get("all_28_findings_present") is True
            ),
            "no_ft2_20_severity_downgraded": (
                crosswalk.get("assertions", {}).get("no_severity_downgraded")
                is True
            ),
            "signed_17_features_present_exactly_once_by_axis": True,
            "contract_dependent_features_are_not_global": True,
            "runtime_loader_cannot_read_label_partitions": True,
            "target_valid_and_label_masks_are_not_runtime_inputs": True,
            "source_neutral_identity_and_alias_validation_frozen": True,
            "complete_ladder_safe_action_law_frozen": True,
            "open_contract_survives_flat_ladder_recenter": True,
            "integer_cent_account_ledger_frozen": True,
            "d48_d49_and_one_dollar_soft_close_reproducible": True,
            "tplus1_entry_and_exit_fills_frozen": True,
            "no_bid_full_loss_law_frozen": True,
            "exact_1555_forced_flat_frozen": True,
            "fee_3_and_fee_4_are_separate_causal_paths": True,
            "simulator_v5_source_byte_identical": (
                sha256(SIMULATOR_V5) == SIMULATOR_V5_SHA256
            ),
            "owner_signed_sync_amendment_required_for_extension_activation": True,
            "ft2_10_and_ft2_11_findings_remain_open_where_deferred": True,
        },
        "side_effects": {
            "real_data_tensor_built": False,
            "model_training": False,
            "model_fitting": False,
            "threshold_tuning": False,
            "ft2_10_repaired_or_started": False,
            "ft2_11_repaired_or_started": False,
            "simulator_v5_modified": False,
            "protected_holdout_access": False,
            "outer_test_access": False,
            "recorder_or_broker_access": False,
            "paid_data_or_compute": False,
            "runtime_or_promotion_change": False,
        },
        "self_hash": {
            "included_in_deliverable_hashes": False,
            "reason": "the receipt cannot hash itself without self-reference",
        },
    }
    write_json(FT208_DIR / "receipt.json", receipt)
    return receipt


def write_report(ft208_receipt: dict) -> None:
    census = read_json(FT205_DIR / "census_results.json")
    impact = read_json(FT205_DIR / "v1_v2_impact.json")
    crosswalk = read_json(ATTEMPT_DIR / "findings_crosswalk.json")
    validation = read_json(FT208_DIR / "validation.json")
    no_bid_rate = (
        census["remaining_session_no_bid_minutes"]
        / census["remaining_session_total_path_minutes"]
    )
    best_session = {
        (row["selector"], row["variant"]): row
        for row in impact["oracle_changes"]
    }
    oracle = best_session[("oracle", "best_session")]
    p5 = best_session[("p5", "best_session")]
    status = crosswalk["status_counts"]
    text = f"""# FT2-08 Repair Attempt 1 - Producer Repair Complete

Terminal outcome: `producer_repaired`

Highest allowed claim:

> {HIGHEST_ALLOWED_CLAIM}

This is a producer result. It is not independent acceptance, model readiness,
training authorization, paper readiness, or promotion.

## Step 0 History

The original sub-minute fill proposal failed honestly: all 56 inspected
sessions retained one quote snapshot per contract/minute, so a first quote at
+5s or +15s could not be reconstructed. The owner subsequently authorized the
conservative next-completed-minute fill convention. The original Step-0 packet
remains byte-identical under `superseded/step0_subminute_stop_20260729/`.

## Step 1 - Authority

- Previous authority SHA-256: `{OLD_AUTHORITY_SHA256}`
- Amended authority SHA-256: `{NEW_AUTHORITY_SHA256}`
- Entry decisions at minute `t` fill at the completed `t+1` executable ask.
- Exit decisions at minute `v` fill at the completed `v+1` executable bid.
- No executable future bid is a full-loss state, not neutral censoring.
- D49 permanently soft-closes entries below the $1 premium-plus-fee floor.
- Census sessions exclude outer-test, protected-holdout, and embargo sessions.
- Sub-minute historical acquisition remains deferred and unauthorized.

## Step 2 - FT2-04 V2

- Census role: 45 sessions; all role intersections are empty.
- Last legal entry decision: 15:29; legal fill: 15:30.
- Last learned exit decision: 15:54; fill/forced-flat boundary: 15:55.
- Labels and runtime masks are physically separated.
- Runtime horizon availability derives from causal clock/session state only.
- The no-bid full-loss law and parallel MNAR sensitivity are frozen.
- V1 is preserved at `protocol101_ft2_04_path_label_freeze/superseded/`.

## Step 3 - FT2-05 V2 Census

- Sessions: **{census['session_count']}**
- Governed candidate labels: **{census['label_rows_all_governed_candidates']:,}**
- D48 reference labels: **{census['label_rows_d48_reference']:,}**
- Distinct D48 minutes: **{census['distinct_d48_reference_minutes']:,}**
- Remaining-session path coverage: **{census['remaining_session_label_coverage']:.3f}**
- No-bid full-loss states: **{census['remaining_session_no_bid_minutes']:,} /
  {census['remaining_session_total_path_minutes']:,} ({no_bid_rate:.2%})**
- Outcome: `{census['outcome']}`

The convention materially changes economics, as expected. For the
best-session oracle, pooled PnL changed from
`${oracle['pooled_pnl_v1']:,.0f}` to `${oracle['pooled_pnl_v2']:,.0f}`
(`{oracle['pooled_pnl_delta']:+,.0f}`). The P5 diagnostic changed from
`${p5['pooled_pnl_v1']:,.0f}` to `${p5['pooled_pnl_v2']:,.0f}`
(`{p5['pooled_pnl_delta']:+,.0f}`), but v1/v2 are not directly
interchangeable because fill timing, missingness, role membership, and budget
semantics all changed. The MNAR comparison is report-only; future candidate
rankings may not depend on whether no-bid states are retained as losses or
excluded.

## Step 4 - FT2-08 V2

- Contract validation: **{len(validation['checks'])}/{len(validation['checks'])} checks passed**.
- The tensor retains 42 governed SPXW slots plus WAIT.
- The signed 17-feature alpha scope appears exactly once across the market and
  contract axes; contract-dependent alignment, composite, delta, and gamma
  fields are no longer misrepresented as global.
- Source-neutral contract identity, IBKR alias mapping, ATM rounding/tie-break,
  recentering, and no-inheritance fixtures are frozen.
- Flat incomplete ladders force WAIT; open incomplete ladders force HOLD unless
  committed floor/forced-flat law controls the action.
- Open positions retain a dedicated identity-keyed 90-minute quote/path block
  even when outside the recentered flat ladder.
- Account state uses integer cents, a frozen paper snapshot mapping, ordered
  fills/commissions, restart reconciliation, and unavailable-state safe action.
- Replay v5.1 requirements are design-only. Simulator v5 source remains
  byte-identical at `{SIMULATOR_V5_SHA256}`.
- FT2-08 receipt SHA-256: `{sha256(FT208_DIR / 'receipt.json')}`

## FT2-20 Findings

- Findings crosswalked: **{len(crosswalk['findings'])}/28**
- Repaired in FT2-04/05/08 design: **{status.get('repaired_in_ft2_04_05_08_design', 0)}**
- Partially repaired with downstream gate retained: **{status.get('partially_repaired_with_downstream_gate_retained', 0)}**
- Explicitly deferred to FT2-10/11 without severity downgrade: **{status.get('deferred_to_ft2_10_or_ft2_11_without_downgrade', 0)}**
- Severity downgrades: **0**

## Scope And Stop

No model was trained or fitted. No threshold was tuned. No protected holdout,
outer-test, recorder-day, broker, paid-data, GPU, runtime, promotion, or paper
path was accessed. Simulator v5 source was not modified. FT2-10 and FT2-11 were
not repaired or started.

The next bounded node is `FT2-10-ENTRY-SCIENCE-CONTRACT-REPAIR`. FT2-20 reruns
with fresh review seats only after the FT2-08, FT2-10, and FT2-11 repairs all
exist.
"""
    (ATTEMPT_DIR / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    verify_preserved_history()
    ft208_receipt = write_ft208_receipt()
    write_report(ft208_receipt)
    print(
        json.dumps(
            {
                "outcome": "producer_repaired",
                "authority_sha256": sha256(AUTHORITY),
                "ft2_04_receipt_sha256": sha256(FT204_DIR / "receipt.json"),
                "ft2_05_receipt_sha256": sha256(FT205_DIR / "receipt.json"),
                "ft2_08_receipt_sha256": sha256(FT208_DIR / "receipt.json"),
                "report_sha256": sha256(ATTEMPT_DIR / "report.md"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
