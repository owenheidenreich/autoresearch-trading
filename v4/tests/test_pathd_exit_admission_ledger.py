from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from v4.research.pathd_exit_admission_ledger import (
    EXIT_FAMILIES,
    EXIT_LEDGER_FEATURE_NAMES,
    EXIT_PARENTS,
    PNL_VELOCITY_FEATURE_NAMES,
    admitted_exit_feature_matrix,
    assert_exit_feature_matrix_admitted,
    generate_exit_ledger,
    verify_exit_ledger,
)
from v4.research.pathd_feature_admission_ledger import AdmissionLedgerError


def test_exit_ledger_covers_every_fitted_feature() -> None:
    """A fitted feature absent from the ledger raises for the wrong reason.

    NOT_IN_EXIT_LEDGER reads like a wiring bug; the honest message is BARRED,
    meaning "known, and not yet certified".
    """

    from v4.research.phase1_exit_model import EXIT_FEATURE_NAMES

    assert set(EXIT_FEATURE_NAMES) <= set(EXIT_LEDGER_FEATURE_NAMES)


def test_pnl_velocity_constant_matches_the_exit_model() -> None:
    """The ledger restates this tuple because importing it would be circular.

    phase1_exit_model imports the ledger for enforcement, so the ledger cannot
    import back. This test is what keeps the two copies honest.
    """

    from v4.research.phase1_exit_model import PNL_VELOCITY_FEATURE_NAMES as fitted

    assert PNL_VELOCITY_FEATURE_NAMES == fitted


def test_pnl_velocity_lives_with_account_state_not_its_own_family() -> None:
    """They are time derivatives of current_net_pnl_dollars.

    A separate family would imply an independent source and an independent
    arrival clock; neither exists.
    """

    assert set(PNL_VELOCITY_FEATURE_NAMES) <= set(
        EXIT_FAMILIES["exit.causal_account_state.v1"]
    )


def test_exit_families_inherit_entry_substrate_rather_than_restating_it() -> None:
    """Exit reads the same feeds as entry, so it must inherit their receipts.

    Track A unblocks the exit OPRA families at the same moment it unblocks the
    entry ones. Declaring independent receipts would imply the arrival clock
    has to be measured twice.
    """

    assert EXIT_PARENTS["exit.opra_cbbo_quote.v1"] == ("entry.opra_cbbo1s_rolling.v1",)
    assert EXIT_PARENTS["exit.causal_account_state.v1"] == (
        "entry.causal_account_state.v1",
    )


def test_signed_exit_ledger_verifies_and_bars_everything_today() -> None:
    payload = verify_exit_ledger()
    statuses = {str(row["status"]) for row in payload["features"]}
    assert statuses == {"BARRED"}
    assert len(payload["features"]) == len(EXIT_LEDGER_FEATURE_NAMES)


def test_exit_fit_is_blocked_at_the_estimator_boundary() -> None:
    from v4.research.phase1_exit_model import EXIT_FEATURE_NAMES

    with pytest.raises(AdmissionLedgerError) as excinfo:
        assert_exit_feature_matrix_admitted(EXIT_FEATURE_NAMES)
    message = str(excinfo.value)
    assert "BARRED" in message
    assert "NOT_IN_EXIT_LEDGER" not in message


def test_greeks_resolve_to_the_root_blocker_across_namespaces(tmp_path: Path) -> None:
    """greeks -> exit quote -> entry 1s rolling must name the entry root.

    Naming exit.opra_cbbo_quote.v1 would point at a family that is itself
    barred for the same missing receipt, hiding the actual blocker.
    """

    path = tmp_path / "exit_ledger.json"
    generate_exit_ledger(ledger_path=path, entry_admitted_families=())
    payload = json.loads(path.read_text())
    greeks = next(
        row
        for row in payload["features"]
        if row["contract_id"] == "exit.self_computed_greeks.v1"
    )
    assert greeks["barred_reason"] == (
        "parent_family_not_admitted:entry.opra_cbbo1s_rolling.v1"
    )


def test_admitting_the_entry_parent_moves_the_exit_blocker_inward(tmp_path: Path) -> None:
    """With the entry root admitted, greeks should blame its direct parent."""

    path = tmp_path / "exit_ledger.json"
    generate_exit_ledger(
        ledger_path=path, entry_admitted_families=("entry.opra_cbbo1s_rolling.v1",)
    )
    payload = json.loads(path.read_text())
    greeks = next(
        row
        for row in payload["features"]
        if row["contract_id"] == "exit.self_computed_greeks.v1"
    )
    assert greeks["barred_reason"] == (
        "parent_family_not_admitted:exit.opra_cbbo_quote.v1"
    )


def test_generate_refuses_to_overwrite_a_verifying_signed_exit_ledger() -> None:
    before = verify_exit_ledger()["ledger_sha256"]
    with pytest.raises(AdmissionLedgerError, match="refusing to overwrite"):
        generate_exit_ledger()
    assert verify_exit_ledger()["ledger_sha256"] == before


def test_tampering_with_the_exit_ledger_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "exit_ledger.json"
    generate_exit_ledger(ledger_path=path, entry_admitted_families=())
    payload = json.loads(path.read_text())
    payload["features"][0]["status"] = "ADMITTED"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    with pytest.raises(AdmissionLedgerError, match="ledger_sha256 mismatch"):
        verify_exit_ledger(path)


def test_admitted_exit_feature_matrix_rejects_a_barred_column() -> None:
    frame = pd.DataFrame({"option_bid": [1.0]})
    with pytest.raises(AdmissionLedgerError, match="option_bid:BARRED"):
        admitted_exit_feature_matrix(frame, ("option_bid",))


def _receipt(tmp_path: Path) -> dict:
    import hashlib

    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    return {
        "receipts": [
            {
                "name": "r",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
        "availability_clock_ms": 12.5,
        "tolerance": 0.0,
    }


def test_a_family_cannot_be_admitted_past_an_unadmitted_parent(tmp_path: Path) -> None:
    """Receipts prove the transform, not when its inputs arrive.

    Admitting a child while its parent lacks an arrival clock is exactly what
    put implied_spot, IV and the greeks into the signed18 look-ahead class on
    the entry side. Supplying receipts must not override the parent gate.
    """

    path = tmp_path / "exit_ledger.json"
    generate_exit_ledger(
        ledger_path=path,
        entry_admitted_families=(),
        admit={"exit.opra_cbbo_quote.v1": _receipt(tmp_path)},
    )
    payload = json.loads(path.read_text())
    quote = next(
        row for row in payload["features"] if row["contract_id"] == "exit.opra_cbbo_quote.v1"
    )
    assert quote["status"] == "BARRED"
    assert quote["barred_reason"] == (
        "parent_family_not_admitted:entry.opra_cbbo1s_rolling.v1"
    )


def test_admission_requires_both_receipts_and_a_clock(tmp_path: Path) -> None:
    path = tmp_path / "exit_ledger.json"
    broken = dict(_receipt(tmp_path))
    broken["availability_clock_ms"] = None
    with pytest.raises(AdmissionLedgerError, match="without receipts and a clock"):
        generate_exit_ledger(
            ledger_path=path,
            entry_admitted_families=("entry.opra_cbbo1s_rolling.v1",),
            admit={"exit.opra_cbbo_quote.v1": broken},
        )


def test_admitted_family_verifies_and_unblocks_its_features(tmp_path: Path) -> None:
    path = tmp_path / "exit_ledger.json"
    generate_exit_ledger(
        ledger_path=path,
        entry_admitted_families=("entry.opra_cbbo1s_rolling.v1",),
        admit={"exit.opra_cbbo_quote.v1": _receipt(tmp_path)},
    )
    verify_exit_ledger(path)
    assert assert_exit_feature_matrix_admitted(("option_bid", "option_ask"), ledger_path=path)


def test_admit_rejects_an_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(AdmissionLedgerError, match="unknown exit family"):
        generate_exit_ledger(
            ledger_path=tmp_path / "l.json",
            admit={"exit.not_a_family.v1": _receipt(tmp_path)},
        )
