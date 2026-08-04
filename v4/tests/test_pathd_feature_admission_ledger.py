from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from v4.research.pathd_feature_admission_ledger import (
    AdmissionLedgerError,
    admitted_feature_matrix,
    generate_ledger,
    ledger_sha256,
    verify_ledger,
)
from v4.research.autoresearch_v2.models import fit_oof


def _ledger(tmp_path: Path) -> Path:
    path = tmp_path / "feature_admission_ledger.json"
    generate_ledger(ledger_path=path, receipt_dir=tmp_path / "receipts")
    return path


def _resign(path: Path, payload: dict) -> None:
    payload["ledger_sha256"] = ledger_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_training_matrix_with_admitted_feature_set_succeeds(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    frame = pd.DataFrame({"is_call": [1.0], "strike": [7500.0]})
    matrix = admitted_feature_matrix(frame, ("is_call", "strike"), ledger_path=path)
    assert matrix.to_dict("records") == [{"is_call": 1.0, "strike": 7500.0}]


def test_single_non_admitted_feature_raises_with_name_and_status(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    frame = pd.DataFrame({"is_call": [1.0], "option_bid": [1.0]})
    with pytest.raises(AdmissionLedgerError, match=r"option_bid:BARRED"):
        admitted_feature_matrix(frame, ("is_call", "option_bid"), ledger_path=path)


def test_oof_training_path_blocks_before_estimator_construction() -> None:
    spec = SimpleNamespace(features=(SimpleNamespace(name="option_bid"),))
    with pytest.raises(AdmissionLedgerError, match=r"option_bid:BARRED"):
        fit_oof(
            pd.DataFrame({"option_bid": [1.0]}),
            spec=spec,
            folds=(),
            foundation_hash="unused",
            cache=SimpleNamespace(),
        )


def test_permanently_barred_intraday_open_interest_raises(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    frame = pd.DataFrame({"intraday_open_interest": [100.0]})
    with pytest.raises(AdmissionLedgerError, match=r"intraday_open_interest:BARRED"):
        admitted_feature_matrix(frame, ("intraday_open_interest",), ledger_path=path)


@pytest.mark.parametrize("kind", ["missing", "empty"])
def test_missing_or_empty_ledger_raises_fail_closed(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "ledger.json"
    if kind == "empty":
        path.write_text("")
    with pytest.raises(AdmissionLedgerError, match="missing_or_empty"):
        admitted_feature_matrix(pd.DataFrame({"is_call": [1.0]}), ("is_call",), ledger_path=path)


def test_tampered_ledger_hash_raises(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    payload = json.loads(path.read_text())
    payload["features"][0]["tolerance"] = 99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(AdmissionLedgerError, match="ledger_sha256 mismatch"):
        verify_ledger(path)


def test_tier2_cannot_be_admitted_before_parent(tmp_path: Path) -> None:
    path = _ledger(tmp_path)
    payload = json.loads(path.read_text())
    admitted_receipts = next(
        row["receipts"] for row in payload["features"] if row["name"] == "is_call"
    )
    for row in payload["features"]:
        if row["contract_id"] == "entry.opra_implied_volatility.v1":
            row["status"] = "ADMITTED"
            row["receipts"] = admitted_receipts
            row["availability_clock_ms"] = 1.0
            row["barred_reason"] = None
    _resign(path, payload)
    with pytest.raises(AdmissionLedgerError, match="tier2_parent_not_admitted"):
        verify_ledger(path)


# --- corrected 2026-08-04: parent graph must match declared data dependencies ---


def _transitive_parents(contract_id: str) -> set[str]:
    from v4.research.pathd_feature_admission_ledger import PARENTS

    seen: set[str] = set()
    queue = list(PARENTS.get(contract_id, ()))
    while queue:
        parent = queue.pop(0)
        if parent in seen:
            continue
        seen.add(parent)
        queue.extend(PARENTS.get(parent, ()))
    return seen


def test_parent_graph_covers_declared_substrate_dependencies() -> None:
    """Any family whose declared source consumes OPRA CBBO-1m must depend on it.

    This is the check that would have caught the Phase-0 spec error: implied_spot
    declares 'owned Path-D OPRA CBBO-1m put/call ladder' as its historical source
    but was listed as having no dependency, so it was admitted carrying a local
    compute p99 as if it were an availability clock.
    """
    from v4.research.autoresearch_v2.entry_live_feature_catalog import (
        ENTRY_FEATURE_FAMILY_CATALOG,
    )

    substrate = "entry.opra_cbbo1m_native.v1"
    offenders = []
    for contract in ENTRY_FEATURE_FAMILY_CATALOG:
        if contract.contract_id == substrate:
            continue
        source = (contract.historical_source or "").lower()
        if "cbbo-1m" not in source:
            continue
        if substrate not in _transitive_parents(contract.contract_id):
            offenders.append(contract.contract_id)
    assert not offenders, (
        "families consume OPRA CBBO-1m but do not declare it as a transitive parent: "
        f"{offenders}"
    )


def test_implied_spot_declares_its_quote_parent() -> None:
    assert "entry.opra_cbbo1m_native.v1" in _transitive_parents("entry.opra_implied_spot.v1")


def test_greeks_reach_the_quote_substrate_transitively() -> None:
    """greeks -> implied_spot -> cbbo1m_native. A grandparent gap must not hide."""
    assert "entry.opra_cbbo1m_native.v1" in _transitive_parents("entry.self_computed_greeks.v1")
    assert "entry.opra_cbbo1m_native.v1" in _transitive_parents("entry.opra_implied_volatility.v1")


def test_compute_clock_kinds_are_not_arrival_clocks() -> None:
    """A local compute p99 may never back an ADMITTED row on its own."""
    from v4.research.pathd_feature_admission_ledger import ARRIVAL_CLOCK_KINDS

    assert "measured_local_shared_parity_adapter_compute_p99" not in ARRIVAL_CLOCK_KINDS
    assert "composed_parent_arrival_plus_local_compute" in ARRIVAL_CLOCK_KINDS
