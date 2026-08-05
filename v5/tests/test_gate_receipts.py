"""Gate ordering must be released by verified receipts, not caller claims."""
from __future__ import annotations

from dataclasses import replace

import pytest

from v5.research import gate_receipts as gr
from v5.research import knobs


def _receipt(tmp_path, gate: str = "G4") -> gr.GatePassReceipt:
    evidence = tmp_path / f"{gate.lower()}_evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    return gr.make_gate_pass_receipt(
        gate=gate,
        passed_on="2026-08-05",
        evidence_paths=[str(evidence)],
        summary=f"synthetic {gate} pass for tests",
    )


def test_a_valid_receipt_round_trips_through_disk(tmp_path) -> None:
    receipt = _receipt(tmp_path)
    path = tmp_path / "receipt.json"
    gr.write_gate_pass_receipt(receipt, path)
    assert gr.load_gate_pass_receipt(path) == receipt
    with pytest.raises(gr.GateReceiptError, match="refusing to overwrite"):
        gr.write_gate_pass_receipt(receipt, path)


def test_a_tampered_receipt_releases_nothing(tmp_path) -> None:
    tampered = replace(_receipt(tmp_path, "G1"), gate="G4")
    with pytest.raises(gr.GateReceiptError, match="self-hash mismatch"):
        tampered.assert_valid()


def test_a_receipt_citing_missing_evidence_is_refused(tmp_path) -> None:
    receipt = gr.make_gate_pass_receipt(
        gate="G1",
        passed_on="2026-08-05",
        evidence_paths=[str(tmp_path / "does_not_exist.json")],
        summary="cites nothing real",
    )
    with pytest.raises(gr.GateReceiptError, match="missing evidence"):
        receipt.assert_valid()


def test_unknown_gates_and_bad_dates_are_refused(tmp_path) -> None:
    evidence = tmp_path / "e.json"
    evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(gr.GateReceiptError, match="unknown gate"):
        gr.make_gate_pass_receipt(
            gate="G99", passed_on="2026-08-05",
            evidence_paths=[str(evidence)], summary="x",
        ).assert_valid()
    with pytest.raises(gr.GateReceiptError, match="invalid pass date"):
        gr.make_gate_pass_receipt(
            gate="G1", passed_on="yesterday",
            evidence_paths=[str(evidence)], summary="x",
        ).assert_valid()


def test_the_caller_asserted_release_path_is_replaced_by_receipts(tmp_path) -> None:
    """The old hole: `released_gates={'G1','G4'}` was accepted on the caller's word.

    The receipt-consuming form releases nothing without verified receipts and
    permits the same search once real G1 and G4 receipts exist.
    """

    params = {
        "model_class": "random_forest",
        "horizon_minutes": 60,
        "entry_threshold": 0.5,
        "hold_minutes": 25,
    }
    with pytest.raises(knobs.KnobError, match="has not passed"):
        gr.assert_search_space_released(params, gate_receipts=[])
    with pytest.raises(knobs.KnobError, match="has not passed"):
        gr.assert_search_space_released(params, gate_receipts=[_receipt(tmp_path, "G1")])
    gr.assert_search_space_released(
        params,
        gate_receipts=[_receipt(tmp_path, "G1"), _receipt(tmp_path, "G4")],
    )


def test_an_invalid_receipt_in_the_set_blocks_the_whole_release(tmp_path) -> None:
    good = _receipt(tmp_path, "G1")
    bad = replace(_receipt(tmp_path, "G4"), summary="edited after signing")
    with pytest.raises(gr.GateReceiptError, match="self-hash mismatch"):
        gr.assert_search_space_released(
            {"horizon_minutes": 60}, gate_receipts=[good, bad]
        )
