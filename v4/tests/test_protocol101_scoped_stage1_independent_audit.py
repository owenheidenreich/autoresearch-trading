from __future__ import annotations

import ast
import inspect
from copy import deepcopy

from v4.scripts import run_protocol101_scoped_stage1_independent_audit as audit
from v4.scripts.run_protocol101_full_trader_stage1_gate_audit_selection_validation import (
    _aggregate,
    _audit,
    _authorities,
    _reseal,
    build_synthetic_campaign,
)
from v4.model.protocol101_stage1_gate_contract import (
    seal_campaign_packet,
    seal_unit,
)
from v4.scripts.run_protocol101_scoped_stage1_gate_aggregator import (
    aggregate_campaign,
)
def test_audit_source_does_not_import_producer_aggregator() -> None:
    tree = ast.parse(inspect.getsource(audit))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(
        "run_protocol101_scoped_stage1_gate_aggregator" in value
        for value in imports
    )


def test_independent_audit_replays_and_agrees_on_all_28_rows() -> None:
    packet = build_synthetic_campaign()
    producer = _aggregate(packet)
    result = _audit(packet, producer)
    assert result["accepted"] is True
    assert result["campaign_valid"] is True
    assert result["comparisons"] == []
    assert result["published_row_count"] == 28
    assert result["all_28_rows_published"] is True
    assert result["freeze_sha256"]
    fold = result["independent_result"]["rows"][0]["seed_results"][0][
        "fold_results"
    ][0]
    assert "candidate_stream_hash" in fold
    assert "candidate_payload_hash" in fold
    assert "skipped_events" in fold
    assert "source_and_realized_clocks" in fold


def test_independent_audit_catches_injected_producer_defect() -> None:
    packet = build_synthetic_campaign()
    producer = _aggregate(packet)
    defect = deepcopy(producer)
    defect["rows"][0][
        "median_seed_fee_adjusted_continuous_strict_serial_net_pnl"
    ] += 1.0
    result = _audit(packet, defect)
    assert result["accepted"] is False
    assert "producer_independent_result_disagreement" in result["defects"]
    assert result["comparisons"]
    assert result["freeze"] is None


def test_independent_audit_rejects_duplicate_trade_before_replay() -> None:
    packet = build_synthetic_campaign()
    packet["units"][0]["candidates"].append(
        deepcopy(packet["units"][0]["candidates"][0])
    )
    producer = _aggregate(packet)
    result = _audit(packet, producer)
    assert result["accepted"] is False
    assert any("trade_identity_duplicate" in value for value in result["defects"])


def test_independent_audit_derives_maxT_row_law() -> None:
    packet = build_synthetic_campaign()
    packet["controls"]["maxT"]["rows"][0]["hard_pass"] = False
    _reseal(packet)
    producer = _aggregate(packet)
    result = _audit(packet, producer)
    assert result["accepted"] is False
    assert "maxT_row_law_invalid:H0/P0" in result["defects"]


def test_independent_audit_enforces_frozen_execution_authority() -> None:
    packet = build_synthetic_campaign()
    frozen = _authorities(packet)
    packet["units"][0]["provenance"]["model_hash"] = "a" * 64
    packet["units"][0] = seal_unit(packet["units"][0])
    packet = seal_campaign_packet(packet)
    producer = aggregate_campaign(packet, **frozen)
    result = audit.audit_campaign(packet, producer, **frozen)
    assert result["accepted"] is False
    assert "execution_authority_binding_mismatch" in result["defects"]


def test_independent_audit_enforces_frozen_control_authority() -> None:
    packet = build_synthetic_campaign()
    frozen = _authorities(packet)
    packet["controls"]["D5"]["receipt_sha256"] = "b" * 64
    packet = seal_campaign_packet(packet)
    producer = aggregate_campaign(packet, **frozen)
    result = audit.audit_campaign(packet, producer, **frozen)
    assert result["accepted"] is False
    assert "control_authority_binding_mismatch" in result["defects"]
