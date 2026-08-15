from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from v4.scripts.run_protocol101_ft2_stage0_serial_failure_attribution import (
    AttributionBlocked,
    classify_skipped_intents,
    decomposition_frame,
    parse_checksum_manifest,
    route_from_contributions,
    validate_checksum_manifest,
)


@pytest.mark.parametrize(
    ("common", "stream", "expected"),
    [
        (1.0, -1.0, "entry_timing_and_reentry_is_binding"),
        (-1.0, 1.0, "one_step_exit_target_is_binding"),
        (-1.0, -1.0, "both_entry_and_exit_are_binding"),
        (0.0, 0.0, "attribution_inconclusive"),
        (1.0, 1.0, "attribution_inconclusive"),
    ],
)
def test_frozen_route_sign_rules(
    common: float, stream: float, expected: str
) -> None:
    assert route_from_contributions(common, stream) == expected


def test_checksum_manifest_validation(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    payload.write_text('{"frozen":true}\n', encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "hashes.sha256"
    manifest.write_text(f"{digest}  payload.json\n", encoding="utf-8")

    assert parse_checksum_manifest(manifest) == [(digest, "payload.json")]
    result = validate_checksum_manifest(manifest, tmp_path)
    assert result[0]["matches"] is True

    payload.write_text('{"frozen":false}\n', encoding="utf-8")
    with pytest.raises(AttributionBlocked) as exc:
        validate_checksum_manifest(manifest, tmp_path)
    assert exc.value.route == "attribution_blocked_artifact_mismatch"


def _trade(
    session: str,
    decision: int,
    contract: str,
    slot: int,
    pnl: float,
    exit_time: int,
) -> dict[str, object]:
    return {
        "session": session,
        "decision_time_ns": decision,
        "contract_id": contract,
        "canonical_strike_slot": slot,
        "right": "C",
        "entry_ask": 10.0,
        "premium_at_risk": 1_000.0,
        "label_realized_exit_time_ns": exit_time,
        "label_source_exit_quote_time_ns": exit_time,
        "label_exit_reason_code": 1,
        "stressed_pnl": pnl,
    }


def test_identity_decomposition_reconciles_common_and_reentry() -> None:
    p5 = pd.DataFrame(
        [_trade("2025-03-04", 100, "A", 10, 100.0, 400)]
    )
    hgb = pd.DataFrame(
        [
            _trade("2025-03-04", 100, "A", 10, 40.0, 200),
            _trade("2025-03-04", 200, "B", 11, -25.0, 300),
        ]
    )

    result = decomposition_frame(p5, hgb, 3.0)
    assert result["classification"].tolist() == ["common", "hgb_only"]
    common = result["common_exit_pnl_delta"].sum()
    stream = result["entry_stream_pnl_component"].sum()
    total = hgb["stressed_pnl"].sum() - p5["stressed_pnl"].sum()
    assert common == -60.0
    assert stream == -25.0
    assert total == common + stream
    assert route_from_contributions(common, stream) == (
        "both_entry_and_exit_are_binding"
    )


def test_skipped_intents_reconstruct_overlap_and_daily_stop() -> None:
    intents = pd.DataFrame(
        [
            {
                "session": "2025-03-04",
                "decision_time_ns": minute,
                "contract_id": contract,
                "canonical_slot": slot,
            }
            for minute, contract, slot in [
                (100, "A", 10),
                (200, "B", 11),
                (300, "C", 12),
                (400, "D", 13),
            ]
        ]
    )
    admitted = pd.DataFrame(
        [
            {
                **_trade(
                    "2025-03-04", 100, "A", 10, -600.0, 300
                ),
                "session_start_equity": 10_000.0,
                "session_realized_pnl_after": -600.0,
            }
        ]
    )

    result = classify_skipped_intents(
        intents, admitted, daily_loss_fraction=0.05
    )
    assert result["reason"].tolist() == [
        "overlap",
        "daily_loss_stop",
        "daily_loss_stop",
    ]
