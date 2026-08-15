from __future__ import annotations

import copy
from pathlib import Path

from v4.live.shadow_parity import (
    ShadowParityConfig,
    load_shadow_observations,
    shadow_observation_template,
    summarize_shadow_parity,
    validate_shadow_observation,
    write_shadow_template,
)


def _row() -> dict:
    return shadow_observation_template(ShadowParityConfig())


def test_shadow_template_passes_no_order_parity() -> None:
    summary = summarize_shadow_parity([_row()])

    assert summary["status"] in {"pass", "warn"}
    assert summary["failed_rows"] == 0
    checks = {check["name"]: check["status"] for check in summary["checks"]}
    assert checks["no_order_shadow_mode"] == "pass"
    assert checks["feature_schema"] == "pass"


def test_shadow_parity_blocks_order_intent() -> None:
    row = _row()
    row["order_intent"] = {"side": "BUY", "quantity": 1}

    result = validate_shadow_observation(row, row_index=0)

    assert result.status == "fail"
    assert any("broker/order field" in error for error in result.errors)


def test_shadow_parity_requires_protocol066_feature_schema() -> None:
    row = _row()
    row["features"] = copy.deepcopy(row["features"])
    row["features"].pop("gamma")

    result = validate_shadow_observation(row, row_index=0)

    assert result.status == "fail"
    assert any("missing required features" in error for error in result.errors)


def test_shadow_parity_blocks_stale_quotes() -> None:
    row = _row()
    row["timestamp_ms"] = 1770000000000
    row["nbbo"]["timestamp_ms"] = 1769999990000

    result = validate_shadow_observation(row, row_index=0)

    assert result.status == "fail"
    assert any("quote age" in error for error in result.errors)


def test_shadow_jsonl_loader(tmp_path: Path) -> None:
    path = tmp_path / "shadow.jsonl"
    write_shadow_template(path)

    rows = load_shadow_observations(path)

    assert len(rows) == 1
    assert rows[0]["protocol_id"] == "protocol066"

