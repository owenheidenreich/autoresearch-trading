from __future__ import annotations

import json
from pathlib import Path

from v5.ops.record_compact_shared_lifecycle_design import build_receipt


def test_design_receipt_uses_built_count_and_refuses_fit_claim(tmp_path: Path) -> None:
    design = tmp_path / "design.md"
    design.write_text("frozen design\n")
    value = tmp_path / "value.json"
    value.write_text(
        json.dumps(
            {
                "result": {
                    "new_architecture_requirement": {
                        "parameters_at_most_for_full_conservative_support": 122
                    }
                }
            }
        )
    )

    got = build_receipt(design_path=design, value_path=value)

    assert got["architecture"]["built_parameter_count"] == 120
    assert got["architecture"]["budget_headroom"] == 2
    assert got["fit_boundary"]["fit_permitted"] is False
    assert got["integrity"]["economics_read"] is False
