from __future__ import annotations

import json
from pathlib import Path

from v4.foundation.formal_validation_governance import (
    BLOCKED,
    PASS,
    PAPER_DEFAULT_BASELINE,
    build_formal_validation_governance,
    cscv_proxy,
    strategy_matrix,
)


def _write_summary(root: Path, name: str, *, deltas: dict[str, float]) -> None:
    path = root / name / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "role_label": name,
                "decision": f"{name}_research_only",
                "paper_default_baseline": PAPER_DEFAULT_BASELINE,
                "stress_results": [
                    {
                        "seed": 1,
                        "slippage_per_side": 0.0,
                        "splits": {
                            split: {
                                "delta_vs_protocol101_same_scope": delta,
                                "total_pnl": 10_000 + delta,
                                "protocol101_same_scope_pnl": 10_000,
                                "trades": 10,
                                "challenger_entries": 2,
                                "all_flat_by_session_end": True,
                                "max_concurrent_positions": 1,
                            }
                            for split, delta in deltas.items()
                        },
                    }
                ],
            }
        )
        + "\n"
    )


def test_strategy_matrix_extracts_protocol101_comparable_stress_results(tmp_path: Path) -> None:
    _write_summary(tmp_path, "strategy_a", deltas={"q1_2026": 100.0, "q3_2025": -50.0, "q4_2025": 200.0})
    (tmp_path / "ignored" / "summary.json").parent.mkdir(parents=True)
    (tmp_path / "ignored" / "summary.json").write_text(json.dumps({"stress_results": []}) + "\n")

    matrix = strategy_matrix(tmp_path)

    assert len(matrix) == 3
    assert set(matrix["strategy_id"]) == {"strategy_a"}
    assert set(matrix["split"]) == {"q1_2026", "q3_2025", "q4_2025"}


def test_cscv_proxy_blocks_when_matrix_is_too_small(tmp_path: Path) -> None:
    _write_summary(tmp_path, "strategy_a", deltas={"q1_2026": 100.0, "q3_2025": -50.0})

    result = cscv_proxy(strategy_matrix(tmp_path))

    assert result["status"] == BLOCKED
    assert result["reason"] == "insufficient_comparable_strategies_or_splits"


def test_formal_validation_controls_ready_with_comparable_matrix(tmp_path: Path) -> None:
    _write_summary(tmp_path, "strategy_a", deltas={"q1_2026": 100.0, "q3_2025": -50.0, "q4_2025": 200.0, "recent_2026": 40.0})
    _write_summary(tmp_path, "strategy_b", deltas={"q1_2026": 20.0, "q3_2025": 10.0, "q4_2025": -30.0, "recent_2026": 60.0})

    payload = build_formal_validation_governance(tmp_path)

    assert payload["control_status"] == PASS
    assert payload["decision"] == "formal_validation_controls_ready"
    assert payload["pbo_cscv_status"] == "formal_validation_controls_ready"
    assert payload["comparable_strategy_count"] == 2
    assert payload["comparable_split_count"] == 4
    assert payload["protected_holdout_scored"] is False
