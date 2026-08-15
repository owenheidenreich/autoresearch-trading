from __future__ import annotations

from v4.scripts.run_protocol120_surface_edge_portability import summarize_edge_rows


def test_protocol120_summarizes_edge_rows() -> None:
    summary = summarize_edge_rows(
        [
            {"edge": 10.0, "right": "C", "valid_token_count": 2},
            {"edge": -5.0, "right": "P", "valid_token_count": 4},
            {"edge": None, "right": None, "valid_token_count": 0},
        ]
    )

    assert summary["scored_decisions"] == 3
    assert summary["finite_edge_count"] == 2
    assert round(summary["finite_edge_fraction"], 6) == round(2 / 3, 6)
    assert summary["positive_edge_fraction"] == 0.5
    assert summary["selected_side_counts"] == {"C": 1, "P": 1}
