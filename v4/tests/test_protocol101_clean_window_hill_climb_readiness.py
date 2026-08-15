from __future__ import annotations

from v4.scripts.run_protocol101_clean_window_hill_climb_readiness import (
    build_summary,
)


def test_clean_window_readiness_is_scoped_to_stable_subset() -> None:
    summary = build_summary()

    assert summary["confident_answer"] is True
    assert (
        summary["decision"]
        == "governed_hill_climbing_ready_on_parity_stable_subset"
    )
    assert summary["family_results"]["D_composites"]["initial_alpha"] is True
    assert (
        summary["family_results"]["C_direct_option_price"]["initial_alpha"]
        is False
    )
    assert summary["paper_ready"] is False
