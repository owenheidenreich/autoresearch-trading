from __future__ import annotations

import numpy as np

from v4.scripts.run_protocol101_h2_calibration_to_g9_graph import (
    NODE_NAMES,
    initial_state,
    route_after_calibration,
    route_after_g9,
)
from v4.scripts.run_protocol101_h2_policy5_calibration_audit import (
    independent_ece,
)


def test_graph_has_exact_bounded_node_order() -> None:
    assert NODE_NAMES == (
        "CAL-PREREGISTER",
        "CAL-MACHINERY-SMOKE",
        "CAL-RUN",
        "CAL-G1-G8-GATE",
        "CAL-G1-G8-INDEPENDENT-AUDIT",
        "C1-G9-MACHINERY",
        "C2-G9-RUN-SEED45",
        "C3-G9-INDEPENDENT-AUDIT",
    )
    state = initial_state()
    assert state["max_mechanical_attempts_per_node"] == 3
    assert all(row["status"] == "pending" for row in state["nodes"].values())


def test_calibration_route_only_accepts_exact_eligible_verdict() -> None:
    assert route_after_calibration("accepted_eligible_G1_G8") == "continue_to_g9"
    assert (
        route_after_calibration("accepted_calibration_repair_failed_G8")
        == "stop_scientific_calibration_result"
    )
    assert (
        route_after_calibration("void_implementation_or_artifact_defect")
        == "stop_scientific_calibration_result"
    )


def test_g9_route_is_terminal() -> None:
    assert (
        route_after_g9("g9_pass_candidate_eligible_for_final_fit")
        == "complete_g9_pass"
    )
    assert (
        route_after_g9("g9_fail_candidate_burned")
        == "complete_g9_scientific_fail"
    )


def test_independent_ece_uses_ten_weighted_bins() -> None:
    confidence = np.asarray([0.1, 0.2, 0.8, 0.9])
    outcomes = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert np.isclose(independent_ece(confidence, outcomes), 0.15)
